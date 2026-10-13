"""Rock Paper Scissors - live Streamlit version (streamlit-webrtc).

Same flow as the desktop game: live camera -> press START -> 3-2-1 countdown
-> gesture is read -> result and score are drawn on the video.

Pipeline: MediaPipe Hand Landmarker (21 landmarks) -> wrist-relative
normalization (63 features) -> Random Forest -> confidence threshold -> game.
"""
import math
import random
import threading
import time
from collections import Counter, deque
from pathlib import Path

import av
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "rps_random_forest.pkl"
HAND_MODEL_PATH = BASE / "models" / "hand_landmarker.task"
DATA_PATH = BASE / "dataset" / "public_rps_normalized.csv"

CHOICES = ["ROCK", "PAPER", "SCISSORS"]
ROUND_TIME = 3
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

st.set_page_config(page_title="RPS Hand Gesture Game", page_icon="🎮", layout="centered")


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model...")
def load_classifier():
    """Load the saved model; if that fails (e.g. sklearn version mismatch),
    retrain from the bundled CSV."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        df = pd.read_csv(DATA_PATH)
        train = df[df["split"] == "train"]
        X = train.drop(columns=["image", "split", "hand_id", "label"])
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X.values, train["label"])
        return model


def normalize(landmarks):
    wrist, ref = landmarks[0], landmarks[9]
    scale = math.sqrt((ref.x - wrist.x) ** 2 + (ref.y - wrist.y) ** 2 + (ref.z - wrist.z) ** 2)
    if scale == 0:
        return None
    feats = []
    for lm in landmarks:
        feats += [(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, (lm.z - wrist.z) / scale]
    return np.array(feats, dtype=np.float32).reshape(1, -1)


def winner(a, b):
    if a == b:
        return "DRAW"
    beats = {("ROCK", "SCISSORS"), ("PAPER", "ROCK"), ("SCISSORS", "PAPER")}
    return "A" if (a, b) in beats else "B"


# ------------------------------------------------------------ processor
class RPSProcessor(VideoProcessorBase):
    """Runs once per connected viewer; receives every camera frame."""

    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()
        self.threshold = 60
        self.mode = "Human vs Computer"
        self.scores = {"p1": 0, "p2": 0, "draw": 0}
        self.round_active = False
        self.round_start = 0.0
        self.result_lines = ["Press START ROUND"]
        self.recent = [deque(maxlen=7), deque(maxlen=7)]   # smoothing per hand
        self.last_ts = 0
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(opts)

    # ---- controls (called from the Streamlit UI thread)
    def start_round(self):
        with self.lock:
            self.round_active = True
            self.round_start = time.time()
            self.result_lines = []
            for d in self.recent:
                d.clear()

    def reset(self):
        with self.lock:
            self.scores = {"p1": 0, "p2": 0, "draw": 0}
            self.round_active = False
            self.result_lines = ["Score reset. Press START ROUND"]

    # ---- helpers
    def _predict(self, landmarks):
        feats = normalize(landmarks)
        if feats is None:
            return "UNKNOWN", 0.0
        pred = self.model.predict(feats)[0]
        conf = float(np.max(self.model.predict_proba(feats)[0]) * 100)
        return (pred if conf >= self.threshold else "UNKNOWN"), conf

    @staticmethod
    def _put(img, text, org, scale=0.8, color=(0, 255, 0), thick=2):
        cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 3)
        cv2.putText(img, text, org, FONT, scale, color, thick)

    def _stable(self, i):
        """Most common valid gesture over the last few frames."""
        valid = [g for g in self.recent[i] if g in CHOICES]
        return Counter(valid).most_common(1)[0][0] if valid else None

    def _finish_round(self, n_hands):
        vs_cpu = self.mode == "Human vs Computer"
        g1 = self._stable(0)
        if vs_cpu:
            if g1 is None:
                self.result_lines = ["INVALID GESTURE", "No clear hand - try again"]
                return
            comp = random.choice(CHOICES)
            r = winner(g1, comp)
            self.scores["p1" if r == "A" else "p2" if r == "B" else "draw"] += 1
            head = {"A": "YOU WIN!", "B": "COMPUTER WINS", "DRAW": "DRAW"}[r]
            self.result_lines = [head, f"You: {g1}   Computer: {comp}"]
        else:
            g2 = self._stable(1)
            if n_hands < 2 or g1 is None or g2 is None:
                self.result_lines = ["INVALID", "Need 2 clear hands - try again"]
                return
            r = winner(g1, g2)
            self.scores["p1" if r == "A" else "p2" if r == "B" else "draw"] += 1
            head = {"A": "PLAYER 1 WINS!", "B": "PLAYER 2 WINS!", "DRAW": "DRAW"}[r]
            self.result_lines = [head, f"P1: {g1}   P2: {g2}"]

    # ---- per-frame
    def recv(self, frame):
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)   # mirror like the desktop game
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ts = max(int(time.time() * 1000), self.last_ts + 1)   # must strictly increase
        self.last_ts = ts
        res = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        vs_cpu = self.mode == "Human vs Computer"
        hands = res.hand_landmarks[:1] if vs_cpu else res.hand_landmarks[:2]
        hands = sorted(hands, key=lambda lms: lms[0].x)       # left -> right (P1, P2)

        for i in range(2):
            if i >= len(hands):
                self.recent[i].append("NO HAND")
        gestures = []
        for i, lms in enumerate(hands):
            pts = [(int(l.x * w), int(l.y * h)) for l in lms]
            for a, b in CONNECTIONS:
                cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
            for p in pts:
                cv2.circle(img, p, 4, (0, 0, 255), -1)
            g, conf = self._predict(lms)
            self.recent[i].append(g)
            gestures.append((g, conf, pts[0]))
            tag = f"P{i + 1}: " if not vs_cpu else ""
            top = min(pt[1] for pt in pts)
            left = min(pt[0] for pt in pts)
            self._put(img, f"{tag}{g} {conf:.0f}%", (max(left, 5), max(top - 12, 70)),
                      0.7, (255, 255, 0))

        with self.lock:
            if self.round_active:
                elapsed = time.time() - self.round_start
                if elapsed < ROUND_TIME:
                    n = ROUND_TIME - int(elapsed)
                    self._put(img, str(n), (w // 2 - 40, h // 2 + 40), 5, (0, 255, 255), 8)
                else:
                    self.round_active = False
                    self._finish_round(len(hands))

            # scoreboard
            s = self.scores
            labels = ("You", "Computer") if vs_cpu else ("P1", "P2")
            self._put(img, f"{labels[0]}: {s['p1']}   Draws: {s['draw']}   {labels[1]}: {s['p2']}",
                      (15, 35), 0.8, (255, 255, 255))

            # result text
            y = h - 60
            for k, line in enumerate(self.result_lines):
                color = (0, 255, 0) if k == 0 and "WIN" in line and "COMPUTER" not in line else \
                        (0, 0, 255) if k == 0 and ("COMPUTER" in line or "INVALID" in line) else (255, 255, 255)
                self._put(img, line, (15, y + k * 32), 0.9 if k == 0 else 0.7, color)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ------------------------------------------------------------------- UI
def get_rtc_config():
    """Google STUN by default. If a TURN server is set in Streamlit secrets
    (needed on some networks), it is added automatically."""
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    try:
        servers += list(st.secrets["ice_servers"])
    except Exception:
        pass
    return {"iceServers": servers}


def main():
    model = load_classifier()
    st.title("🎮 Rock Paper Scissors")
    st.caption("Live hand-gesture game: MediaPipe landmarks + Random Forest")

    with st.sidebar:
        mode = st.radio("Game mode", ["Human vs Computer", "Human vs Human"])
        threshold = st.slider("Confidence threshold (%)", 30, 95, 60,
                              help="Below this the gesture counts as UNKNOWN")
        st.markdown("**How to play**\n1. Click **START** on the video and allow the camera\n"
                    "2. Click **Start round**\n3. Show ✊ 🖐️ or ✌️ before the countdown ends"
                    "\n\nHuman vs Human: both hands in view, Player 1 on the left of the screen.")

    ctx = webrtc_streamer(
        key="rps",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=get_rtc_config(),
        video_processor_factory=lambda: RPSProcessor(model),
        media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                                            "frameRate": {"ideal": 20}}, "audio": False},
        async_processing=True,
    )

    c1, c2 = st.columns(2)
    start = c1.button("▶️ Start round", type="primary")
    reset = c2.button("🔄 Reset score")

    if ctx.video_processor:
        ctx.video_processor.mode = mode
        ctx.video_processor.threshold = threshold
        if start:
            ctx.video_processor.start_round()
        if reset:
            ctx.video_processor.reset()
    elif start or reset:
        st.warning("Click START on the video first and allow camera access.")


main()