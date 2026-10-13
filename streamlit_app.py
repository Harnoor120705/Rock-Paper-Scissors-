"""Rock Paper Scissors - Streamlit version.

Pipeline: browser camera snapshot -> MediaPipe Hand Landmarker (21 landmarks)
-> wrist-relative normalization (63 features) -> Random Forest -> game logic.
"""
import hashlib
import math
import random
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "rps_random_forest.pkl"
HAND_MODEL_PATH = BASE / "models" / "hand_landmarker.task"
DATA_PATH = BASE / "dataset" / "public_rps_normalized.csv"

CHOICES = ["ROCK", "PAPER", "SCISSORS"]
EMOJI = {"ROCK": "✊", "PAPER": "🖐️", "SCISSORS": "✌️"}
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

st.set_page_config(page_title="RPS Hand Gesture Game", page_icon="🎮", layout="centered")


# ---------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading model...")
def load_classifier():
    """Load the saved model; if it can't be loaded (e.g. sklearn version
    mismatch on the server), retrain from the bundled CSV."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        df = pd.read_csv(DATA_PATH)
        train = df[df["split"] == "train"]
        X = train.drop(columns=["image", "split", "hand_id", "label"])
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X.values, train["label"])
        return model


@st.cache_resource(show_spinner="Loading hand detector...")
def load_landmarker(num_hands: int):
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(opts)


# ----------------------------------------------------------------- vision
def normalize(landmarks):
    wrist, ref = landmarks[0], landmarks[9]
    scale = math.sqrt((ref.x - wrist.x) ** 2 + (ref.y - wrist.y) ** 2 + (ref.z - wrist.z) ** 2)
    if scale == 0:
        return None
    feats = []
    for lm in landmarks:
        feats += [(lm.x - wrist.x) / scale, (lm.y - wrist.y) / scale, (lm.z - wrist.z) / scale]
    return np.array(feats, dtype=np.float32).reshape(1, -1)


def classify(model, landmarks, threshold):
    feats = normalize(landmarks)
    if feats is None:
        return "UNKNOWN", 0.0
    pred = model.predict(feats)[0]
    conf = float(np.max(model.predict_proba(feats)[0]) * 100)
    return (pred if conf >= threshold else "UNKNOWN"), conf


def detect(rgb, num_hands, model, threshold):
    """Return (annotated_rgb, [(label, conf, wrist_x), ...]) sorted left->right."""
    result = load_landmarker(num_hands).detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    out = rgb.copy()
    h, w, _ = out.shape
    hands = []
    for lms in result.hand_landmarks:
        pts = [(int(l.x * w), int(l.y * h)) for l in lms]
        for a, b in CONNECTIONS:
            cv2.line(out, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(out, p, 4, (255, 0, 0), -1)
        label, conf = classify(model, lms, threshold)
        hands.append((label, conf, lms[0].x, pts[0]))
    hands.sort(key=lambda t: t[2])
    return out, hands


def winner(a, b):
    if a == b:
        return "DRAW"
    beats = {("ROCK", "SCISSORS"), ("PAPER", "ROCK"), ("SCISSORS", "PAPER")}
    return "A" if (a, b) in beats else "B"


# ------------------------------------------------------------------ state
def init_state():
    st.session_state.setdefault("scores", {"p1": 0, "p2": 0, "draw": 0})
    st.session_state.setdefault("last_hash", None)
    st.session_state.setdefault("last_result", None)


def reset_scores():
    st.session_state.scores = {"p1": 0, "p2": 0, "draw": 0}
    st.session_state.last_result = None


def show_scores(n1, n2):
    s = st.session_state.scores
    c1, c2, c3 = st.columns(3)
    c1.metric(n1, s["p1"])
    c2.metric("Draws", s["draw"])
    c3.metric(n2, s["p2"])


# -------------------------------------------------------------------- app
def main():
    init_state()
    model = load_classifier()

    st.title("🎮 Rock Paper Scissors")
    st.caption("Hand-gesture game: MediaPipe landmarks + Random Forest classifier")

    with st.sidebar:
        mode = st.radio("Game mode", ["Human vs Computer", "Human vs Human"])
        threshold = st.slider("Confidence threshold (%)", 30, 95, 60,
                              help="Below this the gesture is shown as UNKNOWN")
        st.button("Reset score", on_click=reset_scores)
        st.markdown("**How to play**\n1. Allow camera access\n2. Show ✊ 🖐️ or ✌️\n3. Click *Take photo*"
                    "\n\nHuman vs Human: both hands in one photo, "
                    "Player 1 on the left of the image.")

    vs_cpu = mode == "Human vs Computer"
    n1, n2 = ("You", "Computer") if vs_cpu else ("Player 1", "Player 2")
    show_scores(n1, n2)

    shot = st.camera_input("Show your gesture and take a photo", key=f"cam_{mode}")
    if shot is None:
        return

    data = shot.getvalue()
    h = hashlib.md5(data).hexdigest()
    rgb = np.array(Image.open(shot).convert("RGB"))
    annotated, hands = detect(rgb, 1 if vs_cpu else 2, model, threshold)

    for label, conf, _, wrist in hands:
        cv2.putText(annotated, f"{label} {conf:.0f}%", (max(wrist[0] - 40, 5), max(wrist[1] + 30, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    st.image(annotated, caption="Detected landmarks", use_container_width=True)

    new_photo = h != st.session_state.last_hash   # count each photo only once
    st.session_state.last_hash = h
    s = st.session_state.scores

    if vs_cpu:
        if not hands:
            st.warning("No hand detected. Move closer / improve lighting.")
            return
        player, conf, _, _ = hands[0]
        if player not in CHOICES:
            st.warning(f"Unclear gesture (UNKNOWN, {conf:.0f}% confidence). Try again.")
            return
        if new_photo:
            comp = random.choice(CHOICES)
            r = winner(player, comp)
            s["p1" if r == "A" else "p2" if r == "B" else "draw"] += 1
            st.session_state.last_result = (player, comp, r)
            st.rerun()
        if st.session_state.last_result:
            player, comp, r = st.session_state.last_result
            st.subheader(f"{EMOJI[player]} You  vs  Computer {EMOJI[comp]}")
            {"A": st.success, "B": st.error, "DRAW": st.info}[r](
                {"A": "🏆 YOU WIN!", "B": "🤖 COMPUTER WINS", "DRAW": "🤝 DRAW"}[r])
    else:
        if len(hands) < 2:
            st.warning(f"Detected {len(hands)} hand(s). Need 2 hands in the photo.")
            return
        (g1, c1, _, _), (g2, c2, _, _) = hands[0], hands[1]
        if g1 not in CHOICES or g2 not in CHOICES:
            st.warning("One gesture was unclear (UNKNOWN). Try again.")
            return
        if new_photo:
            r = winner(g1, g2)
            s["p1" if r == "A" else "p2" if r == "B" else "draw"] += 1
            st.session_state.last_result = (g1, g2, r)
            st.rerun()
        if st.session_state.last_result:
            g1, g2, r = st.session_state.last_result
            st.subheader(f"{EMOJI[g1]} Player 1  vs  Player 2 {EMOJI[g2]}")
            {"A": st.success, "B": st.success, "DRAW": st.info}[r](
                {"A": "🏆 PLAYER 1 WINS!", "B": "🏆 PLAYER 2 WINS!", "DRAW": "🤝 DRAW"}[r])


main()
