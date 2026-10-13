
import cv2
import mediapipe as mp
import joblib
import numpy as np
import math


# ============================================================
# Paths
# ============================================================

MODEL_PATH = "models/rps_random_forest.pkl"
HAND_MODEL_PATH = "models/hand_landmarker.task"


# ============================================================
# Settings
# ============================================================

CONFIDENCE_THRESHOLD = 60


# ============================================================
# Load Random Forest
# ============================================================

model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully.")


# ============================================================
# MediaPipe Setup
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# Winner Logic
# ============================================================

def get_winner(player1, player2):

    if player1 == player2:
        return "DRAW"

    if player1 == "ROCK" and player2 == "SCISSORS":
        return "PLAYER 1 WINS"

    if player1 == "PAPER" and player2 == "ROCK":
        return "PLAYER 1 WINS"

    if player1 == "SCISSORS" and player2 == "PAPER":
        return "PLAYER 1 WINS"

    return "PLAYER 2 WINS"


# ============================================================
# Draw Landmarks
# ============================================================

def draw_landmarks(frame, hand_landmarks):

    h, w, _ = frame.shape

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),

        (0, 5), (5, 6), (6, 7), (7, 8),

        (5, 9), (9, 10), (10, 11), (11, 12),

        (9, 13), (13, 14), (14, 15), (15, 16),

        (13, 17), (17, 18), (18, 19), (19, 20),

        (0, 17)
    ]

    # Draw connections
    for start, end in connections:

        x1 = int(hand_landmarks[start].x * w)
        y1 = int(hand_landmarks[start].y * h)

        x2 = int(hand_landmarks[end].x * w)
        y2 = int(hand_landmarks[end].y * h)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    # Draw points
    for landmark in hand_landmarks:

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )


# ============================================================
# Predict Gesture
# ============================================================

def predict_gesture(hand_landmarks):

    # Wrist
    wrist = hand_landmarks[0]

    # Middle finger MCP
    reference = hand_landmarks[9]

    # Hand scale
    scale = math.sqrt(
        (reference.x - wrist.x) ** 2 +
        (reference.y - wrist.y) ** 2 +
        (reference.z - wrist.z) ** 2
    )

    if scale == 0:
        return "UNKNOWN", 0.0

    # 21 landmarks × 3 = 63 features
    features = []

    for landmark in hand_landmarks:

        x = (
            landmark.x - wrist.x
        ) / scale

        y = (
            landmark.y - wrist.y
        ) / scale

        z = (
            landmark.z - wrist.z
        ) / scale

        features.extend([
            x,
            y,
            z
        ])

    features = np.array(
        features,
        dtype=np.float32
    ).reshape(1, -1)

    if features.shape[1] != 63:
        return "UNKNOWN", 0.0

    # Prediction
    prediction = model.predict(features)[0]

    # Confidence
    confidence = 0.0

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            features
        )[0]

        confidence = np.max(probabilities) * 100

        if confidence < CONFIDENCE_THRESHOLD:
            prediction = "UNKNOWN"

    return prediction, confidence


# ============================================================
# Start Webcam
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


timestamp = 0


# ============================================================
# Game Variables
# ============================================================

score_player1 = 0
score_player2 = 0
draws = 0

last_player1_gesture = None
last_player2_gesture = None

current_player1_gesture = "UNKNOWN"
current_player2_gesture = "UNKNOWN"

result_text = "Show your gestures"


# ============================================================
# Run Game
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        # BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp += 33

        # Detect hands
        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        # ====================================================
        # Detect Gestures
        # ====================================================

        current_gestures = []

        if result.hand_landmarks:

            for hand_landmarks in result.hand_landmarks:

                gesture, confidence = predict_gesture(
                    hand_landmarks
                )

                current_gestures.append(
                    (gesture, confidence)
                )

                draw_landmarks(
                    frame,
                    hand_landmarks
                )


        # ====================================================
        # Get Player 1 Gesture
        # ====================================================

        if len(current_gestures) >= 1:

            current_player1_gesture = current_gestures[0][0]

            confidence1 = current_gestures[0][1]

        else:

            current_player1_gesture = "UNKNOWN"
            confidence1 = 0.0


        # ====================================================
        # Get Player 2 Gesture
        # ====================================================

        if len(current_gestures) >= 2:

            current_player2_gesture = current_gestures[1][0]

            confidence2 = current_gestures[1][1]

        else:

            current_player2_gesture = "UNKNOWN"
            confidence2 = 0.0


        # ====================================================
        # Display Current Gestures
        # ====================================================

        cv2.putText(
            frame,
            f"Player 1: {current_player1_gesture} "
            f"({confidence1:.1f}%)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Player 2: {current_player2_gesture} "
            f"({confidence2:.1f}%)",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )


        # ====================================================
        # GAME LOGIC
        # ====================================================

        valid_gestures = [
            "ROCK",
            "PAPER",
            "SCISSORS"
        ]


        # Both players must have valid gestures
        if (
            current_player1_gesture in valid_gestures
            and
            current_player2_gesture in valid_gestures
        ):

            # ------------------------------------------------
            # Detect gesture change
            # ------------------------------------------------

            gesture_changed = (
                current_player1_gesture != last_player1_gesture
                or
                current_player2_gesture != last_player2_gesture
            )


            # ------------------------------------------------
            # New round
            # ------------------------------------------------

            if gesture_changed:

                player1 = current_player1_gesture
                player2 = current_player2_gesture

                # Determine winner
                result_text = get_winner(
                    player1,
                    player2
                )


                # Update score
                if result_text == "PLAYER 1 WINS":

                    score_player1 += 1

                elif result_text == "PLAYER 2 WINS":

                    score_player2 += 1

                else:

                    draws += 1


                # Save current gestures
                last_player1_gesture = player1
                last_player2_gesture = player2


        # ====================================================
        # If Invalid / Missing Gestures
        # ====================================================

        else:

            if len(current_gestures) < 2:

                result_text = "Both players show your hands"

            else:

                result_text = "Invalid gesture"


        # ====================================================
        # Display Result
        # ====================================================

        cv2.putText(
            frame,
            result_text,
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3
        )


        # ====================================================
        # Display Last Round
        # ====================================================

        if last_player1_gesture is not None:

            cv2.putText(
                frame,
                f"Last Round: "
                f"P1 {last_player1_gesture} "
                f"vs "
                f"P2 {last_player2_gesture}",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


        # ====================================================
        # Score
        # ====================================================

        cv2.putText(
            frame,
            f"P1: {score_player1}    "
            f"P2: {score_player2}    "
            f"Draws: {draws}",
            (20, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ====================================================
        # Instructions
        # ====================================================

        cv2.putText(
            frame,
            "Change your gesture for next round | Q = Quit",
            (20, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ====================================================
        # Show
        # ====================================================

        cv2.imshow(
            "Human vs Human - Rock Paper Scissors",
            frame
        )


        # ====================================================
        # Keyboard
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# ============================================================
# Cleanup
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Human vs Human game ended.")

