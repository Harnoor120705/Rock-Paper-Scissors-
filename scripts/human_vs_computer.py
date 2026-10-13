import cv2
import mediapipe as mp
import joblib
import numpy as np
import math
import random
import time


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/rps_random_forest.pkl"
HAND_MODEL_PATH = "models/hand_landmarker.task"


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 60

ROUND_TIME = 3

# Hand skeleton connections
CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17)
]


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

model = joblib.load(MODEL_PATH)

print("Random Forest model loaded successfully.")


# ============================================================
# MEDIAPIPE SETUP
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
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# GAME VARIABLES
# ============================================================

choices = [
    "ROCK",
    "PAPER",
    "SCISSORS"
]

player_score = 0
computer_score = 0
draw_score = 0

player_choice = "WAITING..."
computer_choice = "WAITING..."

result_text = "Press R to start"

round_active = False
round_start_time = 0

countdown_number = 0

timestamp = 0


# ============================================================
# GAME LOGIC
# ============================================================

def get_winner(player, computer):

    if player == computer:
        return "DRAW"

    if (
        player == "ROCK"
        and computer == "SCISSORS"
    ):
        return "PLAYER WINS"

    if (
        player == "PAPER"
        and computer == "ROCK"
    ):
        return "PLAYER WINS"

    if (
        player == "SCISSORS"
        and computer == "PAPER"
    ):
        return "PLAYER WINS"

    return "COMPUTER WINS"


# ============================================================
# START WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


# ============================================================
# MEDIAPIPE LANDMARKER
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            break


        # ----------------------------------------------------
        # Mirror webcam
        # ----------------------------------------------------

        frame = cv2.flip(frame, 1)


        # ----------------------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp += 33


        # ----------------------------------------------------
        # Detect hand
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        # ====================================================
        # DEFAULT PREDICTION
        # ====================================================

        prediction = "NO HAND"
        confidence = 0


        # ====================================================
        # PROCESS HAND
        # ====================================================

        if result.hand_landmarks:

            hand_landmarks = result.hand_landmarks[0]


            # ------------------------------------------------
            # DRAW HAND LANDMARKS
            # ------------------------------------------------

            h, w, _ = frame.shape

            points = []

            for landmark in hand_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # ------------------------------------------------
            # DRAW CONNECTIONS
            # ------------------------------------------------

            for start, end in CONNECTIONS:

                if (
                    start < len(points)
                    and end < len(points)
                ):

                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (0, 255, 0),
                        2
                    )


            # =================================================
            # NORMALIZATION
            # Same normalization used during training
            # =================================================

            # Landmark 0 = wrist
            wrist = hand_landmarks[0]

            # Landmark 9 = middle finger MCP
            reference = hand_landmarks[9]


            # ------------------------------------------------
            # Calculate hand scale
            # ------------------------------------------------

            scale = math.sqrt(

                (reference.x - wrist.x) ** 2
                +

                (reference.y - wrist.y) ** 2
                +

                (reference.z - wrist.z) ** 2

            )


            if scale != 0:

                features = []


                # ------------------------------------------------
                # Normalize all 21 landmarks
                # ------------------------------------------------

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


                # ------------------------------------------------
                # Convert to NumPy
                # ------------------------------------------------

                features = np.array(
                    features,
                    dtype=np.float32
                ).reshape(1, -1)


                # ------------------------------------------------
                # Make prediction
                # ------------------------------------------------

                prediction = model.predict(
                    features
                )[0]


                # ------------------------------------------------
                # Confidence
                # ------------------------------------------------

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        features
                    )[0]

                    confidence = (
                        np.max(probabilities) * 100
                    )


                # ------------------------------------------------
                # UNKNOWN threshold
                # ------------------------------------------------

                if confidence < CONFIDENCE_THRESHOLD:

                    prediction = "UNKNOWN"


        # ====================================================
        # GAME ROUND
        # ====================================================

        if round_active:

            elapsed = time.time() - round_start_time

            remaining = ROUND_TIME - int(elapsed)


            # ------------------------------------------------
            # Countdown
            # ------------------------------------------------

            if elapsed < ROUND_TIME:

                countdown_number = remaining + 1

                cv2.putText(
                    frame,
                    str(countdown_number),
                    (frame.shape[1] // 2 - 40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (0, 255, 255),
                    7
                )


            # ------------------------------------------------
            # Finish round
            # ------------------------------------------------

            else:

                round_active = False


                # Player must make a valid gesture
                if prediction in choices:

                    player_choice = prediction

                    computer_choice = random.choice(
                        choices
                    )


                    # Determine winner
                    result_text = get_winner(
                        player_choice,
                        computer_choice
                    )


                    # Update score
                    if result_text == "PLAYER WINS":

                        player_score += 1

                    elif result_text == "COMPUTER WINS":

                        computer_score += 1

                    else:

                        draw_score += 1


                else:

                    player_choice = prediction
                    computer_choice = "-"

                    result_text = "INVALID GESTURE"


        # ====================================================
        # DISPLAY PREDICTION
        # ====================================================

        cv2.putText(
            frame,
            f"Gesture: {prediction}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        if confidence > 0:

            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        # ====================================================
        # GAME DISPLAY
        # ====================================================

        cv2.putText(
            frame,
            f"Player: {player_choice}",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Computer: {computer_choice}",
            (20, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        cv2.putText(
            frame,
            result_text,
            (20, 345),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3
        )


        # ====================================================
        # SCOREBOARD
        # ====================================================

        cv2.putText(
            frame,
            f"Player: {player_score}",
            (20, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Computer: {computer_score}",
            (20, 435),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Draw: {draw_score}",
            (20, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(
            frame,
            "R = New Round",
            (20, frame.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q = Quit",
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(
            "Rock Paper Scissors Game",
            frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------------------------
        # Start new round
        # ----------------------------------------------------

        if key == ord("r"):

            round_active = True
            round_start_time = time.time()

            player_choice = "WAITING..."
            computer_choice = "WAITING..."

            result_text = "Get Ready!"


        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        elif key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nGame ended.")

print(
    f"Final Score -> "
    f"Player: {player_score}, "
    f"Computer: {computer_score}, "
    f"Draw: {draw_score}"
)