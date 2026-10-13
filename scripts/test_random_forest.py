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
# Start Webcam
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


timestamp = 0


# ============================================================
# Run
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
        # Process detected hands
        # ====================================================

        if result.hand_landmarks:

            for hand_index, hand_landmarks in enumerate(
                result.hand_landmarks
            ):

                # --------------------------------------------
                # NORMALIZATION
                # Same normalization used during training
                # --------------------------------------------

                # Landmark 0 = wrist
                wrist = hand_landmarks[0]

                # Landmark 9 = middle finger MCP
                reference = hand_landmarks[9]

                # Calculate hand size
                scale = math.sqrt(
                    (reference.x - wrist.x) ** 2 +
                    (reference.y - wrist.y) ** 2 +
                    (reference.z - wrist.z) ** 2
                )

                # Prevent division by zero
                if scale == 0:
                    continue


                # --------------------------------------------
                # Extract normalized 21 × (x,y,z)
                # = 63 features
                # --------------------------------------------

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


                # Convert to NumPy array
                features = np.array(
                    features,
                    dtype=np.float32
                ).reshape(1, -1)


                # --------------------------------------------
                # Check feature count
                # --------------------------------------------

                if features.shape[1] != 63:
                    print(
                        "ERROR: Expected 63 features, got",
                        features.shape[1]
                    )
                    continue


                # --------------------------------------------
                # Random Forest prediction
                # --------------------------------------------

                prediction = model.predict(
                    features
                )[0]


                # --------------------------------------------
                # Confidence
                # --------------------------------------------

                confidence = None

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        features
                    )[0]

                    confidence = (
                        np.max(probabilities) * 100
                    )
                    if confidence < 60:
                        prediction = "UNKNOWN"


                # --------------------------------------------
                # Draw landmarks
                # --------------------------------------------

                h, w, _ = frame.shape

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


                # --------------------------------------------
                # Display prediction
                # --------------------------------------------

                text = (
                    f"Hand {hand_index + 1}: "
                    f"{prediction}"
                )

                if confidence is not None:

                    text += (
                        f" ({confidence:.1f}%)"
                    )


                cv2.putText(
                    frame,
                    text,
                    (20, 50 + hand_index * 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )


        else:

            cv2.putText(
                frame,
                "No hand detected",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )


        # ====================================================
        # Instructions
        # ====================================================

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # Show webcam
        # ====================================================

        cv2.imshow(
            "Random Forest RPS Test",
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

print("Webcam test ended.")