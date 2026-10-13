import os
import cv2
import mediapipe as mp

INPUT_FOLDER = "public_rps/train"
OUTPUT_FOLDER = "landmark_check"
MODEL_PATH = "models/hand_landmarker.task"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)


# MediaPipe hand connections
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


with HandLandmarker.create_from_options(options) as landmarker:

    for label in ["rock", "paper", "scissors"]:

        input_class = os.path.join(INPUT_FOLDER, label)
        output_class = os.path.join(OUTPUT_FOLDER, label)

        os.makedirs(output_class, exist_ok=True)

        images = [
            f for f in os.listdir(input_class)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        # Check first 10 images from each class
        for filename in images[:10]:

            image_path = os.path.join(
                input_class,
                filename
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            height, width = image.shape[:2]

            mp_image = mp.Image.create_from_file(
                image_path
            )

            result = landmarker.detect(mp_image)

            if result.hand_landmarks:

                landmarks = result.hand_landmarks[0]

                points = []

                # Draw points
                for landmark in landmarks:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    points.append((x, y))

                    cv2.circle(
                        image,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Draw connections
                for start, end in CONNECTIONS:

                    cv2.line(
                        image,
                        points[start],
                        points[end],
                        (255, 0, 0),
                        2
                    )

                # Show class
                cv2.putText(
                    image,
                    label.upper(),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            else:

                cv2.putText(
                    image,
                    "NO HAND DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            output_path = os.path.join(
                output_class,
                filename
            )

            cv2.imwrite(
                output_path,
                image
            )

print("\nVisualization complete!")
print("Open the 'landmark_check' folder.")