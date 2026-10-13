import os
import csv
import mediapipe as mp

# --------------------------------
# Paths
# --------------------------------
INPUT_FOLDER = "frames"
OUTPUT_FILE = "dataset/landmarks.csv"
MODEL_PATH = "models/hand_landmarker.task"

# --------------------------------
# Gesture labels
# --------------------------------
LABELS = {
    "sample_01": "ROCK",
    "sample_02": "PAPER",
    "sample_03": "SCISSORS"
}

# --------------------------------
# MediaPipe setup
# --------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# --------------------------------
# Create dataset folder
# --------------------------------
os.makedirs("dataset", exist_ok=True)

# --------------------------------
# Create CSV header
# --------------------------------
header = ["image", "hand_id"]

for i in range(21):
    header.append(f"x{i}")
    header.append(f"y{i}")
    header.append(f"z{i}")

# ADD LABEL
header.append("label")

# --------------------------------
# Process images
# --------------------------------
with HandLandmarker.create_from_options(options) as landmarker:

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(header)

        total_images = 0
        detected_hands = 0
        images_without_hands = 0
        skipped_images = 0

        image_files = [
            f for f in os.listdir(INPUT_FOLDER)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        image_files.sort()

        for filename in image_files:

            image_path = os.path.join(
                INPUT_FOLDER,
                filename
            )

            try:

                # --------------------------------
                # Find gesture label
                # --------------------------------

                sample_name = filename.split("_0000")[0]

                # Better extraction from filename
                parts = filename.split("_")

                if len(parts) < 2:
                    print(f"Skipped - unknown format: {filename}")
                    skipped_images += 1
                    continue

                sample_id = parts[0] + "_" + parts[1]

                if sample_id not in LABELS:
                    print(
                        f"Skipped - no label for: {filename}"
                    )
                    skipped_images += 1
                    continue

                label = LABELS[sample_id]

                # --------------------------------
                # Load image
                # --------------------------------
                image = mp.Image.create_from_file(
                    image_path
                )

                # --------------------------------
                # Detect hands
                # --------------------------------
                result = landmarker.detect(image)

                total_images += 1

                # No hands detected
                if not result.hand_landmarks:

                    images_without_hands += 1

                    print(
                        f"No hand: {filename}"
                    )

                    continue

                # --------------------------------
                # Process each detected hand
                # --------------------------------
                for hand_id, hand_landmarks in enumerate(
                    result.hand_landmarks
                ):

                    row = [
                        filename,
                        hand_id
                    ]

                    # 21 landmarks
                    for landmark in hand_landmarks:

                        row.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])

                    # ADD LABEL
                    row.append(label)

                    writer.writerow(row)

                    detected_hands += 1

                print(
                    f"Processed: {filename} | "
                    f"Hands: {len(result.hand_landmarks)} | "
                    f"Label: {label}"
                )

            except Exception as e:

                print(
                    f"Error processing {filename}: {e}"
                )

# --------------------------------
# Results
# --------------------------------
print("\n==============================")
print("EXTRACTION COMPLETE")
print("==============================")

print("Total images:", total_images)
print("Detected hands:", detected_hands)
print("Images without hands:", images_without_hands)
print("Skipped images:", skipped_images)

print(
    "CSV saved at:",
    OUTPUT_FILE
)