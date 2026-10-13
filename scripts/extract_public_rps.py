import os
import csv
import mediapipe as mp
import math

# =========================
# PATHS
# =========================
INPUT_FOLDER = "public_rps"
OUTPUT_FILE = "dataset/public_rps_normalized.csv"
MODEL_PATH = "models/hand_landmarker.task"

# =========================
# MEDIAPIPE
# =========================
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
    min_hand_presence_confidence=0.5,
)

os.makedirs("dataset", exist_ok=True)

# =========================
# CSV HEADER
# =========================
header = ["image", "split", "hand_id"]

for i in range(21):
    header.append(f"x{i}")
    header.append(f"y{i}")
    header.append(f"z{i}")

header.append("label")


# =========================
# PROCESS DATASET
# =========================
with HandLandmarker.create_from_options(options) as landmarker:

    with open(OUTPUT_FILE, "w", newline="") as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(header)

        total_images = 0
        detected_hands = 0
        no_hand = 0

        for split in ["train", "test"]:

            split_folder = os.path.join(INPUT_FOLDER, split)

            for label in ["rock", "paper", "scissors"]:

                class_folder = os.path.join(split_folder, label)

                if not os.path.exists(class_folder):
                    print(f"Folder not found: {class_folder}")
                    continue

                image_files = [
                    f for f in os.listdir(class_folder)
                    if f.lower().endswith(
                        (".jpg", ".jpeg", ".png")
                    )
                ]

                print(
                    f"\nProcessing {split}/{label}: "
                    f"{len(image_files)} images"
                )

                for filename in image_files:

                    image_path = os.path.join(
                        class_folder,
                        filename
                    )

                    try:

                        image = mp.Image.create_from_file(
                            image_path
                        )

                        result = landmarker.detect(image)

                        total_images += 1

                        if not result.hand_landmarks:
                            no_hand += 1
                            print(
                                f"No hand detected: "
                                f"{split}/{label}/{filename}"
                            )
                            continue

                        # Use first detected hand
                        hand_landmarks = result.hand_landmarks[0]

                        row = [
                            filename,
                            split,
                            0
                        ]

                        # Wrist = landmark 0
                        wrist = hand_landmarks[0]

# Reference point = middle finger MCP (landmark 9)
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

# Normalize all 21 landmarks
                        for landmark in hand_landmarks:

                            x = (landmark.x - wrist.x) / scale
                            y = (landmark.y - wrist.y) / scale
                            z = (landmark.z - wrist.z) / scale
                            

                            row.extend([x, y, z])

                        row.append(label.upper())

                        writer.writerow(row)

                        detected_hands += 1

                    except Exception as e:

                        print(
                            f"Error: "
                            f"{split}/{label}/{filename}"
                        )

                        print(e)


print("\n==============================")
print("EXTRACTION COMPLETE")
print("==============================")

print("Total images:", total_images)
print("Hands detected:", detected_hands)
print("Images without hand:", no_hand)

print("\nCSV saved to:")
print(OUTPUT_FILE)