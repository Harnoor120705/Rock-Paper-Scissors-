# 🎮 Hand Gesture Rock Paper Scissors

A real-time **Rock Paper Scissors game using Computer Vision and Machine Learning**. The system uses **MediaPipe Hand Landmarker** to extract 21 hand landmarks, normalizes the landmark coordinates, and uses a **Random Forest classifier** to recognize Rock, Paper, and Scissors gestures through a webcam.

The application supports two game modes:

* 👤 **Human vs Computer**
* 👥 **Human vs Human**

---

## 🎯 Project Overview

The complete pipeline of the project is:

```text
Webcam
   ↓
MediaPipe Hand Landmarker
   ↓
21 Hand Landmarks
   ↓
Landmark Normalization
   ↓
63 Numerical Features
   ↓
Random Forest Classifier
   ↓
Rock / Paper / Scissors
   ↓
Confidence Filtering
   ↓
Game Logic
   ↓
Score & Result
```

The project focuses on recognizing hand gestures rather than directly classifying images. MediaPipe extracts the hand's landmark coordinates, which are then transformed into normalized numerical features for the machine learning model.

---

## ✨ Features

* 🖐️ Real-time hand detection using MediaPipe
* 📍 Detection of 21 hand landmarks
* 🔢 63 numerical features per hand
* 📐 Landmark normalization for position and scale independence
* 🌲 Random Forest gesture classifier
* 🎯 Rock, Paper, and Scissors recognition
* ❓ Unknown gesture detection using confidence threshold
* 👤 Human vs Computer mode
* 👥 Human vs Human mode
* 🔄 Gesture-change-based rounds without a countdown timer
* ⭐ 2–3 frame stability mechanism to reduce prediction fluctuations
* 🏠 Main menu for selecting game mode
* 📊 Score tracking

---

## 🧠 Technologies Used

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Main programming language      |
| MediaPipe    | Hand landmark detection        |
| OpenCV       | Webcam and image processing    |
| Scikit-learn | Random Forest machine learning |
| NumPy        | Numerical operations           |
| Pandas       | Dataset processing             |
| Pickle       | Saving/loading trained model   |

---

## 🖐️ Hand Landmark Detection

MediaPipe provides **21 landmarks** for each detected hand.

Each landmark contains:

```text
X coordinate
Y coordinate
Z coordinate
```

Therefore:

```text
21 landmarks × 3 coordinates = 63 features
```

Example landmarks:

```text
Landmark 0 → Wrist
Landmark 9 → Middle Finger MCP
```

The extracted landmarks are used as input features for the Random Forest classifier.

---

## 📐 Landmark Normalization

Raw landmark coordinates can change depending on:

* Hand position in the camera
* Distance from the camera
* Hand size
* Movement within the frame

To make the model more robust, the landmarks are normalized relative to the wrist.

The wrist is used as the reference point:

```text
x' = (x - wrist_x) / scale
y' = (y - wrist_y) / scale
z' = (z - wrist_z) / scale
```

The scale is calculated using the distance between:

```text
Wrist (Landmark 0)
        ↓
Middle Finger MCP (Landmark 9)
```

This helps the model recognize the same gesture even when the hand moves or changes size within the camera frame.

---

## 📊 Dataset

A public Rock Paper Scissors image dataset was processed using MediaPipe.

Dataset structure:

```text
public_rps/
│
├── train/
│   ├── rock/
│   ├── paper/
│   └── scissors/
│
└── test/
    ├── rock/
    ├── paper/
    └── scissors/
```

### Dataset Processing Results

| Category                     | Count |
| ---------------------------- | ----: |
| Total images                 | 2,892 |
| Hands detected               | 2,776 |
| Images without detected hand |   116 |

After MediaPipe landmark extraction:

| Gesture   |   Samples |
| --------- | --------: |
| Scissors  |       964 |
| Paper     |       958 |
| Rock      |       854 |
| **Total** | **2,776** |

Dataset split:

```text
Training samples: 2407
Testing samples:   369
```

---

## 🌲 Machine Learning Model

The project uses a:

```text
RandomForestClassifier
```

with the following configuration:

```python
n_estimators = 200
random_state = 42
```

The model receives:

```text
63 normalized landmark features
```

and predicts one of:

```text
ROCK
PAPER
SCISSORS
```

The trained model is saved as:

```text
models/rps_random_forest.pkl
```

---

## 📈 Model Performance

The latest model achieved:

### Accuracy

```text
96.75%
```

### Classification Report

| Gesture  | Precision | Recall | F1-Score |
| -------- | --------: | -----: | -------: |
| Paper    |      1.00 |   0.90 |     0.95 |
| Rock     |      0.91 |   1.00 |     0.95 |
| Scissors |      1.00 |   1.00 |     1.00 |

The results show that the normalized landmark features work effectively for distinguishing the three RPS gestures.

---

## ❓ Unknown Gesture Detection

A Random Forest classifier normally selects one of its trained classes even when the input does not represent a valid RPS gesture.

To reduce this problem, a confidence threshold is used:

```python
CONFIDENCE_THRESHOLD = 60
```

The prediction logic is:

```text
High confidence
      ↓
ROCK / PAPER / SCISSORS

Low confidence
      ↓
UNKNOWN
```

This prevents low-confidence predictions from immediately being treated as valid game gestures.

> Note: Confidence thresholding improves practical behavior but is not a perfect unknown-class detector.

---

## ⭐ Gesture Stability

Real-time webcam predictions can fluctuate between consecutive frames because of small changes in:

* Hand position
* Finger movement
* Lighting
* MediaPipe landmark detection

For example:

```text
Frame 1 → ROCK
Frame 2 → ROCK
Frame 3 → PAPER
Frame 4 → ROCK
```

Without stabilization, these temporary changes could incorrectly trigger game events.

The application therefore requires a gesture to remain consistent for approximately **2–3 consecutive frames** before accepting it as a stable gesture.

```text
Raw Prediction
      ↓
2–3 Frame Stability Check
      ↓
Stable Gesture
      ↓
Game Logic
```

This makes the real-time game more reliable without requiring a countdown timer.

---

# 🎮 Game Modes

## 👤 Human vs Computer

The player shows a gesture to the webcam.

```text
Player Hand
     ↓
MediaPipe
     ↓
Normalization
     ↓
Random Forest
     ↓
Player Gesture
     ↓
Computer Random Gesture
     ↓
Winner Calculation
     ↓
Score
```

The computer randomly selects:

```text
ROCK
PAPER
SCISSORS
```

Example:

```text
Player:    ROCK
Computer:  SCISSORS

Result: PLAYER WINS
```

---

## 👥 Human vs Human

Two players can play using the webcam.

MediaPipe is configured to detect up to two hands:

```python
num_hands = 2
```

The pipeline is:

```text
Player 1 Hand ──→ MediaPipe ──→ Normalization ──→ Random Forest
                                                        ↓
                                                   Gesture 1

Player 2 Hand ──→ MediaPipe ──→ Normalization ──→ Random Forest
                                                        ↓
                                                   Gesture 2
                                                        ↓
                                                   Game Logic
                                                        ↓
                                                     Score
```

Each detected hand is classified independently.

---

## 🏠 Main Application

The application will provide a main page where the player can select a game mode:

```text
╔════════════════════════════════╗
║                                ║
║     ROCK PAPER SCISSORS        ║
║          AI GAME               ║
║                                ║
║   [ HUMAN VS COMPUTER ]        ║
║                                ║
║   [ HUMAN VS HUMAN ]           ║
║                                ║
╚════════════════════════════════╝
```

The user can return to the home page and switch between the two game modes.

---

## 📁 Project Structure

```text
Hand-Pose-rps-YOLO_AI/
│
├── models/
│   ├── hand_landmarker.task
│   └── rps_random_forest.pkl
│
├── public_rps/
│   ├── train/
│   │   ├── rock/
│   │   ├── paper/
│   │   └── scissors/
│   │
│   └── test/
│       ├── rock/
│       ├── paper/
│       └── scissors/
│
├── dataset/
│   └── public_rps_normalized.csv
│
├── scripts/
│   ├── extract_landmarks.py
│   ├── train_model.py
│   ├── test_random_forest.py
│   ├── visualize_landmarks.py
│   ├── human_vs_computer.py
│   └── human_vs_human.py
│
├── app/
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/AzamFida/Hand-Gesture-Rock-Paper-Scissors-Mediapipe.git
```

### 2. Navigate to the project

```bash
cd Hand-Gesture-Rock-Paper-Scissors-Mediapipe
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

After installing the dependencies, run the appropriate application script.

For example:

```bash
python app/main.py
```

The application will open the webcam and allow the user to select a game mode.

---

## 🔬 Model Training

To train the Random Forest model from the extracted landmark dataset:

```bash
python scripts/train_model.py
```

The trained model will be saved as:

```text
models/rps_random_forest.pkl
```

---

## 🧪 Model Testing

To evaluate the trained model:

```bash
python scripts/test_random_forest.py
```

The evaluation provides:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report

---

## 🔄 Complete System Architecture

```text
                    ┌──────────────────┐
                    │    Webcam Input  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │     MediaPipe    │
                    │  Hand Landmarker │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │  21 Landmarks    │
                    │  X, Y, Z         │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │  Normalization   │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │  63 Features     │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │  Random Forest   │
                    └────────┬─────────┘
                             ↓
               ┌─────────────┼─────────────┐
               ↓             ↓             ↓
             ROCK          PAPER        SCISSORS
               └─────────────┼─────────────┘
                             ↓
                    ┌──────────────────┐
                    │ Confidence Check │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ 2–3 Frame Stable │
                    │    Prediction    │
                    └────────┬─────────┘
                             ↓
                  ┌──────────┴──────────┐
                  ↓                     ↓
          Human vs Computer      Human vs Human
                  ↓                     ↓
             Game Logic             Game Logic
                  └──────────┬──────────┘
                             ↓
                       Score / Result
```

---

## 🎯 Future Improvements

Possible future improvements include:

* Improve unknown gesture detection
* Add more diverse hand poses to the dataset
* Add hand tracking between frames
* Improve two-hand identification
* Add game statistics
* Add sound effects
* Improve UI/UX
* Add difficulty levels for Human vs Computer
* Package the application for easier distribution

---

## 👨‍💻 Author

**Azam Fida**

Flutter Developer | Computer Vision & Machine Learning Enthusiast

---

## 📜 License

This project is developed for educational and portfolio purposes.
