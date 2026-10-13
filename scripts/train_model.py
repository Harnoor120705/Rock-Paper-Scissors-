import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "dataset/public_rps_normalized.csv"
MODEL_PATH = "models/rps_random_forest.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nSplit distribution:")
print(df["split"].value_counts())


# =========================
# FEATURES AND LABEL
# =========================

# Remove metadata columns
X = df.drop(columns=["image", "split", "label", "hand_id"])
y = df["label"]
print("\nClass distribution:")
print(df["label"].value_counts())


# =========================
# TRAIN / TEST SPLIT
# =========================

train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

X_train = train_df.drop(columns=["image", "split", "hand_id", "label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["image", "split", "hand_id", "label"])
y_test = test_df["label"]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# RANDOM FOREST
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel training completed.")


# =========================
# EVALUATION
# =========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================
# SAVE MODEL
# =========================

joblib.dump(model, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)