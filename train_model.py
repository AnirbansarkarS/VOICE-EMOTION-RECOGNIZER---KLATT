import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from utils.audio_features import extract_features

# Mapping RAVDESS emotion codes to labels
emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful"
    # Can include 07 (disgust) or 08 (surprised) if needed
}

X, y = [], []

# Loop through data/ravdess directory
for root, _, files in os.walk("data/ravdess"):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            if emotion_code in emotions:
                path = os.path.join(root, file)
                try:
                    features = extract_features(path)
                    X.append(features)
                    y.append(emotions[emotion_code])
                except Exception as e:
                    print(f"Error with {path}: {e}")

# Convert to NumPy arrays
X, y = np.array(X), np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/emotion_model.pkl")
print("✅ Model trained and saved to models/emotion_model.pkl")
