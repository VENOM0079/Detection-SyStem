import os
import numpy as np
import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Use the same extractor as in the module
from modules.gesture_recognition import GestureRecognizer

def collect_data(data_path):
    """
    Expects data_path to have subfolders: neutral, argument, aggressive
    Each subfolder contains video clips or images.
    """
    recognizer = GestureRecognizer() # Helper for extraction
    X, y = [], []
    labels = ["neutral", "argument", "aggressive"]
    
    for label_idx, label in enumerate(labels):
        class_path = os.path.join(data_path, label)
        if not os.path.exists(class_path): continue
        
        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)
            # Simple implementation: one keypoint vector per file (assuming images or single frame extraction)
            # For videos, you'd iterate through frames
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = recognizer.holistic.process(image)
                keypoints = recognizer.extract_keypoints(results)
                X.append(keypoints)
                y.append(label_idx)
            cap.release()
            
    recognizer.close()
    return np.array(X), np.array(y)

def train_model(X, y, save_path="models/gesture_classifier.joblib"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f"Training Complete. Accuracy: {accuracy_score(y_test, y_pred)}")
    
    # Create models directory if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(clf, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    print("Gesture Classifier Training Script")
    # Example:
    # X, y = collect_data("data/gesture_dataset")
    # train_model(X, y)
