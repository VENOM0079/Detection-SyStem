import cv2
import numpy as np
import joblib
import os

try:
    import mediapipe as mp
    try:
        import mediapipe.python.solutions.holistic as mp_holistic
    except (ImportError, AttributeError):
        try:
            import mediapipe.solutions.holistic as mp_holistic
        except (ImportError, AttributeError):
            mp_holistic = None
except (ImportError, AttributeError, TypeError):
    mp = None
    mp_holistic = None

class GestureRecognizer:
    def __init__(self, model_path=None, confidence=0.7, logger=None):
        self.mp_holistic = mp_holistic
        if self.mp_holistic:
            self.holistic = self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.holistic = None
            print("Warning: MediaPipe Holistic solutions not found. Gesture recognition will be skipped.")
        self.logger = logger
        self.confidence_threshold = confidence
        self.labels = ["neutral", "argument", "aggressive"]
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = None
            print(f"Warning: Gesture model not found at {model_path}. Inference will be skipped.")

    def extract_keypoints(self, results):
        # Extract pose, face, and hand keypoints and flatten them
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def recognize(self, frame, frame_count, person_id=None, roi=None):
        # roi: [x1, y1, x2, y2]
        working_frame = frame
        if roi is not None:
            x1, y1, x2, y2 = map(int, roi)
            # Ensure ROI is within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                working_frame = frame[y1:y2, x1:x2]

        if self.holistic is None:
            return "neutral", 1.0, None

        # MediaPipe expects RGB
        image = cv2.cvtColor(working_frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        
        if self.model:
            keypoints = self.extract_keypoints(results)
            # Reshape for prediction
            prediction = self.model.predict_proba([keypoints])[0]
            max_prob = np.max(prediction)
            label_idx = np.argmax(prediction)
            label = self.labels[label_idx]
            
            if max_prob > self.confidence_threshold:
                event_detail = {
                    "frame": frame_count,
                    "label": label,
                    "probability": round(float(max_prob), 2),
                    "person_id": int(person_id) if person_id is not None else None
                }
                if label != "neutral":
                    if self.logger:
                        self.logger.log_event("GestureRecognition", "gesture_detected", event_detail)
                    return label, max_prob, event_detail
                    
        return "neutral", 1.0, None

    def close(self):
        if self.holistic:
            self.holistic.close()
