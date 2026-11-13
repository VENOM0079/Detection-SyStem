# Modular Behavior Detection System

A comprehensive system for detecting phone usage, monitoring zones for idle behavior, and recognizing aggressive gestures.

## Features
- **Phone Usage Detection**: Uses YOLOv8 to detect overlap between person and phone bounding boxes.
- **Zone Monitoring**: Defines polygon zones and tracks people using ByteTrack. Detects zone entry/exit and idle duration.
- **Gesture Recognition**: Uses MediaPipe Holistic for pose estimation and a RandomForest classifier for identifying aggressive gestures.
- **Unified Logging**: All events are logged to a central JSON/CSV file with timestamps and person IDs.

## Installation
1. Install dependencies:
   ```bash
   pip install ultralytics mediapipe shapely pyyaml joblib scikit-learn opencv-python numpy
   ```
2. Ensure you have the YOLOv8 weights (`yolov8n.pt`).

## Configuration
Edit `config.yaml` to set:
- Video source (webcam index or file path).
- Zone coordinates (polygons).
- Model thresholds and paths.

## Usage
### Running the System
```bash
python main.py
```

## Training Roadmap

This section provides a complete guide on how to train the specialized models for Phone Detection and Gesture Recognition.

### 1. Training the Phone Detector (YOLOv8)
To improve phone detection accuracy beyond the base COCO model, you can fine-tune YOLOv8.

#### Step A: Data Preparation
1. **Gather Data**: Collect images of people using phones in your specific environment (e.g., office, retail).
2. **Annotation**: Use a tool like [LabelImg](https://github.com/HumanSignal/labelImg) or [Roboflow](https://roboflow.com/) to label your images.
   - **Classes**: `0: person`, `1: cell phone`.
3. **Format**: Export in **YOLO format**. Your folder structure should look like this:
   ```text
   dataset/
     train/
       images/ (jpg/png files)
       labels/ (txt files)
     val/
       images/
       labels/
   ```
4. **Configuration**: Create a `data.yaml` file:
   ```yaml
   train: ./dataset/train/images
   val: ./dataset/val/images
   nc: 2
   names: ['person', 'cell phone']
   ```

#### Step B: Run Training
Use the provided template `train_phone_yolo.py` or run directly via CLI:
```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```
- **Output**: Your trained model will be in `runs/detect/train/weights/best.pt`.
- **Deployment**: Copy `best.pt` to `models/` and update `config.yaml` with the new path.

---

### 2. Training the Gesture Classifier (RandomForest)
The gesture classifier recognizes specific actions like "aggressive" or "argument" based on skeletal data.

#### Step A: Data Preparation
1. **Gather Videos**: Collect short video clips (1-3 seconds) for each category.
2. **Organize Folders**:
   ```text
   data/
     gesture_dataset/
       neutral/    (clips of normal behavior)
       argument/   (clips of intense discussion/gestures)
       aggressive/ (clips of physical movement/hitting)
   ```

#### Step B: Feature Extraction & Training
1. **Run the training script**:
   ```bash
   python train_gesture_classifier.py
   ```
   - This script uses `modules/gesture_recognition.py` to extract 1600+ keypoints (Pose, Face, Hands) from each video frame.
   - It trains a **RandomForest Classifier** on these flattened vectors.
2. **Output**: The trained model is saved as `models/gesture_classifier.joblib`.
3. **Deployment**: Ensure the `gesture_model_path` in `config.yaml` points to this file.

---

## Modular Structure
- `modules/phone_detection.py`: YOLOv8 detection and overlap logic.
- `modules/zone_monitor.py`: Polygon intersection and idle tracking.
- `modules/gesture_recognition.py`: MediaPipe keypoint extraction and inference.
- `utils/logger.py`: Centralized audit logging.
- `train_phone_yolo.py`: Template for YOLOv8 fine-tuning.
- `train_gesture_classifier.py`: Script for training the gesture ML model.
