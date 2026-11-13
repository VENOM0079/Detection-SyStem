import cv2
import yaml
import time
import argparse
import numpy as np
from utils.logger import BehaviorLogger
from utils.reporter import LogReporter
from modules.phone_detection import PhoneDetector
from modules.zone_monitor import ZoneMonitor
from modules.gesture_recognition import GestureRecognizer
from ultralytics import YOLO

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Behavior Detection System")
    parser.add_argument("--report", action="store_true", help="Generate a CSV report from existing logs")
    args = parser.parse_args()

    # Handle Report Generation
    if args.report:
        print("Generating CSV report from logs...")
        reporter = LogReporter()
        report_path = reporter.generate_report()
        if report_path:
            print(f"Report successfully generated at: {report_path}")
        return

    # Load Configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize Logger
    logger = BehaviorLogger(log_format=config["output_log_format"])
    
    # Initialize Modules
    phone_detector = PhoneDetector(
        model_path=config["phone_model_path"],
        confidence=config["phone_confidence_threshold"],
        logger=logger
    )
    
    zone_monitor = ZoneMonitor(
        zones_config=config["zones"],
        logger=logger
    )
    
    gesture_recognizer = GestureRecognizer(
        model_path=config["gesture_model_path"],
        confidence=config["gesture_confidence_threshold"],
        logger=logger
    )

    # Main Model for Tracking and Detection
    main_model = YOLO("yolov8n.pt") 

    # Input Source
    input_src = config["input_source"]
    cap = cv2.VideoCapture(input_src)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source '{input_src}'.")
        print("Please check if your camera is connected or if the file path is correct in 'config.yaml'.")
        print("If using a webcam, try changing 'input_source' to 1 or 2.")
        return

    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 1. Detection & Tracking (Person class=0)
            results = main_model.track(frame, persist=True, verbose=False)
            
            tracks = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, track_id, conf, cls in zip(boxes, ids, confs, clss):
                    if cls == 0: # Person
                        tracks.append((box, int(track_id), float(conf)))

            # 2. Zone Monitoring
            zone_monitor.process_tracks(tracks, frame_count)

            # 3. Phone Usage Detection
            # We pass the detections from the main model to avoid redundant inference
            phone_events, phone_detections = phone_detector.process_frame(frame, frame_count, results[0].boxes, tracks)

            # 4. Gesture Recognition
            gesture_alerts = []
            for box, track_id, _ in tracks:
                label, prob, alert = gesture_recognizer.recognize(frame, frame_count, track_id, roi=box)
                if alert:
                    gesture_alerts.append((box, label))

            # --- Visualization ---
            zone_monitor.draw_zones(frame)
            
            for box, track_id, conf in tracks:
                color = (0, 255, 0)
                has_phone = any(e['person_id'] == track_id for e in phone_events)
                if has_phone:
                    color = (0, 255, 255)
                    cv2.putText(frame, "PHONE", (box[0], box[3] + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                for g_box, label in gesture_alerts:
                    if np.array_equal(box, g_box):
                        color = (0, 0, 255)
                        cv2.putText(frame, label.upper(), (box[0], box[1] - 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for ph in phone_detections:
                pb = ph["bbox"]
                cv2.rectangle(frame, (pb[0], pb[1]), (pb[2], pb[3]), (255, 255, 0), 1)

            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Behavior Detection System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        gesture_recognizer.close()

if __name__ == "__main__":
    main()
