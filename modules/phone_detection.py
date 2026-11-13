from ultralytics import YOLO
import cv2
import numpy as np

class PhoneDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.5, logger=None):
        self.model_path = model_path
        self._model = None
        self.confidence = confidence
        self.logger = logger
        self.person_class_id = 0
        self.phone_class_id = 67  # COCO dataset class ID for cell phone

    @property
    def model(self):
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model
        
    def detect(self, frame):
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = results[0].boxes
        
        people = []
        phones = []
        
        for box in detections:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            if cls == self.person_class_id:
                people.append({"bbox": xyxy, "conf": conf, "id": None}) # ID added during tracking
            elif cls == self.phone_class_id:
                phones.append({"bbox": xyxy, "conf": conf})
                
        return people, phones

    def check_overlap(self, person_bbox, phone_bbox):
        # person_bbox: [x1, y1, x2, y2]
        # Check if phone's center is within person's bounding box
        # Or simple Intersection Over Area of phone
        px1, py1, px2, py2 = person_bbox
        phx1, phy1, phx2, phy2 = phone_bbox
        
        # Phone center
        cx = (phx1 + phx2) / 2
        cy = (phy1 + phy2) / 2
        
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            return True
        return False

    def process_frame(self, frame, frame_count, detections, tracks):
        # detections: from the main YOLO model (results[0].boxes)
        # tracks: list of (bbox, id, conf) for people
        
        phones = []
        for box in detections:
            cls = int(box.cls[0])
            if cls == self.phone_class_id:
                phones.append({
                    "bbox": box.xyxy[0].cpu().numpy().astype(int),
                    "conf": float(box.conf[0])
                })

        events = []
        for p_bbox, p_id, p_conf in tracks:
            for phone in phones:
                ph_bbox = phone["bbox"]
                if self.check_overlap(p_bbox, ph_bbox):
                    event_detail = {
                        "frame": frame_count,
                        "person_id": int(p_id),
                        "person_conf": float(p_conf),
                        "phone_conf": phone["conf"],
                        "person_bbox": p_bbox.tolist(),
                        "phone_bbox": ph_bbox.tolist()
                    }
                    
                    if self.logger:
                        self.logger.log_event("PhoneUsage", "usage_detected", event_detail)
                    events.append(event_detail)
                    break # One phone per person alert
                    
        return events, phones
