import cv2
import numpy as np
from shapely.geometry import Point, Polygon
import time

class ZoneMonitor:
    def __init__(self, zones_config, logger=None):
        self.zones = []
        for z in zones_config:
            self.zones.append({
                "name": z["name"],
                "polygon": Polygon(z["polygon"]),
                "idle_threshold": z["idle_threshold_seconds"],
                "active_people": {} # id: {"last_move_time": timestamp, "last_pos": (x,y), "status": "active"|"idle"}
            })
        self.logger = logger
        self.person_last_zones = {} # person_id: zone_name

    def is_in_zone(self, point, zone_poly):
        return zone_poly.contains(Point(point))

    def process_tracks(self, tracks, frame_count):
        # tracks: list of (xyxy, id, conf)
        current_timestamp = time.time()
        
        active_ids_this_frame = [t[1] for t in tracks]
        
        # Cleanup: remove IDs that are no longer active in any zone
        for zone in self.zones:
            ids_to_remove = []
            for track_id in zone["active_people"]:
                if track_id not in active_ids_this_frame:
                    ids_to_remove.append(track_id)
            for tid in ids_to_remove:
                del zone["active_people"][tid]

        for track in tracks:
            xyxy, track_id, conf = track
            # Use center of bottom of bounding box as person's position (feet)
            pos = ((xyxy[0] + xyxy[2]) / 2, xyxy[3])
            
            in_any_zone = False
            for zone in self.zones:
                if self.is_in_zone(pos, zone["polygon"]):
                    in_any_zone = True
                    # Check for zone entry
                    if self.person_last_zones.get(track_id) != zone["name"]:
                        self._log_zone_event(track_id, "entry", zone["name"], frame_count)
                        self.person_last_zones[track_id] = zone["name"]
                    
                    # Update idle tracking
                    self._update_idle_status(track_id, pos, zone, current_timestamp, frame_count)
                    break
            
            if not in_any_zone and track_id in self.person_last_zones:
                last_zone = self.person_last_zones.pop(track_id)
                self._log_zone_event(track_id, "exit", last_zone, frame_count)

    def draw_zones(self, frame):
        for zone in self.zones:
            # Draw polygon
            poly_points = np.array(zone["polygon"].exterior.coords, np.int32)
            cv2.polylines(frame, [poly_points], True, (255, 0, 0), 2)
            
            # Label zone
            x, y = int(poly_points[0][0]), int(poly_points[0][1])
            cv2.putText(frame, zone["name"], (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Mark idle people
            for tid, state in zone["active_people"].items():
                if state["status"] == "idle":
                    pos = (int(state["last_pos"][0]), int(state["last_pos"][1]))
                    cv2.circle(frame, pos, 5, (0, 0, 255), -1)
                    cv2.putText(frame, "IDLE", (pos[0] + 10, pos[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def _update_idle_status(self, track_id, pos, zone, current_time, frame_count):
        if track_id not in zone["active_people"]:
            zone["active_people"][track_id] = {
                "last_move_time": current_time,
                "last_pos": pos,
                "status": "active"
            }
            return

        state = zone["active_people"][track_id]
        # Check if moved significantly (e.g., more than 10 pixels)
        dist = np.sqrt((pos[0] - state["last_pos"][0])**2 + (pos[1] - state["last_pos"][1])**2)
        
        if dist > 10:
            state["last_move_time"] = current_time
            state["last_pos"] = pos
            if state["status"] == "idle":
                state["status"] = "active"
                self._log_idle_event(track_id, zone["name"], "resumed", frame_count)
        else:
            idle_duration = current_time - state["last_move_time"]
            if idle_duration > zone["idle_threshold"] and state["status"] == "active":
                state["status"] = "idle"
                self._log_idle_event(track_id, zone["name"], "idle_detected", frame_count, idle_duration)

    def _log_zone_event(self, track_id, event_type, zone_name, frame):
        details = {"person_id": track_id, "zone": zone_name, "frame": frame}
        if self.logger:
            self.logger.log_event("ZoneMonitor", f"zone_{event_type}", details)

    def _log_idle_event(self, track_id, zone_name, event_type, frame, duration=None):
        details = {"person_id": track_id, "zone": zone_name, "frame": frame}
        if duration:
            details["duration_seconds"] = round(duration, 2)
        if self.logger:
            self.logger.log_event("ZoneMonitor", event_type, details)
