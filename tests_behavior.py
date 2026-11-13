import unittest
import numpy as np
import os
import json
import shutil
from utils.logger import BehaviorLogger
from modules.phone_detection import PhoneDetector
from modules.zone_monitor import ZoneMonitor
from shapely.geometry import Polygon

class TestLogger(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_logs"
        self.logger = BehaviorLogger(log_dir=self.test_dir, log_format="json")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_jsonl_logging(self):
        event = {"test": "data"}
        self.logger.log_event("TestModule", "test_event", event)
        
        # Check if file exists and has correct extension
        files = os.listdir(self.test_dir)
        self.assertTrue(any(f.endswith(".jsonl") for f in files))
        
        filename = os.path.join(self.test_dir, files[0])
        with open(filename, 'r') as f:
            line = f.readline()
            logged_data = json.loads(line)
            self.assertEqual(logged_data["module"], "TestModule")
            self.assertEqual(logged_data["details"], event)

class TestPhoneDetector(unittest.TestCase):
    def setUp(self):
        self.detector = PhoneDetector()

    def test_overlap(self):
        person_bbox = [10, 10, 100, 100]
        phone_bbox = [50, 50, 60, 60] # Fully inside
        self.assertTrue(self.detector.check_overlap(person_bbox, phone_bbox))
        
        phone_bbox_outside = [110, 110, 120, 120]
        self.assertFalse(self.detector.check_overlap(person_bbox, phone_bbox_outside))

class TestZoneMonitor(unittest.TestCase):
    def setUp(self):
        zones_config = [
            {"name": "TestZone", "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]], "idle_threshold_seconds": 1}
        ]
        self.monitor = ZoneMonitor(zones_config)

    def test_is_in_zone(self):
        self.assertTrue(self.monitor.is_in_zone((50, 50), Polygon([[0, 0], [100, 0], [100, 100], [0, 100]])))
        self.assertFalse(self.monitor.is_in_zone((150, 150), Polygon([[0, 0], [100, 0], [100, 100], [0, 100]])))

if __name__ == "__main__":
    unittest.main()
