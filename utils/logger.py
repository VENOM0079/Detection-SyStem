import json
import csv
import os
import datetime
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class BehaviorLogger:
    def __init__(self, log_dir="logs", log_format="json"):
        self.log_dir = log_dir
        self.log_format = log_format.lower()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "jsonl" if self.log_format == "json" else self.log_format
        self.filename = os.path.join(log_dir, f"audit_log_{timestamp}.{ext}")
        
    def log_event(self, module, event_type, details):
        event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "module": module,
            "event_type": event_type,
            "details": details
        }
        
        if self.log_format == "json":
            self._log_jsonl(event)
        else:
            self._log_csv(event)
            
    def _log_jsonl(self, event):
        with open(self.filename, 'a') as f:
            f.write(json.dumps(event, cls=NpEncoder) + "\n")
            
    def _log_csv(self, event):
        file_exists = os.path.exists(self.filename)
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "module", "event_type", "details"])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": event["timestamp"],
                "module": event["module"],
                "event_type": event["event_type"],
                "details": str(event["details"])
            })
