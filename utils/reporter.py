import os
import json
import csv
from datetime import datetime

class LogReporter:
    def __init__(self, log_dir="logs", output_dir="output"):
        self.log_dir = log_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _parse_line(self, line):
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            return None

    def _flatten_details(self, details):
        if isinstance(details, dict):
            return "; ".join([f"{k}: {v}" for k, v in details.items()])
        return str(details)

    def generate_report(self, output_filename="detections_report.csv"):
        all_events = []
        
        if not os.path.exists(self.log_dir):
            print(f"Warning: Log directory '{self.log_dir}' not found.")
            return None

        # Scan for log files
        for filename in os.listdir(self.log_dir):
            filepath = os.path.join(self.log_dir, filename)
            
            if filename.endswith(".jsonl"):
                with open(filepath, 'r') as f:
                    for line in f:
                        event = self._parse_line(line)
                        if event:
                            all_events.append(event)
            
            elif filename.endswith(".json"):
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_events.extend(data)
                        elif isinstance(data, dict):
                            all_events.append(data)
                    except json.JSONDecodeError:
                        print(f"Error parsing {filename}")

        if not all_events:
            print("No detection logs found to report.")
            return None

        # Sort events by timestamp
        try:
            all_events.sort(key=lambda x: x.get("timestamp", ""))
        except Exception as e:
            print(f"Warning: Could not sort events: {e}")

        output_path = os.path.join(self.output_dir, output_filename)
        
        fieldnames = ["timestamp", "module", "event_type", "details"]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in all_events:
                row = {
                    "timestamp": event.get("timestamp"),
                    "module": event.get("module"),
                    "event_type": event.get("event_type"),
                    "details": self._flatten_details(event.get("details"))
                }
                writer.writerow(row)
        
        return output_path

if __name__ == "__main__":
    reporter = LogReporter()
    report_path = reporter.generate_report()
    if report_path:
        print(f"Report generated successfully: {report_path}")
