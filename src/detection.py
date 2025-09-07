# detection.py
import cv2
from ultralytics import YOLO
import torch
import ultralytics.nn.tasks
import ultralytics.nn.modules  # for Conv
import torch.nn.modules.container  # for Sequential if used

class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # ✅ Add safe globals BEFORE loading the model
        torch.serialization.add_safe_globals([
            ultralytics.nn.tasks.DetectionModel,
            ultralytics.nn.modules.Conv,
            torch.nn.modules.container.Sequential
        ])
        
        # Load YOLO model
        self.model = YOLO(model_path)

    def detect_vehicles(self, frame):
        results = self.model(frame, verbose=False)
        lane_counts = {"A":0, "B":0, "C":0, "D":0}
        emergency_detected = False
        annotated_frame = frame.copy()
        frame_width = frame.shape[1]

        for box in results[0].boxes:
            cls = int(box.cls)
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            if cls in [2,3,5,7]:
                label = {2:"Car", 3:"Bike", 5:"Bus", 7:"Truck"}[cls]

                # Assign lane based on x-coordinate of box center
                x_center = (xyxy[0] + xyxy[2]) // 2
                if x_center < frame_width * 0.25:
                    lane = "A"
                elif x_center < frame_width * 0.5:
                    lane = "B"
                elif x_center < frame_width * 0.75:
                    lane = "C"
                else:
                    lane = "D"

                lane_counts[lane] += 1

                if cls == 5:  # Bus = emergency
                    emergency_detected = True

                cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0,255,0), 2)
                cv2.putText(annotated_frame, label, (xyxy[0], xyxy[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        return lane_counts, emergency_detected, annotated_frame
