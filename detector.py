import cv2
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np
import cvzone
import time

class ShopliftingDetector:
    def __init__(self, model_path='trained_model.json', yolo_path='yolo11n-pose.pt'):
        self.model_yolo = YOLO(yolo_path)
        self.model = xgb.Booster()
        try:
            self.model.load_model(model_path)
        except Exception as e:
            print(f"Error loading XGBoost model: {e}")
            raise e

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_tot = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Resize for consistent processing
            frame = cv2.resize(frame, (1018, 600))
            
            # Run YOLO
            results = self.model_yolo(frame, verbose=False)
            annotated_frame = results[0].plot(boxes=False)
            
            detections = []
            
            for r in results:
                bound_box = r.boxes.xyxy
                conf = r.boxes.conf.tolist()
                keypoints = r.keypoints.xyn.tolist()
                
                for index, box in enumerate(bound_box):
                    if conf[index] > 0.55:
                        x1, y1, x2, y2 = box.tolist()
                        
                        # Prepare data
                        data = {}
                        if len(keypoints[index]) == 0:
                            continue

                        for j in range(len(keypoints[index])):
                            data[f'x{j}'] = keypoints[index][j][0]
                            data[f'y{j}'] = keypoints[index][j][1]
                        
                        df = pd.DataFrame(data, index=[0])
                        dmatrix = xgb.DMatrix(df)
                        
                        sus = self.model.predict(dmatrix)
                        # Handle prediction format
                        binary_predictions = (sus > 0.5).astype(int)
                        pred = int(binary_predictions[0]) if hasattr(binary_predictions, '__len__') else int(binary_predictions)
                        
                        label = "Suspicious" if pred == 0 else "Normal"
                        color = (0, 0, 255) if pred == 0 else (0, 255, 0)
                        
                        # Draw bounding box and label
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cvzone.putTextRect(annotated_frame, label, (int(x1), int(y1)), 1, 1)
                        
                        if pred == 0:
                            detections.append({
                                "time": time.strftime("%H:%M:%S"),
                                "frame": frame_tot,
                                "type": "Suspicious Behavior",
                                "confidence": float(sus[0]) if hasattr(sus, '__len__') else float(sus)
                            })
            
            frame_tot += 1
            yield annotated_frame, detections

        cap.release()
