from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolo11n-pose.pt")

def extract_pose_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        keypoints = np.zeros((17, 3), dtype=np.float32)

        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            kp = results[0].keypoints.data[0].cpu().numpy()  # first person
            keypoints[:, :2] = kp[:, :2]
            if kp.shape[1] > 2:
                keypoints[:, 2] = kp[:, 2]
            else:
                keypoints[:, 2] = 1.0

        all_frames.append(keypoints)

    cap.release()
    return np.array(all_frames)   # shape: (T, 17, 3)