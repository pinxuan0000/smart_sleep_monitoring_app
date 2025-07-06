import cv2
import mediapipe as mp
import torch
import numpy as np
from model import SleepPostureModel

# Load model
model = SleepPostureModel()
model.load_state_dict(torch.load("sleep_pose/model/sleep_posture_model_99.pth"))
model.eval()

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Labels
label_map = {0: "left_side", 1: "right_side", 2: "supine", 3: "lie"}

# rtsp_url = "rtsp://Miswork:4080520@172.20.10.2:554/live/profile0/stream2" #用監視器
rtsp_url=0
# Camera setup
cap = cv2.VideoCapture(rtsp_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        keypoints = results.pose_landmarks.landmark
        input_tensor = []

        for i in [ 0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28 ]:  # Relevant landmarks
            input_tensor.extend([keypoints[i].x, keypoints[i].y])

            # Draw circle for each keypoint
            h, w, _ = frame.shape
            x, y = int(keypoints[i].x * w), int(keypoints[i].y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green circle for the landmark

        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()

        label = label_map[predicted_label]
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Sleep Posture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
