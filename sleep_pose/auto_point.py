import cv2
import mediapipe as mp
import os
import json

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

image_folder = "data"
output_json = "annotations.json"

annotations = []

for pose_label in os.listdir(image_folder):
    pose_folder = os.path.join(image_folder, pose_label)

    if os.path.isdir(pose_folder):

        for image_name in os.listdir(pose_folder):
            image_path = os.path.join(pose_folder, image_name)

            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                keypoints = {
                    "nose": None,
                    "left_shoulder": None,
                    "right_shoulder": None,
                    "left_hip": None,
                    "right_hip": None,
                    "left_knee": None,
                    "right_knee": None,
                    "left_ankle": None,
                    "right_ankle": None,
                    "left_elbow": None,
                    "right_elbow": None,
                    "left_wrist": None,
                    "right_wrist": None
                }

            if results.pose_landmarks:
                nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                

                annotations.append({
                    "image_name": image_name,
                    "pose_label": pose_label,  
                    "keypoints": {
                        "nose": {"x": nose.x, "y": nose.y},
                        "left_shoulder": {"x": left_shoulder.x, "y": left_shoulder.y},
                        "right_shoulder": {"x": right_shoulder.x, "y": right_shoulder.y},
                        "left_hip": {"x": left_hip.x, "y": left_hip.y},
                        "right_hip": {"x": right_hip.x, "y": right_hip.y},
                        "left_knee": {"x": left_knee.x, "y": left_knee.y},
                        "right_knee": {"x": right_knee.x, "y": right_knee.y},
                        "left_ankle": {"x": left_ankle.x, "y": left_ankle.y},
                        "right_ankle": {"x": right_ankle.x, "y": right_ankle.y},
                        "left_elbow": {"x": left_elbow.x, "y": left_elbow.y},
                        "right_elbow": {"x": right_elbow.x, "y": right_elbow.y},
                        "left_wrist": {"x": left_wrist.x, "y": left_wrist.y},
                        "right_wrist": {"x": right_wrist.x, "y": right_wrist.y}

                    }
                })

with open(output_json, 'w') as f:
    json.dump(annotations, f, indent=4)

print(f"save to  {output_json}")
