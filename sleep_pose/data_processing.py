# dataprocessing.py
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os

def normalize_keypoints(keypoints):
    base_x, base_y = keypoints["right_hip"]["x"], keypoints["right_hip"]["y"]
    for key in keypoints:
        keypoints[key]["x"] -= base_x
        keypoints[key]["y"] -= base_y
    return keypoints

def augment_keypoints(keypoints):
    # 隨機旋轉
    angle = random.uniform(-15, 15)  # 旋轉角度（-15度到15度之間）
    rad = np.deg2rad(angle)
    cos_val = np.cos(rad)
    sin_val = np.sin(rad)
    
    rotated_keypoints = {}
    for k, v in keypoints.items():
        x, y = v["x"], v["y"]
        new_x = x * cos_val - y * sin_val
        new_y = x * sin_val + y * cos_val
        rotated_keypoints[k] = {"x": new_x, "y": new_y}

    # 隨機縮放
    scale = random.uniform(0.9, 1.1)  # 在 90% 到 110% 之間
    for k in rotated_keypoints:
        rotated_keypoints[k]["x"] *= scale
        rotated_keypoints[k]["y"] *= scale

    # 隨機平移
    trans_x = random.uniform(-0.1, 0.1)  # 可微調
    trans_y = random.uniform(-0.1, 0.1)
    for k in rotated_keypoints:
        rotated_keypoints[k]["x"] += trans_x
        rotated_keypoints[k]["y"] += trans_y

    return rotated_keypoints

class SleepPostureDataset(Dataset):
    

    def __init__(self, annotation_file, augment=False):
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.augment = augment  # 新增參數控制要不要做 data augmentation

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        keypoints = annotation['keypoints']
        pose_label = annotation['pose_label']

        keypoints = normalize_keypoints(keypoints)

        if self.augment:
            keypoints = augment_keypoints(keypoints)

        keypoint_values = []
        for point in keypoints.values():
            keypoint_values.extend([point['x'], point['y']])
        keypoint_tensor = torch.tensor(keypoint_values, dtype=torch.float32)

        label_map = {"left": 0, "right": 1, "supine": 2, "lie": 3}
        if pose_label not in label_map:
            raise ValueError(f"Unexpected pose_label: {pose_label}")
        
        label_tensor = torch.tensor(label_map[pose_label], dtype=torch.long)

        return keypoint_tensor, label_tensor

def get_dataloader(annotation_file, batch_size=32, shuffle=True, augment=False):
    dataset = SleepPostureDataset(annotation_file, augment=augment)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
