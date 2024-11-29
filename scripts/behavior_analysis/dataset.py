import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class BehaviorDataset(Dataset):
    def __init__(self, csv_file, num_frames=30, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # 관절 데이터 열 이름 정의
        self.joint_columns = [
            "x1", "y1", "x3", "y2", "x5", "y3", "x7", "y4", "x9", "y5",
            "x11", "y6", "x13", "y7", "x15", "y8", "x17", "y9", "x19", "y10",
            "x21", "y11", "x23", "y12", "x25", "y13", "x27", "y14", "x29", "y15"
        ]

        # 관절 데이터 읽기 및 변환
        skeleton_data = self.data[self.joint_columns].values.reshape(-1, 15, 2)  # (samples, joints, 2)
        self.skeleton_data = np.expand_dims(skeleton_data, axis=2).repeat(num_frames, axis=2)  # (samples, joints, frames, 2)

        # 라벨 매핑
        label_mapping = {label: idx for idx, label in enumerate(self.data['label'].unique())}
        self.labels = self.data['label'].map(label_mapping).values

    def __len__(self):
        return len(self.skeleton_data)

    def __getitem__(self, idx):
        skeleton = torch.tensor(self.skeleton_data[idx], dtype=torch.float32)  # (joints, frames, 2)
        skeleton = skeleton.permute(2, 1, 0)  # (joints, frames, 2) -> (2, frames, joints)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # 프레임 피처를 더미 데이터로 설정 (예: 0으로 채운 텐서)
        frame_features = torch.zeros(128, dtype=torch.float32)


        return skeleton, frame_features, label
