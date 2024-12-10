import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# 관절 수 및 관절 연결 정의
num_nodes = 15
edges = [
    (0, 1), (0, 3), (2, 3), (3, 4), (4, 5),
    (4, 6), (5, 7), (6, 8), (4, 13), (13, 9),
    (13, 10), (13, 14), (9, 11), (10, 12)
]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

class KeypointDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # 결측값 처리
        self.data.fillna(0, inplace=True)

        # 라벨 및 키포인트 데이터 추출
        self.labels = self.data['label'].values
        self.keypoints = self.data.drop(columns=['label', 'frame_number']).values.astype(float)

        # 데이터 전처리
        self.keypoints = self.preprocess_keypoints(self.keypoints, num_nodes)

    def preprocess_keypoints(self, keypoints, num_nodes):
        expected_size = num_nodes * 2
        processed_keypoints = []
        for data in keypoints:
            if len(data) > expected_size:
                data = data[:expected_size]  # 초과분 제거
            elif len(data) < expected_size:
                data = np.pad(data, (0, expected_size - len(data)), mode='constant')  # 부족분 패딩
            processed_keypoints.append(data)
        return np.array(processed_keypoints)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 키포인트 및 라벨 데이터 로드
        keypoint_data = self.keypoints[idx].reshape(num_nodes, 2)  # (x, y) 좌표
        label = self.labels[idx]

        # 텐서 변환
        x = torch.tensor(keypoint_data, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.long)

        # 그래프 데이터 생성
        graph = Data(x=x, edge_index=edge_index, y=y)
        return graph

# 데이터 로드
csv_file = r"data\csv_file\combined.csv"
dataset = KeypointDataset(csv_file)

# train-test split
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
