# 데이터셋
import pandas as pd
import torch
from torch.utils.data import Dataset

class BehaviorDataset(Dataset):
    def __init__(self, csv_file):
        """
        BehaviorDataset 클래스는 스켈레톤 데이터와 레이블을 처리합니다.

        Args:
            csv_file (str): 스켈레톤 데이터가 저장된 CSV 파일 경로.
        """
        # CSV 파일 읽기
        self.data = pd.read_csv(csv_file)
        print("Available columns:", self.data.columns)

        # x와 y 열 이름 설정
        self.joint_columns = [f'x{i}' for i in range(1, 30, 2)] + [f'y{i}' for i in range(1, 16)]

        # 필요한 열이 실제 데이터프레임에 존재하는지 확인
        missing_columns = [col for col in self.joint_columns if col not in self.data.columns]
        if missing_columns:
            raise KeyError(f"Missing columns in the dataset: {missing_columns}")

        # 스켈레톤 데이터를 numpy 배열로 변환 및 레이블 추출
        self.skeleton_data = self.data[self.joint_columns].values.reshape(-1, 15, 2)  # (샘플 수, 관절 개수, 좌표 수)
        
        # 문자열 레이블을 정수형으로 매핑
        self.label_mapping = {label: idx for idx, label in enumerate(self.data['label'].unique())}
        self.labels = self.data['label'].map(self.label_mapping).values  # 정수형 레이블

    def __len__(self):
        """
        데이터셋의 샘플 수를 반환합니다.
        """
        return len(self.skeleton_data)

    def __getitem__(self, idx):
        """
        주어진 인덱스의 스켈레톤 데이터와 레이블을 반환합니다.

        Args:
            idx (int): 데이터 인덱스.

        Returns:
            torch.Tensor: 스켈레톤 데이터 (형상: [15, 2]).
            torch.Tensor: 레이블 (정수형).
        """
        skeleton = torch.tensor(self.skeleton_data[idx], dtype=torch.float32)  # 스켈레톤 데이터를 Tensor로 변환
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # 정수형 레이블 데이터를 Tensor로 변환
        return skeleton, label
