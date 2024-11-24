import torch
import torch.nn as nn

class STGCN(nn.Module):
    def __init__(self, in_channels=2, num_joints=15, num_classes=12, num_frames=960):
        """
        Spatio-Temporal Graph Convolutional Network (STGCN)

        Args:
            in_channels (int): 입력 채널 개수 (예: 2는 x, y 좌표 사용).
            num_joints (int): 관절 개수.
            num_classes (int): 분류할 행동 클래스 개수.
            num_frames (int): 입력 시퀀스의 프레임 수.
        """
        super(STGCN, self).__init__()

        # GCN 계층: 입력 채널 -> 중간 채널
        self.gcn1 = nn.Conv2d(in_channels, 64, kernel_size=(1, 1))
        self.gcn2 = nn.Conv2d(64, 128, kernel_size=(1, 1))

        # Temporal Convolution을 Conv2d로 변경하여 일관된 데이터 처리
        self.temporal_conv = nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0))  

        # Fully Connected Layer
        self.fc = nn.Linear(128 * num_frames * num_joints, num_classes)

        # 활성화 함수
        self.relu = nn.ReLU()

    def forward(self, x, adjacency_matrix):
        """
        모델의 순전파(Forward) 과정.

        Args:
            x (Tensor): 입력 데이터. 형상 (batch_size, in_channels, time, num_joints).
            adjacency_matrix (Tensor): 그래프의 인접 행렬. 형상 (num_joints, num_joints).

        Returns:
            Tensor: 출력 예측 값. 형상 (batch_size, num_classes).
        """
        # 1. 인접 행렬 검증
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1] == x.shape[-1], (
            "Adjacency matrix shape and input joints do not match."
        )

        # 2. Spatial GCN
        x = torch.einsum("bctv,nv->bctn", x, adjacency_matrix)  # 데이터 재구성 (batch, channels, time, joints)
        x = self.relu(self.gcn1(x))  # 첫 번째 GCN 레이어 적용 및 활성화
        x = self.relu(self.gcn2(x))  # 두 번째 GCN 레이어 적용 및 활성화

        # 3. Temporal Convolution
        x = self.temporal_conv(x)  # Conv2d 적용

        # 4. Flatten and Fully Connected Layer
        batch_size, channels, time, joints = x.shape
        x = x.view(batch_size, -1)  # (batch_size, channels * time * joints)로 평탄화
        x = self.fc(x)  # 완전연결계층으로 클래스 분류

        return x
