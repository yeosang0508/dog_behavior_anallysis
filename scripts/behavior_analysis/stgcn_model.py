import torch
import torch.nn as nn

class UnifiedModel(nn.Module):
    """
    STGCN + SAMURAI 통합 모델
    """
    def __init__(self, in_channels, num_joints, num_classes, num_frames, frame_feature_size=None, hidden_size=64):
        super(UnifiedModel, self).__init__()
        
        # STGCN 부분 (관절 데이터 처리)
        self.stgcn_gcn1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.stgcn_gcn2 = nn.Conv2d(64, 128, kernel_size=1)
        self.stgcn_gcn3 = nn.Conv2d(128, 256, kernel_size=1)
        self.relu = nn.ReLU()
        self.stgcn_fc = nn.Linear(256 * num_frames * num_joints, hidden_size)
        
        # SAMURAI 부분 (프레임 피처 처리)
        self.use_frame_features = frame_feature_size is not None
        if self.use_frame_features:
            self.frame_dense = nn.Sequential(
                nn.Linear(frame_feature_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        
        # 융합 계층
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * (2 if self.use_frame_features else 1), hidden_size),
            nn.ReLU()
        )
        
        # 최종 출력 계층
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, skeletons, adjacency_matrix, frame_features=None):
        """
        순전파 정의
        Args:
            skeletons: 관절 데이터 텐서 (batch, in_channels, frames, joints)
            frame_features: 프레임 피처 텐서 (batch, frame_feature_size)
            adjacency_matrix: 인접 행렬 (joints, joints)
        """
        # STGCN 처리
        x = torch.einsum("bctj,jk->bctk", skeletons, adjacency_matrix)  # 그래프 컨볼루션
        x = self.relu(self.stgcn_gcn1(x))
        x = self.relu(self.stgcn_gcn2(x))
        x = self.relu(self.stgcn_gcn3(x))
        x = x.view(x.size(0), -1)  # 평탄화
        stgcn_features = self.stgcn_fc(x)  # STGCN의 출력 특징
        
        # 프레임 피처가 있으면 처리
        if frame_features is not None and frame_features.sum() != 0:  # 더미 데이터가 아닌 경우
            frame_features = self.frame_dense(frame_features)
            fused_features = torch.cat([stgcn_features, frame_features], dim=1)
        else:
            fused_features = stgcn_features

        # 융합 및 출력
        fused_output = self.fusion_layer(fused_features)
        return self.output_layer(fused_output)
