import torch
import torch.nn as nn

class UnifiedModel3DCNN(nn.Module):
    """
    STGCN + SAMURAI 통합 모델 (3D CNN + 동적 피처 추출기)
    """
    def __init__(self, in_channels, num_joints, num_classes, num_frames, hidden_size=64):
        super(UnifiedModel3DCNN, self).__init__()

        # STGCN 부분
        self.stgcn_gcn1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.stgcn_gcn2 = nn.Conv2d(64, 128, kernel_size=1)
        self.stgcn_gcn3 = nn.Conv2d(128, 256, kernel_size=1)
        self.relu = nn.ReLU()

        # AdaptiveAvgPool2d로 STGCN 출력 크기 고정
        self.stgcn_pool = nn.AdaptiveAvgPool2d((num_frames // 2, num_joints // 2))
        self.stgcn_fc = nn.Linear((num_frames // 2) * (num_joints // 2) * 256, hidden_size)

        # 3D CNN 기반 SAMURAI-like 동적 피처 추출기
        self.dynamic_feature_extractor = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # dynamic_features 출력 크기 고정
        )

        # 융합 계층
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size + 128, hidden_size),  # STGCN(256->hidden_size) + SAMURAI(128)
            nn.ReLU()
        )

        # 최종 출력 계층
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, skeletons, adjacency_matrix):
        # STGCN 처리
        x = torch.einsum("bctj,jk->bctk", skeletons, adjacency_matrix)
        x = self.relu(self.stgcn_gcn1(x))
        x = self.relu(self.stgcn_gcn2(x))
        x = self.relu(self.stgcn_gcn3(x))
        x = self.stgcn_pool(x)  # AdaptiveAvgPool2d 적용
        x = x.view(x.size(0), -1)  # 평탄화
        stgcn_features = self.stgcn_fc(x)

        # SAMURAI-like 동적 피처 추출
        dynamic_input = skeletons.unsqueeze(1)  # (batch_size, 1, in_channels, frames, joints)
        dynamic_features = self.dynamic_feature_extractor(dynamic_input)
        dynamic_features = dynamic_features.view(dynamic_features.size(0), -1)  # 평탄화

        # STGCN과 SAMURAI 출력 결합
        fused_features = torch.cat([stgcn_features, dynamic_features], dim=1)  # (batch_size, hidden_size + 128)
        fused_output = self.fusion_layer(fused_features)

        return self.output_layer(fused_output)

# STGCN 모델 설정
STGCN = UnifiedModel3DCNN
