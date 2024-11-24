import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BehaviorDataset  # 데이터셋 클래스는 별도 파일로 분리
from model import STGCN  # 모델 정의는 별도 파일로 분리
from config.config import Config  # Config는 별도 파일로 분리
import pandas as pd

# Config 인스턴스 생성
config = Config()

# ----- 훈련 함수 -----
def train_model(train_loader, val_loader, config):
    model = STGCN(
        in_channels=2,  # 입력 채널: x, y 좌표
        num_joints=config.num_joints,  # Config에서 관절 개수 가져옴
        num_classes=config.num_classes,  # Config에서 클래스 개수 가져옴
        num_frames=30  # 입력 시퀀스 길이 (프레임 수)
    ).to(config.device)

    # Adjacency matrix 생성
    adjacency_matrix = torch.eye(config.num_joints).to(config.device)  # 단순 자기 연결 행렬

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for skeletons, labels in train_loader:
            skeletons, labels = skeletons.to(config.device), labels.to(config.device)

            # 입력 데이터 형상 변환
            skeletons = skeletons.permute(0, 2, 1).contiguous().unsqueeze(2)  # (batch, joints, 2) -> (batch, 2, 1, joints)
            
             # 모델에 입력
            outputs = model(skeletons, adjacency_matrix)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()


        # 검증 정확도 계산
        val_loss, val_correct = evaluate_model(val_loader, model, criterion, config.device, adjacency_matrix)
        print(f"Epoch {epoch+1}/{config.num_epochs}: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_correct/len(train_loader.dataset):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_correct/len(val_loader.dataset):.4f}")


    # 훈련 완료 후 모델 저장
    torch.save(model.state_dict(), os.path.join(config.models_dir, "stgcn_behavior.pth"))
    print("모델 저장 완료!")

# ----- 평가 함수 -----
def evaluate_model(loader, model, criterion, device, adjacency_matrix):
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for skeletons, labels in loader:
            skeletons, labels = skeletons.to(device), labels.to(device)
             # 입력 데이터 형상 변환
            skeletons = skeletons.permute(0, 2, 1).contiguous().unsqueeze(2)  # (batch, joints, 2) -> (batch, 2, 1, joints)
            
            outputs = model(skeletons, adjacency_matrix)  # Adjacency matrix 전달
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return loss, correct

# ----- 실행 -----
if __name__ == "__main__":
    # Config 로드
    config = Config()

    # 디바이스 설정
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {config.device}")

    # 데이터셋 로드
    train_data = BehaviorDataset(config.train_csv)
    val_data = BehaviorDataset(config.val_csv)

    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(val_data)}")

    # 데이터로더 설정
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    # 모델 학습
    train_model(train_loader, val_loader, config)
