import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BehaviorDataset
from stgcn_model import STGCN
from config.config import Config
import matplotlib.pyplot as plt

# Config 인스턴스 생성
config = Config()

# ----- 훈련 함수 -----
def train_model(train_loader, val_loader, config):
    device = config.device
    model = STGCN(
        in_channels=2,  # x, y 좌표
        num_joints=config.num_joints,  # 관절 수
        num_classes=config.num_classes,
        num_frames=30  # 프레임 수 고정
    ).to(device)

    model.train()  # 모델을 훈련 모드로 설정

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    adjacency_matrix = torch.eye(config.num_joints).to(device)  # (15, 15)

    train_losses, val_losses = [], []
    train_accuracies = []  # 정확도 리스트 초기화
    val_accuracies = []    # 정확도 리스트 초기화

    for epoch in range(config.num_epochs):
        train_loss = 0.0  # 훈련 손실 초기화
        train_correct = 0  # 훈련 정확도 초기화

        for batch_idx, (skeletons, labels) in enumerate(train_loader):
            skeletons, labels = skeletons.to(device), labels.to(device)

            # Debugging shapes
            optimizer.zero_grad()
            outputs = model(skeletons, adjacency_matrix)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # Epoch end
        train_accuracy = train_correct / len(train_loader.dataset)
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # 검증 데이터 평가
        val_loss, val_correct = evaluate_model(val_loader, model, criterion, device, adjacency_matrix)
        val_accuracy = val_correct / len(val_loader.dataset)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        # 결과 출력
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"  Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}")
        print(f"  Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.4f}")

    # 체크포인트 저장
    checkpoint_path = os.path.join(config.models_dir, "stgcn_behavior.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"모델이 {checkpoint_path}에 저장되었습니다.")

    return train_losses, val_losses, train_accuracies, val_accuracies


# ----- 평가 함수 -----
def evaluate_model(loader, model, criterion, device, adjacency_matrix):
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for skeletons, labels in loader:
            skeletons, labels = skeletons.to(device), labels.to(device)
            outputs = model(skeletons, adjacency_matrix)  # Adjacency matrix 전달
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return loss, correct


# ----- 실행 ----- 
if __name__ == "__main__":
    config = Config()
    train_data = BehaviorDataset(config.train_csv)
    val_data = BehaviorDataset(config.val_csv)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    # 모델 학습
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(train_loader, val_loader, config)

    # 학습 결과 시각화
    def plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies):
        epochs = range(1, len(train_losses) + 1)

        # 손실 시각화
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # 정확도 시각화
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        plt.plot(epochs, val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies)
