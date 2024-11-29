import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BehaviorDataset
from stgcn_model import UnifiedModel
from config.config import Config
import matplotlib.pyplot as plt

# Config 인스턴스 생성
config = Config()

# ----- 훈련 함수 -----
def train_unified_model(train_loader, val_loader, config):
    """
    통합 모델 학습 함수
    Args:
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        config: 설정 객체
    """
    device = config.device
    model = UnifiedModel(
        in_channels=2,
        num_joints=config.num_joints,
        num_classes=config.num_classes,
        num_frames=30,
        frame_feature_size=None,  # 프레임 피처 없음
        hidden_size=64
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    adjacency_matrix = torch.eye(config.num_joints).to(device)  # 인접 행렬 (단위 행렬)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for skeletons, frame_features, labels in train_loader:
            skeletons, labels = skeletons.to(device), labels.to(device)
            frame_features = frame_features.to(device) if frame_features is not None else None

            optimizer.zero_grad()
            outputs = model(skeletons, adjacency_matrix, frame_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # 에포크별 훈련 결과 저장
        train_accuracy = train_correct / len(train_loader.dataset)
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # 검증 데이터 평가
        val_loss, val_correct = evaluate_unified_model(val_loader, model, criterion, device, adjacency_matrix)
        val_accuracy = val_correct / len(val_loader.dataset)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        # 결과 출력
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"  Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}")
        print(f"  Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.4f}")

    # 모델 저장
    model_save_path = os.path.join(config.models_dir, "stgcn_behavior.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return model, train_losses, val_losses, train_accuracies, val_accuracies


# ----- 평가 함수 -----
def evaluate_unified_model(loader, model, criterion, device, adjacency_matrix):
    """
    통합 모델 평가 함수
    Args:
        loader: 데이터 로더
        model: 평가할 모델
        criterion: 손실 함수
        device: 디바이스
        adjacency_matrix: 그래프 인접 행렬
    """
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for skeletons, frame_features, labels in loader:
            skeletons, labels = skeletons.to(device), labels.to(device)
            frame_features = frame_features.to(device) if frame_features is not None else None

            outputs = model(skeletons, adjacency_matrix, frame_features)
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()

    return loss / len(loader), correct


# ----- 학습 결과 시각화 -----
def plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # 손실 시각화
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 정확도 시각화
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ----- 실행 -----
if __name__ == "__main__":
    config = Config()
    train_data = BehaviorDataset(config.train_csv)
    val_data = BehaviorDataset(config.val_csv)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    # 모델 학습
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_unified_model(train_loader, val_loader, config)

    # 학습 결과 시각화
    plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies)
