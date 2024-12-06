import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BehaviorDataset
from stgcn_model import UnifiedModel3DCNN
from config.config import Config
import matplotlib.pyplot as plt

# Config 인스턴스 생성
config = Config()

# ----- 훈련 함수 -----
def train_unified_model(train_loader, val_loader, config):
    device = config.device
    model = UnifiedModel3DCNN(
        in_channels=2,
        num_joints=config.num_joints,
        num_classes=config.num_classes,
        num_frames=config.num_frames,
        hidden_size=64
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    feature_maps = []  # 중간 출력 저장

    def hook_fn(module, input, output):
        feature_maps.append(output)

    # Hook 등록 (3D CNN 마지막 Conv3D 레이어)
    model.dynamic_feature_extractor[3].register_forward_hook(hook_fn)

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for skeletons, labels in train_loader:
            skeletons, labels = skeletons.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(skeletons, torch.eye(config.num_joints).to(device))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_accuracy = train_correct / len(train_loader.dataset)
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        val_loss, val_correct = evaluate_unified_model(val_loader, model, criterion, device)
        val_accuracy = val_correct / len(val_loader.dataset)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"  Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}")
        print(f"  Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.4f}")

    model_save_path = os.path.join(config.models_dir, "stgcn_behavior.pth")
    torch.save(model, model_save_path)
    print(f"Model saved to {model_save_path}")
    return model, train_losses, val_losses, train_accuracies, val_accuracies, feature_maps


# ----- 평가 함수 -----
def evaluate_unified_model(loader, model, criterion, device):
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for skeletons, labels in loader:
            skeletons, labels = skeletons.to(device), labels.to(device)
            outputs = model(skeletons, torch.eye(config.num_joints).to(device))
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return loss / len(loader), correct


# ----- 중간 출력 시각화 -----
def visualize_feature_maps(feature_maps):
    feature_map = feature_maps[0][0, :, 0, :, :].cpu().detach().numpy()  # 첫 배치의 첫 번째 프레임
    plt.figure(figsize=(20, 10))
    for i in range(min(feature_map.shape[0], 8)):  # 첫 8개 채널만 표시
        plt.subplot(2, 4, i + 1)
        plt.imshow(feature_map[i], cmap='viridis')
        plt.title(f'Channel {i}')
        plt.axis('off')
    plt.show()


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


def initialize_and_load_sam_model(cfg):
    # `sam_model_path` 키가 없으면 기본 경로 사용
    model_path = getattr(cfg, 'sam_model_path', 'C:/Users/admin/IdeaProjects/VSCode/sam2/checkpoints/sam2_hiera_tiny.pt')
    print(f"모델 경로: {model_path}")
    # 모델 로드 로직 (여기에 모델을 실제로 로드하는 코드를 작성)
    # 예: model = torch.load(model_path)
    return model_path  # 또는 모델 객체 반환



# ----- 실행 -----
if __name__ == "__main__":
    config = Config()
    
    # SAM 모델 초기화 및 로드
    model_path = initialize_and_load_sam_model(config)
    print(f"Initialized SAM Model with weights at: {model_path}")
    
    train_data = BehaviorDataset(config.train_csv, num_frames=config.num_frames)
    val_data = BehaviorDataset(config.val_csv, num_frames=config.num_frames)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    # 모델 학습
    model, train_losses, val_losses, train_accuracies, val_accuracies, feature_maps = train_unified_model(
        train_loader, val_loader, config
    )

    # 학습 결과 시각화
    plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies)

    # 중간 출력 시각화
    visualize_feature_maps(feature_maps)
