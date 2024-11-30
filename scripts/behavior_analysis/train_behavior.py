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
    model = UnifiedModel3DCNN(  # 3D CNN 기반 UnifiedModel
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
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return model, train_losses, val_losses, train_accuracies, val_accuracies


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
    train_data = BehaviorDataset(config.train_csv, num_frames=config.num_frames)
    val_data = BehaviorDataset(config.val_csv, num_frames=config.num_frames)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    # 모델 학습
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_unified_model(train_loader, val_loader, config)

    # 학습 결과 시각화
    plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies)