import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# GPU/CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 클래스
class KeypointDataset(Dataset):
    def __init__(self, csv_file, normalize=True):
        self.data = pd.read_csv(csv_file)
        self.data.fillna(0, inplace=True)

        # 키포인트 열 선택
        self.keypoint_columns = [col for col in self.data.columns if col.startswith("x") or col.startswith("y")]
        keypoints = self.data[self.keypoint_columns].values.astype(np.float32)

        # 키포인트 정규화
        if normalize:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            keypoints = self.scaler.fit_transform(keypoints)

        self.keypoints = keypoints
        self.labels = self.data["behavior_class"].values.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        keypoints = self.keypoints[idx]
        label = self.labels[idx]
        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(label)


# 모델 정의
class KeypointModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(KeypointModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, num_classes)

        # 가중치 초기화
        self._initialize_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(self.bn1(x))
        x = self.fc2(x)
        x = nn.ReLU()(self.bn2(x))
        x = self.fc3(x)
        x = nn.ReLU()(self.bn3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


# 학습 및 검증
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, save_path="best_model.pth"):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        for keypoints, labels in train_loader:
            keypoints, labels = keypoints.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_correct / total)

        # Validation
        model.eval()
        val_loss, val_correct, total = 0, 0, 0
        with torch.no_grad():
            for keypoints, labels in val_loader:
                keypoints, labels = keypoints.to(device), labels.to(device)
                outputs = model(keypoints)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_correct / total)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

        # 모델 저장 부분 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"모델 저장됨: {save_path}")

        scheduler.step(val_loss)

    # 시각화
    visualize_metrics(train_losses, val_losses, train_accuracies, val_accuracies)


# 손실 및 정확도 시각화
def visualize_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()


# 실행 코드
if __name__ == "__main__":
    train_csv = "data/split_data/annotations_train.csv"
    val_csv = "data/split_data/annotations_validation.csv"
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4

    train_loader = DataLoader(KeypointDataset(train_csv), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(KeypointDataset(val_csv), batch_size=batch_size, shuffle=False)

    labels = pd.read_csv(train_csv)["behavior_class"].values
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = KeypointModel(input_size=len(KeypointDataset(train_csv).keypoint_columns), num_classes=len(np.unique(labels)))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)


    train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, save_path="best_model.pth")
