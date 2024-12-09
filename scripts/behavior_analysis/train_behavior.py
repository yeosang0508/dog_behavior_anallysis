import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# GPU/CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 행동 클래스 정의
behavior_classes = {
    0: "bodylower",
    1: "bodyscratch",
    2: "bodyshake",
    3: "feetup",
    4: "footup",
    5: "heading",
    6: "lying",
    7: "mounting",
    8: "sit",
    9: "tailing",
    10: "taillow",
    11: "turn",
    12: "walkrun"
}

class KeypointDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # 결측값 처리
        self.data.fillna(0, inplace=True)

        # 키포인트 정규화
        keypoint_columns = [col for col in self.data.columns if col.startswith("x") or col.startswith("y")]
        self.scaler = StandardScaler()
        self.keypoints = self.scaler.fit_transform(self.data[keypoint_columns].values.astype(np.float32))

        # 행동 클래스
        self.labels = self.data["behavior_class"].values.astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        keypoints = self.keypoints[idx]
        label = self.labels[idx]
        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(label)

class ResidualBlock(nn.Module):
    """Residual Block to add skip connections"""
    def __init__(self, input_size, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.batch_norm = nn.BatchNorm1d(input_size)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batch_norm(x)
        return x + residual

class KeypointModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(KeypointModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.residual_block1 = ResidualBlock(256, 128)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.6)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.residual_block1(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, patience=5, save_path="best_model.pth"):
    model.to(device)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for keypoints, labels in train_loader:
            keypoints, labels = keypoints.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(keypoints)
            loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN 또는 Inf 감지됨, 학습 무시.")
                continue

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_acc)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for keypoints, labels in val_loader:
                keypoints, labels = keypoints.to(device), labels.to(device)
                outputs = model(keypoints)
                loss = criterion(outputs, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    print("NaN detected in validation loss")
                    continue

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Early Stopping & 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model, save_path)
            print(f"모델 저장됨: {save_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

        # Scheduler step
        scheduler.step()

    # 손실 및 정확도 시각화
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    train_csv = "data/split_data/annotations_train.csv"
    val_csv = "data/split_data/annotations_validation.csv"
    num_epochs = 20
    batch_size = 32
    learning_rate = 1e-5
    save_path = "best_model.pth"

    train_loader = DataLoader(KeypointDataset(train_csv), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(KeypointDataset(val_csv), batch_size=batch_size, shuffle=False)

    keypoint_columns = [col for col in pd.read_csv(train_csv).columns if col.startswith("x") or col.startswith("y")]
    input_size = len(keypoint_columns)
    num_classes = len(behavior_classes)

    model = KeypointModel(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("Training model...")
    train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, patience=5, save_path=save_path)
