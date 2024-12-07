import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from stgcn_model import STGCN  # 수정된 STGCN 모델 불러오기
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 행동 라벨 정의
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

class DogBehaviorDataset(Dataset):
    def __init__(self, csv_file, num_frames=1, num_joints=15):
        self.data = pd.read_csv(csv_file)

        # x, y 키포인트 열 추출
        keypoints_cols = [col for col in self.data.columns if col.startswith('x') or col.startswith('y')]

        # 키포인트 열이 홀수인 경우 마지막 열에 0으로 패딩
        if len(keypoints_cols) % 2 != 0:
            print("키포인트 열 개수가 짝수가 아닙니다. 마지막 열에 0으로 패딩 추가.")
            padding_column = f"padding_{len(keypoints_cols)}"
            self.data[padding_column] = 0
            keypoints_data = self.data[keypoints_cols + [padding_column]].values
        else:
            keypoints_data = self.data[keypoints_cols].values

        # 키포인트 데이터 크기 조정
        flattened_data = keypoints_data.flatten()
        expected_size = (flattened_data.size // (2 * num_joints)) * (2 * num_joints)

        # 초과 데이터를 자르거나 패딩 추가
        if flattened_data.size > expected_size:
            print(f"데이터 크기가 맞지 않아 조정: {flattened_data.size} -> {expected_size}. 초과 데이터 제거.")
            flattened_data = flattened_data[:expected_size]
        elif flattened_data.size < expected_size:
            padding_needed = expected_size - flattened_data.size
            print(f"데이터 크기가 맞지 않아 조정: {flattened_data.size} -> {expected_size}. 0으로 패딩 추가.")
            flattened_data = np.pad(flattened_data, (0, padding_needed), mode='constant', constant_values=0)

        # 데이터를 (samples, frames, channels, joints) 형태로 재구성
        try:
            self.features = flattened_data.reshape(-1, num_frames, 2, num_joints)
        except ValueError as e:
            raise ValueError(f"데이터 크기 조정 실패: {e}")

        # 라벨 크기를 피처 크기에 맞게 조정
        total_samples = self.features.shape[0]
        if len(self.data['behavior_class']) > total_samples:
            print(f"라벨 크기 조정: {len(self.data['behavior_class'])} -> {total_samples}. 초과 라벨 제거.")
            self.labels = self.data['behavior_class'].values[:total_samples]
        elif len(self.data['behavior_class']) < total_samples:
            print(f"라벨 크기 조정: {len(self.data['behavior_class'])} -> {total_samples}. 라벨에 패딩 추가.")
            label_padding = total_samples - len(self.data['behavior_class'])
            self.labels = np.pad(
                self.data['behavior_class'].values, 
                (0, label_padding), 
                mode='constant', 
                constant_values=-1  # 패딩된 라벨은 -1로 설정
            )
        else:
            self.labels = self.data['behavior_class'].values

        # 데이터 동기화 확인
        if len(self.labels) != self.features.shape[0]:
            raise ValueError(f"라벨 크기가 피처와 여전히 일치하지 않습니다: {len(self.labels)} != {self.features.shape[0]}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def load_data(train_file, val_file, test_file, batch_size=32):
    train_dataset = DogBehaviorDataset(train_file)
    val_dataset = DogBehaviorDataset(val_file)
    test_dataset = DogBehaviorDataset(test_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        validate_model(model, val_loader, criterion)

    print("Training complete!")

def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f"Validation Loss: {running_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

def test_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    print("Test Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(behavior_classes.values())))

if __name__ == "__main__":
    train_file = "data/split_data/annotations_train.csv"
    val_file = "data/split_data/annotations_validation.csv"
    test_file = "data/split_data/annotations_test.csv"

    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001

    train_loader, val_loader, test_loader = load_data(train_file, val_file, test_file, batch_size)

    input_dim = 2
    num_joints = 15
    num_classes = len(behavior_classes)
    num_frames = 30

    model = STGCN(in_channels=input_dim, num_joints=num_joints, num_classes=num_classes, num_frames=num_frames)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Training model...")
    train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer)

    print("Testing model...")
    test_model(model, test_loader)
