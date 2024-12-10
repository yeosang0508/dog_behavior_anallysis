import torch.optim as optim
from stgcn_model import STGCN
import torch.nn as nn
from sklearn.model_selection import train_test_split
from dataset import KeypointDataset
from torch.utils.data import Dataset, DataLoader

# 데이터셋 준비
dataset = KeypointDataset(r"data\csv_file\combined.csv")
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)


# 모델 초기화
model = STGCN(in_channels=2, hidden_channels=64, num_classes=13)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 학습 루프
for epoch in range(10):  # 에포크 수
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# 평가
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        output = model(batch)
        predictions = output.argmax(dim=1)
        correct += (predictions == batch.y).sum().item()
        total += batch.y.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
