import torch
import torch.nn as nn

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
    """Keypoint-based Action Classification Model"""
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
