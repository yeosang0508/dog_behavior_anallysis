import torch
import torch.nn as nn

class BehaviorClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BehaviorClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    @staticmethod
    def load_model(model_path):
        model = torch.load(model_path)
        model.eval()
        return model
