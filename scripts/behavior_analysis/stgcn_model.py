import torch.nn as nn
from torch_geometric.nn import GCNConv

class STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(STGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = torch.mean(x, dim=0)  # 노드들의 평균
        x = self.fc(x)
        return x
