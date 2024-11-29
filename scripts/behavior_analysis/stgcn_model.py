import torch
import torch.nn as nn

class STGCN(nn.Module):
    def __init__(self, in_channels, num_joints, num_classes, num_frames):
        super(STGCN, self).__init__()
        self.gcn1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.gcn2 = nn.Conv2d(64, 128, kernel_size=1)
        self.gcn3 = nn.Conv2d(128, 256, kernel_size=1)

        self.relu = nn.ReLU()

        # Fully connected layer
        input_features = 256 * num_frames * num_joints
        self.fc = nn.Linear(input_features, num_classes)

    def forward(self, x, adjacency_matrix):
        # x: (batch, channels, frames, joints) -> (batch, channels, frames, joints)
        x = torch.einsum("bctj,jk->bctk", x, adjacency_matrix)

        # Pass through GCN layers
        x = self.relu(self.gcn1(x))  # (batch, 64, frames, joints)
        x = self.relu(self.gcn2(x))  # (batch, 128, frames, joints)
        x = self.relu(self.gcn3(x))  # (batch, 256, frames, joints)

        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)  # (batch, -1)
        x = self.fc(x)
        return x

