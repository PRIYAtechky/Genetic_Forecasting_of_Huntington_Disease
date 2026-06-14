import torch
import torch.nn as nn

class DNANet(nn.Module):
    def __init__(self, input_dim=1000, num_classes=3):  # 3 classes
        super(DNANet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
