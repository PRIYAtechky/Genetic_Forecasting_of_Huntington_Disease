import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class MRIClassifier(nn.Module):
    def __init__(self, num_classes=3):  # 3 classes
        super(MRIClassifier, self).__init__()
        self.vit = vit_b_16(weights=None)  # no pretrained weights
        in_features = self.vit.heads.head.in_features
        self.vit.heads = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
