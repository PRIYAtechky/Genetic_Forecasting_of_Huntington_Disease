import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTClassifier(nn.Module):
    def __init__(self, num_classes=3):  # 3 classes
        super(ViTClassifier, self).__init__()
        # Use pretrained weights for faster/better convergence. Change to None if you prefer.
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
