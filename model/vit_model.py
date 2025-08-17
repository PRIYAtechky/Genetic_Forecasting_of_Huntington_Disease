import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=4, emb_size=256, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        return x + self.pos_embed

class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=256, depth=6, heads=8, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)

class ViT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.embedding = PatchEmbedding()
        self.encoder = TransformerEncoder()
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.head(x[:, 0])
