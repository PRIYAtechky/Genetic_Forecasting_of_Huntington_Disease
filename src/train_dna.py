import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from dna_model import DNANet

# ---- Config ----
CSV_PATH = r"P:\HD_ViT_Project\data\dna_sequences.csv"   # must have columns: sequence, label (0/1/2)
SAVE_PATH = r"P:\HD_ViT_Project\models\dna_model.pth"
INPUT_DIM = 1000
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

# ---- Dataset ----
class DNADataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        assert "sequence" in df.columns and "label" in df.columns, "CSV must contain 'sequence' and 'label'"
        self.seqs = df["sequence"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.seqs)

    @staticmethod
    def _clean(seq: str) -> str:
        return re.sub(r'[^ATCG]', '', seq.upper())

    @staticmethod
    def _featurize(seq: str, dim: int = INPUT_DIM) -> np.ndarray:
        feat = np.zeros(dim, dtype=np.float32)
        # Feature 0: CAG repeat count
        feat[0] = len(re.findall(r'(?:CAG)+', seq))
        # Feature 1: GC content
        if len(seq) > 0:
            feat[1] = (seq.count("G") + seq.count("C")) / len(seq)
        # Feature 2: length (scaled)
        feat[2] = min(len(seq) / 1000.0, 1.0)
        return feat

    def __getitem__(self, idx):
        seq = self._clean(self.seqs[idx])
        x = self._featurize(seq)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def train_dna():
    dataset = DNADataset(CSV_PATH)

    # Train/Validation split
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42, shuffle=True)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNANet(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_correct, total_count = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (out.argmax(1) == y).sum().item()
            total_count += y.size(0)

        train_acc = total_correct / max(1, total_count)

        # Validation
        model.eval()
        val_correct, val_count = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
                val_count += y.size(0)
        val_acc = val_correct / max(1, val_count)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/max(1,len(train_loader)):.4f} "
              f"| Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ DNA model saved at {SAVE_PATH}")

if __name__ == "__main__":
    train_dna()
