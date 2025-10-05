import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dna_model import DNANet
from paths import DNA_DATA_PATH, DNA_MODEL_PATH  # ✅ using paths.py

# ----- Config -----
INPUT_DIM = 1000
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

# Mapping dictionary (text → int)
LABEL_MAP = {"Normal": 0, "Intermediate": 1, "Pathogenic": 2}

# ----- Dataset -----
class DNADataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        # ✅ Detect label column (typo "lable")
        if "label" in self.df.columns:
            col = "label"
        elif "lable" in self.df.columns:
            col = "lable"
        elif "class" in self.df.columns:
            col = "class"
        elif "outcome" in self.df.columns:
            col = "outcome"
        else:
            raise ValueError(f"❌ Could not find a label column. Found: {list(self.df.columns)}")

        # Convert labels (text → int if needed)
        if self.df[col].dtype == object:  # text labels
            self.labels = self.df[col].map(LABEL_MAP).astype(int).tolist()
        else:  # already numeric
            self.labels = self.df[col].astype(int).tolist()

        # ✅ Detect sequence column
        if "sequence" in self.df.columns:
            self.seqs = self.df["sequence"].astype(str).tolist()
        elif "seq" in self.df.columns:
            self.seqs = self.df["seq"].astype(str).tolist()
        else:
            raise ValueError(f"❌ Could not find a sequence column. Found: {list(self.df.columns)}")

    def __len__(self):
        return len(self.seqs)

    @staticmethod
    def _clean(seq):
        return re.sub(r"[^ATCG]", "", seq.upper())

    @staticmethod
    def _featurize(seq, dim=INPUT_DIM):
        feat = np.zeros(dim, dtype=np.float32)
        # handcrafted feature: count of CAG repeats
        feat[0] = len(re.findall(r"(?:CAG)+", seq))
        return feat

    def __getitem__(self, idx):
        seq = self._clean(self.seqs[idx])
        x = self._featurize(seq)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# ----- DataLoader -----
dataset = DNADataset(DNA_DATA_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----- Model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DNANet(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----- Training Loop -----
model.train()
for epoch in range(EPOCHS):
    total_loss, total_correct, total_count = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (out.argmax(1) == y).sum().item()
        total_count += y.size(0)
    acc = total_correct / total_count
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.3f}")

# ----- Save Model -----
os.makedirs(os.path.dirname(MRI_MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MRI_MODEL_PATH)
print(f"✅ MRI model trained and saved at {MRI_MODEL_PATH}")
