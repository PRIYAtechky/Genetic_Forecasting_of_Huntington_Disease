import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from vit_model import ViTClassifier
from paths import MRI_DATA_DIR, MRI_MODEL_PATH  # ✅ use paths.py

# ----- Config -----
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

NUM_CLASSES = 3
CLASS_NAMES = ["normal", "intermediate", "pathogenic"]

# ----- Dataset -----
class MRIDataset(Dataset):
    def __init__(self, img_folder):
        self.images = []
        allowed_ext = {".png", ".jpg", ".jpeg", ".bmp"}
        for cls in CLASS_NAMES:
            folder_path = os.path.join(img_folder, cls)
            if os.path.isdir(folder_path):
                for f in os.listdir(folder_path):
                    if os.path.splitext(f)[1].lower() in allowed_ext:
                        self.images.append((os.path.join(folder_path, f), cls))
        if not self.images:
            raise ValueError("No images found. Check your dataset path or class folders.")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, cls_name = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = CLASS_NAMES.index(cls_name)
        return img, torch.tensor(label, dtype=torch.long)

# ----- DataLoader -----
dataset = MRIDataset(MRI_DATA_DIR)
print(f"✅ Total MRI images: {len(dataset)}")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----- Model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTClassifier(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----- Training Loop -----
model.train()
for epoch in range(EPOCHS):
    total_loss, correct, total = 0, 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    acc = correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(dataloader):.4f} - Acc: {acc:.2%}")

# ----- Save Model -----
os.makedirs(os.path.dirname(MRI_MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MRI_MODEL_PATH)
print(f"✅ MRI model trained and saved at {MRI_MODEL_PATH}")





