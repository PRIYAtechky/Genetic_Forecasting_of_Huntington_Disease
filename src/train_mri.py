import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from vit_model import ViTClassifier

# ----- Config -----
DATA_DIR = r"P:\HD_ViT_Project\data\mri_images"
SAVE_PATH = r"P:\HD_ViT_Project\models\mri_model.pth"
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

def train_mri():
    dataset = MRIDataset(DATA_DIR)
    print(f"✅ Total MRI images: {len(dataset)}")

    # Train/Validation split (80/20)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ViTClassifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc = correct / total if total > 0 else 0.0

        # Validate
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_correct += (out.argmax(1) == y).sum().item()
                v_total += y.size(0)
        val_acc = v_correct / v_total if v_total > 0 else 0.0

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(train_loader):.4f} "
              f"- Train Acc: {train_acc:.2%} - Val Acc: {val_acc:.2%}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ MRI model trained and saved at {SAVE_PATH}")

if __name__ == "__main__":
    train_mri()