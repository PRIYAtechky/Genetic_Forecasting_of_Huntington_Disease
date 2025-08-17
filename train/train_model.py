import torch
from torch.utils.data import DataLoader
from utils.dataset import DNADataset
from model.vit_model import ViT
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim

dataset = DNADataset("data/Huntington_Disease_Dataset.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ViT(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    all_preds, all_labels = [], []
    model.train()
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        all_preds += outputs.argmax(1).tolist()
        all_labels += labels.tolist()

    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}: Accuracy = {acc:.4f}")

torch.save(model.state_dict(), "model/vit_hd.pt")
