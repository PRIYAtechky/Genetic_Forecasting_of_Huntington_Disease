import re
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from vit_model import ViTClassifier
from dna_model import DNANet

CLASS_NAMES = ["Normal", "Intermediate", "Pathogenic"]

# ----- MRI preprocessing -----
mri_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def process_image(img_path_or_file):
    # Works with file path or file-like object (e.g., Streamlit uploaded file)
    img = Image.open(img_path_or_file).convert("RGB")
    return mri_transform(img).unsqueeze(0)

# ----- DNA featurization -----
def clean_dna(seq: str) -> str:
    return re.sub(r'[^ATCG]', '', seq.upper())

def dna_to_features(seq: str, dim: int = 1000) -> torch.Tensor:
    feat = np.zeros(dim, dtype=np.float32)
    # Match training features
    feat[0] = len(re.findall(r'(?:CAG)+', seq))
    if len(seq) > 0:
        feat[1] = (seq.count("G") + seq.count("C")) / len(seq)
    feat[2] = min(len(seq) / 1000.0, 1.0)
    return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

# ----- Load models -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dna_model = DNANet(input_dim=1000, num_classes=3).to(device)
dna_model.load_state_dict(torch.load(r"P:\HD_ViT_Project\models\dna_model.pth", map_location=device))
dna_model.eval()

mri_model = ViTClassifier(num_classes=3).to(device)
mri_model.load_state_dict(torch.load(r"P:\HD_ViT_Project\models\mri_model.pth", map_location=device))
mri_model.eval()

# ----- Predict -----
def predict(dna_seq: str, img_path_or_file=None):
    results = {}

    # DNA prediction (even if empty string, still returns something)
    s = clean_dna(dna_seq or "")
    dna_feat = dna_to_features(s).to(device)
    with torch.no_grad():
        out = dna_model(dna_feat)
        prob = torch.softmax(out, dim=1)[0]
        cls = int(prob.argmax().item())
    results["DNA"] = {"Class": CLASS_NAMES[cls], "Probability": float(prob[cls].item())}

    # MRI prediction
    if img_path_or_file:
        img_tensor = process_image(img_path_or_file).to(device)
        with torch.no_grad():
            out = mri_model(img_tensor)
            prob = torch.softmax(out, dim=1)[0]
            cls = int(prob.argmax().item())
        results["MRI"] = {"Class": CLASS_NAMES[cls], "Probability": float(prob[cls].item())}

    return results
