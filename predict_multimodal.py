# src/predict_multimodal.py
import os
import re
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from vit_model import ViTClassifier
from dna_model import DNANet

# Names should match app.py expectations
CLASS_NAMES = ["Normal", "Intermediate", "Pathogenic"]

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model objects (these use the classes in your repo)
mri_model = ViTClassifier(num_classes=3).to(device)
dna_model = DNANet(input_dim=1000, num_classes=3).to(device)

# Try to load model files from a local ./models folder (optional)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

dna_path = os.path.join(MODEL_DIR, "dna_model.pth")
mri_path = os.path.join(MODEL_DIR, "mri_model.pth")

dna_loaded = False
mri_loaded = False
if os.path.exists(dna_path):
    try:
        dna_model.load_state_dict(torch.load(dna_path, map_location=device))
        dna_loaded = True
    except Exception as e:
        print("Could not load DNA model:", e)

if os.path.exists(mri_path):
    try:
        mri_model.load_state_dict(torch.load(mri_path, map_location=device))
        mri_loaded = True
    except Exception as e:
        print("Could not load MRI model:", e)

mri_model.eval()
dna_model.eval()

# MRI preprocessing (matches your repo's expectation)
mri_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def _process_image(img_path_or_file):
    # Accept either a path string or a file-like object (Streamlit uploader)
    if isinstance(img_path_or_file, str):
        img = Image.open(img_path_or_file).convert("RGB")
    else:
        # Streamlit's uploaded file works with PIL.Image.open()
        img = Image.open(img_path_or_file).convert("RGB")
    return mri_transform(img).unsqueeze(0)  # add batch dim

def _dna_seq_to_tensor(seq, input_dim=1000):
    # simple deterministic encoding (no training required) — for compatibility if a real model is loaded
    seq = re.sub(r'[^ATCG]', '', seq.upper())
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    vec = np.zeros(input_dim, dtype=np.float32)
    for i, ch in enumerate(seq[:input_dim]):
        vec[i] = mapping.get(ch, 0) / 3.0
    return torch.tensor(vec).unsqueeze(0)  # shape (1, input_dim)

def _heuristic_dna(seq):
    """
    Demo heuristic: count 'CAG' repeats (illustrative only).
    - many CAG repeats -> 'Pathogenic' (demo)
    - moderate -> 'Intermediate'
    - otherwise -> 'Normal'
    **NOT** a medical diagnosis—just a demo fallback.
    """
    s = seq.upper()
    cag_count = s.count("CAG")
    if cag_count >= 40:
        return 2, 0.95
    elif cag_count >= 36:
        return 1, 0.65
    else:
        return 0, 0.60

def predict(dna_text, img_path_or_file):
    """
    Returns a dict:
      {"DNA": {"Class": str, "Probability": float}, 
       "MRI": {"Class": str, "Probability": float}}
    This signature matches what app.py expects.
    """
    results = {}

    # ----- DNA prediction -----
    try:
        if dna_loaded:
            x = _dna_seq_to_tensor(dna_text).to(device)
            with torch.no_grad():
                out = dna_model(x)  # shape (1, num_classes)
                prob = torch.softmax(out, dim=1)[0]
                cls = int(prob.argmax().item())
                results["DNA"] = {"Class": CLASS_NAMES[cls], "Probability": float(prob[cls].item())}
        else:
            cls, p = _heuristic_dna(dna_text)
            results["DNA"] = {"Class": CLASS_NAMES[cls], "Probability": float(p)}
    except Exception as e:
        results["DNA"] = {"Class": "Error", "Probability": 0.0, "Error": str(e)}

    # ----- MRI prediction -----
    if img_path_or_file is None:
        results["MRI"] = {"Class": "No image provided", "Probability": 0.0}
        return results

    try:
        img_t = _process_image(img_path_or_file).to(device)
        with torch.no_grad():
            out = mri_model(img_t)  # shape (1,num_classes)
            prob = torch.softmax(out, dim=1)[0]
            cls = int(prob.argmax().item())
            results["MRI"] = {"Class": CLASS_NAMES[cls], "Probability": float(prob[cls].item())}
    except Exception as e:
        results["MRI"] = {"Class": "Error", "Probability": 0.0, "Error": str(e)}

    return results
