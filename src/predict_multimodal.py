import torch
import torch.nn as nn
from dna_model import DNAModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = [
    "Intermediate",
    "Normal",
    "Pathogenic"
]

model = DNAModel().to(DEVICE)
model.load_state_dict(torch.load("models/dna_model.pth", map_location=DEVICE))
model.eval()

DNA_MAP = {
    "A": 0,
    "T": 1,
    "C": 2,
    "G": 3
}

MAX_LEN = 200

def encode_sequence(seq):
    seq = seq.upper()

    encoded = [DNA_MAP.get(ch, 0) for ch in seq]

    if len(encoded) < MAX_LEN:
        encoded += [0] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]

    return torch.tensor(encoded).unsqueeze(0)

def predict(dna_text, image_input=None):
    x = encode_sequence(dna_text).to(DEVICE)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    cls = LABELS[pred.item()]

    return {
        "DNA": {
            "Class": cls,
            "Probability": float(confidence.item())
        }
    }
