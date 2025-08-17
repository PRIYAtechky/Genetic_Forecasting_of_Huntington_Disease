import streamlit as st
import torch
from model.vit_model import ViT
from preprocessing.dna_to_image import dna_sequence_to_matrix
import numpy as np

model = ViT(num_classes=2)
model.load_state_dict(torch.load("model/vit_hd.pt", map_location="cpu"))
model.eval()

st.title("Huntington Disease Risk Prediction")
sequence = st.text_input("Enter DNA Sequence")

if sequence:
    img = dna_sequence_to_matrix(sequence)
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    output = model(img_tensor)
    pred = torch.argmax(output, 1).item()
    label = "High Risk" if pred == 1 else "Low Risk"
    st.success(f"Prediction: {label}")
