import os

# Project root folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data folder
DATA_DIR = os.path.join(BASE_DIR, "data")

# Models folder (where trained models will be saved)
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Paths to data
DNA_DATA_PATH = os.path.join(DATA_DIR, "dna_sequences.csv")
MRI_DATA_DIR = os.path.join(DATA_DIR, "mri_images")

# Paths where models will be saved
DNA_MODEL_PATH = os.path.join(MODEL_DIR, "dna_model.pth")
MRI_MODEL_PATH = os.path.join(MODEL_DIR, "mri_model.pth")
