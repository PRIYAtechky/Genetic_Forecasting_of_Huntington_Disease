import torch
from torch.utils.data import Dataset
import pandas as pd
from preprocessing.dna_to_image import dna_sequence_to_matrix

class DNADataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.labels = self.df['Category'].astype('category').cat.codes
        self.sequences = self.df['Random_Gene_Sequence']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.sequences.iloc[idx]
        image = dna_sequence_to_matrix(sequence)
        image = torch.tensor(image).unsqueeze(0).float()
        label = torch.tensor(self.labels.iloc[idx]).long()
        return image, label
