import pandas as pd
import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, file_path):
        # Load csv as pandas dataframe
        self.df = pd.read_csv(file_path)
        # Extract and remove IDs
        self.ids = self.df["User ID"]
        self.df = self.df.drop(columns=["User ID"])
        # Convert values to numpy
        self.data = self.df.to_numpy()

    def __len__(self):
        # Number of samples
        return len(self.data)

    def __getitem__(self, index):
        # Convert row to tensor
        return torch.tensor(self.data[index], dtype=torch.float32)
    

    def get_ids(self):
        return self.ids
