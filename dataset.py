import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, path: str =None, df: pd.DataFrame =None):
        """Class that loads a dataset from a csv file or pandas dataframe and prepares it for PyTorch.

        Args:
            path (str): path to the csv file. Defaults to None.
            df (pd.DataFrame): pandas dataframe. Defaults to None.
        """
        if path:
            # Load csv as pandas dataframe
            self.df = pd.read_csv(path)
        else:
            # Use provided dataframe
            self.df = df
        
        # Extract and remove IDs
        self.ids = self.df["User ID"]
        self.df = self.df.drop(columns=["User ID"])
        # Compute correlation matrix
        self.correlation_matrix = self.df.corr()
        # Convert data to tensor
        self.data = torch.tensor(self.df.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    @staticmethod
    def min_max_normalize(data: torch.Tensor) -> torch.Tensor:
        """Min-Max normalization

        Args:
            data (torch.Tensor): input data

        Returns:
            torch.Tensor: normalized data
        """
        # Convert data to pandas dataframe (because pandas makes operation column-wise as default)
        df = pd.DataFrame(data.numpy())
        # Normalize
        df = (df - df.min()) / (df.max() - df.min())
        return torch.tensor(df.to_numpy(), dtype=torch.float32)

    @staticmethod
    def z_score_normalize(data: torch.Tensor) -> torch.Tensor:
        """Z-score normalization.

        Args:
            data (torch.Tensor): input data

        Returns:
            torch.Tensor: normalized data
        """
        # Convert data to pandas dataframe (because pandas makes operation column-wise as default)
        df = pd.DataFrame(data.numpy())
        # Normalize
        df = (df - df.mean()) / df.std()
        return torch.tensor(df.to_numpy(), dtype=torch.float32)

