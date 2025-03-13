import torch
from torch.utils.data import DataLoader

from clear import clear_dataset
from csv_dataset import CSVDataset


def main():
    # clear_dataset()

    dataset = CSVDataset("data/smartwatch_cleared.csv")
    dataloader = DataLoader(dataset, batch_size=32)

    # Example: Fetch a batch
    for batch in dataloader:
        print(batch.shape)
        break





if __name__ == "__main__":
    main()