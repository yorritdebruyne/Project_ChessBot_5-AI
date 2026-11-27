import numpy as np
import torch

from torch.utils.data import Dataset


"""
dataset
-------
Positions dataset for the value network, same structure as HousesDataset in Lab 2 (data.py):
__getitem__ returns (data, targets), so the learner loop 'for data, targets in dataloader' stays the same.
"""


class PositionsDataset(Dataset):
    def __init__(self, npz_path: str, split: str):
        # split = "train", "valid" or "test" (the arrays saved by build_dataset.py)
        arrays = np.load(npz_path)

        # Stored as uint8 to save space, the network needs float
        self.data = torch.from_numpy(arrays[split + "_data"]).float()
        self.targets = torch.from_numpy(arrays[split + "_targets"]).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
