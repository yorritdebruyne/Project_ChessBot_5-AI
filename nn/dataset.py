import glob
import os

import numpy as np
import torch

from torch.utils.data import Dataset

from fiveaichess.utilities.encoding import unpack


"""
dataset
-------
Positions dataset for the value network, same structure as HousesDataset in Lab 2 (data.py):
__getitem__ returns (data, targets), so the learner loop 'for data, targets in dataloader' stays the same.

Reads the shards written by build_dataset_lichess.py (data/shards/train_*.npz, val_*.npz, test_*.npz):
- X: 12 uint64 bitboards per position, from the side to move (pack() in fiveaichess/utilities/encoding.py)
- y: centipawns from the side to move, clipped and divided by CLIP -> between -1 and 1
The split is already made in build_dataset_lichess.py (fixed, via a hash of the FEN), so no random_split here.
"""


class PositionsDataset(Dataset):
    def __init__(self, shards_dir: str, split: str):
        # split = "train", "val" or "test" (the shard names written by build_dataset_lichess.py)
        paths = sorted(glob.glob(os.path.join(shards_dir, split + "_*.npz")))
        if not paths:
            raise FileNotFoundError(f"no {split} shards in {shards_dir}")

        bitboards = []
        targets = []
        for path in paths:
            arrays = np.load(path)
            bitboards.append(arrays["X"])
            targets.append(arrays["y"])

        # Kept as 12 uint64 bitboards per position (96 bytes): 5M positions ~ 480 MB instead of 15 GB as 768 floats
        self.data = np.concatenate(bitboards)

        # Labels [-1, 1] -> [0, 1]: score for the side to move, so BCEWithLogitsLoss (logistic regression, L4)
        # and the 0.5 threshold for the accuracy stay the same
        self.targets = torch.from_numpy((np.concatenate(targets) + 1) / 2).float()

        # TODO (Phase 4, policy network): also load the best move (first move of the PV), once the shards contain it

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Unpack per position: only the batch in use exists as 768 floats in memory
        features = torch.from_numpy(unpack(self.data[idx])).float()
        return features, self.targets[idx]
