import numpy as np
import torch

from torch.utils.data import Dataset


"""
dataset
-------
Positions dataset for the value network, same structure as HousesDataset in Lab 2 (data.py):
__getitem__ returns (data, targets), so the learner loop 'for data, targets in dataloader' stays the same.

Every dataset file (.npz) uses the same format, so only npz_path in train_value.py changes:
- keys: train_data, train_targets, valid_data, valid_targets, test_data, test_targets
- data: encode() from fiveaichess/utilities/encoding.py (768 features)
- targets: from white's view, between 0 and 1 (game result, or sigmoid(cp / 400) for Stockfish evaluations)
Kasparov set: data/value_kasparov.npz (build_dataset.py, split by game)
Lichess set: data/value_lichess.npz (Maarten's extraction script, 98/1/1 split per position,
             every position appears only once, so no near-copies between train and test)
"""


class PositionsDataset(Dataset):
    def __init__(self, npz_path: str, split: str):
        # split = "train", "valid" or "test" (the arrays saved by build_dataset.py)
        arrays = np.load(npz_path)

        # Stored as uint8 to save space, the network needs float
        # TODO (Lichess set): don't convert the whole array to float here: 5M positions x 768 = 3.8 GB as uint8,
        #       15 GB as float32 -> doesn't fit in RAM. Keep self.data as uint8 and convert per item in
        #       __getitem__ (self.data[idx].float())
        # TODO (Lichess set): if Maarten stores the positions with np.packbits (96 bytes per position, ~480 MB for 5M),
        #       unpack per item: np.unpackbits(self.data[idx])[:768]
        self.data = torch.from_numpy(arrays[split + "_data"]).float()
        self.targets = torch.from_numpy(arrays[split + "_targets"]).float()

        # TODO (Phase 4, policy network): also load arrays[split + "_moves"] (Stockfish's best move, first move of the PV)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
