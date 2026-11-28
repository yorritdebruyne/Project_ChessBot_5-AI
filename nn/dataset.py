import glob
import os

import numpy as np
import torch

from torch.utils.data import Dataset

from fiveaichess.utilities.encoding import CLIP, unpack


"""
dataset
-------
Positions dataset for the value network, same structure as HousesDataset in Lab 2 (data.py):
__getitem__ returns (data, targets), so the learner loop 'for data, targets in dataloader' stays the same.

Reads the shards written by build_dataset_lichess.py (data/shards/train_*.npz, val_*.npz, test_*.npz):
- X:    12 uint64 bitboards per position, from the side to move (pack() in fiveaichess/utilities/encoding.py)
- cp:   Stockfish evaluation in centipawns from the side to move (mate = +-MATE_CP)
- move: best move (from_square * 64 + to_square), for the policy network (Phase 4)
The label is made here from cp (D2), so linear and sigmoid can be compared without rebuilding the shards.
The split is already made in build_dataset_lichess.py (fixed, via a hash of the FEN), so no random_split here.
"""

# sigmoid(cp / CP_SCALE): +400 cp (about a rook less a pawn up) -> 0.73, +1000 cp -> 0.92
CP_SCALE = 400


def cp_to_label(cp: np.ndarray, label: str) -> np.ndarray:
    # Score for the side to move between 0 and 1, so BCEWithLogitsLoss (logistic regression, L4)
    # and the 0.5 threshold for the accuracy stay the same for both labels
    cp = cp.astype(np.float32)
    if label == "sigmoid":
        # Default (D2): a win probability (L9); only mates (+-MATE_CP, ~18% of the Lichess positions) become 0 or 1
        return 1 / (1 + np.exp(-cp / CP_SCALE))
    if label == "linear":
        # Comparison for the report: clipped to [-CLIP, CLIP], then [-1, 1] -> [0, 1]
        return (np.clip(cp, -CLIP, CLIP) / CLIP + 1) / 2
    raise ValueError(f"unknown label: {label}")


class PositionsDataset(Dataset):
    def __init__(self, shards_dir: str, split: str, label: str = "sigmoid"):
        # split = "train", "val" or "test" (the shard names written by build_dataset_lichess.py)
        paths = sorted(glob.glob(os.path.join(shards_dir, split + "_*.npz")))
        if not paths:
            raise FileNotFoundError(f"no {split} shards in {shards_dir}")

        bitboards = []
        cps = []
        moves = []
        for path in paths:
            arrays = np.load(path)
            bitboards.append(arrays["X"])
            cps.append(arrays["cp"])
            moves.append(arrays["move"])

        # Kept as 12 uint64 bitboards per position (96 bytes): 5M positions ~ 480 MB instead of 15 GB as 768 floats
        self.data = np.concatenate(bitboards)

        self.cp = np.concatenate(cps)
        self.targets = torch.from_numpy(cp_to_label(self.cp, label)).float()

        # Best moves are loaded already, the policy network (Phase 4) uses them as class labels
        self.moves = torch.from_numpy(np.concatenate(moves).astype(np.int64))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Unpack per position: only the batch in use exists as 768 floats in memory
        features = torch.from_numpy(unpack(self.data[idx])).float()
        return features, self.targets[idx]
