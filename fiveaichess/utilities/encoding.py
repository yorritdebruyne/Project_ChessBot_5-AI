import chess
import numpy as np

from chess import Board


"""
encoding
--------
Encodes a board as 768 one-hot features for the neural networks:
12 planes (6 piece types * 2 colours) * 64 squares, 
feature[plane * 64 + square] = 1 when that piece stands on that square.
Shared by the training code (nn/) and the engine (nn_utility.py), so the network
always gets the same input while training and while playing.
"""

NUM_PLANES = 12
NUM_FEATURES = NUM_PLANES * 64  # 768


def plane_index(piece: chess.Piece) -> int:
    # Plane 0-5 = white pawn, knight, bishop, rook, queen, king
    # Plane 6-11 = the same for black
    # TODO: piece.piece_type is 1 (PAWN) .. 6 (KING), piece.color is chess.WHITE (True) or chess.BLACK (False)
    if piece.color == chess.WHITE:
        return piece.piece_type - 1
    else:
        return piece.piece_type + 5



def encode(board: Board) -> np.ndarray:
    # uint8 instead of float: 4x less memory for the whole dataset, the learner converts to float
    features = np.zeros(NUM_FEATURES, dtype=np.uint8)

    # TODO: Loop over board.piece_map().items() -> (square, piece) and set the right feature to 1

    for square, piece in board.piece_map().items():
        features[plane_index(piece) * 64 + square] = 1
    return features


if __name__ == "__main__":
    # Encoding test: start position must have 32 pieces, with the right number per plane
    features = encode(Board())
    planes = features.reshape(NUM_PLANES, 64).sum(axis=1)

    print("ones:", features.sum())  # expected: 32
    print("per plane:", planes)  # expected: [8 2 2 2 1 1 8 2 2 2 1 1]

    assert features.sum() == 32
    assert list(planes) == [8, 2, 2, 2, 1, 1] * 2
    print("encoding test OK")