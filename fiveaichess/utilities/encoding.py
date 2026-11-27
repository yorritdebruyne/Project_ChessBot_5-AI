import chess
import numpy as np

from chess import Board


"""
encoding
--------
Encodes a board for the neural networks, from the perspective of the player to move
(same convention as NNUE): the network learns "good for the side to move" once,
instead of separately for white and for black.
- pack(): 12 bitboards (uint64), compact storage for the dataset (96 bytes per position)
- unpack(): 12 bitboards -> 768 one-hot features, the input of the network
Plane 0-5 = own pawn, knight, bishop, rook, queen, king
Plane 6-11 = the same pieces of the opponent
feature[plane * 64 + square] = 1, square 0 = a1 ... 63 = h8.
When black is to move, the board is mirrored vertically, so the own pieces always start on rank 1.
Shared by the training code (nn/) and the engine (nn_utility.py), so the network
always gets the same input while training and while playing.
"""

# Labels: centipawns from the side to move, clipped to [-CLIP, CLIP] and divided by CLIP -> [-1, 1]
CLIP = 1000
# Mate gets a score far above the clip, so it always becomes -1 or 1
MATE_CP = 10000

NUM_PLANES = 12
NUM_FEATURES = NUM_PLANES * 64  # 768


def pack(board: Board) -> np.ndarray:
    player_color = board.turn
    bitboards = np.zeros(NUM_PLANES, dtype=np.uint64)

    # First the own pieces (plane 0-5), then the opponent's pieces (plane 6-11)
    for side, color in enumerate((player_color, not player_color)):
        for piece_type in range(chess.PAWN, chess.KING + 1):
            bitboard = int(board.pieces(piece_type, color))
            # Black to move: mirror the board vertically (rank 8 <-> rank 1)
            if player_color == chess.BLACK:
                bitboard = chess.flip_vertical(bitboard)
            bitboards[side * 6 + piece_type - 1] = bitboard

    return bitboards


def unpack(bitboards: np.ndarray) -> np.ndarray:
    # 12 uint64 -> 96 bytes -> 768 bits, little bit order: bit 0 (a1) of plane 0 first
    # Result as uint8 (0/1), the dataset converts to float per position
    return np.unpackbits(bitboards.view(np.uint8), bitorder="little")


if __name__ == "__main__":
    # Test 1: start position (white to move) must have 32 pieces, with the right number per plane
    features = unpack(pack(Board()))
    planes = features.reshape(NUM_PLANES, 64).sum(axis=1)

    print("ones:", features.sum())  # expected: 32
    print("per plane:", planes)  # expected: [8 2 2 2 1 1 8 2 2 2 1 1]

    assert features.shape == (NUM_FEATURES,)
    assert features.sum() == 32
    assert list(planes) == [8, 2, 2, 2, 1, 1] * 2

    # Test 2: after 1.e4 black is to move, so plane 0-5 must be the black pieces, mirrored
    board = Board()
    board.push_uci("e2e4")
    features = unpack(pack(board)).reshape(NUM_PLANES, 64)

    assert features.sum() == 32
    # Own (black) king on e8, mirrored -> e1
    assert features[5, chess.E1] == 1
    # Opponent (white) pawn on e4, mirrored -> e5
    assert features[6, chess.E5] == 1

    # Test 3: a position and its colour-mirrored version (same position for the other side to move)
    # must give exactly the same features, otherwise the perspective is wrong
    board = Board("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    assert np.array_equal(unpack(pack(board)), unpack(pack(board.mirror())))

    print("encoding test OK")
