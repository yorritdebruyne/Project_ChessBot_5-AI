import chess
import numpy as np

CLIP = 1000  # centipawns; labels zijn cp / CLIP in [-1, 1]
MATE_CP = 10000


def pack(board: chess.Board) -> np.ndarray:
    """12 bitboards (uint64), vanuit het perspectief van de speler aan zet.
    Index 0-5: eigen pion, paard, loper, toren, dame, koning.
    Index 6-11: dezelfde stukken van de tegenstander.
    Bit 0 = a1 ... bit 63 = h8. Als zwart aan zet is, wordt het bord verticaal gespiegeld."""
    player_color = board.turn
    bitboards = np.zeros(12, dtype=np.uint64)

    for side, color in enumerate((player_color, not player_color)):
        for piece_type in range(chess.PAWN, chess.KING + 1):
            bitboard = int(board.pieces(piece_type, color))
            if player_color == chess.BLACK:
                bitboard = int.from_bytes(
                    bitboard.to_bytes(8, "little")[::-1],
                    "little",
                )
            bitboards[side * 6 + piece_type - 1] = bitboard

    return bitboards