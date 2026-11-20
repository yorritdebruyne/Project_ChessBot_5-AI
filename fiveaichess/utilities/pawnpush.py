import chess

from chess import Board, Move

from fiveaichess.utilities.utility import Utility


"""
Pawn structure utility
----------------------
Evaluates the pawn structure of both sides.
- Passed pawns get more points than normal pawns, since they have more chance to promote.
  The further they are pushed, the more points (especially in the endgame).
- Doubled pawns get less points, since they block each other.
- Isolated pawns (no own pawns on the neighbouring files) get less points.
- Connected pawns get more points, since they protect each other.
Positive score = advantage for WHITE
Negative score = advantage for BLACK
"""

# TODO: Rename file and class to pawn_structure_utility.py / PawnStructureUtility (+ update import)
# TODO: Score must NOT depend on board.turn: always white minus black
# TODO: Rank for black is mirrored: use chess.square_rank(square) for white
#       and 7 - chess.square_rank(square) for black
# TODO: Doubled pawns: count pawns per file (chess.square_file), penalty for each extra pawn
# TODO: Isolated pawns: no own pawns on file - 1 and file + 1
# TODO: Connected pawns: bonus when an own pawn protects it (diagonally behind) or stands next to it
# TODO: Passed pawns: no enemy pawns in front on the same or neighbouring files,
#       bonus per rank (bigger in the endgame)
# TODO: Return middlegame and endgame score in centipawns


class pawnPush(Utility):

    def board_value(self, board: Board):
        return self.pawnPush(board)

    def pawnPush(self, board: Board) -> int:
        whitePawns = board.pieces(chess.PAWN, chess.WHITE)
        blackPawns = board.pieces(chess.PAWN, chess.BLACK)

        # TODO: Implement doubled, isolated and passed pawns (see TODOs at the top)
        return 0