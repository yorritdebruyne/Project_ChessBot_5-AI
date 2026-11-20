import chess

from chess import Board, Move

from fiveaichess.utilities.utility import Utility


"""
MobilityUtility
---------------
Evaluates a board position based on mobility.
- Counts the number of legal moves for both WHITE & BLACK
- Returns the difference between WHITE moves and BLACK moves
Positive score = advantage for WHITE (more options)
Negative score = advantage for BLACK (more options)
"""

# TODO: Rewrite with attacks instead of legal moves:
#       for each knight/bishop/rook/queen: len(board.attacks(square)) minus the squares with own pieces
#       -> no move generation, no changing board.turn (gives wrong results when a king is in check)
# TODO: Different weight per piece type (e.g. knight/bishop more than queen)
# TODO: Return middlegame and endgame score in centipawns (e.g. 2-5 cp per square)


class MobilityUtility(Utility):


    def board_value(self, board: Board):

        # Save the original turn so we can restore it later
        original_board_turn = board.turn

        # Temporary set turn to WHITE
        board.turn = chess.WHITE
        # Count legal moves for WHITE
        mobility_white = len(list(board.legal_moves))

        # Temporary set turn to BLACK
        board.turn = chess.BLACK
        # Count legal moves for BLACK
        mobility_black = len(list(board.legal_moves))

        # Restore the original turn to avoid disturbing the game flow
        board.turn = original_board_turn

        # Return the difference: positive means White has more mobility
        return (mobility_white - mobility_black) * 0.5