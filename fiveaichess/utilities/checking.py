import chess

from chess import Board, Move

from fiveaichess.utilities.utility import Utility


"""
Check utility
-------------
Makes the goal not only to capture as many pieces of the opponent as possible,
but also to take away the moves of the opponent's king.
"""

# TODO: Remove this file once the search handles mate scores and draws (+ remove import in combined_utility.py)

class check(Utility):

    def board_value(self, board: Board):
        if board.turn == chess.WHITE:
            color = True
        else:
            color = False
        score = self.checkmatePossibility(board, color)
        return score

    def checkmatePossibility(self, board: Board, color : bool) -> int:
        if color:
            if board.is_checkmate():
                return -1000
            elif board.is_check():
                return -2
        else:
            if board.is_checkmate():
                return 1000
            elif board.is_check():
                return 2
        return 0