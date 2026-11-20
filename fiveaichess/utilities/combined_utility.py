import chess
from chess import Board

from fiveaichess.utilities.utility import Utility

from fiveaichess.utilities.material_utility import MaterialUtility
from fiveaichess.utilities.mobility_utility import MobilityUtility
from fiveaichess.utilities.king_safety_utility import KingSafetyUtility
from fiveaichess.utilities.checking import check
from fiveaichess.utilities.pawnpush import pawnPush


"""
CombinedUtility
---------------
Evaluates a board position by combining multiple evaluation strategies:
- MaterialUtility: counts the material balance (pieces and their values + piece-square tables)
- MobilityUtility: counts the mobility for both sides
- KingSafetyUtility: pawn shield in front of the king
- pawnPush: pawn structure
The final score is a weighted sum of all evaluations.
Positive score = advantage for WHITE
Negative score = advantage for BLACK
"""

# TODO: Put all utilities on the same scale (centipawns: pawn = 100), now mobility can outweigh a rook
# TODO: Replace the ply-based phases by a game phase based on the remaining material
#       (knight/bishop = 1, rook = 2, queen = 4, max 24) and blend middlegame and endgame
#       score: (mg * phase + eg * (24 - phase)) / 24  (tapered evaluation)
# TODO: Let every utility return a middlegame and an endgame score
# TODO: Remove check utility once the search handles mate and draws
# TODO: Tune the weights (Texel tuning) instead of choosing them by hand
# TODO: Return the score as int centipawns and remove the *0.1


class CombinedUtility(Utility):

    def __init__(self):
        # Create instances of the individual utilities
        self.material = MaterialUtility()
        self.mobility = MobilityUtility()
        self.kingsafety = KingSafetyUtility()
        self.pawnPush = pawnPush()
        self.check = check()

    def board_value(self, board: Board):

        # Define weights: adjustable if a factor is more important
        # TODO: Replace by tapered evaluation (see TODOs at the top)
        if board.ply() < 10: #begingame
            material_weight = 2
            mobility_weight = 0.2
            kingsafety_weight = 1
            checkmate_weight = 0.5
            pawnpush_weight = 0
        elif board.ply() < 25: #middlegame
            material_weight = 1.3
            mobility_weight = 0.7
            kingsafety_weight = 1
            checkmate_weight = 1
            pawnpush_weight = 1
        else:
            material_weight = 1.0
            mobility_weight = 1.3
            kingsafety_weight = 0.2
            checkmate_weight = 3
            pawnpush_weight = 2

        # TODO: Remove when pawn structure utility is rewritten
        pawnpush_weight = 0

        # Get the evaluation from MaterialUtility (piece values)
        material_score = self.material.board_value(board)
        # Get the evaluation from MobilityUtility (number of legal moves)
        mobility_score = self.mobility.board_value(board)
        # Get the evaluation from KingSafetyUtility (score based on filled/empty squares surrounding king)
        kingsafety_score = self.kingsafety.board_value(board)
        # Get the evaluation from check utility (checkmate)
        check_score = self.check.board_value(board)
        # Get the evaluation from pawn structure
        # TODO: Only calculate when the weight is not 0 (saves time at every leaf)
        pawnpush_score = self.pawnPush.board_value(board)

        # Combine the scores using the chosen weights
        combined_score = material_score * material_weight + mobility_score * mobility_weight + \
                         kingsafety_score * kingsafety_weight + check_score * checkmate_weight + \
                         pawnpush_score * pawnpush_weight

        # Return the final combined evaluation
        return combined_score*0.1