import time

import chess

from chess import Board, Move

from fiveaichess.agents.agent import Agent
from fiveaichess.utilities.utility import Utility


class SearchTimeout(Exception):
    pass

"""
Minimax Agent with alpha-beta pruning
-------------------------------------
This agent uses a search algorithm to evaluate moves.
He can use minimax with or without alpha-beta pruning.
Alpha-beta pruning realises the same output as the classic minimax,
but calculates faster by pruning the branches 
when they are not relevant anymore.

"""

# TODO: Re-run the symmetry test after every change to the search
# TODO: Add mate score constant (e.g. MATE = 100000) and return -MATE + ply when checkmated,
#       so the agent prefers the fastest mate
# TODO: Return 0 for draws (stalemate, insufficient material, repetition, 50-move rule)
#       instead of board.is_game_over() + utility (is_game_over() is slow at every node)


class MinimaxAgent(Agent):
    # TODO: Give the agent an own name
    name = "5AIChessAgent"

    author = "Group5"

    def __init__(self, utility : Utility, time_limit_move : float, depth):
        super().__init__(utility, time_limit_move)
        # Save in class
        self.depth = depth
        self.piece_values= {
                chess.PAWN: 100,
                chess.KNIGHT: 320,
                chess.BISHOP: 330,
                chess.ROOK: 500,
                chess.QUEEN: 900,
                chess.KING: 20000
            }
        # TODO: Add transposition table (dict: zobrist hash -> depth, score, flag, best move)
        #       Use chess.polyglot.zobrist_hash(board) as key
        # TODO: Add killer moves (2 per ply) and history table for move ordering

    # Sort moves in order to make alpha-beta pruning faster
    def order_moves(self, board: Board):
        captures = []
        promotions = []
        checks = []
        others = []

        for move in board.legal_moves:
            if board.is_capture(move):
                captures.append(move)
            elif move.promotion:
                promotions.append(move)
            else:
                board.push(move)
                if board.is_check():
                    checks.append(move)
                else:
                    others.append(move)
                board.pop()

        captures.sort(key=lambda move: self.capture_score(board, move), reverse=True)
        promotions.sort(key=lambda move: self.piece_values[move.promotion], reverse=True)

        return captures + promotions + checks + others

    def capture_score(self, board: Board, move: Move):
        captured_piece = board.piece_at(move.to_square)

        if board.is_en_passant(move):
            captured_piece = chess.Piece(chess.PAWN, not board.turn)

        if captured_piece is None:
            return 0

        attacking_piece = board.piece_at(move.from_square)

        captured_value = self.piece_values[captured_piece.piece_type]
        attacking_value = (self.piece_values[attacking_piece.piece_type] if attacking_piece else 0)

        score = captured_value *10 - attacking_value

        if move.promotion:
            score += self.piece_values[move.promotion]

        return score



    # ---------------------------------------
    # --- Classic minimax without pruning ---
    # ---------------------------------------

    # Only kept as reference: compare node count with alpha-beta in the report
    # Recursively simulates future moves to evaluate board positions
    def minimax(self, board : Board , depth : int, maximizing_player : bool):
        self.nodes += 1

        # If depth = empty or the game is over: give evaluation back
        if depth == 0 or board.is_game_over():
            return self.utility.board_value(board)

        # Player is maximizing player
        if maximizing_player:
            best_value = - float('inf')
            for move in list(board.legal_moves):
                # Play move
                board.push(move)
                # Calculate new value
                value = self.minimax(board, depth - 1, False)
                # Reverse move
                board.pop()
                # Evaluate values until the optimal move is determined
                best_value = max(value, best_value)
            return best_value

        # Player is minimizing player
        else:
            worst_value = float('inf')
            for move in list(board.legal_moves):
                # Play move
                board.push(move)
                # Evaluate recursively
                value = self.minimax(board, depth - 1, True)
                # Reverse move
                board.pop()
                # Evaluate values until the optimal move is determined
                worst_value = min(value, worst_value)
            return worst_value

    #-----------------------------------
    # --- Alpha-beta pruning version ---
    # ----------------------------------

    # TODO: Probe the transposition table at the start, store the result at the end
    # TODO: Later: late move reductions, extensions (check, singular) (lecture 8)
    def alphabeta(self, board : Board, depth : int, alpha : float, beta : float, maximizing_player : bool):
        self.nodes += 1 # Nodes expanded

        if self.nodes % 1024 == 0 and time.perf_counter() >= self.search_deadline:
            raise SearchTimeout()

        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -100000
            return 100000

        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
        ):
            return 0

        if depth == 0:
            return self.quiescence(board, alpha, beta)

        # Use sorted order
        ordered_moves = self.order_moves(board)

        # Player is maximizing player
        if maximizing_player:
            best_value = - float('inf')
            for move in ordered_moves:
                # Play move
                board.push(move)
                # Calculate new value
                value = self.alphabeta(board, depth - 1, alpha, beta, False)
                # Reverse move
                board.pop()
                # Evaluate values until the optimal move is determined
                best_value = max(value, best_value)
                # Evaluate if alpha is the highest value
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    # TODO: Store quiet move as killer move + update history
                    break # Prune: no value in searching any further
            return best_value

        # Player is minimizing player
        else:
            worst_value = float('inf')
            for move in ordered_moves:
                # Play move
                board.push(move)
                # Evaluate recursively
                value = self.alphabeta(board, depth - 1, alpha, beta,True)
                # Reverse move
                board.pop()
                # Evaluate values until the optimal move is determined
                worst_value = min(value, worst_value)
                # Evaluate if beta is the lowest value
                beta = min(beta, worst_value)
                if beta <= alpha:
                    break  # Prune: no value in searching any further
            return worst_value

    # ---------------------------
    # --- Quiescence search ---
    # ---------------------------

    # TODO: Implement quiescence search
    #       - Only search captures (and promotions), ordered by MVV-LVA
    #       - Stops the search from ending in the middle of a capture sequence
    def quiescence(self, board: Board, alpha: float, beta: float):
        self.nodes += 1

        if self.nodes % 1024 == 0 and time.perf_counter() >= self.search_deadline:
            raise SearchTimeout()

        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -100000
            return 100000

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        stand_pat = self.utility.board_value(board)

        if board.turn == chess.WHITE:
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return stand_pat
            beta = min(beta, stand_pat)

        tactical_moves = [
            move for move in board.legal_moves
            if board.is_capture(move) or move.promotion
        ]

        tactical_moves.sort(
            key=lambda move: self.capture_score(board, move),
            reverse=True,
        )

        if board.turn == chess.WHITE:
            best_value = stand_pat

            for move in tactical_moves:
                board.push(move)
                try:
                    value = self.quiescence(board, alpha, beta)
                finally:
                    board.pop()

                best_value = max(best_value, value)
                alpha = max(alpha, best_value)

                if alpha >= beta:
                    break

            return best_value

        worst_value = stand_pat

        for move in tactical_moves:
            board.push(move)
            try:
                value = self.quiescence(board, alpha, beta)
            finally:
                board.pop()

            worst_value = min(worst_value, value)
            beta = min(beta, worst_value)

            if beta <= alpha:
                break

        return worst_value

    # ----------------------
    # --- Move Selection ---
    # ----------------------

    # Chooses the best current move by evaluating all options with alpha-beta pruning
    # TODO: Iterative deepening: search depth 1, 2, 3, ... until the time is up,
    #       start each iteration with the best move of the previous one
    # TODO: Time management: calculate the time for this move from constraints
    #       (movetime, or wtime/btime + winc/binc / movestogo), fall back on self.time_limit_move
    # TODO: Check the time inside the search (e.g. every 1000 nodes), not only at the root
    # TODO: Print "info depth .. score cp .. nodes .. time .. pv .." per iteration,
    #       score from the side to move's view (not white's) and "score mate n" for mates
    # TODO: Later: opening book (chess.polyglot) and endgame tablebases (chess.syzygy)
    def calculate_move(self, board: Board, constraints: dict[str, int] = {}) -> Move | None:
        if 'movetime' in constraints:
            search_time = constraints['movetime'] / 1000
        else:
            own_time_key = 'wtime' if board.turn == chess.WHITE else 'btime'
            own_increment_key = 'winc' if board.turn == chess.WHITE else 'binc'
            own_time = constraints.get(own_time_key)
            own_increment = constraints.get(own_increment_key, 0)

            if own_time is not None:
                moves_to_go = constraints.get('movestogo', 40)
                moves_to_go = moves_to_go if moves_to_go > 0 else 40
                search_time = min(
                    self.time_limit_move,
                    (own_time / 1000) / moves_to_go + (own_increment / 1000) * 0.8,
                )
            else:
                search_time = self.time_limit_move

        search_time = max(0.05, search_time)
        self.search_deadline = time.perf_counter() + search_time

        self.nodes = 0 # Reset at the start of every move

        search_board = board.copy(stack=True)
        maximizing_player = True if board.turn == chess.WHITE else False
        alpha = -float('inf')
        beta = float('inf')
        max_depth = constraints.get('depth', self.depth) # depth from the GUI, otherwise our default

        ordered_moves = self.order_moves(search_board)

        if not ordered_moves:
            return None  # No legal moves available

        best_move = ordered_moves[0]

        if maximizing_player:
            best_value = -float('inf')
        else:
            best_value = float('inf')

        for move in ordered_moves:
            # Play move
            search_board.push(move)
            try:
                # Evaluate recursively
                value = self.alphabeta(search_board, max_depth - 1, alpha, beta, not maximizing_player) # Opponent's turn after our move
            except SearchTimeout:
                search_board.pop()
                break
            else:
                # Reverse move
                search_board.pop()

            # Check if current value is the best move
            if maximizing_player and value > best_value or not maximizing_player and value < best_value:
                best_value = value
                best_move = move
                print(
                    f"info depth {max_depth} score cp {int(best_value * 100)} pv {best_move.uci()}",
                    flush=True,
                )

            # Update the bound of the side to move
            if maximizing_player:
                alpha = max(alpha, best_value)
            else:
                beta = min(beta, best_value)
        return best_move
