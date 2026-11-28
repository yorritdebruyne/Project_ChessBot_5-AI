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
# Draw handling is implemented for stalemate, insufficient material and the
# 75-move rule. Add repetition and 50-move-rule handling explicitly if needed.


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
        self.iteration_results: list[tuple[int, Move, float, int]] = []
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
                try:
                    value = self.alphabeta(board, depth - 1, alpha, beta, False)
                finally:
                    # Reverse move even if the search is interrupted by a timeout
                    board.pop()
                # Evaluate values until the optimal move is determined
                best_value = max(value, best_value)
                # Evaluate if alpha is the highest value
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    # TODO: Store quiet moves for killer/history move ordering
                    break # Prune: no value in searching any further
            return best_value

        # Player is minimizing player
        else:
            worst_value = float('inf')
            for move in ordered_moves:
                # Play move
                board.push(move)
                # Evaluate recursively
                try:
                    value = self.alphabeta(board, depth - 1, alpha, beta, True)
                finally:
                    # Reverse move even if the search is interrupted by a timeout
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

    # Extends the search for captures and promotions to avoid evaluating a
    # position in the middle of a tactical exchange. Tactical moves are
    # ordered by the existing MVV-LVA-style capture score.
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

    # Chooses the best move with iterative deepening and alpha-beta pruning.
    # Search time comes from movetime, the remaining clock and increment,
    # or the configured fallback time limit.
    # Each completed iteration stores its depth, score, node count and principal move.
    # The search checks its deadline at the root and every 1024 nodes.
    # TODO: Add UCI time reporting and mate scores ("score mate n").
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
        self.iteration_results.clear() # Reset at the start of every move

        search_board = board.copy(stack=True)
        maximizing_player = True if board.turn == chess.WHITE else False
        max_depth = constraints.get('depth', self.depth) # depth from the GUI, otherwise our default

        ordered_moves = self.order_moves(search_board)
        if not ordered_moves:
            return None  # No legal moves available

        last_best_move = ordered_moves[0]

        for depth in range(1, max_depth + 1):
            iteration_moves = ordered_moves.copy()
            if last_best_move in iteration_moves:
                iteration_moves.remove(last_best_move)
                iteration_moves.insert(0, last_best_move)
            
            iteration_best_move = None
            iteration_best_value = -float('inf') if maximizing_player else float('inf')
            alpha = -float('inf')
            beta = float('inf')
            nodes_before_iteration = self.nodes
            iteration_completed = False
            
            try:
                for move in iteration_moves:
                    if time.perf_counter() >= self.search_deadline:
                        raise SearchTimeout()
                
                    search_board.push(move)
                    try:
                        value = self.alphabeta(search_board, depth - 1, alpha, beta, not maximizing_player)
                    finally:
                        search_board.pop()
                    
                    if maximizing_player:
                        if value > iteration_best_value:
                            iteration_best_value = value
                            iteration_best_move = move
                        alpha = max(alpha, iteration_best_value)
                    else:
                        if value < iteration_best_value:
                            iteration_best_value = value
                            iteration_best_move = move
                        beta = min(beta, iteration_best_value)

                iteration_completed = True
                        
            except SearchTimeout:
                break

            if not iteration_completed or iteration_best_move is None:
                break

            last_best_move = iteration_best_move
            iteration_nodes = self.nodes - nodes_before_iteration
            self.iteration_results.append((depth, iteration_best_move, iteration_best_value, iteration_nodes))
            
            score = iteration_best_value if maximizing_player else -iteration_best_value
            print(
            f"info depth {depth} score cp {int(score * 100)} "
            f"nodes {iteration_nodes} pv {last_best_move.uci()}",
            )
        return last_best_move
        
