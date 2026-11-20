import time
import random

import chess

from chess import Board, Move

from fiveaichess.agents.agent import Agent
from fiveaichess.utilities.utility import Utility

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
        # TODO: Add transposition table (dict: zobrist hash -> depth, score, flag, best move)
        #       Use chess.polyglot.zobrist_hash(board) as key
        # TODO: Add killer moves (2 per ply) and history table for move ordering

    # Sort moves in order to make alpha-beta pruning faster
    def order_moves(self, board: Board):
        captures = []
        others = []

        for move in board.legal_moves:
            if board.is_capture(move):
                captures.append(move)
            else:
                others.append(move)
        # Sort captures before others
        # TODO: Better ordering: TT move > captures sorted by MVV-LVA > promotions > killer moves > history
        return captures + others



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

        # If depth = empty or the game is over: give evaluation back
        # TODO: At depth 0 call self.quiescence() instead of the utility (horizon effect)
        if depth == 0 or board.is_game_over():
            return self.utility.board_value(board)

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
    #       - Stand pat: evaluate the position, return if >= beta
    #       - Only search captures (and promotions), ordered by MVV-LVA
    #       - Stops the search from ending in the middle of a capture sequence
    def quiescence(self, board : Board, alpha : float, beta : float):
        pass

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
    def calculate_move(self, board: Board, constraints: dict[str, int] = {}) -> Move:
        start_time = time.time()

        self.nodes = 0 # Reset at the start of every move

        # TODO: No random fallback: use the first move of the ordered list (reproducible tests),
        #       with iterative deepening: the best move of the last finished depth
        # TODO: Return None ("bestmove 0000") when there are no legal moves, now random.choice crashes
        best_move = random.choice(list(board.legal_moves))
        maximizing_player = True if board.turn == chess.WHITE else False
        alpha = -float('inf')
        beta = float('inf')
        max_depth = constraints.get('depth', self.depth) # depth from the GUI, otherwise our default

        ordered_moves = self.order_moves(board)

        if maximizing_player:
            best_value = -float('inf')
        else:
            best_value = float('inf')

        for move in ordered_moves:
            # TODO: Temporary time check, only between root moves (one root move can still take much longer)
            #       Replace by iterative deepening + time check inside alphabeta every 1000 nodes
            if time.time() - start_time > self.time_limit_move:
                break

            # Play move
            board.push(move)
            # Evaluate recursively
            value = self.alphabeta(board, max_depth - 1, alpha, beta, not maximizing_player) # Opponent's turn after our move
            # Reverse move
            board.pop()

            # Check if current value is the best move
            if maximizing_player and value > best_value or not maximizing_player and value < best_value:
                best_value = value
                best_move = move
                print(f"info depth {max_depth} score cp {int(best_value * 100)} pv {best_move.uci()}")

            # Update the bound of the side to move
            if maximizing_player:
                alpha = max(alpha, best_value)
            else:
                beta = min(beta, best_value)
        return best_move
