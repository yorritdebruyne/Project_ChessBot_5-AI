import os
import math
import argparse
import contextlib
import io
from datetime import datetime

import chess
import chess.engine
import chess.pgn

from fiveaichess.agents.minimax_agent import MinimaxAgent
from fiveaichess.utilities.combined_utility import CombinedUtility


"""
AI-assisted (Claude), test tooling, not part of the engine

Play a match between two players and estimate the Elo difference

A player is one of our agents or a UCI engine (e.g. Stockfish).
Every opening is played twice, once with each color, so the match is fair.
All games are saved in a PGN file in the results folder.

Usage: python scripts/match.py --games 20 --sf-skill 0
       python scripts/match.py --games 20 --sf-elo 1400
Stockfish is found via the environment variable UCI_ENGINE_PATH.
"""

# Short openings (UCI moves) to get different games
OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6",
    "e2e4 c7c5 g1f3 d7d6",
    "e2e4 e7e6 d2d4 d7d5",
    "e2e4 c7c6 d2d4 d7d5",
    "d2d4 d7d5 c2c4 e7e6",
    "d2d4 g8f6 c2c4 g7g6",
    "c2c4 e7e5 b1c3 g8f6",
    "g1f3 d7d5 g2g3 g8f6",
    "e2e4 d7d5 e4d5 d8d5",
    "d2d4 g8f6 c2c4 e7e6",
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')


"""
Player that uses one of our agents
"""
class AgentPlayer:

    def __init__(self, name: str, agent):
        self.name = name
        self.agent = agent

    def play(self, board: chess.Board) -> chess.Move:
        # Hide the "info ..." lines and give the agent a copy, so it can't change the real board
        with contextlib.redirect_stdout(io.StringIO()):
            return self.agent.calculate_move(board.copy())

    def quit(self):
        pass


"""
Player that uses a UCI engine (e.g. Stockfish)
"""
class UCIPlayer:

    def __init__(self, name: str, path: str, options: dict, time_limit: float):
        self.name = name
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure(options)
        self.limit = chess.engine.Limit(time=time_limit)

    def play(self, board: chess.Board) -> chess.Move:
        return self.engine.play(board, self.limit).move

    def quit(self):
        self.engine.quit()


def create_players(args):
    # TODO: Change player A / player B to compare other versions (e.g. old agent vs new agent)
    player_a = AgentPlayer(f"MinimaxAgent d{args.depth}",
                           MinimaxAgent(CombinedUtility(), args.agent_time, depth=args.depth))

    uci_engine_path = os.getenv('UCI_ENGINE_PATH')
    if not uci_engine_path:
        raise EnvironmentError('Environment variable UCI_ENGINE_PATH not defined')

    if args.sf_elo:
        options = {'UCI_LimitStrength': True, 'UCI_Elo': args.sf_elo}
        name = f"Stockfish Elo {args.sf_elo}"
    else:
        options = {'Skill Level': args.sf_skill}
        name = f"Stockfish skill {args.sf_skill}"
    player_b = UCIPlayer(name, uci_engine_path, options, args.sf_time)

    return player_a, player_b


def play_game(white, black, opening: str, max_plies: int) -> chess.pgn.Game:
    board = chess.Board()
    for move in opening.split():
        board.push_uci(move)

    result = None
    while not board.is_game_over(claim_draw=True):
        # Adjudicate as draw when the game takes too long
        if board.ply() >= max_plies:
            result = '1/2-1/2'
            break

        player = white if board.turn == chess.WHITE else black
        move = player.play(board)

        # An illegal move loses the game
        if move is None or move not in board.legal_moves:
            print(f"Illegal move {move} by {player.name}")
            result = '0-1' if board.turn == chess.WHITE else '1-0'
            break

        board.push(move)

    if result is None:
        result = board.result(claim_draw=True)

    game = chess.pgn.Game.from_board(board)
    game.headers['White'] = white.name
    game.headers['Black'] = black.name
    game.headers['Result'] = result
    return game


# Elo difference from a score between 0 and 1
def elo_difference(score: float) -> float:
    if score <= 0:
        return -math.inf
    if score >= 1:
        return math.inf
    return -400 * math.log10(1 / score - 1)


def play_match(args):
    player_a, player_b = create_players(args)
    wins = draws = losses = 0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pgn_path = os.path.join(RESULTS_DIR, f"match_{datetime.now():%Y%m%d_%H%M}.pgn")

    try:
        for i in range(args.games):
            opening = OPENINGS[(i // 2) % len(OPENINGS)]
            # Switch colors every game
            a_is_white = i % 2 == 0
            white, black = (player_a, player_b) if a_is_white else (player_b, player_a)

            game = play_game(white, black, opening, args.max_plies)
            result = game.headers['Result']

            # Count the result from player A's point of view
            if result == '1/2-1/2':
                draws += 1
            elif (result == '1-0') == a_is_white:
                wins += 1
            else:
                losses += 1

            with open(pgn_path, 'a') as pgn_file:
                print(game, file=pgn_file, end='\n\n')

            print(f"Game {i + 1}/{args.games}: {white.name} - {black.name} {result}   "
                  f"(score {player_a.name}: +{wins} ={draws} -{losses})")
    finally:
        player_a.quit()
        player_b.quit()

    games = wins + draws + losses
    score = (wins + 0.5 * draws) / max(games, 1)
    print('-' * 50)
    print(f"{player_a.name} vs {player_b.name}: +{wins} ={draws} -{losses}  score {100 * score:.0f}%")
    print(f"Elo difference: {elo_difference(score):+.0f}  (only reliable with many games)")
    print(f"Games saved in {pgn_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--depth', type=int, default=3, help='depth of our agent')
    parser.add_argument('--agent-time', type=float, default=5.0, help='seconds per move for our agent')
    parser.add_argument('--sf-skill', type=int, default=0, help='Stockfish Skill Level (0-20)')
    parser.add_argument('--sf-elo', type=int, default=None, help='Stockfish UCI_Elo (overrules skill level)')
    parser.add_argument('--sf-time', type=float, default=0.1, help='seconds per move for Stockfish')
    parser.add_argument('--max-plies', type=int, default=300)
    args = parser.parse_args()

    play_match(args)
