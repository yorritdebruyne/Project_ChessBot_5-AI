import time
import argparse
import contextlib
import io

import chess

from fiveaichess.agents.minimax_agent import MinimaxAgent
from fiveaichess.utilities.combined_utility import CombinedUtility
from fiveaichess.utilities.utility import Utility


"""
AI-assisted (Claude), test tooling, not part of the engine

Benchmark the search on a fixed set of positions

For every position the agent searches to a fixed depth (no time limit).
Prints the number of nodes visited, the time and the nodes per second.
(Falls back on counting evaluations = leaf nodes for an agent without a node counter.)
Run this after every change to see the effect of a technique on the search.

Usage: python scripts/bench.py --depth 3
"""

# Fixed set of positions: opening, middlegame, tactical and endgame positions
BENCH_POSITIONS = [
    # Start position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Ruy Lopez after 3.Bb5
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    # Sicilian after 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq - 2 5",
    # Queen's Gambit Declined
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    # Kiwipete (many captures, castling, pins)
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    # Middlegame with both sides castled
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    # Promotion and checks
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    # Tactical position (WAC.001)
    "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
    # Rook endgame
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    # King and pawn endgame
    "8/8/4k3/8/2p5/8/B2K4/8 w - - 0 1",
]


"""
Wraps a utility and counts how many times board_value is called (= number of evaluated positions)
Works for every agent, also before the agent has its own node counter
"""
class CountingUtility(Utility):

    def __init__(self, utility: Utility):
        self.utility = utility
        self.count = 0

    def board_value(self, board: chess.Board):
        self.count += 1
        return self.utility.board_value(board)


def bench(depth: int):
    total_nodes = 0
    total_time = 0.0

    print(f"{'#':>2}  {'move':>6}  {'nodes':>9}  {'time (s)':>9}  {'nodes/s':>8}")

    for i, fen in enumerate(BENCH_POSITIONS):
        board = chess.Board(fen)

        # Create a new agent for every position (so a later transposition table starts empty)
        # TODO: Replace by the agent/utility you want to measure
        utility = CountingUtility(CombinedUtility())
        agent = MinimaxAgent(utility, 1e9, depth=depth)

        start_time = time.time()
        # Hide the "info ..." lines of the agent
        with contextlib.redirect_stdout(io.StringIO()):
            move = agent.calculate_move(board)
        elapsed = time.time() - start_time

        # Use the node counter of the agent, otherwise the number of evaluations (leaf nodes)
        nodes = getattr(agent, 'nodes', None) or utility.count
        total_nodes += nodes
        total_time += elapsed

        print(f"{i + 1:>2}  {move.uci():>6}  {nodes:>9}  {elapsed:>9.2f}  {nodes / max(elapsed, 1e-9):>8.0f}")

    print('-' * 42)
    print(f"depth {depth}: {total_nodes} nodes in {total_time:.2f} s ({total_nodes / max(total_time, 1e-9):.0f} nodes/s)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=3)
    args = parser.parse_args()

    bench(args.depth)
