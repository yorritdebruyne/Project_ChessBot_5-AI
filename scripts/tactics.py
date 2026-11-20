import os
import time
import argparse
import contextlib
import io

import chess

from fiveaichess.agents.minimax_agent import MinimaxAgent
from fiveaichess.utilities.combined_utility import CombinedUtility


"""
AI-assisted (Claude), test tooling, not part of the engine

Run a tactical test suite (EPD file) and count how many positions the agent solves

Every line of the EPD file contains a position and the best move(s) (bm),
or a move that must be avoided (am).
Default suite: Win At Chess (WAC), 300 tactical positions.

Usage: python scripts/tactics.py --time 5 --limit 50
"""

DEFAULT_EPD = os.path.join(os.path.dirname(__file__), '../data/wac.epd')


def run_tactics(epd_path: str, time_limit: float, depth: int, limit: int):
    solved = 0
    tested = 0
    failed_ids = []
    start_total = time.time()

    with open(epd_path) as epd_file:
        lines = [line.strip() for line in epd_file if line.strip()]

    for line in lines[:limit]:
        board, ops = chess.Board.from_epd(line)
        position_id = ops.get('id', f"position {tested + 1}")

        # TODO: Replace by the agent/utility you want to measure
        agent = MinimaxAgent(CombinedUtility(), time_limit, depth=depth)

        # Hide the "info ..." lines of the agent
        with contextlib.redirect_stdout(io.StringIO()):
            move = agent.calculate_move(board, {'movetime': int(time_limit * 1000)})

        # Correct if the move is one of the best moves, or not the move to avoid
        if 'bm' in ops:
            correct = move in ops['bm']
            expected = ' '.join(board.san(m) for m in ops['bm'])
        else:
            correct = move not in ops.get('am', [])
            expected = 'not ' + ' '.join(board.san(m) for m in ops.get('am', []))

        tested += 1
        if correct:
            solved += 1
        else:
            failed_ids.append(position_id)

        print(f"{position_id:>10}  {'OK  ' if correct else 'FAIL'}  played {board.san(move):>7}  expected {expected}")

    print('-' * 50)
    print(f"Solved {solved}/{tested} ({100 * solved / max(tested, 1):.0f}%) in {time.time() - start_total:.0f} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epd', default=DEFAULT_EPD)
    parser.add_argument('--time', type=float, default=5.0, help='seconds per position')
    parser.add_argument('--depth', type=int, default=3, help='maximum depth (until iterative deepening exists)')
    parser.add_argument('--limit', type=int, default=50, help='number of positions to test')
    args = parser.parse_args()

    run_tactics(args.epd, args.time, args.depth, args.limit)
