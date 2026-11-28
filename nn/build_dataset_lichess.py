import argparse
import io
import json
import os
import zlib

import chess
import numpy as np
import zstandard as zstd

from fiveaichess.utilities.encoding import MATE_CP, pack

"""
build_dataset_lichess
---------------------
Converts lichess_db_eval.jsonl.zst into shards (.npz) for the value and policy networks (runs once, offline).
The file is read as a stream, so the 20 GB download is never unpacked as a whole.
Written by Maarten, adapted by Yorrit (28/11, decisions D1-D4 in the plan):
- D1: the shards store raw centipawns (side to move, mate = +-MATE_CP), dataset.py chooses the label
- D3: split 98/1/1 (fixed via a hash of the FEN)
- D4: every k-th line (spread over the whole file), only quiet positions (not in check, best move is
      no capture), and the best move (first move of the PV) is stored for the policy network

Every shard contains:
- X:    12 uint64 bitboards per position, from the side to move (pack() in fiveaichess/utilities/encoding.py)
- cp:   Stockfish evaluation in centipawns from the side to move (int16)
- move: best move as from_square * 64 + to_square, mirrored like the board when black is to move (uint16)

Usage (from the project root):
    python -m nn.build_dataset_lichess --src ..\\datasets\\lichess_db_eval.jsonl.zst --out data/shards_test --limit 1000000
    python -m nn.build_dataset_lichess --src ..\\datasets\\lichess_db_eval.jsonl.zst --out data/shards --max-positions 5000000
"""

SPLITS = ("train", "val", "test")


def split_of(fen: str) -> str:
    # Hash of the position (without move counters): the same position always lands in the same split
    key = " ".join(fen.split()[:4])
    bucket = zlib.crc32(key.encode()) % 100
    if bucket < 98:
        return "train"
    if bucket < 99:
        return "val"
    return "test"


def best_evaluation(evaluations: list[dict]) -> tuple[int, str] | None:
    """Evaluation with the highest depth, first variation. Returns (centipawns from white, first move in UCI) or None."""
    evaluation = max(evaluations, key=lambda item: item.get("depth", 0))
    principal_variation = evaluation["pvs"][0]
    first_move = principal_variation["line"].split()[0]
    if "cp" in principal_variation:
        return principal_variation["cp"], first_move
    if "mate" in principal_variation:
        return (MATE_CP if principal_variation["mate"] > 0 else -MATE_CP), first_move
    return None


def move_index(move: chess.Move, player_color: bool) -> int:
    # Same perspective as pack(): when black is to move, the board is mirrored, so the move is mirrored too
    from_square, to_square = move.from_square, move.to_square
    if player_color == chess.BLACK:
        from_square = chess.square_mirror(from_square)
        to_square = chess.square_mirror(to_square)
    return from_square * 64 + to_square


class ShardWriter:
    def __init__(self, out_dir: str, shard_size: int) -> None:
        self.out_dir, self.shard_size = out_dir, shard_size
        self.features = {split: [] for split in SPLITS}
        self.cps = {split: [] for split in SPLITS}
        self.moves = {split: [] for split in SPLITS}
        self.n_shards = {split: 0 for split in SPLITS}
        self.n_positions = {split: 0 for split in SPLITS}
        os.makedirs(out_dir, exist_ok=True)

    def add(self, split: str, features: np.ndarray, cp: int, move: int) -> None:
        self.features[split].append(features)
        self.cps[split].append(cp)
        self.moves[split].append(move)
        if len(self.cps[split]) >= self.shard_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        if not self.cps[split]:
            return
        features = np.array(self.features[split], dtype=np.uint64)
        cps = np.array(self.cps[split], dtype=np.int16)
        moves = np.array(self.moves[split], dtype=np.uint16)
        permutation = np.random.permutation(len(cps))  # shuffle within the shard
        path = os.path.join(self.out_dir, f"{split}_{self.n_shards[split]:05d}.npz")
        np.savez_compressed(
            path,
            X=features[permutation],
            cp=cps[permutation],
            move=moves[permutation],
        )
        self.n_shards[split] += 1
        self.n_positions[split] += len(cps)
        self.features[split], self.cps[split], self.moves[split] = [], [], []

    def close(self) -> None:
        for split in SPLITS:
            self.flush(split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", default="data/shards")
    parser.add_argument("--shard-size", type=int, default=1_000_000)
    parser.add_argument("--sample-every", type=int, default=100, help="keep every k-th line (spread over the file)")
    parser.add_argument("--limit", type=int, default=None, help="max number of lines to read (test)")
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="stop after this many valid positions",
    )
    args = parser.parse_args()

    writer = ShardWriter(args.out, args.shard_size)
    kept_positions = 0
    skipped_noisy = 0  # in check or best move is a capture
    skipped_lines = 0  # no usable evaluation or broken line

    dctx = zstd.ZstdDecompressor()
    with open(args.src, "rb") as fh, dctx.stream_reader(fh) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        for line_number, line in enumerate(text):
            if args.limit and line_number >= args.limit:
                break
            if args.max_positions and kept_positions >= args.max_positions:
                break

            # Skip before json.loads: only the kept lines cost parsing time
            if line_number % args.sample_every != 0:
                continue

            if line_number % 1_000_000 == 0:
                print(
                    f"read {line_number:,} | kept {kept_positions:,} "
                    f"| skipped noisy {skipped_noisy:,} | skipped broken {skipped_lines:,}",
                    flush=True,
                )

            try:
                record = json.loads(line)
                evaluation = best_evaluation(record["evals"])
                if evaluation is None:
                    skipped_lines += 1
                    continue
                score, first_move = evaluation

                fen = record["fen"]
                if len(fen.split()) == 4:
                    fen += " 0 1"
                board = chess.Board(fen)
                # parse_uci also checks that the move is legal (IllegalMoveError is a ValueError), and turns
                # Lichess' castling notation (king takes rook, e1h1) into the standard one (e1g1), the same
                # as the moves the engine generates -> the policy network learns the moves the engine asks about
                move = board.parse_uci(first_move)

                # Quiet positions only: in the middle of a capture sequence or a check the score swings,
                # so it's a bad training label (same reason as quiescence in the search)
                if board.is_check() or board.is_capture(move):
                    skipped_noisy += 1
                    continue

                # Lichess scores are from white, the network learns from the side to move
                if board.turn == chess.BLACK:
                    score = -score
                score = max(-MATE_CP, min(MATE_CP, score))

                writer.add(split_of(record["fen"]), pack(board), score, move_index(move, board.turn))
                kept_positions += 1
            except (KeyError, ValueError, IndexError):
                skipped_lines += 1

    writer.close()
    summary = {
        split: (writer.n_shards[split], writer.n_positions[split])
        for split in SPLITS
    }
    print(f"kept {kept_positions:,} | skipped noisy {skipped_noisy:,} | skipped broken {skipped_lines:,}")
    print("done (shards, positions):", summary)


if __name__ == "__main__":
    main()
