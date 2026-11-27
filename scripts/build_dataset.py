import argparse
import io
import json
import os
import zlib

import chess
import numpy as np
import zstandard as zstd

from encoding import CLIP, MATE_CP, pack

"""Zet lichess_db_eval.jsonl.zst om naar shards (.npz).

Gebruik:
    python scripts/build_dataset.py --src lichess_db_eval.jsonl.zst --out data/shards --limit 100000
    python scripts/build_dataset.py --src lichess_db_eval.jsonl.zst --out data/shards --max-positions 20000000
"""

SPLITS = ("train", "val", "test")


def split_of(fen: str) -> str:
    key = " ".join(fen.split()[:4])
    bucket = zlib.crc32(key.encode()) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def best_score(evaluations: list[dict]) -> int | None:
    """Evaluatie met de hoogste diepte, eerste variant. Geeft centipawns (vanuit wit) of None."""
    evaluation = max(evaluations, key=lambda item: item.get("depth", 0))
    principal_variation = evaluation["pvs"][0]
    if "cp" in principal_variation:
        return principal_variation["cp"]
    if "mate" in principal_variation:
        return MATE_CP if principal_variation["mate"] > 0 else -MATE_CP
    return None


class ShardWriter:
    def __init__(self, out_dir: str, shard_size: int) -> None:
        self.out_dir, self.shard_size = out_dir, shard_size
        self.features = {split: [] for split in SPLITS}
        self.labels = {split: [] for split in SPLITS}
        self.n_shards = {split: 0 for split in SPLITS}
        self.n_positions = {split: 0 for split in SPLITS}
        os.makedirs(out_dir, exist_ok=True)

    def add(self, split: str, features: np.ndarray, label: float) -> None:
        self.features[split].append(features)
        self.labels[split].append(label)
        if len(self.labels[split]) >= self.shard_size:
            self.flush(split)

    def flush(self, split: str) -> None:
        if not self.labels[split]:
            return
        features = np.array(self.features[split], dtype=np.uint64)
        labels = np.array(self.labels[split], dtype=np.float32)
        permutation = np.random.permutation(len(labels))  # shuffle within the shard
        path = os.path.join(self.out_dir, f"{split}_{self.n_shards[split]:05d}.npz")
        np.savez_compressed(
            path,
            X=features[permutation],
            y=labels[permutation],
        )
        self.n_shards[split] += 1
        self.n_positions[split] += len(labels)
        self.features[split], self.labels[split] = [], []

    def close(self) -> None:
        for split in SPLITS:
            self.flush(split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", default="data/shards")
    parser.add_argument("--shard-size", type=int, default=1_000_000)
    parser.add_argument("--limit", type=int, default=None, help="max aantal regels lezen (test)")
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="stop na zoveel geldige posities",
    )
    args = parser.parse_args()

    writer = ShardWriter(args.out, args.shard_size)
    kept_positions = 0
    skipped_lines = 0

    dctx = zstd.ZstdDecompressor()
    with open(args.src, "rb") as fh, dctx.stream_reader(fh) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        for line_number, line in enumerate(text):
            if args.limit and line_number >= args.limit:
                break
            if args.max_positions and kept_positions >= args.max_positions:
                break

            try:
                record = json.loads(line)
                score = best_score(record["evals"])
                if score is None:
                    skipped_lines += 1
                    continue

                fen = record["fen"]
                if len(fen.split()) == 4:
                    fen += " 0 1"
                board = chess.Board(fen)
                if board.turn == chess.BLACK:
                    score = -score

                label = max(-CLIP, min(CLIP, score)) / CLIP
                writer.add(split_of(record["fen"]), pack(board), label)
                kept_positions += 1
            except (KeyError, ValueError, IndexError):
                skipped_lines += 1

            if line_number % 1_000_000 == 0:
                print(
                    f"gelezen {line_number:,} | bewaard {kept_positions:,} "
                    f"| overgeslagen {skipped_lines:,}",
                    flush=True,
                )

    writer.close()
    summary = {
        split: (writer.n_shards[split], writer.n_positions[split])
        for split in SPLITS
    }
    print("klaar:", summary)


if __name__ == "__main__":
    main()