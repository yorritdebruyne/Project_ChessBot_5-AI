# Results

Every measurement: date, commit, what changed, settings, result.
The report graphs and tables are made from this file.

## Baseline (old engine from last year, with orientation fix in material_utility)

| Date | Commit | Test | Settings | Result |
|---|---|---|---|---|
| 20/11 | c62f345 | bench | depth 3 | 20 330 evals (leaf nodes), 2.85 s, 7 128 evals/s |
| 20/11 | c62f345 | tactics | WAC 1-50, 5 s, depth 3 | 27/50 (54%) |
| 20/11 | c62f345 | match | 20 games vs Stockfish skill 0 (0.1 s/move), agent depth 3 | +1 =2 -17, score 10%, Elo diff -382 (white 0/10, black 2/10; symmetry test OK) |

## Search

| Date | Commit | Technique | bench (nodes / time, depth 3) | tactics | match |
|---|---|---|---|---|---|
| 20/11 | (hash after commit) | One alpha-beta copy at the root + node counter | 22 046 nodes / 2.62 s, same moves as baseline | - | - |

## Evaluation

| Date | Commit | Utility | bench | tactics | match |
|---|---|---|---|---|---|
