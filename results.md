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
| 20/11 | abe95f7 | One alpha-beta copy at the root + node counter | 22 046 nodes / 2.62 s, same moves as baseline | - | - |
| 27/11 | 75d8495 | Maarten: quiescence, terminal states, time management, move ordering | 72 811 nodes / 9.82 s (incl. quiescence nodes, not comparable to 22 046) | 32/50 (+5) | - |

## Data (value and policy network)

| Date | Commit | Dataset | Settings | Result |
|---|---|---|---|---|
| 28/11 | 64343ac | Lichess eval database, test run | `--limit 1000000`, every 100th line | 10 000 lines read: 7 286 kept, 2 714 skipped as noisy (27%: in check or best move is a capture), 0 broken; 7 129 / 82 / 75 |
| 28/11 | 64343ac | Lichess eval database, full run (`data/shards`) | every 100th line of the whole file, quiet positions only, split 98/1/1 via FEN hash | 2 927 057 positions: train 2 868 353 / val 29 289 / test 29 415 (97.99 / 1.00 / 1.00%); mates 11.7%; median cp 0; ~21 pieces per position; 21 min; TODO: kept / skipped noisy from the terminal output |

## Evaluation

| Date | Commit | Utility | bench | tactics | match |
|---|---|---|---|---|---|

## Value network

| Date | Commit | Data | Label | Network | Settings | Baseline loss | Test loss | Test accuracy | Notes |
|---|---|---|---|---|---|---|---|---|---|
