# Report notes

Words, not numbers: why we chose something, which lecture it comes from, what surprised us, what didn't work.
The numbers are in results.md.

## Phase 0 - Baseline (20/11)

- We started from last year's engine instead of from scratch: the minimax + alpha-beta structure (MAX/MIN branches, L8) was correct, it lacked the AI part. The baseline shows where we start: 27/50 tactics, score 10% vs Stockfish skill 0.
- Symmetry test (mirrored positions give the opposite evaluation and the mirrored move) passed, so the difference between white (0/10) and black (2/10) in the baseline match was chance, not a bug in the MAX/MIN branches.

## Phase 3 - Value network data (28/11)

- Data: Lichess evaluation database (Stockfish evaluations of positions played on Lichess) instead of labelling positions ourselves with Stockfish. Why: millions of evaluations at high depth are already there, labelling ourselves would take days of computer time.
- Kasparov.pgn (game results as labels) is paused. The first test (27/11) overfitted from epoch 1: with ~1 700 training games the network recognises the games instead of learning chess, and the validation loss never beat predicting the mean. Game results are noisy labels (an early position of a won game isn't winning); DeepChess needed millions of positions for this. Kept for a noisy vs. clean label comparison later (L4).
- Encoding from the side to move (board mirrored when black is to move, NNUE convention): the network learns "good for the side to move" once, instead of separately for white and for black. The engine flips the value when black is to move, because MinimaxAgent scores from white's view.
- Sample: every 100th line of the whole file instead of the first lines, so the sample is spread over the whole database.
- Quiet positions only (not in check, best move is no capture): in the middle of a capture sequence the score swings, so it's a bad label. Same reason as quiescence search (L8, horizon effect). This removed ~27% of the positions.
- The shards store raw centipawns, the label is made in dataset.py: linear vs. sigmoid can be compared without rebuilding the data. Default label sigmoid(cp / 400): a win probability (L9) between 0 and 1 that fits the cross-entropy loss (logistic regression, L4).
- Split 98/1/1 per position (fixed via a hash of the FEN) instead of by game: in the Lichess database every position appears only once, from millions of different games, so there are hardly any near-copies between train and test. 1% is still ~29 000 positions, enough for a reliable number.
- The best move (first move of Stockfish's variation) is stored too, for the policy network (Phase 4), so no second pass over the 20 GB file is needed.
- Surprise: Lichess writes castling as "king takes rook" (e1h1), python-chess generates e1g1. Without converting, the policy network would learn other moves than the engine asks about.
- 11.7% of the positions are mates: label exactly 0 or 1 even with the sigmoid.

## Phase 3b - Value network v1 (28/11)

- Loss: binary cross-entropy (BCEWithLogitsLoss) instead of CrossEntropyLoss: one output (the score for the side to move) with a soft target between 0 and 1 = logistic regression (L4). CrossEntropyLoss comes back for the W/D/L head (3 classes) and the policy network (4096 move classes), like MNIST in Lab 2.
- The loss can't go to 0 with soft labels: a perfect prediction still has a loss of 0.563 on the test set (entropy of the labels). Comparing with the baseline (always predict the mean, 0.6923) and this floor gives a clearer picture than the loss alone: v1 removes 69% of the reducible loss.
- Accuracy is only measured on clear positions (|cp| > ~80, 54% of the test set): in a balanced position there's no right side of 0.5. 89.5% on the test set.
- CPU is enough: the network is small (~100k weights), so a GPU would only speed up half of the work; loading the data (unpacking the bitboards) takes about as long as the training math. One epoch on 2.87M positions takes ~1.2 min on a laptop CPU.
- Train and validation loss stay close (0.6013 vs 0.6031 after 5 epochs): no overfitting with 2.87M positions and ~100k weights, unlike the Kasparov test (1 700 games).
