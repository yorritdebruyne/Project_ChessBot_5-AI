import chess.pgn
import numpy as np

from fiveaichess.utilities.encoding import encode


"""
build_dataset
-------------
Builds the value-network dataset from data/Kasparov.pgn (runs once, offline).
Label = game result from white's view (1 / 0.5 / 0), same direction as CombinedUtility
(high is good for white).
Split by game into train / validation / test: positions from one game look alike, so a
split per position would put near-copies in train and test.

Run from the project root: python -m nn.build_dataset
"""

## PARAMETERS

pgn_path = "data/Kasparov.pgn"
out_path = "data/value_kasparov.npz"

# Opening positions are almost the same in every game but end in every result -> label noise
skip_plies = 8

split = (0.8, 0.1, 0.1)  # train / validation / test
random_seed = 1


def result_to_label(result: str):
    # '1-0' -> 1.0, '1/2-1/2' -> 0.5, '0-1' -> 0.0, anything else ('*') -> None (skip the game)
    # TODO

    results = {}

def positions_from_game(game: chess.pgn.Game) -> list:
    positions = []
    board = game.board()

    # TODO: Loop over game.mainline_moves(), push each move on the board
    #       and append encode(board) once more than skip_plies plies are played

    return positions


def load_games() -> list:
    # All usable games as (positions, label)
    games = []

    with open(pgn_path, encoding="utf-8", errors="replace") as pgn:
        # TODO: Read games with chess.pgn.read_game(pgn) until it returns None,
        #       label from game.headers["Result"], skip games without a label
        pass

    return games


def to_arrays(games: list):
    # Stack the positions of a list of games into data (n, 768) uint8 and targets (n,) float32,
    # every position gets the label of its game
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    games = load_games()

    # Shuffle the games (not the positions) and cut them in train / validation / test
    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(games))
    n_train = int(split[0] * len(games))
    n_valid = int(split[1] * len(games))
    train_games = [games[i] for i in order[:n_train]]
    valid_games = [games[i] for i in order[n_train:n_train + n_valid]]
    test_games = [games[i] for i in order[n_train + n_valid:]]

    train_data, train_targets = to_arrays(train_games)
    valid_data, valid_targets = to_arrays(valid_games)
    test_data, test_targets = to_arrays(test_games)

    print(f"games: {len(games)}")
    print(f"positions: train {len(train_targets)}, valid {len(valid_targets)}, test {len(test_targets)}")

    # Label distribution: if one result dominates, the network can score well by only predicting that
    for value in (1.0, 0.5, 0.0):
        print(f"label {value}: {100 * np.mean(train_targets == value):.1f}%")
    print(f"mean label (train): {train_targets.mean():.3f}")

    np.savez_compressed(out_path,
                        train_data=train_data, train_targets=train_targets,
                        valid_data=valid_data, valid_targets=valid_targets,
                        test_data=test_data, test_targets=test_targets)
    print("saved", out_path)