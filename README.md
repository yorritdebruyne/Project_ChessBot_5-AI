# 5AI Labs: Chess Project

In this lab you will build an artificially intelligent chess agent.

This repository provides a light-weight framework built on top of `python-chess`, which implements the core chess logic. The framework hides the quirky details of this library and implements the Universal Chess Interface (UCI), an open communication protocol that enables chess engines to communicate with user interfaces and other software.

Follow the instructions below to install, run, and extend the framework.

## Gettting started

Make sure your virtual environment is active if you use one.

To ensure the `fiveaichess` package from your project folder can be discovered from anywhere on your system, install it in **editable mode** by running the following command from your project root folder:

```bash
pip install -e .
```

This will automatically install the `python-chess` package. You can also install it manually:

```bash
pip install python-chess
```

Or use `requirements.txt` if you have more dependencies in your project:

```bash
pip install -r requirements.txt
```

Once installed, you can run your chess engine using `fiveaichess.bat` (Windows) and `fiveaichess.sh` (Linux). These scripts wrap the following command:

```bash
python -m fiveaichess.engines.example_engine
```

> **Note**: On Linux, make sure your script is executable with `chmod +x fiveaichess.sh`

Verify the installation by running the engine and typing the command `uci` in the console.

The engine should respond with:

```
id name 5AIChessAgent
id author UAntwerp
uciok
```

## Documentation

The framework includes base classes and example implementations for **Agents**, **Utilities**, and **Engines**.

> **Note:** The provided example implementation is intentionally simple and naïve. Your task in this lab is to improve it with the techniques discussed in class and by looking into state-of-the-art solution to this problem.

### Agents

An Agent encapsulates the decision-making logic used to search the game tree and select moves during gameplay. This is the appropriate location to implement search algorithms such as minimax or alpha-beta pruning. During search, the agent queries a Utility to obtain an evaluation score for each game state, and uses these evaluations to determine which move to play according to the selected search strategy.

### Utilities

A Utility is responsible for assigning heuristic values to board positions. A utility may implement simple heuristics, such as material count, or more advanced evaluation functions that consider positional factors including piece mobility, king safety, development, and phase of the game. Modifying or replacing the utility directly influences the playing strength and style of the agent.

### Engines

An Engine wraps an Agent and enables interaction with UCI-compatible graphical user interfaces or other software. The `uci_engine.py` module provides a UCI protocol implementation and _generally does not require modification_. The `example_engine.py` module demonstrates how to construct a UCIEngine by combining custom Agent and Utility classes.

## Examples

The `scripts` folder contains example scripts that show how to use the framework.


| Script | Purpose |
|--------|---------|
| `play_self.py` | Lets two agents play a game against each other |
| `play_engine.py` | Lets your agent play against a UCI-compatible chess engine (e.g., Stockfish) |
| `replay_pgn.py` | Reads PGN files and replays recorded chess games |


> **Note:** `play_engine.py` uses the `UCI_ENGINE_PATH` environment variable to locate the engine.

## Stockfish

[Stockfish](https://stockfishchess.org) is a free and open-source chess engine and is among the strongest chess engines in the world today.

Download it from the official website, unzip the files and set `UCI_ENGINE_PATH` to the binary:

**Windows**

```powershell
set UCI_ENGINE_PATH C:\path\to\stockfish\binary
```

**Linux**

```bash
export UCI_ENGINE_PATH=/path/to/stockfish/binary
```

## Chess GUI

You can play against your own egine (or see it in action against other engines) with any UCI-compatible GUI.

We recommend [En Croissant](https://encroissant.org), a free and open-source chess GUI.

Follow the [Getting Started](https://encroissant.org/docs/getting-started) guide to install the application. See [Configure Engines](https://encroissant.org/docs/guides/configure-engines) for more information on how to set up chess engines.

### Adding your own engine

En Croissant's engine selection dialog expects an executable file. It does not allow to register your engine using the provided `fiveaichess.bat` (Windows) or `fiveaichess.sh` (Linux) scripts. Therefore you should manually edit `engines.json` to set up your engine:

- Run En Croissant at least once, then close it before manually modifying the configuration
- Open `engines.json` at the following location (unless you have configured a custom engines directory):

**Windows**

```text
%APPDATA%\org.encroissant.app\engines\engines.json
```

**Linux**

```text
~/.local/share/org.encroissant.app/engines/engines.json
```

- Add an engine entry to the JSON array, keeping any existing entries.

**Windows**

```json
{
  "type": "local",
  "id": "8f9cb50e-1cb6-4e57-a73b-a48a75f214d9",
  "name": "5AIChessAgent",
  "version": "1.0",
  "path": "C:\\path\\to\\chess-project\\fiveaichess.bat",
  "loaded": true,
  "settings": []
}
```

**Linux**

```json
{
  "type": "local",
  "id": "8f9cb50e-1cb6-4e57-a73b-a48a75f214d9",
  "name": "5AIChessAgent",
  "version": "1.0",
  "path": "/home/user/chess-project/fiveaichess.sh",
  "loaded": true,
  "settings": []
}
```

- Replace `path` with the absolute path to your script and use a unique `id` (UUID v4) for each engine. 
- Save the file, restart En Croissant, and select your engine to play or analyze a game

> **Note**: On Windows, backslashes must be escaped (`\\`) in JSON.

> **Note**: The GUI does not have an option to set the Python interpreter. The app will run `fiveaichess.bat` (Windows) or `fiveaichess.sh` (Linux) as provided. If you use a virtual environment, adapt the script to use the Python interpreter from your virtual environment (e.g. `.venv/bin/python`) instead of the system-wide Python interpreter (e.g. `python`). This ensures the dependencies installled in your environment can be imported.

**Have fun building your chess AI!**