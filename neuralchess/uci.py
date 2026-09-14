"""UCI protocol adapter.

Implements enough of the Universal Chess Interface for the engine to be
loaded by Arena, Cute Chess, Banksia or a Lichess bot bridge.  Search
progress is reported as ``info`` lines with a pseudo-centipawn score derived
from the value head, so GUIs display a meaningful evaluation bar.

Run it with ``python -m neuralchess.uci --checkpoint path/to/net.npz``.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

import numpy as np

from .chess.board import START_FEN, Board
from .chess.move import move_uci
from .encoding.planes import PositionHistory
from .nn.modules import NetConfig, PolicyValueNet
from .search.evaluator import Evaluator
from .search.mcts import MCTS, MCTSConfig

ENGINE_NAME = "neuralchess"
ENGINE_AUTHOR = "Tyler Seils"


def value_to_centipawns(value: float) -> int:
    """Map a value in ``(-1, 1)`` onto a conventional centipawn scale."""
    v = max(-0.9999, min(0.9999, value))
    return int(round(111.7 * np.tan(1.5620688 * v)))


class UCIEngine:
    def __init__(self, checkpoint: Optional[str] = None, simulations: int = 400):
        self.options = {
            "Simulations": simulations,
            "CPuct": 1.25,
            "Temperature": 0.0,
            "BatchSize": 16,
            "CacheSize": 200000,
        }
        self.model = self._load(checkpoint)
        self.evaluator = Evaluator(self.model, cache_size=int(self.options["CacheSize"]))
        self.board = Board()
        self.history = PositionHistory(self.board)
        self.rng = np.random.default_rng(0)

    @staticmethod
    def _load(checkpoint: Optional[str]) -> PolicyValueNet:
        if checkpoint:
            return PolicyValueNet.load(checkpoint)
        # A random network still plays legal chess; useful for protocol tests.
        return PolicyValueNet(NetConfig(channels=32, blocks=4))

    # -- protocol -------------------------------------------------------
    def send(self, line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def handle(self, line: str) -> bool:
        """Process one command; returns False when the engine should exit."""
        parts = line.strip().split()
        if not parts:
            return True
        cmd, args = parts[0], parts[1:]

        if cmd == "uci":
            self.send(f"id name {ENGINE_NAME}")
            self.send(f"id author {ENGINE_AUTHOR}")
            self.send("option name Simulations type spin default 400 min 1 max 1000000")
            self.send("option name CPuct type string default 1.25")
            self.send("option name Temperature type string default 0.0")
            self.send("option name BatchSize type spin default 16 min 1 max 512")
            self.send("uciok")
        elif cmd == "isready":
            self.send("readyok")
        elif cmd == "setoption":
            self._set_option(args)
        elif cmd == "ucinewgame":
            self.board = Board()
            self.history = PositionHistory(self.board)
            self.evaluator.clear_cache()
        elif cmd == "position":
            self._set_position(args)
        elif cmd == "go":
            self._go(args)
        elif cmd == "stop":
            pass
        elif cmd in ("quit", "exit"):
            return False
        elif cmd == "d":
            self.send(str(self.board))
            self.send(f"fen: {self.board.fen()}")
        return True

    def _set_option(self, args: List[str]) -> None:
        if "name" not in args:
            return
        name_idx = args.index("name") + 1
        value_idx = args.index("value") + 1 if "value" in args else len(args)
        name = " ".join(args[name_idx:value_idx - 1]) if "value" in args else " ".join(args[name_idx:])
        value = " ".join(args[value_idx:]) if "value" in args else ""
        for key in self.options:
            if key.lower() == name.lower().strip():
                try:
                    self.options[key] = type(self.options[key])(float(value))
                except (TypeError, ValueError):
                    pass

    def _set_position(self, args: List[str]) -> None:
        if not args:
            return
        if args[0] == "startpos":
            fen = START_FEN
            rest = args[1:]
        elif args[0] == "fen":
            fen_parts = []
            rest = []
            i = 1
            while i < len(args) and args[i] != "moves":
                fen_parts.append(args[i])
                i += 1
            rest = args[i:]
            fen = " ".join(fen_parts)
        else:
            return
        self.board = Board(fen)
        self.history = PositionHistory(self.board)
        if rest and rest[0] == "moves":
            for uci in rest[1:]:
                self.board.push_uci(uci)
                self.history.push(self.board)

    def _go(self, args: List[str]) -> None:
        simulations = int(self.options["Simulations"])
        movetime = None
        for i, token in enumerate(args):
            if token == "nodes" and i + 1 < len(args):
                simulations = int(args[i + 1])
            elif token == "movetime" and i + 1 < len(args):
                movetime = int(args[i + 1]) / 1000.0
            elif token == "depth" and i + 1 < len(args):
                simulations = max(simulations, 64 * int(args[i + 1]))

        config = MCTSConfig(simulations=simulations,
                            c_puct_init=float(self.options["CPuct"]),
                            batch_size=int(self.options["BatchSize"]))
        search = MCTS(self.evaluator, config, rng=self.rng)

        start = time.time()
        if movetime is not None:
            root = None
            step = max(16, config.batch_size * 2)
            done = 0
            while time.time() - start < movetime * 0.9:
                done += step
                root = search.run(self.board, self.history.frames(), done, root=root)
                if not root.moves or root.terminal_value is not None:
                    break
        else:
            root = search.run(self.board, self.history.frames(), simulations)

        elapsed = max(time.time() - start, 1e-6)
        if not root.moves:
            self.send("bestmove 0000")
            return
        best = root.best_move()
        pv = root.principal_variation(12)
        score = value_to_centipawns(root.value)
        self.send(
            f"info depth {len(pv)} nodes {int(root.visits)} "
            f"nps {int(root.visits / elapsed)} time {int(elapsed * 1000)} "
            f"score cp {score} pv {' '.join(move_uci(m) for m in pv)}"
        )
        self.send(f"bestmove {move_uci(best)}")

    def loop(self) -> None:
        for line in sys.stdin:
            if not self.handle(line):
                break


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="neuralchess UCI engine")
    parser.add_argument("--checkpoint", help="path to a .npz network checkpoint")
    parser.add_argument("--simulations", type=int, default=400)
    args = parser.parse_args(argv)
    UCIEngine(args.checkpoint, args.simulations).loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
