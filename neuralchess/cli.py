"""Command line interface.

``python -m neuralchess <command>`` is the single entry point for everything:
training, playing in a terminal or a browser, running the arena, exporting
games, and the move-generator correctness suite.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import fields
from typing import List, Optional

import numpy as np

from .chess.board import START_FEN, Board, perft, perft_divide
from .chess.move import move_uci
from .chess.pgn import to_pgn
from .encoding.planes import PositionHistory
from .evaluation.arena import match, match_summary, round_robin
from .evaluation.baselines import (
    AlphaBetaAgent,
    MaterialAgent,
    MCTSAgent,
    PolicyAgent,
    RandomAgent,
)
from .evaluation.elo import fit_elo, format_ladder
from .nn.modules import NetConfig, PolicyValueNet
from .search.evaluator import Evaluator
from .search.mcts import MCTSConfig, describe_root
from .train.loop import TrainConfig, Trainer

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_model(path: Optional[str], channels: int = 32, blocks: int = 4) -> PolicyValueNet:
    if path:
        model = PolicyValueNet.load(path)
        model.eval()
        return model
    model = PolicyValueNet(NetConfig(channels=channels, blocks=blocks))
    model.eval()
    return model


def _build_agent(spec: str, simulations: int, seed: int = 0):
    """Turn a string like ``mcts:runs/x/checkpoints/best.npz`` into an agent."""
    if spec == "random":
        return RandomAgent(seed=seed)
    if spec.startswith("material"):
        noise = float(spec.split(":")[1]) if ":" in spec else 0.0
        return MaterialAgent(seed=seed, noise=noise)
    if spec.startswith("alphabeta"):
        depth = int(spec.split(":")[1]) if ":" in spec else 2
        return AlphaBetaAgent(depth=depth, seed=seed)
    if spec.startswith("policy"):
        path = spec.split(":", 1)[1] if ":" in spec else None
        return PolicyAgent(_load_model(path), seed=seed)
    if spec.startswith("mcts"):
        path = spec.split(":", 1)[1] if ":" in spec else None
        model = _load_model(path)
        return MCTSAgent(Evaluator(model), simulations,
                         MCTSConfig(simulations=simulations), seed=seed,
                         name=f"mcts-{simulations}")
    raise SystemExit(f"unknown agent spec: {spec!r}")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_train(args) -> int:
    overrides = {}
    if args.config:
        with open(args.config) as fh:
            overrides.update(json.load(fh))
    known = {f.name for f in fields(TrainConfig)}
    for key, value in vars(args).items():
        if key in known and value is not None:
            overrides[key] = value
    config = TrainConfig(**{k: v for k, v in overrides.items() if k in known})
    trainer = Trainer(config)
    print(f"run directory : {trainer.run_dir}")
    print(f"network       : {trainer.model.describe()}")
    print(f"iterations    : {config.iterations} x "
          f"({config.games_per_iteration} games, {config.steps_per_iteration} steps)")
    print("-" * 78)
    start = time.time()

    def report(record):
        parts = [f"iter {int(record['iteration']):3d}"]
        for key, label, fmt in (
            ("games", "games", "{:.0f}"), ("buffer_size", "buffer", "{:.0f}"),
            ("total_loss", "loss", "{:.3f}"), ("policy_loss", "pol", "{:.3f}"),
            ("value_loss", "val", "{:.3f}"), ("policy_accuracy", "acc", "{:.3f}"),
            ("gate_score", "gate", "{:.2f}"), ("vs_random_score", "vs.rand", "{:.2f}"),
            ("vs_material_score", "vs.mat", "{:.2f}"),
        ):
            if record.get(key) is not None:
                parts.append(f"{label} " + fmt.format(record[key]))
        parts.append(f"{time.time() - start:6.0f}s")
        print("  ".join(parts), flush=True)

    trainer.run(on_iteration=report)
    print("-" * 78)
    print(f"done in {time.time() - start:.0f}s; best network at "
          f"{os.path.join(trainer.run_dir, 'checkpoints', 'best.npz')}")
    return 0


def cmd_selfplay(args) -> int:
    from .selfplay.game import SelfPlayConfig, play_game

    model = _load_model(args.checkpoint)
    evaluator = Evaluator(model)
    config = SelfPlayConfig(simulations=args.simulations,
                            fast_simulations=max(8, args.simulations // 4),
                            max_moves=args.max_moves)
    games = []
    for i in range(args.games):
        start = time.time()
        game = play_game(evaluator, config, rng=np.random.default_rng(args.seed + i))
        print(f"game {i + 1}: {game.result} in {game.plies} plies "
              f"({game.terminal_reason}, {len(game.samples)} samples, "
              f"{time.time() - start:.1f}s)")
        games.append(game)
    if args.pgn:
        with open(args.pgn, "w") as fh:
            for game in games:
                fh.write(to_pgn(game.moves, game.result))
                fh.write("\n")
        print(f"wrote {args.pgn}")
    return 0


def cmd_arena(args) -> int:
    agents = [_build_agent(spec, args.simulations, seed=i)
              for i, spec in enumerate(args.agents)]
    if len(agents) < 2:
        raise SystemExit("need at least two agents")
    if len(agents) == 2:
        record, logs = match(agents[0], agents[1], args.games, seed=args.seed,
                             opening_plies=args.opening_plies)
        print(record.summary())
        print(json.dumps(match_summary(record), indent=2))
        records = [record]
    else:
        records, logs = round_robin(agents, args.games, seed=args.seed,
                                    opening_plies=args.opening_plies,
                                    progress=lambda s: print(f"  playing {s}", flush=True))
        for record in records:
            print(record.summary())
    ratings = fit_elo(records, anchor=args.anchor)
    print()
    print(format_ladder(ratings))
    if args.pgn:
        with open(args.pgn, "w") as fh:
            for log in logs:
                board = Board(log.start_fen)
                parsed = []
                for uci in log.moves:
                    move = board.parse_uci(uci)
                    parsed.append(move)
                    board.push(move)
                fh.write(to_pgn(parsed, log.result,
                                {"White": log.white, "Black": log.black,
                                 "Event": "neuralchess arena",
                                 "Termination": log.reason},
                                start_fen=log.start_fen))
                fh.write("\n")
        print(f"wrote {args.pgn}")
    return 0


def cmd_play(args) -> int:
    """Play in the terminal."""
    model = _load_model(args.checkpoint)
    agent = MCTSAgent(Evaluator(model), args.simulations,
                      MCTSConfig(simulations=args.simulations))
    board = Board(args.fen)
    history = PositionHistory(board)
    human_white = args.color != "black"
    while board.terminal_value() is None:
        print()
        print(board)
        print(board.fen())
        if (board.turn == 0) == human_white:
            try:
                text = input("your move (uci, or 'quit'): ").strip()
            except EOFError:
                break
            if text in ("quit", "exit"):
                break
            try:
                move = board.parse_uci(text)
            except (ValueError, KeyError):
                print("  illegal or malformed move")
                continue
        else:
            start = time.time()
            move = agent.select_move(board, history)
            print(f"engine plays {move_uci(move)} "
                  f"({time.time() - start:.1f}s)  {describe_root(agent.last_root, 3)}")
        print(f"  {board.san(move)}")
        board.push(move)
        history.push(board)
    print()
    print(board)
    print("result:", board.outcome())
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    serve(args.checkpoint, args.port, args.simulations, args.open)
    return 0


def cmd_uci(args) -> int:
    from .uci import UCIEngine

    UCIEngine(args.checkpoint, args.simulations).loop()
    return 0


def cmd_perft(args) -> int:
    board = Board(args.fen)
    print(board)
    for depth in range(1, args.depth + 1):
        start = time.time()
        nodes = perft(board, depth)
        elapsed = time.time() - start
        print(f"perft({depth}) = {nodes:>12,}   {elapsed:7.2f}s   "
              f"{nodes / max(elapsed, 1e-9):>10,.0f} nps")
    if args.divide:
        print()
        for move, count in sorted(perft_divide(board, args.depth).items()):
            print(f"  {move}: {count}")
    return 0


def cmd_bench(args) -> int:
    """Throughput of the network and the search on this machine."""
    model = _load_model(args.checkpoint, args.channels, args.blocks)
    print(model.describe())
    planes = np.random.randn(args.batch, 119, 8, 8).astype(np.float32)
    for _ in range(2):
        model.predict(planes[:2])
    start = time.time()
    for _ in range(args.repeats):
        model.predict(planes)
    elapsed = time.time() - start
    total = args.repeats * args.batch
    print(f"inference : {total / elapsed:8.0f} positions/s "
          f"(batch {args.batch}, {elapsed:.2f}s)")

    evaluator = Evaluator(model)
    agent = MCTSAgent(evaluator, args.simulations, MCTSConfig(simulations=args.simulations))
    board = Board()
    history = PositionHistory(board)
    start = time.time()
    agent.select_move(board, history)
    elapsed = time.time() - start
    print(f"search    : {args.simulations / elapsed:8.0f} simulations/s")
    print(f"cache     : {evaluator.stats()}")
    return 0


def cmd_render(args) -> int:
    from .render import write_board_svg

    board = Board(args.fen)
    last = None
    for uci in args.moves:
        last = board.push_uci(uci)
    write_board_svg(args.output, board, size=args.size, flipped=args.flip,
                    last_move=last, caption=args.caption)
    print(f"wrote {args.output}")
    return 0


def cmd_info(args) -> int:
    model = _load_model(args.checkpoint, args.channels, args.blocks)
    print(model.describe())
    print(f"config: {model.config.to_dict()}")
    total = 0
    for name, param in model.named_parameters():
        total += param.data.size
        if args.verbose:
            print(f"  {name:34s} {str(param.data.shape):22s} {param.data.size:>9,}")
    print(f"total parameters: {total:,}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neuralchess",
                                     description="AlphaZero-style chess engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="run the self-play training loop")
    p.add_argument("--config", help="JSON file with TrainConfig overrides")
    p.add_argument("--run-name", dest="run_name")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--iterations", type=int)
    p.add_argument("--games-per-iteration", dest="games_per_iteration", type=int)
    p.add_argument("--steps-per-iteration", dest="steps_per_iteration", type=int)
    p.add_argument("--simulations", type=int)
    p.add_argument("--channels", type=int)
    p.add_argument("--blocks", type=int)
    p.add_argument("--history-length", dest="history_length", type=int,
                   help="past positions encoded in the input (8 = AlphaZero, 2 = shallow)")
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"))
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("selfplay", help="generate self-play games")
    p.add_argument("--checkpoint")
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--simulations", type=int, default=128)
    p.add_argument("--max-moves", dest="max_moves", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pgn", help="write the games to this PGN file")
    p.set_defaults(func=cmd_selfplay)

    p = sub.add_parser("arena", help="play agents against each other")
    p.add_argument("agents", nargs="+",
                   help="agent specs: random | material | material:200 | "
                        "alphabeta:2 | policy:CKPT | mcts:CKPT")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--simulations", type=int, default=128)
    p.add_argument("--opening-plies", dest="opening_plies", type=int, default=2)
    p.add_argument("--anchor", default="random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pgn")
    p.set_defaults(func=cmd_arena)

    p = sub.add_parser("play", help="play against the engine in the terminal")
    p.add_argument("--checkpoint")
    p.add_argument("--simulations", type=int, default=300)
    p.add_argument("--color", choices=("white", "black"), default="white")
    p.add_argument("--fen", default=START_FEN)
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("serve", help="play against the engine in a browser")
    p.add_argument("--checkpoint")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--simulations", type=int, default=300)
    p.add_argument("--open", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("uci", help="run as a UCI engine on stdin/stdout")
    p.add_argument("--checkpoint")
    p.add_argument("--simulations", type=int, default=400)
    p.set_defaults(func=cmd_uci)

    p = sub.add_parser("perft", help="move generator correctness / speed")
    p.add_argument("--fen", default=START_FEN)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--divide", action="store_true")
    p.set_defaults(func=cmd_perft)

    p = sub.add_parser("bench", help="measure inference and search throughput")
    p.add_argument("--checkpoint")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--simulations", type=int, default=200)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("render", help="render a position as an SVG")
    p.add_argument("--fen", default=START_FEN)
    p.add_argument("--moves", nargs="*", default=[])
    p.add_argument("--output", "-o", default="board.svg")
    p.add_argument("--size", type=int, default=400)
    p.add_argument("--flip", action="store_true")
    p.add_argument("--caption", default="")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("info", help="describe a checkpoint")
    p.add_argument("--checkpoint")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
