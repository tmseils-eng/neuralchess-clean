#!/usr/bin/env python3
"""Controlled ablation: how much does input history depth drive memorisation?

    python scripts/ablate_history.py --checkpoint assets/neuralchess-5x48.npz \
        --games 90 --steps 400 --depths 8 4 2 1

The hypothesis is that the 112 history planes are a memorisation channel: all
positions in a game share one outcome label, and eight steps of history is
close to a unique fingerprint of *which* game a position came from, so the
network can learn to recall the label instead of judging the position.

A second full training run cannot separate that effect from everything else
that changes between runs.  This script can, because every variant sees the
*same games*, the *same train/validation split*, the same number of gradient
steps and the same seed.  Only the depth of the encoded history differs.

Positions are stored as move lists and re-encoded per variant - encoding once
and slicing would leak the deeper representation into the shallow variants.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from neuralchess.chess.board import START_FEN, Board
from neuralchess.encoding.planes import PositionHistory, input_planes
from neuralchess.encoding.policy_map import POLICY_SIZE
from neuralchess.nn.modules import NetConfig, PolicyValueNet
from neuralchess.nn.optim import build_optimizer
from neuralchess.nn.tensor import no_grad
from neuralchess.search.evaluator import Evaluator
from neuralchess.search.mcts import MCTSConfig
from neuralchess.selfplay.game import SelfPlayConfig, play_game
from neuralchess.train.losses import LossWeights, compute_loss


def generate(checkpoint: str, games: int, simulations: int, seed: int) -> List:
    """Play games once; every variant is trained on this same set."""
    model = PolicyValueNet.load(checkpoint) if checkpoint else PolicyValueNet(
        NetConfig(channels=48, blocks=5))
    model.eval()
    evaluator = Evaluator(model)
    config = SelfPlayConfig(simulations=simulations, fast_simulations=simulations // 4,
                            full_search_probability=0.5, max_moves=110,
                            adjudicate_margin=150)
    out = []
    started = time.time()
    for i in range(games):
        evaluator.clear_cache()
        game = play_game(evaluator, config, MCTSConfig(simulations=simulations, batch_size=24),
                         rng=np.random.default_rng(seed + i))
        out.append(game)
        if (i + 1) % 10 == 0:
            done = time.time() - started
            print(f"  {i + 1}/{games} games  ({done:.0f}s, "
                  f"eta {done / (i + 1) * (games - i - 1):.0f}s)", flush=True)
    return out


def encode_games(games: Sequence, depth: int):
    """Re-encode every recorded position at ``depth`` history steps."""
    planes, policies, masks, values = [], [], [], []
    for game in games:
        by_ply = {s.ply: s for s in game.samples}
        if not by_ply:
            continue
        board = Board(START_FEN)
        history = PositionHistory(board, window=depth)
        for ply, move in enumerate(game.moves):
            sample = by_ply.get(ply)
            if sample is not None:
                planes.append(history.encode(depth))
                policy = np.zeros(POLICY_SIZE, dtype=np.float32)
                mask = np.zeros(POLICY_SIZE, dtype=bool)
                policy[sample.policy_indices] = sample.policy_probs
                mask[sample.policy_indices] = True
                policies.append(policy)
                masks.append(mask)
                values.append(sample.value)
            board.push(move)
            history.push(board)
    return (np.stack(planes), np.stack(policies), np.stack(masks),
            np.array(values, dtype=np.float32))


def train_variant(depth: int, train, val, steps: int, batch: int,
                  channels: int, blocks: int, optimizer: str,
                  weight_decay: float, lr: float, seed: int) -> Dict[str, float]:
    np.random.seed(seed)
    model = PolicyValueNet(NetConfig(channels=channels, blocks=blocks,
                                     history_length=depth))
    opt = build_optimizer(model.parameters(), optimizer, lr=lr,
                          weight_decay=weight_decay)
    weights = LossWeights()
    rng = np.random.default_rng(seed)
    n = train[0].shape[0]
    history = []

    for step in range(steps):
        idx = rng.integers(0, n, batch)
        loss, _ = compute_loss(model, train[0][idx], train[1][idx],
                               train[2][idx], train[3][idx], weights)
        opt.zero_grad()
        loss.backward()
        opt.clip_grad_norm(4.0)
        opt.step()
        if (step + 1) % max(1, steps // 8) == 0:
            history.append(evaluate(model, train, val, rng))
    # Copy, do not alias: ``history[-1]`` is an element of ``history``, so
    # assigning the list into it makes the structure self-referential and
    # json.dump fails with "circular reference".
    final = dict(history[-1]) if history else evaluate(model, train, val, rng)
    final["curve"] = history
    final["parameters"] = model.num_parameters()
    final["input_planes"] = input_planes(depth)
    return final


def evaluate(model, train, val, rng) -> Dict[str, float]:
    """Loss on a batch of training data and on the held-out games."""
    was_training = model.training
    model.eval()
    try:
        with no_grad():
            idx = rng.integers(0, train[0].shape[0], min(256, train[0].shape[0]))
            _, tr = compute_loss(model, train[0][idx], train[1][idx],
                                 train[2][idx], train[3][idx], LossWeights())
            m = min(256, val[0].shape[0])
            _, va = compute_loss(model, val[0][:m], val[1][:m],
                                 val[2][:m], val[3][:m], LossWeights())
    finally:
        model.train(was_training)
    return {
        "train_total": tr["total_loss"], "val_total": va["total_loss"],
        "train_value": tr["value_loss"], "val_value": va["value_loss"],
        "train_policy": tr["policy_loss"], "val_policy": va["policy_loss"],
        "train_value_acc": tr.get("value_accuracy", float("nan")),
        "val_value_acc": va.get("value_accuracy", float("nan")),
        "gap": va["total_loss"] - tr["total_loss"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint")
    ap.add_argument("--games", type=int, default=90)
    ap.add_argument("--simulations", type=int, default=64)
    ap.add_argument("--depths", type=int, nargs="+", default=[8, 4, 2, 1])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--optimizer", default="adam")
    ap.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-fraction", dest="val_fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="assets/ablation_history.json")
    args = ap.parse_args()

    print(f"generating {args.games} games at {args.simulations} simulations…", flush=True)
    games = generate(args.checkpoint, args.games, args.simulations, args.seed)

    # Split by GAME, never by position: positions from one game share an
    # outcome label and overlapping history, so a position split leaks.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(games))
    cut = int(len(games) * (1 - args.val_fraction))
    train_games = [games[i] for i in order[:cut]]
    val_games = [games[i] for i in order[cut:]]
    print(f"{len(train_games)} training games / {len(val_games)} held-out games")

    results = {}
    for depth in args.depths:
        train = encode_games(train_games, depth)
        val = encode_games(val_games, depth)
        print(f"\ndepth T={depth}: {input_planes(depth)} planes, "
              f"{train[0].shape[0]} train / {val[0].shape[0]} val positions", flush=True)
        started = time.time()
        res = train_variant(depth, train, val, args.steps, args.batch,
                            args.channels, args.blocks, args.optimizer,
                            args.weight_decay, args.lr, args.seed)
        res["seconds"] = time.time() - started
        results[str(depth)] = res
        print(f"  train {res['train_total']:.3f} | held out {res['val_total']:.3f} "
              f"| gap {res['gap']:+.3f} | value acc train {res['train_value_acc']:.2f} "
              f"held out {res['val_value_acc']:.2f}  ({res['seconds']:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"config": vars(args), "results": results}, fh, indent=2)

    print("\n" + "=" * 72)
    print(f"{'depth':>6} {'planes':>7} {'train':>8} {'held out':>9} {'gap':>8} "
          f"{'val acc':>8}")
    for depth in args.depths:
        r = results[str(depth)]
        print(f"{depth:>6} {r['input_planes']:>7} {r['train_total']:>8.3f} "
              f"{r['val_total']:>9.3f} {r['gap']:>+8.3f} {r['val_value_acc']:>8.2f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
