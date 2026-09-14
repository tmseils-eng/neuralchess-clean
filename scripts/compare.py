#!/usr/bin/env python3
"""Put several checkpoints and the fixed baselines on one Elo scale.

    python scripts/compare.py --games 12 --simulations 100 \
        --net previous=assets/neuralchess-5x48.npz \
        --net current=runs/t2adamw/checkpoints/best.npz

Every network plays the same opponents with the same paired openings, so the
resulting ladder answers "is this checkpoint stronger than that one" directly
rather than by comparing two separately anchored numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralchess.evaluation.arena import match_summary, round_robin
from neuralchess.evaluation.baselines import (
    MaterialAgent,
    MCTSAgent,
    PolicyAgent,
    RandomAgent,
)
from neuralchess.evaluation.elo import fit_elo, format_ladder
from neuralchess.nn.modules import PolicyValueNet
from neuralchess.search.evaluator import Evaluator
from neuralchess.search.mcts import MCTSConfig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", action="append", default=[], metavar="NAME=PATH",
                    help="a checkpoint to place on the ladder; repeatable")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--simulations", type=int, default=100)
    ap.add_argument("--policy-of", dest="policy_of",
                    help="also enter the search-free policy head of this network")
    ap.add_argument("--out", default="assets/comparison.json")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    agents = [RandomAgent(seed=1), MaterialAgent(seed=7, noise=500.0),
              MaterialAgent(seed=2)]
    for i, spec in enumerate(args.net):
        name, path = spec.split("=", 1)
        model = PolicyValueNet.load(path)
        model.eval()
        print(f"{name}: {model.describe()}")
        agents.append(MCTSAgent(Evaluator(model), args.simulations,
                                MCTSConfig(simulations=args.simulations),
                                seed=20 + i, name=name))
    if args.policy_of:
        model = PolicyValueNet.load(args.policy_of)
        model.eval()
        agent = PolicyAgent(model, seed=9)
        agent.name = "policy-only"
        agents.append(agent)

    started = time.time()
    records, _ = round_robin(agents, args.games, seed=args.seed, opening_plies=4,
                             max_moves=220, adjudicate_margin=500,
                             progress=lambda s: print(f"  {s}", flush=True))
    ratings = fit_elo(records, anchor="random", anchor_rating=0.0)
    print(f"\nfinished in {time.time() - started:.0f}s\n")
    for record in records:
        print("  " + record.summary())
    print()
    print(format_ladder(ratings))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"ratings": ratings, "games": args.games,
                   "simulations": args.simulations,
                   "records": [vars(r) for r in records],
                   "summaries": [match_summary(r) for r in records]}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
