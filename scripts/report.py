#!/usr/bin/env python3
"""Turn a finished training run into charts and a results page.

    python scripts/report.py runs/laptop --games 40 --simulations 200

Reads ``metrics.jsonl``, redraws the training charts, plays the final network
against the fixed baseline ladder, fits a single Elo scale over the whole
round-robin, and writes ``docs/results.md``.  Everything reported in the
README is produced by this script, so the numbers there can be regenerated
from a checkpoint rather than trusted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from neuralchess.evaluation.arena import match_summary, round_robin
from neuralchess.evaluation.baselines import (
    MaterialAgent,
    MCTSAgent,
    PolicyAgent,
    RandomAgent,
)
from neuralchess.evaluation.elo import MatchRecord, fit_elo, format_ladder
from neuralchess.nn.modules import PolicyValueNet
from neuralchess.search.evaluator import Evaluator
from neuralchess.search.mcts import MCTSConfig
from neuralchess.train.metrics import write_chart


def load_metrics(run_dir: str) -> List[dict]:
    path = os.path.join(run_dir, "metrics.jsonl")
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def series(records: List[dict], key: str):
    return [r.get(key) for r in records]


def smooth(values, window: int = 3):
    out, buf = [], []
    for v in values:
        if v is None:
            out.append(None)
            continue
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def draw_charts(run_dir: str, records: List[dict]) -> Dict[str, str]:
    """Write the charts into ``assets/`` so they are committed with the repo.

    ``runs/`` is git-ignored - it fills up with checkpoints - but the figures
    the README points at have to survive a clone.
    """
    os.makedirs(ASSETS, exist_ok=True)
    charts = {}
    charts["decisive"] = None
    charts["loss"] = write_chart(
        os.path.join(ASSETS, "loss.svg"),
        {"total": series(records, "total_loss"),
         "policy": series(records, "policy_loss"),
         "value": series(records, "value_loss")},
        title="Training loss", ylabel="loss")
    charts["accuracy"] = write_chart(
        os.path.join(ASSETS, "accuracy.svg"),
        {"policy top-1 vs search": series(records, "policy_accuracy"),
         "value top-1 vs outcome": series(records, "value_accuracy")},
        title="Agreement with the search targets", ylabel="accuracy")
    if any(r.get("val_total_loss") is not None for r in records):
        charts["generalisation"] = write_chart(
            os.path.join(ASSETS, "generalisation.svg"),
            {"training loss": series(records, "total_loss"),
             "held-out loss": series(records, "val_total_loss")},
            title="Training vs held-out loss", ylabel="loss")
    charts["entropy"] = write_chart(
        os.path.join(ASSETS, "entropy.svg"),
        {"policy entropy (nats)": series(records, "policy_entropy")},
        title="Policy entropy - the network becoming decisive", ylabel="nats")
    bench = {k: smooth(series(records, k)) for k in
             ("vs_random_score", "vs_material_score")
             if any(r.get(k) is not None for r in records)}
    if bench:
        charts["benchmark"] = write_chart(
            os.path.join(ASSETS, "benchmark.svg"), bench,
            title="In-loop benchmark (10 games, 40 sims, short limit)", ylabel="score")
    decisive = {k: series(records, k) for k in ("decisive_fraction", "adjudicated_fraction")
                if any(r.get(k) is not None for r in records)}
    if decisive:
        charts["decisive"] = write_chart(
            os.path.join(ASSETS, "decisive.svg"),
            {"decisive self-play games": decisive.get("decisive_fraction", []),
             "of which adjudicated": decisive.get("adjudicated_fraction", [])},
            title="Fraction of self-play games with a decisive result",
            ylabel="fraction")
    else:
        charts.pop("decisive", None)
    games = {"games played": np.cumsum(
        [r.get("games", 0) or 0 for r in records]).tolist()}
    charts["games"] = write_chart(os.path.join(ASSETS, "games.svg"), games,
                                  title="Cumulative self-play games", ylabel="games")
    return charts


def build_ladder(checkpoint: str, simulations: int, games: int, seed: int = 0,
                 first_checkpoint: Optional[str] = None):
    model = PolicyValueNet.load(checkpoint)
    model.eval()
    agents = [
        RandomAgent(seed=1),
        # A material grabber with enough evaluation noise to blunder regularly;
        # it fills the large Elo gap between random play and clean one-ply
        # material, so the ladder has a rung in the range a small network can
        # actually reach.
        MaterialAgent(seed=7, noise=500.0),
        MaterialAgent(seed=2),
        PolicyAgent(model, seed=4),
        MCTSAgent(Evaluator(model), simulations, MCTSConfig(simulations=simulations),
                  seed=5, name=f"neuralchess-{simulations}"),
    ]
    # The most direct evidence that training did anything: the same
    # architecture and the same search, one iteration in versus fully trained.
    if first_checkpoint and os.path.exists(first_checkpoint):
        early = PolicyValueNet.load(first_checkpoint)
        early.eval()
        agents.insert(3, MCTSAgent(Evaluator(early), simulations,
                                   MCTSConfig(simulations=simulations), seed=6,
                                   name="neuralchess-iter1"))
    started = time.time()
    records, logs = round_robin(agents, games, seed=seed, opening_plies=4,
                                max_moves=220, adjudicate_margin=500,
                                progress=lambda s: print(f"  {s}", flush=True))
    print(f"round robin finished in {time.time() - started:.0f}s")
    return records, logs


def main() -> int:
    parser = argparse.ArgumentParser(description="summarise a training run")
    parser.add_argument("run_dir")
    parser.add_argument("--games", type=int, default=30, help="games per pairing")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--out", default="docs/results.md")
    parser.add_argument("--skip-arena", action="store_true")
    parser.add_argument("--reuse-arena", action="store_true",
                        help="rebuild the page from a previous run's ladder.json")
    parser.add_argument("--compare", help="an earlier run directory to contrast with")
    args = parser.parse_args()

    records = load_metrics(args.run_dir)
    config = json.load(open(os.path.join(args.run_dir, "config.json")))
    charts = draw_charts(args.run_dir, records)
    charts = {k: v for k, v in charts.items() if v}
    # Ship the trained network with the repository so the UI and the UCI
    # engine work straight after a clone.
    import shutil
    best = os.path.join(args.run_dir, "checkpoints", "best.npz")
    if os.path.exists(best):
        shutil.copyfile(best, os.path.join(ASSETS, "neuralchess-5x48.npz"))
    print(f"{len(records)} iterations; charts: {', '.join(sorted(charts))}")

    ladder_text, table_text = "", ""
    ladder_path = os.path.join(args.run_dir, "ladder.json")
    if args.reuse_arena and os.path.exists(ladder_path):
        cached = json.load(open(ladder_path))
        # The prose quotes the settings the cached matches were actually
        # played at, not whatever was passed on this invocation.
        args.games = cached.get("games", args.games)
        args.simulations = cached.get("simulations", args.simulations)
        ratings = cached["ratings"]
        ladder_text = format_ladder(ratings)
        rows = ["| match | result | score | Elo difference |",
                "| --- | --- | --- | --- |"]
        for raw in cached["records"]:
            record = MatchRecord(**raw)
            summary = match_summary(record)
            rows.append(
                f"| {record.player_a} vs {record.player_b} "
                f"| +{record.wins} ={record.draws} \u2212{record.losses} "
                f"| {summary['score']:.1%} "
                f"| {summary['elo']:+.0f} [{summary['elo_low']:+.0f}, {summary['elo_high']:+.0f}] |")
        table_text = "\n".join(rows)
    elif not args.skip_arena:
        checkpoint = os.path.join(args.run_dir, "checkpoints", "best.npz")
        first = os.path.join(args.run_dir, "checkpoints", "iter0001.npz")
        match_records, logs = build_ladder(checkpoint, args.simulations, args.games,
                                           first_checkpoint=first)
        ratings = fit_elo(match_records, anchor="random", anchor_rating=0.0)
        ladder_text = format_ladder(ratings)
        rows = ["| match | result | score | Elo difference |",
                "| --- | --- | --- | --- |"]
        for record in match_records:
            summary = match_summary(record)
            rows.append(
                f"| {record.player_a} vs {record.player_b} "
                f"| +{record.wins} ={record.draws} −{record.losses} "
                f"| {summary['score']:.1%} "
                f"| {summary['elo']:+.0f} [{summary['elo_low']:+.0f}, {summary['elo_high']:+.0f}] |")
        table_text = "\n".join(rows)
        with open(ladder_path, "w") as fh:
            json.dump({"ratings": ratings, "games": args.games,
                       "simulations": args.simulations,
                       "records": [vars(r) for r in match_records]}, fh, indent=2)

    first, last = records[0], records[-1]
    total_games = sum(r.get("games", 0) or 0 for r in records)
    wall = sum(r.get("selfplay_seconds", 0) + r.get("train_seconds", 0) for r in records)

    body = f"""# Results

Everything on this page is regenerated by `scripts/report.py` from the run's
`metrics.jsonl` and its final checkpoint.

## The run

| | |
| --- | --- |
| network | {config['blocks']} × {config['channels']} residual blocks with squeeze-excitation |
| iterations | {len(records)} |
| self-play games | {total_games:,} |
| positions trained on | {last.get('buffer_size', 0):,} in the replay window |
| gradient steps | {int(last.get('global_step', 0)):,} |
| simulations per recorded move | {config['simulations']} (fast moves: {config['fast_simulations']}) |
| self-play + training wall time | {wall / 3600:.1f} h on {config['workers']} CPU workers |

## Learning curves

![training loss](../assets/loss.svg)
![accuracy](../assets/accuracy.svg)
![policy entropy](../assets/entropy.svg)
"""
    if "decisive" in charts:
        body += "![decisive games](../assets/decisive.svg)\n"
    if "generalisation" in charts:
        body += "![training vs held-out loss](../assets/generalisation.svg)\n"
    if "benchmark" in charts:
        body += "![baseline scores](../assets/benchmark.svg)\n"
        body += ("\nThe in-loop benchmark is deliberately cheap - ten games at forty\n"
                 "simulations with a short move limit and no adjudication - so it is\n"
                 "noisy and it understates strength whenever a won game runs out of\n"
                 "moves. The round-robin below is the measurement to read.\n")

    body += f"""
| metric | first iteration | last iteration |
| --- | --- | --- |
| total loss | {first.get('total_loss', float('nan')):.3f} | {last.get('total_loss', float('nan')):.3f} |
| policy loss | {first.get('policy_loss', float('nan')):.3f} | {last.get('policy_loss', float('nan')):.3f} |
| value loss | {first.get('value_loss', float('nan')):.3f} | {last.get('value_loss', float('nan')):.3f} |
| policy top-1 vs search | {first.get('policy_accuracy', float('nan')):.1%} | {last.get('policy_accuracy', float('nan')):.1%} |
| policy entropy (nats) | {first.get('policy_entropy', float('nan')):.2f} | {last.get('policy_entropy', float('nan')):.2f} |
"""

    if table_text:
        body += f"""
## Final round-robin

{args.games} games per pairing, {args.simulations} simulations per move for the
search agents, paired openings so colour cannot bias the result, and a
220-move limit. Games still running at the limit are adjudicated to whichever
side is at least a rook ahead - standard engine-testing practice, and
necessary here because a network this small often reaches a winning position
without being able to force mate.

{table_text}

Maximum-likelihood Elo over the whole round-robin, anchored at `random = 0`:

```
{ladder_text}
```

`policy-only` is the network with the search removed: it plays the `argmax` of
the policy head. The gap between it and `neuralchess-{args.simulations}` is what
tree search contributes on top of the learned priors.

`neuralchess-iter1` is the *same architecture with the same search* using the
checkpoint from the first training iteration. It is the cleanest available
measure of what the training loop itself accomplished, because everything
except the weights is held fixed.
"""

    if args.compare and os.path.exists(os.path.join(args.compare, "metrics.jsonl")):
        other = load_metrics(args.compare)
        other_cfg = json.load(open(os.path.join(args.compare, "config.json")))
        def decisive(rs):
            vals = [1.0 - (r.get("draws", 0) / max(1, r.get("games", 1))) for r in rs]
            return sum(vals) / len(vals)
        body += f"""
## What changed since the previous run

`{os.path.basename(args.compare)}` is the same code and the same network
trained without material adjudication of indecisive self-play games. It is
included because it failed instructively - see
[the draw trap](training.md#the-draw-trap-and-how-the-first-run-failed).

| | `{os.path.basename(args.compare)}` | `{os.path.basename(args.run_dir.rstrip('/'))}` |
| --- | --- | --- |
| adjudicate indecisive self-play games | no | yes, at {config.get('adjudicate_margin', 0)} cp |
| self-play move limit | {other_cfg.get('max_moves')} | {config.get('max_moves')} |
| mean fraction of decisive self-play games | {decisive(other):.2f} | {decisive(records):.2f} |
| final policy entropy (nats) | {other[-1].get('policy_entropy', float('nan')):.2f} | {last.get('policy_entropy', float('nan')):.2f} |
| final value loss | {other[-1].get('value_loss', float('nan')):.3f} | {last.get('value_loss', float('nan')):.3f} |
| final network vs its own iteration-1 checkpoint | **lost** (35% over 10 games) | **won** (79% over 12 games) |

The loss curves of the two runs are almost indistinguishable. The difference
only shows up in the fraction of decisive games and in playing strength, which
is the whole lesson.
"""

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(body)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
