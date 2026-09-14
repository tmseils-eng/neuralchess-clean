"""The AlphaZero training loop.

One *iteration* is: generate self-play games with the current best network,
append them to a replay window, take a number of gradient steps on batches
drawn from that window, then play a gating match between the freshly trained
network and the current best.  The challenger is promoted only if it scores
above a threshold, which stops a bad training run from poisoning the data it
generates next - the single most common failure mode of a self-play loop.

Self-play is embarrassingly parallel, so games are farmed out to worker
processes when more than one is requested.  Workers load the network from the
checkpoint on disk rather than receiving it through a pipe, which keeps the
per-game overhead to a single file read.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from ..evaluation.arena import match, match_summary
from ..evaluation.baselines import MaterialAgent, MCTSAgent, RandomAgent
from ..nn.modules import NetConfig, PolicyValueNet
from ..nn.optim import WarmupCosineLR, build_optimizer
from ..nn.tensor import no_grad
from ..search.evaluator import Evaluator
from ..search.mcts import MCTSConfig
from ..selfplay.game import SelfPlayConfig, play_game
from ..selfplay.replay import ReplayBuffer
from .losses import LossWeights, compute_loss
from .metrics import MetricLogger, write_chart


@dataclass
class TrainConfig:
    """Everything that defines a training run."""

    run_name: str = "run"
    output_dir: str = "runs"
    seed: int = 0

    # network
    channels: int = 64
    blocks: int = 6
    use_se: bool = True
    wdl: bool = True
    history_length: int = 8

    # loop shape
    iterations: int = 20
    games_per_iteration: int = 16
    steps_per_iteration: int = 60
    batch_size: int = 64
    buffer_capacity: int = 60_000
    min_buffer: int = 512
    workers: int = 1

    # Whole games - never individual positions - are held out.  Positions from
    # one game share an outcome and near-identical history planes, so a
    # position-level split leaks the label straight across it.
    validation_fraction: float = 0.06
    validation_capacity: int = 6_000
    validation_batch: int = 256

    # optimisation
    optimizer: str = "adam"
    learning_rate: float = 2e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    grad_clip: float = 4.0
    warmup_steps: int = 100
    policy_weight: float = 1.0
    value_weight: float = 1.0
    label_smoothing: float = 0.02

    # search / self-play
    simulations: int = 128
    fast_simulations: int = 32
    full_search_probability: float = 0.25
    mcts_batch_size: int = 16
    temperature_moves: int = 20
    max_moves: int = 200
    resign_threshold: float = -0.92
    adjudicate_margin: int = 300

    # evaluation
    gate_every: int = 2
    gate_games: int = 20
    gate_threshold: float = 0.55
    gate_simulations: int = 64
    gate_max_moves: int = 160
    benchmark_every: int = 5
    benchmark_games: int = 12
    benchmark_simulations: int = 64
    adjudicate_margin: int = 500      # a rook up at the move limit counts as a win
    resume: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Self-play workers
# ---------------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _pool_context():
    """Pick a multiprocessing start method that is safe on this platform.

    ``fork`` is the cheapest way to hand a worker its configuration, and on
    Linux it is what we want.  On macOS it is a trap: the system frameworks
    and Accelerate (which NumPy links against) are not fork-safe, so a forked
    child that touches BLAS can deadlock or die with an ``objc`` initialise
    error.  Python itself stopped defaulting to fork on macOS in 3.8 for this
    reason.  ``spawn`` re-imports the module instead, which costs a second per
    worker and is why the worker entry points are module-level and the config
    is passed as plain dicts.
    """
    if sys.platform == "darwin":
        return mp.get_context("spawn")
    try:
        return mp.get_context("fork")
    except ValueError:  # pragma: no cover - Windows
        return mp.get_context("spawn")


def _worker_init(checkpoint: str, selfplay_cfg: dict, mcts_cfg: dict) -> None:
    model = PolicyValueNet.load(checkpoint)
    model.eval()
    _WORKER_STATE["evaluator"] = Evaluator(model, cache_size=50_000)
    _WORKER_STATE["selfplay"] = SelfPlayConfig(**selfplay_cfg)
    _WORKER_STATE["mcts"] = MCTSConfig(**mcts_cfg)


def _worker_play(seed: int):
    """Play one game, returning ``None`` rather than propagating a failure.

    ``Pool.map`` aborts the whole batch if any task raises, which on a cluster
    would throw away an iteration of self-play because one game hit a bad
    edge case.  Isolating the failure costs one lost game instead.
    """
    evaluator = _WORKER_STATE["evaluator"]
    evaluator.clear_cache()
    try:
        return play_game(evaluator, _WORKER_STATE["selfplay"], _WORKER_STATE["mcts"],
                         rng=np.random.default_rng(seed))
    except Exception as exc:  # noqa: BLE001 - deliberately broad, then reported
        print(f"self-play game with seed {seed} failed: {exc!r}", file=sys.stderr)
        return None


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.run_dir = os.path.join(config.output_dir, config.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)

        np.random.seed(config.seed)
        self.rng = np.random.default_rng(config.seed)

        net_config = NetConfig(channels=config.channels, blocks=config.blocks,
                               use_se=config.use_se, wdl=config.wdl,
                               history_length=config.history_length)
        self.model = PolicyValueNet(net_config)
        self.best_model = PolicyValueNet(net_config)
        self.best_model.load_state_dict(self.model.state_dict())

        opt_kwargs = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
        if config.optimizer.lower() == "sgd":
            opt_kwargs["momentum"] = config.momentum
        self.optimizer = build_optimizer(self.model.parameters(), config.optimizer, **opt_kwargs)
        total_steps = max(1, config.iterations * config.steps_per_iteration)
        self.schedule = WarmupCosineLR(self.optimizer, total_steps, config.warmup_steps)

        self.buffer = ReplayBuffer(config.buffer_capacity, seed=config.seed)
        self.validation = ReplayBuffer(config.validation_capacity, seed=config.seed + 1)
        self.logger = MetricLogger(os.path.join(self.run_dir, "metrics.jsonl"))
        self.loss_weights = LossWeights(config.policy_weight, config.value_weight,
                                        config.label_smoothing)
        self.global_step = 0
        self._last_train_loss = 0.0
        self.promotions = 0
        self.games_played = 0
        self.start_iteration = 0
        if config.resume:
            self._try_resume()

        with open(os.path.join(self.run_dir, "config.json"), "w") as fh:
            json.dump(config.to_dict(), fh, indent=2)

    def _try_resume(self) -> None:
        """Pick up where a previous invocation of this run left off."""
        best = os.path.join(self.run_dir, "checkpoints", "best.npz")
        if not os.path.exists(best):
            return
        try:
            restored = PolicyValueNet.load(best)
        except (OSError, ValueError, KeyError) as exc:
            # Starting a long run over from scratch because a checkpoint was
            # unreadable is exactly the failure that must never be silent.
            print(f"WARNING: could not resume from {best} ({exc}); "
                  f"starting from a fresh network", file=sys.stderr)
            return
        self.model.load_state_dict(restored.state_dict())
        self.best_model.load_state_dict(restored.state_dict())
        done = [r for r in self.logger.records if "iteration" in r]
        if done:
            self.start_iteration = int(max(r["iteration"] for r in done))
            self.global_step = int(max(r.get("global_step", 0) for r in done))
            self.promotions = int(max(r.get("promotions", 0) for r in done))
        print(f"resumed from {best} at iteration {self.start_iteration}")

    # -- configs --------------------------------------------------------
    def _selfplay_config(self) -> SelfPlayConfig:
        c = self.config
        return SelfPlayConfig(
            simulations=c.simulations, fast_simulations=c.fast_simulations,
            full_search_probability=c.full_search_probability,
            temperature_moves=c.temperature_moves, max_moves=c.max_moves,
            resign_threshold=c.resign_threshold,
            adjudicate_margin=c.adjudicate_margin,
        )

    def _mcts_config(self, simulations: Optional[int] = None) -> MCTSConfig:
        return MCTSConfig(simulations=simulations or self.config.simulations,
                          batch_size=self.config.mcts_batch_size)

    # -- phases ---------------------------------------------------------
    def generate_games(self, iteration: int) -> Dict[str, float]:
        c = self.config
        if not c.gate_every:
            # No gating: self-play always uses the latest weights.  This is
            # what AlphaZero does - the evaluate-and-promote step was an
            # AlphaGo Zero idea that the AlphaZero paper dropped, and on short
            # runs it costs more compute than the stability it buys.
            self.best_model.load_state_dict(self.model.state_dict())
        best_path = os.path.join(self.run_dir, "checkpoints", "best.npz")
        self.best_model.save(best_path)
        seeds = [int(self.rng.integers(0, 2 ** 31 - 1)) for _ in range(c.games_per_iteration)]
        start = time.time()

        if c.workers > 1:
            ctx = _pool_context()
            with ctx.Pool(
                processes=c.workers,
                initializer=_worker_init,
                initargs=(best_path, asdict(self._selfplay_config()),
                          asdict(self._mcts_config())),
                maxtasksperchild=64,   # bound worker RSS on long runs
            ) as pool:
                games = [g for g in pool.map(_worker_play, seeds) if g is not None]
        else:
            evaluator = Evaluator(self.best_model, cache_size=100_000)
            games = []
            for seed in seeds:
                evaluator.clear_cache()
                games.append(play_game(evaluator, self._selfplay_config(),
                                       self._mcts_config(),
                                       rng=np.random.default_rng(seed)))

        samples = 0
        held_out = 0
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
        plies = []
        adjudicated = 0
        for game in games:
            if self.rng.random() < c.validation_fraction:
                held_out += self.validation.add(game.samples)
            else:
                samples += self.buffer.add(game.samples)
            results[game.result] = results.get(game.result, 0) + 1
            plies.append(game.plies)
            adjudicated += int(getattr(game, "adjudicated", False))
        self.games_played += len(games)
        elapsed = time.time() - start
        return {
            "selfplay_seconds": elapsed,
            "games": len(games),
            "failed_games": len(seeds) - len(games),
            "new_samples": samples,
            "buffer_size": len(self.buffer),
            "mean_plies": float(np.mean(plies)) if plies else 0.0,
            "white_wins": results["1-0"],
            "black_wins": results["0-1"],
            "draws": results["1/2-1/2"],
            "games_per_second": len(games) / max(elapsed, 1e-9),
            "decisive_fraction": 1.0 - results["1/2-1/2"] / max(1, len(games)),
            "adjudicated_fraction": adjudicated / max(1, len(games)),
            "held_out_samples": held_out,
            "validation_size": len(self.validation),
            # How many times the optimiser touches each freshly generated
            # position before it ages out of the window.  Above ~4 this run is
            # memorising rather than learning; see docs/training.md.
            "reuse_ratio": (c.steps_per_iteration * c.batch_size) / max(1, samples),
        }

    def train_steps(self) -> Dict[str, float]:
        c = self.config
        if len(self.buffer) < max(c.min_buffer, c.batch_size):
            return {"train_skipped": 1.0}
        self.model.train()
        totals: Dict[str, float] = {}
        start = time.time()
        for _ in range(c.steps_per_iteration):
            lr = self.schedule.step(self.global_step)
            planes, policy, mask, values = self.buffer.sample_batch(c.batch_size)
            loss, parts = compute_loss(self.model, planes, policy, mask, values,
                                       self.loss_weights)
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = self.optimizer.clip_grad_norm(c.grad_clip)
            self.optimizer.step()
            self.global_step += 1
            parts["grad_norm"] = grad_norm
            parts["learning_rate"] = lr
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v
        n = c.steps_per_iteration
        out = {k: v / n for k, v in totals.items()}
        self._last_train_loss = out.get("total_loss", 0.0)
        out["train_seconds"] = time.time() - start
        out["global_step"] = self.global_step
        out.update(self.evaluate_validation())
        return out

    def evaluate_validation(self) -> Dict[str, float]:
        """Loss on games the optimiser has never seen.

        This is the only number in the loop that can distinguish learning from
        memorisation.  Without it a run can drive training loss to 0.25 while
        the value head is no better than the majority class on new positions -
        which is exactly what the first two runs of this project did.
        """
        c = self.config
        # Report as soon as there is enough held-out data to be meaningful
        # rather than waiting for a full batch; a short run would otherwise
        # never produce the one metric that matters.
        floor = min(c.validation_batch, 32)
        if len(self.validation) < floor:
            return {}
        batch = min(c.validation_batch, len(self.validation))
        was_training = self.model.training
        self.model.eval()          # frozen batch-norm statistics, no leakage
        try:
            with no_grad():
                planes, policy, mask, values = self.validation.sample_batch(batch)
                _, parts = compute_loss(self.model, planes, policy, mask, values,
                                        self.loss_weights)
        finally:
            self.model.train(was_training)
        return {
            "val_policy_loss": parts["policy_loss"],
            "val_value_loss": parts["value_loss"],
            "val_total_loss": parts["total_loss"],
            "val_policy_accuracy": parts["policy_accuracy"],
            "val_value_accuracy": parts.get("value_accuracy", float("nan")),
            "val_batch": batch,
            # The number to watch.  A widening gap means the network is
            # memorising the replay window instead of learning chess.
            "generalisation_gap": parts["total_loss"] - self._last_train_loss,
        }

    def gate(self) -> Dict[str, float]:
        """Play the challenger against the incumbent; promote if clearly better."""
        c = self.config
        challenger = MCTSAgent(Evaluator(self.model), c.gate_simulations,
                               self._mcts_config(c.gate_simulations),
                               seed=int(self.rng.integers(0, 2 ** 31 - 1)),
                               name="challenger")
        incumbent = MCTSAgent(Evaluator(self.best_model), c.gate_simulations,
                              self._mcts_config(c.gate_simulations),
                              seed=int(self.rng.integers(0, 2 ** 31 - 1)),
                              name="best")
        record, _ = match(challenger, incumbent, c.gate_games,
                          seed=int(self.rng.integers(0, 2 ** 31 - 1)), opening_plies=4,
                          max_moves=c.gate_max_moves)
        summary = match_summary(record)
        promoted = summary["score"] >= c.gate_threshold
        if promoted:
            self.best_model.load_state_dict(self.model.state_dict())
            self.promotions += 1
        return {
            "gate_score": summary["score"],
            "gate_elo": summary["elo"],
            "gate_los": summary["los"],
            "gate_promoted": float(promoted),
            "promotions": self.promotions,
        }

    def benchmark(self) -> Dict[str, float]:
        """Measure the current best against the fixed baseline ladder."""
        c = self.config
        agent = MCTSAgent(Evaluator(self.best_model), c.benchmark_simulations,
                          self._mcts_config(c.benchmark_simulations), name="neuralchess")
        out: Dict[str, float] = {}
        opponents = [RandomAgent(seed=7), MaterialAgent(seed=8)]
        for opponent in opponents:
            record, _ = match(agent, opponent, c.benchmark_games,
                              seed=int(self.rng.integers(0, 2 ** 31 - 1)), opening_plies=2,
                              max_moves=c.gate_max_moves,
                              adjudicate_margin=c.adjudicate_margin)
            summary = match_summary(record)
            out[f"vs_{opponent.name}_score"] = summary["score"]
            out[f"vs_{opponent.name}_elo"] = summary["elo"]
        return out

    # -- driver ---------------------------------------------------------
    def run(self, on_iteration=None) -> List[dict]:
        c = self.config
        for iteration in range(self.start_iteration + 1, c.iterations + 1):
            record: Dict[str, float] = {"iteration": iteration}
            record.update(self.generate_games(iteration))
            record.update(self.train_steps())
            if c.gate_every and iteration % c.gate_every == 0:
                record.update(self.gate())
            if c.benchmark_every and iteration % c.benchmark_every == 0:
                record.update(self.benchmark())
            self.logger.log(**record)
            self.save_checkpoint(iteration)
            self.write_plots()
            if on_iteration:
                on_iteration(record)
        return self.logger.records

    def save_checkpoint(self, iteration: int) -> str:
        path = os.path.join(self.run_dir, "checkpoints", f"iter{iteration:04d}.npz")
        self.model.save(path)
        self.best_model.save(os.path.join(self.run_dir, "checkpoints", "best.npz"))
        return path

    def write_plots(self) -> None:
        records = self.logger.records
        if len(records) < 2:
            return
        def series(key):
            return [r.get(key) for r in records]
        write_chart(os.path.join(self.run_dir, "loss.svg"),
                    {"total": series("total_loss"), "policy": series("policy_loss"),
                     "value": series("value_loss")},
                    title="Training loss", ylabel="loss")
        write_chart(os.path.join(self.run_dir, "accuracy.svg"),
                    {"policy top-1": series("policy_accuracy"),
                     "value top-1": series("value_accuracy")},
                    title="Prediction accuracy vs search targets", ylabel="accuracy")
        bench = {k: series(k) for k in ("vs_random_score", "vs_material_score")
                 if any(r.get(k) is not None for r in records)}
        if bench:
            write_chart(os.path.join(self.run_dir, "benchmark.svg"), bench,
                        title="Score against fixed baselines", ylabel="score")
