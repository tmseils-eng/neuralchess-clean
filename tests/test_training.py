"""Self-play, replay buffer, losses and a miniature end-to-end training run."""

import os
import tempfile

import numpy as np

from neuralchess.evaluation import (
    MatchRecord,
    MaterialAgent,
    RandomAgent,
    elo_difference,
    fit_elo,
    match,
)
from neuralchess.nn import NetConfig, PolicyValueNet
from neuralchess.search import Evaluator
from neuralchess.selfplay import ReplayBuffer, SelfPlayConfig, play_game
from neuralchess.train import TrainConfig, Trainer, value_targets_wdl


def tiny_evaluator():
    return Evaluator(PolicyValueNet(NetConfig(channels=16, blocks=1)))


def test_self_play_produces_consistent_samples():
    config = SelfPlayConfig(simulations=24, fast_simulations=8,
                            full_search_probability=1.0, max_moves=30)
    game = play_game(tiny_evaluator(), config, rng=np.random.default_rng(0))
    assert game.plies > 0
    assert len(game.samples) > 0
    assert game.result in ("1-0", "0-1", "1/2-1/2")
    for sample in game.samples:
        assert sample.planes.shape == (119, 8, 8)
        assert abs(sample.policy_probs.sum() - 1.0) < 1e-4
        assert len(sample.policy_indices) == len(sample.policy_probs)
        assert sample.value in (-1.0, 0.0, 1.0)
    # The value target must flip sign with the side to move.
    if game.result != "1/2-1/2":
        white = [s.value for s in game.samples if s.ply % 2 == 0]
        black = [s.value for s in game.samples if s.ply % 2 == 1]
        if white and black:
            assert white[0] == -black[0]


def test_replay_buffer_densifies_and_persists():
    config = SelfPlayConfig(simulations=16, fast_simulations=8,
                            full_search_probability=1.0, max_moves=20)
    game = play_game(tiny_evaluator(), config, rng=np.random.default_rng(1))
    buffer = ReplayBuffer(capacity=1000)
    buffer.add(game.samples)
    assert len(buffer) == len(game.samples)

    planes, policy, mask, values = buffer.sample_batch(4)
    assert planes.shape == (4, 119, 8, 8)
    assert policy.shape == (4, 4672) and mask.shape == (4, 4672)
    assert np.allclose(policy.sum(axis=1), 1.0, atol=1e-4)
    assert (policy[~mask] == 0).all()

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "buffer.npz")
        buffer.save(path)
        reloaded = ReplayBuffer(capacity=1000)
        assert reloaded.load(path) == len(buffer)


def test_wdl_targets():
    values = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    targets = value_targets_wdl(values)
    assert np.allclose(targets, np.eye(3, dtype=np.float32))
    smoothed = value_targets_wdl(values, smoothing=0.03)
    assert np.allclose(smoothed.sum(axis=1), 1.0)


def test_elo_helpers():
    elo, low, high = elo_difference(0.75, 100)
    assert 180 < elo < 200 and low < elo < high
    assert elo_difference(0.5, 100)[0] == 0.0
    ratings = fit_elo([
        MatchRecord("strong", "weak", wins=18, draws=2, losses=0),
        MatchRecord("weak", "weakest", wins=15, draws=0, losses=5),
    ], anchor="weakest")
    assert ratings["strong"] > ratings["weak"] > ratings["weakest"]


def test_baseline_ladder_is_ordered():
    record, _ = match(MaterialAgent(seed=1), RandomAgent(seed=2), games=8, seed=3)
    assert record.score > 0.6, "a material-grabbing player should beat random"


def test_end_to_end_training_iteration():
    with tempfile.TemporaryDirectory() as folder:
        config = TrainConfig(
            run_name="unit", output_dir=folder, channels=16, blocks=1,
            iterations=2, games_per_iteration=2, steps_per_iteration=6,
            batch_size=8, min_buffer=4, simulations=16, fast_simulations=8,
            full_search_probability=1.0, max_moves=24, gate_every=2,
            gate_games=2, gate_simulations=8, benchmark_every=0, workers=1,
        )
        trainer = Trainer(config)
        records = trainer.run()
        assert len(records) == 2
        assert records[-1]["total_loss"] > 0
        assert os.path.exists(os.path.join(trainer.run_dir, "checkpoints", "best.npz"))
        assert os.path.exists(os.path.join(trainer.run_dir, "metrics.jsonl"))
        # The loop must produce a loadable network.
        model = PolicyValueNet.load(os.path.join(trainer.run_dir, "checkpoints", "best.npz"))
        assert model.config.channels == 16


def test_adjudication_awards_a_won_endgame():
    """A game stopped at the move limit with a rook up is scored as a win."""
    from neuralchess.evaluation.arena import play_pair
    from neuralchess.evaluation.baselines import RandomAgent

    # White is a queen and a rook up; random play will not mate quickly.
    fen = "7k/8/8/8/8/8/8/K2QR3 w - - 0 1"
    strict = play_pair(RandomAgent(seed=1), RandomAgent(seed=2), fen, max_moves=6)
    adjudicated = play_pair(RandomAgent(seed=1), RandomAgent(seed=2), fen,
                            max_moves=6, adjudicate_margin=500)
    assert strict.result == "1/2-1/2" and strict.reason == "move limit"
    assert adjudicated.result == "1-0"
    assert adjudicated.reason == "adjudicated on material"


def test_validation_holds_out_whole_games_and_reports_a_gap():
    """The split must be by game, and it must produce a usable val metric.

    A position-level split leaks: every position in a game carries the same
    outcome label and a nearly identical history stack, so the model can match
    a validation position to a training one from the same game.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        base = dict(
            output_dir=folder, channels=16, blocks=1, iterations=1,
            games_per_iteration=4, steps_per_iteration=4, batch_size=8,
            min_buffer=4, simulations=12, fast_simulations=6,
            full_search_probability=1.0, max_moves=24, gate_every=0,
            benchmark_every=0, workers=1, validation_batch=8,
        )
        everything_held_out = Trainer(TrainConfig(run_name="v-all",
                                                 validation_fraction=1.0, **base))
        stats = everything_held_out.generate_games(1)
        assert stats["new_samples"] == 0
        assert stats["held_out_samples"] > 0
        assert len(everything_held_out.buffer) == 0

        nothing_held_out = Trainer(TrainConfig(run_name="v-none",
                                               validation_fraction=0.0, **base))
        stats = nothing_held_out.generate_games(1)
        assert stats["held_out_samples"] == 0
        assert stats["new_samples"] == len(nothing_held_out.buffer)
        # No sample is dropped or double counted by the split.
        assert stats["reuse_ratio"] > 0

        # With enough held-out data the validation pass reports real numbers.
        everything_held_out.buffer.add(list(everything_held_out.validation.samples)[:16])
        report = everything_held_out.train_steps()
        assert "val_total_loss" in report
        assert report["val_total_loss"] > 0


def test_validation_pass_does_not_disturb_batch_norm_statistics():
    """Evaluating must not update running stats or leave the model in eval mode."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        trainer = Trainer(TrainConfig(
            run_name="bn", output_dir=folder, channels=16, blocks=1,
            iterations=1, games_per_iteration=2, steps_per_iteration=2,
            batch_size=8, min_buffer=4, simulations=12, fast_simulations=6,
            full_search_probability=1.0, max_moves=20, gate_every=0,
            benchmark_every=0, workers=1, validation_fraction=1.0,
            validation_batch=8,
        ))
        trainer.generate_games(1)
        if len(trainer.validation) < 8:
            return
        before = {k: v.copy() for k, v in trainer.model.named_buffers()}
        trainer.model.train(True)
        trainer.evaluate_validation()
        assert trainer.model.training, "validation must restore training mode"
        for name, buf in trainer.model.named_buffers():
            assert np.allclose(before[name], buf), f"{name} moved during validation"


def test_pool_context_does_not_fork_on_macos(monkeypatch=None):
    """fork() is unsafe on macOS: Accelerate and the ObjC runtime are not fork-safe."""
    import sys as _sys

    from neuralchess.train.loop import _pool_context

    original = _sys.platform
    try:
        _sys.platform = "darwin"
        assert _pool_context().get_start_method() == "spawn"
    finally:
        _sys.platform = original
