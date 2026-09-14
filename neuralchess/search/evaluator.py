"""Network evaluation for the search, with batching and a position cache."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence, Tuple

import numpy as np

from ..encoding.planes import HISTORY_LENGTH


class Evaluator:
    """Wraps a policy-value network for use inside MCTS.

    Two things matter for throughput: evaluating leaves in batches (a single
    NumPy matmul over 16 positions costs barely more than one over a single
    position) and never evaluating the same position twice within a search.

    The cache stores the *expanded* result - the legal move list and its
    normalised priors - rather than the raw 4,672-wide logit vector.  Caching
    the logits is the obvious implementation and it is a memory trap: at
    18 KB per entry a 100k-entry cache is 1.9 GB per process, which on a
    multi-worker self-play run is the difference between fitting in RAM and
    thrashing.  The expanded form is ~600 bytes.
    """

    def __init__(self, model, cache_size: int = 60_000):
        self.model = model
        self.cache_size = cache_size
        self._cache: "OrderedDict[int, Tuple[list, np.ndarray, float]]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.batches = 0
        self.positions = 0

    @property
    def history_length(self) -> int:
        """History depth this network was trained with; encoders must match."""
        return getattr(self.model.config, "history_length", HISTORY_LENGTH)

    # -- cache ----------------------------------------------------------
    def lookup(self, key: int):
        entry = self._cache.get(key)
        if entry is not None:
            self._cache.move_to_end(key)
            self.hits += 1
        return entry

    def store(self, key: int, moves, priors: np.ndarray, value: float) -> None:
        self._cache[key] = (moves, priors, value)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- evaluation -----------------------------------------------------
    def evaluate_batch(self, planes: Sequence[np.ndarray]):
        """Run the network on a list of ``(119, 8, 8)`` inputs."""
        if not planes:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)
        batch = np.stack(planes).astype(np.float32, copy=False)
        self.batches += 1
        self.positions += batch.shape[0]
        self.misses += batch.shape[0]
        return self.model.predict(batch)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "batches": self.batches,
            "positions": self.positions,
            "avg_batch": (self.positions / self.batches) if self.batches else 0.0,
        }


class RandomEvaluator:
    """Uniform policy, zero value - the untrained control for experiments."""

    def __init__(self, seed: int = 0, history_length: int = HISTORY_LENGTH):
        self.rng = np.random.default_rng(seed)
        self.history_length = history_length
        self.hits = self.misses = self.batches = self.positions = 0

    def lookup(self, key: int):
        return None

    def store(self, key: int, moves, priors: np.ndarray, value: float) -> None:
        pass

    def clear_cache(self) -> None:
        pass

    def evaluate_batch(self, planes: Sequence[np.ndarray]):
        n = len(planes)
        from ..encoding.policy_map import POLICY_SIZE

        return np.zeros((n, POLICY_SIZE), dtype=np.float32), np.zeros(n, dtype=np.float32)

    def stats(self) -> dict:
        return {"cache_hits": 0, "cache_misses": 0, "hit_rate": 0.0,
                "batches": 0, "positions": 0, "avg_batch": 0.0}
