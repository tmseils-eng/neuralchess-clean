"""Replay buffer: a fixed-size window over the most recent self-play data.

Policy targets are stored sparsely (only the legal moves carry probability),
which is roughly a 100x memory saving over dense 4672-vectors and lets a
laptop hold a window of hundreds of thousands of positions.
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Iterable

import numpy as np

from ..encoding.policy_map import POLICY_SIZE
from .game import Sample


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000, seed: int = 0):
        self.capacity = capacity
        self.samples: "deque[Sample]" = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        self.total_added = 0

    def __len__(self) -> int:
        return len(self.samples)

    def add(self, samples: Iterable[Sample]) -> int:
        count = 0
        for sample in samples:
            self.samples.append(sample)
            count += 1
        self.total_added += count
        return count

    def sample_batch(self, batch_size: int):
        """Draw a uniform batch and densify the policy targets."""
        if not self.samples:
            raise ValueError("replay buffer is empty")
        picks = [self.samples[self.rng.randrange(len(self.samples))]
                 for _ in range(batch_size)]
        planes = np.stack([p.planes for p in picks]).astype(np.float32)
        policy = np.zeros((batch_size, POLICY_SIZE), dtype=np.float32)
        mask = np.zeros((batch_size, POLICY_SIZE), dtype=bool)
        for i, p in enumerate(picks):
            policy[i, p.policy_indices] = p.policy_probs
            mask[i, p.policy_indices] = True
        values = np.array([p.value for p in picks], dtype=np.float32)
        return planes, policy, mask, values

    # -- persistence ----------------------------------------------------
    def save(self, path: str) -> None:
        if not self.samples:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        planes = np.stack([s.planes for s in self.samples]).astype(np.uint8)
        values = np.array([s.value for s in self.samples], dtype=np.float32)
        plies = np.array([s.ply for s in self.samples], dtype=np.int32)
        lengths = np.array([len(s.policy_indices) for s in self.samples], dtype=np.int32)
        indices = np.concatenate([s.policy_indices for s in self.samples]).astype(np.int32)
        probs = np.concatenate([s.policy_probs for s in self.samples]).astype(np.float32)
        np.savez_compressed(path, planes=planes, values=values, plies=plies,
                            lengths=lengths, indices=indices, probs=probs)

    def load(self, path: str) -> int:
        data = np.load(path)
        planes, values = data["planes"], data["values"]
        plies, lengths = data["plies"], data["lengths"]
        indices, probs = data["indices"], data["probs"]
        offset = 0
        added = 0
        for i, length in enumerate(lengths):
            sample = Sample(
                planes=planes[i].astype(np.float32),
                policy_indices=indices[offset:offset + length],
                policy_probs=probs[offset:offset + length],
                value=float(values[i]),
                ply=int(plies[i]),
            )
            offset += length
            self.samples.append(sample)
            added += 1
        self.total_added += added
        return added

    def stats(self) -> dict:
        if not self.samples:
            return {"size": 0}
        values = np.array([s.value for s in self.samples])
        return {
            "size": len(self.samples),
            "capacity": self.capacity,
            "total_added": self.total_added,
            "mean_value": float(values.mean()),
            "draw_fraction": float((values == 0).mean()),
        }
