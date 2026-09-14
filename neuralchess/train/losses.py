"""Training objective.

AlphaZero minimises

.. math:: \\ell = (z - v)^2 - \\pi^\\top \\log p + c\\lVert\\theta\\rVert^2

Here the value head is a three-way win/draw/loss classifier by default, so
the squared error is replaced by a cross entropy against the outcome, which
is better calibrated and gives the search a usable draw probability.  The
weight-decay term lives in the optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..nn.functional import cross_entropy_soft, mse_loss, softmax_np
from ..nn.tensor import Tensor


@dataclass
class LossWeights:
    policy: float = 1.0
    value: float = 1.0
    label_smoothing: float = 0.0


def value_targets_wdl(values: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
    """Map ``z`` in ``{-1, 0, +1}`` to win/draw/loss target distributions."""
    n = values.shape[0]
    target = np.zeros((n, 3), dtype=np.float32)
    target[values > 0.5, 0] = 1.0
    target[np.abs(values) <= 0.5, 1] = 1.0
    target[values < -0.5, 2] = 1.0
    if smoothing > 0:
        target = target * (1.0 - smoothing) + smoothing / 3.0
    return target


def compute_loss(model, planes: np.ndarray, policy_target: np.ndarray,
                 legal_mask: np.ndarray, values: np.ndarray,
                 weights: Optional[LossWeights] = None):
    """Forward pass plus loss; returns ``(total, parts_dict)``."""
    weights = weights or LossWeights()
    policy_logits, value_out = model(Tensor(planes))

    policy_loss = cross_entropy_soft(policy_logits, policy_target, legal_mask)
    if model.config.wdl:
        target = value_targets_wdl(values, weights.label_smoothing)
        value_loss = cross_entropy_soft(value_out, target)
    else:
        value_loss = mse_loss(value_out.reshape(-1), values)

    total = policy_loss * weights.policy + value_loss * weights.value

    with_np = policy_logits.data
    parts = {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "total_loss": float(total.item()),
        "policy_accuracy": float(_top1_accuracy(with_np, policy_target, legal_mask)),
        "policy_entropy": float(_entropy(with_np, legal_mask)),
    }
    if model.config.wdl:
        parts["value_accuracy"] = float(
            (softmax_np(value_out.data).argmax(1) == value_targets_wdl(values).argmax(1)).mean()
        )
    return total, parts


def _top1_accuracy(logits: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    masked = np.where(mask, logits, -np.inf)
    return float((masked.argmax(axis=1) == target.argmax(axis=1)).mean())


def _entropy(logits: np.ndarray, mask: np.ndarray) -> float:
    masked = np.where(mask, logits, -1e9)
    probs = softmax_np(masked, axis=1)
    return float(-(probs * np.log(probs + 1e-9)).sum(axis=1).mean())
