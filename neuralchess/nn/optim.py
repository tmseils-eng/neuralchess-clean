"""Optimisers and learning-rate schedules.

AlphaZero trains with plain SGD plus Nesterov momentum, weight decay and a
step-wise learning-rate drop.  Adam is provided as well because it makes the
short runs used for smoke tests and CI converge in far fewer steps.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np

from .tensor import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor], lr: float):
        self.params: List[Tensor] = [p for p in params if p.requires_grad]
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def clip_grad_norm(self, max_norm: float) -> float:
        """Global L2 gradient clipping; returns the pre-clip norm."""
        total = 0.0
        for p in self.params:
            if p.grad is not None:
                total += float((p.grad ** 2).sum())
        total = math.sqrt(total)
        if max_norm > 0 and total > max_norm:
            scale = max_norm / (total + 1e-6)
            for p in self.params:
                if p.grad is not None:
                    p.grad *= scale
        return total


class SGD(Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.9,
                 weight_decay: float = 1e-4, nesterov: bool = True):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self._velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            v = self._velocity[i]
            v *= self.momentum
            v += g
            update = g + self.momentum * v if self.nesterov else v
            p.data -= self.lr * update


class AdamW(Optimizer):
    """Adam with *decoupled* weight decay (Loshchilov & Hutter, 2019).

    Adding ``wd * p`` to the gradient - what :class:`Adam` below does, and what
    ``weight_decay`` means almost everywhere by default - routes the penalty
    through the adaptive denominator.  Parameters with large gradient second
    moments then get *less* decay than parameters with small ones, which is the
    opposite of the intent and makes the effective decay depend on the loss
    scale.  Applying it directly to the weights instead keeps the penalty
    proportional to the learning rate alone.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self) -> None:
        self._t += 1
        bc1 = 1.0 - self.b1 ** self._t
        bc2 = 1.0 - self.b2 ** self._t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self._m[i] *= self.b1
            self._m[i] += (1 - self.b1) * g
            self._v[i] *= self.b2
            self._v[i] += (1 - self.b2) * (g * g)
            update = (self._m[i] / bc1) / (np.sqrt(self._v[i] / bc2) + self.eps)
            if self.weight_decay:
                update = update + self.weight_decay * p.data   # decoupled
            p.data -= self.lr * update


class Adam(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self) -> None:
        self._t += 1
        bc1 = 1.0 - self.b1 ** self._t
        bc2 = 1.0 - self.b2 ** self._t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            self._m[i] *= self.b1
            self._m[i] += (1 - self.b1) * g
            self._v[i] *= self.b2
            self._v[i] += (1 - self.b2) * (g * g)
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

class StepLR:
    """Multiply the learning rate by ``gamma`` at each milestone step."""

    def __init__(self, optimizer: Optimizer, milestones: Sequence[int], gamma: float = 0.1):
        self.optimizer = optimizer
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.base_lr = optimizer.lr

    def step(self, global_step: int) -> float:
        drops = sum(1 for m in self.milestones if global_step >= m)
        self.optimizer.lr = self.base_lr * (self.gamma ** drops)
        return self.optimizer.lr


class WarmupCosineLR:
    """Linear warmup then cosine decay - stable for short from-scratch runs."""

    def __init__(self, optimizer: Optimizer, total_steps: int,
                 warmup_steps: int = 100, min_lr_ratio: float = 0.05):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = warmup_steps
        self.min_ratio = min_lr_ratio
        self.base_lr = optimizer.lr

    def step(self, global_step: int) -> float:
        if global_step < self.warmup_steps:
            ratio = (global_step + 1) / max(1, self.warmup_steps)
        else:
            progress = (global_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(1.0, max(0.0, progress))
            ratio = self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        self.optimizer.lr = self.base_lr * ratio
        return self.optimizer.lr


def build_optimizer(params, name: str = "sgd", **kwargs) -> Optimizer:
    name = name.lower()
    if name == "sgd":
        return SGD(params, **kwargs)
    if name == "adam":
        return Adam(params, **kwargs)
    if name == "adamw":
        return AdamW(params, **kwargs)
    raise ValueError(f"unknown optimizer {name!r}")
