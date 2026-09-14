"""Differentiable convolution, normalisation and loss primitives.

These are the operations the autograd engine in :mod:`.tensor` does not get
for free from elementwise arithmetic.  Convolution uses the classic
``im2col`` reformulation: patches are gathered into a matrix so the forward
pass is one ``matmul``, and both gradients fall out of the same matmul
transposed.  On an 8x8 board this is comfortably fast in pure NumPy.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .tensor import Tensor

# ---------------------------------------------------------------------------
# im2col / col2im
# ---------------------------------------------------------------------------

def im2col(x: np.ndarray, kh: int, kw: int, pad: int) -> np.ndarray:
    """``(N, C, H, W) -> (N, C*kh*kw, Ho*Wo)`` patch matrix."""
    n, c, h, w = x.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    ho, wo = h + 2 * pad - kh + 1, w + 2 * pad - kw + 1
    cols = np.empty((n, c, kh, kw, ho, wo), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = x[:, :, i:i + ho, j:j + wo]
    return cols.reshape(n, c * kh * kw, ho * wo)


def col2im(cols: np.ndarray, shape: Tuple[int, int, int, int],
           kh: int, kw: int, pad: int) -> np.ndarray:
    """Adjoint of :func:`im2col`: scatter-add patches back to an image."""
    n, c, h, w = shape
    ho, wo = h + 2 * pad - kh + 1, w + 2 * pad - kw + 1
    cols = cols.reshape(n, c, kh, kw, ho, wo)
    out = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    for i in range(kh):
        for j in range(kw):
            out[:, :, i:i + ho, j:j + wo] += cols[:, :, i, j]
    return out[:, :, pad:pad + h, pad:pad + w] if pad else out


def conv2d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
           padding: int = 0) -> Tensor:
    """2-D cross-correlation with unit stride and square padding."""
    n, c, h, w = x.data.shape
    o, ci, kh, kw = weight.data.shape
    if ci != c:
        raise ValueError(f"channel mismatch: input {c}, weight {ci}")
    ho, wo = h + 2 * padding - kh + 1, w + 2 * padding - kw + 1

    cols = im2col(x.data, kh, kw, padding)                 # (N, C*kh*kw, L)
    wmat = weight.data.reshape(o, -1)                      # (O, C*kh*kw)
    out_data = np.einsum("ok,nkl->nol", wmat, cols, optimize=True)
    if bias is not None:
        out_data = out_data + bias.data.reshape(1, o, 1)
    out_data = out_data.reshape(n, o, ho, wo)

    parents = [x, weight] + ([bias] if bias is not None else [])
    out = Tensor(out_data,
                 x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad),
                 parents, "conv2d")

    def _bw(grad):
        g = grad.reshape(n, o, ho * wo)                     # (N, O, L)
        if weight.requires_grad:
            weight.accumulate(
                np.einsum("nol,nkl->ok", g, cols, optimize=True).reshape(weight.data.shape)
            )
        if bias is not None and bias.requires_grad:
            bias.accumulate(g.sum(axis=(0, 2)))
        if x.requires_grad:
            dcols = np.einsum("ok,nol->nkl", wmat, g, optimize=True)
            x.accumulate(col2im(dcols, (n, c, h, w), kh, kw, padding))

    out._backward = _bw
    return out


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def batch_norm(x: Tensor, gamma: Tensor, beta: Tensor,
               running_mean: np.ndarray, running_var: np.ndarray,
               training: bool, momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    """Batch normalisation over the channel axis of an ``(N, C, H, W)`` input."""
    axes = (0, 2, 3)
    shape = (1, -1, 1, 1)
    if training:
        mean = x.data.mean(axis=axes)
        var = x.data.var(axis=axes)
        running_mean *= (1.0 - momentum)
        running_mean += momentum * mean
        running_var *= (1.0 - momentum)
        running_var += momentum * var
    else:
        mean, var = running_mean, running_var

    inv_std = 1.0 / np.sqrt(var + eps)
    xhat = (x.data - mean.reshape(shape)) * inv_std.reshape(shape)
    out_data = xhat * gamma.data.reshape(shape) + beta.data.reshape(shape)
    out = Tensor(out_data, x.requires_grad or gamma.requires_grad, (x, gamma, beta), "batch_norm")

    def _bw(g):
        if gamma.requires_grad:
            gamma.accumulate((g * xhat).sum(axis=axes))
        if beta.requires_grad:
            beta.accumulate(g.sum(axis=axes))
        if not x.requires_grad:
            return
        gx = g * gamma.data.reshape(shape)
        if training:
            dx = (gx - gx.mean(axis=axes, keepdims=True)
                  - xhat * (gx * xhat).mean(axis=axes, keepdims=True)) * inv_std.reshape(shape)
        else:
            dx = gx * inv_std.reshape(shape)
        x.accumulate(dx)

    out._backward = _bw
    return out


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def cross_entropy_soft(logits: Tensor, targets: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> Tensor:
    """Cross entropy against a *distribution* target (the MCTS visit policy).

    ``mask`` optionally marks illegal moves, whose logits are pushed to
    ``-inf`` before the softmax so they can never receive probability.
    """
    if mask is not None:
        logits = logits.masked_fill(~mask, -1e9)
    logp = logits.log_softmax(axis=-1)
    target = Tensor(targets)
    nll = (logp * target).sum(axis=-1) * -1.0
    return nll.mean()


def mse_loss(pred: Tensor, target: np.ndarray) -> Tensor:
    diff = pred - Tensor(target)
    return (diff * diff).mean()


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)
