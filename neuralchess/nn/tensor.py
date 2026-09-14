"""A small reverse-mode autograd engine built on NumPy.

The project needs to train a residual CNN in environments where PyTorch is
not available, so the default backend implements backpropagation directly.
The design is the standard one: every :class:`Tensor` remembers the parents
it was produced from and a closure that pushes gradient back to them; a
topological sort over that graph gives the backward pass.

One implementation detail is load-bearing.  The natural way to write these
closures is to have them read ``out.grad`` from the tensor they belong to -
which makes ``out._backward`` hold a reference to ``out``, a reference cycle
that Python can only reclaim through the cyclic collector.  The collector
triggers on object *counts*, not bytes, so a training loop accumulates whole
computation graphs - hundreds of megabytes each - long before it fires.  The
fix is to pass the gradient in as an argument, which leaves every reference
pointing strictly from child to parent and lets refcounting free each graph
the moment the loss goes out of scope.

The API is a small subset of PyTorch's so that ``nn/torch_backend.py`` can
mirror it closely.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Global switch used by :class:`no_grad`.  When inference-only work is being
# done (every MCTS leaf evaluation, for instance) there is no reason to keep
# the graph alive, and dropping it frees the activation memory immediately.
_GRAD_ENABLED = True


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


class no_grad:
    """Context manager (and decorator) that disables graph construction."""

    def __enter__(self):
        global _GRAD_ENABLED
        self._prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, *exc):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev
        return False

    def __call__(self, fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with no_grad():
                return fn(*args, **kwargs)

        return wrapper


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` back to ``shape`` after NumPy broadcasting."""
    if grad.shape == shape:
        return grad
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


def _noop(grad: np.ndarray) -> None:
    return None


class Tensor:
    """An n-dimensional array that records operations for backpropagation."""

    # ``__weakref__`` is listed so the test suite can assert that graphs are
    # released by refcounting alone - see test_graph_is_freed_without_gc.
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_parents", "_op",
                 "__weakref__")

    def __init__(self, data, requires_grad: bool = False,
                 _parents: Sequence["Tensor"] = (), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float32)
        if not _GRAD_ENABLED:
            requires_grad = False
            _parents = ()
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._backward: Callable[[np.ndarray], None] = _noop
        self._parents: Tuple["Tensor", ...] = tuple(_parents)
        self._op = _op

    # -- plumbing -------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op or 'leaf'})"

    def accumulate(self, grad: np.ndarray) -> None:
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = grad.astype(np.float32, copy=True)
        else:
            self.grad += grad

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> "Tensor":
        return Tensor(self.data, requires_grad=False)

    def item(self) -> float:
        return float(self.data.reshape(-1)[0])

    def numpy(self) -> np.ndarray:
        return self.data

    # -- graph ----------------------------------------------------------
    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """Backpropagate from this tensor (must be a scalar unless ``grad``)."""
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() on a non-scalar needs an explicit gradient")
            grad = np.ones_like(self.data)

        topo: List[Tensor] = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if id(parent) not in visited:
                    stack.append((parent, False))

        self.accumulate(np.asarray(grad, dtype=np.float32))
        for node in reversed(topo):
            if node.grad is not None:
                node._backward(node.grad)

    # -- elementwise ops ------------------------------------------------
    @staticmethod
    def _wrap(other) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data + other.data,
                     self.requires_grad or other.requires_grad, (self, other), "add")
        a, b = self, other

        def _bw(g):
            a.accumulate(_unbroadcast(g, a.data.shape))
            b.accumulate(_unbroadcast(g, b.data.shape))

        out._backward = _bw
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data * other.data,
                     self.requires_grad or other.requires_grad, (self, other), "mul")
        a, b = self, other

        def _bw(g):
            a.accumulate(_unbroadcast(g * b.data, a.data.shape))
            b.accumulate(_unbroadcast(g * a.data, b.data.shape))

        out._backward = _bw
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (self._wrap(other) * -1.0)

    def __rsub__(self, other):
        return self._wrap(other) + (self * -1.0)

    def __truediv__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data / other.data,
                     self.requires_grad or other.requires_grad, (self, other), "div")
        a, b = self, other

        def _bw(g):
            a.accumulate(_unbroadcast(g / b.data, a.data.shape))
            b.accumulate(_unbroadcast(-g * a.data / (b.data ** 2), b.data.shape))

        out._backward = _bw
        return out

    def __pow__(self, power: float):
        out = Tensor(self.data ** power, self.requires_grad, (self,), "pow")
        a = self

        def _bw(g):
            a.accumulate(g * power * a.data ** (power - 1))

        out._backward = _bw
        return out

    # -- reductions and shape -------------------------------------------
    def sum(self, axis=None, keepdims: bool = False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     self.requires_grad, (self,), "sum")
        a = self
        shape = self.data.shape

        def _bw(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            a.accumulate(np.broadcast_to(g, shape).copy())

        out._backward = _bw
        return out

    def mean(self, axis=None, keepdims: bool = False):
        count = self.data.size if axis is None else int(np.prod(
            [self.data.shape[a] for a in (axis if isinstance(axis, tuple) else (axis,))]
        ))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(count))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), self.requires_grad, (self,), "reshape")
        a = self
        original = self.data.shape

        def _bw(g):
            a.accumulate(g.reshape(original))

        out._backward = _bw
        return out

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = Tensor(self.data.transpose(axes), self.requires_grad, (self,), "transpose")
        a = self
        inverse = tuple(np.argsort(axes))

        def _bw(g):
            a.accumulate(g.transpose(inverse))

        out._backward = _bw
        return out

    def matmul(self, other: "Tensor"):
        out = Tensor(self.data @ other.data,
                     self.requires_grad or other.requires_grad, (self, other), "matmul")
        a, b = self, other

        def _bw(g):
            a.accumulate(_unbroadcast(g @ np.swapaxes(b.data, -1, -2), a.data.shape))
            b.accumulate(_unbroadcast(np.swapaxes(a.data, -1, -2) @ g, b.data.shape))

        out._backward = _bw
        return out

    __matmul__ = matmul

    # -- activations ----------------------------------------------------
    def relu(self):
        mask = self.data > 0
        out = Tensor(self.data * mask, self.requires_grad, (self,), "relu")
        a = self

        def _bw(g):
            a.accumulate(g * mask)

        out._backward = _bw
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60.0, 60.0)))
        out = Tensor(s, self.requires_grad, (self,), "sigmoid")
        a = self

        def _bw(g):
            a.accumulate(g * s * (1.0 - s))

        out._backward = _bw
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,), "tanh")
        a = self

        def _bw(g):
            a.accumulate(g * (1.0 - t * t))

        out._backward = _bw
        return out

    def log_softmax(self, axis: int = -1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        value = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        out = Tensor(value, self.requires_grad, (self,), "log_softmax")
        softmax = np.exp(value)
        a = self

        def _bw(g):
            a.accumulate(g - softmax * g.sum(axis=axis, keepdims=True))

        out._backward = _bw
        return out

    def masked_fill(self, mask: np.ndarray, value: float):
        """Set entries where ``mask`` is True to ``value`` (used for illegal moves)."""
        out = Tensor(np.where(mask, np.float32(value), self.data),
                     self.requires_grad, (self,), "masked_fill")
        keep = ~mask
        a = self

        def _bw(g):
            a.accumulate(g * keep)

        out._backward = _bw
        return out


def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def zeros(shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad)


def ones(shape, requires_grad: bool = False) -> Tensor:
    return Tensor(np.ones(shape, dtype=np.float32), requires_grad)


def concat(tensors: Iterable[Tensor], axis: int = 0) -> Tensor:
    parts = list(tensors)
    out = Tensor(np.concatenate([t.data for t in parts], axis=axis),
                 any(t.requires_grad for t in parts), parts, "concat")
    sizes = [t.data.shape[axis] for t in parts]

    def _bw(g):
        start = 0
        for t, size in zip(parts, sizes):
            slicer = [slice(None)] * g.ndim
            slicer[axis] = slice(start, start + size)
            t.accumulate(g[tuple(slicer)])
            start += size

    out._backward = _bw
    return out
