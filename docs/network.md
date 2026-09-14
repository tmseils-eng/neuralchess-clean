# The network, and the autograd engine under it

## Architecture

```
input 119×8×8
   └─ 3×3 conv → batchnorm → ReLU                          (stem)
   └─ N × residual block:
          3×3 conv → BN → ReLU → 3×3 conv → BN
          → squeeze-excitation gate
          → + skip → ReLU
   ├─ policy head:  3×3 conv → BN → ReLU → 1×1 conv → 73 planes → 4,672 logits
   └─ value head:   1×1 conv → BN → ReLU → FC(128) → ReLU → FC(3)  [W/D/L]
```

Two departures from the 2017 AlphaZero paper, both deliberate:

**Squeeze-and-excitation blocks.** Each residual block ends with a global
average pool, a two-layer bottleneck, and a per-channel gate *and* bias
(the Leela Chess Zero variant). A 3×3 convolution stack can only propagate
information about material balance or king safety at one square per layer;
an SE gate lets a whole feature map be modulated by global context in a
single step. It costs a few percent more parameters and consistently earns
them back.

**A win/draw/loss head instead of a scalar.** Chess draws constantly, and a
scalar value trained on `z ∈ {−1, 0, +1}` collapses "certainly drawn" and
"equally likely to win or lose" onto the same output. Three softmax outputs
keep them apart; the search uses `P(win) − P(loss)` as its scalar, so nothing
downstream changes, but the head can now express drawishness. Set
`wdl: false` for the original scalar-with-`tanh` formulation.

Sizes used in this repository:

| config | blocks × channels | parameters |
| --- | --- | --- |
| `configs/tiny.json` | 4 × 32 | 399K |
| `configs/laptop.json` | 5 × 48 | 566K |
| `configs/cluster.json` | 10 × 128 | 3.6M |

## Automatic differentiation from scratch

`neuralchess/nn/tensor.py` is a small reverse-mode autograd engine. Every
`Tensor` records the parents it came from and a closure that pushes gradient
back to them; `backward()` topologically sorts that graph and runs the
closures in reverse. Broadcasting is undone explicitly on the way back, which
is the one place a hand-rolled engine usually goes quietly wrong.

Convolution uses the `im2col` reformulation: patches are gathered into a
matrix so the forward pass is a single `einsum`, and both gradients are the
same contraction with different indices free. On an 8×8 board this is fast
enough that the search, not the arithmetic, is the bottleneck.

Batch normalisation implements the full training-mode backward pass
(including the mean and variance paths, which are what people usually drop),
and keeps running statistics for inference.

Every one of these is checked against a central-difference numerical
gradient in `tests/test_autograd.py`, to a relative tolerance of 10⁻³:

```
conv2d      dx  4.99e-05   dW  4.60e-05   db  2.09e-05
batchnorm   dx  1.44e-04
log_softmax     < 1e-3
```

`no_grad()` disables graph construction entirely, which is what the search
runs under: without it, every leaf evaluation would retain its activations
until the tree was discarded.

### One subtlety that cost a training run

The natural way to write a backward closure is to have it read `out.grad` from
the tensor it belongs to:

```python
def _bw():
    if out.grad is None:
        return
    parent.accumulate(out.grad * mask)
out._backward = _bw          # out._backward -> closure -> out
```

That last line is a reference cycle, and Python can only reclaim it through
the cyclic collector — which triggers on object *counts*, not bytes. A
training step allocates a few hundred small `Tensor` objects holding a few
hundred megabytes of activations, so the collector sees nothing alarming while
resident memory climbs by ~100 MB per step. The first long run in this
repository was killed by the OOM reaper at 6 GB after two iterations.

The fix is to pass the gradient in as an argument:

```python
def _bw(g):
    parent.accumulate(g * mask)
out._backward = _bw          # every reference now points child -> parent
```

Every reference then points strictly from child to parent, so the whole graph
is freed by refcounting the moment the loss goes out of scope. Memory during
training is now flat at ~630 MB for a 5×48 network at batch 96.
`tests/test_autograd.py` pins the invariant with a weak reference taken while
the cyclic collector is disabled.

## PyTorch backend

`neuralchess/nn/torch_backend.py` mirrors the architecture layer for layer in
PyTorch so a long run can move to CUDA, Apple MPS, or a cluster GPU.
Checkpoints are interchangeable in both directions — the only fixup is that
linear weights are stored `(in, out)` here and `(out, in)` by `torch.nn`, so
they are transposed on the way across. PyTorch remains an optional
dependency; nothing in the tests or the default training path needs it.
