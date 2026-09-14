"""Numerical gradient checks for the hand-written autograd engine."""

import numpy as np

from neuralchess.nn import NetConfig, PolicyValueNet
from neuralchess.nn.functional import batch_norm, conv2d, cross_entropy_soft
from neuralchess.nn.modules import ResidualBlock, SqueezeExcitation
from neuralchess.nn.optim import Adam
from neuralchess.nn.tensor import Tensor, no_grad


def numeric_gradient(fn, x, eps=1e-2):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        original = x[i]
        x[i] = original + eps
        plus = fn(x)
        x[i] = original - eps
        minus = fn(x)
        x[i] = original
        grad[i] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a, b):
    return np.abs(a - b).max() / max(1e-9, np.abs(b).max())


def test_elementwise_gradients():
    rng = np.random.default_rng(0)
    cases = {
        "square": lambda t: (t * t).sum(),
        "relu": lambda t: (t.relu() * 3.0).sum(),
        "tanh": lambda t: (t.tanh() ** 2).sum(),
        "sigmoid": lambda t: t.sigmoid().sum(),
        "div": lambda t: (t / (t * t + 5.0)).sum(),
    }
    for name, fn in cases.items():
        x = rng.standard_normal((3, 4)).astype(np.float32)
        node = Tensor(x.copy(), requires_grad=True)
        fn(node).backward()
        numeric = numeric_gradient(
            lambda z, fn=fn: float(fn(Tensor(z)).data.sum()), x.copy())
        assert relative_error(node.grad, numeric) < 1e-3, name


def test_log_softmax_gradient():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 7)).astype(np.float32)
    weights = rng.random((4, 7)).astype(np.float32)

    def loss(arr):
        return (Tensor(arr).log_softmax(-1) * Tensor(weights)).sum()

    node = Tensor(x.copy(), requires_grad=True)
    (node.log_softmax(-1) * Tensor(weights)).sum().backward()
    numeric = numeric_gradient(lambda z: float(loss(z).data), x.copy())
    assert relative_error(node.grad, numeric) < 1e-3


def test_conv2d_gradients():
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((2, 3, 5, 5)) * 0.5).astype(np.float32)
    w = (rng.standard_normal((4, 3, 3, 3)) * 0.3).astype(np.float32)
    b = (rng.standard_normal(4) * 0.1).astype(np.float32)

    def build(xv=None, wv=None, bv=None):
        xt = Tensor(x if xv is None else xv, requires_grad=True)
        wt = Tensor(w if wv is None else wv, requires_grad=True)
        bt = Tensor(b if bv is None else bv, requires_grad=True)
        out = conv2d(xt, wt, bt, padding=1)
        return (out * out).sum(), xt, wt, bt

    loss, xt, wt, bt = build()
    loss.backward()
    for key, array, node in (("xv", x, xt), ("wv", w, wt), ("bv", b, bt)):
        numeric = numeric_gradient(
            lambda z, key=key: float(build(**{key: z})[0].data), array.copy())
        assert relative_error(node.grad, numeric) < 1e-3, key


def test_batch_norm_gradient():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((4, 3, 4, 4)) * 0.7).astype(np.float32)
    coeff = rng.standard_normal((4, 3, 4, 4)).astype(np.float32)

    def build(arr):
        xt = Tensor(arr, requires_grad=True)
        gamma = Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
        beta = Tensor(np.zeros(3, dtype=np.float32), requires_grad=True)
        out = batch_norm(xt, gamma, beta, np.zeros(3, dtype=np.float32),
                         np.ones(3, dtype=np.float32), True)
        return (out * Tensor(coeff)).sum(), xt

    loss, node = build(x.copy())
    loss.backward()
    numeric = numeric_gradient(lambda z: float(build(z)[0].data), x.copy())
    assert relative_error(node.grad, numeric) < 5e-3


def test_squeeze_excitation_and_residual_shapes():
    rng = np.random.default_rng(4)
    x = Tensor(rng.standard_normal((2, 8, 8, 8)).astype(np.float32), requires_grad=True)
    assert SqueezeExcitation(8)(x).shape == (2, 8, 8, 8)
    block = ResidualBlock(8)
    out = block(x)
    assert out.shape == (2, 8, 8, 8)
    out.sum().backward()
    assert x.grad is not None and np.isfinite(x.grad).all()


def test_no_grad_frees_the_graph():
    x = Tensor(np.ones((2, 2)), requires_grad=True)
    with no_grad():
        y = (x * x).sum()
    assert y._parents == () and not y.requires_grad


def test_network_can_overfit_a_single_batch():
    np.random.seed(0)
    model = PolicyValueNet(NetConfig(channels=16, blocks=1))
    optimizer = Adam(model.parameters(), lr=4e-3)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 119, 8, 8)).astype(np.float32)
    target = np.zeros((4, 4672), dtype=np.float32)
    indices = rng.integers(0, 4672, 4)
    target[np.arange(4), indices] = 1.0

    first = None
    for _ in range(60):
        policy, _ = model(Tensor(x))
        loss = cross_entropy_soft(policy, target)
        first = first if first is not None else loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logits, _ = model.predict(x)
    assert loss.item() < first * 0.1
    assert (logits.argmax(axis=1) == indices).all()


def test_checkpoint_round_trip(tmp_path=None):
    import os
    import tempfile

    model = PolicyValueNet(NetConfig(channels=16, blocks=2, wdl=True))
    x = np.random.RandomState(0).randn(2, 119, 8, 8).astype(np.float32)
    before = model.predict(x)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "net.npz")
        model.save(path)
        restored = PolicyValueNet.load(path)
    after = restored.predict(x)
    assert np.allclose(before[0], after[0], atol=1e-5)
    assert np.allclose(before[1], after[1], atol=1e-5)
    assert restored.config.channels == 16 and restored.config.blocks == 2


def test_graph_is_freed_without_gc():
    """Backward graphs must be reclaimable by refcounting alone.

    If a tensor's backward closure captures the tensor it belongs to, the pair
    forms a reference cycle.  Python's cyclic collector triggers on object
    counts rather than bytes, so a training loop then accumulates whole
    computation graphs - hundreds of megabytes each - and is eventually killed
    by the OOM reaper.  This test pins the invariant that made that bug
    possible.
    """
    import gc
    import weakref

    gc.collect()
    gc.disable()
    try:
        x = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32), requires_grad=True)
        w = Tensor(np.random.randn(2, 3, 3, 3).astype(np.float32), requires_grad=True)
        hidden = conv2d(x, w, padding=1).relu()
        loss = (hidden * hidden).sum()
        ref = weakref.ref(hidden)
        loss.backward()
        assert x.grad is not None
        del hidden, loss
        assert ref() is None, "the intermediate tensor outlived its graph"
    finally:
        gc.enable()


def test_repeated_steps_do_not_accumulate_tensors():
    """Sixty optimiser steps must not leave sixty graphs behind."""
    import gc

    np.random.seed(1)
    model = PolicyValueNet(NetConfig(channels=8, blocks=1))
    optimizer = Adam(model.parameters(), lr=1e-3)
    x = np.random.randn(4, 119, 8, 8).astype(np.float32)
    target = np.zeros((4, 4672), dtype=np.float32)
    target[:, 0] = 1.0

    def live_tensors():
        gc.collect()
        return sum(1 for obj in gc.get_objects() if isinstance(obj, Tensor))

    for _ in range(3):
        policy, _ = model(Tensor(x))
        loss = cross_entropy_soft(policy, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    baseline = live_tensors()

    for _ in range(30):
        policy, _ = model(Tensor(x))
        loss = cross_entropy_soft(policy, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    del policy, loss
    assert live_tensors() <= baseline + 5


def test_adamw_decay_scales_with_the_decay_setting():
    """Under AdamW the shrinkage is exactly ``lr * wd * p`` per step.

    Adam-with-L2 folds ``wd * p`` into the gradient, where the adaptive
    denominator immediately normalises it away: with no other gradient signal
    the update is ~``lr`` per step whatever ``weight_decay`` is set to.  So
    raising the setting tenfold barely changes Adam's behaviour while it
    changes AdamW's by the same factor.  That ratio is the discriminating
    property, and it is why the setting means what you expect only in AdamW.
    """
    from neuralchess.nn.optim import Adam, AdamW

    def shrinkage(cls, weight_decay, steps=100):
        w = Tensor(np.ones((4,), dtype=np.float32), requires_grad=True)
        opt = cls([w], lr=1e-3, weight_decay=weight_decay)
        for _ in range(steps):
            w.grad = np.zeros((4,), dtype=np.float32)
            opt.step()
        return 1.0 - float(w.data[0])

    decoupled = shrinkage(AdamW, 0.1) / shrinkage(AdamW, 0.01)
    coupled = shrinkage(Adam, 0.1) / shrinkage(Adam, 0.01)
    assert decoupled > 5.0, f"AdamW decay should scale with wd, got {decoupled:.2f}x"
    assert coupled < 2.0, f"Adam+L2 decay should be flattened by adaptation, got {coupled:.2f}x"


def test_adamw_is_selectable_by_name():
    from neuralchess.nn.optim import AdamW, build_optimizer

    w = Tensor(np.ones((2,), dtype=np.float32), requires_grad=True)
    assert isinstance(build_optimizer([w], "adamw", lr=1e-3), AdamW)
