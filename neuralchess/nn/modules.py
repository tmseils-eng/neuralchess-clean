"""Network layers and the AlphaZero-style policy-value architecture.

The tower follows the AlphaZero recipe with one modern addition borrowed
from Leela Chess Zero: a squeeze-and-excitation gate at the end of every
residual block, which lets the network modulate whole feature maps from
global board context (material balance, king safety) rather than from the
3x3 neighbourhood alone.  On small networks this is a consistently good
trade: a few percent more parameters for a noticeably better policy.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..encoding.planes import HISTORY_LENGTH, input_planes
from ..encoding.policy_map import NUM_MOVE_PLANES, POLICY_SIZE
from . import functional as F
from .tensor import Tensor, no_grad


class Module:
    """Minimal PyTorch-shaped base class: parameters, buffers, train/eval."""

    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}
        self._buffers: Dict[str, np.ndarray] = {}
        self._modules: Dict[str, "Module"] = {}
        self.training = True

    # -- registration ---------------------------------------------------
    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            self.__dict__.setdefault("_parameters", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        object.__setattr__(self, name, value)

    def register_buffer(self, name: str, array: np.ndarray) -> None:
        self._buffers[name] = array
        object.__setattr__(self, name, array)

    # -- traversal ------------------------------------------------------
    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Tensor]]:
        for name, param in self._parameters.items():
            yield f"{prefix}{name}", param
        for name, module in self._modules.items():
            yield from module.named_parameters(f"{prefix}{name}.")

    def parameters(self) -> List[Tensor]:
        return [p for _, p in self.named_parameters()]

    def named_buffers(self, prefix: str = "") -> Iterator[Tuple[str, np.ndarray]]:
        for name, buf in self._buffers.items():
            yield f"{prefix}{name}", buf
        for name, module in self._modules.items():
            yield from module.named_buffers(f"{prefix}{name}.")

    def modules(self) -> Iterator["Module"]:
        yield self
        for module in self._modules.values():
            yield from module.modules()

    # -- state ----------------------------------------------------------
    def train(self, mode: bool = True) -> "Module":
        for module in self.modules():
            object.__setattr__(module, "training", mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    def state_dict(self) -> Dict[str, np.ndarray]:
        state = {name: param.data.copy() for name, param in self.named_parameters()}
        state.update({f"buffer::{name}": buf.copy() for name, buf in self.named_buffers()})
        return state

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        params = dict(self.named_parameters())
        for name, param in params.items():
            if name in state:
                param.data = np.asarray(state[name], dtype=np.float32).reshape(param.data.shape)
        for name, buf in self.named_buffers():
            key = f"buffer::{name}"
            if key in state:
                buf[...] = np.asarray(state[key], dtype=np.float32).reshape(buf.shape)

    def num_parameters(self) -> int:
        return sum(int(p.data.size) for p in self.parameters())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Primitive layers
# ---------------------------------------------------------------------------

class Conv2d(Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 padding: Optional[int] = None, bias: bool = False):
        super().__init__()
        self.kernel = kernel
        self.padding = kernel // 2 if padding is None else padding
        fan_in = in_ch * kernel * kernel
        scale = math.sqrt(2.0 / fan_in)          # He initialisation for ReLU nets
        self.weight = Tensor(
            np.random.randn(out_ch, in_ch, kernel, kernel).astype(np.float32) * scale,
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_ch, dtype=np.float32), requires_grad=True) if bias else None
        if self.bias is None:
            self._parameters.pop("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, self.bias, padding=self.padding)


class BatchNorm2d(Module):
    def __init__(self, channels: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = Tensor(np.ones(channels, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(np.zeros(channels, dtype=np.float32), requires_grad=True)
        self.register_buffer("running_mean", np.zeros(channels, dtype=np.float32))
        self.register_buffer("running_var", np.ones(channels, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        return F.batch_norm(x, self.weight, self.bias, self.running_mean, self.running_var,
                            self.training, self.momentum, self.eps)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        scale = math.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features).astype(np.float32) * scale,
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True) if bias else None
        if self.bias is None:
            self._parameters.pop("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weight)
        return out + self.bias if self.bias is not None else out


class SqueezeExcitation(Module):
    """Global gate: pool the board, predict a per-channel scale and bias.

    Uses the Leela variant that emits *both* a multiplicative gate and an
    additive bias, which empirically converges faster than the classic
    scale-only formulation.
    """

    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        hidden = max(4, channels // ratio)
        self.fc1 = Linear(channels, hidden)
        self.fc2 = Linear(hidden, channels * 2)
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        pooled = x.mean(axis=(2, 3))                       # (N, C)
        hidden = self.fc1(pooled).relu()
        both = self.fc2(hidden).reshape(n, 2, c)           # gate | bias
        gate_t = _slice_channels(both, 0).sigmoid().reshape(n, c, 1, 1)
        bias_t = _slice_channels(both, 1).reshape(n, c, 1, 1)
        return x * gate_t + bias_t


def _slice_channels(t: Tensor, index: int) -> Tensor:
    """Differentiable ``t[:, index, :]`` for a ``(N, 2, C)`` tensor."""
    out = Tensor(t.data[:, index], t.requires_grad, (t,), "slice")

    def _bw(grad):
        g = np.zeros_like(t.data)
        g[:, index] = grad
        t.accumulate(g)

    out._backward = _bw
    return out


class ResidualBlock(Module):
    def __init__(self, channels: int, se_ratio: int = 4, use_se: bool = True):
        super().__init__()
        self.conv1 = Conv2d(channels, channels, 3)
        self.bn1 = BatchNorm2d(channels)
        self.conv2 = Conv2d(channels, channels, 3)
        self.bn2 = BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, se_ratio) if use_se else None
        if self.se is None:
            self._modules.pop("se", None)

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        return (out + x).relu()


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

@dataclass
class NetConfig:
    """Hyper-parameters of the policy-value network."""

    channels: int = 64
    blocks: int = 6
    se_ratio: int = 4
    use_se: bool = True
    value_hidden: int = 128
    value_channels: int = 32
    policy_channels: int = 64
    wdl: bool = True          # 3-way win/draw/loss head instead of a scalar
    # How many past positions the input encodes.  Stored on the checkpoint
    # because the input width depends on it: 8 steps is 119 planes, 2 is 35.
    # Older checkpoints predate the field and default to AlphaZero's 8.
    history_length: int = HISTORY_LENGTH

    @property
    def input_planes(self) -> int:
        return input_planes(self.history_length)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NetConfig":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


class PolicyValueNet(Module):
    """Residual tower with a 4672-way policy head and a WDL value head."""

    def __init__(self, config: Optional[NetConfig] = None):
        super().__init__()
        self.config = config or NetConfig()
        c = self.config.channels

        self.stem_conv = Conv2d(self.config.input_planes, c, 3)
        self.stem_bn = BatchNorm2d(c)
        self.blocks = [
            ResidualBlock(c, self.config.se_ratio, self.config.use_se)
            for _ in range(self.config.blocks)
        ]
        for i, block in enumerate(self.blocks):
            self._modules[f"block{i}"] = block

        self.policy_conv = Conv2d(c, self.config.policy_channels, 3)
        self.policy_bn = BatchNorm2d(self.config.policy_channels)
        self.policy_out = Conv2d(self.config.policy_channels, NUM_MOVE_PLANES, 1, bias=True)

        self.value_conv = Conv2d(c, self.config.value_channels, 1)
        self.value_bn = BatchNorm2d(self.config.value_channels)
        self.value_fc1 = Linear(self.config.value_channels * 64, self.config.value_hidden)
        self.value_fc2 = Linear(self.config.value_hidden, 3 if self.config.wdl else 1)

    # -- forward --------------------------------------------------------
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        n = x.shape[0]
        h = self.stem_bn(self.stem_conv(x)).relu()
        for block in self.blocks:
            h = block(h)

        p = self.policy_bn(self.policy_conv(h)).relu()
        p = self.policy_out(p)                              # (N, 73, 8, 8)
        # Flatten as square-major so index == from_square * 73 + plane.
        policy = p.transpose(0, 2, 3, 1).reshape(n, POLICY_SIZE)

        v = self.value_bn(self.value_conv(h)).relu()
        v = v.reshape(n, self.config.value_channels * 64)
        v = self.value_fc1(v).relu()
        value = self.value_fc2(v)
        if not self.config.wdl:
            value = value.tanh()
        return policy, value

    # -- inference helpers ----------------------------------------------
    @no_grad()
    def predict(self, planes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch inference. Returns ``(policy_logits, scalar_value)``.

        ``scalar_value`` is in ``[-1, 1]`` from the mover's point of view; for
        a WDL head it is ``P(win) - P(loss)``.
        """
        was_training = self.training
        self.eval()
        try:
            policy, value = self.forward(Tensor(planes))
            if self.config.wdl:
                probs = F.softmax_np(value.data, axis=-1)
                scalar = probs[:, 0] - probs[:, 2]
            else:
                scalar = value.data.reshape(-1)
            return policy.data, scalar.astype(np.float32)
        finally:
            self.train(was_training)

    @no_grad()
    def predict_wdl(self, planes: np.ndarray) -> np.ndarray:
        was_training = self.training
        self.eval()
        try:
            _, value = self.forward(Tensor(planes))
            if self.config.wdl:
                return F.softmax_np(value.data, axis=-1)
            v = value.data.reshape(-1)
            return np.stack([(1 + v) / 2, np.zeros_like(v), (1 - v) / 2], axis=-1)
        finally:
            self.train(was_training)

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        state = self.state_dict()
        np.savez_compressed(path, __config__=np.array(str(self.config.to_dict())), **state)

    @classmethod
    def load(cls, path: str) -> "PolicyValueNet":
        data = np.load(path, allow_pickle=False)
        import ast

        config = NetConfig.from_dict(ast.literal_eval(str(data["__config__"])))
        model = cls(config)
        model.load_state_dict({k: data[k] for k in data.files if k != "__config__"})
        return model

    def describe(self) -> str:
        c = self.config
        return (
            f"PolicyValueNet({c.blocks}x{c.channels}"
            f"{'+SE' if c.use_se else ''}, "
            f"{'WDL' if c.wdl else 'scalar'} value, "
            f"T={c.history_length} ({c.input_planes} planes), "
            f"{self.num_parameters():,} parameters)"
        )
