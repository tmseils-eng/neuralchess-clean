"""Optional PyTorch backend for training the same architecture on a GPU.

The NumPy implementation is the reference: it defines the architecture, and
it is what the tests exercise.  This module mirrors it layer for layer in
PyTorch so a long run can be moved to a GPU (CUDA), an Apple-silicon laptop
(MPS) or a cluster node without changing anything else in the project.

Checkpoints are interchangeable in both directions.  The only wrinkle is the
linear layers: this project stores them as ``(in, out)`` while ``torch.nn``
stores ``(out, in)``, so those tensors are transposed on the way across.

PyTorch is an optional dependency; importing this module without it raises a
clear error rather than failing at some later, more confusing point.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..encoding.planes import input_planes
from ..encoding.policy_map import NUM_MOVE_PLANES, POLICY_SIZE
from .modules import NetConfig

try:  # pragma: no cover - exercised only where torch is installed
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None
    nn = object


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "the PyTorch backend needs torch installed: pip install 'neuralchess[torch]'"
        )


def best_device() -> str:
    """CUDA if present, else Apple MPS, else CPU."""
    require_torch()
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if TORCH_AVAILABLE:

    class SqueezeExcitation(nn.Module):
        def __init__(self, channels: int, ratio: int = 4):
            super().__init__()
            hidden = max(4, channels // ratio)
            self.fc1 = nn.Linear(channels, hidden)
            self.fc2 = nn.Linear(hidden, channels * 2)
            self.channels = channels

        def forward(self, x):
            n, c, _, _ = x.shape
            pooled = x.mean(dim=(2, 3))
            both = self.fc2(F.relu(self.fc1(pooled))).view(n, 2, c)
            gate = torch.sigmoid(both[:, 0]).view(n, c, 1, 1)
            bias = both[:, 1].view(n, c, 1, 1)
            return x * gate + bias

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, se_ratio: int = 4, use_se: bool = True):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.se = SqueezeExcitation(channels, se_ratio) if use_se else None

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.se is not None:
                out = self.se(out)
            return F.relu(out + x)

    class TorchPolicyValueNet(nn.Module):
        """The same tower as :class:`neuralchess.nn.modules.PolicyValueNet`."""

        def __init__(self, config: Optional[NetConfig] = None):
            super().__init__()
            self.config = config or NetConfig()
            c = self.config.channels
            self.stem_conv = nn.Conv2d(input_planes(self.config.history_length),
                                       c, 3, padding=1, bias=False)
            self.stem_bn = nn.BatchNorm2d(c)
            self.blocks = nn.ModuleList([
                ResidualBlock(c, self.config.se_ratio, self.config.use_se)
                for _ in range(self.config.blocks)
            ])
            self.policy_conv = nn.Conv2d(c, self.config.policy_channels, 3, padding=1, bias=False)
            self.policy_bn = nn.BatchNorm2d(self.config.policy_channels)
            self.policy_out = nn.Conv2d(self.config.policy_channels, NUM_MOVE_PLANES, 1)
            self.value_conv = nn.Conv2d(c, self.config.value_channels, 1, bias=False)
            self.value_bn = nn.BatchNorm2d(self.config.value_channels)
            self.value_fc1 = nn.Linear(self.config.value_channels * 64, self.config.value_hidden)
            self.value_fc2 = nn.Linear(self.config.value_hidden, 3 if self.config.wdl else 1)

        def forward(self, x):
            n = x.shape[0]
            h = F.relu(self.stem_bn(self.stem_conv(x)))
            for block in self.blocks:
                h = block(h)
            p = F.relu(self.policy_bn(self.policy_conv(h)))
            p = self.policy_out(p)
            policy = p.permute(0, 2, 3, 1).reshape(n, POLICY_SIZE)
            v = F.relu(self.value_bn(self.value_conv(h)))
            v = F.relu(self.value_fc1(v.reshape(n, -1)))
            value = self.value_fc2(v)
            if not self.config.wdl:
                value = torch.tanh(value)
            return policy, value

        # -- inference interface expected by the search --------------------
        @torch.no_grad()
        def predict(self, planes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            device = next(self.parameters()).device
            was_training = self.training
            self.eval()
            x = torch.from_numpy(np.ascontiguousarray(planes)).to(device)
            policy, value = self.forward(x)
            if self.config.wdl:
                probs = torch.softmax(value, dim=-1)
                scalar = probs[:, 0] - probs[:, 2]
            else:
                scalar = value.reshape(-1)
            self.train(was_training)
            return (policy.cpu().numpy().astype(np.float32),
                    scalar.cpu().numpy().astype(np.float32))

        def describe(self) -> str:
            total = sum(p.numel() for p in self.parameters())
            c = self.config
            return (f"TorchPolicyValueNet({c.blocks}x{c.channels}"
                    f"{'+SE' if c.use_se else ''}, {total:,} parameters)")

        # -- checkpoint interchange ---------------------------------------
        def save(self, path: str) -> None:
            """Write a checkpoint the NumPy backend can load."""
            state = {}
            for name, tensor in self.state_dict().items():
                array = tensor.detach().cpu().numpy()
                if name.endswith(".weight") and array.ndim == 2:
                    array = array.T                       # (out, in) -> (in, out)
                if "running_mean" in name or "running_var" in name:
                    state[f"buffer::{name}"] = array
                elif "num_batches_tracked" in name:
                    continue
                else:
                    state[name] = array
            np.savez_compressed(path, __config__=np.array(str(self.config.to_dict())), **state)

        @classmethod
        def load(cls, path: str, device: Optional[str] = None) -> "TorchPolicyValueNet":
            """Load a checkpoint written by either backend."""
            import ast

            data = np.load(path, allow_pickle=False)
            config = NetConfig.from_dict(ast.literal_eval(str(data["__config__"])))
            model = cls(config)
            target = model.state_dict()
            loaded = {}
            for key in data.files:
                if key == "__config__":
                    continue
                name = key[len("buffer::"):] if key.startswith("buffer::") else key
                if name not in target:
                    continue
                array = data[key]
                if name.endswith(".weight") and array.ndim == 2:
                    array = array.T
                loaded[name] = torch.from_numpy(np.ascontiguousarray(array)).to(
                    target[name].dtype)
            # strict=False: a checkpoint from the NumPy backend carries no
            # num_batches_tracked buffers, and their absence is expected.
            model.load_state_dict(loaded, strict=False)
            model.to(device or best_device())
            return model

    def from_numpy_model(model) -> "TorchPolicyValueNet":
        """Copy a NumPy :class:`PolicyValueNet` into a torch module."""
        require_torch()
        torch_model = TorchPolicyValueNet(model.config)
        state = model.state_dict()
        target = torch_model.state_dict()
        loaded = {}
        for key, array in state.items():
            name = key[len("buffer::"):] if key.startswith("buffer::") else key
            if name not in target:
                continue
            if name.endswith(".weight") and array.ndim == 2:
                array = array.T
            loaded[name] = torch.from_numpy(np.ascontiguousarray(array)).to(target[name].dtype)
        torch_model.load_state_dict(loaded, strict=False)
        return torch_model
