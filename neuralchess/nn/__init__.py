"""Neural-network backend.

The NumPy backend in this package is self-contained and is what the default
configuration trains with.  When PyTorch is installed the equivalent model in
:mod:`neuralchess.nn.torch_backend` can be selected instead to train the same
architecture on a GPU or Apple-silicon MPS device; both save and load the same
``.npz`` checkpoint format, so a network trained on a cluster can be played
against here without conversion.
"""

from .modules import (
    BatchNorm2d,
    Conv2d,
    Linear,
    Module,
    NetConfig,
    PolicyValueNet,
    ResidualBlock,
    SqueezeExcitation,
)
from .optim import SGD, Adam, AdamW, StepLR, WarmupCosineLR, build_optimizer
from .tensor import Tensor, no_grad

__all__ = [
    "Tensor", "no_grad", "Module", "Conv2d", "BatchNorm2d", "Linear",
    "SqueezeExcitation", "ResidualBlock", "PolicyValueNet", "NetConfig",
    "SGD", "Adam", "AdamW", "StepLR", "WarmupCosineLR", "build_optimizer",
]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True
