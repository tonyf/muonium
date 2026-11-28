"""muonium: a distributed-ready Muon optimizer built on PyTorch DTensor."""

from .optim import (
    AdjustLrFn,
    BufferSpec,
    Muon,
    MuonAlgorithm,
    get_algorithm,
    register_muon_algorithm,
    registered_algorithms,
)
from .orthogonalize import NsCoefficients, OrthogonalizationStrategy

__all__ = [
    "AdjustLrFn",
    "BufferSpec",
    "Muon",
    "MuonAlgorithm",
    "NsCoefficients",
    "OrthogonalizationStrategy",
    "get_algorithm",
    "register_muon_algorithm",
    "registered_algorithms",
]

try:
    from .optim import MuonLP

    __all__ += ["MuonLP"]
except ImportError:
    pass
