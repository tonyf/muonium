"""Pluggable Muon algorithm variants.

Selected per param group via ``{"params": ..., "algorithm": "<name>"}``. Register
your own variant by subclassing :class:`MuonAlgorithm` and decorating it with
:func:`register_muon_algorithm`.
"""

from .base import (
    AdjustLrFn,
    BufferSpec,
    MuonAlgorithm,
    compute_lr_ratio,
    get_algorithm,
    orthogonalize_batch,
    orthogonalize_single,
    register_muon_algorithm,
    registered_algorithms,
    split_lr_scales,
)

# Importing these modules registers the built-in algorithms.
from .dimuon import Dimuon
from .hyperball import AdamHyperball, MuonHyperball, NorMuonHyperball
from .muon import MuonBaseline, MuonC
from .normuon import NorMuon

__all__ = [
    "AdamHyperball",
    "AdjustLrFn",
    "BufferSpec",
    "Dimuon",
    "MuonAlgorithm",
    "MuonBaseline",
    "MuonC",
    "MuonHyperball",
    "NorMuon",
    "NorMuonHyperball",
    "compute_lr_ratio",
    "get_algorithm",
    "orthogonalize_batch",
    "orthogonalize_single",
    "register_muon_algorithm",
    "registered_algorithms",
    "split_lr_scales",
]
