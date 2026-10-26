"""Pluggable Muon algorithm variants.

Selected per param group via ``{"params": ..., "algorithm": "<name>"}``. Register
your own variant by subclassing :class:`MuonAlgorithm` and decorating it with
:func:`register_muon_algorithm`.
"""

from .base import (
    BufferSpec,
    MuonAlgorithm,
    get_algorithm,
    orthogonalize_batch,
    orthogonalize_single,
    register_muon_algorithm,
    registered_algorithms,
    split_lr_scales,
)

# Importing these modules registers the built-in algorithms.
from .hyperball import AdamHyperball, MuonHyperball, NorMuonHyperball
from .muon import MuonBaseline
from .muon2 import Muon2, Muon2F
from .normuon import NorMuon

__all__ = [
    "AdamHyperball",
    "BufferSpec",
    "Muon2",
    "Muon2F",
    "MuonAlgorithm",
    "MuonBaseline",
    "MuonHyperball",
    "NorMuon",
    "NorMuonHyperball",
    "get_algorithm",
    "orthogonalize_batch",
    "orthogonalize_single",
    "register_muon_algorithm",
    "registered_algorithms",
    "split_lr_scales",
]
