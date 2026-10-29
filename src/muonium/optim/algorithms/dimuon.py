"""Dimuon: Muon with two-sided diagonal preconditioning of the momentum.

Port of the update rule in https://github.com/catid/dimuon (``dimuon.py``). Per
weight matrix, row and column conditioners ``L = (e_row / max(e_row) + damping)^-1/4``
(and likewise ``R`` for columns) are built from the *previous* step's momentum energy.
The new EMA momentum is preconditioned to ``L M R``, orthogonalized, mapped back with
the same conditioners to ``L O R``, normalized to unit Frobenius norm, and applied with
step ``lr * sqrt(r) * (r / 32)^0.175`` where ``r = min(rows, cols)``.
"""

import math

import torch
from torch import Tensor

from muonium.orthogonalize import OrthogonalizationStrategy

from .base import (
    MuonAlgorithm,
    orthogonalize_batch,
    orthogonalize_single,
    register_muon_algorithm,
)
from .hyperball import _matrix_sum


def _conditioner(energy: Tensor, dim: int, damping: float) -> Tensor:
    """``(energy / max(energy) + damping)^-1/4`` along ``dim``; identity when all zero."""
    peak = energy.amax(dim=dim, keepdim=True)
    cond = (energy / peak.clamp(min=torch.finfo(energy.dtype).tiny) + damping) ** -0.25
    return torch.where(peak > 0, cond, torch.ones_like(cond))


def _dimuon_conditioners(m: Tensor, damping: float) -> tuple[Tensor, Tensor]:
    """Row (``[..., rows, 1]``) and column (``[..., 1, cols]``) conditioners of ``m``."""
    energy = m * m
    left = _conditioner(energy.sum(dim=-1, keepdim=True), -2, damping)
    right = _conditioner(energy.sum(dim=-2, keepdim=True), -1, damping)
    return left, right


def _dimuon_scale(u: Tensor, left: Tensor, right: Tensor) -> Tensor:
    """Map orthogonalized ``u`` back through the conditioners and apply the rank step size.

    Zero momentum orthogonalizes to zero and stays zero through the clamped norm.
    """
    d = left * u.to(left.dtype) * right
    rank = min(d.shape[-2], d.shape[-1])
    target = math.sqrt(rank) * (rank / 32) ** 0.175
    return d * (target / _matrix_sum(d * d).sqrt().clamp(min=1e-30))


def _dimuon_direction(
    old: Tensor,
    m_new: Tensor,
    damping: float,
    *,
    ns_steps: int,
    strategy: OrthogonalizationStrategy,
    split_sizes: tuple[int, ...] | None = None,
) -> Tensor:
    """Per-tensor Dimuon update direction (before ``lr`` and weight decay).

    With ``split_sizes``, each row block is conditioned, orthogonalized and
    rank-scaled on its own, matching the update separate parameters would get.
    """
    if split_sizes is not None:
        return torch.cat(
            [
                _dimuon_direction(o, m, damping, ns_steps=ns_steps, strategy=strategy)
                for o, m in zip(
                    old.split(list(split_sizes), dim=-2),
                    m_new.split(list(split_sizes), dim=-2),
                    strict=True,
                )
            ],
            dim=-2,
        )
    left, right = _dimuon_conditioners(old, damping)
    u = orthogonalize_single(left * m_new * right, ns_steps=ns_steps, strategy=strategy)
    return _dimuon_scale(u, left, right)


def _apply(param: Tensor, d: Tensor, lr: Tensor, weight_decay: float, cautious_wd: bool):
    p = param.to(torch.float32)
    d = d.view_as(p)
    if weight_decay != 0:
        if cautious_wd:
            d = d + weight_decay * p * (d * p > 0)
        else:
            d = d + weight_decay * p
    param.copy_(p - lr * d)


@register_muon_algorithm
class Dimuon(MuonAlgorithm):
    """Dimuon (https://github.com/catid/dimuon): diagonally preconditioned Muon.

    ``momentum`` is the EMA coefficient (beta1; the reference uses 0.9) and
    ``nesterov`` is ignored. The step size ``lr * sqrt(r) * (r / 32)^0.175``
    replaces the shape-based ``lr_ratio``. ``dimuon_damping`` floors the
    normalized row/column energies before the ``-1/4`` power. Weight decay is not
    part of the reference; it is applied as in Muon (``lr * wd * p``) and can be
    disabled with ``wd=0``.
    """

    name = "dimuon"
    options = {"dimuon_damping": 1e-4}

    def update(
        self,
        param: Tensor,
        grad: Tensor,
        state: dict[str, Tensor],
        lr_ratio: Tensor,
        *,
        lr: Tensor,
        weight_decay: float,
        cautious_wd: bool,
        momentum: float,
        maximize: bool,
        ns_steps: int,
        orthogonalization_strategy: OrthogonalizationStrategy,
        split_sizes: tuple[int, ...] | None,
        dimuon_damping: float = 1e-4,
        **opts,
    ) -> None:
        momentum_buffer = state["momentum_buffer"]
        if maximize:
            grad = -grad

        # Out-of-place until state copy-back (see MuonBaseline.update).
        old = momentum_buffer.to(torch.float32)
        m_new = torch.lerp(old, grad.to(torch.float32), 1 - momentum)
        momentum_buffer.copy_(m_new)

        d = _dimuon_direction(
            old,
            m_new,
            dimuon_damping,
            ns_steps=ns_steps,
            strategy=orthogonalization_strategy,
            split_sizes=split_sizes,
        )
        _apply(param, d, lr, weight_decay, cautious_wd)

    def foreach_update(
        self,
        params: list[Tensor],
        grads: list[Tensor],
        state: dict[str, list[Tensor]],
        lr_ratios: list[Tensor],
        *,
        lr: Tensor,
        weight_decay: float,
        cautious_wd: bool,
        momentum: float,
        nesterov: bool,
        maximize: bool,
        ns_steps: int,
        orthogonalization_strategy: OrthogonalizationStrategy,
        split_sizes: tuple[int, ...] | None,
        dimuon_damping: float = 1e-4,
        **opts,
    ) -> None:
        """Batched Dimuon update. Assumes all tensors are same shape and device."""
        if split_sizes is not None:
            # Per-block conditioning doesn't batch; use the per-tensor reference.
            return super().foreach_update(
                params,
                grads,
                state,
                lr_ratios,
                lr=lr,
                weight_decay=weight_decay,
                cautious_wd=cautious_wd,
                momentum=momentum,
                nesterov=nesterov,
                maximize=maximize,
                ns_steps=ns_steps,
                orthogonalization_strategy=orthogonalization_strategy,
                split_sizes=split_sizes,
                dimuon_damping=dimuon_damping,
                **opts,
            )

        g = grads
        buf = state["momentum_buffer"]
        if maximize:
            g = list(torch._foreach_neg(g))

        # Conditioners come from the momentum *before* this step's EMA update.
        conds = [_dimuon_conditioners(b, dimuon_damping) for b in buf]
        torch._foreach_lerp_(buf, g, 1 - momentum)

        u = orthogonalize_batch(
            [left * b * right for b, (left, right) in zip(buf, conds, strict=True)],
            ns_steps=ns_steps,
            strategy=orthogonalization_strategy,
        )

        # Per tensor, like NorMuon's normalization: a few reductions that compile
        # fuses, and placement-agnostic for DTensor inputs.
        for p, u_, (left, right) in zip(params, u, conds, strict=True):
            _apply(p, _dimuon_scale(u_, left, right), lr, weight_decay, cautious_wd)
