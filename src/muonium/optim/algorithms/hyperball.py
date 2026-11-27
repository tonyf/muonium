"""Hyperball optimization: MuonH, NorMuonH and AdamH.

Implements the Hyperball wrapper from Wen, Dang, Lyu, Ma & Liang, "Fantastic
Pretraining Optimizers and Where to Find Them II: Hyperball Optimization" (2026),
https://arxiv.org/abs/2606.16899. Each weight matrix is held on the Frobenius sphere
of its initial radius ``R = ||W_0||_F``: the base update is normalized to unit norm,
applied with step ``lr * R``, and the result projected back to radius ``R``
(Algorithm 1 in the paper). The paper applies it to attention/MLP weight matrices
only; embeddings, norm gains and scalars stay on plain Adam.
"""
import torch
from torch import Tensor

from muonium.orthogonalize import OrthogonalizationStrategy

from .base import (
    BufferSpec,
    MuonAlgorithm,
    orthogonalize_batch,
    orthogonalize_single,
    register_muon_algorithm,
)
from .normuon import NorMuon, _normuon_normalize


def _matrix_sum(x: Tensor) -> Tensor:
    # Per-dim sums rather than sum(dim=(-2, -1)): the fused 2-dim reduction hits an
    # Inductor CPU codegen KeyError on 3D inputs when the result is written back
    # into the reduced param in place.
    return x.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)


def _hyperball_step(
    p: Tensor,
    u: Tensor,
    lr: Tensor,
    split_sizes: tuple[int, ...] | None = None,
) -> Tensor:
    """Step ``p`` by ``lr * R`` along unit-Frobenius ``u``, then project back to radius R.

    The Hyperball step from Algorithm 1 of https://arxiv.org/abs/2606.16899.

    Both inputs are shaped like the (possibly flattened) grad, so norms are per
    trailing matrix. With ``split_sizes``, each row block is projected onto its
    own Frobenius sphere, matching separate per-block parameters. Returns the
    new parameter out-of-place.
    """
    if split_sizes is not None:
        return torch.cat(
            [
                _hyperball_step(p_block, u_block, lr)
                for p_block, u_block in zip(
                    p.split(list(split_sizes), dim=-2),
                    u.split(list(split_sizes), dim=-2),
                    strict=True,
                )
            ],
            dim=-2,
        )
    # ponytail: R is re-read from the current weights instead of stored at init;
    # the projection keeps ||W|| == R, so this only drifts by rounding (bf16 params).
    u = u.to(p.dtype)
    radius_sq = _matrix_sum(p * p)
    radius = radius_sq.sqrt()
    u_norm = _matrix_sum(u * u).sqrt()
    # Zero updates must leave p untouched rather than shrinking by 1/sqrt(1 + lr^2).
    inv_u_norm = torch.where(u_norm > 1e-8, 1.0 / u_norm.clamp(min=1e-8), 0.0)
    u_hat_norm_sq = (u_norm * inv_u_norm).square()
    # ||p - lr R u_hat||^2 expanded without materializing normalized u_hat.
    dot = _matrix_sum(p * u) * inv_u_norm
    w_norm = (
        (radius_sq * (1.0 + lr * lr * u_hat_norm_sq) - 2.0 * lr * radius * dot)
        .clamp(min=1e-16)
        .sqrt()
    )
    p_scale = radius / w_norm
    u_scale = p_scale * lr * radius * inv_u_norm
    return p * p_scale - u * u_scale


@register_muon_algorithm
class MuonHyperball(MuonAlgorithm):
    """MuonH: Muon wrapped in Hyperball (https://arxiv.org/abs/2606.16899).

    Each weight matrix stays on its initial-norm sphere.

    The orthogonalized update is normalized to unit Frobenius norm, applied with
    step ``lr * ||W||_F``, and the result rescaled back to ``||W||_F``. ``lr`` is
    therefore an angular step size; ``lr_ratio`` cancels in the normalization, and
    ``wd`` / ``use_cautious_wd`` are ignored (the projection replaces weight decay).
    Zero-initialized weights have radius 0 and never move: route them elsewhere.
    """

    name = "muonh"

    def direction(
        self,
        u: Tensor,
        state: dict[str, Tensor],
        split_sizes: tuple[int, ...] | None,
        **opts,
    ) -> Tensor:
        """Hook between orthogonalization and the Hyperball step; identity for MuonH."""
        return u

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
        nesterov: bool,
        maximize: bool,
        ns_steps: int,
        orthogonalization_strategy: OrthogonalizationStrategy,
        split_sizes: tuple[int, ...] | None,
        **opts,
    ) -> None:
        momentum_buffer = state["momentum_buffer"]
        if maximize:
            grad = -grad

        grad_fp32 = grad.to(torch.float32)
        # Out-of-place until state copy-back (see MuonBaseline.update).
        momentum_buffer_fp32 = momentum_buffer.to(torch.float32) * momentum + grad_fp32
        momentum_buffer.copy_(momentum_buffer_fp32)

        if nesterov:
            grad_fp32 = grad_fp32 + momentum_buffer_fp32 * momentum
        else:
            grad_fp32 = momentum_buffer_fp32

        u = orthogonalize_single(
            grad_fp32,
            ns_steps=ns_steps,
            strategy=orthogonalization_strategy,
            split_sizes=split_sizes,
            apply_lr_scales=False,
        )
        u = self.direction(u, state, split_sizes, **opts)
        p = param.to(torch.float32).view_as(grad)
        param.copy_(_hyperball_step(p, u, lr, split_sizes=split_sizes).view_as(param))

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
        **opts,
    ) -> None:
        """Batched MuonH update. Assumes all tensors are same shape and device."""
        g = grads
        buf = state["momentum_buffer"]
        if maximize:
            g = list(torch._foreach_neg(g))

        torch._foreach_mul_(buf, momentum)
        torch._foreach_add_(buf, g)

        if nesterov:
            # Out-of-place: avoids mutating p.grad and promotes bf16/fp16 grads to fp32.
            g = list(torch._foreach_add(g, buf, alpha=momentum))
        else:
            g = buf

        u = orthogonalize_batch(
            g,
            ns_steps=ns_steps,
            strategy=orthogonalization_strategy,
            split_sizes=split_sizes,
            apply_lr_scales=False,
        )

        # Per tensor, like NorMuon's normalization: a few reductions that compile
        # fuses, and placement-agnostic for DTensor inputs.
        for i, (p, g_, u_) in enumerate(zip(params, g, u, strict=True)):
            u_ = self.direction(
                u_, {key: buffers[i] for key, buffers in state.items()}, split_sizes, **opts
            )
            p.copy_(
                _hyperball_step(p.to(torch.float32).view_as(g_), u_, lr, split_sizes=split_sizes)
                .view_as(p)
            )


@register_muon_algorithm
class NorMuonHyperball(MuonHyperball):
    """NorMuonH: NorMuon wrapped in Hyperball.

    The NorMuon direction (https://arxiv.org/abs/2510.05491: orthogonalize, then
    normalize each row by a per-neuron second-moment EMA) followed by the Hyperball
    step (https://arxiv.org/abs/2606.16899). NorMuon's Frobenius-norm rescale is
    cancelled by Hyperball's unit normalization, so only the row reweighting
    survives. Same caveats as :class:`MuonHyperball`.
    """

    name = "normuonh"
    options = NorMuon.options
    state_spec = NorMuon.state_spec

    def direction(
        self,
        u: Tensor,
        state: dict[str, Tensor],
        split_sizes: tuple[int, ...] | None,
        muon_beta2: float = 0.95,
        **opts,
    ) -> Tensor:
        variance_neuron = state["variance_neuron"]
        u, v_new = _normuon_normalize(
            u,
            variance_neuron.to(torch.float32),
            muon_beta2,
            split_sizes,
            preserve_norm=False,
        )
        variance_neuron.copy_(v_new)
        return u


@register_muon_algorithm
class AdamHyperball(MuonAlgorithm):
    """AdamH: Adam wrapped in Hyperball (https://arxiv.org/abs/2606.16899).

    Applies the Hyperball projection to the Adam direction ``m / (sqrt(v) + eps)``.
    Same caveats as :class:`MuonHyperball`; ``momentum`` is Adam's beta1.
    Bias correction is skipped: it rescales the direction by a scalar, which the
    unit-norm normalization cancels, leaving only an ``eps * sqrt(1 - beta2^t)``
    shift in the effective epsilon. The default per-tensor foreach loop is kept since
    there is no orthogonalization to batch.
    """

    name = "adamh"
    options = {"beta2": 0.95, "eps": 1e-8}
    state_spec = {
        "exp_avg": BufferSpec(like="grad", signed=True),
        "exp_avg_sq": BufferSpec(like="grad", signed=False),
    }

    def update(
        self,
        param: Tensor,
        grad: Tensor,
        state: dict[str, Tensor],
        lr_ratio: Tensor,
        *,
        lr: Tensor,
        momentum: float,
        maximize: bool,
        split_sizes: tuple[int, ...] | None = None,
        beta2: float = 0.95,
        eps: float = 1e-8,
        **opts,
    ) -> None:
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        if maximize:
            grad = -grad

        grad_fp32 = grad.to(torch.float32)
        # Out-of-place until state copy-back (see MuonBaseline.update).
        m = torch.lerp(exp_avg.to(torch.float32), grad_fp32, 1 - momentum)
        v = torch.lerp(exp_avg_sq.to(torch.float32), grad_fp32 * grad_fp32, 1 - beta2)
        exp_avg.copy_(m)
        exp_avg_sq.copy_(v)

        p = param.to(torch.float32).view_as(grad)
        param.copy_(
            _hyperball_step(p, m / (v.sqrt() + eps), lr, split_sizes=split_sizes).view_as(param)
        )

    def foreach_update(
        self,
        params: list[Tensor],
        grads: list[Tensor],
        state: dict[str, list[Tensor]],
        lr_ratios: list[Tensor],
        *,
        lr: Tensor,
        momentum: float,
        maximize: bool,
        split_sizes: tuple[int, ...] | None = None,
        beta2: float = 0.95,
        eps: float = 1e-8,
        **opts,
    ) -> None:
        """Batched AdamH update. Assumes all tensors are same shape and device."""
        g = [g_.to(torch.float32) for g_ in grads]
        if maximize:
            g = list(torch._foreach_neg(g))

        m = state["exp_avg"]
        v = state["exp_avg_sq"]
        torch._foreach_lerp_(m, g, 1 - momentum)
        g_sq = torch._foreach_mul(g, g)
        torch._foreach_lerp_(v, g_sq, 1 - beta2)

        denom = list(torch._foreach_sqrt(v))
        torch._foreach_add_(denom, eps)
        u = torch._foreach_div(m, denom)

        for p, g_, u_ in zip(params, g, u, strict=True):
            p.copy_(
                _hyperball_step(
                    p.to(torch.float32).view_as(g_), u_, lr, split_sizes=split_sizes
                ).view_as(p)
            )
