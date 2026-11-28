# Baseline Muon update rule (Keller Jordan's Muon), per-tensor and foreach forms.
import torch
from torch import Tensor

from muonium.orthogonalize import OrthogonalizationStrategy

from .base import (
    AdjustLrFn,
    MuonAlgorithm,
    orthogonalize_batch,
    orthogonalize_single,
    register_muon_algorithm,
)


@register_muon_algorithm
class MuonBaseline(MuonAlgorithm):
    """The original Muon update: momentum, orthogonalize, shape-scaled apply."""

    name = "muon"

    def weight_decay_scale(self, lr: Tensor, **opts) -> Tensor | None:
        """Extra multiplier on ``weight_decay`` (``None`` = 1); see :class:`MuonC`."""
        return None

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
        adjust_lr_fn: AdjustLrFn = "original",
        **opts,
    ) -> None:
        momentum_buffer = state["momentum_buffer"]
        if maximize:
            grad = -grad

        param_fp32 = param.to(torch.float32)
        grad_fp32 = grad.to(torch.float32)
        momentum_buffer_fp32 = momentum_buffer.to(torch.float32)

        # Keep the compiled math out-of-place until copying state back. This
        # avoids functionalization aliasing the fp32 views with fp32 inputs.
        momentum_buffer_fp32 = momentum_buffer_fp32 * momentum + grad_fp32
        momentum_buffer.copy_(momentum_buffer_fp32)

        # Update gradient
        if nesterov:
            grad_fp32 = grad_fp32 + momentum_buffer_fp32 * momentum
        else:
            grad_fp32 = momentum_buffer_fp32

        # Zero power via orthogonalization (Newton-Schulz or Polar Express)
        u = orthogonalize_single(
            grad_fp32,
            ns_steps=ns_steps,
            strategy=orthogonalization_strategy,
            split_sizes=split_sizes,
            adjust_lr_fn=adjust_lr_fn,
        )
        u = u.view_as(param_fp32)

        # Scale update by shape-adjusted LR; decoupled weight decay uses raw lr
        # so tall matrices and split_sizes blocks are not over-decayed.
        adjusted_lr = lr_ratio * lr
        update = adjusted_lr * u
        if weight_decay != 0:
            scale = self.weight_decay_scale(lr, **opts)
            wd_lr = lr * weight_decay if scale is None else lr * weight_decay * scale
            if cautious_wd:
                update = update + wd_lr * param_fp32 * (u * param_fp32 > 0)
            else:
                update = update + wd_lr * param_fp32

        param.copy_(param_fp32 - update)

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
        adjust_lr_fn: AdjustLrFn = "original",
        **opts,
    ) -> None:
        """Batched Muon update. Assumes all tensors are same shape and device."""
        p = params
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

        # Independent of u: enqueued before orthogonalization so on the DTensor
        # path it overlaps the async redistribute instead of queuing behind its
        # stream wait (forced at the first pointwise op on u below).
        adjusted_lr = torch._foreach_mul(lr_ratios, lr)

        # Zero power via orthogonalization
        u = orthogonalize_batch(
            g,
            ns_steps=ns_steps,
            strategy=orthogonalization_strategy,
            split_sizes=split_sizes,
            adjust_lr_fn=adjust_lr_fn,
        )
        u = [u_.to(torch.float32).view_as(p_) for u_, p_ in zip(u, p, strict=True)]

        # Scale update direction by adjusted_lr; decoupled weight decay uses raw lr.
        if weight_decay != 0:
            scale = self.weight_decay_scale(lr, **opts)
            wd_scale = lr * weight_decay if scale is None else lr * weight_decay * scale
            if cautious_wd:
                # decay += wd_scale * p, only where u * p > 0
                mask = [(u_ * p_ > 0).to(torch.float32) for u_, p_ in zip(u, p, strict=True)]
                torch._foreach_mul_(u, adjusted_lr)
                wd_term = torch._foreach_mul(p, mask)
                torch._foreach_mul_(wd_term, wd_scale)
                torch._foreach_add_(u, wd_term)
            else:
                torch._foreach_mul_(u, adjusted_lr)
                wd_term = torch._foreach_mul(p, wd_scale)
                torch._foreach_add_(u, wd_term)
        else:
            torch._foreach_mul_(u, adjusted_lr)

        torch._foreach_add_(p, u, alpha=-1)


@register_muon_algorithm
class MuonC(MuonBaseline):
    """Muon with corrected weight decay, from https://arxiv.org/abs/2506.02285.

    Weight decay is scaled by ``lr / max_lr``, so the per-step decay is
    ``lr^2 / max_lr * wd`` instead of ``lr * wd`` (Algorithm 1's "corrected weight
    decay", the AdamC rule applied to Muon). With a decaying schedule this keeps the
    steady-state weight norm of normalized layers from collapsing toward the end of
    training. ``max_lr`` defaults to the group's ``lr`` at construction; set it to the
    schedule's peak if the initial ``lr`` is not the maximum.
    """

    name = "muonc"
    options = {"max_lr": None}

    def resolve_options(self, group: dict) -> None:
        if group["max_lr"] is None:
            group["max_lr"] = float(group["lr"])
        if not group["max_lr"] > 0:
            raise ValueError(f"muonc requires max_lr > 0, got {group['max_lr']}")

    def weight_decay_scale(self, lr: Tensor, max_lr: float = 1.0, **opts) -> Tensor:
        return lr / max_lr
