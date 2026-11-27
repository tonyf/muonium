# Baseline Muon update rule (Keller Jordan's Muon), per-tensor and foreach forms.
import torch
from torch import Tensor

from muonium.orthogonalize import OrthogonalizationStrategy

from .base import (
    MuonAlgorithm,
    orthogonalize_batch,
    orthogonalize_single,
    register_muon_algorithm,
)


@register_muon_algorithm
class MuonBaseline(MuonAlgorithm):
    """The original Muon update: momentum, orthogonalize, shape-scaled apply."""

    name = "muon"

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
        )
        u = u.view_as(param_fp32)

        # Scale update by shape-adjusted LR; decoupled weight decay uses raw lr
        # so tall matrices and split_sizes blocks are not over-decayed.
        adjusted_lr = lr_ratio * lr
        update = adjusted_lr * u
        if weight_decay != 0:
            if cautious_wd:
                update = update + (lr * weight_decay) * param_fp32 * (u * param_fp32 > 0)
            else:
                update = update + (lr * weight_decay) * param_fp32

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
        )
        u = [u_.to(torch.float32).view_as(p_) for u_, p_ in zip(u, p, strict=True)]

        # Scale update direction by adjusted_lr; decoupled weight decay uses raw lr.
        if weight_decay != 0:
            wd_scale = lr * weight_decay
            if cautious_wd:
                # decay += lr * weight_decay * p, only where u * p > 0
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
