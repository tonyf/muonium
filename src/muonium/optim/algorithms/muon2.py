"""Muon²: Muon with Adam-style second-moment preconditioning.

Implements Liu, Zhang, Wang, Zhao, Su, Yang & Zhang, "Muon²: Boosting Muon via
Adaptive Second-Moment Preconditioning" (2026), https://arxiv.org/abs/2604.09967.
The momentum is divided elementwise by ``sqrt(V) + eps`` before orthogonalization,
where ``V`` is an EMA of the squared gradient (Algorithm 1). This improves the
momentum's conditioning, so Newton-Schulz converges in fewer steps; the paper
reports matching or beating Muon with 3 steps instead of 5 (set ``ns_steps``).
Muon²-F (Algorithm 2) stores ``V`` in Adafactor-style factored form.

Momentum, Nesterov, decoupled weight decay and the ``lr_ratio`` shape factor follow
this package's Muon. The paper's EMA momentum differs from the summed buffer used here
only by a scale factor, which Newton-Schulz normalizes away (up to ``eps``).
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


@register_muon_algorithm
class Muon2(MuonAlgorithm):
    """Muon² (https://arxiv.org/abs/2604.09967, Algorithm 1).

    Preconditions the momentum by an elementwise second moment of the gradient
    (decay ``muon_beta2``) before orthogonalization. Adds a grad-sized
    ``exp_avg_sq`` buffer; :class:`Muon2F` is the memory-efficient variant.
    """

    name = "muon2"
    options = {"muon_beta2": 0.95, "muon_eps": 1e-8}
    state_spec = {
        "momentum_buffer": BufferSpec(like="grad", signed=True),
        "exp_avg_sq": BufferSpec(like="grad", signed=False),
    }

    def second_moment(self, grad: Tensor, state: dict[str, Tensor], beta2: float) -> Tensor:
        """Advance the second-moment state with ``grad`` (fp32); return V, grad-shaped."""
        exp_avg_sq = state["exp_avg_sq"]
        v = torch.lerp(exp_avg_sq.to(torch.float32), grad * grad, 1 - beta2)
        exp_avg_sq.copy_(v)
        return v

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
        muon_beta2: float = 0.95,
        muon_eps: float = 1e-8,
        **opts,
    ) -> None:
        momentum_buffer = state["momentum_buffer"]
        if maximize:
            grad = -grad

        param_fp32 = param.to(torch.float32)
        grad_fp32 = grad.to(torch.float32)

        # Out-of-place until state copy-back (see MuonBaseline.update).
        momentum_buffer_fp32 = momentum_buffer.to(torch.float32) * momentum + grad_fp32
        momentum_buffer.copy_(momentum_buffer_fp32)
        v = self.second_moment(grad_fp32, state, muon_beta2)

        if nesterov:
            m = grad_fp32 + momentum_buffer_fp32 * momentum
        else:
            m = momentum_buffer_fp32

        u = orthogonalize_single(
            m / (v.sqrt() + muon_eps),
            ns_steps=ns_steps,
            strategy=orthogonalization_strategy,
            split_sizes=split_sizes,
        )
        u = u.view_as(param_fp32)

        if weight_decay != 0:
            if cautious_wd:
                u = u + weight_decay * param_fp32 * (u * param_fp32 > 0)
            else:
                u = u + weight_decay * param_fp32

        adjusted_lr = lr_ratio * lr
        param.copy_(param_fp32 - adjusted_lr * u)

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
        muon_beta2: float = 0.95,
        muon_eps: float = 1e-8,
        **opts,
    ) -> None:
        """Batched Muon² update. Assumes all tensors are same shape and device."""
        p = params
        g = grads
        buf = state["momentum_buffer"]
        if maximize:
            g = list(torch._foreach_neg(g))

        # Second moments read the raw gradient, so advance them before nesterov
        # folds the momentum into g in place.
        vs = [
            self.second_moment(g_, {key: buffers[i] for key, buffers in state.items()}, muon_beta2)
            for i, g_ in enumerate(g)
        ]

        torch._foreach_mul_(buf, momentum)
        torch._foreach_add_(buf, g)

        if nesterov:
            torch._foreach_add_(g, buf, alpha=momentum)
        else:
            g = buf

        # Independent of u: enqueued before orthogonalization so on the DTensor
        # path it overlaps the async redistribute (see MuonBaseline.foreach_update).
        adjusted_lr = torch._foreach_mul(lr_ratios, lr)

        denom = torch._foreach_add(torch._foreach_sqrt(vs), muon_eps)
        u = orthogonalize_batch(
            list(torch._foreach_div(g, denom)),
            ns_steps=ns_steps,
            strategy=orthogonalization_strategy,
            split_sizes=split_sizes,
        )
        u = [u_.view_as(p_) for u_, p_ in zip(u, p, strict=True)]

        if weight_decay != 0:
            if cautious_wd:
                mask = [(u_ * p_ > 0).to(p_.dtype) for u_, p_ in zip(u, p, strict=True)]
                masked_p = torch._foreach_mul(p, mask)
                torch._foreach_add_(u, masked_p, alpha=weight_decay)
            else:
                torch._foreach_add_(u, p, alpha=weight_decay)

        torch._foreach_mul_(u, adjusted_lr)

        torch._foreach_add_(p, u, alpha=-1)


@register_muon_algorithm
class Muon2F(Muon2):
    """Muon²-F (https://arxiv.org/abs/2604.09967, Algorithm 2): factored second moment.

    Keeps Adafactor-style EMAs of the row sums ``r`` and column sums ``c`` of the
    squared gradient and reconstructs ``V = r c^T / sum(r)`` per matrix, so the
    extra state is ``N + M`` values instead of ``N * M``. (Algorithm 2 line 11
    prints ``V_t`` in the preconditioner; the factored ``V̂_t`` is clearly meant.)
    With ``split_sizes``, the factors span the whole fused matrix.
    """

    name = "muon2f"
    state_spec = {
        "momentum_buffer": BufferSpec(like="grad", signed=True),
        "exp_avg_sq_row": BufferSpec(like="grad_rows", signed=False),
        "exp_avg_sq_col": BufferSpec(like="grad_cols", signed=False),
    }

    def second_moment(self, grad: Tensor, state: dict[str, Tensor], beta2: float) -> Tensor:
        row, col = state["exp_avg_sq_row"], state["exp_avg_sq_col"]
        g2 = grad * grad
        r = torch.lerp(row.to(torch.float32), g2.sum(dim=-1, keepdim=True), 1 - beta2)
        c = torch.lerp(col.to(torch.float32), g2.sum(dim=-2, keepdim=True), 1 - beta2)
        row.copy_(r)
        col.copy_(c)
        return r * c / r.sum(dim=-2, keepdim=True).clamp(min=1e-30)
