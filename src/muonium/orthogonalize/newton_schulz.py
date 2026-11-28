"""
Newton-Schulz iteration for matrix orthogonalization.

This module contains the PyTorch implementation of Newton-Schulz iteration.
The triton kernel is in the kernel submodule.
"""

from typing import Annotated

import torch
from torch import Tensor

from muonium.kernels.gram import gram_


def _fused_addmm(
    input: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    *,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> Tensor:
    """Compute ``beta * input + alpha * (mat1 @ mat2)`` via out-of-place addmm/baddbmm.

    Handles 2D (``addmm``), 3D (``baddbmm``), and 4D+ (flattening leading batch
    dims into 3D ``baddbmm`` and restoring shape) so cuBLAS accumulates the
    epilogue in FP32 before rounding to BF16 once. Out-of-place avoids the
    Inductor 2D functionalization aliasing bug.
    """
    if input.ndim == 2:
        return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)
    if input.ndim == 3:
        return torch.baddbmm(input, mat1, mat2, beta=beta, alpha=alpha)
    batch_shape = input.shape[:-2]
    m, n = input.shape[-2], input.shape[-1]
    k = mat1.shape[-1]
    out = torch.baddbmm(
        input.reshape(-1, m, n),
        mat1.reshape(-1, m, k),
        mat2.reshape(-1, k, n),
        beta=beta,
        alpha=alpha,
    )
    return out.reshape(*batch_shape, m, n)


@torch.compile(fullgraph=True)
def ns_loop(
    X: Annotated[Tensor, "B N M"],
    steps: int = 5,
    *,
    a: float = 3.4445,
    b: float = -4.7750,
    c: float = 2.0315,
    eps: float = 1e-7,
) -> Annotated[Tensor, "B N M"]:
    """
    Newton-Schulz iteration for computing matrix sign function.

    Args:
        X: Input tensor of shape (B, N, M) where B is batch size
        steps: Number of iteration steps
        a, b, c: Newton-Schulz coefficients
        eps: Small constant for numerical stability

    Returns:
        Orthogonalized tensor of same shape as input
    """
    if X.size(0) == 0:
        return X

    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.transpose(-2, -1)

    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)

    for _ in range(steps):
        A = torch.matmul(X, X.transpose(-1, -2))
        B = _fused_addmm(A, A, A, beta=b, alpha=c)
        X = _fused_addmm(X, B, X, beta=a)

    if transpose:
        X = X.transpose(-2, -1)

    return X


@torch.compile(fullgraph=True)
def ns_loop_triton(
    X: Tensor,
    steps: int = 5,
    *,
    a: float = 3.4445,
    b: float = -4.7750,
    c: float = 2.0315,
    eps: float = 1e-7,
) -> Tensor:
    """
    Batched/ND Newton-Schulz orthogonalization over the last 2 dims.
    """
    assert X.ndim >= 2
    if X.size(0) == 0:
        return X

    transpose = X.size(-2) > X.size(-1)
    if transpose:
        # Materialize the transposed layout once before the loop so gram_() does
        # not copy X on every iteration and B @ X sees contiguous operands.
        X = X.mT.contiguous()

    # Ensure spectral norm <= 1
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)

    M = X.size(-2)
    buf_shape = (*X.shape[:-2], M, M)
    A = torch.empty(buf_shape, dtype=X.dtype, device=X.device)
    B = torch.empty_like(A)

    for _ in range(steps):
        # A = X X^T
        gram_(X, A)

        # B = A A^T == A^2 since A is symmetric
        gram_(A, B)

        # B := B = b*A + c*A^2   (reuse B storage)
        B.mul_(c).add_(A, alpha=b)

        X = _fused_addmm(X, B, X, beta=a)

    if transpose:
        X = X.mT
    return X
