"""
Shared orthogonalization utilities for Muon optimizer.

This module provides the dispatch logic and DTensor handling for
different orthogonalization strategies (Newton-Schulz, Polar Express).
"""

from collections.abc import Sequence
import math
from typing import Annotated, Literal, cast

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard

from .newton_schulz import ns_loop, ns_loop_triton
from .polar_express import POLAR_EXPRESS_COEFFS, pe_loop, pe_loop_triton

OrthogonalizationStrategy = Literal["newton_schulz", "polar_express"]
NsCoefficients = tuple[tuple[float, float, float], ...]


def get_dtensor_metadata(dtensors: DTensor | list[DTensor], run_check: bool = True):
    """Extract metadata from DTensor(s) for reconstruction after operations."""
    tensor = dtensors if isinstance(dtensors, DTensor) else dtensors[0]
    metadata = {
        "device_mesh": tensor.device_mesh,
        "placements": tensor.placements,
        "shape": tensor.shape,
        "stride": tensor.stride(),
    }

    # The consistency assert builds dicts of stride SymInts, which Dynamo cannot
    # represent when this runs inside a compiled region under dynamic shapes —
    # skip the purely defensive check there.
    if run_check and isinstance(dtensors, list) and not torch.compiler.is_compiling():
        assert all(
            metadata
            == {
                "device_mesh": dtensor.device_mesh,
                "placements": dtensor.placements,
                "shape": dtensor.shape,
                "stride": dtensor.stride(),
            }
            for dtensor in dtensors
        )

    return metadata


def _get_orthogonalization_fn(strategy: OrthogonalizationStrategy, use_triton: bool):
    """Get the orthogonalization function based on strategy."""
    if strategy == "newton_schulz":
        return ns_loop_triton if use_triton else ns_loop
    elif strategy == "polar_express":
        return pe_loop_triton if use_triton else pe_loop
    else:
        raise ValueError(f"Unknown orthogonalization strategy: {strategy}")


def normalize_ns_coefficients(
    coefficients: Sequence[Sequence[float]] | None,
) -> NsCoefficients | None:
    """Validate and normalize a custom ``(a, b, c)`` coefficient schedule."""
    if coefficients is None:
        return None
    if isinstance(coefficients, (str, bytes)) or not isinstance(coefficients, Sequence):
        raise ValueError(
            "ns_coefficients must be a non-empty sequence of (a, b, c) float triples."
        )
    if len(coefficients) == 0:
        raise ValueError("ns_coefficients must be non-empty.")
    normalized: list[tuple[float, float, float]] = []
    for entry in coefficients:
        if (
            isinstance(entry, (str, bytes))
            or not isinstance(entry, Sequence)
            or len(entry) != 3
        ):
            raise ValueError(
                f"Each ns_coefficients entry must be a 3-tuple of floats, got {entry!r}."
            )
        triple: list[float] = []
        for value in entry:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"ns_coefficients values must be finite numbers, got {value!r}."
                )
            fval = float(value)
            if not math.isfinite(fval):
                raise ValueError(
                    f"ns_coefficients values must be finite numbers, got {value!r}."
                )
            triple.append(fval)
        normalized.append((triple[0], triple[1], triple[2]))
    return tuple(normalized)


def _resolve_steps_and_coefficients(
    strategy: OrthogonalizationStrategy,
    steps: int | None,
    coefficients: Sequence[Sequence[float]] | None,
) -> tuple[int, NsCoefficients | None]:
    norm_coeffs = normalize_ns_coefficients(coefficients)
    if norm_coeffs is None:
        resolved_steps = 5 if steps is None else steps
        if strategy == "polar_express":
            assert resolved_steps <= len(POLAR_EXPRESS_COEFFS), (
                f"polar express orthogonalization only supports up to "
                f"{len(POLAR_EXPRESS_COEFFS)} optimization steps without custom coefficients."
            )
        return resolved_steps, None

    resolved_steps = len(norm_coeffs) if steps is None else steps
    if resolved_steps < 0:
        raise ValueError(f"steps must be non-negative, got {resolved_steps}")
    if len(norm_coeffs) == 1:
        schedule = norm_coeffs * resolved_steps
    else:
        if resolved_steps > len(norm_coeffs):
            raise ValueError(
                f"steps ({resolved_steps}) exceeds length of custom coefficients "
                f"({len(norm_coeffs)})."
            )
        schedule = norm_coeffs[:resolved_steps]
    return resolved_steps, schedule


def _resolve_eps(strategy: OrthogonalizationStrategy, eps: float | None) -> float:
    if eps is not None:
        return eps
    return 1e-6 if strategy == "polar_express" else 1e-7


def _run_orthogonalize(
    X: Tensor,
    *,
    strategy: OrthogonalizationStrategy,
    steps: int,
    eps: float,
    use_triton: bool,
    schedule: NsCoefficients | None,
) -> Tensor:
    if schedule is not None:
        fn = pe_loop_triton if use_triton else pe_loop
        return fn(X, steps=steps, eps=eps, coefficients=schedule, norm_factor=1.0)
    orthogonalize_fn = _get_orthogonalization_fn(strategy, use_triton)
    return orthogonalize_fn(X, steps=steps, eps=eps)


def zeropower(
    G: Annotated[Tensor, "N M"] | Annotated[DTensor, "N M"],
    steps: int | None = None,
    eps: float | None = None,
    use_triton: bool = True,
    strategy: OrthogonalizationStrategy = "newton_schulz",
    coefficients: Sequence[Sequence[float]] | None = None,
) -> Annotated[Tensor, "N M"] | Annotated[DTensor, "N M"]:
    """
    Compute zero-power (sign) of a single gradient tensor using orthogonalization.

    Args:
        G: Gradient tensor (2D, regular or DTensor)
        steps: Number of iteration steps (defaults to ``len(coefficients)`` when
            custom ``coefficients`` are provided, otherwise 5)
        eps: Small constant for numerical stability (defaults to 1e-7 for
            newton_schulz and 1e-6 for polar_express)
        strategy: Orthogonalization algorithm to use:
            - "newton_schulz": Classic Newton-Schulz iteration (default)
            - "polar_express": Polar Express algorithm (arxiv.org/pdf/2505.16932)
        coefficients: Optional custom ``(a, b, c)`` schedule. A single-entry
            schedule repeats for ``steps`` iterations; multi-entry schedules
            supply per-step coefficients (no implicit 1.02 norm factor).

    Returns:
        Orthogonalized tensor with same structure as input
    """
    # Validate strategy before running even when custom coefficients are given.
    _get_orthogonalization_fn(strategy, False)
    resolved_steps, schedule = _resolve_steps_and_coefficients(strategy, steps, coefficients)
    resolved_eps = _resolve_eps(strategy, eps)
    triton_enabled = use_triton and G.is_cuda

    if isinstance(G, DTensor):
        X = cast(DTensor, G.bfloat16()).full_tensor()
        U = _run_orthogonalize(
            X,
            strategy=strategy,
            steps=resolved_steps,
            eps=resolved_eps,
            use_triton=triton_enabled,
            schedule=schedule,
        )
        U = DTensor.from_local(
            U, device_mesh=G.device_mesh, placements=(Replicate(),) * G.device_mesh.ndim
        )
        return U.redistribute(placements=G.placements, async_op=True)
    else:
        return _run_orthogonalize(
            G.bfloat16(),
            strategy=strategy,
            steps=resolved_steps,
            eps=resolved_eps,
            use_triton=triton_enabled,
            schedule=schedule,
        )


def foreach_zeropower(
    Gs: list[Annotated[Tensor, "N M"]]
    | list[Annotated[DTensor, "N M"]]
    | list[Annotated[Tensor, "G N M"]]
    | list[Annotated[DTensor, "G N M"]],
    steps: int | None = None,
    eps: float | None = None,
    use_triton: bool = True,
    strategy: OrthogonalizationStrategy = "newton_schulz",
    coefficients: Sequence[Sequence[float]] | None = None,
) -> (
    list[Annotated[Tensor, "N M"]]
    | list[Annotated[DTensor, "N M"]]
    | list[Annotated[Tensor, "G N M"]]
    | list[Annotated[DTensor, "G N M"]]
):
    """
    Compute zero-power (sign) of gradients using orthogonalization.

    Args:
        Gs: List of gradient tensors (2D or 3D, regular or DTensor)
        steps: Number of iteration steps (defaults to ``len(coefficients)`` when
            custom ``coefficients`` are provided, otherwise 5)
        eps: Small constant for numerical stability (defaults to 1e-7 for
            newton_schulz and 1e-6 for polar_express)
        use_triton: Whether to use triton kernel on CUDA
        strategy: Orthogonalization algorithm to use:
            - "newton_schulz": Classic Newton-Schulz iteration (default)
            - "polar_express": Polar Express algorithm (arxiv.org/pdf/2505.16932)
        coefficients: Optional custom ``(a, b, c)`` schedule.

    Returns:
        List of orthogonalized tensors with same structure as input
    """
    # Validate strategy before indexing Gs so bogus strategies always raise ValueError
    _get_orthogonalization_fn(strategy, False)
    resolved_steps, schedule = _resolve_steps_and_coefficients(strategy, steps, coefficients)
    resolved_eps = _resolve_eps(strategy, eps)
    triton_enabled = use_triton and Gs[0].is_cuda

    if isinstance(Gs[0], DTensor):
        Gs = cast(list[DTensor], Gs)
        metadata = get_dtensor_metadata(Gs)
        shard_0_pl = (Shard(0), *[Replicate()] * (metadata["device_mesh"].ndim - 1))

        X = cast(DTensor, torch.stack(list(Gs), dim=0).bfloat16())

        # Distribute X chunks across devices for orthogonalization
        X_dist = X.redistribute(placements=shard_0_pl, async_op=True)
        U_dist = _run_orthogonalize(
            X_dist.to_local(),
            strategy=strategy,
            steps=resolved_steps,
            eps=resolved_eps,
            use_triton=triton_enabled,
            schedule=schedule,
        )
        U_dist = DTensor.from_local(U_dist, **get_dtensor_metadata(X_dist))
        U = U_dist.redistribute(placements=X.placements, async_op=True)

        # Unbind back into list (on local tensors)
        U_local = U.to_local()
        Us_local = list(U_local.unbind(0))
        Us = [DTensor.from_local(u, **metadata) for u in Us_local]

        return Us

    else:
        X = torch.stack([g for g in Gs])
        Us = _run_orthogonalize(
            X.bfloat16(),
            strategy=strategy,
            steps=resolved_steps,
            eps=resolved_eps,
            use_triton=triton_enabled,
            schedule=schedule,
        )
        return list(Us.unbind(0))


def is_fsdp_3d_sharded(xs: list[DTensor] | list[Tensor]) -> bool:
    return all(
        (
            isinstance(x, DTensor)
            and x.ndim == 3
            and sum(p == Shard(0) for p in x.placements) == 1
            and sum(p == Replicate() for p in x.placements) == len(x.placements) - 1
        )
        for x in xs
    )


def foreach_zeropower_3d_fsdp(
    Gs: list[Annotated[DTensor, "G N M"]],
    steps: int | None = None,
    eps: float | None = None,
    use_triton: bool = True,
    strategy: OrthogonalizationStrategy = "newton_schulz",
    coefficients: Sequence[Sequence[float]] | None = None,
) -> (
    list[Annotated[Tensor, "N M"]]
    | list[Annotated[DTensor, "N M"]]
    | list[Annotated[Tensor, "G N M"]]
    | list[Annotated[DTensor, "G N M"]]
):
    _get_orthogonalization_fn(strategy, False)
    resolved_steps, schedule = _resolve_steps_and_coefficients(strategy, steps, coefficients)
    resolved_eps = _resolve_eps(strategy, eps)
    assert is_fsdp_3d_sharded(Gs), (
        "foreach_zeropower_3d_fsdp only works for DTensors that are *only* sharded along the first dimension"
    )

    triton_enabled = use_triton and Gs[0].is_cuda
    metadata = get_dtensor_metadata(Gs)

    # List[B, G_shard, N, M], distributed across G shards
    X_dist = torch.stack([g.to_local().bfloat16() for g in Gs])
    U_dist = _run_orthogonalize(
        X_dist,
        strategy=strategy,
        steps=resolved_steps,
        eps=resolved_eps,
        use_triton=triton_enabled,
        schedule=schedule,
    )

    # Unbind back into list (on local tensors)
    Us_local = list(U_dist.unbind(0))
    Us = [DTensor.from_local(u, **metadata) for u in Us_local]

    return Us
