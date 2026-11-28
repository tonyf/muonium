import math
from typing import Any

import pytest
import torch

import muonium.optim.algorithms.base as algo_base
from muonium.optim.algorithms import get_algorithm
from muonium.optim.optim import Muon

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_standalone_step_uses_private_compiled_muon_callable():
    p = torch.nn.Parameter(torch.randn(2, 2))
    p.grad = torch.randn_like(p)
    optimizer = Muon([p])
    calls = []

    def compiled_muon(*args, **kwargs):
        calls.append((args, kwargs))

    optimizer._muon_impls["muon"] = compiled_muon
    optimizer.step()

    assert len(calls) == 1
    assert calls[0][0][0] == [p]


def test_external_torch_compile_captures_muon_update(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(4, 2))
    ref = torch.nn.Parameter(p.detach().clone())
    optimizer = Muon(
        [p],
        lr=0.1,
        wd=0.0,
        momentum=0.0,
        nesterov=False,
    )
    ref_optimizer = Muon([ref], lr=0.1, wd=0.0, momentum=0.0, nesterov=False)
    ref_optimizer._muon_impls["muon"] = ref_optimizer._raw_muon_impl(get_algorithm("muon"))
    graphs = []

    def fail_nested_compile(*args, **kwargs):
        raise AssertionError("external compilation must use the raw Muon kernel")

    def recording_backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    optimizer._muon_impls["muon"] = fail_nested_compile

    @torch.compile(backend=recording_backend)
    def compiled_step():
        optimizer.step()

    for value in (0.5, 0.25):
        p.grad = torch.full_like(p, value)
        ref.grad = torch.full_like(ref, value)
        compiled_step()
        ref_optimizer.step()

    assert graphs
    torch.testing.assert_close(p, ref)
    torch.testing.assert_close(optimizer.state[p]["momentum_buffer"], ref_optimizer.state[ref]["momentum_buffer"])
    torch.testing.assert_close(optimizer.state[p]["step"], ref_optimizer.state[ref]["step"])


@pytest.mark.parametrize("tensor_lr", [False, True])
def test_external_torch_compile_captures_adam_update(tensor_lr):
    params = [torch.nn.Parameter(torch.randn(4)) for _ in range(3)]
    refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
    optimizer = Muon(
        [
            {
                "params": params,
                "algorithm": "adamw",
                "lr": torch.tensor(0.01) if tensor_lr else 0.01,
                "wd": 0.1,
                "fused": False,
                "foreach": True,
            }
        ]
    )
    ref_optimizer = torch.optim.AdamW(
        refs,
        lr=0.01,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        foreach=True,
    )
    graphs = []

    def recording_backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    @torch.compile(backend=recording_backend)
    def compiled_step():
        optimizer.step()

    for _ in range(2):
        for p, ref in zip(params, refs, strict=True):
            grad = torch.randn_like(p)
            p.grad = grad.clone()
            ref.grad = grad.clone()
        compiled_step()
        ref_optimizer.step()

    assert graphs
    for p, ref in zip(params, refs, strict=True):
        torch.testing.assert_close(p, ref)


@requires_cuda
def test_standalone_compiled_muon_group_steps_on_cuda():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 4, device="cuda"))
    before = p.detach().clone()
    p.grad = torch.randn_like(p)
    optimizer = Muon(
        [p],
        lr=0.01,
        wd=0.0,
        momentum=0.0,
        nesterov=False,
        orthogonalization_strategy="newton_schulz",
    )

    optimizer.step()

    assert not torch.equal(p, before)
    assert torch.equal(optimizer.state[p]["step"], torch.tensor(1.0))


@requires_cuda
@pytest.mark.parametrize("strategy", ["newton_schulz", "polar_express"])
def test_external_torch_compile_runs_real_muon_kernel_on_cuda(strategy):
    p = torch.nn.Parameter(torch.randn(8, 4, device="cuda"))
    before = p.detach().clone()
    p.grad = torch.randn_like(p)
    optimizer = Muon(
        [p],
        lr=0.01,
        wd=0.0,
        momentum=0.0,
        nesterov=False,
        orthogonalization_strategy=strategy,
    )

    def fail_nested_compile(*args, **kwargs):
        raise AssertionError("external compilation must use the raw Muon kernel")

    optimizer._muon_impls["muon"] = fail_nested_compile

    @torch.compile
    def compiled_step():
        optimizer.step()

    compiled_step()

    assert not torch.equal(p, before)
    assert torch.equal(optimizer.state[p]["step"], torch.tensor(1.0))


@requires_cuda
def test_fused_adam_cuda_state_step_is_on_param_device():
    p = torch.nn.Parameter(torch.randn(4, device="cuda"))
    optimizer = Muon([{"params": [p], "algorithm": "adamw", "fused": True}])
    p.grad = torch.randn_like(p)

    optimizer.step()

    assert optimizer.state[p]["step"].device == p.device
    assert optimizer.state[p]["step"].dtype is torch.float32


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lr": -1.0}, "Invalid learning rate"),
        ({"adam_eps": -1.0}, "Invalid epsilon"),
        ({"adam_betas": (1.0, 0.9)}, "index 0"),
        ({"adam_betas": (0.9, -0.1)}, "index 1"),
    ],
)
def test_constructor_validates_scalar_hyperparameters(kwargs, match):
    p = torch.nn.Parameter(torch.randn(2, 2))

    with pytest.raises(ValueError, match=match):
        Muon([p], **kwargs)


def test_constructor_rejects_single_param_group_dict():
    p = torch.nn.Parameter(torch.randn(2, 2))

    with pytest.raises(TypeError, match="not a single dict"):
        Muon({"params": [p]})


def test_constructor_rejects_empty_params():
    with pytest.raises(ValueError, match="empty parameter list"):
        Muon([])


def test_constructor_rejects_group_missing_params_key():
    with pytest.raises(ValueError, match="'params' key"):
        Muon([{"algorithm": "muon"}])


def test_constructor_skips_empty_groups_when_other_groups_exist():
    p = torch.nn.Parameter(torch.randn(2, 2))

    optimizer = Muon([{"params": []}, {"params": [p]}])

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["params"] == [p]


def test_constructor_rejects_unknown_algorithm():
    p = torch.nn.Parameter(torch.randn(2, 2))

    with pytest.raises(ValueError, match="Unknown algorithm"):
        Muon([{"params": [p], "algorithm": "rmsprop"}])


def test_constructor_rejects_invalid_muon_parameters():
    one_d = torch.nn.Parameter(torch.randn(4))
    complex_param = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.cfloat))

    with pytest.raises(ValueError, match="2D\\+"):
        Muon([one_d])
    with pytest.raises(NotImplementedError, match="Complex parameters"):
        Muon([complex_param])


def test_constructor_normalizes_muon_and_adam_groups():
    muon_param = torch.nn.Parameter(torch.randn(2, 2))
    adam_param = torch.nn.Parameter(torch.randn(4))

    optimizer = Muon(
        [
            {"params": [muon_param], "lr": 0.5, "wd": 0.25},
            {"params": [adam_param], "algorithm": "adamw", "lr": 0.3, "wd": 0.1},
        ],
        lr=0.01,
        wd=0.9,
    )

    muon_group, adam_group = optimizer.param_groups
    assert muon_group["use_muon"] is True
    assert torch.equal(muon_group["lr"], torch.tensor(0.5))
    assert muon_group["wd"] == 0.25
    assert adam_group["use_muon"] is False
    assert adam_group["lr"] == 0.3
    assert adam_group["wd"] == 0.1


def test_constructor_allows_3d_muon_and_complex_adam_parameters():
    muon_param = torch.nn.Parameter(torch.randn(2, 3, 4))
    adam_param = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.cfloat))

    optimizer = Muon(
        [
            {"params": [muon_param]},
            {"params": [adam_param], "algorithm": "adam", "fused": False},
        ]
    )

    assert optimizer.param_groups[0]["use_muon"] is True
    assert optimizer.param_groups[1]["has_complex"] is True


def test_adam_group_matches_torch_adamw_on_cpu():
    p = torch.nn.Parameter(torch.randn(3, 4))
    ref = torch.nn.Parameter(p.detach().clone())
    optimizer = Muon(
        [
            {
                "params": [p],
                "algorithm": "adamw",
                "lr": 0.01,
                "wd": 0.1,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "fused": False,
            }
        ]
    )
    ref_optimizer = torch.optim.AdamW(
        [ref], lr=0.01, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8
    )

    for _ in range(3):
        grad = torch.randn_like(p)
        p.grad = grad.clone()
        ref.grad = grad.clone()
        optimizer.step()
        ref_optimizer.step()

    torch.testing.assert_close(p, ref)


def test_algorithm_adam_matches_torch_adam_coupled_weight_decay_on_cpu():
    p = torch.nn.Parameter(torch.randn(3, 4))
    ref = torch.nn.Parameter(p.detach().clone())
    optimizer = Muon(
        [
            {
                "params": [p],
                "algorithm": "adam",
                "lr": 0.01,
                "wd": 0.1,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "fused": False,
            }
        ]
    )
    ref_optimizer = torch.optim.Adam(
        [ref], lr=0.01, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8
    )

    for _ in range(3):
        grad = torch.randn_like(p)
        p.grad = grad.clone()
        ref.grad = grad.clone()
        optimizer.step()
        ref_optimizer.step()

    torch.testing.assert_close(p, ref)


def test_adam_group_amsgrad_matches_torch_adamw_on_cpu():
    p = torch.nn.Parameter(torch.randn(3, 4))
    ref = torch.nn.Parameter(p.detach().clone())
    optimizer = Muon(
        [
            {
                "params": [p],
                "algorithm": "adamw",
                "amsgrad": True,
                "lr": 0.01,
                "wd": 0.1,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "fused": False,
            }
        ]
    )
    ref_optimizer = torch.optim.AdamW(
        [ref],
        lr=0.01,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        amsgrad=True,
    )

    for _ in range(3):
        grad = torch.randn_like(p)
        p.grad = grad.clone()
        ref.grad = grad.clone()
        optimizer.step()
        ref_optimizer.step()

    assert "max_exp_avg_sq" in optimizer.state[p]
    torch.testing.assert_close(p, ref)


def test_complex_adam_group_steps_on_cpu():
    p = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.cfloat))
    optimizer = Muon([{"params": [p], "algorithm": "adam", "fused": False}])
    p.grad = torch.randn_like(p)

    optimizer.step()

    assert optimizer.state[p]["step"].item() == 1


def test_step_returns_closure_loss():
    p = torch.nn.Parameter(torch.randn(3, 4))
    optimizer = Muon([{"params": [p], "algorithm": "adamw", "fused": False}])

    def closure():
        return p.square().sum()

    assert optimizer.step(closure) == closure()


def test_muon_direct_update_initializes_state_and_uses_shape_lr_ratio(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(4, 2))
    p.grad = torch.full_like(p, 0.5)
    optimizer = Muon([p], lr=0.1, wd=0.0, momentum=0.0, nesterov=False)

    optimizer.step()

    state = optimizer.state[p]
    assert torch.equal(state["step"], torch.tensor(1.0))
    assert torch.equal(state["lr_ratio"], torch.tensor(math.sqrt(2.0)))
    torch.testing.assert_close(state["momentum_buffer"], torch.full_like(p, 0.5))
    torch.testing.assert_close(p, torch.ones_like(p) - math.sqrt(2.0) * 0.1 * 0.5)


def test_muon_nesterov_uses_updated_momentum_buffer(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(2, 2))
    p.grad = torch.full_like(p, 0.5)
    optimizer = Muon([p], lr=0.1, wd=0.0, momentum=0.5, nesterov=True)

    optimizer.step()

    # buf = grad; nesterov update = grad + momentum * buf = 0.75
    torch.testing.assert_close(optimizer.state[p]["momentum_buffer"], torch.full_like(p, 0.5))
    torch.testing.assert_close(p, torch.ones_like(p) - 0.1 * 0.75)


def test_muon_cautious_weight_decay_masks_by_update_param_sign(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.tensor([[1.0, -1.0], [1.0, -1.0]]))
    p.grad = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    optimizer = Muon(
        [p],
        lr=1.0,
        wd=1.0,
        momentum=0.0,
        nesterov=False,
        use_cautious_wd=True,
    )

    optimizer.step()

    torch.testing.assert_close(p, torch.tensor([[-1.0, -2.0], [2.0, 1.0]]))


def test_muon_non_cautious_weight_decay_adds_param_everywhere(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.tensor([[1.0, -1.0], [1.0, -1.0]]))
    p.grad = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    optimizer = Muon(
        [p],
        lr=1.0,
        wd=1.0,
        momentum=0.0,
        nesterov=False,
        use_cautious_wd=False,
    )

    optimizer.step()

    torch.testing.assert_close(p, torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))


@pytest.mark.parametrize("cautious", [True, False])
def test_muon_weight_decay_is_not_scaled_by_lr_ratio(monkeypatch, cautious: bool):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    # Tall 8x2 matrix has lr_ratio = sqrt(4) = 2.0, while WD must use raw lr.
    p = torch.nn.Parameter(torch.full((8, 2), 2.0))
    p.grad = torch.full((8, 2), 1.0)
    optimizer = Muon(
        [p],
        lr=0.1,
        wd=0.5,
        momentum=0.0,
        nesterov=False,
        use_cautious_wd=cautious,
    )

    optimizer.step()

    # p_new = p - (lr_ratio * lr * u + lr * wd * p) = 2.0 - (2.0 * 0.1 * 1.0 + 0.1 * 0.5 * 2.0)
    expected = torch.full((8, 2), 2.0 - (2.0 * 0.1 * 1.0 + 0.1 * 0.5 * 2.0))
    torch.testing.assert_close(p.detach(), expected)


def test_muon_maximize_negates_grad_in_place(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(2, 2))
    grad = torch.full_like(p, 0.5)
    p.grad = grad
    optimizer = Muon([p], lr=0.1, wd=0.0, momentum=0.0, nesterov=False, maximize=True)

    optimizer.step()

    torch.testing.assert_close(grad, torch.full_like(grad, 0.5))
    torch.testing.assert_close(p, torch.full_like(p, 1.05))


def test_muon_maximize_reused_grad_keeps_maximize_direction(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(2, 2))
    grad = torch.full_like(p, 0.5)
    p.grad = grad
    optimizer = Muon([p], lr=0.1, wd=0.0, momentum=0.0, nesterov=False, maximize=True)

    optimizer.step()
    after_first = p.detach().clone()
    optimizer.step()

    torch.testing.assert_close(grad, torch.full_like(grad, 0.5))
    torch.testing.assert_close(after_first, torch.full_like(after_first, 1.05))
    torch.testing.assert_close(p, torch.full_like(p, 1.10))


def test_muon_momentum_buffer_uses_fp32_for_low_precision_params(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
    p.grad = torch.full_like(p, 0.5)
    optimizer = Muon([p], lr=0.1, wd=0.0, momentum=0.9, nesterov=False)

    optimizer.step()

    assert optimizer.state[p]["momentum_buffer"].dtype is torch.float32


def test_muon_skips_params_without_grad(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    with_grad = torch.nn.Parameter(torch.ones(2, 2))
    without_grad = torch.nn.Parameter(torch.ones(2, 2))
    with_grad.grad = torch.full_like(with_grad, 0.5)
    optimizer = Muon([with_grad, without_grad], lr=0.1, wd=0.0, momentum=0.0)

    optimizer.step()

    assert with_grad in optimizer.state
    assert without_grad not in optimizer.state
    torch.testing.assert_close(without_grad, torch.ones_like(without_grad))


def test_muon_flatten_true_updates_ndim_greater_than_three(monkeypatch):
    seen_shapes = []

    def fake_zeropower(g, **_):
        seen_shapes.append(tuple(g.shape))
        return g

    monkeypatch.setattr(algo_base, "zeropower", fake_zeropower)
    p = torch.nn.Parameter(torch.ones(2, 2, 2, 2))
    p.grad = torch.full_like(p, 0.5)
    optimizer = Muon([{"params": [p], "flatten": True}], lr=0.1, wd=0.0, momentum=0.0)

    optimizer.step()

    assert seen_shapes == [(2, 8)]
    assert torch.equal(optimizer.state[p]["lr_ratio"], torch.tensor(1.0))
    torch.testing.assert_close(p, torch.ones_like(p) - 0.1 * 0.5)


def test_muon_flatten_false_batches_leading_dims_of_4d(monkeypatch):
    seen_shapes = []

    def fake_zeropower(g, **_):
        seen_shapes.append(tuple(g.shape))
        return g

    monkeypatch.setattr(algo_base, "zeropower", fake_zeropower)
    p = torch.nn.Parameter(torch.ones(2, 2, 2, 2))
    p.grad = torch.full_like(p, 0.5)
    optimizer = Muon([{"params": [p], "flatten": False}], lr=0.1, wd=0.0, momentum=0.0)

    optimizer.step()

    # Leading dims fold into one batch dim; each trailing 2x2 matrix is
    # orthogonalized independently (dion semantics).
    assert seen_shapes == [(4, 2, 2)]
    assert torch.equal(optimizer.state[p]["lr_ratio"], torch.tensor(1.0))
    torch.testing.assert_close(p, torch.ones_like(p) - 0.1 * 0.5)


def test_muon_default_treats_3d_as_batch_of_matrices(monkeypatch):
    seen_shapes = []

    def fake_zeropower(g, **_):
        seen_shapes.append(tuple(g.shape))
        return g

    monkeypatch.setattr(algo_base, "zeropower", fake_zeropower)
    p = torch.nn.Parameter(torch.ones(2, 4, 2))
    p.grad = torch.full_like(p, 0.5)
    optimizer = Muon([p], lr=0.1, wd=0.0, momentum=0.0)

    optimizer.step()

    # flatten defaults to False: no reshape, lr_ratio from the matrix dims.
    assert seen_shapes == [(2, 4, 2)]
    assert torch.equal(optimizer.state[p]["lr_ratio"], torch.tensor(math.sqrt(2.0)))


def test_muon_flatten_false_steps_3d_grad_without_flattening(monkeypatch):
    seen_shapes = []

    def fake_zeropower(g, **_):
        seen_shapes.append(tuple(g.shape))
        return g

    monkeypatch.setattr(algo_base, "zeropower", fake_zeropower)
    p = torch.nn.Parameter(torch.ones(2, 2, 2))
    p.grad = torch.full_like(p, 0.5)
    optimizer = Muon([{"params": [p], "flatten": False}], lr=0.1, wd=0.0, momentum=0.0)

    optimizer.step()

    assert seen_shapes == [(2, 2, 2)]
    torch.testing.assert_close(p, torch.ones_like(p) - 0.1 * 0.5)


def test_load_state_dict_applies_legacy_defaults_and_tensor_steps():
    p = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = Muon([p])
    group = optimizer.param_groups[0]
    for key in [
        "ns_steps",
        "nesterov",
        "flatten",
        "use_cautious_wd",
        "orthogonalization_strategy",
        "adjust_lr_fn",
        "maximize",
    ]:
        del group[key]
    group["lr"] = 0.1
    optimizer.state[p]["step"] = 3.0

    optimizer.__setstate__({"state": optimizer.state, "param_groups": optimizer.param_groups})

    assert group["ns_steps"] == 5
    assert group["nesterov"] is True
    assert group["flatten"] is False
    assert group["use_cautious_wd"] is True
    assert group["maximize"] is False
    # Legacy default matches the constructor default ("polar_express"), so a
    # checkpoint missing the key doesn't silently switch the orthogonalization scheme.
    assert group["orthogonalization_strategy"] == "polar_express"
    assert group["adjust_lr_fn"] == "original"
    lr = group["lr"]
    assert isinstance(lr, torch.Tensor)
    assert torch.equal(lr, torch.tensor(0.1))
    assert torch.equal(optimizer.state[p]["step"], torch.tensor(3.0))


def test_load_state_dict_applies_adam_legacy_defaults_and_step_tensor():
    p = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = Muon([{"params": [p], "algorithm": "adamw", "fused": False}])
    group = optimizer.param_groups[0]
    for key in ["amsgrad", "maximize", "foreach", "decoupled_weight_decay", "fused"]:
        group.pop(key, None)
    optimizer.state[p]["step"] = 4.0

    optimizer.__setstate__({"state": optimizer.state, "param_groups": optimizer.param_groups})

    assert group["amsgrad"] is False
    assert group["maximize"] is False
    assert group["foreach"] is None
    assert group["decoupled_weight_decay"] is False
    assert group["fused"] is None
    assert torch.equal(optimizer.state[p]["step"], torch.tensor(4.0))


def test_setstate_clears_compiled_muon_impl_cache_and_recompiles_lazily(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    compiled = []

    def fake_compile(fn, *, dynamic):
        compiled.append((fn, dynamic))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    p = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = Muon([p])
    optimizer._muon_impls["muon"] = "stale-compiled-artifact"

    optimizer.__setstate__({"state": optimizer.state, "param_groups": optimizer.param_groups})

    # A stale compiled kernel must not survive checkpoint load; the next step
    # compiles a fresh per-algorithm entry.
    assert optimizer._muon_impls == {}
    p.grad = torch.full_like(p, 0.5)
    optimizer.step()
    assert len(compiled) == 1
    assert compiled[0][1] is True
    assert "muon" in optimizer._muon_impls


def test_state_dict_round_trip_preserves_muon_training_continuity(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(4, 2))
    uninterrupted_p = torch.nn.Parameter(p.detach().clone())
    loaded_p = torch.nn.Parameter(p.detach().clone())
    kwargs: dict[str, Any] = dict(lr=0.1, wd=0.0, momentum=0.5, nesterov=False)
    uninterrupted = Muon([uninterrupted_p], **kwargs)
    to_save = Muon([p], **kwargs)
    loaded = Muon([loaded_p], **kwargs)

    first_grad = torch.full_like(p, 0.5)
    for param in (uninterrupted_p, p):
        param.grad = first_grad.clone()
    uninterrupted.step()
    to_save.step()
    loaded_p.data.copy_(p.data)
    loaded.load_state_dict(to_save.state_dict())

    second_grad = torch.full_like(p, 0.25)
    for param in (uninterrupted_p, loaded_p):
        param.grad = second_grad.clone()
    uninterrupted.step()
    loaded.step()

    torch.testing.assert_close(loaded_p, uninterrupted_p)
    for key in ["step", "momentum_buffer", "lr_ratio"]:
        torch.testing.assert_close(loaded.state[loaded_p][key], uninterrupted.state[uninterrupted_p][key])


def test_mixed_muon_and_adam_groups_step_on_cpu(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    muon_param = torch.nn.Parameter(torch.ones(2, 2))
    adam_param = torch.nn.Parameter(torch.randn(3))
    adam_ref = torch.nn.Parameter(adam_param.detach().clone())
    optimizer = Muon(
        [
            {"params": [muon_param], "lr": 0.1, "wd": 0.0, "momentum": 0.0},
            {
                "params": [adam_param],
                "algorithm": "adamw",
                "lr": 0.01,
                "wd": 0.1,
                "fused": False,
            },
        ]
    )
    ref_optimizer = torch.optim.AdamW([adam_ref], lr=0.01, weight_decay=0.1)

    muon_param.grad = torch.full_like(muon_param, 0.5)
    adam_grad = torch.randn_like(adam_param)
    adam_param.grad = adam_grad.clone()
    adam_ref.grad = adam_grad.clone()

    optimizer.step()
    ref_optimizer.step()

    torch.testing.assert_close(muon_param, torch.full_like(muon_param, 0.95))
    torch.testing.assert_close(adam_param, adam_ref)


@pytest.mark.parametrize(
    ("adjust_lr_fn", "shape", "expected_ratio"),
    [
        ("original", (512, 1536), 1.0),
        (None, (512, 1536), 1.0),
        ("spectral_unclamped", (512, 1536), math.sqrt(512 / 1536)),
        ("match_rms_adamw", (512, 1536), 0.2 * math.sqrt(1536)),
        ("original", (1536, 512), math.sqrt(3.0)),
        ("spectral_unclamped", (1536, 512), math.sqrt(3.0)),
        ("match_rms_adamw", (1536, 512), 0.2 * math.sqrt(1536)),
        ("original", (4, 4), 1.0),
        ("spectral_unclamped", (4, 4), 1.0),
        ("match_rms_adamw", (4, 4), 0.2 * math.sqrt(4)),
        ("spectral_unclamped", (2, 4, 8), math.sqrt(4 / 8)),
        ("match_rms_adamw", (2, 4, 8), 0.2 * math.sqrt(8)),
    ],
)
def test_adjust_lr_fn_scales_muon_update(monkeypatch, adjust_lr_fn, shape, expected_ratio):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(shape))
    p.grad = torch.full_like(p, 0.5)
    lr = 0.1
    optimizer = Muon(
        [p],
        lr=lr,
        wd=0.0,
        momentum=0.0,
        nesterov=False,
        adjust_lr_fn=adjust_lr_fn,
    )

    optimizer.step()

    expected_name = "original" if adjust_lr_fn is None else adjust_lr_fn
    assert optimizer.param_groups[0]["adjust_lr_fn"] == expected_name
    torch.testing.assert_close(optimizer.state[p]["lr_ratio"], torch.tensor(expected_ratio))
    torch.testing.assert_close(p, torch.full_like(p, 1.0 - lr * expected_ratio * 0.5))


def test_adjust_lr_fn_per_group_override_and_validation(monkeypatch):
    monkeypatch.setattr(algo_base, "zeropower", lambda g, **_: g)
    p_orig = torch.nn.Parameter(torch.ones(4, 12))
    p_spec = torch.nn.Parameter(torch.ones(4, 12))
    p_rms = torch.nn.Parameter(torch.ones(4, 12))
    for p in (p_orig, p_spec, p_rms):
        p.grad = torch.full_like(p, 0.5)

    optimizer = Muon(
        [
            {"params": [p_orig]},
            {"params": [p_spec], "adjust_lr_fn": "spectral_unclamped"},
            {"params": [p_rms], "adjust_lr_fn": "match_rms_adamw"},
        ],
        lr=0.1,
        wd=0.0,
        momentum=0.0,
        nesterov=False,
        adjust_lr_fn="original",
    )
    optimizer.step()

    torch.testing.assert_close(optimizer.state[p_orig]["lr_ratio"], torch.tensor(1.0))
    torch.testing.assert_close(
        optimizer.state[p_spec]["lr_ratio"], torch.tensor(math.sqrt(4 / 12))
    )
    torch.testing.assert_close(
        optimizer.state[p_rms]["lr_ratio"], torch.tensor(0.2 * math.sqrt(12))
    )

    with pytest.raises(ValueError, match="Unknown adjust_lr_fn"):
        Muon([torch.nn.Parameter(torch.randn(4, 4))], adjust_lr_fn="bogus")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="Unknown adjust_lr_fn"):
        Muon([{"params": [torch.nn.Parameter(torch.randn(4, 4))], "adjust_lr_fn": "bogus"}])


def test_ns_coefficients_configuration_and_step() -> None:
    torch.manual_seed(0)
    ns_triple = (3.4445, -4.7750, 2.0315)
    eight_step = (ns_triple,) * 8

    # Omitting ns_steps defaults to len(ns_coefficients); matches 8-step newton_schulz.
    p_custom = torch.nn.Parameter(torch.randn(16, 8))
    p_ref = torch.nn.Parameter(p_custom.detach().clone())
    grad = torch.randn_like(p_custom)
    p_custom.grad = grad.clone()
    p_ref.grad = grad.clone()

    opt_custom = Muon(
        [p_custom],
        lr=0.05,
        wd=0.0,
        momentum=0.0,
        nesterov=False,
        orthogonalization_strategy="newton_schulz",
        ns_coefficients=eight_step,
    )
    opt_ref = Muon(
        [p_ref],
        lr=0.05,
        wd=0.0,
        momentum=0.0,
        nesterov=False,
        orthogonalization_strategy="newton_schulz",
        ns_steps=8,
    )
    assert opt_custom.ns_steps == 8
    assert opt_custom.param_groups[0]["ns_steps"] == 8
    assert opt_custom.param_groups[0]["ns_coefficients"] == eight_step

    opt_custom.step()
    opt_ref.step()
    torch.testing.assert_close(p_custom, p_ref, rtol=1e-5, atol=1e-6)

    # Per-group override defaults ns_steps to len(group["ns_coefficients"]).
    p_g = torch.nn.Parameter(torch.randn(8, 8))
    opt_group = Muon([{"params": [p_g], "ns_coefficients": eight_step}])
    assert opt_group.param_groups[0]["ns_steps"] == 8

    # Rejects ns_steps > len(ns_coefficients) at construction.
    with pytest.raises(ValueError, match="exceeds length"):
        Muon([p_g], ns_steps=10, ns_coefficients=eight_step)

    # __setstate__ normalizes list-of-lists from serialized checkpoints back to tuples.
    opt_custom.param_groups[0]["ns_coefficients"] = [list(c) for c in eight_step]
    opt_custom.__setstate__(
        {"state": opt_custom.state, "param_groups": opt_custom.param_groups}
    )
    assert opt_custom.param_groups[0]["ns_coefficients"] == eight_step
