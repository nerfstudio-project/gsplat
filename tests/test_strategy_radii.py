# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gsplat import DefaultStrategy


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="No CUDA device"
            ),
        ),
    ]
)
def tensor_device(request):
    return torch.device(request.param)


def _make_params(n, device):
    means = torch.zeros(n, 3, device=device)
    means[:, 0] = torch.arange(n, device=device)
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(torch.full_like(means, 0.02).log()),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(torch.zeros(n, device=device)),
        }
    )
    optimizers = {name: torch.optim.Adam([p]) for name, p in params.items()}
    return params, optimizers


def _observe(strategy, params, optimizers, state, radii, packed, step=1):
    n_cameras = radii.shape[0]
    if packed:
        visible = (radii > 0).all(dim=-1)
        gaussian_ids = torch.where(visible)[1]
        radii = radii[visible]
    else:
        gaussian_ids = None
    means2d = torch.zeros(*radii.shape[:-1], 2, device=radii.device, requires_grad=True)
    info = {
        "width": 100,
        "height": 50,
        "n_cameras": n_cameras,
        "radii": radii,
        "gaussian_ids": gaussian_ids,
        "means2d": means2d,
    }
    strategy.step_pre_backward(params, optimizers, state, step, info)
    means2d[..., 0].sum().backward()
    strategy.step_post_backward(params, optimizers, state, step, info, packed=packed)


def _camera_radii(device):
    return torch.tensor(
        [
            [[80, 30], [10, 0], [5, 10], [0, 0]],
            [[12, 3], [0, 9], [60, 20], [0, 0]],
            [[7, 4], [3, 15], [4, 8], [0, 0]],
        ],
        dtype=torch.int32,
        device=device,
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("camera_order", [(0, 1, 2), (2, 1, 0)])
def test_screen_radii_reduce_all_camera_observations(
    tensor_device, packed, camera_order
):
    strategy = DefaultStrategy(refine_scale2d_stop_iter=100)
    params, optimizers = _make_params(4, tensor_device)
    state = strategy.initialize_state()
    radii = _camera_radii(tensor_device)[list(camera_order)]

    _observe(strategy, params, optimizers, state, radii, packed)

    torch.testing.assert_close(
        state["radii"], torch.tensor([0.8, 0.15, 0.6, 0.0], device=tensor_device)
    )
    # Gradient sums and visibility counts still include every visible observation.
    torch.testing.assert_close(
        state["count"], torch.tensor([3.0, 1.0, 3.0, 0.0], device=tensor_device)
    )
    torch.testing.assert_close(
        state["grad2d"], torch.tensor([450.0, 150.0, 450.0, 0.0], device=tensor_device)
    )


@pytest.mark.parametrize("packed", [False, True])
def test_screen_radii_keep_history_and_unobserved_gaussians(tensor_device, packed):
    strategy = DefaultStrategy(refine_scale2d_stop_iter=100)
    params, optimizers = _make_params(4, tensor_device)
    state = strategy.initialize_state()
    state["radii"] = torch.tensor([0.9, 0.25, 0.35, 0.45], device=tensor_device)

    _observe(strategy, params, optimizers, state, _camera_radii(tensor_device), packed)

    torch.testing.assert_close(
        state["radii"], torch.tensor([0.9, 0.25, 0.6, 0.45], device=tensor_device)
    )
    before = {key: state[key].clone() for key in ("radii", "count", "grad2d")}
    _observe(
        strategy,
        params,
        optimizers,
        state,
        torch.zeros(2, 4, 2, dtype=torch.int32, device=tensor_device),
        packed,
        step=2,
    )
    for key, value in before.items():
        torch.testing.assert_close(state[key], value)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("refinement", ["split", "prune"])
def test_largest_camera_radius_controls_refinement(tensor_device, packed, refinement):
    strategy = DefaultStrategy(
        grow_grad2d=float("inf"),
        grow_scale2d=0.15 if refinement == "split" else 1.0,
        prune_scale2d=0.15,
        refine_scale2d_stop_iter=10,
        refine_start_iter=0,
        refine_every=1,
        reset_every=100 if refinement == "split" else 1,
    )
    params, optimizers = _make_params(2, tensor_device)
    strategy.check_sanity(params, optimizers)
    # Initialize real Adam moments so refinement also exercises optimizer resizing.
    sum(p.square().sum() for p in params.values()).backward()
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad()
    surviving_mean = params["means"][1].detach().clone()
    state = strategy.initialize_state()
    radii = torch.tensor(
        [[[20, 18], [1, 1]], [[1, 1], [1, 1]]],
        dtype=torch.int32,
        device=tensor_device,
    )

    _observe(strategy, params, optimizers, state, radii, packed, step=2)

    expected_count = 3 if refinement == "split" else 1
    for name, param in params.items():
        assert len(param) == expected_count
        assert optimizers[name].param_groups[0]["params"][0] is param
        for key in ("exp_avg", "exp_avg_sq"):
            assert optimizers[name].state[param][key].shape == param.shape
    if refinement == "prune":
        torch.testing.assert_close(params["means"][0], surviving_mean)
    for key in ("radii", "count", "grad2d"):
        torch.testing.assert_close(
            state[key], torch.zeros(expected_count, device=tensor_device)
        )
    # Optimization must remain usable after the split/prune permutation.
    sum(p.square().sum() for p in params.values()).backward()
    for optimizer in optimizers.values():
        optimizer.step()


@pytest.mark.parametrize("packed", [False, True])
def test_disabled_screen_refinement_does_not_track_radii(tensor_device, packed):
    strategy = DefaultStrategy()
    params, optimizers = _make_params(4, tensor_device)
    state = strategy.initialize_state()

    _observe(strategy, params, optimizers, state, _camera_radii(tensor_device), packed)

    assert "radii" not in state
    torch.testing.assert_close(
        state["count"], torch.tensor([3.0, 1.0, 3.0, 0.0], device=tensor_device)
    )
