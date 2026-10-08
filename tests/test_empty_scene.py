# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise empty scenes in subprocesses so native crashes cannot kill pytest."""

import os
from datetime import timedelta
from itertools import product
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _run_worker(renderer, packed, *args):
    env = os.environ.copy()
    repo = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([repo, env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, __file__, renderer, str(int(packed)), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"{renderer}, packed={packed}: exit {result.returncode}\n"
        f"{result.stdout}\n{result.stderr}"
    )


@pytest.mark.parametrize("renderer", ["3dgs", "2dgs"])
@pytest.mark.parametrize("packed", [False, True])
def test_rasterization_empty_scene(renderer, packed):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import gsplat

    if not getattr(gsplat, f"has_{renderer}")():
        pytest.skip(f"{renderer} support wasn't built")
    _run_worker(renderer, packed)


@pytest.mark.parametrize("packed", [False, True])
def test_rasterization_distributed_empty_scene(packed, tmp_path):
    if torch.cuda.device_count() < 2 or not dist.is_nccl_available():
        pytest.skip("two CUDA devices and NCCL required")
    import gsplat

    if not gsplat.has_3dgs():
        pytest.skip("3DGS support wasn't built")
    _run_worker("distributed", packed, (tmp_path / "rendezvous").as_uri())


def test_rasterization_ut_empty_scene():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import gsplat

    if not gsplat.has_3dgut():
        pytest.skip("3DGUT support wasn't built")
    _run_worker("ut", False)


def _check_empty_scene(
    renderer,
    packed,
    batch_dims,
    cameras,
    layout,
    mode,
    with_background,
    depth_mode="expected",
):
    import gsplat

    device = torch.device("cuda:0")
    width, height, channels = 16, 16, 3
    means = torch.empty((*batch_dims, 0, 3), device=device, requires_grad=True)
    quats = torch.empty((*batch_dims, 0, 4), device=device, requires_grad=True)
    scales = torch.empty_like(means, requires_grad=True)
    opacities = torch.empty((*batch_dims, 0), device=device, requires_grad=True)
    color_shape = {
        "shared": (*batch_dims, 0, channels),
        "per_view": (*batch_dims, cameras, 0, channels),
        "sh": (0, 1, channels),
    }[layout]
    colors = torch.empty(color_shape, device=device, requires_grad=True)
    has_color = mode.startswith("RGB")
    has_depth = mode != "RGB"
    sh_degree = 0 if layout == "sh" and has_color else None
    viewmats = torch.eye(4, device=device).repeat(*batch_dims, cameras, 1, 1)
    viewmats.requires_grad_()
    Ks = torch.tensor(
        [[16.0, 0.0, 8.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]], device=device
    ).repeat(*batch_dims, cameras, 1, 1)
    backgrounds = None
    if with_background:
        backgrounds = torch.rand(
            (*batch_dims, cameras, channels), device=device, requires_grad=True
        )
    kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors if has_color or renderer == "2dgs" else None,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        packed=packed,
        sh_degree=sh_degree,
        backgrounds=backgrounds,
        render_mode=mode,
    )
    if renderer == "2dgs":
        kwargs["distloss"] = has_depth
        kwargs["depth_mode"] = depth_mode
        result = gsplat.rasterization_2dgs(**kwargs)
    else:
        result = gsplat.rasterization(**kwargs)
    renders, alphas, meta = result[0], result[1], result[-1]
    image_shape = (*batch_dims, cameras, height, width)
    output_channels = (channels if has_color else 0) + int(has_depth)
    assert renders.shape == (*image_shape, output_channels)
    assert alphas.shape == (*image_shape, 1)
    assert renders.dtype == means.dtype and renders.device == device
    assert alphas.dtype == means.dtype and alphas.device == device
    expected = torch.zeros_like(renders)
    if backgrounds is not None and has_color:
        expected[..., :channels] = backgrounds[..., None, None, :]
    torch.testing.assert_close(renders, expected, rtol=0, atol=0)
    torch.testing.assert_close(alphas, torch.zeros_like(alphas), rtol=0, atol=0)
    projection_shape = (0,) if packed else (*batch_dims, cameras, 0)
    assert meta["radii"].shape == (*projection_shape, 2)
    assert meta["means2d"].shape == (*projection_shape, 2)
    assert meta["depths"].shape == projection_shape
    assert meta["opacities"].shape == projection_shape
    assert meta["tiles_per_gauss"].shape == projection_shape
    assert meta["isect_ids"].shape == (0,)
    assert meta["flatten_ids"].shape == (0,)
    assert meta["isect_offsets"].shape == (
        *batch_dims,
        cameras,
        meta["tile_height"],
        meta["tile_width"],
    )
    assert torch.count_nonzero(meta["isect_offsets"]) == 0
    assert meta["n_cameras"] == cameras
    for name in ("camera_ids", "gaussian_ids"):
        if packed:
            assert meta[name].shape == (0,)
    if renderer == "2dgs":
        for index, shape in (
            (2, (*image_shape, 3)),
            (4, (*image_shape, 1)),
            (5, (*image_shape, 1)),
        ):
            assert result[index].shape == shape
            torch.testing.assert_close(
                result[index], torch.zeros_like(result[index]), rtol=0, atol=0
            )
        if has_color and has_depth:
            assert result[3] is not None
            torch.testing.assert_close(
                result[3], torch.zeros_like(result[3]), rtol=0, atol=0
            )
        assert meta["ray_transforms"].shape == (*projection_shape, 3, 3)
        assert meta["gradient_2dgs"].shape == (*projection_shape, 2)
    else:
        assert meta["conics"].shape == (*projection_shape, 3)

    (renders.sum() + alphas.sum()).backward()
    for tensor in (means, quats, scales, opacities):
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape
        assert tensor.grad.numel() == 0
    if has_color:
        assert colors.grad is not None and colors.grad.shape == colors.shape
    torch.testing.assert_close(
        viewmats.grad, torch.zeros_like(viewmats), rtol=0, atol=0
    )
    if backgrounds is not None and has_color:
        torch.testing.assert_close(
            backgrounds.grad,
            torch.full_like(backgrounds, height * width),
            rtol=0,
            atol=0,
        )


def _empty_scene_worker(renderer, packed):
    cases = product(
        [(), (2,), (1, 2)],
        [1, 2],
        ["shared", "per_view", "sh"],
        ["RGB", "D", "ED", "RGB+D", "RGB+ED"],
        [False, True],
    )
    for batch_dims, cameras, layout, mode, with_background in cases:
        if mode in ("D", "ED") and layout != "shared":
            continue
        case = (renderer, packed, batch_dims, cameras, layout, mode, with_background)
        print(case, flush=True)
        _check_empty_scene(*case)
    if renderer == "2dgs":
        for mode in ("RGB+D", "RGB+ED"):
            _check_empty_scene(renderer, packed, (1, 2), 2, "sh", mode, True, "median")


def _distributed_worker(rank, init_method, packed):
    import gsplat

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        for all_empty in (False, True):
            count = 0 if all_empty else 1
            scene = dict(
                means=torch.tensor([[0.0, 0.0, 3.0]], device=device)[:count],
                quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)[:count],
                scales=torch.full((count, 3), 0.15, device=device),
                opacities=torch.full((count,), 0.8, device=device),
                colors=torch.full((count, 3), 0.5, device=device),
            )
            viewmats = torch.eye(4, device=device)[None]
            viewmats[:, 0, 3] = rank * 0.1
            Ks = torch.tensor(
                [[[16.0, 0.0, 8.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]]], device=device
            )
            static = dict(viewmats=viewmats, Ks=Ks, width=16, height=16, packed=packed)
            reference_inputs = {
                key: value.clone().requires_grad_() for key, value in scene.items()
            }
            local_count = count if rank == 1 else 0
            local_inputs = {
                key: value[:local_count].clone().requires_grad_()
                for key, value in scene.items()
            }
            reference = gsplat.rasterization(**reference_inputs, **static)
            rendered = gsplat.rasterization(**local_inputs, **static, distributed=True)
            for actual, expected in zip(rendered[:2], reference[:2]):
                torch.testing.assert_close(actual, expected)
            # Every rank renders a camera; the centralized gradient is their sum.
            (reference[0].sum() + reference[1].sum()).backward()
            for tensor in reference_inputs.values():
                dist.all_reduce(tensor.grad)
            (rendered[0].sum() + rendered[1].sum()).backward()
            for key, tensor in local_inputs.items():
                assert tensor.grad is not None
                torch.testing.assert_close(
                    tensor.grad, reference_inputs[key].grad[:local_count]
                )
    finally:
        dist.destroy_process_group()


def _ut_worker():
    import gsplat

    device = torch.device("cuda:0")
    backgrounds = torch.rand((2, 2, 3), device=device)
    kwargs = dict(
        means=torch.empty((2, 0, 3), device=device),
        quats=torch.empty((2, 0, 4), device=device),
        scales=torch.empty((2, 0, 3), device=device),
        opacities=torch.empty((2, 0), device=device),
        colors=torch.empty((2, 0, 3), device=device),
        viewmats=torch.eye(4, device=device).repeat(2, 2, 1, 1),
        Ks=torch.tensor(
            [[16.0, 0.0, 8.0], [0.0, 16.0, 8.0], [0.0, 0.0, 1.0]], device=device
        ).repeat(2, 2, 1, 1),
        width=16,
        height=16,
        backgrounds=backgrounds,
        render_mode="RGB+ED",
        packed=False,
        with_ut=True,
    )
    for with_eval3d in [False, True] if gsplat.has_3dgs() else [True]:
        renders, alphas, meta = gsplat.rasterization(
            **kwargs, with_eval3d=with_eval3d, return_normals=with_eval3d
        )
        expected = torch.zeros((2, 2, 16, 16, 4), device=device)
        expected[..., :3] = backgrounds[..., None, None, :]
        torch.testing.assert_close(renders, expected, rtol=0, atol=0)
        torch.testing.assert_close(alphas, torch.zeros_like(alphas), rtol=0, atol=0)
        if with_eval3d:
            torch.testing.assert_close(
                meta["normals"],
                torch.zeros_like(meta["normals"]),
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    renderer, packed = sys.argv[1], bool(int(sys.argv[2]))
    if renderer == "distributed":
        mp.spawn(_distributed_worker, args=(sys.argv[3], packed), nprocs=2, join=True)
    elif renderer == "ut":
        _ut_worker()
    else:
        _empty_scene_worker(renderer, packed)
