# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import inspect
from itertools import product, chain
from types import SimpleNamespace
from typing import Optional, Tuple

import gsplat
import pytest
import torch

from tests.test_cameras import parse_lidar_camera
from gsplat.rendering import (
    RenderMode,
    RendererConfig,
    RendererConfig_MixedBatch,
    RendererConfig_ParallelBatch,
    render_mode_has_color,
    render_mode_has_depth_channel,
    render_mode_has_hit_distance,
)
from gsplat.cuda._constants import ALPHA_THRESHOLD

device = torch.device("cuda:0")

_RENDER_EXECUTION = "render"
_TRAINING_EXECUTION = "training"


def _append_renderer_execution(params, renderer_execution_cases):
    for base_params in params:
        for renderer_config, execution_mode in renderer_execution_cases:
            yield (*base_params, renderer_config, execution_mode)


_MIXED_BATCH_RENDER_CASES = ((RendererConfig_MixedBatch(), _RENDER_EXECUTION),)

_EVAL3D_RENDERER_CASES = (
    (RendererConfig_MixedBatch(), _RENDER_EXECUTION),
    (RendererConfig_MixedBatch(), _TRAINING_EXECUTION),
    (RendererConfig_ParallelBatch(), _RENDER_EXECUTION),
    (RendererConfig_ParallelBatch(), _TRAINING_EXECUTION),
)


def _minimal_rasterization_args():
    return dict(
        means=torch.zeros((1, 3), dtype=torch.float32),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        scales=torch.ones((1, 3), dtype=torch.float32) * 0.01,
        opacities=torch.ones((1,), dtype=torch.float32),
        colors=torch.zeros((1, 3), dtype=torch.float32),
        viewmats=torch.eye(4, dtype=torch.float32).unsqueeze(0),
        Ks=torch.eye(3, dtype=torch.float32).unsqueeze(0),
        width=1,
        height=1,
    )


def test_renderer_config_base_public_api():
    assert gsplat.RendererConfig is RendererConfig
    with pytest.raises(TypeError, match="RendererConfig_MixedBatch"):
        RendererConfig()


@pytest.mark.parametrize(
    "config_type",
    [
        RendererConfig_MixedBatch,
        RendererConfig_ParallelBatch,
    ],
    ids=[
        "mixed_batch",
        "parallel_batch",
    ],
)
def test_renderer_config_public_api(config_type):
    assert getattr(gsplat, config_type.__name__) is config_type
    assert issubclass(config_type, RendererConfig)


def test_rasterization_default_renderer_config():
    renderer_config = inspect.signature(gsplat.rasterization).parameters[
        "renderer_config"
    ]
    assert renderer_config.default is None


class RendererConfig_Future(RendererConfig):
    pass


@pytest.mark.parametrize(
    ("renderer_config", "error_type", "match"),
    [
        pytest.param(object(), TypeError, "renderer_config", id="object"),
        pytest.param(
            RendererConfig_Future(),
            NotImplementedError,
            "RendererConfig_Future",
            id="unsupported_subclass",
        ),
    ],
)
def test_rasterization_rejects_invalid_renderer_config(
    renderer_config, error_type, match
):
    with pytest.raises(error_type, match=match):
        gsplat.rasterization(
            **_minimal_rasterization_args(),
            renderer_config=renderer_config,
        )


def test_rasterization_rejects_parallel_renderer_config_without_eval3d():
    with pytest.raises(ValueError, match="with_eval3d=True"):
        gsplat.rasterization(
            **_minimal_rasterization_args(),
            renderer_config=RendererConfig_ParallelBatch(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_rasterization_3dgut_only_build_shape():
    """Public UT/from-world rasterization must work without GSPLAT_BUILD_3DGS.

    The 3DGUT path routes through the C++ ``rasterization_3dgs`` op, which is
    guarded ``GSPLAT_BUILD_3DGS || GSPLAT_BUILD_3DGUT``. A 3dgut-only build
    (``BUILD_3DGUT=1`` with ``BUILD_3DGS=0`` — a shape Config.h documents) must
    still resolve and run it; this guards against the op being re-narrowed to
    3DGS-only, which silently breaks that build shape. When 3DGS is absent, the
    classic (non-UT) path must reject cleanly rather than fail to resolve the op.
    """
    from gsplat.rendering import rasterization

    device = "cuda"
    torch.manual_seed(0)
    N, C, H, W = 200, 1, 64, 64
    means = torch.randn(N, 3, device=device) * 0.3
    quats = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.rand(N, 3, device=device) * 0.05 + 0.01
    opacities = torch.rand(N, device=device)
    colors = torch.rand(N, 3, device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(C, 1, 1)
    viewmats[:, 2, 3] = 3.0  # world->cam: gaussians near origin land in front (+z)
    focal = 50.0
    Ks = torch.tensor(
        [[[focal, 0.0, W / 2], [0.0, focal, H / 2], [0.0, 0.0, 1.0]]],
        device=device,
    ).repeat(C, 1, 1)

    # The UT / from-world path must resolve and render in any 3DGUT build,
    # including a 3dgut-only one. (packed is invalid with eval3d.)
    with torch.no_grad():
        renders, alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            sh_degree=None,
            render_mode="RGB",
            camera_model="pinhole",
            with_ut=True,
            with_eval3d=True,
            packed=False,
        )
    assert renders.shape == (C, H, W, 3)
    assert alphas.shape == (C, H, W, 1)
    assert torch.isfinite(renders).all()

    # Without 3DGS, the classic (non-UT) projection path must reject cleanly with
    # an actionable error, not fail to resolve the rasterization_3dgs op.
    if not gsplat.has_3dgs():
        with pytest.raises(RuntimeError, match="GSPLAT_BUILD_3DGS"):
            with torch.no_grad():
                rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=W,
                    height=H,
                    sh_degree=None,
                    render_mode="RGB",
                    camera_model="pinhole",
                    with_ut=False,
                    with_eval3d=False,
                    packed=False,
                )


def _rasterization_param_id(value):
    if type(value) is RendererConfig_MixedBatch:
        return "mixed_batch"
    if type(value) is RendererConfig_ParallelBatch:
        return "parallel_batch"
    if value in (_RENDER_EXECUTION, _TRAINING_EXECUTION):
        return value
    return None


def _execution_tensor(tensor: Optional[torch.Tensor], execution_mode: str):
    if tensor is None or execution_mode == _RENDER_EXECUTION:
        return tensor
    assert execution_mode == _TRAINING_EXECUTION
    return tensor.detach().clone().requires_grad_(tensor.is_floating_point())


@pytest.fixture
def sensor_model(
    C: int,
    batch_dims: Tuple[int, ...],
    camera_model: str,
):
    sensor_model = SimpleNamespace()

    if camera_model == "lidar":
        # This test consumes randomness before lidar setup, so fix the lidar
        # param seed explicitly to keep the preprocessing cache reusable.
        lidar_params, angles_to_columns_map, tiling = parse_lidar_camera(
            "at128", batch_dims, 0, 0, device=device, seed=42
        )
        sensor_model.lidar = gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
            lidar_params, angles_to_columns_map, tiling
        )
        sensor_model.width = sensor_model.lidar.n_columns
        sensor_model.height = sensor_model.lidar.n_rows
        focal = sensor_model.width
    else:
        sensor_model.width, sensor_model.height = 300, 200
        focal = 300.0
        sensor_model.lidar = None

    sensor_model.Ks = torch.tensor(
        [
            [focal, 0.0, sensor_model.width / 2.0],
            [0.0, focal, sensor_model.height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
    ).expand(batch_dims + (C, -1, -1))
    sensor_model.viewmats = torch.eye(4, device=device).expand(batch_dims + (C, -1, -1))
    return sensor_model


@pytest.fixture
def gaussians(
    C: int,
    N: int,
    batch_dims: Tuple[int, ...],
    per_view_color: bool,
    sh_degree: Optional[int],
    render_mode: RenderMode,
    extra_signals_info: Optional[tuple],
):
    gaussians = SimpleNamespace()
    gaussians.means = torch.rand(batch_dims + (N, 3), device=device)
    gaussians.quats = torch.randn(batch_dims + (N, 4), device=device)
    gaussians.scales = torch.rand(batch_dims + (N, 3), device=device)
    gaussians.opacities = torch.rand(batch_dims + (N,), device=device)

    # Depth-only modes with no SH and no per_view_color pass colors=None.
    # SH coefficients are deduplicated as (N, K, 3) — shared across batch and
    # camera dims — so per_view_color/batch_dims do not apply when sh_degree is set.
    if not render_mode_has_color(render_mode):
        gaussians.colors = None
    elif sh_degree is not None:
        gaussians.colors = torch.rand((N, (sh_degree + 1) ** 2, 3), device=device)
    elif per_view_color:
        gaussians.colors = torch.rand(batch_dims + (C, N, 3), device=device)
    else:
        gaussians.colors = torch.rand(batch_dims + (N, 3), device=device)

    if extra_signals_info is None:
        gaussians.extra_signals_sh_degree = None
        gaussians.extra_signals = None
    else:
        gaussians.extra_signals_sh_degree = extra_signals_info[0]
        extra_signals_size = extra_signals_info[1]

        if gaussians.extra_signals_sh_degree is not None:
            gaussians.extra_signals = torch.rand(
                (
                    N,
                    (gaussians.extra_signals_sh_degree + 1) ** 2,
                    extra_signals_size,
                ),
                device=device,
            )
        elif per_view_color:
            gaussians.extra_signals = torch.rand(
                batch_dims + (C, N, extra_signals_size), device=device
            )
        else:
            gaussians.extra_signals = torch.rand(
                batch_dims + (N, extra_signals_size), device=device
            )

    return gaussians


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize(
    "per_view_color,sh_degree,render_mode,packed,batch_dims,with_eval3d,with_ut,camera_model,extra_signals_info,distributed,C,N,renderer_config,execution_mode",
    [
        pytest.param(
            *params,
            marks=[
                # test based on with_eval3d (5)  and with_ut (6)
                pytest.mark.skipif(
                    (params[5] == True or params[6] == True) and not gsplat.has_3dgut(),
                    reason="3DGUT support isn't built in",
                ),
                pytest.mark.skipif(
                    (params[5] == False or params[6] == False)
                    and not gsplat.has_3dgs(),
                    reason="3DGS support isn't built in",
                ),
                # sh_degree (1) and per_view_color (0) require colors, which requires a color render_mode (2)
                pytest.mark.skipif(
                    params[1] is not None and not render_mode_has_color(params[2]),
                    reason="sh_degree requires colors, invalid with depth-only render_mode",
                ),
                pytest.mark.skipif(
                    params[0] == True and not render_mode_has_color(params[2]),
                    reason="per_view_color requires colors, invalid with depth-only render_mode",
                ),
                pytest.mark.skipif(
                    params[0] == True and params[1] is not None,
                    reason="per_view_color is a no-op when sh_degree is set; skip the duplicate",
                ),
            ],
        )
        for params in
        # Standard tests: all combinations with with_eval3d=False
        chain(
            _append_renderer_execution(
                product(
                    [True, False],  # per_view_color
                    [None, 3],  # sh_degree
                    ["RGB", "RGB+D", "D"],  # render_mode
                    [True, False],  # packed
                    [(), (2,), (1, 2)],  # batch_dims
                    [False],  # with_eval3d
                    [False],  # with_ut
                    ["pinhole"],  # camera_model
                    [None],  # extra_signals_info
                    [False],  # distributed
                    [3],  # C (number of cameras)
                    [10_000],  # N (number of gaussians)
                ),
                _MIXED_BATCH_RENDER_CASES,
            ),
            # 3DGUT
            _append_renderer_execution(
                product(
                    [True, False],  # per_view_color
                    [None, 3],  # sh_degree
                    ["RGB"],  # render_mode (only RGB)
                    [False],  # packed (must be False)
                    [(), (2,)],  # batch_dims (reduced subset)
                    [True],  # with_eval3d
                    [True],  # with_ut
                    ["pinhole", "lidar"],  # camera_model
                    [
                        None,
                        (None, 20),
                        (3, 3),
                    ],  # extra_signals_info (extra_signals_sh_degree,extra_signals_size)
                    [False],  # distributed
                    [3],  # C (number of cameras)
                    [10_000],  # N (number of gaussians)
                ),
                _EVAL3D_RENDERER_CASES,
            ),
            # 3DGUT hit-distance modes: exercises `use_hit_distance` with the
            # extra-signals plumbing. Channel counts produced (e.g. 24 = 3 RGB
            # + 20 extra + 1 depth) must be present in `GSPLAT_NUM_CHANNELS` —
            # see `pytest.ini` `NUM_CHANNELS`.
            _append_renderer_execution(
                product(
                    [False],  # per_view_color
                    [None],  # sh_degree
                    ["RGB-d", "d"],  # render_mode
                    [False],  # packed (must be False)
                    [()],  # batch_dims
                    [True],  # with_eval3d
                    [True],  # with_ut
                    ["pinhole"],  # camera_model
                    [None, (None, 20)],  # extra_signals_info
                    [False],  # distributed
                    [3],  # C (number of cameras)
                    [10_000],  # N (number of gaussians)
                ),
                _EVAL3D_RENDERER_CASES,
            ),
            # Distributed rendering (single-rank): exercises the all-gather /
            # scatter code path.  Constraints: batch_dims=(), no per_view_color.
            _append_renderer_execution(
                product(
                    [False],  # per_view_color (distributed forbids per-view)
                    [None, 3],  # sh_degree
                    ["RGB", "RGB+D", "D"],  # render_mode
                    [True, False],  # packed
                    [()],  # batch_dims (distributed requires ())
                    [False],  # with_eval3d
                    [False],  # with_ut
                    ["pinhole"],  # camera_model
                    [None],  # extra_signals_info
                    [True],  # distributed
                    [3],  # C (number of cameras)
                    [10_000],  # N (number of gaussians)
                ),
                _MIXED_BATCH_RENDER_CASES,
            ),
        )
    ],
    ids=_rasterization_param_id,
)
def test_rasterization(
    per_view_color: bool,
    sh_degree: Optional[int],
    render_mode: RenderMode,
    packed: bool,
    batch_dims: Tuple[int, ...],
    with_eval3d: bool,
    with_ut: bool,
    camera_model: str,
    extra_signals_info: Optional[tuple],
    distributed: bool,
    C: int,
    N: int,
    renderer_config: RendererConfig,
    execution_mode: str,
    sensor_model: SimpleNamespace,
    gaussians: SimpleNamespace,
    dist_init,
):
    if distributed and not torch.distributed.is_initialized():
        pytest.skip("distributed process group not initialized")
    from gsplat.rendering import _rasterization, rasterization

    torch.manual_seed(42)

    means = _execution_tensor(gaussians.means, execution_mode)
    quats = _execution_tensor(gaussians.quats, execution_mode)
    scales = _execution_tensor(gaussians.scales, execution_mode)
    opacities = _execution_tensor(gaussians.opacities, execution_mode)
    colors = _execution_tensor(gaussians.colors, execution_mode)
    extra_signals = _execution_tensor(gaussians.extra_signals, execution_mode)

    execution_context = (
        torch.enable_grad()
        if execution_mode == _TRAINING_EXECUTION
        else torch.no_grad()
    )
    with execution_context:
        renders, alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=sensor_model.viewmats,
            Ks=sensor_model.Ks,
            width=sensor_model.width,
            height=sensor_model.height,
            sh_degree=sh_degree,
            render_mode=render_mode,
            packed=packed,
            with_eval3d=with_eval3d,
            with_ut=with_ut,
            camera_model=camera_model,
            lidar_coeffs=sensor_model.lidar,
            extra_signals=extra_signals,
            extra_signals_sh_degree=gaussians.extra_signals_sh_degree,
            distributed=distributed,
            renderer_config=renderer_config,
        )

    assert renders.requires_grad == (execution_mode == _TRAINING_EXECUTION)

    with torch.no_grad():
        _renders, _alphas, _meta = _rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=sensor_model.viewmats,
            Ks=sensor_model.Ks,
            width=sensor_model.width,
            height=sensor_model.height,
            sh_degree=sh_degree,
            render_mode=render_mode,
            with_eval3d=with_eval3d,
            with_ut=with_ut,
            camera_model=camera_model,
            lidar_coeffs=sensor_model.lidar,
            extra_signals=extra_signals,
            extra_signals_sh_degree=gaussians.extra_signals_sh_degree,
        )

    expected_channels = 0
    if render_mode_has_color(render_mode):
        expected_channels += 3
    if render_mode_has_depth_channel(render_mode):
        expected_channels += 1
    assert renders.shape == (
        *batch_dims,
        C,
        sensor_model.height,
        sensor_model.width,
        expected_channels,
    )

    rtol = 1e-4
    atol = 1e-4
    if (
        execution_mode == _RENDER_EXECUTION
        and with_eval3d
        and type(renderer_config) is RendererConfig_ParallelBatch
    ):
        # Render execution intentionally runs under no_grad, so public
        # ParallelBatch dispatch uses its fwd-only path. That path skips exact
        # batch-replay and finalizes terminal batches from batch-scan
        # summaries; the priming chain stores transmittance as a rounded fp16
        # upper bound, so pixels near the saturation boundary can differ
        # slightly from the exact-metadata reference.
        rtol = 3e-4
        atol = 3e-4

    torch.testing.assert_close(alphas, _alphas, rtol=rtol, atol=atol)
    if gaussians.extra_signals is not None:
        torch.testing.assert_close(
            meta["render_extra_signals"],
            _meta["render_extra_signals"],
            rtol=rtol,
            atol=atol,
        )

    if render_mode_has_hit_distance(render_mode):
        # For combined RGB+depth modes, verify RGB channels match.
        if render_mode_has_color(render_mode):
            torch.testing.assert_close(
                renders[..., :3], _renders[..., :3], rtol=rtol, atol=atol
            )

        # Verify hit distance is non-zero where pixels are visible.
        mask = alphas[..., 0] > ALPHA_THRESHOLD
        if mask.any():
            hit_dist = renders[..., -1][mask]
            assert (hit_dist > 0).all(), (
                f"Hit-distance is zero for visible pixels "
                f"(render_mode={render_mode}, extra_signals={extra_signals_info})"
            )
    else:
        torch.testing.assert_close(renders, _renders, rtol=rtol, atol=atol)

    if execution_mode == _TRAINING_EXECUTION:
        loss = renders.sum() + alphas.sum()
        if "render_extra_signals" in meta:
            loss = loss + meta["render_extra_signals"].sum()
        loss.backward()

        grad_inputs = (
            ("means", means),
            ("quats", quats),
            ("scales", scales),
            ("opacities", opacities),
            ("colors", colors),
            ("extra_signals", extra_signals),
        )
        for name, tensor in grad_inputs:
            if tensor is None:
                continue
            assert tensor.grad is not None, f"{name} should receive gradients"
            assert torch.isfinite(tensor.grad).all()

        # ParallelBatch shares the batch-parallel backward with MixedBatch but
        # feeds it ParallelBatch-only saved state (the round-major CSR slot map
        # and compose_c_stop). Finite, non-None grads alone would not catch a
        # systematically-wrong gradient from mis-indexing that state, so compare
        # the ParallelBatch grads against a MixedBatch reference run on
        # identical inputs.
        if type(renderer_config) is RendererConfig_ParallelBatch:
            ref_inputs = {
                name: (
                    None
                    if tensor is None
                    else tensor.detach().clone().requires_grad_(tensor.requires_grad)
                )
                for name, tensor in grad_inputs
            }
            with torch.enable_grad():
                ref_renders, ref_alphas, ref_meta = rasterization(
                    means=ref_inputs["means"],
                    quats=ref_inputs["quats"],
                    scales=ref_inputs["scales"],
                    opacities=ref_inputs["opacities"],
                    colors=ref_inputs["colors"],
                    viewmats=sensor_model.viewmats,
                    Ks=sensor_model.Ks,
                    width=sensor_model.width,
                    height=sensor_model.height,
                    sh_degree=sh_degree,
                    render_mode=render_mode,
                    packed=packed,
                    with_eval3d=with_eval3d,
                    with_ut=with_ut,
                    camera_model=camera_model,
                    lidar_coeffs=sensor_model.lidar,
                    extra_signals=ref_inputs["extra_signals"],
                    extra_signals_sh_degree=gaussians.extra_signals_sh_degree,
                    distributed=distributed,
                    renderer_config=RendererConfig_MixedBatch(),
                )
            ref_loss = ref_renders.sum() + ref_alphas.sum()
            if "render_extra_signals" in ref_meta:
                ref_loss = ref_loss + ref_meta["render_extra_signals"].sum()
            ref_loss.backward()

            for name, tensor in grad_inputs:
                if tensor is None:
                    continue
                torch.testing.assert_close(
                    tensor.grad,
                    ref_inputs[name].grad,
                    rtol=1e-3,
                    atol=1e-3,
                    msg=lambda formatted, n=name: (
                        f"ParallelBatch {n} grad diverges from the MixedBatch "
                        f"reference:\n{formatted}"
                    ),
                )


def _make_distributed_validation_scene() -> dict:
    C = 2
    N = 8
    width = 16
    height = 12
    means = torch.rand((N, 3), device=device)
    means[:, 2] = means[:, 2] + 2.0
    return {
        "means": means,
        "quats": torch.randn((N, 4), device=device),
        "scales": torch.rand((N, 3), device=device) * 0.05 + 0.01,
        "opacities": torch.rand((N,), device=device),
        "colors": torch.rand((N, 3), device=device),
        "viewmats": torch.eye(4, device=device).expand(C, -1, -1).clone(),
        "Ks": torch.tensor(
            [[20.0, 0.0, width / 2], [0.0, 20.0, height / 2], [0.0, 0.0, 1.0]],
            device=device,
        )
        .expand(C, -1, -1)
        .clone(),
        "width": width,
        "height": height,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize(
    "case,match",
    [
        ("batch_dims", "batch dimensions"),
        ("with_eval3d", "with_eval3d=True"),
        ("with_ut", "with_ut=True"),
        ("camera_model", "camera_model='ortho'"),
        ("sparse_grad", "sparse_grad=True"),
        ("absgrad", "absgrad=True"),
        ("rolling_shutter", "rolling shutter"),
        ("distortion", "camera distortion"),
        ("global_z_order", "global_z_order=False"),
        ("per_view_color", "per-Gaussian colors"),
        ("rays", "rays"),
        ("return_normals", "return_normals=True"),
        ("lidar", "lidar coefficients"),
    ],
)
def test_rasterization_distributed_rejects_unsupported_configs(
    case: str,
    match: str,
):
    # Every case raises during pure-Python validation before any distributed
    # call, so no NCCL process group is needed.

    kwargs = _make_distributed_validation_scene()
    C = kwargs["viewmats"].shape[0]
    N = kwargs["means"].shape[0]

    if case == "batch_dims":
        for key in ("means", "quats", "scales", "opacities", "colors"):
            kwargs[key] = kwargs[key].expand(2, *kwargs[key].shape).clone()
        kwargs["viewmats"] = kwargs["viewmats"].expand(2, C, 4, 4).clone()
        kwargs["Ks"] = kwargs["Ks"].expand(2, C, 3, 3).clone()
    elif case == "with_eval3d":
        kwargs["with_eval3d"] = True
    elif case == "with_ut":
        kwargs["with_ut"] = True
    elif case == "camera_model":
        kwargs["camera_model"] = "ortho"
    elif case == "sparse_grad":
        kwargs["sparse_grad"] = True
    elif case == "absgrad":
        kwargs["absgrad"] = True
    elif case == "rays":
        kwargs["rays"] = torch.zeros(
            (C, kwargs["height"], kwargs["width"], 6), device=device
        )
    elif case == "rolling_shutter":
        kwargs["rolling_shutter"] = gsplat.RollingShutterType.ROLLING_TOP_TO_BOTTOM
        kwargs["viewmats_rs"] = kwargs["viewmats"].clone()
    elif case == "distortion":
        kwargs["radial_coeffs"] = torch.zeros((C, 6), device=device)
    elif case == "global_z_order":
        kwargs["global_z_order"] = False
    elif case == "per_view_color":
        kwargs["colors"] = torch.rand((C, N, 3), device=device)
    elif case == "return_normals":
        kwargs["return_normals"] = True
    elif case == "lidar":
        from types import SimpleNamespace

        # SimpleNamespace satisfies the n_columns/n_rows read that happens before
        # the distributed-rejection block; the ValueError fires before any deeper
        # lidar processing that would require a real LiDAR object.
        kwargs["lidar_coeffs"] = SimpleNamespace(
            n_columns=kwargs["width"], n_rows=kwargs["height"]
        )
    else:
        raise AssertionError(f"unhandled case {case}")

    with pytest.raises(ValueError, match=match):
        gsplat.rasterization(**kwargs, distributed=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_rasterization_external_distortion_requires_ut():
    kwargs = _make_distributed_validation_scene()
    # The validation fires on `external_distortion_coeffs is not None` before
    # any CUDA call, so any sentinel object suffices here.
    kwargs["external_distortion_coeffs"] = object()
    with pytest.raises(
        (ValueError, AssertionError),
        match="with_ut=True",
    ):
        gsplat.rasterization(**kwargs, with_ut=False)
