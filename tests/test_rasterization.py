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

from itertools import product, chain
from types import SimpleNamespace
from typing import Optional, Tuple

import gsplat
import pytest
import torch

from tests.test_cameras import parse_lidar_camera
from gsplat.rendering import (
    RenderMode,
    render_mode_has_color,
    render_mode_has_depth_channel,
    render_mode_has_hit_distance,
)
from gsplat.cuda._constants import ALPHA_THRESHOLD

device = torch.device("cuda:0")


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
    "per_view_color,sh_degree,render_mode,packed,batch_dims,with_eval3d,with_ut,camera_model,extra_signals_info,distributed,C,N",
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
            # 3DGUT
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
            # 3DGUT hit-distance modes: exercises `use_hit_distance` with the
            # extra-signals plumbing. Channel counts produced (e.g. 24 = 3 RGB
            # + 20 extra + 1 depth) must be present in `GSPLAT_NUM_CHANNELS` —
            # see `pytest.ini` `NUM_CHANNELS`.
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
            # Distributed rendering (single-rank): exercises the all-gather /
            # scatter code path.  Constraints: batch_dims=(), no per_view_color.
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
        )
    ],
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
    sensor_model: SimpleNamespace,
    gaussians: SimpleNamespace,
    dist_init,
):
    if distributed and not torch.distributed.is_initialized():
        pytest.skip("distributed process group not initialized")
    from gsplat.rendering import _rasterization, rasterization

    torch.manual_seed(42)

    renders, alphas, meta = rasterization(
        means=gaussians.means,
        quats=gaussians.quats,
        scales=gaussians.scales,
        opacities=gaussians.opacities,
        colors=gaussians.colors,
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
        extra_signals=gaussians.extra_signals,
        extra_signals_sh_degree=gaussians.extra_signals_sh_degree,
        distributed=distributed,
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

    _renders, _alphas, _meta = _rasterization(
        means=gaussians.means,
        quats=gaussians.quats,
        scales=gaussians.scales,
        opacities=gaussians.opacities,
        colors=gaussians.colors,
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
        extra_signals=gaussians.extra_signals,
        extra_signals_sh_degree=gaussians.extra_signals_sh_degree,
    )

    rtol = 1e-4
    atol = 1e-4

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
