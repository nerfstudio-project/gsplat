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


# Only 3dgs is being tested as per default args with_ut==False and with_eval3d==False
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize(
    "per_view_color,sh_degree,render_mode,packed,batch_dims,with_eval3d,with_ut,camera_model,extra_signals_info,distributed",
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
            ),
            # 3DGUT hit-distance modes: exercises the padding + use_hit_distance
            # interaction in rasterize_to_pixels_eval3d.  The (None,20) extra
            # signals case is the critical one — it produces 24 channels
            # (3 RGB + 20 extra + 1 depth) which gets padded to CDIM=32.
            product(
                [False],  # per_view_color
                [None],  # sh_degree
                ["RGB-d", "d"],  # render_mode
                [False],  # packed (must be False)
                [()],  # batch_dims
                [True],  # with_eval3d
                [True],  # with_ut
                ["pinhole"],  # camera_model
                [None, (None, 20)],  # extra_signals_info — (None,20) triggers padding
                [False],  # distributed
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
    dist_init,
):
    if distributed and not torch.distributed.is_initialized():
        pytest.skip("distributed process group not initialized")
    from gsplat.rendering import _rasterization, rasterization

    torch.manual_seed(42)

    C, N = 3, 10_000
    means = torch.rand(batch_dims + (N, 3), device=device)
    quats = torch.randn(batch_dims + (N, 4), device=device)
    scales = torch.rand(batch_dims + (N, 3), device=device)
    opacities = torch.rand(batch_dims + (N,), device=device)
    # Depth-only modes with no SH and no per_view_color pass colors=None.
    # (per_view_color and sh_degree combos are already filtered by skipif above.)
    if not render_mode_has_color(render_mode):
        colors = None
    elif per_view_color:
        if sh_degree is None:
            colors = torch.rand(batch_dims + (C, N, 3), device=device)
        else:
            colors = torch.rand(
                batch_dims + (C, N, (sh_degree + 1) ** 2, 3), device=device
            )
    else:
        if sh_degree is None:
            colors = torch.rand(batch_dims + (N, 3), device=device)
        else:
            colors = torch.rand(
                batch_dims + (N, (sh_degree + 1) ** 2, 3), device=device
            )

    if extra_signals_info is None:
        extra_signals_sh_degree = None
        extra_signals = None
    else:
        extra_signals_sh_degree = extra_signals_info[0]
        extra_signals_size = extra_signals_info[1]

        if per_view_color:
            if extra_signals_sh_degree is None:
                extra_signals = torch.rand(
                    batch_dims + (C, N, extra_signals_size), device=device
                )
            else:
                extra_signals = torch.rand(
                    batch_dims
                    + (C, N, (extra_signals_sh_degree + 1) ** 2, extra_signals_size),
                    device=device,
                )
        else:
            if extra_signals_sh_degree is None:
                extra_signals = torch.rand(
                    batch_dims + (N, extra_signals_size), device=device
                )
            else:
                extra_signals = torch.rand(
                    batch_dims
                    + (N, (extra_signals_sh_degree + 1) ** 2, extra_signals_size),
                    device=device,
                )

    if camera_model == "lidar":
        # This test consumes randomness before lidar setup, so fix the lidar
        # param seed explicitly to keep the preprocessing cache reusable.
        lidar_params, angles_to_columns_map, tiling = parse_lidar_camera(
            "at128", batch_dims, 0, 0, device=device, seed=42
        )
        lidar = gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
            lidar_params, angles_to_columns_map, tiling
        )
        width = lidar.n_columns
        height = lidar.n_rows
        focal = width
    else:
        width, height = 300, 200
        focal = 300.0
        lidar = None

    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    ).expand(batch_dims + (C, -1, -1))
    viewmats = torch.eye(4, device=device).expand(batch_dims + (C, -1, -1))

    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
        packed=packed,
        with_eval3d=with_eval3d,
        with_ut=with_ut,
        camera_model=camera_model,
        lidar_coeffs=lidar,
        extra_signals=extra_signals,
        extra_signals_sh_degree=extra_signals_sh_degree,
        distributed=distributed,
    )

    expected_channels = 0
    if render_mode_has_color(render_mode):
        expected_channels += 3
    if render_mode_has_depth_channel(render_mode):
        expected_channels += 1
    assert renders.shape == (*batch_dims, C, height, width, expected_channels)

    _renders, _alphas, _meta = _rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
        with_eval3d=with_eval3d,
        with_ut=with_ut,
        camera_model=camera_model,
        lidar_coeffs=lidar,
        extra_signals=extra_signals,
        extra_signals_sh_degree=extra_signals_sh_degree,
    )

    rtol = 1e-4
    atol = 1e-4

    torch.testing.assert_close(alphas, _alphas, rtol=rtol, atol=atol)
    if extra_signals is not None:
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
