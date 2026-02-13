# SPDX-FileCopyrightText: Copyright 2024-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import pytest
import torch
import gsplat

from gsplat.rendering import RenderMode

device = torch.device("cuda:0")


# Only 3dgs is being tested as per default args with_ut==False and with_eval3d==False
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize(
    "per_view_color,sh_degree,render_mode,packed,batch_dims,with_eval3d,with_ut,extra_signals_info",
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
                [None],  # extra_signals_info
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
                [
                    None,
                    (None, 20),
                    (3, 3),
                ],  # extra_signals_info (extra_signals_sh_degree,extra_signals_size)
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
    extra_signals_info: Optional[tuple],
):
    from gsplat.rendering import _rasterization, rasterization

    torch.manual_seed(42)

    C, N = 3, 10_000
    means = torch.rand(batch_dims + (N, 3), device=device)
    quats = torch.randn(batch_dims + (N, 4), device=device)
    scales = torch.rand(batch_dims + (N, 3), device=device)
    opacities = torch.rand(batch_dims + (N,), device=device)
    if per_view_color:
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

    width, height = 300, 200
    focal = 300.0

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
        extra_signals=extra_signals,
        extra_signals_sh_degree=extra_signals_sh_degree,
    )

    if render_mode == "D":
        assert renders.shape == batch_dims + (C, height, width, 1)
    elif render_mode == "RGB":
        assert renders.shape == batch_dims + (C, height, width, 3)
    elif render_mode == "RGB+D":
        assert renders.shape == batch_dims + (C, height, width, 4)

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
        extra_signals=extra_signals,
        extra_signals_sh_degree=extra_signals_sh_degree,
    )
    # Use more lenient tolerances for eval3d implementation
    rtol = 5e-4 if with_eval3d else 1e-4
    atol = 5e-4 if with_eval3d else 1e-4
    torch.testing.assert_close(renders, _renders, rtol=rtol, atol=atol)
    torch.testing.assert_close(alphas, _alphas, rtol=rtol, atol=atol)
    if extra_signals is not None:
        torch.testing.assert_close(
            meta["render_extra_signals"], _meta["render_extra_signals"]
        )
