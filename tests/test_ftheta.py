"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

from typing import Optional, Tuple

import pytest
import torch

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("per_view_color", [True])
@pytest.mark.parametrize("sh_degree", [None])
@pytest.mark.parametrize("render_mode", ["RGB"])
@pytest.mark.parametrize("batch_dims", [(1, 2)])
def test_rasterization(
    per_view_color: bool,
    sh_degree: Optional[int],
    render_mode: str,
    batch_dims: Tuple[int, ...],
):
    from gsplat.rendering import (
        rasterization,
        FThetaCameraDistortionParameters,
        FThetaPolynomialType,
    )

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
    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    ).expand(batch_dims + (C, -1, -1))
    viewmats = torch.eye(4, device=device).expand(batch_dims + (C, -1, -1))

    # distortion parameters
    camera_model = "fisheye"
    distortion_params = FThetaCameraDistortionParameters(
        reference_poly=FThetaPolynomialType.ANGLE_TO_PIXELDIST,
        pixeldist_to_angle_poly=(
            0.0,
            8.4335003e-03,
            2.3174282e-06,
            -5.0478608e-08,
            6.1392608e-10,
            -1.7447865e-12,
        ),
        angle_to_pixeldist_poly=(
            0.0,
            118.43232,
            -2.562147,
            6.317949,
            -10.41861,
            3.6694396,
        ),
        max_angle=1.3446016557364315,
        linear_cde=(9.9968284e-01, 1.8735906e-05, 1.7659619e-05),
    )

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
        camera_model=camera_model,
        packed=False,
        ftheta_coeffs=distortion_params,
        with_ut=True,
        with_eval3d=True,
    )

    if render_mode == "D":
        assert renders.shape == batch_dims + (C, height, width, 1)
    elif render_mode == "RGB":
        assert renders.shape == batch_dims + (C, height, width, 3)
    elif render_mode == "RGB+D":
        assert renders.shape == batch_dims + (C, height, width, 4)
