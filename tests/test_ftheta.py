"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import os
from typing import Tuple

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")


def expand(data: dict, batch_dims: Tuple[int, ...]):
    # append multiple batch dimensions to the front of the tensor
    # eg. x.shape = [N, 3], batch_dims = (1, 2), return shape is [1, 2, N, 3]
    # eg. x.shape = [N, 3], batch_dims = (), return shape is [N, 3]
    ret = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and len(batch_dims) > 0:
            new_shape = batch_dims + v.shape
            ret[k] = v.expand(new_shape)
        else:
            ret[k] = v
    return ret


@pytest.fixture
def test_data():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    return {
        "means": means,  # [N, 3]
        "quats": quats,  # [N, 4]
        "scales": scales,  # [N, 3]
        "opacities": opacities,  # [N]
        "viewmats": viewmats,  # [C, 4, 4]
        "Ks": Ks,  # [C, 3, 3]
        "colors": colors,  # [C, N, 3]
        "width": width,
        "height": height,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("render_mode", ["RGB"])
def test_rasterization(
    test_data,
    render_mode: str,
):
    from gsplat.rendering import (
        rasterization,
        FThetaCameraDistortionParameters,
        FThetaPolynomialType,
    )

    torch.manual_seed(42)
    C = test_data["Ks"].shape[0]
    N = test_data["opacities"].shape[0]

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]
    opacities = test_data["opacities"]
    colors = test_data["colors"].repeat(C, 1, 1)

    # distortion parameters
    camera_model = "ftheta"
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
        max_angle=1000,
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
        render_mode=render_mode,
        camera_model=camera_model,
        packed=False,
        ftheta_coeffs=distortion_params,
        with_ut=True,
        with_eval3d=True,
    )

    if render_mode == "D":
        assert renders.shape == (C, height, width, 1)
    elif render_mode == "RGB":
        assert renders.shape == (C, height, width, 3)
    elif render_mode == "RGB+D":
        assert renders.shape == (C, height, width, 4)

    # save the renders
    # import imageio
    # import numpy as np
    # imageio.imwrite(
    #     "test_ftheta.png", (renders[0, :, :, :3].cpu().numpy() * 255).astype(np.uint8)
    # )
