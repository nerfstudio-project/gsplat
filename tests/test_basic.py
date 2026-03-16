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

import math
import struct
import os
from itertools import chain, product
from types import SimpleNamespace

import pytest
import torch
from typing_extensions import Literal, Tuple, assert_never
import torch.nn.functional as F

from gsplat._helper import (
    load_test_data,
    get_inlier_abserror_mask,
    assert_mismatch_ratio,
)
import gsplat

from gsplat.cuda._backend import _C

if _C is None:
    pytest.skip("gsplat CUDA extension not available", allow_module_level=True)

from gsplat._helper import (
    load_test_data,
    get_inlier_abserror_mask,
    assert_mismatch_ratio,
)

from gsplat.cuda._wrapper import (
    CameraModel,
    RollingShutterType,
    UnscentedTransformParameters,
    _make_lazy_cuda_cls,
    has_camera_wrappers,
    create_camera_model,
)
from gsplat.cuda._math import _safe_normalize
from gsplat.cuda._torch_cameras import _viewmat_to_pose
from gsplat.cuda._constants import ALPHA_THRESHOLD
from tests.test_cameras import parse_lidar_camera
from gsplat.cuda._torch_impl_lidar import ANGLE_TO_PIXEL_SCALING_FACTOR

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
        "width": width,
        "height": height,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("triu", [False, True])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_quat_scale_to_covar_preci(test_data, triu: bool, batch_dims: Tuple[int, ...]):
    from gsplat.cuda._math import _quat_scale_to_covar_preci
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci

    torch.manual_seed(42)

    test_data = expand(test_data, batch_dims)
    quats = test_data["quats"]
    scales = test_data["scales"]
    quats.requires_grad = True
    scales.requires_grad = True

    # forward
    covars, precis = quat_scale_to_covar_preci(quats, scales, triu=triu)
    _covars, _precis = _quat_scale_to_covar_preci(quats, scales, triu=triu)
    torch.testing.assert_close(covars, _covars)
    # This test is disabled because the numerical instability.
    # torch.testing.assert_close(precis, _precis, rtol=2e-2, atol=1e-2)
    # if not triu:
    #     I = torch.eye(3, device=device).expand(len(covars), 3, 3)
    #     torch.testing.assert_close(torch.bmm(covars, precis), I)
    #     torch.testing.assert_close(torch.bmm(precis, covars), I)

    # backward
    v_covars = torch.randn_like(covars)
    v_precis = torch.randn_like(precis) * 0.01
    v_quats, v_scales = torch.autograd.grad(
        (covars * v_covars + precis * v_precis).sum(),
        (quats, scales),
    )
    _v_quats, _v_scales = torch.autograd.grad(
        (_covars * v_covars + _precis * v_precis).sum(),
        (quats, scales),
    )
    torch.testing.assert_close(v_quats, _v_quats, rtol=1e0, atol=1e-1)
    torch.testing.assert_close(v_scales, _v_scales, rtol=1e0, atol=1e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_proj(
    test_data,
    camera_model: CameraModel,
    batch_dims: Tuple[int, ...],
):
    from gsplat.cuda._torch_impl import (
        _fisheye_proj,
        _ortho_proj,
        _persp_proj,
        _world_to_cam,
    )
    from gsplat.cuda._wrapper import proj, quat_scale_to_covar_preci

    torch.manual_seed(42)

    test_data = expand(test_data, batch_dims)
    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]

    covars, _ = quat_scale_to_covar_preci(test_data["quats"], test_data["scales"])
    means, covars = _world_to_cam(test_data["means"], covars, viewmats)
    means.requires_grad = True
    covars.requires_grad = True

    # forward
    means2d, covars2d = proj(means, covars, Ks, width, height, camera_model)
    if camera_model == "ortho":
        _means2d, _covars2d = _ortho_proj(means, covars, Ks, width, height)
    elif camera_model == "fisheye":
        _means2d, _covars2d = _fisheye_proj(means, covars, Ks, width, height)
    elif camera_model == "pinhole":
        _means2d, _covars2d = _persp_proj(means, covars, Ks, width, height)
    else:
        assert_never(camera_model)

    torch.testing.assert_close(means2d, _means2d, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(covars2d, _covars2d, rtol=1e-1, atol=3e-2)

    # backward
    v_means2d = torch.randn_like(means2d)
    v_covars2d = torch.randn_like(covars2d)
    v_means, v_covars = torch.autograd.grad(
        (means2d * v_means2d).sum() + (covars2d * v_covars2d).sum(),
        (means, covars),
    )
    _v_means, _v_covars = torch.autograd.grad(
        (_means2d * v_means2d).sum() + (_covars2d * v_covars2d).sum(),
        (means, covars),
    )
    torch.testing.assert_close(v_means, _v_means, rtol=6e-1, atol=1e-2)
    torch.testing.assert_close(v_covars, _v_covars, rtol=1e-1, atol=1e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("calc_compensations", [True, False])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_projection(
    test_data,
    fused: bool,
    calc_compensations: bool,
    camera_model: CameraModel,
    batch_dims: Tuple[int, ...],
):
    from gsplat.cuda._torch_impl import _fully_fused_projection
    from gsplat.cuda._wrapper import fully_fused_projection, quat_scale_to_covar_preci

    torch.manual_seed(42)

    test_data = expand(test_data, batch_dims)
    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]

    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True

    # forward
    if fused:
        radii, means2d, depths, conics, compensations = fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
    else:
        covars, _ = quat_scale_to_covar_preci(quats, scales, triu=True)  # [..., N, 6]
        radii, means2d, depths, conics, compensations = fully_fused_projection(
            means,
            covars,
            None,
            None,
            viewmats,
            Ks,
            width,
            height,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
    _covars, _ = quat_scale_to_covar_preci(quats, scales, triu=False)  # [..., N, 3, 3]
    _radii, _means2d, _depths, _conics, _compensations = _fully_fused_projection(
        means,
        _covars,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=calc_compensations,
        camera_model=camera_model,
    )

    # radii is integer so we allow for 1 unit difference
    valid = (radii > 0).all(dim=-1) & (_radii > 0).all(dim=-1)
    torch.testing.assert_close(radii, _radii, rtol=0, atol=1)
    torch.testing.assert_close(means2d[valid], _means2d[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(depths[valid], _depths[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(conics[valid], _conics[valid], rtol=1e-4, atol=1e-4)
    if calc_compensations:
        torch.testing.assert_close(
            compensations[valid], _compensations[valid], rtol=1e-4, atol=1e-3
        )

    # backward
    v_means2d = torch.randn_like(means2d) * valid[..., None]
    v_depths = torch.randn_like(depths) * valid
    v_conics = torch.randn_like(conics) * valid[..., None]
    if calc_compensations:
        v_compensations = torch.randn_like(compensations) * valid
    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d).sum()
        + (depths * v_depths).sum()
        + (conics * v_conics).sum()
        + ((compensations * v_compensations).sum() if calc_compensations else 0),
        (viewmats, quats, scales, means),
    )
    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_conics * v_conics).sum()
        + ((_compensations * v_compensations).sum() if calc_compensations else 0),
        (viewmats, quats, scales, means),
    )

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=2e-3, atol=2e-3)
    # Slightly relaxed tolerance for quats due to numerical differences between triu=True
    # (CUDA path with triangular storage) and triu=False (PyTorch reference with full 3x3).
    # Both are mathematically equivalent but have different FP operation order.
    torch.testing.assert_close(v_quats, _v_quats, rtol=3e-1, atol=3e-2)
    torch.testing.assert_close(v_scales, _v_scales, rtol=5e-1, atol=2e-1)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-2, atol=6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("sparse_grad", [False])
@pytest.mark.parametrize("calc_compensations", [False, True])
@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_fully_fused_projection_packed(
    test_data,
    fused: bool,
    sparse_grad: bool,
    calc_compensations: bool,
    camera_model: CameraModel,
    batch_dims: Tuple[int, ...],
):
    from gsplat.cuda._wrapper import fully_fused_projection, quat_scale_to_covar_preci

    torch.manual_seed(42)

    test_data = expand(test_data, batch_dims)
    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]

    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True

    # forward
    if fused:
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            indptr,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            packed=True,
            sparse_grad=sparse_grad,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
        _radii, _means2d, _depths, _conics, _compensations = fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            packed=False,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
    else:
        covars, _ = quat_scale_to_covar_preci(quats, scales, triu=True)  # [..., N, 6]
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            indptr,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = fully_fused_projection(
            means,
            covars,
            None,
            None,
            viewmats,
            Ks,
            width,
            height,
            packed=True,
            sparse_grad=sparse_grad,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
        _radii, _means2d, _depths, _conics, _compensations = fully_fused_projection(
            means,
            covars,
            None,
            None,
            viewmats,
            Ks,
            width,
            height,
            packed=False,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )

    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]

    # recover packed tensors to full matrices for testing
    __radii = torch.sparse_coo_tensor(
        torch.stack([batch_ids, camera_ids, gaussian_ids]), radii, (B, C, N, 2)
    ).to_dense()
    __radii = __radii.reshape(batch_dims + (C, N, 2))
    __means2d = torch.sparse_coo_tensor(
        torch.stack([batch_ids, camera_ids, gaussian_ids]), means2d, (B, C, N, 2)
    ).to_dense()
    __means2d = __means2d.reshape(batch_dims + (C, N, 2))
    __depths = torch.sparse_coo_tensor(
        torch.stack([batch_ids, camera_ids, gaussian_ids]), depths, (B, C, N)
    ).to_dense()
    __depths = __depths.reshape(batch_dims + (C, N))
    __conics = torch.sparse_coo_tensor(
        torch.stack([batch_ids, camera_ids, gaussian_ids]), conics, (B, C, N, 3)
    ).to_dense()
    __conics = __conics.reshape(batch_dims + (C, N, 3))
    if calc_compensations:
        __compensations = torch.sparse_coo_tensor(
            torch.stack([batch_ids, camera_ids, gaussian_ids]),
            compensations,
            (B, C, N),
        ).to_dense()
        __compensations = __compensations.reshape(batch_dims + (C, N))
    sel = (__radii > 0).all(dim=-1) & (_radii > 0).all(dim=-1)
    torch.testing.assert_close(__radii[sel], _radii[sel], rtol=0, atol=1)
    torch.testing.assert_close(__means2d[sel], _means2d[sel], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(__depths[sel], _depths[sel], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(__conics[sel], _conics[sel], rtol=1e-4, atol=1e-4)
    if calc_compensations:
        torch.testing.assert_close(
            __compensations[sel], _compensations[sel], rtol=1e-4, atol=1e-3
        )

    # backward
    v_means2d = torch.randn_like(_means2d) * sel[..., None]
    v_depths = torch.randn_like(_depths) * sel
    v_conics = torch.randn_like(_conics) * sel[..., None]
    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_conics * v_conics).sum(),
        (viewmats, quats, scales, means),
        retain_graph=True,
    )
    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d[(__radii > 0).all(dim=-1)]).sum()
        + (depths * v_depths[(__radii > 0).all(dim=-1)]).sum()
        + (conics * v_conics[(__radii > 0).all(dim=-1)]).sum(),
        (viewmats, quats, scales, means),
        retain_graph=True,
    )
    if sparse_grad:
        v_quats = v_quats.to_dense()
        v_scales = v_scales.to_dense()
        v_means = v_means.to_dense()

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_quats, _v_quats, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_scales, _v_scales, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for UT projection"
)
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
@pytest.mark.parametrize(
    "require_all_valid", [True, False], ids=["allvalid", "somevalid"]
)
@pytest.mark.parametrize(
    "rolling_shutter",
    [RollingShutterType.GLOBAL, RollingShutterType.ROLLING_TOP_TO_BOTTOM],
)
@pytest.mark.parametrize("global_z_order", [True, False])
def test_fully_fused_projection_ut(
    test_data,
    batch_dims: Tuple[int, ...],
    require_all_valid: bool,
    rolling_shutter: RollingShutterType,
    global_z_order: bool,
):
    """Unified test for UT projection with CUDA vs PyTorch reference.

    Args:
        test_data: Test data fixture (test_garden.npz)
        batch_dims: Batch dimensions to test
        require_all_valid: UT parameter for sigma point validity
        rolling_shutter: Rolling shutter mode (GLOBAL, ROLLING_*, etc.)
    """
    from gsplat.cuda._torch_impl_ut import _fully_fused_projection_with_ut
    from gsplat.cuda._wrapper import fully_fused_projection_with_ut

    # Expand test data to batch dimensions
    test_data = expand(test_data, batch_dims)

    means = test_data["means"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    opacities = test_data["opacities"]
    viewmats = test_data["viewmats"]
    Ks = test_data["Ks"]
    width = test_data["width"]
    height = test_data["height"]

    # Create UT parameters
    ut_params = UnscentedTransformParameters(
        require_all_sigma_points_valid=require_all_valid
    )

    # Setup rolling shutter (end viewmats) if not GLOBAL
    if rolling_shutter != RollingShutterType.GLOBAL:
        viewmats_rs = viewmats.clone()
        # Add small translation to simulate camera motion (5% of scene scale)
        viewmats_rs[..., :3, 3] += torch.randn_like(viewmats[..., :3, 3]) * 0.05
    else:
        viewmats_rs = None

    means.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    opacities.requires_grad = True

    # ========================================================================
    # FORWARD PASS: CUDA vs PyTorch reference
    # ========================================================================

    parameters = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
        "camera_model": "pinhole",
        "eps2d": 0.3,
        "near_plane": 0.01,
        "far_plane": 1e10,
        "radius_clip": 0.0,
        "calc_compensations": True,
        "ut_params": ut_params,
        "rolling_shutter": rolling_shutter,
        "viewmats_rs": viewmats_rs,
        "global_z_order": global_z_order,
    }

    # Run CUDA implementation
    (
        radii_cuda,
        means2d_cuda,
        depths_cuda,
        conics_cuda,
        comps_cuda,
    ) = fully_fused_projection_with_ut(**parameters)

    # Run PyTorch reference implementation
    (
        radii_torch,
        means2d_torch,
        depths_torch,
        conics_torch,
        comps_torch,
    ) = _fully_fused_projection_with_ut(**parameters)

    # Compare outputs - use same selection pattern as test_basic.py
    # Only compare Gaussians that BOTH implementations marked as valid
    cuda_sel = (radii_cuda > 0).all(dim=-1)
    torch_sel = (radii_torch > 0).all(dim=-1)

    # Check that the number of mismatches is small (< 0.1% of total Gaussians)
    # Numerical differences can cause edge-case Gaussians to be culled differently
    assert_mismatch_ratio(cuda_sel, torch_sel, max=1e-3)

    sel = cuda_sel & torch_sel

    assert sel.any(), f"No valid Gaussians found"

    # ========================================================================
    # ASSERTIONS: Compare outputs with appropriate tolerances
    # ========================================================================

    # Radii: Integer values, allow small differences due to ceil() rounding
    # With require_all_valid=True, early-exit can cause larger FP32 differences
    # Rolling shutter uses iterative refinement (10 iterations) which accumulates
    # numerical differences through the opacity-aware radius computation pipeline
    if rolling_shutter == RollingShutterType.GLOBAL:
        radii_atol = 2.0  # Only ceil() differences for global shutter
    else:
        radii_atol = (
            10.0  # Rolling shutter: iterative refinement amplifies FP32 differences
        )

    torch.testing.assert_close(
        radii_cuda[sel].float(), radii_torch[sel].float(), rtol=0, atol=radii_atol
    )

    # means2d: Sub-pixel precision expected
    # Relaxed tolerances appropriate for UT's multi-step numerical pipeline
    # UT involves sigma point generation, projection, and weighted averaging
    # which accumulates small FP32 differences between CUDA and PyTorch
    torch.testing.assert_close(
        means2d_cuda[sel],
        means2d_torch[sel],
        rtol=0.5,  # 50% relative tolerance (handles high rel diff at near-zero values)
        atol=0.05,  # 0.05 pixel absolute tolerance (great sub-pixel accuracy)
    )

    # depths: High precision expected
    # Depths are computed from camera transformation, less accumulation than means2d
    torch.testing.assert_close(
        depths_cuda[sel],
        depths_torch[sel],
        rtol=1e-6,  # 0.0001% relative tolerance (excellent precision)
        atol=2e-6,  # 2e-6 absolute tolerance (2x safety margin)
    )

    # conics: Moderate precision
    # Conics involve covariance inverse which can amplify small numerical differences
    # Near-zero conic values can have large relative differences but small absolute differences
    # Rolling shutter: iterative refinement affects 2D covariance which propagates to conics
    if rolling_shutter == RollingShutterType.GLOBAL:
        conics_rtol, conics_atol = 1e-2, 1e-2
    else:
        conics_rtol, conics_atol = (
            10.0,
            1.0,
        )  # Rolling shutter amplifies differences in conic computation

    torch.testing.assert_close(
        conics_cuda[sel], conics_torch[sel], rtol=conics_rtol, atol=conics_atol
    )

    # compensations: Moderate precision
    # Compensations involve sqrt(det_orig/det_blur) which can be sensitive
    # Small differences in determinants can cause larger differences in sqrt ratio
    # Rolling shutter: changes in 2D covariance determinant cascade through compensation
    if rolling_shutter == RollingShutterType.GLOBAL:
        comps_rtol, comps_atol = 0.1, 0.01  # Global: max_abs=0.0034, max_rel=4.3%
    else:
        comps_rtol, comps_atol = (
            0.25,
            0.15,
        )  # Rolling shutter: max_abs=0.110, max_rel=18.4%

    torch.testing.assert_close(
        comps_cuda[sel], comps_torch[sel], rtol=comps_rtol, atol=comps_atol
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT/Lidar support isn't built in")
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_isect(test_data, batch_dims: Tuple[int, ...]):
    from gsplat.cuda._torch_impl import _isect_offset_encode, _isect_tiles
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    torch.manual_seed(42)

    B = math.prod(batch_dims)
    C, N = 3, 1000
    I = B * C
    width, height = 40, 60

    test_data = {
        "means2d": torch.randn(C, N, 2, device=device) * width,
        "radii": torch.randint(0, width, (C, N, 2), device=device, dtype=torch.int32),
        "depths": torch.rand(C, N, device=device),
    }
    test_data = expand(test_data, batch_dims)
    means2d = test_data["means2d"]
    radii = test_data["radii"]
    depths = test_data["depths"]

    tile_size = 16
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)

    _tiles_per_gauss, _isect_ids, _gauss_ids = _isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    _isect_offsets = _isect_offset_encode(_isect_ids, I, tile_width, tile_height)

    torch.testing.assert_close(tiles_per_gauss, _tiles_per_gauss)
    torch.testing.assert_close(isect_ids, _isect_ids)
    torch.testing.assert_close(flatten_ids, _gauss_ids)
    torch.testing.assert_close(isect_offsets, _isect_offsets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT/Lidar support isn't built in")
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
@pytest.mark.parametrize("lidar_model", ["pandar128", "at128"])
def test_isect_lidar(lidar_model, batch_dims: Tuple[int, ...]):
    from gsplat.cuda._torch_impl_lidar import (
        _isect_tiles_lidar,
        ANGLE_TO_PIXEL_SCALING_FACTOR,
    )
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles_lidar

    torch.manual_seed(42)

    params = parse_lidar_camera(lidar_model, batch_dims, 0, 0, device=device)
    lidar = gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(**params)

    C, N = 3, 1000  # cameras and gaussians

    test_data = {
        "means2d": torch.randn(C, N, 2, device=device)
        * torch.tensor([math.pi, 2 * math.pi], device=device),
        # TODO: assuming .x=elevation and .y=azimuth. This must be transposed.
        "radii": torch.randn(C, N, 2, device=device).abs().clamp(max=1)
        * torch.tensor([lidar.fov_vert_rad.span / 2, math.pi], device=device),
        "depths": torch.rand(C, N, device=device),
    }
    test_data = expand(test_data, batch_dims)
    means2d = test_data["means2d"] * ANGLE_TO_PIXEL_SCALING_FACTOR
    radii = torch.ceil(test_data["radii"] * ANGLE_TO_PIXEL_SCALING_FACTOR).to(
        torch.int32
    )
    depths = test_data["depths"]

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_lidar(
        lidar, means2d, radii, depths, sort=True
    )

    _tiles_per_gauss, _isect_ids, _flatten_ids = _isect_tiles_lidar(
        lidar, means2d, radii, depths, sort=True
    )

    torch.testing.assert_close(tiles_per_gauss, _tiles_per_gauss)
    torch.testing.assert_close(isect_ids, _isect_ids)
    torch.testing.assert_close(flatten_ids, _flatten_ids)


@pytest.fixture
def lidar_param(hfov_span_deg, ray_location, n_dense_tiles_azimuth):
    from gsplat.cuda._lidar import LidarTiling

    # The actual vfov span value is not relevant, as long as it's within elevation range.
    vfov_span_deg = 90
    spinning_direction = gsplat.SpinningDirection.COUNTER_CLOCKWISE

    # Scale FOV to be multiple of a "pixel"
    hfov_span_rad = (
        math.ceil((hfov_span_deg / 180 * math.pi) * ANGLE_TO_PIXEL_SCALING_FACTOR)
        / ANGLE_TO_PIXEL_SCALING_FACTOR
    )
    vfov_span_rad = (
        math.ceil((vfov_span_deg / 180 * math.pi) * ANGLE_TO_PIXEL_SCALING_FACTOR)
        / ANGLE_TO_PIXEL_SCALING_FACTOR
    )

    hfov_span_pix = hfov_span_rad * ANGLE_TO_PIXEL_SCALING_FACTOR

    assert (hfov_span_deg < 360) == (hfov_span_rad < 2 * math.pi)

    # If the dense tile's size to be a multiple of a pixel...
    if n_dense_tiles_azimuth < 0:
        n_dense_tiles_azimuth = math.ceil(hfov_span_pix / (-n_dense_tiles_azimuth))

    # clockwise
    row_elevations_rad = torch.tensor(
        [vfov_span_rad / 2, 0.0, -vfov_span_rad / 2], dtype=torch.float32, device=device
    )

    # Keep base columns strictly within <2*pi span to satisfy lidar parameter assertions.
    # Row offsets then expand the effective FoV to the requested hfov_span_rad.
    offset_amp = min(0.2, hfov_span_rad * 0.2)
    base_hfov_span_rad = max(1e-3, hfov_span_rad - 2.0 * offset_amp)

    if spinning_direction == gsplat.SpinningDirection.CLOCKWISE:
        column_azimuths_rad = torch.linspace(
            # Angles go in decreasing order.
            base_hfov_span_rad / 2,
            -base_hfov_span_rad / 2,
            5,
            dtype=torch.float32,
            device=device,
        )
    else:
        column_azimuths_rad = torch.linspace(
            # Angles go in increasing order.
            -base_hfov_span_rad / 2,
            base_hfov_span_rad / 2,
            5,
            dtype=torch.float32,
            device=device,
        )
    row_azimuth_offsets_rad = torch.tensor(
        [offset_amp, 0.0, -offset_amp], dtype=torch.float32, device=device
    )

    n_bins_elevation = 11
    n_bins_azimuth = 21
    # Use a non-identity CDF (multiple dense bins per coarse bin) so that
    # the elevation CDF half-open range logic is exercised in every test.
    cdf_resolution_elevation = 33
    cdf_resolution_azimuth = n_dense_tiles_azimuth

    # Build sparse mask from requested ray azimuth position(s).

    # Define the ray location angle
    if ray_location == "none":
        rays_rel_az_pix = []
    elif ray_location == "middle":
        rays_rel_az_pix = [hfov_span_pix * 0.5]
    elif ray_location == "min":
        rays_rel_az_pix = [0.0]
    elif ray_location == "max":
        rays_rel_az_pix = [hfov_span_pix - 1.0]
    elif ray_location == "both":
        rays_rel_az_pix = [0.0, hfov_span_pix - 1.0]
    else:
        assert False, f"Invalid ray location: {ray_location=}"

    dense_mask = torch.zeros(
        (cdf_resolution_azimuth, cdf_resolution_elevation),
        dtype=torch.int32,
        device=device,
    )
    if len(rays_rel_az_pix) > 0:
        all_az_rad = column_azimuths_rad[None, :] + row_azimuth_offsets_rad[:, None]
        fov_min_rad = all_az_rad.min().item()
        fov_max_rad = all_az_rad.max().item()

        fov_span_az_pix = (fov_max_rad - fov_min_rad) * ANGLE_TO_PIXEL_SCALING_FACTOR

        for ray_rel_az_pix in rays_rel_az_pix:
            az_norm = ray_rel_az_pix / fov_span_az_pix
            az_idx = int(math.floor(az_norm * cdf_resolution_azimuth))
            az_idx = max(0, min(az_idx, cdf_resolution_azimuth - 1))

            # Always use the middle elevation, as only azimuth requires
            # special testing (periodic, non-periodic, etc).
            middle_el_idx = dense_mask.shape[1] // 2

            dense_mask[az_idx, middle_el_idx] = 1

    # Create the cdf dense ray mask given the dense mask with number of rays in a dense cell.
    cdf = torch.zeros(
        (dense_mask.shape[0] + 1, dense_mask.shape[1] + 1),
        dtype=torch.int32,
        device=dense_mask.device,
    )
    cdf[1:, 1:] = dense_mask.to(torch.int32)
    cdf_dense_ray_mask = cdf.cumsum(dim=0).cumsum(dim=1).to(torch.int32)

    # Build elevation CDF: maps dense elevation index -> coarse elevation bin.
    # Identity when cdf_resolution_elevation == n_bins_elevation;
    # multiple dense bins per coarse bin otherwise.
    cdf_elevation = torch.tensor(
        [
            i * n_bins_elevation // cdf_resolution_elevation
            for i in range(cdf_resolution_elevation + 1)
        ],
        dtype=torch.int32,
        device=device,
    )
    # The CDF must have plateaus (consecutive equal values) so that the
    # elevation half-open range logic is actually exercised.
    assert torch.any(cdf_elevation[1:] == cdf_elevation[:-1])

    # These aren't used, but need to have the correct shape.
    tiles_pack_info = torch.zeros(
        (n_bins_azimuth * n_bins_elevation, 2), dtype=torch.int32, device=device
    )
    tiles_to_elements_map = torch.zeros((1, 2), dtype=torch.int32, device=device)

    # Finally create the LidarTiling object
    tiling = LidarTiling(
        n_bins_azimuth=n_bins_azimuth,
        n_bins_elevation=n_bins_elevation,
        cdf_elevation=cdf_elevation,
        cdf_dense_ray_mask=cdf_dense_ray_mask,
        tiles_pack_info=tiles_pack_info,
        tiles_to_elements_map=tiles_to_elements_map,
    )

    # And the lidar whole parameters
    lidar = gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
        row_elevations_rad=row_elevations_rad,
        column_azimuths_rad=column_azimuths_rad,
        row_azimuth_offsets_rad=row_azimuth_offsets_rad,
        spinning_direction=spinning_direction,
        spinning_frequency_hz=10.0,
        angles_to_columns_map=torch.zeros((3, 5), dtype=torch.int32, device=device),
        tiling=tiling,
    )

    assert lidar.fov_horiz_rad.span * 180.0 / math.pi >= hfov_span_deg
    return lidar


@pytest.fixture
def gaussian_param(lidar_param, gauss_start_pos, gauss_end_pos):
    lidar = lidar_param

    hfov_start_pix = lidar.fov_horiz_rad.start * ANGLE_TO_PIXEL_SCALING_FACTOR
    hfov_span_pix = lidar.fov_horiz_rad.span * ANGLE_TO_PIXEL_SCALING_FACTOR
    hfov_end_pix = hfov_start_pix + hfov_span_pix

    full_circle_pix = 2 * math.pi * ANGLE_TO_PIXEL_SCALING_FACTOR

    def get_hfov_angle_az_pix(boundary_name):
        if "far_min_outside" in boundary_name:
            return hfov_start_pix - hfov_span_pix
        elif "far_max_outside" in boundary_name:
            return hfov_end_pix + hfov_span_pix
        elif "min_outside" in boundary_name:
            return hfov_start_pix - 1
        elif "min_inside" in boundary_name:
            return hfov_start_pix
        elif "low_inside" in boundary_name:
            return hfov_start_pix + hfov_span_pix * 0.25
        elif "high_inside" in boundary_name:
            return hfov_start_pix + hfov_span_pix * 0.75
        elif "max_inside" in boundary_name:
            return hfov_end_pix - 1
        elif "max_outside" in boundary_name:
            return hfov_end_pix
        else:
            assert False, f"Invalid boundary: {boundary_name}"

    gaussian_start_az_pix = get_hfov_angle_az_pix(gauss_start_pos)
    gaussian_end_az_pix = get_hfov_angle_az_pix(gauss_end_pos)

    periodic_azimuth = hfov_span_pix >= full_circle_pix
    behind_sensor = not periodic_azimuth and gaussian_start_az_pix > gaussian_end_az_pix

    # Calculate the raw radius (in floating point)
    radius_az_pix = abs(gaussian_end_az_pix - gaussian_start_az_pix) / 2
    if gaussian_start_az_pix > gaussian_end_az_pix:
        radius_az_pix = full_circle_pix / 2 - radius_az_pix

    assert ("exact" not in gauss_start_pos) or (
        "exact" not in gauss_end_pos
    ), "Both boundaries can't be exact"

    # If either gaussian extremity is exactly positioned (i.e. pinned)
    if "exact" in gauss_start_pos or "exact" in gauss_end_pos:
        if "exact" in gauss_start_pos:
            exact_boundary = gauss_start_pos
            non_exact_boundary = gauss_end_pos
        else:
            exact_boundary = gauss_end_pos
            non_exact_boundary = gauss_start_pos

        # We round to the direction that won't change the classification of the non-exact boundary
        round_up = non_exact_boundary in ("min_inside", "max_outside")
        # When gaussian is behind the sensor, increasing the radius make its edges go inwards the hfov,
        # so we need to round in the opposite direction.
        if behind_sensor and exact_boundary == gauss_end_pos:
            round_up = not round_up

        if round_up:
            radius_az_pix = math.ceil(radius_az_pix)
        else:
            radius_az_pix = math.floor(radius_az_pix)

        # Make sure the radius ends up not being 0 due to rounding.
        if radius_az_pix == 0:
            radius_az_pix += 1

        def calc_mean_az_pix(exact_edge: float, radius: float) -> float:
            gauss_mean = exact_edge + radius
            # Convert from float64 to float32 (that's what we use in torch)
            gauss_mean_fp32 = struct.unpack("!f", struct.pack("!f", float(gauss_mean)))[
                0
            ]
            # Estimate where the edge would land with the current gaussian mean
            edge_estim = gauss_mean_fp32 - radius
            # Calc the new gaussian mean so that the mean+radius lands exactly on the exact edge
            return gauss_mean + (exact_edge - edge_estim)

        if exact_boundary == gauss_start_pos:
            mean_az_pix = calc_mean_az_pix(gaussian_start_az_pix, radius_az_pix)
        else:
            assert exact_boundary == gauss_end_pos
            mean_az_pix = calc_mean_az_pix(gaussian_end_az_pix, -radius_az_pix)
    else:
        radius_az_pix = round(radius_az_pix)
        mean_az_pix = gaussian_start_az_pix + radius_az_pix

    assert not behind_sensor or (
        mean_az_pix < hfov_start_pix or mean_az_pix >= hfov_end_pix
    ), "If the gaussian is behind the sensor, its mean must be outside hfov"

    return SimpleNamespace(
        means2d=torch.tensor(
            [[[0.0, mean_az_pix]]],
            dtype=torch.float32,
            device=device,
        ),
        radii=torch.tensor(
            [[[1, radius_az_pix]]],
            dtype=torch.int32,
            device=device,
        ),
        depths=torch.tensor([[1.0]], dtype=torch.float32, device=device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize(
    # Verifies that isect_tiles_lidar (CUDA) and _isect_tiles_lidar (Python ref)
    # agree on the number of tiles intersected by a single Gaussian at the
    # boundary of the horizontal FOV.
    #
    # Parameters:
    #   hfov_span_deg:
    #       - horizontal FOV span (180 = non-periodic, 360 = periodic azimuth).
    #   gauss_start_pos / gauss_end_pos:
    #       - where each edge of the Gaussian sits relative to the hfov, in the spinning direction.
    #         When start > end, the Gaussian wraps through the back of the sensor.
    #         Positions are:
    #           min_outside  = just outside the low boundary
    #           min_inside   = on the low boundary
    #           low_inside   = well inside, lower quarter
    #           high_inside  = well inside, upper quarter
    #           max_inside   = on the high boundary
    #           max_outside  = just outside the high boundary
    #       Appending "_exact" pins that edge precisely on the boundary pixel (in fp32 precision).
    #       Inexact positions might be nudged so that the computed gaussian radius remain an integer.
    #       Nudging is done so that the classification of the inexact position doesn't change (when possible).
    #   ray_location: where the single ray is placed in the CDF mask
    #       ("none" / "min" / "max" / "middle").
    #   n_dense_tiles_azimuth: number of dense azimuth tiles (-1 = 1 tile per pixel).
    #   expected_tiles: how many tiles the Gaussian should intersect.
    "hfov_span_deg,gauss_start_pos,gauss_end_pos,ray_location,n_dense_tiles_azimuth,expected_tiles",
    [
        # Special case for full cover returning all tiles.
        # beg<=0 and end>=span triggers full_cover, which bypasses the ray
        # check and returns all n_bins_azimuth tiles.
        (
            120,
            "min_inside_exact",
            "max_outside",
            "none",
            21,
            21,
        ),  # beg=0, end=span -> full_cover -> 21
        (
            120,
            "min_outside",
            "max_outside_exact",
            "none",
            21,
            21,
        ),  # beg<0, end=span -> full_cover -> 21
        (
            360,
            "min_inside_exact",
            "max_outside",
            "none",
            21,
            21,
        ),  # beg=0, end=span -> full_cover -> 21
        (
            360,
            "min_outside",
            "max_outside_exact",
            "none",
            21,
            21,
        ),  # beg<0, end=span -> full_cover -> 21
        # Full cover with rays present: full_cover bypasses has_any_rays_in_tile,
        # so the result must be 21 regardless of ray placement.
        # Use min_inside_exact+max_outside: exact pinning at min ensures
        # radius rounds up, making end > span and guaranteeing full_cover.
        (
            120,
            "min_inside_exact",
            "max_outside",
            "min",
            21,
            21,
        ),  # full_cover + ray@min -> still 21
        (
            120,
            "min_inside_exact",
            "max_outside",
            "both",
            21,
            21,
        ),  # full_cover + both rays -> still 21
        (
            360,
            "min_inside_exact",
            "max_outside",
            "max",
            21,
            21,
        ),  # full_cover + ray@max -> still 21
        # Near-full cover: the exact pinning at max_outside rounds the radius down,
        # so the start edge lands at ~hfov_start-0.02 instead of -1.
        # The gaussian nearly spans the full FOV but just barely misses
        # full_cover due to fp32 precision. The A/B split must not
        # double-count the boundary tile.
        (
            360,
            "min_outside",
            "max_outside_exact",
            "max",
            21,
            21,
        ),  # A=[0,fc-~0.02), B=[fc-~0.02,fc); tile 20 only counted once -> 21
        # Almost-full cover: end=span-1 fails the full_cover check
        # (end < span), and with no rays the gaussian is culled.
        # We use -1 to denote 1 tile per pixel.
        #
        # NOTE: With n_dense=-1 (1 dense cell per pixel), the float32
        # computation ceil((span-1)/span * span) can overshoot to span
        # for certain FOV values, falsely triggering the full_cover
        # optimization in has_any_rays_in_tile and returning 21 instead
        # of 0. This is a known float32 precision issue.
        # This would only affect performance, nothing to worry about.
        (
            120,
            "min_inside_exact",
            "max_inside",
            "none",
            -1,
            0,
        ),  # end=span-1 < span -> not full_cover, no rays -> 0
        (
            120,
            "min_outside",
            "max_inside_exact",
            "none",
            -1,
            0,
        ),  # end=span-1 < span -> not full_cover, no rays -> 0
        (
            360,
            "min_inside_exact",
            "max_inside",
            "none",
            -1,
            0,
        ),  # end=span-1 < span -> not full_cover, no rays -> 0
        (
            360,
            "min_outside",
            "max_inside_exact",
            "none",
            -1,
            0,
        ),  # end=span-1 < span -> not full_cover, no rays -> 0
        # Normal inside gaussians.
        (
            120,
            "low_inside",
            "high_inside",
            "middle",
            21,
            11,
        ),  # A covers [0.25,0.75)*span -> tiles [5,16); ray@cell10 hit -> 11
        (
            120,
            "high_inside",
            "low_inside",
            "middle",
            21,
            0,
        ),  # behind sensor: tips at min/max edges, ray@middle in gap -> 0
        # 360deg normal inside: same geometry but periodic.
        (
            360,
            "low_inside",
            "high_inside",
            "middle",
            21,
            11,
        ),  # same as 120deg, normal middle coverage -> 11
        # 360deg wrap-around: high->low wraps through the opposite side
        # (not behind_sensor since periodic). Produces A=[0.75*span,fc)
        # and B=[0,0.25*span) via overflow, covering both edges of the FOV.
        (
            360,
            "high_inside",
            "low_inside",
            "middle",
            21,
            0,
        ),  # ray@middle in gap between A and B -> 0
        (
            360,
            "high_inside",
            "low_inside",
            "both",
            21,
            12,
        ),  # A=6 tiles (hit@max), B=6 tiles (hit@min) -> 12
        # Behind-sensor: gaussian wraps through the back of the sensor.
        # The overflow creates A (max tip) and B (min tip) ranges.
        #
        # start_exact@max_inside, end@min_inside:
        #   A=[span-1,span)=tile 20, B=[0,~0) via overflow=tile 0
        (
            120,
            "max_inside_exact",
            "min_inside",
            "min",
            21,
            1,
        ),  # ray@min in B(tile 0) -> 1
        (
            120,
            "max_inside_exact",
            "min_inside",
            "max",
            21,
            1,
        ),  # ray@max in A(tile 20) -> 1
        (
            120,
            "max_inside_exact",
            "min_inside",
            "both",
            21,
            2,
        ),  # ray@min in B + ray@max in A -> 2
        # start@max_inside, end_exact@min_inside:
        #   end exact@0 -> B=[0,0)=empty; A=[span-1,span)=tile 20
        (
            120,
            "max_inside",
            "min_inside_exact",
            "min",
            21,
            0,
        ),  # B empty; ray@min not in A -> 0
        (
            120,
            "max_inside",
            "min_inside_exact",
            "max",
            21,
            1,
        ),  # B empty; ray@max in A(tile 20) -> 1
        (
            120,
            "max_inside",
            "min_inside_exact",
            "both",
            21,
            1,
        ),  # B empty; only ray@max in A -> 1
        # start_exact@max_outside: start at span boundary -> A=[span,span)=empty.
        # Only B (overflow tip at min) can contribute.
        (
            120,
            "max_outside_exact",
            "min_inside",
            "min",
            21,
            1,
        ),  # A empty, B=tile 0; ray@min in B -> 1
        (
            120,
            "max_outside_exact",
            "min_inside",
            "max",
            21,
            0,
        ),  # A empty, B=tile 0; ray@max not in B -> 0
        (
            120,
            "max_outside_exact",
            "min_inside",
            "both",
            21,
            1,
        ),  # A empty, B=tile 0; only ray@min in B -> 1
        # start@max_outside, end_exact@min_inside:
        #   beg=span+1 -> A clamped empty; end exact@0 -> B=[0,0)=empty
        (120, "max_outside", "min_inside_exact", "min", 21, 0),  # A/B both empty -> 0
        (120, "max_outside", "min_inside_exact", "max", 21, 0),  # A/B both empty -> 0
        (120, "max_outside", "min_inside_exact", "both", 21, 0),  # A/B both empty -> 0
        # start_exact@max_inside, end@min_outside:
        #   end outside -> beg=span-1, end=fc-1, no overflow.
        #   A clamped to [span-1,span)=tile 20, B empty.
        (
            120,
            "max_inside_exact",
            "min_outside",
            "min",
            21,
            0,
        ),  # ray@min not in A(tile 20) -> 0
        (
            120,
            "max_inside_exact",
            "min_outside",
            "max",
            21,
            1,
        ),  # ray@max in A(tile 20) -> 1
        (
            120,
            "max_inside_exact",
            "min_outside",
            "both",
            21,
            1,
        ),  # B empty; only ray@max in A -> 1
        (
            120,
            "max_inside",
            "min_outside_exact",
            "min",
            21,
            0,
        ),  # A=tile 20; ray@min not in A -> 0
        (
            120,
            "max_inside",
            "min_outside_exact",
            "max",
            21,
            1,
        ),  # A=tile 20; ray@max in A -> 1
        (
            120,
            "max_inside",
            "min_outside_exact",
            "both",
            21,
            1,
        ),  # A=tile 20; only ray@max in A -> 1
        # Both edges outside the FOV -> A/B clamp to empty.
        (120, "max_outside_exact", "min_outside", "min", 21, 0),  # both outside -> 0
        (120, "max_outside_exact", "min_outside", "max", 21, 0),  # both outside -> 0
        (120, "max_outside_exact", "min_outside", "both", 21, 0),  # both outside -> 0
        (120, "max_outside", "min_outside_exact", "min", 21, 0),  # both outside -> 0
        (120, "max_outside", "min_outside_exact", "max", 21, 0),  # both outside -> 0
        (120, "max_outside", "min_outside_exact", "both", 21, 0),  # both outside -> 0
        # 1 non-periodic azimuth
        # 1.1. Gaussian edge on min hfov boundary
        # 1.1.1 ray on min boundary
        (
            120,
            "min_outside_exact",
            "min_inside",
            "min",
            21,
            1,
        ),  # underflow: A=[0,1)=cell 0, B clamped empty; ray@min in A -> 1
        (
            120,
            "min_outside",
            "min_outside_exact",
            "min",
            21,
            0,
        ),  # entirely outside FOV (near fc), clamped empty -> 0
        (
            120,
            "min_inside_exact",
            "min_inside",
            "min",
            21,
            1,
        ),  # A=[0,2)=cell 0; ray@min in A -> 1
        (
            120,
            "min_outside",
            "min_inside_exact",
            "min",
            21,
            0,
        ),  # mean outside, end exact@0 -> clamped empty -> 0
        # 1.1.2 ray on max boundary
        (
            120,
            "min_outside_exact",
            "min_inside",
            "max",
            21,
            0,
        ),  # A=[0,1)=cell 0; ray@max(cell 20) not in A -> 0
        (
            120,
            "min_outside",
            "min_outside_exact",
            "max",
            21,
            0,
        ),  # entirely outside -> 0
        (
            120,
            "min_inside_exact",
            "min_inside",
            "max",
            21,
            0,
        ),  # A=[0,2)=cell 0; ray@max not in A -> 0
        (120, "min_outside", "min_inside_exact", "max", 21, 0),  # clamped empty -> 0
        # 1.1.3 ray on both boundaries (non-periodic B is empty, only min side matters)
        (
            120,
            "min_outside_exact",
            "min_inside",
            "both",
            21,
            1,
        ),  # A=cell 0; only ray@min hit -> 1
        (
            120,
            "min_outside",
            "min_outside_exact",
            "both",
            21,
            0,
        ),  # entirely outside -> 0
        (
            120,
            "min_inside_exact",
            "min_inside",
            "both",
            21,
            1,
        ),  # A=cell 0; only ray@min hit -> 1
        (120, "min_outside", "min_inside_exact", "both", 21, 0),  # clamped empty -> 0
        # 1.2. Gaussian edge on max hfov boundary
        # 1.2.1 ray on min boundary
        (
            120,
            "max_outside_exact",
            "max_outside",
            "min",
            21,
            0,
        ),  # both at span, clamped to [span,span+2) -> empty -> 0
        (
            120,
            "max_inside",
            "max_outside_exact",
            "min",
            21,
            0,
        ),  # A=[span-2,span)=cell 20; ray@min not in A -> 0
        (
            120,
            "max_inside_exact",
            "max_outside",
            "min",
            21,
            0,
        ),  # A=[span-1,span)=cell 20; ray@min not in A -> 0
        (
            120,
            "max_inside",
            "max_inside_exact",
            "min",
            21,
            0,
        ),  # A near max=cell 20; ray@min not in A -> 0
        # 1.2.2 ray on max boundary
        (
            120,
            "max_outside_exact",
            "max_outside",
            "max",
            21,
            0,
        ),  # entirely outside (clamped empty) -> 0
        (
            120,
            "max_inside",
            "max_outside_exact",
            "max",
            21,
            1,
        ),  # A=[span-2,span)=cell 20; ray@max in A -> 1
        (
            120,
            "max_inside_exact",
            "max_outside",
            "max",
            21,
            1,
        ),  # A=[span-1,span)=cell 20; ray@max in A -> 1
        (
            120,
            "max_inside",
            "max_inside_exact",
            "max",
            21,
            1,
        ),  # A near max=cell 20; ray@max in A -> 1
        # 1.2.3 ray on both boundaries (non-periodic B is empty, only max side matters)
        (
            120,
            "max_outside_exact",
            "max_outside",
            "both",
            21,
            0,
        ),  # entirely outside -> 0
        (
            120,
            "max_inside",
            "max_outside_exact",
            "both",
            21,
            1,
        ),  # A=cell 20; only ray@max hit -> 1
        (
            120,
            "max_inside_exact",
            "max_outside",
            "both",
            21,
            1,
        ),  # A=cell 20; only ray@max hit -> 1
        (
            120,
            "max_inside",
            "max_inside_exact",
            "both",
            21,
            1,
        ),  # A=cell 20; only ray@max hit -> 1
        # 2 periodic azimuth
        # 2.1. Gaussian edge on min hfov boundary
        # 2.1.1 ray on min hfov boundary
        (
            360,
            "min_outside_exact",
            "min_inside",
            "min",
            21,
            1,
        ),  # underflow: A=[0,1)=cell 0, B=[fc-1,fc)=cell 20; ray@min in A -> 1
        (
            360,
            "min_outside",
            "min_outside_exact",
            "min",
            21,
            0,
        ),  # A=[fc-2,fc)=cell 20; ray@min(cell 0) not in A -> 0
        (
            360,
            "min_inside_exact",
            "min_inside",
            "min",
            21,
            1,
        ),  # A=[0,2)=cell 0; ray@min in A -> 1
        (
            360,
            "min_outside",
            "min_inside_exact",
            "min",
            21,
            0,
        ),  # A near fc=cell 20; ray@min not in A -> 0
        # 2.1.2 ray on max boundary
        (
            360,
            "min_outside_exact",
            "min_inside",
            "max",
            21,
            1,
        ),  # A=cell 0, B=cell 20; ray@max in B -> 1
        (
            360,
            "min_outside",
            "min_outside_exact",
            "max",
            21,
            1,
        ),  # A=[fc-2,fc)=cell 20; ray@max in A -> 1
        (
            360,
            "min_inside_exact",
            "min_inside",
            "max",
            21,
            0,
        ),  # A=[0,2)=cell 0; ray@max not in A -> 0
        (
            360,
            "min_outside",
            "min_inside_exact",
            "max",
            21,
            1,
        ),  # A near fc=cell 20; ray@max in A -> 1
        # 2.1.3 ray on both boundaries
        (
            360,
            "min_outside_exact",
            "min_inside",
            "both",
            21,
            2,
        ),  # A=cell 0 (hit@min), B=cell 20 (hit@max) -> 2
        (
            360,
            "min_outside",
            "min_outside_exact",
            "both",
            21,
            1,
        ),  # A=cell 20; ray@max hit, ray@min miss -> 1
        (
            360,
            "min_inside_exact",
            "min_inside",
            "both",
            21,
            1,
        ),  # A=cell 0; ray@min hit, ray@max miss -> 1
        (
            360,
            "min_outside",
            "min_inside_exact",
            "both",
            21,
            1,
        ),  # A=cell 20; ray@max hit -> 1
        # 2.2. Gaussian edge on max hfov boundary
        (
            360,
            "max_outside_exact",
            "max_outside",
            "min",
            21,
            1,
        ),  # wraps to A=[0,2)=cell 0; ray@min in A -> 1
        (
            360,
            "max_outside_exact",
            "min_inside",
            "min",
            21,
            1,
        ),  # same angles for 360: wraps to cell 0; ray@min in A -> 1
        (
            360,
            "max_inside",
            "max_outside_exact",
            "min",
            21,
            0,
        ),  # A=[fc-2,fc)=cell 20; ray@min not in A -> 0
        (
            360,
            "max_inside",
            "min_inside_exact",
            "min",
            21,
            0,
        ),  # same geometry as above -> 0
        (
            360,
            "max_inside_exact",
            "max_outside",
            "min",
            21,
            1,
        ),  # underflow: A=[0,1), B=[fc-1,fc); ray@min in A -> 1
        (
            360,
            "max_inside_exact",
            "min_inside",
            "min",
            21,
            1,
        ),  # same geometry as above -> 1
        (
            360,
            "max_inside",
            "max_inside_exact",
            "min",
            21,
            0,
        ),  # A=[fc-3,fc-1)=cells 19-20; ray@min not in A -> 0
        (
            360,
            "max_inside",
            "min_outside_exact",
            "min",
            21,
            0,
        ),  # same geometry as above -> 0
        # 2.2.2 ray on max boundary
        (
            360,
            "max_outside_exact",
            "max_outside",
            "max",
            21,
            0,
        ),  # wraps to A=[0,2)=cell 0; ray@max not in A -> 0
        (
            360,
            "max_inside",
            "max_outside_exact",
            "max",
            21,
            1,
        ),  # A=[fc-2,fc)=cell 20; ray@max in A -> 1
        (
            360,
            "max_inside_exact",
            "max_outside",
            "max",
            21,
            1,
        ),  # A=cell 0, B=cell 20; ray@max in B -> 1
        (
            360,
            "max_inside",
            "max_inside_exact",
            "max",
            21,
            1,
        ),  # A=[fc-3,fc-1)=cells 19-20; ray@max(cell 20) in A -> 1
        # 2.2.3 ray on both boundaries
        (
            360,
            "max_inside_exact",
            "max_outside",
            "both",
            21,
            2,
        ),  # A=cell 0 (hit@min), B=cell 20 (hit@max) -> 2
        (
            360,
            "max_outside_exact",
            "min_outside",
            "both",
            21,
            0,
        ),  # extent wraps >full_circle -> negative radius -> culled -> 0
        (
            360,
            "max_outside",
            "min_outside_exact",
            "both",
            21,
            1,
        ),  # radius rounds to 1, A covers cell 20; ray@max hit -> 1
        # 3 Dense tile resolution different from tile resolution.
        # n_dense=7 (coarser than n_bins=21) verifies the independent
        # scaling of dense and tile grids doesn't cause off-by-one errors.
        (
            120,
            "min_outside_exact",
            "min_inside",
            "min",
            7,
            1,
        ),  # coarser dense: A dense=[0,1), ray@min hit -> 1
        (
            120,
            "max_inside",
            "max_outside_exact",
            "max",
            7,
            1,
        ),  # coarser dense: A dense=[6,7), ray@max hit -> 1
        (
            120,
            "low_inside",
            "high_inside",
            "middle",
            7,
            11,
        ),  # coarser dense doesn't change tile count -> 11
        (
            360,
            "min_outside_exact",
            "min_inside",
            "min",
            7,
            1,
        ),  # periodic with coarser dense -> 1
        # 4 Large-extent gaussians (extent > fov_span/2).
        # Regression test: the old code incorrectly filtered these because
        # the dense tile index range for both gaussian edges ended up on the
        # same side of the FOV, causing has_any_rays_in_tile to misclassify
        # the gaussian as "entirely outside."
        (
            120,
            "far_min_outside",
            "far_max_outside",
            "both",
            21,
            21,
        ),  # extent=1.5*span; full_cover -> 21
        (
            120,
            "far_min_outside",
            "far_max_outside",
            "none",
            21,
            21,
        ),  # full_cover triggers conservative shortcut -> 21
        (200, "far_min_outside", "far_max_outside", "both", 21, 21),  # full_cover -> 21
        (
            200,
            "far_min_outside",
            "far_max_outside",
            "none",
            21,
            21,
        ),  # full_cover triggers conservative shortcut -> 21
        (
            360,
            "far_min_outside",
            "far_max_outside",
            "middle",
            21,
            21,
        ),  # periodic, extent=1.5*span; region B covers full FOV -> 21
        (
            360,
            "far_min_outside",
            "far_max_outside",
            "none",
            21,
            21,
        ),  # periodic, region B triggers conservative full-cover shortcut -> 21
    ],
)
def test_isect_lidar_corner_cases(
    lidar_param,
    gaussian_param,
    expected_tiles: int,
):
    from gsplat.cuda._torch_impl_lidar import _isect_tiles_lidar
    from gsplat.cuda._wrapper import isect_tiles_lidar
    from gsplat.cuda._lidar import relative_angle

    # Convenient aliases
    lidar = lidar_param
    gaussian = gaussian_param

    # Call cuda implementation
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_lidar(
        lidar, gaussian.means2d, gaussian.radii, gaussian.depths, sort=False
    )
    # Call reference implementation
    _tiles_per_gauss, _isect_ids, _flatten_ids = _isect_tiles_lidar(
        lidar, gaussian.means2d, gaussian.radii, gaussian.depths, sort=False
    )

    # CUDA and PyTorch reference must agree for this corner-case setup.
    torch.testing.assert_close(tiles_per_gauss, _tiles_per_gauss)
    torch.testing.assert_close(isect_ids, _isect_ids)
    torch.testing.assert_close(flatten_ids, _flatten_ids)

    assert tiles_per_gauss.shape[0] == 1
    observed_tiles = int(tiles_per_gauss.item())

    hfov_start_pix = lidar.fov_horiz_rad.start * ANGLE_TO_PIXEL_SCALING_FACTOR

    def abs_to_rel_az(abs_az_pix: float) -> float:
        return relative_angle(
            hfov_start_pix,
            abs_az_pix,
            lidar.spinning_direction,
            scale=ANGLE_TO_PIXEL_SCALING_FACTOR,
        )

    hfov_span_pix = lidar.fov_horiz_rad.span * ANGLE_TO_PIXEL_SCALING_FACTOR
    gauss_mean_pix = gaussian_param.means2d[0, 0, 1].item()
    gauss_radius_pix = gaussian_param.radii[0, 0, 1].item()
    gauss_mean_rel_pix = abs_to_rel_az(gauss_mean_pix)
    gauss_min_rel_pix = abs_to_rel_az(gauss_mean_pix - gauss_radius_pix)
    gauss_max_rel_pix = abs_to_rel_az(gauss_mean_pix + gauss_radius_pix)

    assert (
        observed_tiles == expected_tiles
    ), f"Expected tiles={expected_tiles}, got tiles={observed_tiles} -> gaussian:[{gauss_min_rel_pix};{gauss_radius_pix}@{gauss_mean_rel_pix}:{gauss_max_rel_pix}), hfov-span:{hfov_span_pix}"

    if expected_tiles > 0:
        assert isect_ids.numel() == observed_tiles
        assert flatten_ids.numel() == observed_tiles
        assert torch.all(flatten_ids == 0)
    else:
        assert observed_tiles == 0
        assert isect_ids.numel() == 0
        assert flatten_ids.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("channels", [3, 32, 128])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_rasterize_to_pixels(test_data, channels: int, batch_dims: Tuple[int, ...]):
    from gsplat.cuda._torch_impl import _rasterize_to_pixels
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
        quat_scale_to_covar_preci,
        rasterize_to_pixels,
    )

    torch.manual_seed(42)

    N = test_data["means"].shape[-2]
    C = test_data["viewmats"].shape[-3]
    I = math.prod(batch_dims) * C
    test_data.update(
        {
            "colors": torch.rand(C, N, channels, device=device),
            "backgrounds": torch.rand((C, channels), device=device),
        }
    )
    test_data = expand(test_data, batch_dims)
    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"] * 0.1
    means = test_data["means"]
    opacities = test_data["opacities"]
    colors = test_data["colors"]
    backgrounds = test_data["backgrounds"]

    covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=True)

    # Project Gaussians to 2D
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means, covars, None, None, viewmats, Ks, width, height
    )
    opacities = torch.broadcast_to(opacities[..., None, :], batch_dims + (C, N))

    # Identify intersecting tiles
    tile_size = 16 if channels <= 32 else 4
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    means2d.requires_grad = True
    conics.requires_grad = True
    colors.requires_grad = True
    opacities.requires_grad = True
    backgrounds.requires_grad = True

    # forward
    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )
    _render_colors, _render_alphas = _rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )
    torch.testing.assert_close(render_colors, _render_colors)
    torch.testing.assert_close(render_alphas, _render_alphas)

    # backward
    v_render_colors = torch.randn_like(render_colors)
    v_render_alphas = torch.randn_like(render_alphas)

    v_means2d, v_conics, v_colors, v_opacities, v_backgrounds = torch.autograd.grad(
        (render_colors * v_render_colors).sum()
        + (render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, opacities, backgrounds),
    )
    (
        _v_means2d,
        _v_conics,
        _v_colors,
        _v_opacities,
        _v_backgrounds,
    ) = torch.autograd.grad(
        (_render_colors * v_render_colors).sum()
        + (_render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, opacities, backgrounds),
    )
    torch.testing.assert_close(v_means2d, _v_means2d, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(v_conics, _v_conics, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_colors, _v_colors, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_opacities, _v_opacities, rtol=8e-3, atol=6e-3)
    torch.testing.assert_close(v_backgrounds, _v_backgrounds, rtol=1e-3, atol=1e-3)


def _quat_rotation_y(angle_rad: float, device: torch.device):
    """Quaternion for rotation around Y axis (w, x, y, z)."""
    half = angle_rad / 2.0
    w = math.cos(half)
    y = math.sin(half)
    return torch.tensor([[w, 0.0, y, 0.0]], device=device, dtype=torch.float32)


def _expected_hit_distance_canonical_ray_distance(
    ray_o: torch.Tensor,
    ray_d: torch.Tensor,
    xyz: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
) -> float:
    """Expected hit distance using the formula: length(scale * grd * hit_t)."""
    from gsplat.cuda._math import _quat_scale_to_preci_half
    from gsplat.cuda._torch_impl_eval3d import _safe_normalize

    # iscl_rot = M^T with M = R * S (inverse scale-rotation); batch size 1
    M = _quat_scale_to_preci_half(quats, scales)
    iscl_rot = M.transpose(-2, -1)
    # ray_o, ray_d, xyz are [3]; broadcast for matmul [1,3,3] @ [1,3,1]
    gro = (
        torch.matmul(iscl_rot, (ray_o - xyz).unsqueeze(0).unsqueeze(-1))
        .squeeze(-1)
        .squeeze(0)
    )
    grd = (
        torch.matmul(iscl_rot, ray_d.unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(0)
    )
    grd = _safe_normalize(grd)
    hit_t = (grd * (-gro)).sum().item()
    grds = scales.squeeze(0) * grd * hit_t
    return torch.linalg.norm(grds).item()


def _pixel_ray_dir_pinhole(
    px: float,
    py: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: torch.device,
) -> torch.Tensor:
    """Ray direction in world space for pinhole camera at origin (viewmat identity).
    Returns unit vector: normalize(( (px-cx)/fx, (py-cy)/fy, 1 )).
    """
    dx = (px - cx) / fx
    dy = (py - cy) / fy
    d = torch.tensor([dx, dy, 1.0], device=device, dtype=torch.float32)
    return d / torch.linalg.norm(d)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize(
    "means_list,quats_choice,scales_list,pixel_dx,pixel_dy",
    [
        # On-axis, mid distance (baseline); sample at center
        ([[0.0, 0.0, 5.0]], "identity", (0.15, 0.15, 0.15), 0, 0),
        # On-axis, near camera
        ([[0.0, 0.0, 1.0]], "identity", (0.15, 0.15, 0.15), 0, 0),
        # On-axis, far from camera (larger scale so gaussian visible at D=50)
        ([[0.0, 0.0, 50.0]], "identity", (0.6, 0.6, 0.6), 0, 0),
        # Anisotropic scaling (on-axis)
        ([[0.0, 0.0, 5.0]], "identity", (0.2, 0.05, 0.15), 0, 0),
        # Off-axis: gaussian not on principal ray; sample at center
        ([[1.5, 0.5, 5.0]], "identity", (0.2, 0.2, 0.2), 0, 0),
        # Rotated gaussian (45 deg around Y), isotropic scale
        ([[0.0, 0.0, 5.0]], "rotated_y_45", (0.15, 0.15, 0.15), 0, 0),
        # Rotated + anisotropic
        ([[0.0, 0.0, 5.0]], "rotated_y_45", (0.2, 0.05, 0.15), 0, 0),
        # Flat / disk-like
        ([[0.0, 0.0, 5.0]], "identity", (0.08, 0.08, 0.25), 0, 0),
        # Off-center rays: sample pixel offset from gaussian center so ray does NOT pass through center.
        # Offset in pixels from gaussian center (means2d); keeps pixel inside splat.
        ([[0.0, 0.0, 5.0]], "identity", (0.4, 0.4, 0.4), 3, 0),
        ([[0.0, 0.0, 5.0]], "identity", (0.4, 0.4, 0.4), -2, 2),
        ([[0.0, 0.0, 5.0]], "identity", (0.4, 0.4, 0.4), 0, -3),
        ([[1.5, 0.0, 5.0]], "identity", (0.35, 0.35, 0.35), -2, 0),
        ([[-1.0, 1.0, 6.0]], "identity", (0.35, 0.35, 0.35), 2, -1),
    ],
    ids=[
        "on_axis_mid",
        "on_axis_near",
        "on_axis_far",
        "on_axis_anisotropic",
        "off_axis",
        "rotated_isotropic",
        "rotated_anisotropic",
        "flat_disk",
        "off_center_right",
        "off_center_left_up",
        "off_center_down",
        "off_center_gauss_right_pixel_left",
        "off_center_gauss_ul_pixel_lr",
    ],
)
def test_rasterize_to_pixels_hit_distance_principal_axis(
    means_list, quats_choice, scales_list, pixel_dx, pixel_dy
):
    """Check that hit distance (accumulated/alpha) matches the hit-distance formula:
    length(scale * grd * hit_t) with hit_t = dot(grd, -gro). Includes center and off-center
    pixels. Failures indicate the rasterization hit-distance code should be analyzed and fixed.
    """
    from gsplat.cuda._wrapper import (
        fully_fused_projection_with_ut,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels_eval3d_extra,
    )

    width = height = 32
    tile_size = 16
    N = 1
    C = 1
    I = C
    fx = fy = float(width)
    cx = width / 2.0
    cy = height / 2.0

    means = torch.tensor(means_list, device=device, dtype=torch.float32)
    if quats_choice == "identity":
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    elif quats_choice == "rotated_y_45":
        quats = _quat_rotation_y(math.pi / 4.0, device)
    else:
        raise ValueError(f"Unknown quats_choice: {quats_choice}")

    scales = torch.tensor([list(scales_list)], device=device, dtype=torch.float32)
    opacities = torch.tensor([1.0], device=device, dtype=torch.float32)

    viewmats = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    Ks = torch.tensor(
        [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )

    colors = torch.zeros(C, N, 3, device=device, dtype=torch.float32)
    opacities_broadcast = opacities.unsqueeze(0)

    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        means, quats, scales, opacities, viewmats, Ks, width, height
    )
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)

    render_colors, render_alphas, _, _, _ = rasterize_to_pixels_eval3d_extra(
        means,
        quats,
        scales,
        colors,
        opacities_broadcast,
        viewmats,
        Ks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        use_hit_distance=True,
        return_sample_counts=False,
        return_normals=False,
    )

    # Sample pixel: center of gaussian projection, or offset from center (means2d + dx, dy) for off-center rays
    cx_f = means2d[0, 0, 0].item()
    cy_f = means2d[0, 0, 1].item()
    if pixel_dx == 0 and pixel_dy == 0:
        px_f = cx_f
        py_f = cy_f
    else:
        px_f = cx_f + pixel_dx
        py_f = cy_f + pixel_dy
    px_int = int(round(px_f))
    py_int = int(round(py_f))
    px_int = max(0, min(width - 1, px_int))
    py_int = max(0, min(height - 1, py_int))

    accumulated_hit = render_colors[0, py_int, px_int, -1].item()
    alpha = render_alphas[0, py_int, px_int, 0].item()

    assert alpha > 0, (
        f"Pixel ({px_int}, {py_int}) should hit the gaussian "
        f"(sample px_f={px_f}, py_f={py_f}; means2d=({cx_f}, {cy_f}))"
    )

    ray_o = torch.zeros(3, device=device, dtype=torch.float32)
    pixel_center_x = px_int + 0.5
    pixel_center_y = py_int + 0.5
    ray_d = _pixel_ray_dir_pinhole(
        pixel_center_x, pixel_center_y, fx, fy, cx, cy, device
    )
    expected_hit = _expected_hit_distance_canonical_ray_distance(
        ray_o, ray_d, means[0], quats, scales
    )

    observed_hit_distance = accumulated_hit / alpha

    torch.testing.assert_close(
        torch.tensor(observed_hit_distance, device=device),
        torch.tensor(expected_hit, device=device),
        rtol=2.5e-4,
        atol=2e-3,
    )


# Since we have comprehensive camera model tests, we don't need to add
# a camera model axis to this test. We use perfect pinhole model instead.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize(
    "channels,batch_dims,rs_type,use_hit_distance,use_rays,return_normals",
    [
        pytest.param(
            *params,
            marks=[
                # test based on use_rays (4)
                pytest.mark.skipif(
                    params[4] and not has_camera_wrappers(),
                    reason="Camera wrapper support isn't built in",
                )
            ],
        )
        for params in chain(
            # Main test combinations with return_normals=False
            product(
                [3],  # channels
                [(), (2,), (1, 2)],  # batch_dims
                [
                    RollingShutterType.GLOBAL,
                    RollingShutterType.ROLLING_TOP_TO_BOTTOM,
                ],  # rs_type
                [True, False],  # use_hit_distance
                [True, False],  # use_rays
                [False],  # return_normals
            ),
            # Dedicated test for return_normals=True with one configuration
            [(3, (), RollingShutterType.ROLLING_TOP_TO_BOTTOM, True, True, True)],
        )
    ],
)
def test_rasterize_to_pixels_eval3d(
    test_data,
    channels: int,
    batch_dims: Tuple[int, ...],
    rs_type: RollingShutterType,
    use_hit_distance: bool,
    use_rays: bool,
    return_normals: bool,
):
    from gsplat.cuda._torch_impl_eval3d import _rasterize_to_pixels_eval3d
    from gsplat.cuda._wrapper import (
        fully_fused_projection_with_ut,
        isect_offset_encode,
        isect_tiles,
        quat_scale_to_covar_preci,
        rasterize_to_pixels_eval3d_extra,
        RollingShutterType,
    )

    N = test_data["means"].shape[-2]  # number of Gaussians
    C = test_data["viewmats"].shape[-3]  # number of cameras
    I = math.prod(batch_dims) * C  # number of images (I = B * C)

    downscale = 4 * I

    # Reduce image dimensions by half to save memory
    test_data["height"] = test_data["height"] // downscale
    test_data["width"] = test_data["width"] // downscale
    # Adjust camera intrinsics for smaller image
    test_data["Ks"] = test_data["Ks"].clone()
    test_data["Ks"][..., 0, 0] /= downscale  # fx
    test_data["Ks"][..., 1, 1] /= downscale  # fy
    test_data["Ks"][..., 0, 2] /= downscale  # cx
    test_data["Ks"][..., 1, 2] /= downscale  # cy

    test_data.update(
        {
            "colors": torch.rand(C, N, channels, device=device),
            "backgrounds": torch.rand((C, channels), device=device),
        }
    )
    test_data = expand(test_data, batch_dims)
    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"] * 0.1
    means = test_data["means"]
    opacities = test_data["opacities"]
    colors = test_data["colors"]
    backgrounds = test_data["backgrounds"]

    # Create viewmats_rs for rolling shutter testing
    if rs_type != RollingShutterType.GLOBAL:
        # Simulate camera motion with small perturbation
        viewmats_rs = viewmats.clone()
        # Add small translation (5% of scene scale)
        viewmats_rs[..., :3, 3] += torch.randn_like(viewmats[..., :3, 3]) * 0.05
    else:
        viewmats_rs = None

    if use_rays:
        camera = create_camera_model(
            width=width,
            height=height,
            camera_model="pinhole",
            focal_lengths=Ks[..., [0, 1], [0, 1]].contiguous(),
            principal_points=Ks[..., [0, 1], [2, 2]].contiguous(),
            rs_type=rs_type,
        )

        gridx, gridy = torch.meshgrid(
            [
                torch.arange(0, width, device=device, dtype=torch.float32),
                torch.arange(0, height, device=device, dtype=torch.float32),
            ],
            indexing="ij",
        )

        batch_shape = Ks.shape[:-2]

        grid = torch.stack([gridx, gridy], dim=-1)
        grid = grid.expand(*batch_shape, *grid.shape).reshape(
            *batch_shape, width * height, 2
        )

        pose_start = _viewmat_to_pose(viewmats)
        if viewmats_rs is None:
            pose_end = pose_start
        else:
            pose_end = _viewmat_to_pose(viewmats_rs)

        rori, rdir, rvalid = camera.image_point_to_world_ray_shutter_pose(
            grid, pose_start, pose_end
        )
        assert (rvalid == False).sum() == 0
        rays = torch.cat([rori, rdir], -1)
        rays.reshape(*batch_shape, height, width, 6)
    else:
        rays = None

    # Project Gaussians to 2D for tile intersections
    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        means, quats, scales, opacities, viewmats, Ks, width, height
    )
    opacities_broadcast = torch.broadcast_to(
        opacities[..., None, :], batch_dims + (C, N)
    )

    # Identify intersecting tiles
    tile_size = 16
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    means.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    colors.requires_grad = True
    opacities_broadcast.requires_grad = True
    backgrounds.requires_grad = True
    if use_rays:
        rays.requires_grad = True

    # forward - CUDA implementation
    (
        render_colors,
        render_alphas,
        render_last_ids,
        render_sample_counts,
        render_normals,
    ) = rasterize_to_pixels_eval3d_extra(
        means,
        quats,
        scales,
        colors,
        opacities_broadcast,
        viewmats,
        Ks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        return_sample_counts=True,
        rolling_shutter=rs_type,
        viewmats_rs=viewmats_rs,
        use_hit_distance=use_hit_distance,
        rays=rays,
        return_normals=return_normals,
    )

    # forward - PyTorch reference implementation (with tiling optimization)
    (
        _render_colors,
        _render_alphas,
        _render_last_ids,
        _render_sample_counts,
        *_render_normals,
    ) = _rasterize_to_pixels_eval3d(
        means,
        quats,
        scales,
        colors,
        opacities_broadcast,
        viewmats,
        Ks,
        width,
        height,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
        return_last_ids=True,
        return_sample_counts=True,
        rs_type=rs_type,
        viewmats_rs=viewmats_rs,
        use_hit_distance=use_hit_distance,
        rays=rays,
        return_normals=return_normals,
    )

    _render_normals = _render_normals[0] if return_normals else None

    # Validate: last_ids and alpha must be consistent
    # Alpha > 0 if and only if there's a Gaussian intersecting with the pixel's ray (last_ids >= 0)
    cuda_consistent = (render_last_ids >= 0) == (render_alphas.squeeze(-1) > 0)
    ref_consistent = (_render_last_ids >= 0) == (_render_alphas.squeeze(-1) > 0)

    torch.testing.assert_close(
        ref_consistent.float(),
        torch.ones_like(ref_consistent, dtype=torch.float32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        cuda_consistent.float(),
        torch.ones_like(cuda_consistent, dtype=torch.float32),
        atol=0,
        rtol=0,
    )

    # Validate: last_ids and sample_counts must be consistent:
    # Sample counts must be > 0 if and only if there's
    # a Gaussian intersecting with the pixel's ray (last_ids >= 0)
    cuda_consistent = (render_sample_counts > 0) == (render_last_ids >= 0)
    ref_consistent = (_render_sample_counts > 0) == (_render_last_ids >= 0)

    torch.testing.assert_close(
        ref_consistent.float(),
        torch.ones_like(ref_consistent, dtype=torch.float32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        cuda_consistent.float(),
        torch.ones_like(cuda_consistent, dtype=torch.float32),
        atol=0,
        rtol=0,
    )

    # Categorize pixels into 3 groups with different error tolerances.
    # 1. The number of samples accumulated per pixel are the same in both ref and cuda.
    #    Error tolerance must be low, and due to small numerical errors.
    # 2. One has samples accumulated, the other doesn't
    #    This might happen again due to small numerical errors, but related to the
    #    threshold for discarding samples with low alpha.
    #    Error tolerance is even higher and related to the alpha threshold.
    # 3. Both have samples accumulated, but different counts
    #    This happens due to small numerical errors that lead to the sample being
    #    discarded in one and not in the other due to the low transmission threshold.
    #    Error tolerance is higher,

    # match: last_ids match (same accumulation endpoint)
    cuda_has_isect = render_last_ids >= 0  # [batch, C, H, W]
    ref_has_isect = _render_last_ids >= 0

    # 1. count_match: same number of samples accumulated
    count_match = render_sample_counts == _render_sample_counts  # [batch, C, H, W]
    # 2. vis_mismatch: One has samples, one doesn't (not visible)
    vis_mismatch = cuda_has_isect ^ ref_has_isect
    # 3. count_mismatch: Both have samples but different counts
    count_mismatch = ~(count_match | vis_mismatch)

    # Each pixel must be in only one of the three groups
    assert not (count_mismatch & vis_mismatch).any()
    assert not (count_match & vis_mismatch).any()
    assert not (count_mismatch & count_match).any()

    count_match = count_match.unsqueeze(-1).float()
    count_mismatch = count_mismatch.unsqueeze(-1).float()
    vis_mismatch = vis_mismatch.unsqueeze(-1).float()

    assert count_match.sum() > 0

    # Compare alphas for each group
    torch.testing.assert_close(
        render_alphas * count_match, _render_alphas * count_match, rtol=1e-2, atol=2e-3
    )
    torch.testing.assert_close(
        render_alphas * vis_mismatch,
        _render_alphas * vis_mismatch,
        rtol=0,
        atol=ALPHA_THRESHOLD + 1e-5,
    )
    torch.testing.assert_close(
        render_alphas * count_mismatch,
        _render_alphas * count_mismatch,
        rtol=0,
        atol=5e-3,
    )

    # Compare colors for each group (expand masks to [batch, C, H, W, 3])
    count_match = count_match.expand_as(render_colors)
    vis_mismatch = vis_mismatch.expand_as(render_colors)
    count_mismatch = count_mismatch.expand_as(render_colors)

    torch.testing.assert_close(
        render_colors * count_match, _render_colors * count_match, rtol=3e-3, atol=1e-3
    )
    # Bumped tolerance due to release mode optimizations. In debug mode it's ALPHA_THRESHOLD+1e-5.
    torch.testing.assert_close(
        render_colors * vis_mismatch,
        _render_colors * vis_mismatch,
        rtol=0,
        atol=ALPHA_THRESHOLD + 5e-3,
    )
    torch.testing.assert_close(
        render_colors * count_mismatch,
        _render_colors * count_mismatch,
        rtol=2e-2,
        atol=3e-3,
    )

    # Compare normals if computed
    if return_normals:
        assert (
            render_normals is not None
        ), "CUDA render_normals should not be None when return_normals=True"
        assert (
            _render_normals is not None
        ), "PyTorch render_normals should not be None when return_normals=True"

        # With the default tolerances, the error is quite small:
        # Greatest absolute difference: 5.447864532470703e-05 at index (2, 11, 16, 2) (up to 1e-05 allowed)
        # Greatest relative difference: 0.00024596037110313773 at index (0, 27, 39, 1) (up to 1.3e-06 allowed)
        # Setting tolerances small enough to ignore these errors.
        torch.testing.assert_close(
            render_normals, _render_normals, rtol=3e-4, atol=6e-5
        )
    else:
        assert (
            render_normals is None
        ), "CUDA render_normals should be None when return_normals=False"
        assert (
            _render_normals is None
        ), "PyTorch render_normals should be None when return_normals=False"

    # Test the gradients now

    v_render_colors = torch.randn_like(render_colors)
    v_render_alphas = torch.randn_like(render_alphas)
    v_render_normals = torch.randn_like(render_normals) if return_normals else None

    torch.manual_seed(42)
    perm_idx = torch.randperm(render_colors.shape[0])

    def randperm(x):
        return x[perm_idx]

    # Build the loss for backward pass
    loss_cuda = randperm(
        render_colors * v_render_colors
        + render_alphas * v_render_alphas
        + (render_normals * v_render_normals if return_normals else 0)
    ).sum()
    loss_ref = randperm(
        _render_colors * v_render_colors
        + _render_alphas * v_render_alphas
        + (_render_normals * v_render_normals if return_normals else 0)
    ).sum()

    if use_rays:
        (
            v_means,
            v_quats,
            v_scales,
            v_colors,
            v_opacities,
            v_backgrounds,
            v_rays,
        ) = torch.autograd.grad(
            loss_cuda,
            (means, quats, scales, colors, opacities_broadcast, backgrounds, rays),
            retain_graph=True,
        )

        (
            _v_means,
            _v_quats,
            _v_scales,
            _v_colors,
            _v_opacities,
            _v_backgrounds,
            _v_rays,
        ) = torch.autograd.grad(
            loss_ref,
            (means, quats, scales, colors, opacities_broadcast, backgrounds, rays),
            retain_graph=True,
        )
    else:
        (
            v_means,
            v_quats,
            v_scales,
            v_colors,
            v_opacities,
            v_backgrounds,
        ) = torch.autograd.grad(
            loss_cuda,
            (means, quats, scales, colors, opacities_broadcast, backgrounds),
            retain_graph=True,
        )

        (
            _v_means,
            _v_quats,
            _v_scales,
            _v_colors,
            _v_opacities,
            _v_backgrounds,
        ) = torch.autograd.grad(
            loss_ref,
            (means, quats, scales, colors, opacities_broadcast, backgrounds),
            retain_graph=True,
        )

    # Create visibility mask [N] for Gaussians
    # We want to consider only visible gaussians so that the invisible ones don't
    # skew the error statistics.
    visible_mask = torch.zeros(N, dtype=torch.bool, device=device)
    visible_mask[(flatten_ids % N).unique().long()] = True

    # Reshape gradients to uniform shape [C_or_1, N, channels] for consistent masking
    v_means = v_means[None, ...]  # [N, 3] → [1, N, 3]
    _v_means = _v_means[None, ...]
    v_quats = v_quats[None, ...]  # [N, 4] → [1, N, 4]
    _v_quats = _v_quats[None, ...]
    v_scales = v_scales[None, ...]  # [N, 3] → [1, N, 3]
    _v_scales = _v_scales[None, ...]
    v_opacities = v_opacities[..., None]  # [C, N] → [C, N, 1]
    _v_opacities = _v_opacities[..., None]
    # v_colors already [C, N, 3]

    # Expand visibility mask to [1, N, 1] for broadcasting
    visible_mask = visible_mask[None, :, None]

    assert visible_mask.sum() > 0

    # Extract visible elements once (use reshaped gradients for expand_as)
    means_mask = visible_mask.expand_as(v_means) & get_inlier_abserror_mask(
        v_means, _v_means, quantile=0.90
    )
    scales_mask = visible_mask.expand_as(v_scales) & get_inlier_abserror_mask(
        v_scales, _v_scales, quantile=0.90
    )
    quats_mask = visible_mask.expand_as(v_quats) & get_inlier_abserror_mask(
        v_quats, _v_quats, quantile=0.99
    )
    colors_mask = visible_mask.expand_as(v_colors) & get_inlier_abserror_mask(
        v_colors, _v_colors, quantile=0.99
    )
    opacities_mask = visible_mask.expand_as(v_opacities) & get_inlier_abserror_mask(
        v_opacities, _v_opacities, quantile=0.99
    )
    backgrounds_mask = get_inlier_abserror_mask(
        v_backgrounds, _v_backgrounds, quantile=0.99
    )

    assert means_mask.sum() > 0
    assert scales_mask.sum() > 0
    assert quats_mask.sum() > 0
    assert colors_mask.sum() > 0
    assert opacities_mask.sum() > 0
    assert backgrounds_mask.sum() > 0

    # Compare backward gradients, excluding the ones that fall above the quantile threshold.
    torch.testing.assert_close(
        v_means * means_mask.float(), _v_means * means_mask.float(), rtol=0, atol=4e-2
    )
    torch.testing.assert_close(
        v_scales * scales_mask.float(),
        _v_scales * scales_mask.float(),
        rtol=0,
        atol=5e-2,
    )
    # Relax quat/opacity tolerances when use_hit_distance=True due to accumulated floating-point errors
    # in hit distance calculation (normalize + dot product + length operations)
    quat_atol = 6e-3 if use_hit_distance else 5e-4
    opacity_atol = 1e-3 if use_hit_distance else 1.5e-4
    torch.testing.assert_close(
        v_quats * quats_mask.float(),
        _v_quats * quats_mask.float(),
        rtol=0,
        atol=quat_atol,
    )
    torch.testing.assert_close(
        v_colors * colors_mask.float(),
        _v_colors * colors_mask.float(),
        rtol=0,
        atol=1e-4,
    )
    torch.testing.assert_close(
        v_opacities * opacities_mask.float(),
        _v_opacities * opacities_mask.float(),
        rtol=0,
        atol=opacity_atol,
    )
    torch.testing.assert_close(
        v_backgrounds * backgrounds_mask.float(),
        _v_backgrounds * backgrounds_mask.float(),
        rtol=0,
        atol=1.6e-2,
    )

    if use_rays:
        rays_mask = get_inlier_abserror_mask(v_rays, _v_rays, quantile=0.95)
        assert rays_mask.sum() > 0
        torch.testing.assert_close(
            v_rays * rays_mask.float(), _v_rays * rays_mask.float(), rtol=0, atol=4e-2
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_sh(test_data, sh_degree: int, batch_dims: Tuple[int, ...]):
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from gsplat.cuda._wrapper import spherical_harmonics

    torch.manual_seed(42)

    N = 1000
    test_data = {
        "coeffs": torch.randn(N, (4 + 1) ** 2, 3, device=device),
        "dirs": torch.randn(N, 3, device=device),
    }
    test_data = expand(test_data, batch_dims)
    coeffs = test_data["coeffs"]
    dirs = test_data["dirs"]
    coeffs.requires_grad = True
    dirs.requires_grad = True

    colors = spherical_harmonics(sh_degree, dirs, coeffs)
    _colors = _spherical_harmonics(sh_degree, dirs, coeffs)
    torch.testing.assert_close(colors, _colors, rtol=1e-4, atol=1e-4)

    v_colors = torch.randn_like(colors)

    v_coeffs, v_dirs = torch.autograd.grad(
        (colors * v_colors).sum(), (coeffs, dirs), retain_graph=True, allow_unused=True
    )
    _v_coeffs, _v_dirs = torch.autograd.grad(
        (_colors * v_colors).sum(), (coeffs, dirs), retain_graph=True, allow_unused=True
    )
    torch.testing.assert_close(v_coeffs, _v_coeffs, rtol=1e-4, atol=1e-4)
    if sh_degree > 0:
        torch.testing.assert_close(v_dirs, _v_dirs, rtol=1e-4, atol=1e-4)
