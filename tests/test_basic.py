# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
import os
from itertools import product

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

from gsplat.cuda._wrapper import (
    CameraModel,
    RollingShutterType,
    UnscentedTransformParameters,
    _make_lazy_cuda_obj,
)
from gsplat.cuda._math import _safe_normalize
from gsplat.cuda._torch_cameras import _viewmat_to_pose

BaseCameraModelCUDA = _make_lazy_cuda_obj("BaseCameraModel")

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


# Since we have comprehensive camera model tests, we don't need to add
# a camera model axis to this test. We use perfect pinhole model instead.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize("channels", [3])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
@pytest.mark.parametrize(
    "rs_type", [RollingShutterType.GLOBAL, RollingShutterType.ROLLING_TOP_TO_BOTTOM]
)
@pytest.mark.parametrize("use_hit_distance", [True, False], ids=["hitdist", "depth"])
@pytest.mark.parametrize("use_rays", [True, False], ids=["rays", "sensor"])
def test_rasterize_to_pixels_eval3d(
    test_data,
    channels: int,
    batch_dims: Tuple[int, ...],
    rs_type: RollingShutterType,
    use_hit_distance: bool,
    use_rays: bool,
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
        camera = BaseCameraModelCUDA.create(
            width=width,
            height=height,
            camera_model="pinhole",
            focal_lengths=Ks[..., [0, 1], [0, 1]].contiguous(),
            principal_points=Ks[..., [0, 1], [2, 2]].contiguous(),
            rs_type=rs_type.to_cpp(),
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
    radii, means2d, depths, conics, compensations = fully_fused_projection_with_ut(
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
    )

    # forward - PyTorch reference implementation (with tiling optimization)
    (
        _render_colors,
        _render_alphas,
        _render_last_ids,
        _render_sample_counts,
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
    )

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

    alpha_threshold = 1.0 / 255.0

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
        atol=alpha_threshold + 1e-5,
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
    # Bumped tolerance due to release mode optimizations. In debug mode it's alpha_threshold+1e-5.
    torch.testing.assert_close(
        render_colors * vis_mismatch,
        _render_colors * vis_mismatch,
        rtol=0,
        atol=alpha_threshold + 5e-3,
    )
    torch.testing.assert_close(
        render_colors * count_mismatch,
        _render_colors * count_mismatch,
        rtol=2e-2,
        atol=3e-3,
    )

    # Test the gradients now

    v_render_colors = torch.randn_like(render_colors)
    v_render_alphas = torch.randn_like(render_alphas)

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
            (render_colors * v_render_colors).sum()
            + (render_alphas * v_render_alphas).sum(),
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
            (_render_colors * v_render_colors).sum()
            + (_render_alphas * v_render_alphas).sum(),
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
            (render_colors * v_render_colors).sum()
            + (render_alphas * v_render_alphas).sum(),
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
            (_render_colors * v_render_colors).sum()
            + (_render_alphas * v_render_alphas).sum(),
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
