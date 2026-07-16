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

import math
import struct
import os
from enum import Enum
from itertools import chain, product
from types import SimpleNamespace

import pytest
import torch
from typing_extensions import Literal, Tuple, assert_never
import torch.nn.functional as F

import gsplat

from gsplat.cuda._backend import _C

if _C is None:
    pytest.skip("gsplat CUDA extension not available", allow_module_level=True)

from gsplat._helper import (
    load_test_data,
    get_inlier_abserror_mask,
    assert_mismatch_ratio,
    assert_close_with_boundary_band,
    assert_grad_reference_close,
    assert_grad_sparsity,
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
        data_path=os.path.join(
            os.path.dirname(__file__), "../../assets/test_garden.npz"
        ),
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

    covars_only, no_precis = quat_scale_to_covar_preci(
        quats, scales, compute_preci=False, triu=triu
    )
    assert no_precis is None
    torch.testing.assert_close(covars_only, covars)

    no_covars, precis_only = quat_scale_to_covar_preci(
        quats, scales, compute_covar=False, triu=triu
    )
    assert no_covars is None
    torch.testing.assert_close(precis_only, precis)

    no_covars, no_precis = quat_scale_to_covar_preci(
        quats, scales, compute_covar=False, compute_preci=False, triu=triu
    )
    assert no_covars is None
    assert no_precis is None

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

    # Gradient comparison via assert_close_with_boundary_band.
    # Per-element value diff with a "ref near-zero" boundary band: an element
    # whose reference value sits below the FP noise floor (|ref| < nz_thresh)
    # has unbounded relative tolerance and is admitted into the band; the
    # flip predicate fires when CUDA's value rises significantly above that
    # floor (catches "false non-zero" sparsity bugs). Off-band elements get
    # a tight per-element rtol -- catches magnitude bias, sign flip, "false
    # zero" sparsity bugs (CUDA outputs 0 where ref outputs non-zero), and
    # most off-by-one bugs in the backward chain.
    for name, vc, vt in [
        ("v_quats", v_quats, _v_quats),
        ("v_scales", v_scales, _v_scales),
    ]:
        assert not (
            torch.isnan(vc).any() or torch.isinf(vc).any()
        ), f"{name}: CUDA produced NaN/Inf"
        assert_grad_sparsity(vc, vt, min_ratio=0.1, msg=f"{name} backward sparsity")
        # Tight nz_thresh (1e-7 of max) -- this test has narrow gradient
        # magnitude distribution, so most elements are in interior and
        # 0.3% bias is reliably caught.
        nz_thresh = vt.abs().max().item() * 1e-7
        boundary_mask = vt.abs() < nz_thresh
        # interior_rtol envelope x 1.05:
        #   RTX PRO 2000  worst rtol_req=1.2e-3   (triu=False)
        #   RTX PRO 6000  worst rtol_req=1.55e-2  (triu=True)
        assert_close_with_boundary_band(
            vc,
            vt,
            boundary_mask=boundary_mask,
            interior_atol=nz_thresh,
            interior_rtol=1.65e-2,
            boundary_max_flip_ratio=1e-3,
            boundary_symmetry_tol=0.5,
            flip_predicate=lambda a, e: a.abs() > 10 * nz_thresh,
            msg=f"{name} backward (triu={triu})",
        )


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
    # covars2d: pinhole projection involves J*Sigma*J^T where J is the
    # 2x2 affine Jacobian; FP32 cancellation in the matrix product produces
    # non-trivial per-element noise on the tail (worst observed ~6% rel, ~0.12
    # abs).  Use band+sparsity so the bulk gets a tight check while the tail
    # is absorbed in a small flip budget.
    nz_thresh = _covars2d.abs().max().item() * 1e-3
    # interior_rtol envelope x 1.05:
    #   RTX PRO 2000  worst rtol_req=7e-7
    #   RTX PRO 6000  worst rtol_req=6.55e-6
    # interior_atol stays at 2e-3 -- the FP32 cancellation noise floor on
    # this matrix product.
    assert_close_with_boundary_band(
        covars2d,
        _covars2d,
        boundary_mask=_covars2d.abs() < nz_thresh,
        interior_atol=2e-3,
        interior_rtol=7e-6,
        boundary_max_flip_ratio=1e-3,
        boundary_symmetry_tol=0.5,
        flip_predicate=lambda a, e, _t=nz_thresh: a.abs() > 10 * _t,
        msg="proj covars2d (forward)",
    )
    # Outlier guard: even admitted in-band flips must fit a per-element
    # bound matching the original test's loose tolerance, so a NaN / catastrophic
    # single-element bug cannot hide inside the boundary flip budget.  The
    # original was rtol=0.1, atol=3e-2; covars2d magnitudes reach ~1e11 on
    # near-zero-depth Gaussians, where a small absolute outlier guard would
    # false-fire on FP noise relative to that magnitude.
    _diff_cv = (covars2d - _covars2d).abs()
    _outlier_bound_cv = 3e-2 + 0.1 * _covars2d.abs()
    assert (_diff_cv <= _outlier_bound_cv).all(), (
        f"proj covars2d: outlier diff {_diff_cv.max().item():.4e} exceeds "
        f"loose bound (atol=3e-2, rtol=0.1)"
    )

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
    # Per-element band+sparsity check (cluster-A pattern: rel-diff blows up
    # at near-zero gradient values; off-band tight rtol catches systematic
    # bias / sign flip / sparsity bug, in-band absorbs FP noise).
    #
    # Fisheye-only geometric band: forward-facing fisheye is undefined at
    # theta = atan2(xy, z) > pi/2 (z <= 0, ray pointing backward). The
    # projection `focal * theta * (x,y) / xy_len` is FP-sensitive there:
    # small xy_norm with z<0 amplifies CUDA-vs-torch ULP differences into
    # ~1e3-magnitude v_means disagreements. v_covars stays clean because
    # the bilinear J^T*v*J cancels.
    if camera_model == "fisheye":
        geom_band_pt = means.detach()[..., 2] <= 0  # [..., n_gauss]
    else:
        geom_band_pt = torch.zeros(
            means.shape[:-1], dtype=torch.bool, device=means.device
        )
    for name, vc, vt in [
        ("v_means", v_means, _v_means),
        ("v_covars", v_covars, _v_covars),
    ]:
        assert not (
            torch.isnan(vc).any() or torch.isinf(vc).any()
        ), f"{name}: CUDA produced NaN/Inf"
        assert_grad_sparsity(vc, vt, min_ratio=0.1, msg=f"{name} backward sparsity")
        nz_thresh = vt.abs().max().item() * 1e-5
        # interior_rtol envelope x 1.05:
        #   RTX PRO 2000  rtol_req=0       (atol absorbs all on pinhole/ortho)
        #   RTX PRO 6000  rtol_req=6.5e-4  (fisheye v_means)
        gb = geom_band_pt
        while gb.ndim < vc.ndim:
            gb = gb[..., None]
        boundary_mask = (vt.abs() < nz_thresh) | gb.expand_as(vc)
        # flip_predicate: magnitude-based predicate ("CUDA overflowed where
        # ref was zero") works for the magnitude-only band; the fisheye geom
        # band catches z<=0 Gaussians with legitimately large magnitudes in
        # both impls, where disagreement is the meaningful flip signal.
        # symmetry: disabled for fisheye (single-digit n_flips, all drift in
        # the same direction due to projection-formula coupling).
        if camera_model == "fisheye":
            flip_pred = lambda a, e, _t=nz_thresh: (a - e).abs() > 10 * _t
            sym_tol = 1.0
        else:
            flip_pred = lambda a, e: a.abs() > 10 * nz_thresh
            sym_tol = 0.5
        assert_close_with_boundary_band(
            vc,
            vt,
            boundary_mask=boundary_mask,
            interior_atol=nz_thresh,
            interior_rtol=7e-4,
            boundary_max_flip_ratio=1e-3,
            boundary_symmetry_tol=sym_tol,
            flip_predicate=flip_pred,
            msg=f"{name} backward (proj)",
        )


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

    # v_viewmats stays tight (already well-conditioned).
    assert_grad_reference_close(
        v_viewmats,
        _v_viewmats,
        rtol=2e-3,
        atol=2e-3,
        max_rel_l2=1e-2,
        max_rel_l1=1e-2,
        min_cosine=0.999,
        max_signed_bias=1e-2,
        msg="v_viewmats",
    )

    # Per-element band+sparsity check on the conditioning-sensitive
    # gradients (v_quats, v_scales, v_means flow through quat_scale_to_covar
    # which has 1/scale**2 singularity at small scales).
    # interior_rtol = 1.05 x envelope across calibrated GPUs (RTX PRO 2000 / 6000):
    for name, vc, vt, rtol in [
        (
            "v_quats",
            v_quats,
            _v_quats,
            1.17e-3,
        ),  # RTX PRO 2000=4.8e-5, RTX PRO 6000=5.91e-4, L40S=1.108e-3
        (
            "v_scales",
            v_scales,
            _v_scales,
            1.75e-2,
        ),  # RTX PRO 2000=2.9e-3, RTX PRO 6000=1.638e-2
        ("v_means", v_means, _v_means, 1e-5),
    ]:
        assert not (
            torch.isnan(vc).any() or torch.isinf(vc).any()
        ), f"{name}: CUDA produced NaN/Inf"
        assert_grad_sparsity(vc, vt, min_ratio=0.1, msg=f"{name} backward sparsity")
        nz_thresh = vt.abs().max().item() * 1e-5
        assert_close_with_boundary_band(
            vc,
            vt,
            boundary_mask=vt.abs() < nz_thresh,
            interior_atol=nz_thresh,
            interior_rtol=rtol,
            boundary_max_flip_ratio=1e-3,
            boundary_symmetry_tol=0.5,
            flip_predicate=lambda a, e: a.abs() > 10 * nz_thresh,
            msg=f"{name} backward",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("packed", [False, True])
def test_projection_batched_warp_reduction_labels(packed: bool):
    """Warp reductions must not merge equal Gaussian/camera IDs across batches."""
    from gsplat.cuda._wrapper import fully_fused_projection

    B, C, N = 2, 2, 8
    width = height = 32

    means = torch.zeros(B, N, 3, device=device)
    means[..., 2] = 2.0
    means.requires_grad_(True)

    covars = torch.zeros(B, N, 6, device=device)
    covars[..., 0] = 0.01
    covars[..., 3] = 0.01
    covars[..., 5] = 0.01

    viewmats = torch.eye(4, device=device).expand(B, C, 4, 4).clone()
    viewmats.requires_grad_(True)
    Ks = torch.eye(3, device=device).expand(B, C, 3, 3).clone()
    Ks[..., 0, 0] = 20.0
    Ks[..., 1, 1] = 20.0
    Ks[..., 0, 2] = width / 2
    Ks[..., 1, 2] = height / 2

    outputs = fully_fused_projection(
        means,
        covars,
        None,
        None,
        viewmats,
        Ks,
        width,
        height,
        packed=packed,
        camera_model="ortho",
    )
    depths = outputs[6] if packed else outputs[2]
    assert depths.numel() == B * C * N

    v_means, v_viewmats = torch.autograd.grad(depths.sum(), (means, viewmats))
    expected_means = torch.zeros_like(means)
    expected_means[..., 2] = C
    expected_viewmats = torch.zeros_like(viewmats)
    expected_viewmats[..., 2, 2] = 2.0 * N
    expected_viewmats[..., 2, 3] = N

    torch.testing.assert_close(v_means, expected_means, rtol=0, atol=0)
    torch.testing.assert_close(v_viewmats, expected_viewmats, rtol=0, atol=0)


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

    # Per-element band+sparsity check (catches sparsity bugs + sign flips
    # that the previous magnitude-only asserts admitted).
    # interior_rtol = 1.05 x worst observed required rtol per gradient.
    for name, vc, vt, rtol in [
        ("v_viewmats", v_viewmats, _v_viewmats, 1e-5),  # rtol_req=0
        ("v_quats", v_quats, _v_quats, 2.6e-5),  # rtol_req=2.45e-5 (L40S)
        ("v_scales", v_scales, _v_scales, 1e-5),  # rtol_req=0
        ("v_means", v_means, _v_means, 4e-4),  # rtol_req=3.78e-4
    ]:
        assert not (
            torch.isnan(vc).any() or torch.isinf(vc).any()
        ), f"{name}: CUDA produced NaN/Inf"
        assert_grad_sparsity(vc, vt, min_ratio=0.1, msg=f"{name} backward sparsity")
        nz_thresh = vt.abs().max().item() * 1e-5
        assert_close_with_boundary_band(
            vc,
            vt,
            boundary_mask=vt.abs() < nz_thresh,
            interior_atol=nz_thresh,
            interior_rtol=rtol,
            boundary_max_flip_ratio=1e-3,
            boundary_symmetry_tol=0.5,
            flip_predicate=lambda a, e: a.abs() > 10 * nz_thresh,
            msg=f"{name} backward (packed)",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for UT projection"
)
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize("camera_model", ["pinhole", "ortho"])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
@pytest.mark.parametrize(
    "require_all_valid", [True, False], ids=["allvalid", "somevalid"]
)
@pytest.mark.parametrize(
    "rolling_shutter",
    [RollingShutterType.GLOBAL, RollingShutterType.ROLLING_TOP_TO_BOTTOM],
)
@pytest.mark.parametrize("global_z_order", [True, False], ids=["globalz", "distsensor"])
def test_fully_fused_projection_ut(
    test_data,
    camera_model: CameraModel,
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
        "camera_model": camera_model,
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

    # Radii: ceil()'d integer pixel extents.
    # - GLOBAL shutter: inputs are smooth, so only ceil() rounding differs and a
    #   tight per-element check (atol=2) holds.
    # - ROLLING shutter: the iterative refinement amplifies FP32 differences, and
    #   a sub-pixel difference at a ceil() step tips a boundary Gaussian's extent
    #   by a whole integer.  Mirror the means2d/conics rolling-shutter checks: a
    #   tight bulk per-element bound (atol=10) with a small fail-rate cap so
    #   systematic bias still fails, plus an outlier guard so a single large
    #   radius error cannot hide inside the fail-rate budget.
    if rolling_shutter == RollingShutterType.GLOBAL:
        torch.testing.assert_close(
            radii_cuda[sel].float(), radii_torch[sel].float(), rtol=0, atol=2.0
        )
    else:
        _diff_r = (radii_cuda[sel].float() - radii_torch[sel].float()).abs()
        _fail_r = _diff_r > 10.0
        _fr_r = _fail_r.float().mean().item()
        # Bulk bound atol=10; worst observed 1/783256 (~0.00013%) tipping over on
        # L40S (NVRTC).  Cap at 0.01% for GPU/codegen margin.
        assert _fr_r <= 1e-4, (
            f"UT radii (rolling): fail-rate {_fr_r:.4%} > cap 0.01% "
            f"(atol=10, {int(_fail_r.sum().item())}/{_fail_r.numel()})"
        )
        # Outlier guard: admitted outliers stay bounded.  For rolling shutter,
        # near-plane Gaussians can have footprints thousands of pixels wide;
        # the same RS covariance drift that is tolerated by the conics check
        # can then move the ceil()'d radius by many pixels while remaining a
        # small relative error.  Keep an absolute floor for ordinary radii and
        # a relative cap for very large footprints.
        _scale_r = torch.maximum(
            radii_cuda[sel].float().abs(), radii_torch[sel].float().abs()
        )
        _outlier_bound_r = 32.0 + 0.15 * _scale_r
        _outlier_r = _diff_r > _outlier_bound_r
        _rel_r = _diff_r / _scale_r.clamp(min=1.0)
        assert not _outlier_r.any(), (
            f"UT radii (rolling): {int(_outlier_r.sum().item())} elements exceed "
            f"outlier bound (atol=32 + 15% radius); worst diff "
            f"{_diff_r.max().item():.1f}, worst rel {_rel_r.max().item():.2%}"
        )

    # means2d: split by shutter mode.  GLOBAL is smooth in inputs and admits
    # a tight per-element check.  ROLLING does a 10-iter refinement whose
    # convergence basin can shift on a small fraction of
    # Gaussians, producing tail elements with up to ~16% rel-diff.  Use a
    # bounded per-element check with a small fail-rate cap so the tight bulk
    # check still catches systematic bias.  Replaces rtol=0.5, atol=0.05.
    _means2d_atol = 5e-2
    _means2d_rtol = 2e-3
    if rolling_shutter == RollingShutterType.GLOBAL:
        torch.testing.assert_close(
            means2d_cuda[sel],
            means2d_torch[sel],
            rtol=_means2d_rtol,
            atol=_means2d_atol,
        )
    else:
        _diff = (means2d_cuda[sel] - means2d_torch[sel]).abs()
        _bound = _means2d_atol + _means2d_rtol * means2d_torch[sel].abs()
        _fail = _diff > _bound
        _fr = _fail.float().mean().item()
        # On Blackwell with CUDA 12.8, orthographic projection has a slightly
        # larger boundary tail than pinhole projection.  Keep separate caps so
        # that accommodating it does not reduce the pinhole test's regression
        # sensitivity.
        _fail_cap = 1.8e-4 if camera_model == "ortho" else 1.4e-4
        assert _fr <= _fail_cap, (
            f"UT means2d (rolling): fail-rate {_fr:.4%} > cap {_fail_cap:.3%} "
            f"(atol={_means2d_atol:g}, rtol={_means2d_rtol:g}, "
            f"{int(_fail.sum().item())}/{_fail.numel()})"
        )
        # Outlier guard: even admitted outliers must satisfy a per-element
        # bound so a single-element catastrophic bug cannot hide inside the
        # fail-rate budget.  The rolling-shutter floor discontinuity can move
        # a few boundary Gaussians by O(10) pixels, including coordinates near
        # zero where a purely relative bound is too tight, so combine the old
        # relative guard with an absolute pixel cap.
        _outlier_bound = torch.maximum(
            torch.full_like(_diff, 16.0), 0.07 + 0.14 * means2d_torch[sel].abs()
        )
        _outlier_fail = _diff > _outlier_bound
        assert not _outlier_fail.any(), (
            f"UT means2d (rolling): {int(_outlier_fail.sum().item())} elements "
            f"exceed outlier bound (abs=16px or atol=0.07, rtol=0.14); worst diff "
            f"{_diff.max().item():.4e}"
        )

    # depths: High precision expected
    # Depths are computed from camera transformation, less accumulation than means2d
    torch.testing.assert_close(
        depths_cuda[sel],
        depths_torch[sel],
        rtol=1e-6,  # 0.0001% relative tolerance (excellent precision)
        atol=2e-6,  # 2e-6 absolute tolerance (2x safety margin)
    )

    # conics: covariance inverse amplifies small numerical differences.
    # GLOBAL is smooth and admits a tight per-element check.  ROLLING goes
    # through the 10-iter refinement, producing a tail of elements with
    # high rel-diff -- previously rtol=10.0 admitted any error.  Use a
    # bounded per-element check with a fail-rate cap.
    if rolling_shutter == RollingShutterType.GLOBAL:
        torch.testing.assert_close(
            conics_cuda[sel],
            conics_torch[sel],
            rtol=2e-3,
            atol=2e-3,
        )
    else:
        _diff_c = (conics_cuda[sel] - conics_torch[sel]).abs()
        _bound_c = 1e-2 + 1e-2 * conics_torch[sel].abs()
        _fail_c = _diff_c > _bound_c
        _fr_c = _fail_c.float().mean().item()
        # fail_cap = 1.05 x envelope:
        #   RTX PRO 2000  worst fail-rate 0.0161%
        #   RTX PRO 6000  worst fail-rate 0.0191%
        assert _fr_c <= 2.1e-4, (
            f"UT conics (rolling): fail-rate {_fr_c:.4%} > cap 0.021% "
            f"(atol=1e-2, rtol=1e-2, {int(_fail_c.sum().item())}/{_fail_c.numel()})"
        )
        # Outlier guard tightened to 1.05 x worst observed (1.855) -> 2.0.
        _outlier_bound_c = 2.0
        _outlier_fail_c = _diff_c > _outlier_bound_c
        assert not _outlier_fail_c.any(), (
            f"UT conics (rolling): {int(_outlier_fail_c.sum().item())} elements "
            f"exceed outlier bound (atol=2.0); worst diff "
            f"{_diff_c.max().item():.4e}"
        )

    # compensations: Moderate precision
    # Compensations involve sqrt(det_orig/det_blur) which can be sensitive.
    # Small differences in determinants can cause larger differences in sqrt
    # ratio. For GLOBAL shutter the comp computation is smooth in the inputs
    # and admits a tight per-element check.  For rolling-shutter modes, the
    # 10-iter shutter refinement uses floor(image_point.y) inside
    # shutter_relative_frame_time -- a step discontinuity at every integer y.
    # ULP noise in the iteration can flip floor() between two
    # values, jumping `relative_time` by 1/(H-1) and routing the iteration
    # into a different basin of attraction.  We therefore split into:
    #   interior:     Gaussians whose 2D footprint stays clear of any
    #                 integer y -> tight assert_close.
    #   boundary band: Gaussians whose footprint crosses an integer y ->
    #                 budgeted, symmetric flip-rate check.
    if rolling_shutter == RollingShutterType.GLOBAL:
        torch.testing.assert_close(
            comps_cuda[sel], comps_torch[sel], rtol=0.01, atol=0.01
        )
    else:
        # Boundary mask = "any of the 7 UT sigma points might project within
        # ULP of an integer y".  UT sigma-point analytical half-spread =
        # sqrt((n + lambda) * Sigma_i), which for our params (alpha=0.1,
        # kappa=0, n=3) gives lambda = -2.97 and a sqrt(0.03) ~ 0.173 scale
        # factor along each principal axis.  In the projected 2D image we use
        # 1/sqrt(conic[2]) ~ sigma_y as a cheap monotone proxy and apply the
        # same scale; double for safety against per-camera anisotropy
        # amplification through projection.  A Gaussian's sigma-point cluster
        # straddles an integer iff dist_to_int < sigma-point half-spread.
        y_ref = means2d_torch[sel][..., 1]
        sigma_y = (1.0 / conics_torch[sel][..., 2].clamp(min=1e-6)).sqrt()
        # Boundary half-spread = K * sigma_y. K must cover the worst observed
        # sigma-point projection drift:
        #   analytical static half-spread = 0.35 * sigma  (UT, alpha=0.1, n=3)
        #   RTX PRO 2000  10-iter drift cap ~ 0.6 * sigma  (1.7x)
        #   RTX PRO 6000  10-iter drift cap ~ 0.675 * sigma (1.93x)
        #   L40S          one ortho T2B point at 0.756 * sigma
        # K = 0.8 leaves headroom while preserving non-empty interior coverage.
        sp_half_spread = 0.8 * sigma_y
        dist_to_int = (y_ref - y_ref.round()).abs()
        boundary_mask = dist_to_int < sp_half_spread

        # Calibration trace -- envelope x 1.05:
        #   - interior assert atol=7e-3, rtol=0.01:
        #       RTX PRO 2000  worst <7e-3  (passes)
        #       RTX PRO 6000  worst <7e-3  (passes after K=0.8)
        #   - in-band flips at flip_predicate (a-e).abs() > 7e-3:
        #       RTX PRO 2000  50/188261 (0.0266%)   globalz/distsensor allvalid
        #                     58/188485 (0.0308%)   globalz/distsensor somevalid
        #                     historic single-Gaussian outlier diff=0.398 included
        #       RTX PRO 6000  similar magnitudes, bounded by the 4.5e-4 cap below.
        #   - asymmetry not enforced: too few flips for stable bias estimate.
        # See plan: ~/.claude/plans/how-to-devise-tests-hidden-fox.md
        # CUDA does not expose per-sigma-point projections, so a true cross
        # predicate cannot be written here; the boundary mask is a geometric
        # proxy.
        assert_close_with_boundary_band(
            comps_cuda[sel],
            comps_torch[sel],
            boundary_mask=boundary_mask,
            interior_atol=7e-3,
            interior_rtol=0.01,
            boundary_max_flip_ratio=4.5e-4,  # 1.05 x observed worst (4.23e-4)
            boundary_symmetry_tol=1.0,  # disabled: too few flips meaningful
            flip_predicate=lambda a, e: (a - e).abs() > 7e-3,
            boundary_cross_predicate=None,
            msg="cluster A: rolling-shutter floor() discontinuity",
        )
        # Outlier guard: even admitted in-band flips must satisfy a loose
        # absolute bound, so a catastrophic single-Gaussian bug cannot hide
        # inside the boundary flip budget.  Tightened to 1.05 x worst
        # observed in-band diff ~0.428 -> atol=0.46.
        _diff_a = (comps_cuda[sel] - comps_torch[sel]).abs()
        _outlier_a = (_diff_a > 0.46) & boundary_mask
        assert not _outlier_a.any(), (
            f"cluster A: {int(_outlier_a.sum().item())} in-band elements "
            f"exceed outlier bound atol=0.46; worst diff "
            f"{_diff_a.max().item():.4e}"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for UT projection"
)
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_fully_fused_projection_ut_ortho_rejects_radial_coeffs(test_data):
    from gsplat.cuda._wrapper import fully_fused_projection_with_ut

    N = test_data["means"].shape[-2]
    C = test_data["viewmats"].shape[-3]
    radial_coeffs = torch.zeros((C, 6), device=device, dtype=torch.float32)
    match = (
        "ortho camera model does not support radial_coeffs, "
        "tangential_coeffs, or thin_prism_coeffs parameters"
    )

    with pytest.raises(RuntimeError, match=match):
        fully_fused_projection_with_ut(
            test_data["means"],
            test_data["quats"],
            test_data["scales"],
            test_data["opacities"],
            test_data["viewmats"],
            test_data["Ks"],
            test_data["width"],
            test_data["height"],
            camera_model="ortho",
            radial_coeffs=radial_coeffs,
        )

    colors = torch.zeros((N, 3), device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match=match):
        gsplat.rasterization(
            test_data["means"],
            test_data["quats"],
            test_data["scales"],
            test_data["opacities"],
            colors,
            test_data["viewmats"],
            test_data["Ks"],
            test_data["width"],
            test_data["height"],
            packed=False,
            with_ut=True,
            camera_model="ortho",
            radial_coeffs=radial_coeffs,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT/Lidar support isn't built in")
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_isect(test_data, dtype: torch.dtype, batch_dims: Tuple[int, ...]):
    from gsplat.cuda._torch_impl import _isect_offset_encode, _isect_tiles
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    torch.manual_seed(42)

    B = math.prod(batch_dims)
    C, N = 3, 1000
    I = B * C
    width, height = 40, 60

    # Cover float64 as well as float32: intersect_tile_kernel is instantiated
    # for scalar_t=double (dispatch keys on means2d), and its 32-bit depth sort
    # key must narrow to float32 to stay a monotonic ordering -- a bare 32-bit
    # reinterpret of a double reads only half its bits and scrambles the
    # within-tile front-to-back order. The Python reference narrows the key to
    # float32, so the float64 CUDA path must agree with it.
    test_data = {
        "means2d": torch.randn(C, N, 2, device=device, dtype=dtype) * width,
        "radii": torch.randint(0, width, (C, N, 2), device=device, dtype=torch.int32),
        "depths": torch.rand(C, N, device=device, dtype=dtype),
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
    from tests.core.test_cameras import parse_lidar_camera

    torch.manual_seed(42)

    lidar_params, angles_to_columns_map, tiling = parse_lidar_camera(
        lidar_model, batch_dims, 0, 0, device=device
    )
    lidar = gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
        lidar_params, angles_to_columns_map, tiling
    )

    C, N = 3, 1000  # cameras and gaussians

    test_data = {
        "means2d": torch.randn(C, N, 2, device=device)
        * torch.tensor([2 * math.pi, math.pi], device=device),
        # .x=azimuth, .y=elevation
        "radii": torch.randn(C, N, 2, device=device).abs().clamp(max=1)
        * torch.tensor([math.pi, lidar.fov_vert_rad.span / 2], device=device),
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
        (cdf_resolution_elevation, cdf_resolution_azimuth),
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
            middle_el_idx = dense_mask.shape[0] // 2

            dense_mask[middle_el_idx, az_idx] = 1

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
    base_params = gsplat.RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=row_elevations_rad,
        column_azimuths_rad=column_azimuths_rad,
        row_azimuth_offsets_rad=row_azimuth_offsets_rad,
        spinning_direction=spinning_direction,
        spinning_frequency_hz=10.0,
    )
    lidar = gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
        base_params,
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
            [[[mean_az_pix, 0.0]]],
            dtype=torch.float32,
            device=device,
        ),
        radii=torch.tensor(
            [[[radius_az_pix, 1]]],
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
    gauss_mean_pix = gaussian_param.means2d[0, 0, 0].item()
    gauss_radius_pix = gaussian_param.radii[0, 0, 0].item()
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
@pytest.mark.parametrize("channels", [3, 32])
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

    # Identify intersecting tiles.
    tile_size = 16
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

    # CUDA forward
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
    v_render_colors = torch.randn_like(render_colors)
    v_render_alphas = torch.randn_like(render_alphas)

    # CUDA backward — `torch.autograd.grad` defaults to retain_graph=False,
    # so the saved-for-backward tensors of the CUDA forward graph are released
    # here. The reference forward below builds its own independent graph.
    v_means2d, v_conics, v_colors, v_opacities, v_backgrounds = torch.autograd.grad(
        (render_colors * v_render_colors).sum()
        + (render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, opacities, backgrounds),
    )

    # Reference forward
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

    # Compare values, then drop the CUDA-pass renders before the reference
    # backward runs to keep the test's peak memory bounded.
    torch.testing.assert_close(render_colors, _render_colors)
    torch.testing.assert_close(render_alphas, _render_alphas)
    del render_colors, render_alphas

    # Reference backward
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
    # Per-element band+sparsity check on rasterization backward.
    # - interior_atol absorbs absolute FP cancellation noise at small
    #   magnitudes; interior_rtol catches systematic bias at large.
    # - interior_rtol = 1.05 x envelope across calibrated GPUs (RTX PRO 2000 / 6000).
    # - v_means2d atol covers channel-scaled alpha-compositing accumulation.
    for name, vc, vt, rtol, atol in [
        (
            "v_means2d",
            v_means2d,
            _v_means2d,
            2.5e-4,
            1.6e-3,
        ),
        ("v_conics", v_conics, _v_conics, 1e-5, 1e-3),
        ("v_colors", v_colors, _v_colors, 1e-5, 1e-3),
        ("v_opacities", v_opacities, _v_opacities, 1e-5, 2e-3),
        ("v_backgrounds", v_backgrounds, _v_backgrounds, 1e-5, 1e-3),
    ]:
        assert not (
            torch.isnan(vc).any() or torch.isinf(vc).any()
        ), f"{name}: CUDA produced NaN/Inf"
        assert_grad_sparsity(vc, vt, min_ratio=0.1, msg=f"{name} sparsity")
        nz_thresh = vt.abs().max().item() * 1e-5
        assert_close_with_boundary_band(
            vc,
            vt,
            boundary_mask=vt.abs() < nz_thresh,
            interior_atol=atol,
            interior_rtol=rtol,
            boundary_max_flip_ratio=1e-3,
            boundary_symmetry_tol=0.5,
            flip_predicate=lambda a, e, _t=nz_thresh: a.abs() > 10 * _t,
            msg=f"{name} backward (rasterize)",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("tile_size", [4, 16])
def test_rasterize_num_contributing_gaussians(tile_size: int):
    from gsplat.cuda._wrapper import (
        isect_offset_encode,
        isect_tiles,
        rasterize_num_contributing_gaussians,
        rasterize_to_indices_in_range,
        rasterize_to_pixels,
    )

    width, height = 18, 17
    means2d = torch.tensor(
        [[[4.5, 4.5], [7.5, 4.5], [4.5, 7.5], [12.5, 12.5]]],
        device=device,
    )
    conics = torch.tensor(
        [[[0.08, 0.0, 0.08], [0.10, 0.0, 0.10], [0.12, 0.0, 0.12], [0.07, 0.0, 0.07]]],
        device=device,
    )
    opacities = torch.tensor([[0.45, 0.50, 0.55, 0.35]], device=device)
    colors = torch.zeros((*opacities.shape, 3), device=device)
    radii = torch.full((*opacities.shape, 2), 8, dtype=torch.int32, device=device)
    depths = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device)

    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    tile_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)

    num_contributing, alphas = rasterize_num_contributing_gaussians(
        means2d,
        conics,
        opacities,
        tile_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
    )

    _, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        tile_offsets,
        flatten_ids,
    )
    torch.testing.assert_close(alphas, render_alphas.squeeze(-1))

    gauss_ids, pixel_ids, image_ids = rasterize_to_indices_in_range(
        0,
        2**30,
        torch.ones_like(alphas),
        means2d,
        conics,
        opacities,
        width,
        height,
        tile_size,
        tile_offsets,
        flatten_ids,
    )
    assert gauss_ids.numel() > 0
    expected_counts = torch.zeros(
        num_contributing.numel(), dtype=torch.int32, device=device
    )
    linear_ids = image_ids * (height * width) + pixel_ids
    expected_counts.index_add_(
        0, linear_ids, torch.ones_like(linear_ids, dtype=torch.int32)
    )
    expected_counts = expected_counts.reshape_as(num_contributing)
    torch.testing.assert_close(num_contributing, expected_counts)


def _assert_contributor_output_invariants(
    ids: torch.Tensor,
    weights: torch.Tensor,
    max_gaussian_id: int,
    expected_counts: torch.Tensor,
):
    valid = ids >= 0

    assert ids.dtype == torch.int32
    assert weights.dtype == torch.float32
    assert weights.shape == ids.shape
    torch.testing.assert_close(valid.sum(dim=-1).to(torch.int32), expected_counts)

    if ids.shape[-1] > 1:
        invalid_before_valid = (~valid[..., :-1]) & valid[..., 1:]
        sorted_ids = ids.sort(dim=-1).values
        duplicate_ids = (sorted_ids[..., 1:] == sorted_ids[..., :-1]) & (
            sorted_ids[..., 1:] >= 0
        )
        assert not bool(invalid_before_valid.any().item())
        assert not bool(duplicate_ids.any().item())

    if bool(valid.any().item()):
        valid_ids = ids[valid]
        assert int(valid_ids.min().item()) >= 0
        assert int(valid_ids.max().item()) < max_gaussian_id
        assert bool((weights[valid] > 0).all().item())

    torch.testing.assert_close(weights[~valid], torch.zeros_like(weights[~valid]))
    assert bool(torch.isfinite(weights).all().item())
    assert bool((weights >= 0).all().item())
    assert bool((weights <= 1).all().item())


def _assert_contributor_ids_are_depth_ordered(
    ids: torch.Tensor,
    depths: torch.Tensor,
    packed: bool,
):
    if ids.shape[-1] <= 1:
        return

    clamped_ids = ids.clamp_min(0).to(torch.long)
    if packed:
        sample_depths = depths.reshape(-1)[clamped_ids]
    else:
        image_count = math.prod(ids.shape[:-3])
        num_gaussians = depths.shape[-1]
        ids_flat = clamped_ids.reshape(image_count, -1, ids.shape[-1])
        depths_flat = depths.reshape(image_count, num_gaussians)
        sample_depths = torch.gather(
            depths_flat[:, None, :].expand(-1, ids_flat.shape[1], -1),
            2,
            ids_flat,
        ).reshape_as(ids)

    valid = ids >= 0
    adjacent_valid = valid[..., :-1] & valid[..., 1:]
    ordered = sample_depths[..., :-1] <= sample_depths[..., 1:]
    assert bool((ordered | ~adjacent_valid).all().item())


def _assert_top_contributors_are_largest_weights(
    top_ids: torch.Tensor,
    top_weights: torch.Tensor,
    all_ids: torch.Tensor,
    all_weights: torch.Tensor,
):
    num_depth_samples = top_ids.shape[-1]
    if all_weights.shape[-1] < num_depth_samples:
        pad_shape = all_weights.shape[:-1] + (
            num_depth_samples - all_weights.shape[-1],
        )
        all_weights = torch.cat(
            [
                all_weights,
                torch.zeros(
                    pad_shape,
                    dtype=all_weights.dtype,
                    device=all_weights.device,
                ),
            ],
            dim=-1,
        )

    expected_weights = torch.topk(all_weights, num_depth_samples, dim=-1).values
    actual_weights = top_weights.sort(dim=-1, descending=True).values
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-5, atol=2e-6)

    for k in range(num_depth_samples):
        valid = top_ids[..., k] >= 0
        id_match = all_ids == top_ids[..., k, None]
        weight_match = torch.isclose(
            all_weights[..., : all_ids.shape[-1]],
            top_weights[..., k, None],
            rtol=1e-5,
            atol=2e-6,
        )
        present = (id_match & weight_match).any(dim=-1)
        assert bool((present | ~valid).all().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("tile_size", [4, 16])
@pytest.mark.parametrize("packed", [False, True])
def test_rasterize_contributing_gaussian_ids(tile_size: int, packed: bool):
    from gsplat.cuda._wrapper import (
        isect_offset_encode,
        isect_tiles,
        rasterize_contributing_gaussian_ids,
        rasterize_num_contributing_gaussians,
    )

    width, height = 18, 17
    means2d = torch.tensor(
        [
            [[4.5, 4.5], [7.5, 4.5], [4.5, 7.5], [12.5, 12.5], [8.0, 8.0]],
            [[5.5, 5.5], [9.5, 5.5], [5.5, 9.5], [13.5, 13.5], [9.0, 9.0]],
        ],
        device=device,
    )
    conics = torch.tensor(
        [
            [
                [0.08, 0.00, 0.08],
                [0.10, 0.02, 0.10],
                [0.12, -0.01, 0.12],
                [0.07, 0.00, 0.07],
                [0.05, 0.01, 0.09],
            ],
            [
                [0.09, 0.01, 0.08],
                [0.11, -0.02, 0.10],
                [0.12, 0.00, 0.11],
                [0.08, 0.01, 0.08],
                [0.06, -0.01, 0.08],
            ],
        ],
        device=device,
    )
    opacities = torch.tensor(
        [[0.45, 0.70, 0.55, 0.35, 0.60], [0.50, 0.65, 0.40, 0.75, 0.58]],
        device=device,
    )
    radii = torch.full((*opacities.shape, 2), 8, dtype=torch.int32, device=device)
    depths = torch.tensor(
        [[1.0, 2.5, 2.0, 4.0, 3.0], [0.8, 2.2, 1.8, 4.4, 3.1]],
        device=device,
    )

    I, N = opacities.shape
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    if packed:
        image_ids = torch.arange(I, device=device, dtype=torch.long).repeat_interleave(
            N
        )
        gaussian_ids = torch.arange(N, device=device, dtype=torch.long).repeat(I)
        means2d = means2d.reshape(I * N, 2)
        conics = conics.reshape(I * N, 3)
        opacities = opacities.reshape(I * N)
        radii = radii.reshape(I * N, 2)
        depths = depths.reshape(I * N)
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
            packed=True,
            n_images=I,
            image_ids=image_ids,
            gaussian_ids=gaussian_ids,
        )
    else:
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
        )
    tile_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)

    num_contributing, _ = rasterize_num_contributing_gaussians(
        means2d,
        conics,
        opacities,
        tile_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
    )
    actual_ids, actual_weights = rasterize_contributing_gaussian_ids(
        means2d,
        conics,
        opacities,
        tile_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
        num_contributing,
    )
    max_num_contributing = int(num_contributing.max().item())
    max_gaussian_id = means2d.shape[0] if packed else N

    assert actual_ids.shape == num_contributing.shape + (max_num_contributing,)
    assert bool((actual_ids >= 0).any().item())
    _assert_contributor_output_invariants(
        actual_ids,
        actual_weights,
        max_gaussian_id,
        num_contributing,
    )
    _assert_contributor_ids_are_depth_ordered(actual_ids, depths, packed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("tile_size", [4, 16])
def test_rasterize_contributing_gaussian_ids_early_termination(tile_size: int):
    from gsplat.cuda._wrapper import (
        isect_offset_encode,
        isect_tiles,
        rasterize_contributing_gaussian_ids,
        rasterize_num_contributing_gaussians,
    )

    width, height = 9, 9
    N = 8
    means2d = torch.full((1, N, 2), 4.5, device=device)
    conics = torch.tensor([0.04, 0.0, 0.04], device=device).expand(1, N, 3)
    opacities = torch.full((1, N), 0.98, device=device)
    radii = torch.full((1, N, 2), 8, dtype=torch.int32, device=device)
    depths = torch.arange(1, N + 1, device=device, dtype=torch.float32).reshape(1, N)

    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    tile_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    num_contributing, _ = rasterize_num_contributing_gaussians(
        means2d,
        conics,
        opacities,
        tile_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
    )
    actual_ids, actual_weights = rasterize_contributing_gaussian_ids(
        means2d,
        conics,
        opacities,
        tile_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
        num_contributing,
    )
    max_num_contributing = int(num_contributing.max().item())

    assert actual_ids.shape == num_contributing.shape + (max_num_contributing,)
    _assert_contributor_output_invariants(
        actual_ids, actual_weights, N, num_contributing
    )
    _assert_contributor_ids_are_depth_ordered(actual_ids, depths, packed=False)
    assert int(num_contributing[0, 4, 4].item()) < N


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("tile_size", [4, 16])
def test_rasterize_contributing_gaussian_ids_empty_counts(tile_size: int):
    from gsplat.cuda._wrapper import rasterize_contributing_gaussian_ids

    width, height = 7, 6
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    means2d = torch.empty((1, 0, 2), device=device)
    conics = torch.empty((1, 0, 3), device=device)
    opacities = torch.empty((1, 0), device=device)
    tile_offsets = torch.zeros(
        (1, tile_height, tile_width), dtype=torch.int32, device=device
    )
    flatten_ids = torch.empty((0,), dtype=torch.int32, device=device)
    num_contributing = torch.zeros((1, height, width), dtype=torch.int32, device=device)

    actual_ids, actual_weights = rasterize_contributing_gaussian_ids(
        means2d,
        conics,
        opacities,
        tile_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
        num_contributing,
    )

    assert actual_ids.shape == (1, height, width, 0)
    assert actual_weights.shape == actual_ids.shape
    assert actual_ids.dtype == torch.int32
    assert actual_weights.dtype == torch.float32
    _assert_contributor_output_invariants(
        actual_ids, actual_weights, 0, num_contributing
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("tile_size", [4, 16])
@pytest.mark.parametrize("num_depth_samples", [1, 3])
@pytest.mark.parametrize("packed", [False, True])
def test_rasterize_top_contributing_gaussian_ids(
    tile_size: int, num_depth_samples: int, packed: bool
):
    from gsplat.cuda._wrapper import (
        isect_offset_encode,
        isect_tiles,
        rasterize_contributing_gaussian_ids,
        rasterize_num_contributing_gaussians,
        rasterize_top_contributing_gaussian_ids,
    )

    width, height = 18, 17
    means2d = torch.tensor(
        [
            [[4.5, 4.5], [7.5, 4.5], [4.5, 7.5], [12.5, 12.5], [8.0, 8.0]],
            [[5.5, 5.5], [9.5, 5.5], [5.5, 9.5], [13.5, 13.5], [9.0, 9.0]],
        ],
        device=device,
    )
    conics = torch.tensor(
        [
            [
                [0.08, 0.00, 0.08],
                [0.10, 0.02, 0.10],
                [0.12, -0.01, 0.12],
                [0.07, 0.00, 0.07],
                [0.05, 0.01, 0.09],
            ],
            [
                [0.09, 0.01, 0.08],
                [0.11, -0.02, 0.10],
                [0.12, 0.00, 0.11],
                [0.08, 0.01, 0.08],
                [0.06, -0.01, 0.08],
            ],
        ],
        device=device,
    )
    opacities = torch.tensor(
        [[0.45, 0.70, 0.55, 0.35, 0.60], [0.50, 0.65, 0.40, 0.75, 0.58]],
        device=device,
    )
    radii = torch.full((*opacities.shape, 2), 8, dtype=torch.int32, device=device)
    depths = torch.tensor(
        [[1.0, 2.5, 2.0, 4.0, 3.0], [0.8, 2.2, 1.8, 4.4, 3.1]],
        device=device,
    )

    I, N = opacities.shape
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    if packed:
        image_ids = torch.arange(I, device=device, dtype=torch.long).repeat_interleave(
            N
        )
        gaussian_ids = torch.arange(N, device=device, dtype=torch.long).repeat(I)
        means2d = means2d.reshape(I * N, 2)
        conics = conics.reshape(I * N, 3)
        opacities = opacities.reshape(I * N)
        radii = radii.reshape(I * N, 2)
        depths = depths.reshape(I * N)
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
            packed=True,
            n_images=I,
            image_ids=image_ids,
            gaussian_ids=gaussian_ids,
        )
    else:
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
        )
    tile_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)

    actual_ids, actual_weights = rasterize_top_contributing_gaussian_ids(
        means2d,
        conics,
        opacities,
        tile_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
        num_depth_samples,
    )
    num_contributing, _ = rasterize_num_contributing_gaussians(
        means2d,
        conics,
        opacities,
        tile_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
    )
    all_ids, all_weights = rasterize_contributing_gaussian_ids(
        means2d,
        conics,
        opacities,
        tile_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
        num_contributing,
    )
    expected_counts = torch.minimum(
        num_contributing,
        torch.full_like(num_contributing, num_depth_samples),
    )
    max_gaussian_id = means2d.shape[0] if packed else N

    assert actual_ids.shape == num_contributing.shape + (num_depth_samples,)
    assert bool((actual_ids >= 0).any().item())
    _assert_contributor_output_invariants(
        all_ids,
        all_weights,
        max_gaussian_id,
        num_contributing,
    )
    _assert_contributor_output_invariants(
        actual_ids,
        actual_weights,
        max_gaussian_id,
        expected_counts,
    )
    _assert_contributor_ids_are_depth_ordered(all_ids, depths, packed)
    _assert_contributor_ids_are_depth_ordered(actual_ids, depths, packed)
    _assert_top_contributors_are_largest_weights(
        actual_ids,
        actual_weights,
        all_ids,
        all_weights,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("tile_size", [4, 16])
def test_rasterize_top_contributing_gaussian_ids_early_termination(tile_size: int):
    from gsplat.cuda._wrapper import (
        isect_offset_encode,
        isect_tiles,
        rasterize_contributing_gaussian_ids,
        rasterize_num_contributing_gaussians,
        rasterize_top_contributing_gaussian_ids,
    )

    width, height = 9, 9
    num_depth_samples = 4
    N = 8
    means2d = torch.full((1, N, 2), 4.5, device=device)
    conics = torch.tensor([0.04, 0.0, 0.04], device=device).expand(1, N, 3)
    opacities = torch.full((1, N), 0.98, device=device)
    radii = torch.full((1, N, 2), 8, dtype=torch.int32, device=device)
    depths = torch.arange(1, N + 1, device=device, dtype=torch.float32).reshape(1, N)

    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    tile_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)

    actual_ids, actual_weights = rasterize_top_contributing_gaussian_ids(
        means2d,
        conics,
        opacities,
        tile_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
        num_depth_samples,
    )
    num_contributing, _ = rasterize_num_contributing_gaussians(
        means2d,
        conics,
        opacities,
        tile_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
    )
    all_ids, all_weights = rasterize_contributing_gaussian_ids(
        means2d,
        conics,
        opacities,
        tile_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
        num_contributing,
    )
    expected_counts = torch.minimum(
        num_contributing,
        torch.full_like(num_contributing, num_depth_samples),
    )

    assert actual_ids.shape == num_contributing.shape + (num_depth_samples,)
    _assert_contributor_output_invariants(all_ids, all_weights, N, num_contributing)
    _assert_contributor_output_invariants(
        actual_ids,
        actual_weights,
        N,
        expected_counts,
    )
    _assert_contributor_ids_are_depth_ordered(all_ids, depths, packed=False)
    _assert_contributor_ids_are_depth_ordered(actual_ids, depths, packed=False)
    _assert_top_contributors_are_largest_weights(
        actual_ids,
        actual_weights,
        all_ids,
        all_weights,
    )
    assert int((actual_ids[0, 4, 4] >= 0).sum().item()) < num_depth_samples


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
@pytest.mark.parametrize("tile_size", [8, 16], ids=["tile8", "tile16"])
def test_rasterize_to_pixels_hit_distance_principal_axis(
    means_list, quats_choice, scales_list, pixel_dx, pixel_dy, tile_size
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
        return_last_ids=False,
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
    "channels,batch_dims,rs_type,use_hit_distance,use_rays,return_normals,camera_model,tile_size",
    [
        pytest.param(
            *params,
            "pinhole",
            tile_size,
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
        # Camera path: cover both 3DGUT compile-time tile_size dispatches.
        # tile_size=8 (CTA=32, PPT=2) is the compact-CTA path used at sub-1080p;
        # tile_size=16 (CTA=256, PPT=1) is the thread-per-pixel path used at
        # 1080p+. Both are expected to produce identical RGB/alphas vs the
        # Python reference within tolerance.
        for tile_size in (8, 16)
    ]
    + [
        # Lidar: cover both tile_size dispatches. The auto-fallback in
        # rendering.py always picks 8 for production lidars (n_rows < 1080,
        # max_pts_per_tile = 8*8 = 64 in parse_lidar_camera), so tile=16 here
        # underutilizes the 256-thread CTA. We test it anyway to verify the
        # <CDIM,16,256> kernel instantiation is correct on lidar inputs and
        # to catch any TILE_SIZE-dependent assumption that silently bakes in 8.
        pytest.param(
            3, (), RollingShutterType.GLOBAL, True, True, False, "lidar", tile_size
        )
        for tile_size in (8, 16)
    ]
    + [
        pytest.param(
            3, (), RollingShutterType.GLOBAL, True, False, False, "lidar", tile_size
        )
        for tile_size in (8, 16)
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
    camera_model: CameraModel,
    tile_size: int,
):
    from gsplat.cuda._torch_impl_eval3d import _rasterize_to_pixels_eval3d
    from gsplat.cuda._wrapper import (
        fully_fused_projection_with_ut,
        isect_offset_encode,
        isect_tiles,
        isect_tiles_lidar,
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

    # Setup lidar
    if camera_model == "lidar":
        from tests.core.test_cameras import parse_lidar_camera

        # This test consumes randomness before lidar setup, so fix the lidar
        # param seed explicitly to keep the preprocessing cache reusable.
        lidar_params, angles_to_columns_map, tiling = parse_lidar_camera(
            "at128", batch_dims, 0, 0, device=device, seed=42
        )
        lidar_coeffs = gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
            lidar_params, angles_to_columns_map, tiling
        )
        width = lidar_coeffs.n_columns
        height = lidar_coeffs.n_rows
        focal = float(width)
        Ks = torch.tensor(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            device=device,
        ).expand(batch_dims + (C, -1, -1))
    else:
        lidar_coeffs = None

    # Create viewmats_rs for rolling shutter testing
    if rs_type != RollingShutterType.GLOBAL:
        # Simulate camera motion with small perturbation
        viewmats_rs = viewmats.clone()
        # Add small translation (5% of scene scale)
        viewmats_rs[..., :3, 3] += torch.randn_like(viewmats[..., :3, 3]) * 0.05
    else:
        viewmats_rs = None

    if use_rays:
        from gsplat.cuda._torch_cameras import _BaseCameraModel
        from gsplat.cuda._torch_impl_eval3d import _generate_rays

        camera = _BaseCameraModel.create(
            width=width,
            height=height,
            camera_model=camera_model,
            focal_lengths=Ks.reshape(I, 3, 3)[:, [0, 1], [0, 1]],
            principal_points=Ks.reshape(I, 3, 3)[:, [0, 1], [2, 2]],
            rs_type=rs_type,
            lidar_coeffs=lidar_coeffs,
        )
        rays = (
            _generate_rays(
                camera,
                width,
                height,
                viewmats.reshape(I, 4, 4),
                viewmats_rs.reshape(I, 4, 4) if viewmats_rs is not None else None,
            )
            .detach()
            .reshape(*batch_dims, C, -1, 6)
        )
    else:
        rays = None

    # Project Gaussians to 2D for tile intersections
    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        means,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        camera_model=camera_model,
        lidar_coeffs=lidar_coeffs,
    )
    opacities_broadcast = torch.broadcast_to(
        opacities[..., None, :], batch_dims + (C, N)
    )

    # Identify intersecting tiles. tile_size flows in from the parametrize and
    # selects which 3DGUT kernel instantiation is dispatched (<TILE,CTA> in
    # {<8,32>, <16,256>}).
    if camera_model == "lidar":
        tile_width = lidar_coeffs.tiling.n_bins_azimuth
        tile_height = lidar_coeffs.tiling.n_bins_elevation
        _tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_lidar(
            lidar_coeffs,
            means2d,
            radii,
            depths,
        )
    else:
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
        camera_model=camera_model,
        lidar_coeffs=lidar_coeffs,
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
        lidar_coeffs=lidar_coeffs,
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

    # On count-matched pixels, the last contributing gaussian's flatten_idx
    # must match exactly. An off-by-one, such as over-counting the saturating
    # gaussian past TRANSMITTANCE_THRESHOLD, can shift last_ids by 1 while
    # sample_counts still coincidentally match.
    last_ids_match = render_last_ids[count_match] == _render_last_ids[count_match]
    assert last_ids_match.all(), (
        f"last_ids diverge on {(~last_ids_match).sum().item()} of "
        f"{count_match.sum().item()} count-matched pixels"
    )

    count_match = count_match.unsqueeze(-1).float()
    count_mismatch = count_mismatch.unsqueeze(-1).float()
    vis_mismatch = vis_mismatch.unsqueeze(-1).float()

    assert count_match.sum() > 0

    # Compare alphas for each group.
    # - lidar uses element_to_image_point CUDA / _generate_rays ref, which
    #   diverge by ~1 pixel at tile boundaries, so its envelope is wider.
    # - Non-lidar: tight on both RTX PRO 2000 and RTX PRO 6000; lidar RTX PRO 6000 worst:
    #   rtol=8.4e-3, atol=2.27e-3 (count_match alpha diff at lidar tile edge).
    count_match_rtol = 9e-3 if camera_model == "lidar" else 1e-3
    count_match_atol = 2.5e-3 if camera_model == "lidar" else 2e-3
    torch.testing.assert_close(
        render_alphas * count_match,
        _render_alphas * count_match,
        rtol=count_match_rtol,
        atol=count_match_atol,
    )
    # For lidar, the CUDA kernel generates rays via element_to_image_point
    # while the ref uses _generate_rays, producing ~1 pixel mismatch at tile
    # boundaries (observed: 0.00392 at (1, 0, 566, 0)).
    vis_mismatch_atol = 1e-2 if camera_model == "lidar" else ALPHA_THRESHOLD + 1e-5
    torch.testing.assert_close(
        render_alphas * vis_mismatch,
        _render_alphas * vis_mismatch,
        rtol=0,
        atol=vis_mismatch_atol,
    )
    # For lidar, bump tolerance: CUDA's element_to_image_point and the ref's
    # _generate_image_points may produce slightly different scaled-angle values
    # for the same element, causing ~1 pixel count mismatch at tile boundaries.
    count_mismatch_atol = 1e-2 if camera_model == "lidar" else 5e-3
    torch.testing.assert_close(
        render_alphas * count_mismatch,
        _render_alphas * count_mismatch,
        rtol=0,
        atol=count_mismatch_atol,
    )

    # Compare colors for each group (expand masks to [batch, C, H, W, 3])
    count_match = count_match.expand_as(render_colors)
    vis_mismatch = vis_mismatch.expand_as(render_colors)
    count_mismatch = count_mismatch.expand_as(render_colors)

    # For lidar use_rays=False, the CUDA kernel and Python ref generate rays
    # independently (CUDA via element_to_image_point, ref via _generate_rays).
    # Tiny FP differences in the scaled-angle computation cause ~1-2 pixels at
    # tile boundaries to accumulate slightly different colors/alphas.
    # For lidar use_rays=False, use 10x tolerance to accommodate the ~1-2
    # boundary pixels where CUDA and ref rays differ.
    _lidar_tol = 10.0 if (camera_model == "lidar" and not use_rays) else 1.0

    torch.testing.assert_close(
        render_colors * count_match,
        _render_colors * count_match,
        rtol=1e-3 * _lidar_tol,
        atol=1e-3 * _lidar_tol,
    )
    # Bumped tolerance due to release mode optimizations. In debug mode it's ALPHA_THRESHOLD+1e-5.
    torch.testing.assert_close(
        render_colors * vis_mismatch,
        _render_colors * vis_mismatch,
        rtol=0,
        atol=(ALPHA_THRESHOLD + 5e-3) * _lidar_tol,
    )
    torch.testing.assert_close(
        render_colors * count_mismatch,
        _render_colors * count_mismatch,
        rtol=0,
        atol=3e-3 * _lidar_tol,
    )

    # Compare normals if computed
    if return_normals:
        assert (
            render_normals is not None
        ), "CUDA render_normals should not be None when return_normals=True"
        assert (
            _render_normals is not None
        ), "PyTorch render_normals should not be None when return_normals=True"

        # Old tolerance: atol=6e-5. Bumped to 2.5e-4 after switching ray
        # generation from the CUDA camera model to _generate_rays (Python ref).
        # The rays match to ~4e-7, but the slightly different values shift which
        # pixel has the worst-case CUDA-vs-ref normal error.
        # Old worst case (CUDA rays): 1.20e-4 at a 1-sample pixel.
        # New worst case (ref rays):  2.30e-4 at a 3-sample pixel.
        torch.testing.assert_close(
            render_normals, _render_normals, rtol=3e-4, atol=2.5e-4
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

    # Background gradients are a direct reduction of per-pixel transmittance.
    # Compare only the structurally matched pixels; vis/count mismatches are
    # already validated separately above with looser forward tolerances.
    v_backgrounds_struct = (
        v_render_colors * (1.0 - render_alphas).float() * count_match
    ).sum(dim=(-3, -2))
    _v_backgrounds_struct = (
        v_render_colors * (1.0 - _render_alphas).float() * count_match
    ).sum(dim=(-3, -2))
    backgrounds_mask = get_inlier_abserror_mask(
        v_backgrounds_struct, _v_backgrounds_struct, quantile=0.99
    )
    assert backgrounds_mask.sum() > 0

    # Per-Gaussian sparsity + sign check.  The per-element bound check
    # below absorbs absolute-bias and FP noise via atol+rtol+fail_cap,
    # but two bug classes can still slip through:
    #   1. Sparsity: CUDA zeros (or explodes) one Gaussian's gradient.  The
    #      cap admits this if the Gaussian touches few elements.  Catch it
    #      by aggregating |grad| per Gaussian and bounding the CUDA/ref
    #      magnitude ratio.
    #   2. Sign flip: CUDA returns -ref on a gradient.  The bound admits
    #      this on small-|e| elements (2|e| < atol).  Catch it by requiring
    #      the per-Gaussian dot product (cuda . ref) > 0 for any Gaussian
    #      with significant gradient magnitude.
    # Knobs:
    #   - magnitude floor 1% of max(ref) admits the small-magnitude tail
    #     where rasterizer FP noise dominates (worst observed ratio
    #     1.6e-2 at this floor).
    #   - min_ratio 5e-3 catches zero/200x-explosion sparsity bugs.
    vis_g = visible_mask[0, :, 0]  # [N]

    def _per_gaussian_mag(t):
        m = t.abs().sum(dim=-1)
        while m.ndim > 1:
            m = m.sum(dim=0)
        return m  # [N]

    def _per_gaussian_dot(a, b):
        s = (a * b).sum(dim=-1)
        while s.ndim > 1:
            s = s.sum(dim=0)
        return s  # [N]

    for name, vc, vt in [
        ("v_means", v_means, _v_means),
        ("v_quats", v_quats, _v_quats),
        ("v_scales", v_scales, _v_scales),
        ("v_colors", v_colors, _v_colors),
        ("v_opacities", v_opacities, _v_opacities),
    ]:
        assert not (
            torch.isnan(vc).any() or torch.isinf(vc).any()
        ), f"eval3d {name}: NaN/Inf in CUDA grad"
        mag_c = _per_gaussian_mag(vc)[vis_g]
        mag_t = _per_gaussian_mag(vt)[vis_g]
        larger = torch.maximum(mag_c, mag_t)
        smaller = torch.minimum(mag_c, mag_t)
        sig = larger > mag_t.max() * 1e-2
        if sig.any():
            ratio = smaller[sig] / larger[sig].clamp(min=1e-30)
            n_bad = int((ratio < 5e-3).sum().item())
            assert n_bad == 0, (
                f"eval3d {name}: {n_bad} significant Gaussians with >200x "
                f"grad-magnitude mismatch (worst ratio {ratio.min().item():.4e})"
            )
            dot = _per_gaussian_dot(vc, vt)[vis_g][sig]
            n_neg = int((dot < 0).sum().item())
            n_sig = int(sig.sum().item())
            # 1% cap: a few outlier Gaussians out of thousands are
            # noise; a systematic sign flip pushes n_neg toward n_sig.
            assert n_neg <= max(2, int(n_sig * 1e-2)), (
                f"eval3d {name}: {n_neg}/{n_sig} significant Gaussians with "
                f"negative dot(cuda, ref) (>1% indicates sign-flip class)"
            )
        # Aggregate-magnitude check: per-Gaussian noise averages out across
        # thousands of visible Gaussians, so the ratio of total |grad| sums
        # is a stable summary statistic that catches systematic magnitude
        # bias on small-|e| elements (which the per-element fail_cap absorbs).
        total_c = mag_c.sum().item()
        total_t = mag_t.sum().item()
        if total_t > 0:
            agg_ratio = total_c / total_t
            assert 0.99 <= agg_ratio <= 1.01, (
                f"eval3d {name}: aggregate |cuda|/|ref| = {agg_ratio:.4f} "
                f"outside [0.99, 1.01] (systematic magnitude bias)"
            )

    # Per-element strict bound check + fail-rate cap.
    #
    # The previous quantile=0.9/0.99 mask + atol-only assert_close was
    # degenerate: when most elements are 0 on both sides (typical of a
    # rasterizer backward), the X-th percentile of |a-e| is 0, so the mask
    # kept only exact-match elements and admitted any systematic bias.
    # Replace it with an explicit per-element bound (atol + rtol*|e|) and
    # cap the fail rate over visible elements.  rtol catches percentage
    # bias on large-magnitude elements; atol absorbs the absolute noise
    # floor on small-magnitude elements; fail_cap admits the rasterizer's
    # tile-edge accumulation noise.  Knobs derived from the
    # observed baseline fail-rate distribution per gradient.
    quat_atol = 1e-4
    opacity_atol = 1e-4

    def _bounded_fail_check(name, a, e, vmask, atol, rtol, fail_cap):
        diff = (a - e).abs()
        bound = atol + rtol * e.abs()
        fail = (diff > bound) & vmask
        n_fail = int(fail.sum().item())
        n_tot = int(vmask.sum().item())
        fr = n_fail / max(n_tot, 1)
        assert fr <= fail_cap, (
            f"eval3d {name}: fail-rate {fr:.3%} > cap {fail_cap:.3%} "
            f"(atol={atol:.1e}, rtol={rtol}, {n_fail}/{n_tot})"
        )

    # fail_cap = 1.05 x worst observed baseline fail-rate:
    #   v_means    0.527% -> 0.55%
    #   v_scales   1.598% (L40S) -> 1.68%
    #   v_quats    0.18%  -> 0.19%
    #   v_colors   0.067% -> 0.071%
    #   v_opacities 0.056% -> 0.059%
    _bounded_fail_check(
        "v_means",
        v_means,
        _v_means,
        visible_mask.expand_as(v_means),
        atol=1e-3 * _lidar_tol,
        rtol=0.04,
        fail_cap=0.0055,
    )
    _bounded_fail_check(
        "v_scales",
        v_scales,
        _v_scales,
        visible_mask.expand_as(v_scales),
        atol=1e-3 * _lidar_tol,
        rtol=0.04,
        fail_cap=0.0168,
    )
    _bounded_fail_check(
        "v_quats",
        v_quats,
        _v_quats,
        visible_mask.expand_as(v_quats),
        atol=quat_atol * _lidar_tol,
        rtol=0.04,
        fail_cap=0.0019,
    )
    _bounded_fail_check(
        "v_colors",
        v_colors,
        _v_colors,
        visible_mask.expand_as(v_colors),
        atol=1e-4 * _lidar_tol,
        rtol=0.04,
        fail_cap=0.00071,
    )
    # Structural invariant for use_hit_distance:
    # - fwd overwrites pix_out[..., -1] with per-pixel hit_distance,
    #   computed from means/quats/scales/rays (not from colors[..., -1]).
    # - colors[..., -1] is therefore absent from the rendered output, so
    #   d(loss)/d(colors[..., -1]) is structurally zero.
    # - PyTorch ref produces 0 via
    #   `torch.cat([gauss_colors[..., :-1], hitDist[..., None]], -1)`.
    # - The CUDA bwd's per-Gaussian color VJP must match exactly (atol=0).
    if use_hit_distance:
        last_grad = v_colors[..., -1]
        assert torch.allclose(last_grad, torch.zeros_like(last_grad), atol=0.0), (
            "v_colors[..., -1] must be 0 when use_hit_distance=True (the "
            "last colors channel is replaced by hit_distance in fwd, so "
            "d(loss)/d(colors[..., -1]) is structurally zero). Got max="
            f"{last_grad.abs().max().item():.3e}, mean="
            f"{last_grad.abs().mean().item():.3e}."
        )
    _bounded_fail_check(
        "v_opacities",
        v_opacities,
        _v_opacities,
        visible_mask.expand_as(v_opacities),
        atol=opacity_atol * _lidar_tol,
        rtol=0.04,
        fail_cap=0.00059,
    )
    # Rolling-shutter viewmat perturbations are now seeded, so the background
    # structural-gradient bracket is repeatable. The deterministic tile_size=16
    # no-ray case reaches 1.7891e-3; keep the old global-shutter cap and use a
    # measured rolling-shutter cap with a small margin. Blackwell lidar global
    # shutter can land just above the old 1.6e-3 cap in a single background
    # gradient entry, so keep that path narrowly widened too.
    background_atol = 1.9e-3 if rs_type != RollingShutterType.GLOBAL else 2.0e-3
    assert_grad_reference_close(
        v_backgrounds_struct * backgrounds_mask.float(),
        _v_backgrounds_struct * backgrounds_mask.float(),
        rtol=0,
        atol=background_atol * _lidar_tol,
        max_rel_l2=5e-2,
        max_rel_l1=5e-2,
        min_cosine=0.999,
        max_signed_bias=5e-2,
        msg="eval3d v_backgrounds",
    )

    if use_rays:
        rays_mask = get_inlier_abserror_mask(v_rays, _v_rays, quantile=0.95)
        assert rays_mask.sum() > 0

        # Old tolerance: atol=5e-3. Bumped to 2.5e-2 after fixing the ray grid
        # from col-major to row-major (matching the pix_id = y*width+x convention).
        # The col-major layout assigned rays to wrong pixels in both CUDA and ref,
        # masking the real CUDA-vs-autograd backward divergence. With correct
        # row-major layout, v_rays magnitudes reach ~12000 and the structural
        # difference between the hand-written CUDA backward (back-to-front with
        # recomputed intermediates) and PyTorch autograd (front-to-back via
        # nerfacc) produces up to ~0.022 abs diff (confirmed with FAST_MATH=0).
        # CUDA 12.8 shows a slightly larger pre-existing worst case
        # (~0.0257568 at index (2, 1841, 1)) in this explicit-ray pinhole test,
        # while CUDA >= 12.9 stays within 2.5e-2. Keep a small margin so 12.8
        # passes without masking larger regressions.
        #
        # Fwd-state-reuse bwd path: deriving per-batch starting accumulators
        # from `dot(pix_out_final - pix_out_at_boundary, v_render_c)` introduces
        # subtraction cancellation vs. the old K1's per-Gaussian running dot,
        # pushing the worst case to ~0.0279 on 0.3% of elements. The assertion
        # uses atol=3.5e-2 to accept that drift with a small margin; keep the
        # cap and measured trace in sync when recalibrating.
        #
        # Worst cases observed (release build, FAST_MATH=1):
        #   Mismatched elements: 2511 / 34020 (7.4%)
        #   Greatest absolute difference: 0.02257537841796875 at index (2, 1388, 0) (up to 0.005 allowed)
        #   Greatest relative difference: 0.060736533254384995 at index (0, 1678, 4) (up to 0 allowed)
        assert_grad_reference_close(
            v_rays * rays_mask.float(),
            _v_rays * rays_mask.float(),
            rtol=0,
            atol=3.5e-2,
            max_rel_l2=5e-2,
            max_rel_l1=5e-2,
            min_cosine=0.999,
            max_signed_bias=5e-2,
            msg="eval3d v_rays",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize("tile_size", [8, 16], ids=["tile8", "tile16"])
@pytest.mark.parametrize(
    "renderer_config",
    [gsplat.RendererConfig_MixedBatch(), gsplat.RendererConfig_ParallelBatch()],
    ids=["mixed_batch", "parallel_batch"],
)
def test_eval3d_masked_tile_writes_safe_defaults(tile_size, renderer_config):
    # The public binding allocates outputs internally with at::empty. If a
    # kernel store is accidentally skipped, stale allocator contents could
    # still match these expected defaults and let this test pass.
    from gsplat.cuda._wrapper import rasterize_to_pixels_eval3d_extra

    width = height = tile_size
    channels = 3

    means = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.ones((1, 3), device=device)
    colors = torch.zeros((1, 1, channels), device=device)
    opacities = torch.ones((1, 1), device=device)
    backgrounds = torch.tensor([[0.2, 0.4, 0.6]], device=device)
    masks = torch.zeros((1, 1, 1), dtype=torch.bool, device=device)

    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor(
        [
            [
                [float(width), 0.0, width / 2.0],
                [0.0, float(height), height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        ],
        device=device,
    )
    isect_offsets = torch.zeros((1, 1, 1), dtype=torch.int32, device=device)
    flatten_ids = torch.empty((0,), dtype=torch.int32, device=device)

    public_render_colors, public_render_alphas = gsplat.rasterize_to_pixels_eval3d(
        means,
        quats,
        scales,
        colors,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        masks=masks,
        renderer_config=renderer_config,
    )

    expected_background = backgrounds.reshape(1, 1, 1, channels).expand_as(
        public_render_colors
    )
    torch.testing.assert_close(public_render_colors, expected_background)
    torch.testing.assert_close(
        public_render_alphas, torch.zeros_like(public_render_alphas)
    )

    (
        render_colors,
        render_alphas,
        last_ids,
        sample_counts,
        render_normals,
    ) = rasterize_to_pixels_eval3d_extra(
        means,
        quats,
        scales,
        colors,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        masks=masks,
        return_sample_counts=True,
        return_normals=True,
        renderer_config=renderer_config,
    )

    expected_background = expected_background.expand_as(render_colors)
    torch.testing.assert_close(render_colors, expected_background)
    torch.testing.assert_close(render_alphas, torch.zeros_like(render_alphas))
    torch.testing.assert_close(render_normals, torch.zeros_like(render_normals))
    torch.testing.assert_close(last_ids, torch.full_like(last_ids, -1))
    torch.testing.assert_close(sample_counts, torch.zeros_like(sample_counts))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize("tile_size", [8, 16], ids=["tile8", "tile16"])
@pytest.mark.parametrize(
    "renderer_config",
    [gsplat.RendererConfig_MixedBatch(), gsplat.RendererConfig_ParallelBatch()],
    ids=["mixed_batch", "parallel_batch"],
)
def test_eval3d_unsafe_masked_tile_outputs_match_safe_outputs_on_active_tiles(
    tile_size,
    renderer_config,
):
    from gsplat.cuda._wrapper import (
        fully_fused_projection_with_ut,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels_eval3d_extra,
    )

    # Exercise several active/masked tile transitions in a single row.
    mask_pattern = [True, False, False, True, False, True]
    tile_width = len(mask_pattern)
    tile_height = 1
    active_tile_xs = [i for i, is_active in enumerate(mask_pattern) if is_active]

    width = tile_size * tile_width
    height = tile_size
    channels = 3

    active_tile_centers = torch.tensor(
        [(tile_x + 0.5) * tile_size for tile_x in active_tile_xs],
        device=device,
    )
    xs = (active_tile_centers - width / 2.0) / width
    means = torch.stack(
        [xs, torch.zeros_like(xs), torch.ones_like(xs)],
        dim=-1,
    )
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(
        len(active_tile_xs), -1
    )
    scales = torch.full((len(active_tile_xs), 3), 0.2, device=device)
    colors = torch.tensor(
        [[[0.7, 0.1, 0.3], [0.1, 0.8, 0.2], [0.2, 0.3, 0.9]]],
        device=device,
    )
    opacities = torch.full((1, len(active_tile_xs)), 0.8, device=device)
    backgrounds = torch.tensor([[0.2, 0.4, 0.6]], device=device)
    masks = torch.tensor([[mask_pattern]], dtype=torch.bool, device=device)

    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor(
        [
            [
                [float(width), 0.0, width / 2.0],
                [0.0, float(height), height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        ],
        device=device,
    )

    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        means,
        quats,
        scales,
        opacities[0],
        viewmats,
        Ks,
        width,
        height,
    )
    _tpg, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(1, tile_height, tile_width)

    def run(unsafe_masked_tile_outputs):
        return rasterize_to_pixels_eval3d_extra(
            means,
            quats,
            scales,
            colors,
            opacities,
            viewmats,
            Ks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            masks=masks,
            return_sample_counts=True,
            return_normals=True,
            unsafe_masked_tile_outputs=unsafe_masked_tile_outputs,
            renderer_config=renderer_config,
        )

    safe_outputs = run(False)
    unsafe_outputs = run(True)

    for active_tile_x in active_tile_xs:
        # Compare only active tile pixels; masked tile outputs are undefined
        # when unsafe_masked_tile_outputs=True.
        active = (
            slice(None),
            slice(None),
            slice(active_tile_x * tile_size, (active_tile_x + 1) * tile_size),
        )
        active_with_channels = active + (slice(None),)

        assert torch.any(safe_outputs[1][active_with_channels] > 0.0)
        assert torch.any(safe_outputs[2][active] >= 0)
        assert torch.any(safe_outputs[3][active] > 0)

        for safe, unsafe in zip(safe_outputs, unsafe_outputs):
            active_slice = active_with_channels if safe.ndim == 4 else active
            if safe.is_floating_point():
                torch.testing.assert_close(
                    unsafe[active_slice], safe[active_slice], rtol=0.0, atol=1e-6
                )
            else:
                assert torch.equal(unsafe[active_slice], safe[active_slice])


@pytest.fixture
def small_scene():
    from gsplat.cuda._wrapper import (
        fully_fused_projection_with_ut,
        isect_offset_encode,
        isect_tiles,
    )

    width = height = 32
    tile_size = 16
    C = 1
    N = 4

    means = torch.tensor(
        [
            [0.0, 0.0, 4.0],
            [0.1, 0.0, 4.2],
            [-0.1, 0.1, 4.4],
            [0.0, -0.1, 4.6],
        ],
        device=device,
        dtype=torch.float32,
    )
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(N, 4).clone()
    scales = torch.full((N, 3), 0.25, device=device, dtype=torch.float32)
    opacities = torch.full((N,), 0.5, device=device, dtype=torch.float32)
    colors = torch.tensor(
        [[[0.2, 0.4, 0.6], [0.7, 0.3, 0.1], [0.1, 0.8, 0.2], [0.9, 0.9, 0.1]]],
        device=device,
        dtype=torch.float32,
    )
    viewmats = torch.eye(4, device=device, dtype=torch.float32).expand(C, 4, 4).clone()
    Ks = torch.tensor(
        [[[24.0, 0.0, 16.0], [0.0, 24.0, 16.0], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )

    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        means, quats, scales, opacities, viewmats, Ks, width, height
    )
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(C, tile_height, tile_width)

    assert flatten_ids.numel() > 0, "test setup must rasterize at least one Gaussian"
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "colors": colors,
        "opacities": opacities.unsqueeze(0).clone(),
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "isect_offsets": isect_offsets,
        "flatten_ids": flatten_ids,
    }


class InputGrad(Enum):
    REQ_GRAD = "req_grad"
    NO_GRAD = "no_grad"


class EnableGrad(Enum):
    ENABLE = "enable"
    DISABLE = "disable"
    INFERENCE = "inference"


class GradResult(Enum):
    HAS_GRAD = "has_grad"
    NO_GRAD = "no_grad"


def _call_rasterize_eval3d_extra(
    inputs, renderer_config, return_last_ids=False, return_sample_counts=False
):
    from gsplat.cuda._wrapper import rasterize_to_pixels_eval3d_extra

    return rasterize_to_pixels_eval3d_extra(
        inputs["means"],
        inputs["quats"],
        inputs["scales"],
        inputs["colors"],
        inputs["opacities"],
        inputs["viewmats"],
        inputs["Ks"],
        inputs["width"],
        inputs["height"],
        inputs["tile_size"],
        inputs["isect_offsets"],
        inputs["flatten_ids"],
        return_last_ids=return_last_ids,
        return_sample_counts=return_sample_counts,
        renderer_config=renderer_config,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize(
    "renderer_config",
    [
        gsplat.RendererConfig_MixedBatch(),
        gsplat.RendererConfig_ParallelBatch(),
    ],
    ids=["mixed_batch", "parallel_batch"],
)
@pytest.mark.parametrize(
    "request_metadata",
    [False, True],
    ids=["render_only", "with_metadata"],
)
@pytest.mark.parametrize(
    ("enable_grad", "input_grad", "grad_result"),
    [
        (EnableGrad.DISABLE, InputGrad.REQ_GRAD, GradResult.NO_GRAD),
        (EnableGrad.INFERENCE, InputGrad.REQ_GRAD, GradResult.NO_GRAD),
        (EnableGrad.ENABLE, InputGrad.NO_GRAD, GradResult.NO_GRAD),
        (EnableGrad.ENABLE, InputGrad.REQ_GRAD, GradResult.HAS_GRAD),
    ],
)
def test_rasterize_eval3d_grad_modes_save_backward_state(
    small_scene, renderer_config, enable_grad, input_grad, grad_result, request_metadata
):
    saved = []
    differentiable_names = ("means", "quats", "scales", "colors", "opacities")
    requires_grad = input_grad is InputGrad.REQ_GRAD
    inputs = dict(small_scene)
    for name in differentiable_names:
        inputs[name] = small_scene[name].detach().clone().requires_grad_(requires_grad)

    def pack(tensor):
        saved.append(True)
        return tensor

    def unpack(tensor):
        return tensor

    def call():
        return _call_rasterize_eval3d_extra(
            inputs,
            renderer_config,
            return_last_ids=request_metadata,
            return_sample_counts=request_metadata,
        )

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        if enable_grad is EnableGrad.ENABLE:
            outputs = call()
        elif enable_grad is EnableGrad.DISABLE:
            with torch.no_grad():
                outputs = call()
        elif enable_grad is EnableGrad.INFERENCE:
            with torch.inference_mode():
                outputs = call()
        else:
            raise AssertionError(f"unknown grad mode: {enable_grad}")

        if grad_result is GradResult.HAS_GRAD:
            (outputs[0].sum() + outputs[1].sum()).backward()
        torch.cuda.synchronize()

    if grad_result is GradResult.HAS_GRAD:
        assert saved
        assert outputs[0].requires_grad
        for name in differentiable_names:
            grad = inputs[name].grad
            assert grad is not None, f"{name} should receive gradients"
            assert torch.isfinite(grad).all()
    else:
        assert saved == []
        assert not outputs[0].requires_grad
        for name in differentiable_names:
            assert inputs[name].grad is None

    # Exact-metadata outputs must survive every grad mode. The no-grad / inference
    # paths route through the state-light CUDA-key forward, which must request the
    # full forward when last_ids / sample_counts are asked for rather than taking
    # the fwd-only shortcut (else the shared forward rejects the combination).
    if request_metadata:
        assert outputs[2] is not None, "last_ids should be returned"
        assert outputs[3] is not None, "sample_counts should be returned"
    else:
        assert outputs[2] is None
        assert outputs[3] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize(
    "renderer_config",
    [
        gsplat.RendererConfig_MixedBatch(),
        gsplat.RendererConfig_ParallelBatch(),
    ],
    ids=["mixed_batch", "parallel_batch"],
)
def test_rasterize_eval3d_autograd_tracks_any_tensor_input(
    small_scene, renderer_config
):
    differentiable_names = ("means", "quats", "scales", "colors", "opacities")
    inputs = dict(small_scene)
    for name in differentiable_names:
        inputs[name] = small_scene[name].detach().clone().requires_grad_(False)

    # The C++ adapter's fast path mirrors torch::autograd::Function::apply():
    # any Tensor input requiring gradients keeps the autograd path active, even
    # for inputs whose gradients this op does not currently implement.
    inputs["viewmats"] = small_scene["viewmats"].detach().clone().requires_grad_(True)

    outputs = _call_rasterize_eval3d_extra(inputs, renderer_config)
    assert outputs[0].requires_grad

    (outputs[0].sum() + outputs[1].sum()).backward()
    torch.cuda.synchronize()

    assert inputs["viewmats"].grad is None
    for name in differentiable_names:
        assert inputs[name].grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_rasterize_eval3d_parallel_batch_no_t_underflow_across_batches():
    """ParallelBatch fwd transmittance must not underflow across batches."""
    from gsplat.cuda._torch_impl_eval3d import _rasterize_to_pixels_eval3d
    from gsplat.cuda._wrapper import (
        fully_fused_projection_with_ut,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels_eval3d_extra,
        RollingShutterType,
    )

    tile_size = 16
    width = height = 32
    N = 30_000

    torch.manual_seed(0)
    means = torch.randn(N, 3, device=device) * 0.30
    quats = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.rand(N, 3, device=device).mul_(0.020).add_(0.005)
    opacities = torch.rand(N, device=device).mul_(0.40).add_(0.55)
    colors = torch.rand(1, N, 3, device=device)

    viewmats = torch.eye(4, device=device).unsqueeze(0)
    viewmats[..., 2, 3] = 2.5
    focal = 200.0
    Ks = torch.tensor(
        [[[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]],
        device=device,
    )

    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        means, quats, scales, opacities, viewmats, Ks, width, height
    )
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _tpg, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)

    # This regression targets the for-backward path, where ParallelBatch must
    # preserve enough state to match the exact serial saturation boundary.
    means = means.detach().requires_grad_(True)
    opacities_bc = opacities[None, :].contiguous()

    rc, ra, _, _, _ = rasterize_to_pixels_eval3d_extra(
        means,
        quats,
        scales,
        colors,
        opacities_bc,
        viewmats,
        Ks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        return_last_ids=False,
        rolling_shutter=RollingShutterType.GLOBAL,
        camera_model="pinhole",
        renderer_config=gsplat.RendererConfig_ParallelBatch(),
    )
    _rc, _ra, *_ = _rasterize_to_pixels_eval3d(
        means,
        quats,
        scales,
        colors,
        opacities_bc,
        viewmats,
        Ks,
        width,
        height,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        rs_type=RollingShutterType.GLOBAL,
    )

    # The serial semantic breaks before applying the Gaussian that would make
    # `T * (1 - alpha) <= TRANSMITTANCE_THRESHOLD`, so alpha should not reach
    # exactly one. Batch-local summaries that underflow only during compose
    # violate that contract and show up here before broad color tolerances can
    # hide the regression.
    assert (
        ra.max().item() < 1.0
    ), f"alpha reached {ra.max().item()} because T underflowed across batches"
    torch.testing.assert_close(ra, _ra, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(rc, _rc, rtol=3e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_rasterize_eval3d_parallel_batch_backward_accepts_unused_normals():
    from gsplat.cuda._wrapper import (
        fully_fused_projection_with_ut,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels_eval3d_extra,
    )

    torch.manual_seed(7)
    tile_size = 8
    width = height = 16
    N = 16
    C = 1

    means = torch.cat(
        [
            torch.randn(N, 2, device=device).mul_(0.10),
            torch.full((N, 1), 3.0, device=device),
        ],
        dim=-1,
    ).requires_grad_()
    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0
    quats.requires_grad_()
    scales = torch.full((N, 3), 0.08, device=device).requires_grad_()
    opacities = torch.full((N,), 0.50, device=device).requires_grad_()
    colors = torch.rand(C, N, 3, device=device, requires_grad=True)

    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor(
        [[[20.0, 0.0, width / 2.0], [0.0, 20.0, height / 2.0], [0.0, 0.0, 1.0]]],
        device=device,
    )

    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        means, quats, scales, opacities, viewmats, Ks, width, height
    )
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    (
        render_colors,
        render_alphas,
        _,
        _,
        render_normals,
    ) = rasterize_to_pixels_eval3d_extra(
        means,
        quats,
        scales,
        colors,
        opacities.unsqueeze(0),
        viewmats,
        Ks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        return_normals=True,
        renderer_config=gsplat.RendererConfig_ParallelBatch(),
    )
    assert render_normals is not None

    # The caller may request normals for inspection while optimizing only RGB /
    # alpha. Backward must still use the normal-inclusive forward state layout.
    (render_colors.sum() + render_alphas.sum()).backward()

    for grad in (means.grad, quats.grad, scales.grad, colors.grad, opacities.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize(
    "renderer_config_name",
    ["mixed_batch", "parallel_batch"],
)
def test_fwd_last_ids_match_ref_under_saturation(renderer_config_name):
    """Strict last_ids equality vs Python ref under heavy saturation."""
    from gsplat.cuda._torch_impl_eval3d import _rasterize_to_pixels_eval3d
    from gsplat.cuda._wrapper import (
        fully_fused_projection_with_ut,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels_eval3d_extra,
        RollingShutterType,
    )

    renderer_config_factories = {
        "mixed_batch": gsplat.RendererConfig_MixedBatch,
        "parallel_batch": gsplat.RendererConfig_ParallelBatch,
    }
    renderer_config = renderer_config_factories[renderer_config_name]()

    tile_size = 16
    width = height = tile_size  # single tile

    # Saturation-only fixture with little alpha-threshold ambiguity: huge
    # scales make every Gaussian cover the tile, so alpha is close to opacity
    # for all 256 pixels. With opacity=0.95, T evolves as
    # 1, 0.05, 2.5e-3, 1.25e-4, 6.25e-6, ...
    # and saturation fires unambiguously at Gaussian 3. Correct rendering
    # stops before that threshold-crossing Gaussian, leaving last_id=2 and
    # sample_count=3 for every pixel.
    num_gaussians = 8
    opacity_val = 0.95
    scale_val = 100.0

    fx = fy = float(width)
    cx = width / 2.0
    cy = height / 2.0

    means = torch.zeros(num_gaussians, 3, device=device)
    means[:, 2] = 1.0 + torch.arange(num_gaussians, device=device).float() * 1e-3
    quats = (
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        .expand(num_gaussians, 4)
        .contiguous()
    )
    scales = torch.full((num_gaussians, 3), scale_val, device=device)
    opacities = torch.full((num_gaussians,), opacity_val, device=device)
    channels = 3
    torch.manual_seed(0)
    colors = torch.rand(1, num_gaussians, channels, device=device)
    opacities_bc = opacities[None, :].contiguous()
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor(
        [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]],
        device=device,
    )

    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        means, quats, scales, opacities, viewmats, Ks, width, height
    )
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _tpg, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    means = means.detach().requires_grad_(True)

    rc, ra, render_last_ids, render_sample_counts, _ = rasterize_to_pixels_eval3d_extra(
        means,
        quats,
        scales,
        colors,
        opacities_bc,
        viewmats,
        Ks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        return_sample_counts=True,
        rolling_shutter=RollingShutterType.GLOBAL,
        camera_model="pinhole",
        renderer_config=renderer_config,
    )
    _rc, _ra, _last_ids, _sample_counts = _rasterize_to_pixels_eval3d(
        means,
        quats,
        scales,
        colors,
        opacities_bc,
        viewmats,
        Ks,
        width,
        height,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        rs_type=RollingShutterType.GLOBAL,
        return_last_ids=True,
        return_sample_counts=True,
    )

    assert (
        render_last_ids >= 0
    ).all(), "every pixel should accumulate at least one gaussian"
    assert ra.max().item() < 1.0, (
        f"expected saturation truncation to keep alpha < 1; got {ra.max().item()}. "
        "T underflowed all the way to ~0, so truncation is broken."
    )

    last_ids_diff = render_last_ids != _last_ids
    assert not last_ids_diff.any(), (
        f"last_ids diverge on {int(last_ids_diff.sum().item())} of "
        f"{render_last_ids.numel()} pixels; max(|cuda - ref|) = "
        f"{int((render_last_ids - _last_ids).abs().max().item())}. "
        "Likely cause: saturation truncation includes the threshold-crossing "
        "gaussian instead of stopping before it."
    )

    counts_diff = render_sample_counts != _sample_counts
    assert not counts_diff.any(), (
        f"sample_counts diverge on {int(counts_diff.sum().item())} pixels; "
        f"max(|cuda - ref|) = "
        f"{int((render_sample_counts - _sample_counts).abs().max().item())}."
    )

    torch.testing.assert_close(ra, _ra, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(rc, _rc, rtol=3e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_parallel_batch_fwd_only_omits_debug_metadata():
    """Fwd-only ParallelBatch should match exact output without metadata."""
    from gsplat.cuda._wrapper import (
        fully_fused_projection_with_ut,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels_eval3d_extra,
        RollingShutterType,
    )

    tile_size = 16
    width = height = tile_size
    num_gaussians = 8
    channels = 3

    means = torch.zeros(num_gaussians, 3, device=device)
    means[:, 2] = 1.0 + torch.arange(num_gaussians, device=device).float() * 1e-3
    quats = (
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        .expand(num_gaussians, 4)
        .contiguous()
    )
    scales = torch.full((num_gaussians, 3), 100.0, device=device)
    opacities = torch.full((num_gaussians,), 0.95, device=device)
    torch.manual_seed(0)
    colors = torch.rand(1, num_gaussians, channels, device=device)
    opacities_bc = opacities[None, :].contiguous()
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor(
        [
            [
                [float(width), 0.0, width / 2.0],
                [0.0, float(height), height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        ],
        device=device,
    )

    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        means, quats, scales, opacities, viewmats, Ks, width, height
    )
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _tpg, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    renderer_config = gsplat.RendererConfig_ParallelBatch()
    exact_means = means.detach().clone().requires_grad_(True)

    (
        exact_colors,
        exact_alphas,
        exact_last_ids,
        exact_sample_counts,
        _,
    ) = rasterize_to_pixels_eval3d_extra(
        exact_means,
        quats,
        scales,
        colors,
        opacities_bc,
        viewmats,
        Ks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        return_sample_counts=True,
        rolling_shutter=RollingShutterType.GLOBAL,
        camera_model="pinhole",
        renderer_config=renderer_config,
    )
    (
        render_colors,
        render_alphas,
        last_ids,
        sample_counts,
        _,
    ) = rasterize_to_pixels_eval3d_extra(
        means,
        quats,
        scales,
        colors,
        opacities_bc,
        viewmats,
        Ks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        return_last_ids=False,
        rolling_shutter=RollingShutterType.GLOBAL,
        camera_model="pinhole",
        renderer_config=renderer_config,
    )

    assert exact_last_ids is not None
    assert exact_sample_counts is not None
    assert last_ids is None
    assert sample_counts is None
    torch.testing.assert_close(render_alphas, exact_alphas, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(render_colors, exact_colors, rtol=3e-3, atol=1e-3)


@pytest.fixture
def _ghost_lobe_scene():
    """Single needle gaussian + 32x32 pinhole camera at the origin."""
    means = torch.tensor([[0.4, 0.0, 1.0]], device=device, dtype=torch.float32)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    scales = torch.tensor([[0.01, 0.01, 0.5]], device=device, dtype=torch.float32)
    opacities = torch.tensor([0.99], device=device, dtype=torch.float32)
    width, height = 32, 32
    fx = 64.0
    cx = width / 2.0
    Ks = torch.tensor(
        [[[fx, 0.0, cx], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )
    viewmats = torch.eye(4, device=device, dtype=torch.float32)[None]
    return SimpleNamespace(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        opacities_broadcast=opacities[None, :],
        colors=torch.ones(1, 1, 3, device=device, dtype=torch.float32),
        backgrounds=torch.zeros(1, 3, device=device, dtype=torch.float32),
        Ks=Ks,
        viewmats=viewmats,
        width=width,
        height=height,
        cx_int=int(cx),
    )


@pytest.fixture
def _ghost_lobe_isect(_ghost_lobe_scene):
    """Intersection list with radii inflated to cover every tile."""
    from gsplat.cuda._wrapper import (
        isect_offset_encode,
        isect_tiles,
    )

    s = _ghost_lobe_scene

    # This test intentionally bypasses projection so it can force the
    # off-image gaussian into every tile and exercise the 3D ray hit_t clamp.
    # The actual projection culls this gaussian, and the CUDA projection
    # kernel only guarantees radii=0 on culled entries.  means2d/depths are
    # allocated with at::empty and must not be used after culling.
    means2d = s.means.new_tensor([[[float(s.cx_int), float(s.height) * 0.5]]])
    depths = s.means.new_ones((1, 1))
    radii = torch.empty((1, 1, 2), device=s.means.device, dtype=torch.int32)
    radii[..., 0] = s.width
    radii[..., 1] = s.height

    tile_size = 16
    tw = math.ceil(s.width / tile_size)
    th = math.ceil(s.height / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, tile_size, tw, th)
    isect_offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(1, th, tw)
    return SimpleNamespace(
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        tile_size=tile_size,
    )


@pytest.fixture
def _ghost_lobe_rays(_ghost_lobe_scene):
    """Rays tensor: left half → ghost direction (hit_t<0), right half → visible.

    Ghost ray analysis (iscl_rot = diag(100,100,2), gro = (-40,0,-2)):
      hit_t = -dot(grd_n, gro) ≈ -40 < 0  →  GHOST; |cross|² ≈ 7.84
    """
    s = _ghost_lobe_scene
    ray_ghost = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=torch.float32)
    ray_ghost /= ray_ghost.norm()
    ray_visible = torch.tensor([1.0, 0.0, 1.0], device=device, dtype=torch.float32)
    ray_visible /= ray_visible.norm()
    col_idx = torch.arange(s.width, device=device)
    ghost_mask = (col_idx < s.cx_int)[:, None]
    rays_d = torch.where(ghost_mask, ray_ghost, ray_visible)  # [W, 3]
    rays_d = rays_d.unsqueeze(0).expand(s.height, -1, -1)  # [H, W, 3]
    rays_o = torch.zeros(s.height * s.width, 3, device=device, dtype=torch.float32)
    return torch.cat([rays_o, rays_d.reshape(-1, 3)], dim=-1).reshape(
        1, 1, s.height, s.width, 6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_rasterize_eval3d_no_behind_camera_ghost_lobe(
    _ghost_lobe_scene, _ghost_lobe_isect, _ghost_lobe_rays
):
    """A 3DGUT gaussian with hit_t = -dot(grd, gro) < 0 must contribute zero
    alpha (fwd) and zero gradient (bwd) to that pixel.

    The alpha formula uses |cross(grd, gro)|² — distance to the infinite ray
    line — so without a hit_t >= 0 clamp it produces a bilateral ghost lobe.
    """
    from gsplat.cuda._torch_impl_eval3d import _rasterize_to_pixels_eval3d
    from gsplat.cuda._wrapper import (
        rasterize_to_pixels_eval3d_extra,
        RollingShutterType,
    )

    s, isect, rays = _ghost_lobe_scene, _ghost_lobe_isect, _ghost_lobe_rays

    _, cuda_alphas, *_ = rasterize_to_pixels_eval3d_extra(
        s.means,
        s.quats,
        s.scales,
        s.colors,
        s.opacities_broadcast,
        s.viewmats,
        s.Ks,
        s.width,
        s.height,
        isect.tile_size,
        isect.isect_offsets,
        isect.flatten_ids,
        backgrounds=s.backgrounds,
        rolling_shutter=RollingShutterType.GLOBAL,
        use_hit_distance=False,
        return_last_ids=False,
        return_normals=False,
        camera_model="pinhole",
        rays=rays,
    )
    _, ref_alphas, *_ = _rasterize_to_pixels_eval3d(
        s.means,
        s.quats,
        s.scales,
        s.colors,
        s.opacities_broadcast,
        s.viewmats,
        s.Ks,
        s.width,
        s.height,
        tile_size=isect.tile_size,
        isect_offsets=isect.isect_offsets,
        flatten_ids=isect.flatten_ids,
        backgrounds=s.backgrounds,
        rs_type=RollingShutterType.GLOBAL,
        use_hit_distance=False,
        return_normals=False,
        rays=rays,
    )

    # Precondition: right half must have visible alpha (otherwise test is vacuous).
    for tag, alphas in [("CUDA", cuda_alphas), ("ref", ref_alphas)]:
        assert (
            alphas[..., s.cx_int :, :].max().item() > ALPHA_THRESHOLD
        ), f"{tag}: visible-direction pixels have no alpha — test setup is degenerate."

    # Fwd invariant: ghost-direction pixels (hit_t<0) must have zero alpha.
    for tag, alphas in [("CUDA", cuda_alphas), ("ref", ref_alphas)]:
        left_max = alphas[..., : s.cx_int, :].max().item()
        assert left_max < ALPHA_THRESHOLD, (
            f"{tag}: ghost lobe present (max alpha={left_max:.4e}). "
            f"The 3DGUT formula must skip gaussians with hit_t < 0."
        )

    # Bwd invariant: gradients from ghost pixels must be zero.
    means_bwd = s.means.detach().requires_grad_(True)
    quats_bwd = s.quats.detach().requires_grad_(True)
    scales_bwd = s.scales.detach().requires_grad_(True)
    opac_bwd = s.opacities_broadcast.detach().requires_grad_(True)
    _, alphas_bwd, *_ = rasterize_to_pixels_eval3d_extra(
        means_bwd,
        quats_bwd,
        scales_bwd,
        s.colors,
        opac_bwd,
        s.viewmats,
        s.Ks,
        s.width,
        s.height,
        isect.tile_size,
        isect.isect_offsets,
        isect.flatten_ids,
        backgrounds=s.backgrounds,
        rolling_shutter=RollingShutterType.GLOBAL,
        use_hit_distance=False,
        return_normals=False,
        camera_model="pinhole",
        rays=rays,
    )
    alphas_bwd[..., : s.cx_int, :].sum().backward()
    for name, tensor in [
        ("means", means_bwd),
        ("quats", quats_bwd),
        ("scales", scales_bwd),
        ("opacities", opac_bwd),
    ]:
        grad_max = tensor.grad.abs().max().item() if tensor.grad is not None else 0.0
        assert grad_max == 0.0, (
            f"CUDA bwd: nonzero gradient into {name} from hit_t<0 pixels "
            f"(max |grad|={grad_max:.4e})."
        )


def _make_cpp_classic_rasterization_scene(
    *,
    batch_dims: Tuple[int, ...] = (),
    n_gaussians: int = 24,
    n_cameras: int = 2,
    n_channels: int = 3,
    use_color_sh: bool = False,
    use_extra_signals: bool = False,
    use_extra_sh: bool = False,
):
    shape = batch_dims + (n_gaussians,)

    means = torch.randn(*shape, 3, device=device) * 0.25
    means[..., 2] = means[..., 2].abs() + 2.2

    quats = torch.randn(*shape, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(*shape, 3, device=device) * 0.04 + 0.02
    opacities = torch.rand(*shape, device=device) * 0.8 + 0.1

    viewmats = torch.eye(4, device=device).expand(*batch_dims, n_cameras, 4, 4).clone()
    if n_cameras > 1:
        viewmats[..., 1, 0, 3] = 0.12

    Ks = torch.eye(3, device=device).expand(*batch_dims, n_cameras, 3, 3).clone()
    Ks[..., 0, 0] = 70.0
    Ks[..., 1, 1] = 72.0
    Ks[..., 0, 2] = 24.0
    Ks[..., 1, 2] = 20.0

    if use_color_sh:
        colors = torch.rand(n_gaussians, 4, 3, device=device) * 0.4
        sh_degree = 1
    else:
        colors = torch.rand(*shape, n_channels, device=device)
        sh_degree = None

    extra_signals = None
    extra_signals_sh_degree = None
    if use_extra_signals:
        if use_extra_sh:
            extra_signals = torch.rand(n_gaussians, 4, 3, device=device) * 0.25
            extra_signals_sh_degree = 1
        else:
            extra_signals = torch.rand(*shape, 2, device=device)

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "sh_degree": sh_degree,
        "extra_signals": extra_signals,
        "extra_signals_sh_degree": extra_signals_sh_degree,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize(
    "batch_dims,packed,render_mode,rasterize_mode,n_channels,use_color_sh,use_extra_signals,use_extra_sh",
    [
        ((), True, "RGB", "classic", 46, False, False, False),
        ((2,), False, "RGB+D", "antialiased", 3, False, False, False),
        ((), True, "RGB", "classic", 3, True, False, False),
        ((), True, "RGB", "classic", 3, False, True, True),
        ((), True, "ED", "classic", 3, False, False, False),
        ((), True, "RGB+ED", "classic", 3, False, True, False),
    ],
    ids=[
        "packed_chunked_rgb",
        "batch_nonpacked_rgbd_aa",
        "packed_sh_rgb",
        "packed_extra_sh_rgb",
        "packed_depth_only_ed",
        "packed_extra_rgb_expected_depth",
    ],
)
def test_rasterization_cpp_classic_matches_python_reference(
    batch_dims: Tuple[int, ...],
    packed: bool,
    render_mode: str,
    rasterize_mode: str,
    n_channels: int,
    use_color_sh: bool,
    use_extra_signals: bool,
    use_extra_sh: bool,
):
    from gsplat.rendering import _rasterization

    torch.manual_seed(11)
    scene = _make_cpp_classic_rasterization_scene(
        batch_dims=batch_dims,
        n_channels=n_channels,
        use_color_sh=use_color_sh,
        use_extra_signals=use_extra_signals,
        use_extra_sh=use_extra_sh,
    )
    if render_mode in ("D", "ED"):
        scene["colors"] = None
        scene["sh_degree"] = None

    cpp_colors, cpp_alphas, cpp_meta = gsplat.rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=16,
        render_mode=render_mode,
        rasterize_mode=rasterize_mode,
        packed=packed,
    )
    ref_colors, ref_alphas, ref_meta = _rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=16,
        render_mode=render_mode,
        rasterize_mode=rasterize_mode,
    )

    torch.testing.assert_close(cpp_colors, ref_colors, rtol=1e-4, atol=5e-5)
    torch.testing.assert_close(cpp_alphas, ref_alphas, rtol=1e-4, atol=5e-5)

    if use_extra_signals:
        torch.testing.assert_close(
            cpp_meta["render_extra_signals"],
            ref_meta["render_extra_signals"],
            rtol=1e-4,
            atol=5e-5,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_rasterization_cpp_classic_absgrad_is_optional():
    torch.manual_seed(13)
    scene = _make_cpp_classic_rasterization_scene()

    _, _, meta = gsplat.rasterization(
        **scene,
        width=48,
        height=40,
        render_mode="RGB",
        packed=True,
        absgrad=False,
    )
    assert not hasattr(meta["means2d"], "absgrad")

    scene = _make_cpp_classic_rasterization_scene(n_channels=3)
    for tensor in (
        scene["means"],
        scene["quats"],
        scene["scales"],
        scene["opacities"],
        scene["colors"],
    ):
        tensor.requires_grad_(True)

    colors, alphas, meta = gsplat.rasterization(
        **scene,
        width=48,
        height=40,
        render_mode="RGB",
        packed=True,
        absgrad=True,
    )
    assert hasattr(meta["means2d"], "absgrad")
    assert meta["means2d"].absgrad.shape == meta["means2d"].shape

    (colors.sum() + alphas.sum()).backward()
    assert meta["means2d"].absgrad.abs().sum() > 0

    multichunk_scene = _make_cpp_classic_rasterization_scene(n_channels=46)
    with pytest.raises(RuntimeError, match="does not support absgrad with multiple"):
        gsplat.rasterization(
            **multichunk_scene,
            width=48,
            height=40,
            render_mode="RGB",
            packed=True,
            absgrad=True,
        )


def _set_rasterization_scene_requires_grad(scene: dict, names: Tuple[str, ...]) -> None:
    for name in names:
        tensor = scene[name]
        if tensor is not None:
            tensor.requires_grad_(True)


def _clone_rasterization_scene(scene: dict) -> dict:
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in scene.items()
    }


def _assert_public_rasterization_grad_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
    rtol: float = 5e-3,
    atol: float = 5e-3,
) -> None:
    assert not (
        torch.isnan(actual).any() or torch.isinf(actual).any()
    ), f"{name}: public rasterization produced NaN/Inf gradient"
    assert_grad_sparsity(actual, expected, min_ratio=0.1, msg=f"{name} sparsity")
    assert_grad_reference_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        max_rel_l2=5e-2,
        max_rel_l1=5e-2,
        min_cosine=0.999,
        max_signed_bias=5e-2,
        msg=f"{name} public rasterization gradient",
    )


def _classic_rasterization_loss_terms(colors: torch.Tensor, alphas: torch.Tensor):
    return torch.randn_like(colors), torch.randn_like(alphas)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize(
    "packed,n_channels,backgrounds",
    [
        pytest.param(True, 46, False, id="packed_chunked"),
        pytest.param(False, 6, True, id="nonpacked_backgrounds"),
    ],
)
def test_rasterization_cpp_classic_backward_matches_python_reference(
    packed: bool, n_channels: int, backgrounds: bool
):
    from gsplat.rendering import _rasterization

    torch.manual_seed(17)
    scene = _make_cpp_classic_rasterization_scene(n_channels=n_channels)
    grad_names = ("means", "quats", "scales", "opacities", "colors", "viewmats")
    _set_rasterization_scene_requires_grad(scene, grad_names)

    background_tensor = None
    if backgrounds:
        C = scene["viewmats"].shape[-3]
        background_tensor = torch.rand(C, scene["colors"].shape[-1], device=device)
        background_tensor.requires_grad_(True)
        grad_names = grad_names + ("backgrounds",)
        scene_with_backgrounds = {**scene, "backgrounds": background_tensor}
    else:
        scene_with_backgrounds = scene

    public_colors, public_alphas, _ = gsplat.rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=16,
        render_mode="RGB",
        rasterize_mode="classic",
        packed=packed,
        backgrounds=background_tensor,
    )
    v_colors, v_alphas = _classic_rasterization_loss_terms(public_colors, public_alphas)
    public_inputs = tuple(scene_with_backgrounds[name] for name in grad_names)
    public_grads = torch.autograd.grad(
        (public_colors * v_colors).sum() + (public_alphas * v_alphas).sum(),
        public_inputs,
    )

    ref_colors, ref_alphas, _ = _rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=16,
        render_mode="RGB",
        rasterize_mode="classic",
        backgrounds=background_tensor,
    )
    ref_inputs = tuple(scene_with_backgrounds[name] for name in grad_names)
    ref_grads = torch.autograd.grad(
        (ref_colors * v_colors).sum() + (ref_alphas * v_alphas).sum(),
        ref_inputs,
    )

    for name, actual, expected in zip(grad_names, public_grads, ref_grads):
        _assert_public_rasterization_grad_close(actual, expected, name=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("packed", [True, False])
def test_rasterization_cpp_classic_sh_backward_matches_python_reference(packed: bool):
    from gsplat.rendering import _rasterization

    torch.manual_seed(18)
    scene = _make_cpp_classic_rasterization_scene(use_color_sh=True)
    grad_names = ("means", "colors", "viewmats")
    _set_rasterization_scene_requires_grad(scene, grad_names)

    public_colors, public_alphas, _ = gsplat.rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=16,
        render_mode="RGB",
        rasterize_mode="classic",
        packed=packed,
    )
    v_colors, v_alphas = _classic_rasterization_loss_terms(public_colors, public_alphas)
    public_inputs = tuple(scene[name] for name in grad_names)
    public_grads = torch.autograd.grad(
        (public_colors * v_colors).sum() + (public_alphas * v_alphas).sum(),
        public_inputs,
    )

    ref_colors, ref_alphas, _ = _rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=16,
        render_mode="RGB",
        rasterize_mode="classic",
    )
    ref_inputs = tuple(scene[name] for name in grad_names)
    ref_grads = torch.autograd.grad(
        (ref_colors * v_colors).sum() + (ref_alphas * v_alphas).sum(),
        ref_inputs,
    )

    for name, actual, expected in zip(grad_names, public_grads, ref_grads):
        assert expected.abs().sum() > 0, f"{name}: reference gradient is zero"
        _assert_public_rasterization_grad_close(actual, expected, name=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_rasterization_cpp_classic_sparse_grad_layout():
    torch.manual_seed(19)
    scene = _make_cpp_classic_rasterization_scene(n_cameras=1)
    _set_rasterization_scene_requires_grad(
        scene, ("means", "quats", "scales", "opacities", "colors")
    )

    colors, alphas, _ = gsplat.rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=16,
        render_mode="RGB",
        packed=True,
        sparse_grad=True,
    )
    (colors.sum() + alphas.sum()).backward()

    for name in ("means", "quats", "scales"):
        grad = scene[name].grad
        assert grad is not None, f"{name} gradient was not produced"
        assert grad.is_sparse, f"{name} gradient should use sparse COO layout"
        assert grad._nnz() > 0, f"{name} sparse gradient should have entries"

    for name in ("opacities", "colors"):
        grad = scene[name].grad
        assert grad is not None, f"{name} gradient was not produced"
        assert not grad.is_sparse, f"{name} gradient should remain dense"
        assert grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_rasterization_cpp_classic_covars_input_path():
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci

    torch.manual_seed(23)
    scene = _make_cpp_classic_rasterization_scene()
    covars, _ = quat_scale_to_covar_preci(
        scene["quats"], scene["scales"], compute_preci=False, triu=False
    )

    quat_colors, quat_alphas, _ = gsplat.rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=16,
        render_mode="RGB",
        packed=True,
    )
    covar_colors, covar_alphas, _ = gsplat.rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=16,
        render_mode="RGB",
        packed=True,
        covars=covars,
    )

    torch.testing.assert_close(covar_colors, quat_colors, rtol=1e-4, atol=5e-5)
    torch.testing.assert_close(covar_alphas, quat_alphas, rtol=1e-4, atol=5e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize(
    "n_channels,render_mode,use_extra_signals,return_normals,absgrad",
    [
        pytest.param(46, "RGB", False, False, False, id="rgb_chunked"),
        pytest.param(
            3,
            "RGB-Ed",
            True,
            True,
            False,
            id="hit_expected_extra_normals",
        ),
        pytest.param(3, "RGB", False, False, True, id="absgrad_ignored"),
    ],
)
def test_rasterization_cpp_eval3d_matches_python_reference(
    n_channels: int,
    render_mode: str,
    use_extra_signals: bool,
    return_normals: bool,
    absgrad: bool,
):
    torch.manual_seed(29)
    scene = _make_cpp_classic_rasterization_scene(
        n_channels=n_channels,
        use_extra_signals=use_extra_signals,
    )

    cpp_colors, cpp_alphas, cpp_meta = gsplat.rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=8,
        render_mode=render_mode,
        packed=False,
        with_eval3d=True,
        return_normals=return_normals,
        absgrad=absgrad,
    )
    ref_colors, ref_alphas, ref_meta = gsplat.rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=8,
        render_mode=render_mode,
        packed=False,
        with_eval3d=True,
        return_normals=return_normals,
        # Eval3D ignores absgrad. Keep this second call on the public path to
        # verify that the flag remains forward-neutral after it stops gating
        # C++ orchestration.
        absgrad=True,
    )

    torch.testing.assert_close(cpp_colors, ref_colors, rtol=1e-4, atol=5e-5)
    torch.testing.assert_close(cpp_alphas, ref_alphas, rtol=1e-4, atol=5e-5)

    if use_extra_signals:
        torch.testing.assert_close(
            cpp_meta["render_extra_signals"],
            ref_meta["render_extra_signals"],
            rtol=1e-4,
            atol=5e-5,
        )
    if return_normals:
        assert "normals" in cpp_meta
        assert "normals" in ref_meta
        torch.testing.assert_close(
            cpp_meta["normals"], ref_meta["normals"], rtol=1e-4, atol=5e-5
        )
    if absgrad:
        assert not hasattr(cpp_meta["means2d"], "absgrad")
        assert not hasattr(ref_meta["means2d"], "absgrad")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize(
    "renderer_config",
    [
        gsplat.RendererConfig_MixedBatch(),
        gsplat.RendererConfig_ParallelBatch(),
    ],
    ids=["mixed_batch", "parallel_batch"],
)
def test_rasterization_cpp_eval3d_multichunk_hit_distance_is_final_only(
    renderer_config,
):
    torch.manual_seed(30)
    n_channels = 34
    scene = _make_cpp_classic_rasterization_scene(n_channels=n_channels)
    n_cameras = scene["viewmats"].shape[-3]
    backgrounds = torch.rand(n_cameras, n_channels, device=device)

    rgb, rgb_alpha, _ = gsplat.rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=8,
        backgrounds=backgrounds,
        render_mode="RGB",
        packed=False,
        with_eval3d=True,
        renderer_config=renderer_config,
    )
    rgb_hit, hit_alpha, _ = gsplat.rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=8,
        backgrounds=backgrounds,
        render_mode="RGB-d",
        packed=False,
        with_eval3d=True,
        renderer_config=renderer_config,
    )

    # Every primary feature must survive unchanged. Previously each chunk had
    # hit-distance enabled, which replaced channel 31 as well as the true final
    # depth placeholder.
    torch.testing.assert_close(rgb_hit[..., :n_channels], rgb, rtol=1e-4, atol=5e-5)
    torch.testing.assert_close(hit_alpha, rgb_alpha, rtol=1e-4, atol=5e-5)
    assert torch.isfinite(rgb_hit[..., -1]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_rasterization_cpp_eval3d_backward_matches_python_reference():
    from gsplat.rendering import _rasterization

    torch.manual_seed(31)
    # With the test build's compiled widths, the minimum plan for 46 channels
    # uses two launches of width 23.
    scene = _make_cpp_classic_rasterization_scene(n_channels=46)
    grad_names = ("means", "quats", "scales", "opacities", "colors")
    _set_rasterization_scene_requires_grad(scene, grad_names)

    public_colors, public_alphas, _ = gsplat.rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=8,
        render_mode="RGB",
        packed=False,
        with_eval3d=True,
    )
    v_colors, v_alphas = _classic_rasterization_loss_terms(public_colors, public_alphas)
    public_inputs = tuple(scene[name] for name in grad_names)
    public_grads = torch.autograd.grad(
        (public_colors * v_colors).sum() + (public_alphas * v_alphas).sum(),
        public_inputs,
    )

    ref_colors, ref_alphas, _ = _rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=8,
        render_mode="RGB",
        # The private Eval3D reference dispatches each uniform slice directly;
        # use the same exact 23+23 decomposition selected by the C++ planner.
        _max_channels_per_launch=23,
        with_eval3d=True,
    )
    ref_inputs = tuple(scene[name] for name in grad_names)
    ref_grads = torch.autograd.grad(
        (ref_colors * v_colors).sum() + (ref_alphas * v_alphas).sum(),
        ref_inputs,
    )

    for name, actual, expected in zip(grad_names, public_grads, ref_grads):
        _assert_public_rasterization_grad_close(actual, expected, name=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize(
    "with_eval3d,render_mode,return_normals,tile_size",
    [
        pytest.param(False, "RGB+ED", False, 16, id="projected_expected_depth"),
        pytest.param(True, "RGB-d", True, 8, id="hit_distance_normals"),
    ],
)
def test_rasterization_cpp_ut_camera_absgrad_forward_neutral(
    with_eval3d: bool,
    render_mode: str,
    return_normals: bool,
    tile_size: int,
):
    torch.manual_seed(37)
    scene = _make_cpp_classic_rasterization_scene(n_channels=3)
    C = scene["viewmats"].shape[-3]

    viewmats_rs = scene["viewmats"].clone()
    viewmats_rs[..., 0, 3] += 0.01
    radial_coeffs = torch.zeros(C, 6, device=device)
    radial_coeffs[..., 0] = 1.0e-4

    kwargs = dict(
        **scene,
        width=48,
        height=40,
        tile_size=tile_size,
        render_mode=render_mode,
        rasterize_mode="classic",
        packed=False,
        with_ut=True,
        with_eval3d=with_eval3d,
        rolling_shutter=RollingShutterType.ROLLING_TOP_TO_BOTTOM,
        viewmats_rs=viewmats_rs,
        radial_coeffs=radial_coeffs,
        return_normals=return_normals,
    )

    cpp_colors, cpp_alphas, cpp_meta = gsplat.rasterization(**kwargs)
    ref_colors, ref_alphas, ref_meta = gsplat.rasterization(**kwargs, absgrad=True)

    torch.testing.assert_close(cpp_colors, ref_colors, rtol=1e-4, atol=5e-5)
    torch.testing.assert_close(cpp_alphas, ref_alphas, rtol=1e-4, atol=5e-5)
    if return_normals:
        torch.testing.assert_close(
            cpp_meta["normals"], ref_meta["normals"], rtol=1e-4, atol=5e-5
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_rasterization_cpp_ut_rolling_shutter_sh_backward():
    torch.manual_seed(38)
    scene = _make_cpp_classic_rasterization_scene(use_color_sh=True)

    viewmats = scene["viewmats"].detach().requires_grad_(True)
    viewmats_rs = scene["viewmats"].detach().clone()
    viewmats_rs[..., 0, 3] += 0.03
    viewmats_rs[..., 1, 3] -= 0.02
    viewmats_rs.requires_grad_(True)
    scene["viewmats"] = viewmats

    render_colors, render_alphas, _ = gsplat.rasterization(
        **scene,
        width=48,
        height=40,
        tile_size=16,
        render_mode="RGB",
        rasterize_mode="classic",
        packed=False,
        with_ut=True,
        rolling_shutter=RollingShutterType.ROLLING_TOP_TO_BOTTOM,
        viewmats_rs=viewmats_rs,
    )

    assert torch.isfinite(render_colors).all()
    assert torch.isfinite(render_alphas).all()

    v_colors = torch.randn_like(render_colors)
    (render_colors * v_colors).sum().backward()
    for name, tensor in (("viewmats", viewmats), ("viewmats_rs", viewmats_rs)):
        assert tensor.grad is not None, f"{name} gradient was not produced"
        assert torch.isfinite(tensor.grad).all(), f"{name} gradient contains NaN/Inf"
        assert tensor.grad.abs().sum() > 0, f"{name} gradient is zero"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize("camera_model", ["pinhole", "lidar"])
def test_rasterization_cpp_ut_projected_absgrad_matches_python_reference(
    camera_model: str,
):
    torch.manual_seed(39)
    base_scene = _make_cpp_classic_rasterization_scene(
        n_channels=3,
        n_cameras=1,
    )
    kwargs = dict(
        width=48,
        height=40,
        tile_size=16,
        render_mode="RGB+D",
        rasterize_mode="classic",
        packed=False,
        with_ut=True,
        absgrad=True,
    )

    if camera_model == "lidar":
        from tests.core.test_cameras import parse_lidar_camera

        base_scene["means"] = base_scene["means"].clone()
        base_scene["means"][..., 0] = base_scene["means"][..., 0].abs() + 2.2
        base_scene["means"][..., 1:] *= 0.1

        lidar_params, angles_to_columns_map, tiling = parse_lidar_camera(
            "at128", (), 0, 0, device=device, seed=42
        )
        lidar_coeffs = gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
            lidar_params, angles_to_columns_map, tiling
        )
        width = lidar_coeffs.n_columns
        height = lidar_coeffs.n_rows
        focal = float(width)
        base_scene["Ks"] = torch.tensor(
            [
                [
                    [focal, 0.0, width / 2.0],
                    [0.0, focal, height / 2.0],
                    [0.0, 0.0, 1.0],
                ]
            ],
            device=device,
        )
        kwargs.update(
            width=width,
            height=height,
            camera_model="lidar",
            lidar_coeffs=lidar_coeffs,
        )

    grad_names = ("means", "quats", "scales", "opacities", "colors")
    public_scene = _clone_rasterization_scene(base_scene)
    ref_scene = _clone_rasterization_scene(base_scene)
    _set_rasterization_scene_requires_grad(public_scene, grad_names)
    _set_rasterization_scene_requires_grad(ref_scene, grad_names)

    public_colors, public_alphas, public_meta = gsplat.rasterization(
        **public_scene, **kwargs
    )
    ref_kwargs = dict(kwargs)
    ref_kwargs["absgrad"] = False
    ref_colors, ref_alphas, _ = gsplat.rasterization(**ref_scene, **ref_kwargs)

    v_colors, v_alphas = _classic_rasterization_loss_terms(public_colors, public_alphas)
    public_inputs = tuple(public_scene[name] for name in grad_names)
    torch.autograd.grad(
        (public_colors * v_colors).sum() + (public_alphas * v_alphas).sum(),
        public_inputs,
        allow_unused=True,
    )

    torch.testing.assert_close(public_colors, ref_colors, rtol=1e-4, atol=5e-5)
    torch.testing.assert_close(public_alphas, ref_alphas, rtol=1e-4, atol=5e-5)
    assert hasattr(public_meta["means2d"], "absgrad")
    assert public_meta["means2d"].absgrad.shape == public_meta["means2d"].shape
    if camera_model != "lidar":
        assert public_meta["means2d"].absgrad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize(
    "with_eval3d,render_mode,tile_size",
    [
        # RGB + two extra signals + projected depth has six rasterized channels.
        # The internal policy selects one exact CDIM=6 launch, keeping this
        # absgrad coverage within a single chunk.
        pytest.param(False, "RGB+D", 16, id="projected_depth"),
        pytest.param(True, "RGB-d", 8, id="hit_distance"),
    ],
)
def test_rasterization_cpp_ut_lidar_absgrad_forward_neutral(
    with_eval3d: bool,
    render_mode: str,
    tile_size: int,
):
    from tests.core.test_cameras import parse_lidar_camera

    torch.manual_seed(41)
    scene = _make_cpp_classic_rasterization_scene(
        n_channels=3,
        n_cameras=1,
        use_extra_signals=True,
    )
    lidar_params, angles_to_columns_map, tiling = parse_lidar_camera(
        "at128", (), 0, 0, device=device, seed=42
    )
    lidar_coeffs = gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
        lidar_params, angles_to_columns_map, tiling
    )
    width = lidar_coeffs.n_columns
    height = lidar_coeffs.n_rows
    focal = float(width)
    Ks = torch.tensor(
        [[[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    scene = {**scene, "Ks": Ks}
    kwargs = dict(
        **scene,
        width=width,
        height=height,
        tile_size=tile_size,
        render_mode=render_mode,
        packed=False,
        with_ut=True,
        with_eval3d=with_eval3d,
        camera_model="lidar",
        lidar_coeffs=lidar_coeffs,
    )

    cpp_colors, cpp_alphas, cpp_meta = gsplat.rasterization(**kwargs)
    ref_colors, ref_alphas, ref_meta = gsplat.rasterization(**kwargs, absgrad=True)

    torch.testing.assert_close(cpp_colors, ref_colors, rtol=1e-4, atol=5e-5)
    torch.testing.assert_close(cpp_alphas, ref_alphas, rtol=1e-4, atol=5e-5)
    torch.testing.assert_close(
        cpp_meta["render_extra_signals"],
        ref_meta["render_extra_signals"],
        rtol=1e-4,
        atol=5e-5,
    )
    assert cpp_meta["tile_width"] == lidar_coeffs.tiling.n_bins_azimuth
    assert cpp_meta["tile_height"] == lidar_coeffs.tiling.n_bins_elevation


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize(
    "width,height,expected_tile_size",
    [
        (540, 540, 8),  # well below 1080p
        (960, 540, 8),  # 540p training default
        (1280, 720, 8),  # 720p, min(W,H)=720 < 1080
        (1920, 1080, 16),  # 1080p, min(W,H)=1080 boundary (>= 1080)
        (1080, 1920, 16),  # vertical 1080p
        (3840, 2160, 16),  # 4K
    ],
    ids=["540x540", "540p", "720p", "1080p", "vertical_1080p", "4k"],
)
def test_rasterization_auto_tile_size_dispatch_3dgut(
    width: int, height: int, expected_tile_size: int
):
    """Verify the resolution-based tile_size fallback in gsplat.rasterization.

    When tile_size=None and with_eval3d=True (3DGUT), gsplat picks tile=8
    below 1080p (compact CTA, training-friendly) and tile=16 at 1080p+ (one
    thread per pixel, render-friendly). The gate is min(W,H) >= 1080: lidar
    grids are wide-but-shallow (n_rows ≤ 128) so they always pick 8.

    The two dispatches share the same kernel body (same CUDA template, only
    <TILE_SIZE, CTA_SIZE> constants differ) — separate parametrized tests
    above (test_rasterize_to_pixels_eval3d, etc.) verify they produce the
    same RGB/alpha output. This test only verifies the dispatch picks the
    expected tile_size.
    """
    # Minimal scene: 4 gaussians spread across the field of view.
    N = 4
    means = torch.zeros(N, 3, device=device)
    means[:, 2] = 5.0  # 5m in front of camera
    means[:, 0] = torch.linspace(-0.5, 0.5, N, device=device)
    quats = (
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(N, 4).contiguous()
    )
    scales = torch.full((N, 3), 0.2, device=device)
    opacities = torch.full((N,), 0.5, device=device)
    colors = torch.rand(1, N, 3, device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    fx = fy = float(width)
    Ks = torch.tensor(
        [[[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]]],
        device=device,
    )

    _, _, meta = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        tile_size=None,  # let gsplat pick from resolution
        camera_model="pinhole",
        packed=False,
        with_ut=True,
        with_eval3d=True,
    )
    assert meta["tile_size"] == expected_tile_size, (
        f"width={width} height={height} min(W,H)={min(width, height)}: "
        f"expected tile_size={expected_tile_size}, got {meta['tile_size']}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize(
    "width,height,explicit_tile_size",
    [
        # Force tile_size=16 at sub-1080p (would auto-pick 8).
        (540, 540, 16),
        # Force tile_size=8 at 4K (would auto-pick 16).
        (3840, 2160, 8),
    ],
    ids=["force16_at_540p", "force8_at_4k"],
)
def test_rasterization_explicit_tile_size_overrides_auto_3dgut(
    width: int, height: int, explicit_tile_size: int
):
    """Explicit tile_size kwarg must override the resolution-based fallback."""
    N = 4
    means = torch.zeros(N, 3, device=device)
    means[:, 2] = 5.0
    means[:, 0] = torch.linspace(-0.5, 0.5, N, device=device)
    quats = (
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(N, 4).contiguous()
    )
    scales = torch.full((N, 3), 0.2, device=device)
    opacities = torch.full((N,), 0.5, device=device)
    colors = torch.rand(1, N, 3, device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    fx = fy = float(width)
    Ks = torch.tensor(
        [[[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]]],
        device=device,
    )

    _, _, meta = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        tile_size=explicit_tile_size,
        camera_model="pinhole",
        packed=False,
        with_ut=True,
        with_eval3d=True,
    )
    assert meta["tile_size"] == explicit_tile_size, (
        f"explicit tile_size={explicit_tile_size} not respected at "
        f"{width}x{height}: meta has {meta['tile_size']}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_rasterization_eval3d_accepts_broadcastable_rays_shape():
    torch.manual_seed(30)
    n_gaussians = 4
    width = 48
    height = 40

    means = torch.zeros(n_gaussians, 3, device=device)
    means[:, 2] = 5.0
    means[:, 0] = torch.linspace(-0.5, 0.5, n_gaussians, device=device)
    quats = (
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        .expand(n_gaussians, 4)
        .contiguous()
    )
    scales = torch.full((n_gaussians, 3), 0.2, device=device)
    opacities = torch.full((n_gaussians,), 0.5, device=device)
    colors = torch.rand(n_gaussians, 3, device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    Ks = torch.tensor(
        [
            [
                [float(width), 0.0, width / 2.0],
                [0.0, float(width), height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        ],
        device=device,
    )

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    directions = torch.stack(
        [
            (x - width * 0.5) / width,
            (y - height * 0.5) / height,
            torch.ones_like(x),
        ],
        dim=-1,
    )
    directions = F.normalize(directions, dim=-1)
    origins = torch.zeros_like(directions)
    rays = torch.cat([origins, directions], dim=-1)

    common_kwargs = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        tile_size=8,
        render_mode="RGB",
        packed=False,
        with_eval3d=True,
    )
    broadcast_colors, broadcast_alphas, _ = gsplat.rasterization(
        **common_kwargs,
        rays=rays,
    )
    exact_colors, exact_alphas, _ = gsplat.rasterization(
        **common_kwargs,
        rays=rays.unsqueeze(0).contiguous(),
    )

    torch.testing.assert_close(broadcast_colors, exact_colors, rtol=0, atol=0)
    torch.testing.assert_close(broadcast_alphas, exact_alphas, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("D", [1, 3, 5])
def test_sh(sh_degree: int, batch_dims: Tuple[int, ...], packed: bool, D: int):
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from gsplat.cuda._wrapper import spherical_harmonics

    if packed and batch_dims != ():
        pytest.skip("packed inputs use explicit batch IDs; batch_dims is irrelevant")

    torch.manual_seed(42)

    N = 1000
    C = 3
    K = (4 + 1) ** 2
    coeffs_src = torch.randn(N, K, D, device=device, requires_grad=True)
    means = torch.randn(*batch_dims, N, 3, device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device).expand(*batch_dims, C, 4, 4).clone()
    angles = torch.randn(*batch_dims, C, device=device)
    viewmats[..., 0, 0] = angles.cos()
    viewmats[..., 0, 1] = -angles.sin()
    viewmats[..., 1, 0] = angles.sin()
    viewmats[..., 1, 1] = angles.cos()
    viewmats[..., :3, 3] = torch.randn(*batch_dims, C, 3, device=device)
    viewmats.requires_grad_(True)

    if packed:
        # Mirror the packed call site (rendering.py): a [N, K, D] source of
        # per-Gaussian coeffs is gathered into [nnz, K, D] via gaussian_ids
        # (one row per visible (Gaussian, camera) pair).
        # nnz > N with random ids exercises the duplicate-gaussian regime that
        # the unpacked broadcast path never hits.
        nnz = 3000
        gaussian_ids = torch.randint(0, N, (nnz,), device=device)
        batch_ids = torch.zeros(nnz, dtype=torch.long, device=device)
        camera_ids = torch.randint(0, C, (nnz,), device=device)
        coeffs = coeffs_src[gaussian_ids]  # [nnz, K, D]
        rotations = viewmats[camera_ids, :3, :3]
        translations = viewmats[camera_ids, :3, 3]
        dirs = means[gaussian_ids] + torch.bmm(
            rotations.transpose(-1, -2), translations.unsqueeze(-1)
        ).squeeze(-1)
        expected_colors_shape = (nnz, D)
    else:
        coeffs = coeffs_src
        batch_ids = camera_ids = gaussian_ids = None
        camera_offsets = torch.matmul(
            viewmats[..., :3, :3].transpose(-1, -2),
            viewmats[..., :3, 3].unsqueeze(-1),
        ).squeeze(-1)
        dirs = means[..., None, :, :] + camera_offsets[..., :, None, :]
        expected_colors_shape = (*batch_dims, C, N, D)

    colors = spherical_harmonics(
        sh_degree,
        means,
        viewmats,
        coeffs,
        batch_ids=batch_ids,
        camera_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    _colors = _spherical_harmonics(sh_degree, dirs, coeffs)
    assert colors.shape == expected_colors_shape, colors.shape
    torch.testing.assert_close(colors, _colors, rtol=1e-4, atol=1e-4)

    v_colors = torch.randn_like(colors)

    # Take grads w.r.t. coeffs_src (the [N, K, D] leaf) so packed mode also
    # exercises the gather VJP that accumulates duplicate-id rows back to source.
    v_coeffs_src, v_means, v_viewmats = torch.autograd.grad(
        (colors * v_colors).sum(),
        (coeffs_src, means, viewmats),
        retain_graph=True,
        allow_unused=True,
    )
    _v_coeffs_src, _v_means, _v_viewmats = torch.autograd.grad(
        (_colors * v_colors).sum(),
        (coeffs_src, means, viewmats),
        retain_graph=True,
        allow_unused=True,
    )
    assert v_coeffs_src.shape == (N, K, D), v_coeffs_src.shape
    assert_grad_reference_close(
        v_coeffs_src,
        _v_coeffs_src,
        rtol=1e-4,
        atol=1e-4,
        max_rel_l2=1e-3,
        max_rel_l1=1e-3,
        min_cosine=0.999999,
        max_signed_bias=1e-3,
        msg="v_coeffs_src",
    )
    if sh_degree > 0:
        assert v_means.shape == means.shape, v_means.shape
        assert_grad_reference_close(
            v_means,
            _v_means,
            rtol=1e-4,
            atol=1e-4,
            max_rel_l2=1e-3,
            max_rel_l1=1e-3,
            min_cosine=0.999999,
            max_signed_bias=1e-3,
            msg="v_means",
        )
        assert_grad_reference_close(
            v_viewmats,
            _v_viewmats,
            rtol=1e-4,
            # atomicAdd accumulation over N gaussians is non-deterministic, so
            # small-magnitude entries can drift slightly past a 1e-4 floor; the
            # aggregate guards below keep the overall gradient tight.
            atol=2e-3,
            max_rel_l2=1e-3,
            max_rel_l1=1e-3,
            min_cosine=0.999999,
            max_signed_bias=1e-3,
            msg="v_viewmats",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("batch_dims", [(), (2,)])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("D", [1, 3])
def test_sh_split_invariant(
    sh_degree: int, batch_dims: Tuple[int, ...], packed: bool, D: int
):
    """Split sh0/shN evaluation matches evaluating their concatenation."""
    from gsplat.cuda._wrapper import (
        spherical_harmonics,
        spherical_harmonics_l0,
        spherical_harmonics_l1_plus,
    )

    if packed and batch_dims != ():
        pytest.skip("packed inputs use explicit batch IDs; batch_dims is irrelevant")

    torch.manual_seed(42)

    N = 127
    C = 2
    K = (4 + 1) ** 2
    sh0_src = torch.randn(N, 1, D, device=device, requires_grad=True)
    shN_src = torch.randn(N, K - 1, D, device=device, requires_grad=True)
    means = torch.randn(*batch_dims, N, 3, device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device).expand(*batch_dims, C, 4, 4).clone()
    viewmats[..., :3, 3] = torch.randn(*batch_dims, C, 3, device=device)
    viewmats.requires_grad_(True)

    if packed:
        nnz = 311
        gaussian_ids = torch.randint(0, N, (nnz,), device=device)
        batch_ids = torch.zeros(nnz, dtype=torch.long, device=device)
        camera_ids = torch.randint(0, C, (nnz,), device=device)
        sh0 = sh0_src[gaussian_ids]
        shN = shN_src[gaussian_ids]
    else:
        batch_ids = camera_ids = gaussian_ids = None
        sh0 = sh0_src
        shN = shN_src

    sh_args = dict(
        batch_ids=batch_ids,
        camera_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    colors = spherical_harmonics(
        sh_degree, means, viewmats, torch.cat([sh0, shN], dim=1), **sh_args
    )
    l0 = spherical_harmonics_l0(sh0)
    l1_plus = spherical_harmonics_l1_plus(sh_degree, means, viewmats, shN, **sh_args)
    split_colors = l0 + l1_plus

    assert l0.shape == (sh0.shape[0], D)
    assert l1_plus.shape == colors.shape
    torch.testing.assert_close(colors, split_colors, rtol=1e-4, atol=1e-4)

    v_colors = torch.randn_like(colors)
    full_grads = torch.autograd.grad(
        (colors * v_colors).sum(),
        (sh0_src, shN_src, means, viewmats),
        retain_graph=True,
    )
    split_grads = torch.autograd.grad(
        (split_colors * v_colors).sum(),
        (sh0_src, shN_src, means, viewmats),
        retain_graph=True,
    )

    # The full coefficient gradient is reconstructed by concatenating the two
    # split coefficient gradients in the same layout used by simple_trainer.py.
    full_v_coeffs = torch.cat(full_grads[:2], dim=1)
    split_v_coeffs = torch.cat(split_grads[:2], dim=1)
    assert_grad_reference_close(
        split_v_coeffs,
        full_v_coeffs,
        rtol=1e-4,
        atol=1e-4,
        max_rel_l2=1e-3,
        max_rel_l1=1e-3,
        min_cosine=0.999999,
        max_signed_bias=1e-3,
        msg="split v_coeffs",
    )
    assert_grad_reference_close(
        split_grads[2],
        full_grads[2],
        rtol=1e-4,
        atol=1e-4,
        max_rel_l2=1e-3,
        max_rel_l1=1e-3,
        min_cosine=0.999999,
        max_signed_bias=1e-3,
        msg="split v_means",
    )
    assert_grad_reference_close(
        split_grads[3],
        full_grads[3],
        rtol=1e-4,
        atol=1e-4,
        max_rel_l2=1e-3,
        max_rel_l1=1e-3,
        min_cosine=0.999999,
        max_signed_bias=1e-3,
        msg="split v_viewmats",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_sh_split_rolling_shutter_backward():
    """Split SH matches full SH with gradients to both shutter endpoints."""
    from gsplat.cuda._wrapper import (
        spherical_harmonics,
        spherical_harmonics_l0,
        spherical_harmonics_l1_plus,
    )

    torch.manual_seed(43)

    degree = 2
    N, C, D = 127, 2, 3
    K = (degree + 1) ** 2
    sh0 = torch.randn(N, 1, D, device=device, requires_grad=True)
    shN = torch.randn(N, K - 1, D, device=device, requires_grad=True)
    means = torch.randn(N, 3, device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device).expand(C, 4, 4).clone()
    viewmats[..., :3, 3] = torch.randn(C, 3, device=device)
    viewmats.requires_grad_(True)
    viewmats_rs = viewmats.detach().clone()
    viewmats_rs[..., :3, 3] += torch.randn(C, 3, device=device) * 0.1
    viewmats_rs.requires_grad_(True)

    colors = spherical_harmonics(
        degree,
        means,
        viewmats,
        torch.cat([sh0, shN], dim=1),
        viewmats_rs=viewmats_rs,
    )
    split_colors = spherical_harmonics_l0(sh0) + spherical_harmonics_l1_plus(
        degree, means, viewmats, shN, viewmats_rs=viewmats_rs
    )
    torch.testing.assert_close(colors, split_colors, rtol=1e-4, atol=1e-4)

    inputs = (sh0, shN, means, viewmats, viewmats_rs)
    v_colors = torch.randn_like(colors)
    full_grads = torch.autograd.grad(
        (colors * v_colors).sum(), inputs, retain_graph=True
    )
    split_grads = torch.autograd.grad(
        (split_colors * v_colors).sum(), inputs, retain_graph=True
    )
    for name, split_grad, full_grad in zip(
        ("sh0", "shN", "means", "viewmats", "viewmats_rs"),
        split_grads,
        full_grads,
    ):
        assert_grad_reference_close(
            split_grad,
            full_grad,
            rtol=1e-4,
            atol=1e-4,
            max_rel_l2=1e-3,
            max_rel_l1=1e-3,
            min_cosine=0.999999,
            max_signed_bias=1e-3,
            msg=f"split v_{name}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_sh_split_invariant_trainer_layout_fp16():
    """Trainer-layout FP16 split evaluation matches the K=16 RGB path."""
    from gsplat.cuda._wrapper import (
        spherical_harmonics,
        spherical_harmonics_l0,
        spherical_harmonics_l1_plus,
    )

    torch.manual_seed(42)

    N, D, sh_degree = 257, 3, 3
    sh0 = torch.randn(N, 1, D, device=device, dtype=torch.float16, requires_grad=True)
    # Fifteen coefficients is the exact minimum for degree 3 after omitting sh0.
    shN = torch.randn(N, 15, D, device=device, dtype=torch.float16, requires_grad=True)
    means = torch.randn(N, 3, device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device).expand(2, 4, 4).clone()
    viewmats[:, :3, 3] = torch.randn(2, 3, device=device)
    viewmats.requires_grad_(True)

    coeffs = torch.cat([sh0, shN], dim=1)
    assert coeffs.shape == (N, 16, D)
    assert coeffs.data_ptr() % 16 == 0

    colors = spherical_harmonics(sh_degree, means, viewmats, coeffs)
    l0 = spherical_harmonics_l0(sh0)
    l1_plus = spherical_harmonics_l1_plus(sh_degree, means, viewmats, shN)
    split_colors = l0 + l1_plus

    torch.testing.assert_close(split_colors, colors, rtol=1e-3, atol=2e-3)

    v_colors = torch.randn_like(colors)
    full_grads = torch.autograd.grad(
        (colors * v_colors).sum(), (sh0, shN, means, viewmats), retain_graph=True
    )
    split_grads = torch.autograd.grad(
        (split_colors * v_colors).sum(),
        (sh0, shN, means, viewmats),
        retain_graph=True,
    )

    full_v_coeffs = torch.cat(full_grads[:2], dim=1).float()
    split_v_coeffs = torch.cat(split_grads[:2], dim=1).float()
    assert_grad_reference_close(
        split_v_coeffs,
        full_v_coeffs,
        rtol=2e-3,
        atol=5e-4,
        max_rel_l2=5e-3,
        max_rel_l1=5e-3,
        min_cosine=0.99999,
        max_signed_bias=5e-3,
        msg="trainer FP16 split v_coeffs",
    )
    assert_grad_reference_close(
        split_grads[2],
        full_grads[2],
        rtol=1e-4,
        atol=1e-4,
        max_rel_l2=1e-3,
        max_rel_l1=1e-3,
        min_cosine=0.999999,
        max_signed_bias=1e-3,
        msg="trainer FP16 split v_means",
    )
    assert_grad_reference_close(
        split_grads[3],
        full_grads[3],
        rtol=1e-4,
        atol=1e-4,
        max_rel_l2=1e-3,
        max_rel_l1=1e-3,
        min_cosine=0.999999,
        max_signed_bias=1e-3,
        msg="trainer FP16 split v_viewmats",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_sh_split_l0_accepts_empty_shn():
    from gsplat.cuda._wrapper import (
        spherical_harmonics,
        spherical_harmonics_l0,
        spherical_harmonics_l1_plus,
    )

    torch.manual_seed(42)
    N, D = 32, 3
    sh0 = torch.randn(N, 1, D, device=device, requires_grad=True)
    shN = torch.empty(N, 0, D, device=device, requires_grad=True)
    means = torch.randn(N, 3, device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device).expand(2, 4, 4).clone().requires_grad_(True)

    colors = spherical_harmonics(0, means, viewmats, sh0)
    l0 = spherical_harmonics_l0(sh0)
    l1_plus = spherical_harmonics_l1_plus(0, means, viewmats, shN)

    torch.testing.assert_close(l1_plus, torch.zeros_like(l1_plus))
    torch.testing.assert_close(colors, l0 + l1_plus)

    v_shN, v_means, v_viewmats = torch.autograd.grad(
        l1_plus.sum(), (shN, means, viewmats), allow_unused=False
    )
    torch.testing.assert_close(v_shN, torch.zeros_like(shN))
    torch.testing.assert_close(v_means, torch.zeros_like(means))
    torch.testing.assert_close(v_viewmats, torch.zeros_like(viewmats))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_sh_backward_accepts_strided_output_grad():
    """SH backward should accept channel-slice gradients from downstream cats."""
    from gsplat.cuda._wrapper import spherical_harmonics

    torch.manual_seed(42)

    N, K, D = 128, (3 + 1) ** 2, 3
    means = torch.randn(N, 3, device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device).expand(2, 4, 4).clone().requires_grad_(True)
    coeffs = torch.randn(N, K, D, device=device, requires_grad=True)

    colors = spherical_harmonics(3, means, viewmats, coeffs)
    grad_storage = torch.randn(2, N, D * 2, device=device)
    v_colors = grad_storage[..., :D]
    assert not v_colors.is_contiguous()

    v_coeffs, v_means, v_viewmats = torch.autograd.grad(
        colors, (coeffs, means, viewmats), v_colors
    )

    assert torch.isfinite(v_coeffs).all()
    assert torch.isfinite(v_means).all()
    assert torch.isfinite(v_viewmats).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_sh_zero_channels():
    """Test that an error is thrown when D = 0 (i.e. empty per-Gaussian feature)"""
    from gsplat.cuda._wrapper import spherical_harmonics

    N = 8
    K = (4 + 1) ** 2
    means = torch.randn(N, 3, device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    coeffs = torch.randn(N, K, 0, device=device)
    with pytest.raises(RuntimeError):
        spherical_harmonics(0, means, viewmats, coeffs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("kernel_path", ["generic", "specialized"])
def test_sh_fp16_coeffs(sh_degree: int, kernel_path: str):
    """fp16 SH coeffs through the generic kernel (K != 16) and the K=16 specialized kernel."""
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from gsplat.cuda._wrapper import spherical_harmonics

    if kernel_path == "specialized":
        if sh_degree == 4:
            pytest.skip("K=16 caps at sh_degree=3")
        K = 16
    else:
        K = (sh_degree + 1) ** 2
        if K == 16:
            pytest.skip("K=16 is covered by the specialized path")

    torch.manual_seed(42)

    N = 1000
    coeffs_fp32 = torch.randn(N, K, 3, device=device)
    means_src = torch.randn(N, 3, device=device)
    viewmats_src = torch.eye(4, device=device).unsqueeze(0)
    viewmats_src[:, :3, 3] = torch.randn(1, 3, device=device)

    # fp16 coefficients through CUDA kernel
    coeffs_h = coeffs_fp32.half().requires_grad_(True)
    means_h = means_src.clone().requires_grad_(True)
    viewmats_h = viewmats_src.clone().requires_grad_(True)
    colors_h = spherical_harmonics(sh_degree, means_h, viewmats_h, coeffs_h)

    # Reference 1: roundtripped fp16->fp32 through pure-PyTorch (isolates kernel correctness)
    coeffs_ref = coeffs_fp32.half().float().requires_grad_(True)
    means_ref = means_src.clone().requires_grad_(True)
    viewmats_ref = viewmats_src.clone().requires_grad_(True)
    dirs_ref = (
        means_ref[None]
        + torch.matmul(
            viewmats_ref[:, :3, :3].transpose(-1, -2),
            viewmats_ref[:, :3, 3].unsqueeze(-1),
        ).squeeze(-1)[:, None, :]
    )
    colors_ref = _spherical_harmonics(sh_degree, dirs_ref, coeffs_ref)

    # Reference 2: true fp32 through pure-PyTorch (measures total fp16 precision loss)
    coeffs_fp32_ref = coeffs_fp32.clone().requires_grad_(True)
    means_fp32_ref = means_src.clone().requires_grad_(True)
    viewmats_fp32_ref = viewmats_src.clone().requires_grad_(True)
    dirs_fp32_ref = (
        means_fp32_ref[None]
        + torch.matmul(
            viewmats_fp32_ref[:, :3, :3].transpose(-1, -2),
            viewmats_fp32_ref[:, :3, 3].unsqueeze(-1),
        ).squeeze(-1)[:, None, :]
    )
    colors_fp32_ref = _spherical_harmonics(sh_degree, dirs_fp32_ref, coeffs_fp32_ref)

    # Forward: kernel correctness (tight, same quantized inputs)
    torch.testing.assert_close(colors_h, colors_ref, rtol=1e-4, atol=1e-4)
    # Forward: total fp16 precision loss vs true fp32
    torch.testing.assert_close(colors_h, colors_fp32_ref, rtol=1e-3, atol=2e-3)

    # Backward check
    v_colors = torch.randn_like(colors_h)

    v_coeffs_h, v_means_h, v_viewmats_h = torch.autograd.grad(
        (colors_h * v_colors).sum(),
        (coeffs_h, means_h, viewmats_h),
        retain_graph=True,
        allow_unused=True,
    )
    v_coeffs_ref, v_means_ref, v_viewmats_ref = torch.autograd.grad(
        (colors_ref * v_colors).sum(),
        (coeffs_ref, means_ref, viewmats_ref),
        retain_graph=True,
        allow_unused=True,
    )
    v_coeffs_fp32_ref, v_means_fp32_ref, v_viewmats_fp32_ref = torch.autograd.grad(
        (colors_fp32_ref * v_colors).sum(),
        (coeffs_fp32_ref, means_fp32_ref, viewmats_fp32_ref),
        retain_graph=True,
        allow_unused=True,
    )

    # v_coeffs kernel correctness (wider than forward, fp16 quantization on write-back)
    assert_grad_reference_close(
        v_coeffs_h.float(),
        v_coeffs_ref,
        rtol=2e-3,
        atol=5e-4,
        max_rel_l2=5e-3,
        max_rel_l1=5e-3,
        min_cosine=0.99999,
        max_signed_bias=5e-3,
        msg="v_coeffs_h vs fp16-ref",
    )
    # v_coeffs total precision loss vs true fp32
    assert_grad_reference_close(
        v_coeffs_h.float(),
        v_coeffs_fp32_ref,
        rtol=1e-2,
        atol=1e-3,
        max_rel_l2=5e-2,
        max_rel_l1=5e-2,
        min_cosine=0.999,
        max_signed_bias=5e-2,
        msg="v_coeffs_h vs fp32-ref",
    )
    if sh_degree > 0:
        assert_grad_reference_close(
            v_means_h,
            v_means_ref,
            rtol=1e-4,
            atol=1e-4,
            max_rel_l2=1e-3,
            max_rel_l1=1e-3,
            min_cosine=0.999999,
            max_signed_bias=1e-3,
            msg="v_means_h vs fp16-ref",
        )
        # Geometry-gradient precision loss versus true fp32; higher-order bands
        # amplify coefficient quantization error.
        assert_grad_reference_close(
            v_means_h,
            v_means_fp32_ref,
            rtol=5e-2,
            atol=1e-2,
            max_rel_l2=1e-1,
            max_rel_l1=1e-1,
            min_cosine=0.99,
            max_signed_bias=1e-1,
            msg="v_means_h vs fp32-ref",
        )
        assert_grad_reference_close(
            v_viewmats_h,
            v_viewmats_ref,
            rtol=1e-4,
            atol=1e-4,
            max_rel_l2=1e-3,
            max_rel_l1=1e-3,
            min_cosine=0.999999,
            max_signed_bias=1e-3,
            msg="v_viewmats_h vs fp16-ref",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize(
    "dtype, sh_degree, storage_offset",
    [
        # storage_offset = 1 fp16 elem (2 bytes): fails ushort4 (8-byte) and uint4 (16-byte).
        (torch.float16, 0, 1),
        (torch.float16, 3, 1),
        # storage_offset = 4 fp16 elems (8 bytes): passes ushort4 but fails uint4.
        # fp16 + degree >= 2 uses uint4, so this must still fall back to scalar.
        (torch.float16, 3, 4),
        # fp32: storage_offset = 1 elem (4 bytes): fails uint4 (16-byte).
        (torch.float32, 0, 1),
        (torch.float32, 3, 1),
        # fp32 + storage_offset = 2 elems (8 bytes): still fails uint4 (need 16).
        (torch.float32, 3, 2),
    ],
)
def test_sh_k16_misaligned_coeffs(dtype, sh_degree, storage_offset):
    """K=16 with a contiguous-but-misaligned coeffs view should fall back to the generic kernel.

    Wide-load alignment by (dtype, degree):
      ushort4 (8-byte): fp16 + degree <= 1
      uint4 (16-byte):  fp16 + degree >= 2, or fp32 at any degree
    """
    from gsplat.cuda._wrapper import spherical_harmonics

    torch.manual_seed(42)
    N, K = 1000, 16

    coeffs_aligned = torch.randn(N, K, 3, device=device, dtype=dtype)
    means = torch.randn(N, 3, device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)

    storage = torch.empty(N * K * 3 + storage_offset, device=device, dtype=dtype)
    storage[:storage_offset] = 0
    coeffs_misaligned = storage[storage_offset:].view(N, K, 3)
    coeffs_misaligned.copy_(coeffs_aligned)
    assert coeffs_misaligned.is_contiguous()
    assert coeffs_misaligned.storage_offset() == storage_offset

    colors_aligned = spherical_harmonics(sh_degree, means, viewmats, coeffs_aligned)
    colors_misaligned = spherical_harmonics(
        sh_degree, means, viewmats, coeffs_misaligned
    )

    torch.testing.assert_close(colors_misaligned, colors_aligned)


# ============================================================================
# NaN/wrong-value safety tests for the 3DGUT code path
# ============================================================================


def _render_alpha(data, quats, scales, means2d, radii, depths, tile_size=8):
    """Render the scene and return the per-pixel alpha tensor.

    `tile_size` selects the 3DGUT kernel instantiation (<TILE,CTA> in
    {<8,32>, <16,256>}); default 8 since these helpers exist for projection
    NaN-safety tests where the rasterizer's tile_size is incidental.
    """
    from gsplat.cuda._wrapper import (
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels_eval3d_extra,
    )

    N = data["means"].shape[0]
    C = data["viewmats"].shape[0]
    W, H = data["width"], data["height"]

    tw = math.ceil(W / tile_size)
    th = math.ceil(H / tile_size)
    _, iids, fids = isect_tiles(means2d, radii, depths, tile_size, tw, th)
    ioff = isect_offset_encode(iids, C, tw, th).reshape(C, th, tw)
    colors = torch.ones(C, N, 3, device=device)
    opac_bc = data["opacities"].unsqueeze(0).expand(C, N)
    _, ra, _, _, _ = rasterize_to_pixels_eval3d_extra(
        data["means"],
        quats,
        scales,
        colors,
        opac_bc,
        data["viewmats"],
        data["Ks"],
        W,
        H,
        tile_size,
        ioff,
        fids,
        return_last_ids=False,
    )
    return ra


@pytest.fixture
def nan_test_data():
    """Small synthetic dataset for NaN-safety tests.

    Gaussians are placed directly in front of the camera, guaranteed
    visible with the default projection parameters.  Opacity is set low
    (0.1) so that any bug producing MAX_ALPHA (0.99) is provably wrong:
    a single Gaussian with opacity p can produce at most alpha=p per pixel.
    """
    torch.manual_seed(42)
    N = 8
    C = 1

    # Center all Gaussians at z=3 (well inside near/far planes) and near the
    # optical axis so they definitely project into the image.
    means = torch.zeros(N, 3, device=device)
    means[:, 2] = 3.0

    quats = _safe_normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.ones(N, 3, device=device) * 0.2
    # Low opacity: a single Gaussian can produce at most alpha=0.1 per pixel.
    # Under the bug, fminf(MAX_ALPHA, 0.1 * NaN) = 0.99, which is impossible.
    opacities = torch.full((N,), 0.1, device=device)

    viewmats = torch.eye(4, device=device).unsqueeze(0).expand(C, 4, 4).contiguous()
    Ks = (
        torch.tensor(
            [
                [200.0, 0.0, 32.0],
                [0.0, 200.0, 24.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
        )
        .unsqueeze(0)
        .expand(C, 3, 3)
        .contiguous()
    )

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": 64,
        "height": 48,
    }


# --------------------------------------------------------------------------
# UT parameter validation
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_ut_params_invalid_kappa_rejected():
    """kappa < -D makes sqrt(D + lambda) produce NaN.  The constructor must reject it."""
    with pytest.raises(RuntimeError, match=r"alpha.*kappa"):
        UnscentedTransformParameters(alpha=0.1, kappa=-4.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_ut_params_valid_accepted():
    """Default and typical UT parameters must be accepted without error."""
    UnscentedTransformParameters()  # defaults
    UnscentedTransformParameters(alpha=1.0, beta=0.0, kappa=0.0)
    UnscentedTransformParameters(alpha=0.5, kappa=2.0)


# --------------------------------------------------------------------------
# Zero-quaternion Gaussians culled in projection
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_projection_ut_zero_quaternion(nan_test_data):
    """A zero-quaternion Gaussian must be culled (radii=0).

    Without the fix, under --use_fast_math, glm::normalize(0) returns (0,0,0,0)
    instead of NaN.  glm::mat3_cast then produces an identity rotation, so the
    Gaussian passes projection with nonzero radii.  In the rasterization kernel,
    quat_to_rotmat(0) produces NaN rotation → NaN power → exp(NaN) = NaN →
    fminf(MAX_ALPHA, opacity * NaN) = MAX_ALPHA.  A single Gaussian with
    opacity=0.1 renders alpha=0.99 — an impossible value.
    """
    from gsplat.cuda._wrapper import fully_fused_projection_with_ut
    from gsplat.cuda._torch_impl_ut import _fully_fused_projection_with_ut

    data = nan_test_data
    N = data["means"].shape[0]
    W, H = data["width"], data["height"]
    OPACITY = data["opacities"][0].item()

    # Precondition: Gaussian 0 is visible with its valid quaternion
    radii_ref, _, _, _, _ = fully_fused_projection_with_ut(
        data["means"],
        data["quats"],
        data["scales"],
        data["opacities"],
        data["viewmats"],
        data["Ks"],
        W,
        H,
    )
    assert (
        radii_ref[..., 0, :] > 0
    ).all(), "Gaussian 0 must be visible with valid quat"

    # Inject zero quaternion at index 0
    quats = data["quats"].clone()
    quats[0] = 0.0

    proj_params = dict(
        means=data["means"],
        quats=quats,
        scales=data["scales"],
        opacities=data["opacities"],
        viewmats=data["viewmats"],
        Ks=data["Ks"],
        width=W,
        height=H,
    )

    # CUDA implementation
    radii_gpu, means2d_gpu, depths_gpu, conics_gpu, _ = fully_fused_projection_with_ut(
        **proj_params
    )
    # Python reference implementation
    radii_ref, means2d_ref, depths_ref, conics_ref, _ = _fully_fused_projection_with_ut(
        **proj_params
    )

    # Both must cull the zero-quat Gaussian (both x and y radii)
    assert (
        radii_gpu[..., 0, :] == 0
    ).all(), f"CUDA: zero-quat Gaussian should have radii=0, got {radii_gpu[..., 0, :].tolist()}"
    assert (
        radii_ref[..., 0, :] == 0
    ).all(), f"Ref: zero-quat Gaussian should have radii=0, got {radii_ref[..., 0, :].tolist()}"

    # Both must agree on which Gaussians are visible
    sel_gpu = (radii_gpu > 0).all(dim=-1)
    sel_ref = (radii_ref > 0).all(dim=-1)
    assert_mismatch_ratio(sel_gpu, sel_ref, max=1e-3)
    sel = sel_gpu & sel_ref

    # Valid Gaussians must match between CUDA and Python ref
    if sel.any():
        torch.testing.assert_close(
            means2d_gpu[sel], means2d_ref[sel], rtol=2e-3, atol=1e-3
        )
        torch.testing.assert_close(
            depths_gpu[sel], depths_ref[sel], rtol=1e-6, atol=2e-6
        )
        torch.testing.assert_close(
            conics_gpu[sel], conics_ref[sel], rtol=1e-4, atol=1e-4
        )

    # End-to-end: render and verify no pixel exceeds the opacity bound.
    # With N Gaussians each at opacity p, each contributes at most alpha=p per pixel.
    # After N, max render_alpha = 1 - (1-p)^N.
    max_possible_alpha = 1.0 - (1.0 - OPACITY) ** N
    ra = _render_alpha(data, quats, data["scales"], means2d_gpu, radii_gpu, depths_gpu)
    assert ra.max().item() <= max_possible_alpha + 1e-3, (
        f"render_alpha {ra.max().item():.4f} exceeds theoretical max "
        f"{max_possible_alpha:.4f} for opacity={OPACITY} with {N} Gaussians"
    )


# --------------------------------------------------------------------------
# Zero-scale (single axis) Gaussians culled in projection
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_projection_ut_zero_scale_single_axis(nan_test_data):
    """A Gaussian with a single zero-scale axis must be culled (radii=0).

    Without the fix, a single-axis zero scale still produces a valid-looking
    2D covariance from the UT, so it passes projection with nonzero radii.
    In rasterization, 1/scale[i] = inf → precision matrix diverges →
    under --use_fast_math, fminf(MAX_ALPHA, opacity * NaN) = MAX_ALPHA.
    A single Gaussian with opacity=0.1 renders alpha=0.99 — impossible.
    """
    from gsplat.cuda._wrapper import fully_fused_projection_with_ut
    from gsplat.cuda._torch_impl_ut import _fully_fused_projection_with_ut

    data = nan_test_data
    N = data["means"].shape[0]
    W, H = data["width"], data["height"]
    OPACITY = data["opacities"][0].item()

    # Precondition: Gaussian 0 is visible with valid scales
    radii_ref, _, _, _, _ = fully_fused_projection_with_ut(
        data["means"],
        data["quats"],
        data["scales"],
        data["opacities"],
        data["viewmats"],
        data["Ks"],
        W,
        H,
    )
    assert (
        radii_ref[..., 0, :] > 0
    ).all(), "Gaussian 0 must be visible with valid scales"

    # Zero a single axis
    scales = data["scales"].clone()
    scales[0, 0] = 0.0

    proj_params = dict(
        means=data["means"],
        quats=data["quats"],
        scales=scales,
        opacities=data["opacities"],
        viewmats=data["viewmats"],
        Ks=data["Ks"],
        width=W,
        height=H,
    )

    # CUDA implementation
    radii_gpu, means2d_gpu, depths_gpu, conics_gpu, _ = fully_fused_projection_with_ut(
        **proj_params
    )
    # Python reference implementation
    radii_ref, means2d_ref, depths_ref, conics_ref, _ = _fully_fused_projection_with_ut(
        **proj_params
    )

    # Both must cull the degenerate Gaussian (both x and y radii)
    assert (
        radii_gpu[..., 0, :] == 0
    ).all(), f"CUDA: single-axis zero-scale should have radii=0, got {radii_gpu[..., 0, :].tolist()}"
    assert (
        radii_ref[..., 0, :] == 0
    ).all(), f"Ref: single-axis zero-scale should have radii=0, got {radii_ref[..., 0, :].tolist()}"

    # Both must agree on which Gaussians are visible
    sel_gpu = (radii_gpu > 0).all(dim=-1)
    sel_ref = (radii_ref > 0).all(dim=-1)
    assert_mismatch_ratio(sel_gpu, sel_ref, max=1e-3)
    sel = sel_gpu & sel_ref

    # Valid Gaussians must match between CUDA and Python ref
    if sel.any():
        torch.testing.assert_close(
            means2d_gpu[sel], means2d_ref[sel], rtol=2e-3, atol=1e-3
        )
        torch.testing.assert_close(
            depths_gpu[sel], depths_ref[sel], rtol=1e-6, atol=2e-6
        )
        torch.testing.assert_close(
            conics_gpu[sel], conics_ref[sel], rtol=1e-4, atol=1e-4
        )

    # End-to-end: render and verify no pixel exceeds the opacity bound.
    max_possible_alpha = 1.0 - (1.0 - OPACITY) ** N
    ra = _render_alpha(data, data["quats"], scales, means2d_gpu, radii_gpu, depths_gpu)
    assert ra.max().item() <= max_possible_alpha + 1e-3, (
        f"render_alpha {ra.max().item():.4f} exceeds theoretical max "
        f"{max_possible_alpha:.4f} for opacity={OPACITY} with {N} Gaussians"
    )


# --------------------------------------------------------------------------
# Dgenerate Gaussians must not corrupt rasterization
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize("tile_size", [8, 16], ids=["tile8", "tile16"])
def test_rasterize_eval3d_degenerate_gaussians_culled(nan_test_data, tile_size):
    """End-to-end: degenerate Gaussians (zero quat, zero scale) must be
    culled in projection so they never reach the rasterization kernel.

    Without the fix, under --use_fast_math, fminf(MAX_ALPHA, NaN) = MAX_ALPHA,
    so every overlapping pixel gets an opaque splat.  With opacity=0.1 and N=8
    Gaussians, the theoretical maximum render_alpha is 1-(1-0.1)^8 ≈ 0.57.
    The bug produces render_alpha=0.99, exceeding this bound.

    Compares CUDA rasterization against reference to verify they agree on the
    rendered image when degenerate Gaussians are present.
    """
    from gsplat.cuda._wrapper import (
        fully_fused_projection_with_ut,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels_eval3d_extra,
    )
    from gsplat.cuda._torch_impl_eval3d import _rasterize_to_pixels_eval3d

    data = nan_test_data
    N = data["means"].shape[0]
    C = data["viewmats"].shape[0]
    W, H = data["width"], data["height"]
    OPACITY = data["opacities"][0].item()
    max_possible_alpha = 1.0 - (1.0 - OPACITY) ** N

    colors = torch.rand(C, N, 3, device=device)

    # Inject degenerate Gaussians
    quats_bad = data["quats"].clone()
    scales_bad = data["scales"].clone()
    quats_bad[0] = 0.0  # zero quaternion
    scales_bad[1, 0] = 0.0  # single-axis zero scale

    # Projection (shared between CUDA and ref rasterization)
    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        data["means"],
        quats_bad,
        scales_bad,
        data["opacities"],
        data["viewmats"],
        data["Ks"],
        W,
        H,
    )

    # Both degenerate Gaussians must be culled (both x and y radii)
    assert (radii[..., 0, :] == 0).all(), "Zero-quat Gaussian should be culled"
    assert (radii[..., 1, :] == 0).all(), "Zero-scale Gaussian should be culled"

    # Replace degenerate values with safe dummies for rasterization.
    # The degenerate Gaussians are already culled (radii=0) so they won't
    # enter the tile intersection list.  But the Python reference rasterization
    # iterates over all Gaussians and asserts positive scales.
    quats_safe = quats_bad.clone()
    scales_safe = scales_bad.clone()
    quats_safe[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    scales_safe[1, 0] = 1.0

    # Tile intersection. tile_size flows in from the parametrize and selects
    # the 3DGUT kernel instantiation (<CDIM,8,32> or <CDIM,16,256>).
    tw = math.ceil(W / tile_size)
    th = math.ceil(H / tile_size)
    _tpg, iids, fids = isect_tiles(means2d, radii, depths, tile_size, tw, th)
    ioff = isect_offset_encode(iids, C, tw, th).reshape(C, th, tw)
    opac_bc = data["opacities"].unsqueeze(0).expand(C, N)

    # CUDA rasterization
    rc_gpu, ra_gpu, _, _, _ = rasterize_to_pixels_eval3d_extra(
        data["means"],
        quats_safe,
        scales_safe,
        colors,
        opac_bc,
        data["viewmats"],
        data["Ks"],
        W,
        H,
        tile_size,
        ioff,
        fids,
        return_last_ids=False,
    )

    # Python reference rasterization
    ref_outputs = _rasterize_to_pixels_eval3d(
        data["means"],
        quats_safe,
        scales_safe,
        colors,
        opac_bc,
        data["viewmats"],
        data["Ks"],
        W,
        H,
        tile_size=tile_size,
        isect_offsets=ioff,
        flatten_ids=fids,
    )
    rc_ref, ra_ref = ref_outputs[0], ref_outputs[1]

    # Both must be finite
    assert torch.isfinite(rc_gpu).all(), "NaN/inf in CUDA render_colors"
    assert torch.isfinite(ra_gpu).all(), "NaN/inf in CUDA render_alphas"
    assert torch.isfinite(rc_ref).all(), "NaN/inf in ref render_colors"
    assert torch.isfinite(ra_ref).all(), "NaN/inf in ref render_alphas"

    # CUDA and reference must agree
    torch.testing.assert_close(rc_gpu, rc_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ra_gpu, ra_ref, rtol=1e-3, atol=1e-3)

    # Render alpha must not exceed theoretical max
    assert ra_gpu.max().item() <= max_possible_alpha + 1e-3, (
        f"CUDA render_alpha {ra_gpu.max().item():.4f} exceeds theoretical max "
        f"{max_possible_alpha:.4f} for opacity={OPACITY} with {N} Gaussians"
    )
    assert ra_ref.max().item() <= max_possible_alpha + 1e-3, (
        f"Ref render_alpha {ra_ref.max().item():.4f} exceeds theoretical max "
        f"{max_possible_alpha:.4f} for opacity={OPACITY} with {N} Gaussians"
    )


# --------------------------------------------------------------------------
# Python reference — NaN from torch.sqrt(negative cov diag) in UT projection
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_projection_ut_python_ref_no_nan(nan_test_data):
    """CUDA and Python ref must agree on degenerate inputs, with no NaN.

    Under default UT params (alpha=0.1), the center covariance weight is ≈ -96.
    Without the fix, torch.sqrt(cov_diag) on a negative diagonal produces NaN,
    and torch.linalg.inv on a singular covariance produces NaN/inf.
    """
    from gsplat.cuda._wrapper import fully_fused_projection_with_ut
    from gsplat.cuda._torch_impl_ut import _fully_fused_projection_with_ut

    data = nan_test_data
    quats = data["quats"].clone()
    scales = data["scales"].clone()
    quats[0] = 0.0  # zero quaternion
    scales[1] = 0.0  # all-zero scale

    ut_params = UnscentedTransformParameters(alpha=0.1, beta=2.0, kappa=0.0)

    proj_params = dict(
        means=data["means"],
        quats=quats,
        scales=scales,
        opacities=data["opacities"],
        viewmats=data["viewmats"],
        Ks=data["Ks"],
        width=data["width"],
        height=data["height"],
        ut_params=ut_params,
    )

    radii_gpu, means2d_gpu, depths_gpu, conics_gpu, _ = fully_fused_projection_with_ut(
        **proj_params
    )
    radii_ref, means2d_ref, depths_ref, conics_ref, _ = _fully_fused_projection_with_ut(
        **proj_params
    )

    # Both must agree on culling degenerate Gaussians
    sel_gpu = (radii_gpu > 0).all(dim=-1)
    sel_ref = (radii_ref > 0).all(dim=-1)
    assert_mismatch_ratio(sel_gpu, sel_ref, max=1e-3)
    sel = sel_gpu & sel_ref

    # Valid Gaussians must have finite outputs in both implementations
    if sel.any():
        assert torch.isfinite(means2d_gpu[sel]).all(), "NaN in CUDA means2d"
        assert torch.isfinite(conics_gpu[sel]).all(), "NaN in CUDA conics"
        assert torch.isfinite(means2d_ref[sel]).all(), "NaN in ref means2d"
        assert torch.isfinite(conics_ref[sel]).all(), "NaN in ref conics"

        # CUDA and ref must agree on valid Gaussian outputs
        torch.testing.assert_close(
            means2d_gpu[sel], means2d_ref[sel], rtol=2e-3, atol=1e-3
        )
        torch.testing.assert_close(
            depths_gpu[sel], depths_ref[sel], rtol=1e-6, atol=2e-6
        )
        torch.testing.assert_close(
            conics_gpu[sel], conics_ref[sel], rtol=1e-4, atol=1e-4
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_projection_ut_negative_preblur_diag_survives_eps2d():
    """A valid UT covariance may have a negative diagonal before blur.

    This fixture was found by an independent UT/pinhole search. Before blur,
    ``covar2d.xx`` is negative (-2.4e-4), but adding eps2d=0.3 makes the
    covariance valid. The projection path must preserve the explicit UT-valid
    flag and avoid treating ``covar2d.xx < 0`` as an invalid sentinel.
    """
    from gsplat.cuda._wrapper import fully_fused_projection_with_ut
    from gsplat.cuda._torch_impl_ut import _fully_fused_projection_with_ut

    width = 1_000_000
    height = 1_000_000
    proj_params = dict(
        means=torch.tensor(
            [[-16.73026466369629, 33.509700775146484, 0.1956430971622467]],
            device=device,
        ),
        quats=torch.tensor(
            [
                [
                    -0.40441247820854187,
                    0.5313093066215515,
                    -0.4663550853729248,
                    0.58023601770401,
                ]
            ],
            device=device,
        ),
        scales=torch.tensor(
            [[0.08389746397733688, 0.10447215288877487, 0.0018264513928443193]],
            device=device,
        ),
        opacities=None,
        viewmats=torch.eye(4, device=device).reshape(1, 4, 4),
        Ks=torch.tensor(
            [
                [
                    [0.01, 0.0, 0.5 * width],
                    [0.0, 0.0107, 0.5 * height],
                    [0.0, 0.0, 1.0],
                ]
            ],
            device=device,
        ),
        width=width,
        height=height,
        eps2d=0.3,
        calc_compensations=True,
        ut_params=UnscentedTransformParameters(alpha=0.1, beta=2.0, kappa=0.0),
    )

    (
        radii_gpu,
        means2d_gpu,
        depths_gpu,
        conics_gpu,
        comps_gpu,
    ) = fully_fused_projection_with_ut(**proj_params)
    (
        radii_ref,
        means2d_ref,
        depths_ref,
        conics_ref,
        comps_ref,
    ) = _fully_fused_projection_with_ut(**proj_params)

    assert (radii_gpu[0, 0] > 0).all()
    assert (radii_ref[0, 0] > 0).all()
    assert torch.isfinite(means2d_gpu[0, 0]).all()
    assert torch.isfinite(depths_gpu[0, 0])
    assert torch.isfinite(conics_gpu[0, 0]).all()
    assert torch.isfinite(comps_gpu[0, 0])

    # The existing CUDA/reference UT integration check allows small radius
    # differences because the Python reference uses torch.linalg.inv/radius math
    # rather than the CUDA kernel's exact fp32 expression graph. This fixture
    # pins selection and finite matching outputs for the recovered covariance.
    #
    # means2d here is near 5e5 (image center is width/2 = 5e5). Projecting a
    # near-origin point onto that center subtracts large, nearly equal values, so
    # the CUDA and reference expression graphs round differently. rtol=4e-6 allows
    # ~4e-6 * 5e5 = 2 pixels (roughly 64 representable float32 steps at this
    # magnitude); the tighter atol still guards small coordinates.
    torch.testing.assert_close(means2d_gpu, means2d_ref, rtol=4e-6, atol=1e-3)
    torch.testing.assert_close(depths_gpu, depths_ref, rtol=0, atol=0)

    # CUDA directly inverts the packed float32 covariance, while the float32
    # reference adds 1e-6 to the diagonal before calling torch.linalg.inv_ex.
    # For this near-degenerate fixture, that regularization and the different
    # expression graph produce ~10% off-diagonal differences, so check that the
    # CUDA conic is finite and positive-definite rather than elementwise-close.
    a, b, c = conics_gpu[0, 0, 0], conics_gpu[0, 0, 1], conics_gpu[0, 0, 2]
    assert torch.isfinite(comps_gpu[0, 0]).all()
    assert (a > 0).item() and (c > 0).item()
    assert (a * c - b * b > 0).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_projection_ut_tiny_positive_det_blur_keeps_reciprocal_path():
    """A tiny positive blurred determinant must still produce finite conics.

    This pins the regression where packed covariance inversion special-cased
    ``det_blur <= float::epsilon`` to a zero inverse. Baseline ``glm::inverse``
    divides by every positive determinant, even tiny ones.
    """
    from gsplat.cuda._wrapper import fully_fused_projection_with_ut

    eps2d = 1e-4
    means = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.full((1, 3), 1e-5, device=device)
    viewmats = torch.eye(4, device=device).reshape(1, 4, 4)
    Ks = torch.tensor(
        [[[1.0, 0.0, 8.0], [0.0, 1.0, 8.0], [0.0, 0.0, 1.0]]],
        device=device,
    )

    radii, means2d, depths, conics, compensations = fully_fused_projection_with_ut(
        means=means,
        quats=quats,
        scales=scales,
        opacities=None,
        viewmats=viewmats,
        Ks=Ks,
        width=16,
        height=16,
        eps2d=eps2d,
        calc_compensations=True,
        ut_params=UnscentedTransformParameters(alpha=0.1, beta=2.0, kappa=0.0),
    )

    assert (radii[0, 0] > 0).all()
    assert torch.isfinite(means2d[0, 0]).all()
    assert torch.isfinite(depths[0, 0]).all()
    assert torch.isfinite(conics[0, 0]).all()
    assert torch.isfinite(compensations[0, 0])
    assert conics[0, 0, 0] > 0.0
    assert conics[0, 0, 2] > 0.0
    assert torch.count_nonzero(conics[0, 0]).item() > 0


# --------------------------------------------------------------------------
# Backward stability: 1/(1-alpha) clamp when alpha -> 1
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
@pytest.mark.parametrize("tile_size", [8, 16], ids=["tile8", "tile16"])
def test_backward_high_opacity_no_nan(tile_size):
    """Backward pass must produce finite gradients when alpha approaches 1.0.

    High-opacity Gaussians stacked at the same location drive per-pixel alpha
    toward 1.0.  In the backward pass, T_(i-1) = T_i / (1 - alpha_i).
    The MIN_ONE_MINUS_ALPHA clamp guards this division.

    Uses opacity=0.99 with N=16 Gaussians packed at the same (x,y) to
    guarantee alpha→1.0.  Cross-validates CUDA forward against the Python
    reference (which uses autograd, not the explicit 1/(1-alpha) backward).

    .. note::
        The MIN_ONE_MINUS_ALPHA clamp is **defense-in-depth**: the forward pass
        already clamps ``alpha <= MAX_ALPHA (0.99)``, so ``1-alpha >= 0.01``
        and the backward division stays bounded.  This test cannot trigger the
        MIN_ONE_MINUS_ALPHA clamp directly but serves as a **regression test**
        ensuring the high-opacity backward path produces finite gradients and
        matches the Python reference.

    .. note::
        Only tests the 3DGUT/FromWorld path.  The 3DGS and 2DGS backward
        kernels have the same one-line MIN_ONE_MINUS_ALPHA fix and are covered
        by the parametrized ``test_rasterize`` gradient-correctness tests.
        # TODO: add dedicated high-opacity tests for 3DGS/2DGS paths.
    """
    from gsplat.cuda._wrapper import (
        fully_fused_projection_with_ut,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels_eval3d_extra,
    )
    from gsplat.cuda._torch_impl_eval3d import _rasterize_to_pixels_eval3d

    torch.manual_seed(123)
    N = 16
    C = 1
    W, H = 64, 48

    # Pack all Gaussians at the same (x,y) with opacity=1.0 to drive alpha
    # to MAX_ALPHA (0.99) on every covered pixel.  opacity=1.0 is the most
    # extreme real-world scenario and guarantees every Gaussian contributes
    # the maximum possible alpha per pixel.
    means = torch.zeros(N, 3, device=device)
    means[:, 2] = 3.0
    means[:, :2] = torch.randn(N, 2, device=device) * 0.001  # tight cluster

    quats = _safe_normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.ones(N, 3, device=device) * 0.5
    opacities = torch.full((N,), 1.0, device=device)

    viewmats = torch.eye(4, device=device).unsqueeze(0).expand(C, 4, 4).contiguous()
    Ks = (
        torch.tensor(
            [
                [200.0, 0.0, 32.0],
                [0.0, 200.0, 24.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
        )
        .unsqueeze(0)
        .expand(C, 3, 3)
        .contiguous()
    )

    means.requires_grad_(True)
    quats.requires_grad_(True)
    scales.requires_grad_(True)

    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        means,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        W,
        H,
    )

    # tile_size flows in from the parametrize and selects the 3DGUT kernel
    # instantiation (<CDIM,8,32> or <CDIM,16,256>).
    tw = math.ceil(W / tile_size)
    th = math.ceil(H / tile_size)
    _, iids, fids = isect_tiles(means2d, radii, depths, tile_size, tw, th)
    ioff = isect_offset_encode(iids, C, tw, th).reshape(C, th, tw)

    colors = torch.rand(C, N, 3, device=device, requires_grad=True)
    opac_bc = opacities.unsqueeze(0).expand(C, N)

    # CUDA forward
    render_colors, render_alphas, _, _, _ = rasterize_to_pixels_eval3d_extra(
        means,
        quats,
        scales,
        colors,
        opac_bc,
        viewmats,
        Ks,
        W,
        H,
        tile_size,
        ioff,
        fids,
    )

    # Precondition: alpha must actually approach 1.0 to exercise the
    # high-opacity backward path where 1/(1-alpha) is large
    assert render_alphas.max() > 0.99, (
        f"Precondition failed: max render_alpha={render_alphas.max().item():.4f}, "
        f"need >0.99 to stress the 1/(1-alpha) backward path"
    )

    assert torch.isfinite(render_colors).all(), "NaN/Inf in CUDA forward render_colors"
    assert torch.isfinite(render_alphas).all(), "NaN/Inf in CUDA forward render_alphas"

    # Python reference forward (uses autograd, no explicit 1/(1-alpha) backward)
    ref_outputs = _rasterize_to_pixels_eval3d(
        means,
        quats,
        scales,
        colors,
        opac_bc,
        viewmats,
        Ks,
        W,
        H,
        tile_size=tile_size,
        isect_offsets=ioff,
        flatten_ids=fids,
    )
    rc_ref, ra_ref = ref_outputs[0], ref_outputs[1]

    assert torch.isfinite(rc_ref).all(), "NaN/Inf in ref forward render_colors"
    assert torch.isfinite(ra_ref).all(), "NaN/Inf in ref forward render_alphas"

    # CUDA and reference forward must agree
    torch.testing.assert_close(render_colors, rc_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(render_alphas, ra_ref, rtol=1e-3, atol=1e-3)

    # Backward: gradients must be finite
    loss = render_colors.sum() + render_alphas.sum()
    loss.backward()

    for name, param in [
        ("means", means),
        ("quats", quats),
        ("scales", scales),
        ("colors", colors),
    ]:
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(
            param.grad
        ).all(), f"NaN/Inf in {name}.grad (max={param.grad.abs().max().item():.2e})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("I", [1, 2])
def test_rasterize_to_pixels_3dgs_masked_tile_outputs_initialized(packed: bool, I: int):
    # A masked-out tile takes the forward rasterizer's early-return branch.
    # The op allocates its outputs with at::empty, so render_colors and
    # render_alphas must still be explicitly initialized before returning.
    # last_ids is the op's 4th output (saved by the Python autograd's
    # setup_context); the public rasterize_to_pixels_3dgs wrapper drops it.

    tile_size = 16
    width = height = tile_size  # single tile
    channels = 3
    N = 1

    means2d = torch.full((I, N, 2), width / 2.0, device=device)
    conics = torch.tensor([1.0, 0.0, 1.0], device=device).expand(I, N, 3).clone()
    colors = torch.ones((I, N, channels), device=device)
    opacities = torch.ones((I, N), device=device)
    if packed:
        means2d = means2d.flatten(0, 1)
        conics = conics.flatten(0, 1)
        colors = colors.flatten(0, 1)
        opacities = opacities.flatten(0, 1)

    backgrounds_batched = torch.tensor(
        [[0.2, 0.4, 0.6], [0.6, 0.4, 0.2]], device=device
    )[:I]
    # Packed single-image callers historically passed [channels]. Preserve
    # that safe legacy shape alongside the canonical [I, channels] form.
    backgrounds = (
        backgrounds_batched[0].clone()
        if packed and I == 1
        else backgrounds_batched.clone()
    ).requires_grad_()
    masks = torch.zeros((I, 1, 1), dtype=torch.bool, device=device)  # tiles masked off
    isect_offsets = torch.zeros((I, 1, 1), dtype=torch.int32, device=device)
    flatten_ids = torch.empty((0,), dtype=torch.int32, device=device)

    (
        render_colors,
        render_alphas,
        means2d_absgrad,
        _last_ids,
    ) = torch.ops.gsplat.rasterize_to_pixels_3dgs(
        means2d,
        conics,
        colors,
        opacities,
        backgrounds,
        masks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        packed,
        False,  # absgrad
    )

    torch.testing.assert_close(
        render_colors,
        backgrounds_batched[:, None, None, :].expand_as(render_colors),
    )
    torch.testing.assert_close(render_alphas, torch.zeros_like(render_alphas))
    assert means2d_absgrad.numel() == 0

    render_colors.sum().backward()
    torch.testing.assert_close(
        backgrounds.grad,
        torch.full_like(backgrounds, width * height),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT/Lidar support isn't built in")
def test_isect_tiles_lidar_double_depth():
    # intersect_tile_lidar_kernel narrows the depth sort key to float32. With
    # float64 depths a bare 32-bit slice of the double is its low mantissa bits
    # (non-monotonic), scrambling the within-tile front-to-back order. The
    # double path is reachable from Python (no dtype cast in the wrapper or
    # binding), so the float64 result must match the float32 result tile-for-tile.
    from gsplat.cuda._torch_impl_lidar import ANGLE_TO_PIXEL_SCALING_FACTOR
    from gsplat.cuda._wrapper import isect_tiles_lidar
    from tests.core.test_cameras import parse_lidar_camera

    torch.manual_seed(42)

    lidar_params, angles_to_columns_map, tiling = parse_lidar_camera(
        "pandar128", (), 0, 0, device=device
    )
    lidar = gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
        lidar_params, angles_to_columns_map, tiling
    )

    C, N = 3, 1000
    means2d = (
        torch.randn(C, N, 2, device=device)
        * torch.tensor([2 * math.pi, math.pi], device=device)
    ) * ANGLE_TO_PIXEL_SCALING_FACTOR
    radii = torch.ceil(
        torch.randn(C, N, 2, device=device).abs().clamp(max=1)
        * torch.tensor([math.pi, lidar.fov_vert_rad.span / 2], device=device)
        * ANGLE_TO_PIXEL_SCALING_FACTOR
    ).to(torch.int32)
    depths = torch.rand(C, N, device=device)

    tpg32, iid32, fid32 = isect_tiles_lidar(lidar, means2d, radii, depths, sort=True)
    # Same values, double dtype -> exercises the scalar_t=double instantiation.
    tpg64, iid64, fid64 = isect_tiles_lidar(
        lidar, means2d.double(), radii, depths.double(), sort=True
    )

    torch.testing.assert_close(tpg64, tpg32)
    torch.testing.assert_close(iid64, iid32)
    torch.testing.assert_close(fid64, fid32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT/Lidar support isn't built in")
def test_isect_tiles_packed_segmented_rejected():
    # In packed mode tiles_per_gauss is 1-D [nnz], so the per-image segment
    # offsets collapse to a single [0, total] buffer; a segmented radix sort
    # over n_images segments would read past it. The op must reject the
    # combination rather than corrupt the sort.
    from gsplat.cuda._wrapper import isect_tiles

    torch.manual_seed(42)
    nnz = 8
    means2d = torch.randn(nnz, 2, device=device) * 40
    radii = torch.randint(1, 10, (nnz, 2), device=device, dtype=torch.int32)
    depths = torch.rand(nnz, device=device)
    image_ids = torch.zeros(nnz, dtype=torch.int32, device=device)
    gaussian_ids = torch.arange(nnz, dtype=torch.int32, device=device)

    with pytest.raises(
        RuntimeError, match="segmented sort is not supported for packed inputs"
    ):
        isect_tiles(
            means2d,
            radii,
            depths,
            tile_size=16,
            tile_width=4,
            tile_height=4,
            sort=True,
            segmented=True,
            packed=True,
            n_images=1,
            image_ids=image_ids,
            gaussian_ids=gaussian_ids,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT/Lidar support isn't built in")
def test_isect_tiles_lidar_packed_segmented_rejected():
    # Lidar sibling of the camera packed+segmented guard.
    from gsplat.cuda._wrapper import isect_tiles_lidar
    from tests.core.test_cameras import parse_lidar_camera

    torch.manual_seed(42)
    lidar_params, angles_to_columns_map, tiling = parse_lidar_camera(
        "pandar128", (), 0, 0, device=device
    )
    lidar = gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
        lidar_params, angles_to_columns_map, tiling
    )

    nnz = 8
    means2d = torch.randn(nnz, 2, device=device)
    radii = torch.randint(1, 10, (nnz, 2), device=device, dtype=torch.int32)
    depths = torch.rand(nnz, device=device)
    image_ids = torch.zeros(nnz, dtype=torch.int32, device=device)
    gaussian_ids = torch.arange(nnz, dtype=torch.int32, device=device)

    with pytest.raises(
        RuntimeError, match="segmented sort is not supported for packed inputs"
    ):
        isect_tiles_lidar(
            lidar,
            means2d,
            radii,
            depths,
            sort=True,
            segmented=True,
            packed=True,
            n_images=1,
            image_ids=image_ids,
            gaussian_ids=gaussian_ids,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("C", [1, 3])
def test_fully_fused_projection_packed_empty(C):
    # Packed EWA-3DGS projection computed the batch count as numel/(N*3) before
    # the empty-input guard; with N==0 that divisor is zero (host-side integer
    # division by zero) in BOTH the forward and backward launchers. An empty
    # gaussian set must be a clean no-op (empty packed tensors) through fwd+bwd.
    # C > 1 also pins the indptr shape contract: a length-1 indptr would pass a
    # bare indptr[-1] check yet drop the per-camera rows the CSR API promises.
    from gsplat.cuda._wrapper import fully_fused_projection

    W, H = 64, 64
    means = torch.zeros(0, 3, device=device, requires_grad=True)
    quats = torch.zeros(0, 4, device=device, requires_grad=True)
    scales = torch.ones(0, 3, device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device)[None].expand(C, 4, 4).contiguous()
    Ks = (
        torch.tensor(
            [[float(W), 0.0, W / 2.0], [0.0, float(W), H / 2.0], [0.0, 0.0, 1.0]],
            device=device,
        )[None]
        .expand(C, 3, 3)
        .contiguous()
    )

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
        means, None, quats, scales, viewmats, Ks, W, H, packed=True
    )

    assert gaussian_ids.numel() == 0
    assert means2d.shape[0] == 0
    assert radii.shape[0] == 0
    # B == 1 for means.shape == (0, 3), so indptr has B * C + 1 == C + 1 zero
    # entries -- one row per camera plus the trailing total.
    assert indptr.shape == (C + 1,)
    assert bool((indptr == 0).all())

    # Backward over the empty projection must not divide by zero in the launcher.
    (means2d.sum() + depths.sum() + conics.sum()).backward()
    assert means.grad.shape == means.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_2dgs(), reason="2DGS support isn't built in")
@pytest.mark.parametrize("C", [1, 3])
def test_fully_fused_projection_2dgs_packed_empty(C):
    # Same empty-input batch-count division-by-zero as the EWA packed projection,
    # in the 2DGS packed launcher's forward and backward. An empty gaussian set
    # must be a clean no-op (empty packed tensors) through fwd+bwd.
    # C > 1 also pins the indptr shape contract: a length-1 indptr would pass a
    # bare indptr[-1] check yet drop the per-camera rows the CSR API promises.
    from gsplat.cuda._wrapper import fully_fused_projection_2dgs

    W, H = 64, 64
    means = torch.zeros(0, 3, device=device, requires_grad=True)
    quats = torch.zeros(0, 4, device=device, requires_grad=True)
    scales = torch.ones(0, 3, device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device)[None].expand(C, 4, 4).contiguous()
    Ks = (
        torch.tensor(
            [[float(W), 0.0, W / 2.0], [0.0, float(W), H / 2.0], [0.0, 0.0, 1.0]],
            device=device,
        )[None]
        .expand(C, 3, 3)
        .contiguous()
    )

    (
        batch_ids,
        camera_ids,
        gaussian_ids,
        indptr,
        radii,
        means2d,
        depths,
        ray_transforms,
        normals,
    ) = fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, W, H, packed=True
    )

    assert gaussian_ids.numel() == 0
    assert means2d.shape[0] == 0
    assert radii.shape[0] == 0
    # B == 1 for means.shape == (0, 3), so indptr has B * C + 1 == C + 1 zero
    # entries -- one row per camera plus the trailing total.
    assert indptr.shape == (C + 1,)
    assert bool((indptr == 0).all())

    # Backward over the empty projection must not divide by zero in the launcher.
    (means2d.sum() + depths.sum() + ray_transforms.sum() + normals.sum()).backward()
    assert means.grad.shape == means.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_fully_fused_projection_packed_grid_y_limit():
    # The packed forward launcher maps B*C onto grid.y, which CUDA caps at
    # 65535. A larger B*C must raise a clear error instead of issuing an invalid
    # kernel launch. Here B=1 and C=65536 (one camera over the cap).
    from gsplat.cuda._wrapper import fully_fused_projection

    W, H = 32, 32
    N, C = 1, 65536
    means = torch.randn(N, 3, device=device)
    means[:, 2] += 5.0
    quats = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.rand(N, 3, device=device) * 0.1
    viewmats = torch.eye(4, device=device).expand(C, 4, 4).contiguous()
    Ks = (
        torch.tensor(
            [[float(W), 0.0, W / 2.0], [0.0, float(W), H / 2.0], [0.0, 0.0, 1.0]],
            device=device,
        )
        .expand(C, 3, 3)
        .contiguous()
    )

    with pytest.raises(RuntimeError, match=r"exceeds the CUDA grid\.y limit"):
        fully_fused_projection(
            means, None, quats, scales, viewmats, Ks, W, H, packed=True
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT/Lidar support isn't built in")
@pytest.mark.parametrize("tile_width,tile_height", [(2, 2), (4, 4), (8, 4)])
def test_isect_offset_power_of_two_tiles(tile_width: int, tile_height: int):
    # n_tiles = tile_width*tile_height is a power of two here, the case where the
    # packed (image, tile) id field width disagrees between bits_for_count
    # (bit_width(n-1)) and the old floor(log2)+1 (off by one bit). A pack/unpack
    # mismatch corrupts the decoded tile ids and the offset buffer, so the CUDA
    # packing, the CUDA offset decode, and the Python reference must all agree
    # at these counts.
    from gsplat.cuda._torch_impl import _isect_offset_encode, _isect_tiles
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    torch.manual_seed(42)
    # C >= 3 so the image id occupies bits above the tile field: a wrong
    # tile_n_bits in the offset decode then mis-splits image vs tile bits. With
    # C <= 2 the off-by-one bit folds back into the flat (image*n_tiles+tile)
    # index losslessly and would hide the bug.
    C, N = 4, 800
    tile_size = 16
    width = tile_width * tile_size
    height = tile_height * tile_size
    I = C
    n_tiles = tile_width * tile_height
    assert n_tiles & (n_tiles - 1) == 0, "n_tiles must be a power of two for this test"

    means2d = torch.randn(C, N, 2, device=device) * width
    radii = torch.randint(0, tile_size, (C, N, 2), device=device, dtype=torch.int32)
    depths = torch.rand(C, N, device=device)

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
def test_rasterize_to_indices_in_range_empty():
    # rasterize_to_indices_3dgs computed the image count as numel/(2*N) before
    # any emptiness guard; with zero gaussians (N == 0) the divisor is zero -- a
    # host-side integer division by zero. An empty gaussian set must be a clean
    # no-op returning empty index tensors.
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range

    C = 1
    W = H = 16
    tile_size = 16
    means2d = torch.zeros(C, 0, 2, device=device)
    conics = torch.zeros(C, 0, 3, device=device)
    opacities = torch.zeros(C, 0, device=device)
    transmittances = torch.ones(C, H, W, device=device)
    isect_offsets = torch.zeros(C, 1, 1, dtype=torch.int32, device=device)
    flatten_ids = torch.empty(0, dtype=torch.int32, device=device)

    gauss_ids, pixel_ids, image_ids = rasterize_to_indices_in_range(
        0,
        2**30,
        transmittances,
        means2d,
        conics,
        opacities,
        W,
        H,
        tile_size,
        isect_offsets,
        flatten_ids,
    )
    assert gauss_ids.numel() == 0
    assert pixel_ids.numel() == 0
    assert image_ids.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_rasterize_to_pixels_eval3d_empty():
    # The world-space rasterizer computed the batch count as numel/(N*3) before
    # any emptiness guard; with zero gaussians (N == 0) the divisor is zero -- a
    # host-side integer division by zero. An empty gaussian set must be a clean
    # no-op rather than crashing.
    from gsplat.cuda._wrapper import rasterize_to_pixels_eval3d_extra

    C = 1
    W = H = 16
    tile_size = 16
    channels = 3
    means = torch.zeros(0, 3, device=device)
    quats = torch.zeros(0, 4, device=device)
    scales = torch.ones(0, 3, device=device)
    colors = torch.zeros(C, 0, channels, device=device)
    opacities = torch.zeros(C, 0, device=device)
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor(
        [[[float(W), 0.0, W / 2.0], [0.0, float(H), H / 2.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    isect_offsets = torch.zeros(C, 1, 1, dtype=torch.int32, device=device)
    flatten_ids = torch.empty(0, dtype=torch.int32, device=device)

    # Must not crash (pre-fix: host-side division by zero on the empty input),
    # and every output must hold its empty/safe-default value instead of
    # uninitialized at::empty memory.
    (
        render_colors,
        render_alphas,
        last_ids,
        sample_counts,
        render_normals,
    ) = rasterize_to_pixels_eval3d_extra(
        means,
        quats,
        scales,
        colors,
        opacities,
        viewmats,
        Ks,
        W,
        H,
        tile_size,
        isect_offsets,
        flatten_ids,
        return_sample_counts=True,
        return_normals=True,
    )
    assert torch.isfinite(render_colors).all()
    torch.testing.assert_close(render_colors, torch.zeros_like(render_colors))
    torch.testing.assert_close(render_alphas, torch.zeros_like(render_alphas))
    torch.testing.assert_close(render_normals, torch.zeros_like(render_normals))
    torch.testing.assert_close(last_ids, torch.full_like(last_ids, -1))
    torch.testing.assert_close(sample_counts, torch.zeros_like(sample_counts))
