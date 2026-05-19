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
    assert_close_with_boundary_band,
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
    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=2e-3, atol=2e-3)

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

    # means2d: split by shutter mode.  GLOBAL is smooth in inputs and admits
    # a tight per-element check.  ROLLING does a 10-iter refinement whose
    # convergence basin can shift on a small fraction of
    # Gaussians, producing tail elements with up to ~16% rel-diff.  Use a
    # bounded per-element check with a small fail-rate cap so the tight bulk
    # check still catches systematic bias.  Replaces rtol=0.5, atol=0.05.
    if rolling_shutter == RollingShutterType.GLOBAL:
        torch.testing.assert_close(
            means2d_cuda[sel],
            means2d_torch[sel],
            rtol=2e-3,
            atol=5e-2,
        )
    else:
        _diff = (means2d_cuda[sel] - means2d_torch[sel]).abs()
        _bound = 5e-2 + 2e-3 * means2d_torch[sel].abs()
        _fail = _diff > _bound
        _fr = _fail.float().mean().item()
        # fail_cap = 1.05 x worst observed (0.013145%) -> 0.014%.
        assert _fr <= 1.4e-4, (
            f"UT means2d (rolling): fail-rate {_fr:.4%} > cap 0.014% "
            f"(atol=5e-2, rtol=2e-3, {int(_fail.sum().item())}/{_fail.numel()})"
        )
        # Outlier guard: even admitted outliers must satisfy a per-element
        # bound so a single-element catastrophic bug cannot hide inside the
        # fail-rate budget.  Tightened to 1.05 x worst observed excess
        # (0.66) over old (atol=0.1, rtol=0.2) -> atol=0.07, rtol=0.14.
        _outlier_bound = 0.07 + 0.14 * means2d_torch[sel].abs()
        _outlier_fail = _diff > _outlier_bound
        assert not _outlier_fail.any(), (
            f"UT means2d (rolling): {int(_outlier_fail.sum().item())} elements "
            f"exceed outlier bound (atol=0.07, rtol=0.14); worst diff "
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
        # K = 0.75 (env x 1.05) leaves headroom for both.
        sp_half_spread = 0.75 * sigma_y
        dist_to_int = (y_ref - y_ref.round()).abs()
        boundary_mask = dist_to_int < sp_half_spread

        # Calibration trace -- envelope x 1.05:
        #   - interior assert atol=7e-3, rtol=0.01:
        #       RTX PRO 2000  worst <7e-3  (passes)
        #       RTX PRO 6000  worst <7e-3  (passes after K=0.75)
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
        # observed in-band diff ~0.398 -> atol=0.42.
        _diff_a = (comps_cuda[sel] - comps_torch[sel]).abs()
        _outlier_a = (_diff_a > 0.42) & boundary_mask
        assert not _outlier_a.any(), (
            f"cluster A: {int(_outlier_a.sum().item())} in-band elements "
            f"exceed outlier bound atol=0.42; worst diff "
            f"{_diff_a.max().item():.4e}"
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
    from tests.test_cameras import parse_lidar_camera

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

    # Identify intersecting tiles.
    # NOTE: CDIM=128 needs the smaller tile so the backward's per-block
    # shared memory (tile_size^2 * CDIM * 4 bytes) stays below the device limit;
    # the forward dispatches to a matching (TILE_SIZE=4, CTA_SIZE=16) kernel variant.
    # When we also convert the 3DGS backward rasterizer pass, it will require much
    # less shared memory since we're iterating with PPT=4 and will therefore be able
    # to remove tile_size=4 both here and in the forward rasterizer.
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
    # backward runs; keeping both render buffers live is what made the heavy
    # arms (channels=128, batch_dims=(2,)/(1,2)) peak above the laptop VRAM.
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
    # - v_means2d atol scales with channel count: more channels accumulate
    #   more alpha-compositing terms (channels=3,32 stay under 1e-3;
    #   channels=128 reaches ~1.4e-3).
    # - v_means2d batch_dims*-128 OOMed on RTX PRO 2000, so its envelope is
    #   RTX PRO 6000-only.
    for name, vc, vt, rtol, atol in [
        (
            "v_means2d",
            v_means2d,
            _v_means2d,
            2.5e-4,
            1.6e-3,
        ),  # RTX PRO 2000=OOM, RTX PRO 6000: rtol=2.14e-4 atol=1.47e-3
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
        from tests.test_cameras import parse_lidar_camera

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
    torch.testing.assert_close(
        v_backgrounds_struct * backgrounds_mask.float(),
        _v_backgrounds_struct * backgrounds_mask.float(),
        rtol=0,
        atol=1.6e-3 * _lidar_tol,
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
        # Fwd-state-reuse bwd path: deriving per-chunk starting accumulators
        # from `dot(pix_out_final - pix_out_at_boundary, v_render_c)` introduces
        # subtraction cancellation vs. the old K1's per-Gaussian running dot,
        # pushing the worst case to ~0.0279 on 0.3% of elements. Bumped atol
        # 2.6e-2 -> 3.0e-2 to accept the drift; same magnitude slack as the
        # 5e-3 -> 2.6e-2 bump above.
        #
        # Worst cases observed (release build, FAST_MATH=1):
        #   Mismatched elements: 2511 / 34020 (7.4%)
        #   Greatest absolute difference: 0.02257537841796875 at index (2, 1388, 0) (up to 0.005 allowed)
        #   Greatest relative difference: 0.060736533254384995 at index (0, 1678, 4) (up to 0 allowed)
        torch.testing.assert_close(
            v_rays * rays_mask.float(), _v_rays * rays_mask.float(), rtol=0, atol=3.0e-2
        )


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
@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
@pytest.mark.parametrize("packed", [False, True])
def test_sh(sh_degree: int, batch_dims: Tuple[int, ...], packed: bool):
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from gsplat.cuda._wrapper import spherical_harmonics

    if packed and batch_dims != ():
        pytest.skip("packed inputs are always rank-2 dirs; batch_dims is irrelevant")

    torch.manual_seed(42)

    N = 1000
    K = (4 + 1) ** 2
    coeffs_src = torch.randn(N, K, 3, device=device, requires_grad=True)

    if packed:
        # Mirror the packed call site (rendering.py): a [N, K, 3] source of
        # per-Gaussian coeffs is gathered into [nnz, K, 3] via gaussian_ids
        # (one row per visible (Gaussian, camera) pair), and dirs is [nnz, 3].
        # nnz > N with random ids exercises the duplicate-gaussian regime that
        # the unpacked broadcast path never hits.
        nnz = 3000
        gaussian_ids = torch.randint(0, N, (nnz,), device=device)
        coeffs = coeffs_src[gaussian_ids]  # [nnz, K, 3]
        dirs = torch.randn(nnz, 3, device=device, requires_grad=True)
        expected_colors_shape = (nnz, 3)
    else:
        coeffs = coeffs_src
        dirs = torch.randn(*batch_dims, N, 3, device=device, requires_grad=True)
        expected_colors_shape = (*batch_dims, N, 3)

    colors = spherical_harmonics(sh_degree, dirs, coeffs)
    _colors = _spherical_harmonics(sh_degree, dirs, coeffs)
    assert colors.shape == expected_colors_shape, colors.shape
    torch.testing.assert_close(colors, _colors, rtol=1e-4, atol=1e-4)

    v_colors = torch.randn_like(colors)

    # Take grads w.r.t. coeffs_src (the [N, K, 3] leaf) so packed mode also
    # exercises the gather VJP that accumulates duplicate-id rows back to source.
    v_coeffs_src, v_dirs = torch.autograd.grad(
        (colors * v_colors).sum(),
        (coeffs_src, dirs),
        retain_graph=True,
        allow_unused=True,
    )
    _v_coeffs_src, _v_dirs = torch.autograd.grad(
        (_colors * v_colors).sum(),
        (coeffs_src, dirs),
        retain_graph=True,
        allow_unused=True,
    )
    assert v_coeffs_src.shape == (N, K, 3), v_coeffs_src.shape
    torch.testing.assert_close(v_coeffs_src, _v_coeffs_src, rtol=1e-4, atol=1e-4)
    if sh_degree > 0:
        assert v_dirs.shape == dirs.shape, v_dirs.shape
        torch.testing.assert_close(v_dirs, _v_dirs, rtol=1e-4, atol=1e-4)


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
