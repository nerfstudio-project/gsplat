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

import math

import pytest
import torch
from typing_extensions import Tuple
import gsplat
from gsplat._helper import (
    assert_close_with_boundary_band,
    assert_grad_sparsity,
)

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
    C = 3
    N = 1000
    means = torch.randn(N, 3, device=device)
    quats = torch.randn(N, 4, device=device)
    quats = torch.nn.functional.normalize(quats, dim=-1)
    scales = torch.ones(N, 3, device=device)
    scales[..., :2] *= 0.1
    opacities = torch.rand(C, N, device=device) * 0.5
    colors = torch.rand(C, N, 3, device=device)
    viewmats = torch.broadcast_to(torch.eye(4, device=device), (C, 4, 4))
    # W, H = 24, 20
    W, H = 640, 480
    fx, fy, cx, cy = W, W, W // 2, H // 2
    Ks = torch.broadcast_to(
        torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device),
        (C, 3, 3),
    )
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": W,
        "height": H,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_2dgs(), reason="2DGS support wasn't built")
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_projection_2dgs(test_data, batch_dims: Tuple[int, ...]):
    from gsplat.cuda._torch_impl_2dgs import _fully_fused_projection_2dgs
    from gsplat.cuda._wrapper import fully_fused_projection_2dgs

    torch.manual_seed(42)

    test_data = expand(test_data, batch_dims)
    Ks = test_data["Ks"]  # [..., C, 3, 3]
    viewmats = test_data["viewmats"]  # [..., C, 4, 4]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]  # [..., N, 4]
    scales = test_data["scales"]  # [..., N, 3]
    means = test_data["means"]  # [..., N, 3]
    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True

    # forward
    _radii, _means2d, _depths, _ray_transforms, _normals = _fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, width, height
    )

    radii, means2d, depths, ray_transforms, normals = fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, width, height
    )

    # TODO (WZ): is the following true for 2dgs as while?
    # radii is integer so we allow for 1 unit difference
    valid = ((radii > 0) & (_radii > 0)).all(dim=-1)
    torch.testing.assert_close(radii, _radii, rtol=1e-3, atol=1)
    torch.testing.assert_close(means2d[valid], _means2d[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(depths[valid], _depths[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        ray_transforms[valid], _ray_transforms[valid], rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(normals[valid], _normals[valid], rtol=1e-4, atol=1e-4)

    # backward
    v_means2d = torch.randn_like(means2d) * valid[..., None]
    v_depths = torch.randn_like(depths) * valid
    v_ray_transforms = torch.randn_like(ray_transforms) * valid[..., None, None]
    v_normals = torch.randn_like(normals) * valid[..., None]

    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d).sum()
        + (depths * v_depths).sum()
        + (ray_transforms * v_ray_transforms).sum()
        + (normals * v_normals).sum(),
        (viewmats, quats, scales, means),
    )
    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_ray_transforms * v_ray_transforms).sum()
        + (_normals * v_normals).sum(),
        (viewmats, quats, scales, means),
    )

    # Per-element band+sparsity check on the conditioning-sensitive gradients
    # (2DGS projection backward; rtol blowup at near-zero gradient values).
    # Note: v_scales is sliced to [..., :2] because 2DGS uses 2D scales for
    # the on-plane covariance; the third axis is degenerate.
    # interior_rtol = 1.05 x worst observed required rtol per gradient
    # (msg suffix "(2dgs proj)").
    for name, vc, vt, rtol in [
        ("v_viewmats", v_viewmats, _v_viewmats, 1.9e-5),  # rtol_req=1.83e-5
        ("v_quats", v_quats, _v_quats, 1e-5),  # rtol_req=0
        (
            "v_scales[:2]",
            v_scales[..., :2],
            _v_scales[..., :2],
            1.9e-5,
        ),  # rtol_req=1.83e-5
        ("v_means", v_means, _v_means, 1.6e-5),  # rtol_req=1.52e-5
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
            interior_atol=nz_thresh,
            interior_rtol=rtol,
            boundary_max_flip_ratio=1e-3,
            boundary_symmetry_tol=0.5,
            flip_predicate=lambda a, e, _t=nz_thresh: a.abs() > 10 * _t,
            msg=f"{name} backward (2dgs proj)",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_2dgs(), reason="2DGS support wasn't built")
@pytest.mark.parametrize("sparse_grad", [False])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_fully_fused_projection_packed_2dgs(
    test_data, sparse_grad: bool, batch_dims: Tuple[int, ...]
):
    from gsplat.cuda._wrapper import fully_fused_projection_2dgs

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
        means,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        packed=True,
        sparse_grad=sparse_grad,
    )

    _radii, _means2d, _depths, _ray_transforms, _normals = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        packed=False,
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
    __ray_transforms = torch.sparse_coo_tensor(
        torch.stack([batch_ids, camera_ids, gaussian_ids]),
        ray_transforms,
        (B, C, N, 3, 3),
    ).to_dense()
    __ray_transforms = __ray_transforms.reshape(batch_dims + (C, N, 3, 3))
    __normals = torch.sparse_coo_tensor(
        torch.stack([batch_ids, camera_ids, gaussian_ids]), normals, (B, C, N, 3)
    ).to_dense()
    __normals = __normals.reshape(batch_dims + (C, N, 3))

    sel = ((__radii > 0) & (_radii > 0)).all(dim=-1)
    torch.testing.assert_close(__radii[sel], _radii[sel], rtol=0, atol=1)
    torch.testing.assert_close(__means2d[sel], _means2d[sel], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(__depths[sel], _depths[sel], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        __ray_transforms[sel], _ray_transforms[sel], rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(__normals[sel], _normals[sel], rtol=1e-4, atol=1e-4)

    # backward
    v_means2d = torch.randn_like(_means2d) * sel[..., None]
    v_depths = torch.randn_like(_depths) * sel
    v_ray_transforms = torch.randn_like(_ray_transforms) * sel[..., None, None]
    v_normals = torch.randn_like(_normals) * sel[..., None]
    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_ray_transforms * v_ray_transforms).sum()
        + (_normals * v_normals).sum(),
        (viewmats, quats, scales, means),
        retain_graph=True,
    )
    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d[(__radii > 0).all(dim=-1)]).sum()
        + (depths * v_depths[(__radii > 0).all(dim=-1)]).sum()
        + (ray_transforms * v_ray_transforms[(__radii > 0).all(dim=-1)]).sum()
        + (normals * v_normals[(__radii > 0).all(dim=-1)]).sum(),
        (viewmats, quats, scales, means),
        retain_graph=True,
    )
    if sparse_grad:
        v_quats = v_quats.to_dense()
        v_scales = v_scales.to_dense()
        v_means = v_means.to_dense()

    # Per-element band+sparsity check (2DGS packed projection backward).
    # interior_rtol = 1.05 x worst required (probed; all 4 grads rtol_req=0).
    for name, vc, vt, rtol in [
        ("v_viewmats", v_viewmats, _v_viewmats, 1e-5),
        ("v_quats", v_quats, _v_quats, 1e-5),
        ("v_scales", v_scales, _v_scales, 1e-5),
        ("v_means", v_means, _v_means, 1e-5),
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
            interior_atol=nz_thresh,
            interior_rtol=rtol,
            boundary_max_flip_ratio=1e-3,
            boundary_symmetry_tol=0.5,
            flip_predicate=lambda a, e, _t=nz_thresh: a.abs() > 10 * _t,
            msg=f"{name} backward (2dgs packed)",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_2dgs(), reason="2DGS support wasn't built")
@pytest.mark.parametrize("channels", [3, 31])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_rasterize_to_pixels_2dgs(
    test_data, channels: int, batch_dims: Tuple[int, ...]
):
    from gsplat.cuda._torch_impl_2dgs import _rasterize_to_pixels_2dgs
    from gsplat.cuda._wrapper import (
        fully_fused_projection_2dgs,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels_2dgs,
    )

    torch.manual_seed(42)

    N = test_data["means"].shape[-2]
    C = test_data["viewmats"].shape[-3]
    I = math.prod(batch_dims) * C
    test_data.update(
        {
            "colors": torch.rand(C, N, channels, device=device),
            "backgrounds": torch.rand(C, channels, device=device),
        }
    )

    test_data = expand(test_data, batch_dims)
    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]
    opacities = test_data["opacities"]
    colors = test_data["colors"]
    backgrounds = test_data["backgrounds"]

    radii, means2d, depths, ray_transforms, normals = fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, width, height
    )
    colors = torch.cat([colors, depths[..., None]], dim=-1)
    backgrounds = torch.zeros(batch_dims + (C, channels + 1), device=device)

    # Identify intersecting tiles
    tile_size = 16
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))
    densify = torch.zeros_like(means2d, device=means2d.device)

    means2d.requires_grad = True
    ray_transforms.requires_grad = True
    colors.requires_grad = True
    opacities.requires_grad = True
    backgrounds.requires_grad = True
    normals.requires_grad = True
    densify.requires_grad = True

    (render_colors, render_alphas, render_normals, _, _,) = rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        distloss=True,
    )

    _render_colors, _render_alphas, _render_normals = _rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        colors,
        normals,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )

    v_render_colors = torch.rand_like(render_colors)
    v_render_alphas = torch.rand_like(render_alphas)
    v_render_normals = torch.rand_like(render_normals)

    (
        v_means2d,
        v_ray_transforms,
        v_colors,
        v_opacities,
        v_backgrounds,
        v_normals,
    ) = torch.autograd.grad(
        (render_colors * v_render_colors).sum()
        + (render_alphas * v_render_alphas).sum()
        + (render_normals * v_render_normals).sum(),
        (means2d, ray_transforms, colors, opacities, backgrounds, normals),
    )

    (
        _v_means2d,
        _v_ray_transforms,
        _v_colors,
        _v_opacities,
        _v_backgrounds,
        _v_normals,
    ) = torch.autograd.grad(
        (_render_colors * v_render_colors).sum()
        + (_render_alphas * v_render_alphas).sum()
        + (_render_normals * v_render_normals).sum(),
        (means2d, ray_transforms, colors, opacities, backgrounds, normals),
    )

    # assert close forward
    torch.testing.assert_close(render_colors, _render_colors, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(render_alphas, _render_alphas, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(render_normals, _render_normals, atol=1e-3, rtol=1e-3)

    # assert close backward.  Per-output band+sparsity check; v_ray_transforms
    # used to be rtol=2e-1 (200x looser than the others) due to FP-noisy 3x3
    # matrix backward; the band absorbs that noise without admitting bias.
    # interior_rtol = 1.05 x worst observed required rtol per gradient
    # (msg suffix "(2dgs raster)").
    for name, vc, vt, rtol in [
        ("v_means2d", v_means2d, _v_means2d, 1e-4),  # rtol_req=8.24e-5
        (
            "v_ray_transforms",
            v_ray_transforms,
            _v_ray_transforms,
            2.5e-5,
        ),  # rtol_req=2.34e-5 (RTX PRO 6000)
        ("v_colors", v_colors, _v_colors, 3e-5),  # rtol_req=2.77e-5
        ("v_opacities", v_opacities, _v_opacities, 1e-5),  # rtol_req=0
        ("v_backgrounds", v_backgrounds, _v_backgrounds, 1e-5),  # rtol_req=0
        ("v_normals", v_normals, _v_normals, 2.7e-5),  # rtol_req=2.48e-5
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
            interior_atol=nz_thresh,
            interior_rtol=rtol,
            boundary_max_flip_ratio=1e-3,
            boundary_symmetry_tol=0.5,
            flip_predicate=lambda a, e, _t=nz_thresh: a.abs() > 10 * _t,
            msg=f"{name} backward (2dgs raster)",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_2dgs(), reason="2DGS support wasn't built")
@pytest.mark.parametrize("batch_dims", [(2,), (1, 2)])
def test_rasterization_packed_2dgs(test_data, batch_dims: Tuple[int, ...]):
    from gsplat.rendering import rasterization_2dgs

    torch.manual_seed(42)

    N = test_data["means"].shape[-2]
    C = test_data["viewmats"].shape[-3]
    sh_degree = 3
    K = (sh_degree + 1) ** 2

    test_data = expand(test_data, batch_dims)
    render_colors, *_ = rasterization_2dgs(
        means=test_data["means"].contiguous(),
        quats=test_data["quats"].contiguous(),
        scales=test_data["scales"].contiguous(),
        opacities=torch.rand(batch_dims + (N,), device=device) * 0.5,
        colors=torch.randn(N, K, 3, device=device),
        viewmats=test_data["viewmats"],
        Ks=test_data["Ks"],
        width=test_data["width"],
        height=test_data["height"],
        sh_degree=sh_degree,
        packed=True,
    )
    assert render_colors.shape == batch_dims + (
        C,
        test_data["height"],
        test_data["width"],
        3,
    )
    assert torch.isfinite(render_colors).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_2dgs(), reason="2DGS support wasn't built")
def test_rasterization_packed_2dgs_pose_grad_large_nnz():
    # compute_directions switches to a per-(batch, camera) Python loop when
    #   nnz / (B*C) > 10000 and viewmats.requires_grad
    # (gsplat/rendering.py: _compute_view_dirs_packed). That loop reads
    # indptr.cpu(); the 2DGS call site must thread indptr through or it
    # crashes with AttributeError on None.
    from gsplat.rendering import rasterization_2dgs

    torch.manual_seed(42)

    # N tuned so nnz/(B*C) > 10000: with C=3 default-frustum cameras roughly
    # half the randn-positioned gaussians project, so N=80000 yields nnz ~ 1.2e5.
    N = 80000
    C = 2
    means = torch.randn(N, 3, device=device)
    quats = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.ones(N, 3, device=device)
    scales[..., :2] *= 0.1
    opacities = torch.rand(N, device=device) * 0.5
    sh_degree = 3
    sh_coeffs = torch.randn(N, (sh_degree + 1) ** 2, 3, device=device)

    viewmats = torch.eye(4, device=device).expand(C, 4, 4).clone()
    viewmats.requires_grad = True
    W, H = 640, 480
    Ks = torch.tensor(
        [[W, 0.0, W / 2], [0.0, W, H / 2], [0.0, 0.0, 1.0]], device=device
    ).expand(C, 3, 3)

    render_colors, *_ = rasterization_2dgs(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh_coeffs,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        sh_degree=sh_degree,
        packed=True,
    )
    render_colors.sum().backward()
    assert viewmats.grad is not None
    assert torch.isfinite(viewmats.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_2dgs(), reason="2DGS support wasn't built")
def test_rasterize_to_pixels_2dgs_masked_tile_outputs_initialized():
    # A masked-out tile takes the forward rasterizer's early-return branch,
    # which must initialize ALL of its pixel outputs (the binding allocates
    # them with at::empty, leaving render_alphas / render_normals /
    # render_distort / render_median / last_ids / median_ids uninitialized
    # otherwise). The low-level fwd op exposes last_ids / median_ids directly.
    # Stale allocator contents could coincidentally match these defaults, so
    # this only reliably catches a missing store when uninitialized memory
    # differs from the default.
    from gsplat.cuda._wrapper import _make_lazy_cuda_func

    tile_size = 16
    width = height = tile_size  # single tile
    channels = 3
    N = 1

    means2d = torch.tensor([[[width / 2.0, height / 2.0]]], device=device)  # [1, 1, 2]
    ray_transforms = torch.eye(3, device=device).reshape(1, 1, 3, 3)  # [1, 1, 3, 3]
    colors = torch.ones((1, N, channels), device=device)
    opacities = torch.ones((1, N), device=device)
    normals = torch.zeros((1, N, 3), device=device)
    backgrounds = torch.zeros((1, channels), device=device)
    masks = torch.zeros((1, 1, 1), dtype=torch.bool, device=device)  # tile masked off
    isect_offsets = torch.zeros((1, 1, 1), dtype=torch.int32, device=device)
    flatten_ids = torch.empty((0,), dtype=torch.int32, device=device)

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        last_ids,
        median_ids,
    ) = _make_lazy_cuda_func("rasterize_to_pixels_2dgs_fwd")(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        backgrounds,
        masks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )

    torch.testing.assert_close(
        render_colors, backgrounds.reshape(1, 1, 1, channels).expand_as(render_colors)
    )
    torch.testing.assert_close(render_alphas, torch.zeros_like(render_alphas))
    torch.testing.assert_close(render_normals, torch.zeros_like(render_normals))
    torch.testing.assert_close(render_distort, torch.zeros_like(render_distort))
    torch.testing.assert_close(render_median, torch.zeros_like(render_median))
    torch.testing.assert_close(last_ids, torch.zeros_like(last_ids))
    torch.testing.assert_close(median_ids, torch.zeros_like(median_ids))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_2dgs(), reason="2DGS support wasn't built")
def test_rasterize_to_pixels_2dgs_densify_gradient():
    # The densification gradient is the gradient w.r.t. the `densify` input and
    # equals the depth-scaled ray-transform z-gradients:
    #   v_densify[..., 0] == v_ray_transforms[..., 0, 2] * depth
    #   v_densify[..., 1] == v_ray_transforms[..., 1, 2] * depth
    # The backward must accumulate it race-free (warp-reduce + atomicAdd), the
    # same path as v_ray_transforms. Reading the still-accumulating global
    # v_ray_transforms instead yields a last-writer-wins partial sum that
    # undercounts whenever a gaussian spans multiple tiles, so the identity
    # holds only for the race-free accumulation.
    from gsplat.cuda._wrapper import (
        fully_fused_projection_2dgs,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels_2dgs,
    )

    torch.manual_seed(42)
    C = 1
    N = 1000
    means = torch.randn(N, 3, device=device)
    quats = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.ones(N, 3, device=device)
    scales[..., :2] *= 0.1
    opacities = torch.rand(C, N, device=device) * 0.5
    W, H = 640, 480
    viewmats = torch.broadcast_to(torch.eye(4, device=device), (C, 4, 4))
    Ks = torch.broadcast_to(
        torch.tensor(
            [[float(W), 0.0, W / 2.0], [0.0, float(W), H / 2.0], [0.0, 0.0, 1.0]],
            device=device,
        ),
        (C, 3, 3),
    )
    channels = 3
    colors = torch.rand(C, N, channels, device=device)

    radii, means2d, depths, ray_transforms, normals = fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, W, H
    )
    colors = torch.cat([colors, depths[..., None]], dim=-1)
    backgrounds = torch.zeros(C, channels + 1, device=device)

    tile_size = 16
    tile_width = math.ceil(W / float(tile_size))
    tile_height = math.ceil(H / float(tile_size))
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
    densify = torch.zeros_like(means2d)

    ray_transforms.requires_grad = True
    densify.requires_grad = True

    render_colors, render_alphas, render_normals, _, _ = rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        W,
        H,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        distloss=True,
    )

    v_render_colors = torch.randn_like(render_colors)
    v_densify, v_ray_transforms = torch.autograd.grad(
        (render_colors * v_render_colors).sum(),
        (densify, ray_transforms),
    )

    expected = torch.stack(
        [
            v_ray_transforms[..., 0, 2] * depths,
            v_ray_transforms[..., 1, 2] * depths,
        ],
        dim=-1,
    )

    # The densify gradient is zero for non-contributing gaussians; compare only
    # entries with meaningful magnitude.
    scale = expected.abs().max().item()
    mask = expected.abs() > 1e-4 * scale
    torch.testing.assert_close(
        v_densify[mask], expected[mask], rtol=2e-2, atol=1e-3 * scale
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_2dgs(), reason="2DGS support wasn't built")
def test_fully_fused_projection_packed_2dgs_empty():
    # Packed 2DGS projection computed the batch count as numel/(N*3) before the
    # empty-input guard; with N==0 that divisor is zero (host-side integer
    # division by zero). An empty gaussian set must be a clean no-op returning
    # empty packed tensors.
    from gsplat.cuda._wrapper import fully_fused_projection_2dgs

    W, H = 64, 64
    means = torch.zeros(0, 3, device=device)
    quats = torch.zeros(0, 4, device=device)
    scales = torch.ones(0, 3, device=device)
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor(
        [[float(W), 0.0, W / 2.0], [0.0, float(W), H / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    )[None]

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
    assert int(indptr[-1].item()) == 0
