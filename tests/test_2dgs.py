import math

import pytest
import torch

from tests.test_basic import expand

device = torch.device("cuda:0")


@pytest.fixture
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
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
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_projection_2dgs(test_data, batch_dims: tuple[int]):
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

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=6e-2, atol=1e-3)
    torch.testing.assert_close(v_quats, _v_quats, rtol=2e-1, atol=1e-2)
    torch.testing.assert_close(
        v_scales[..., :2], _v_scales[..., :2], rtol=1e-1, atol=2e-1
    )
    torch.testing.assert_close(v_means, _v_means, rtol=1e-2, atol=6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("sparse_grad", [False])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_fully_fused_projection_packed_2dgs(
    test_data, sparse_grad: bool, batch_dims: tuple[int]
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

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_scales, _v_scales, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_quats, _v_quats, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [3, 31])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_rasterize_to_pixels_2dgs(test_data, channels: int, batch_dims: tuple[int]):
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

    (
        render_colors,
        render_alphas,
        render_normals,
        _,
        _,
    ) = rasterize_to_pixels_2dgs(
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

    # assert close backward
    torch.testing.assert_close(v_means2d, _v_means2d, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        v_ray_transforms, _v_ray_transforms, rtol=2e-1, atol=5e-2
    )
    torch.testing.assert_close(v_colors, _v_colors, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_opacities, _v_opacities, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_backgrounds, _v_backgrounds, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_normals, _v_normals, rtol=1e-3, atol=1e-3)
