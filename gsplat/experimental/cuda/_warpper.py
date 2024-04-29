from typing import Callable, Tuple

import torch
from torch import Tensor


def quat_scale_to_covar_perci(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
    compute_covar: bool = True,
    compute_perci: bool = True,
    triu: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Converts quaternions and scales to covariance and precision matrices.

    Args:
        quats: Normalized quaternions. [N, 4]
        scales: Scales. [N, 3]
        compute_covar: Whether to compute covariance matrices. Default: True. If False,
            the returned covariance matrices will be None.
        compute_perci: Whether to compute precision matrices. Default: True. If False,
            the returned precision matrices will be None.
        triu: If True, the return matrices will be upper triangular. Default: False.

    Returns:
        A tuple of:
        - Covariance matrices. If `triu` is True the returned shape is [N, 6], otherwise [N, 3, 3].
        - Precision matrices. If `triu` is True the returned shape is [N, 6], otherwise [N, 3, 3].
    """
    assert quats.dim() == 2 and quats.size(1) == 4, quats.size()
    assert scales.dim() == 2 and scales.size(1) == 3, scales.size()
    quats = quats.contiguous()
    scales = scales.contiguous()
    covars, percis = _QuatScaleToCovarPerci.apply(
        quats, scales, compute_covar, compute_perci, triu
    )
    return covars if compute_covar else None, percis if compute_perci else None


def persp_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """Perspective projection on Gaussians.

    Args:
        means: Gaussian means. [C, N, 3]
        covars: Gaussian covariances. [C, N, 3, 3]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.

    Returns:
        A tuple of:
        - Projected means. [C, N, 2]
        - Projected covariances. [C, N, 2, 2]
    """
    C, N, _ = means.shape
    assert means.shape == (C, N, 3), means.size()
    assert covars.shape == (C, N, 3, 3), covars.size()
    assert Ks.shape == (C, 3, 3), Ks.size()
    means = means.contiguous()
    covars = covars.contiguous()
    Ks = Ks.contiguous()
    return _PerspProj.apply(means, covars, Ks, width, height)


def world_to_cam(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """Transforms Gaussians from world to camera space.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances. [N, 3, 3]
        viewmats: Camera-to-world matrices. [C, 4, 4]

    Returns:
        A tuple of:
        - Gaussian means in camera space. [C, N, 3]
        - Gaussian covariances in camera space. [C, N, 3, 3]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert covars.size() == (N, 3, 3), covars.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    means = means.contiguous()
    covars = covars.contiguous()
    viewmats = viewmats.contiguous()
    return _WorldToCam.apply(means, covars, viewmats)


def projection(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 6]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D.

    Note:
        During projection, we ignore the Gaussians that are outside of the camera frustum.
        So not all the elements in the output tensors are valid. `Radii` could serve as
        an indicator, in which zero radii means the corresponding elements are invalid in
        the output tensors and will be ignored in the next rasterization process.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances (flattened upper triangle). [N, 6]
        viewmats: Camera-to-world matrices. [C, 4, 4]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.
        eps2d: A epsilon added to the projected covariance for numerical stability. Default: 0.3.
        near_plane: Near plane distance. Default: 0.01.

    Returns:
        A tuple of:
        - Radii. The maximum radius of the projected Gaussians in pixel unit.
            Int32 tensor of shape [C, N].
        - Projected means. [C, N, 2]
        - Depths. The z-depth of the projected Gaussians. [C, N]
        - Conics. Inverse of the projected covariances. Return the flattend upper
            triangle with [C, N, 3]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert covars.size() == (N, 6), covars.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    assert Ks.size() == (C, 3, 3), Ks.size()
    means = means.contiguous()
    covars = covars.contiguous()
    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()
    return _Projection.apply(
        means, covars, viewmats, Ks, width, height, eps2d, near_plane
    )


@torch.no_grad()
def isect_tiles(
    means2d: Tensor,  # [C, N, 2]
    radii: Tensor,  # [C, N]
    depths: Tensor,  # [C, N]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles.

    Args:
        means2d: Projected Gaussian means. [C, N, 2]
        radii: Maximum radii of the projected Gaussians. [C, N]
        depths: Z-depth of the projected Gaussians. [C, N]
        tile_size: Tile size.
        tile_width: Tile width.
        tile_height: Tile height.
        sort: If True, the returned intersections will be sorted by the intersection
            ids. Default: True.

    Returns:
        A tuple of:
        - Tiles per Gaussian. The number of tiles intersected by each Gaussian. Int32 [C, N]
        - Intersection ids. Each id is an 64-bit integer with the following
            information: camera_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
            Xc and Xt are the maximum number of bits required to represent the camera
            and tile ids, respectively. Int64 [n_isects]
        - Gaussian ids of the intersections. Int32 [n_isects]
    """
    C, N, _ = means2d.shape
    assert means2d.shape == (C, N, 2), means2d.size()
    assert radii.shape == (C, N), radii.size()
    assert depths.shape == (C, N), depths.size()
    tiles_per_gauss, isect_ids, gauss_ids = _make_lazy_cuda_func("isect_tiles")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        tile_size,
        tile_width,
        tile_height,
        sort,
    )
    return tiles_per_gauss, isect_ids, gauss_ids


@torch.no_grad()
def isect_offset_encode(
    isect_ids: Tensor, n_cameras: int, tile_width: int, tile_height: int
) -> Tensor:
    """Encodes intersection ids to offsets.

    Args:
        isect_ids: Intersection ids. [n_isects]
        n_cameras: Number of cameras.
        tile_width: Tile width.
        tile_height: Tile height.

    Returns:
        Offsets. [C, tile_height, tile_width]
    """
    return _make_lazy_cuda_func("isect_offset_encode")(
        isect_ids.contiguous(), n_cameras, tile_width, tile_height
    )


def rasterize_to_pixels(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    colors: Tensor,  # [C, N, channels]
    opacities: Tensor,  # [N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    gauss_ids: Tensor,  # [n_isects]
) -> Tuple[Tensor, Tensor]:
    C, N, _ = means2d.shape
    assert conics.shape == (C, N, 3), conics.shape
    assert colors.shape[:2] == (C, N), colors.shape
    assert opacities.shape == (N,), opacities.shape
    assert isect_offsets.shape[0] == C, isect_offsets.shape

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[2]
    if channels > 512 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [colors, torch.empty(C, N, padded_channels, device=colors.device)], dim=-1
        )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    render_colors, render_alphas = _RasterizeToPixels.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        gauss_ids.contiguous(),
    )
    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]
    return render_colors, render_alphas


class _QuatScaleToCovarPerci(torch.autograd.Function):
    """Converts quaternions and scales to covariance and precision matrices."""

    @staticmethod
    def forward(
        ctx,
        quats: Tensor,  # [N, 4],
        scales: Tensor,  # [N, 3],
        compute_covar: bool = True,
        compute_perci: bool = True,
        triu: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        covars, percis = _make_lazy_cuda_func("quat_scale_to_covar_perci_fwd")(
            quats, scales, compute_covar, compute_perci, triu
        )
        ctx.save_for_backward(quats, scales)
        ctx.compute_covar = compute_covar
        ctx.compute_perci = compute_perci
        ctx.triu = triu
        return covars, percis

    @staticmethod
    def backward(ctx, v_covars: Tensor, v_percis: Tensor):
        quats, scales = ctx.saved_tensors
        compute_covar = ctx.compute_covar
        compute_perci = ctx.compute_perci
        triu = ctx.triu
        v_quats, v_scales = _make_lazy_cuda_func("quat_scale_to_covar_perci_bwd")(
            quats,
            scales,
            v_covars.contiguous() if compute_covar else None,
            v_percis.contiguous() if compute_perci else None,
            triu,
        )
        return v_quats, v_scales, None, None, None


class _PerspProj(torch.autograd.Function):
    """Perspective projection on Gaussians."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [C, N, 3]
        covars: Tensor,  # [C, N, 3, 3]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
    ) -> Tuple[Tensor, Tensor]:
        means2d, covars2d = _make_lazy_cuda_func("persp_proj_fwd")(
            means, covars, Ks, width, height
        )
        ctx.save_for_backward(means, covars, Ks)
        ctx.width = width
        ctx.height = height
        return means2d, covars2d

    @staticmethod
    def backward(ctx, v_means2d: Tensor, v_covars2d: Tensor):
        means, covars, Ks = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        v_means, v_covars = _make_lazy_cuda_func("persp_proj_bwd")(
            means,
            covars,
            Ks,
            width,
            height,
            v_means2d.contiguous(),
            v_covars2d.contiguous(),
        )
        return v_means, v_covars, None, None, None


class _WorldToCam(torch.autograd.Function):
    """Transforms Gaussians from world to camera space."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 3, 3]
        viewmats: Tensor,  # [C, 4, 4]
    ) -> Tuple[Tensor, Tensor]:
        means_c, covars_c = _make_lazy_cuda_func("world_to_cam_fwd")(
            means, covars, viewmats
        )
        ctx.save_for_backward(means, covars, viewmats)
        return means_c, covars_c

    @staticmethod
    def backward(ctx, v_means_c: Tensor, v_covars_c: Tensor):
        means, covars, viewmats = ctx.saved_tensors
        v_means, v_covars, v_viewmats = _make_lazy_cuda_func("world_to_cam_bwd")(
            means,
            covars,
            viewmats,
            v_means_c.contiguous(),
            v_covars_c.contiguous(),
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_viewmats = None
        return v_means, v_covars, v_viewmats


class _Projection(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6]
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        radii, means2d, depths, conics = _make_lazy_cuda_func("projection_fwd")(
            means, covars, viewmats, Ks, width, height, eps2d, near_plane
        )
        ctx.save_for_backward(means, covars, viewmats, Ks, radii, conics)
        ctx.width = width
        ctx.height = height

        return radii, means2d, depths, conics

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_conics):
        means, covars, viewmats, Ks, radii, conics = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        v_means, v_covars, v_viewmats = _make_lazy_cuda_func("projection_bwd")(
            means,
            covars,
            viewmats,
            Ks,
            width,
            height,
            radii,
            conics,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            ctx.needs_input_grad[2],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_viewmats = None
        return v_means, v_covars, v_viewmats, None, None, None, None, None


class _RasterizeToPixels(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [C, N, 2]
        conics: Tensor,  # [C, N, 3]
        colors: Tensor,  # [C, N, 3]
        opacities: Tensor,  # [N]
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        gauss_ids: Tensor,  # [n_isects]
    ) -> Tuple[Tensor, Tensor]:
        render_colors, render_alphas, last_ids = _make_lazy_cuda_func(
            "rasterize_to_pixels_fwd"
        )(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            gauss_ids,
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            isect_offsets,
            gauss_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size

        # double to float
        render_alphas = render_alphas.float()
        return render_colors, render_alphas

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [C, H, W, 3]
        v_render_alphas: Tensor,  # [C, H, W, 1]
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            isect_offsets,
            gauss_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size

        (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            gauss_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
        )

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            None,
            None,
            None,
            None,
            None,
        )


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda
