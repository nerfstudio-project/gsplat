import math
import warnings
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


def world_to_cam(
    means: Tensor,  # [B, N, 3]
    covars: Tensor,  # [B, N, 3, 3]
    viewmats: Tensor,  # [B, C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """Transforms Gaussians from world to camera coordinate system.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances. [N, 3, 3]
        viewmats: World-to-camera transformation matrices. [C, 4, 4]

    Returns:
        A tuple:

        - **Gaussian means in camera coordinate system**. [C, N, 3]
        - **Gaussian covariances in camera coordinate system**. [C, N, 3, 3]
    """
    from ._torch_impl import _world_to_cam

    warnings.warn(
        "world_to_cam() is removed from the CUDA backend as it's relatively easy to "
        "implement in PyTorch. Currently use the PyTorch implementation instead. "
        "This function will be completely removed in a future release.",
        DeprecationWarning,
    )
    B, N, _ = means.shape
    C = viewmats.shape[1]
    assert means.shape == (B, N, 3), means.shape
    assert covars.shape == (B, N, 3, 3), covars.shape
    assert viewmats.shape == (B, C, 4, 4), viewmats.shape
    means = means.contiguous()
    covars = covars.contiguous()
    viewmats = viewmats.contiguous()
    return _world_to_cam(means, covars, viewmats)


def adam(
    param: Tensor,
    param_grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    valid: Tensor,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
) -> None:
    _make_lazy_cuda_func("adam")(
        param, param_grad, exp_avg, exp_avg_sq, valid, lr, b1, b2, eps
    )


def _make_lazy_cuda_obj(name: str) -> Any:
    # pylint: disable=import-outside-toplevel
    from ._backend import _C

    obj = _C
    for name_split in name.split("."):
        obj = getattr(_C, name_split)
    return obj


def spherical_harmonics(
    degrees_to_use: int,
    dirs: Tensor,  # [..., 3]
    coeffs: Tensor,  # [..., K, 3]
    masks: Optional[Tensor] = None,  # [...,]
) -> Tensor:
    """Computes spherical harmonics.

    Args:
        degrees_to_use: The degree to be used.
        dirs: Directions. [..., 3]
        coeffs: Coefficients. [..., K, 3]
        masks: Optional boolen masks to skip some computation. [...,] Default: None.

    Returns:
        Spherical harmonics. [..., 3]
    """
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], coeffs.shape
    batch_dims = dirs.shape[:-1]
    assert dirs.shape == batch_dims + (3,), dirs.shape
    assert (
        (len(coeffs.shape) == len(batch_dims) + 2)
        and coeffs.shape[:-2] == batch_dims
        and coeffs.shape[-1] == 3
    ), coeffs.shape
    if masks is not None:
        assert masks.shape == batch_dims, masks.shape
        masks = masks.contiguous()
    return _SphericalHarmonics.apply(
        degrees_to_use, dirs.contiguous(), coeffs.contiguous(), masks
    )


def quat_scale_to_covar_preci(
    quats: Tensor,  # [..., 4],
    scales: Tensor,  # [..., 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Converts quaternions and scales to covariance and precision matrices.

    Args:
        quats: Quaternions (No need to be normalized). [..., 4]
        scales: Scales. [..., 3]
        compute_covar: Whether to compute covariance matrices. Default: True. If False,
            the returned covariance matrices will be None.
        compute_preci: Whether to compute precision matrices. Default: True. If False,
            the returned precision matrices will be None.
        triu: If True, the return matrices will be upper triangular. Default: False.

    Returns:
        A tuple:

        - **Covariance matrices**. If `triu` is True the returned shape is [..., 6], otherwise [..., 3, 3].
        - **Precision matrices**. If `triu` is True the returned shape is [..., 6], otherwise [..., 3, 3].
    """
    batch_dims = quats.shape[:-1]
    assert quats.shape == batch_dims + (4,), quats.shape
    assert scales.shape == batch_dims + (3,), scales.shape
    quats = quats.contiguous()
    scales = scales.contiguous()
    covars, precis = _QuatScaleToCovarPreci.apply(
        quats, scales, compute_covar, compute_preci, triu
    )
    return covars if compute_covar else None, precis if compute_preci else None


def persp_proj(
    means: Tensor,  # [B, C, N, 3]
    covars: Tensor,  # [B, C, N, 3, 3]
    Ks: Tensor,  # [B, C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """Perspective projection on Gaussians.
    DEPRECATED: please use `proj` with `ortho=False` instead.

    Args:
        means: Gaussian means. [B, C, N, 3]
        covars: Gaussian covariances. [B, C, N, 3, 3]
        Ks: Camera intrinsics. [B, C, 3, 3]
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **Projected means**. [B, C, N, 2]
        - **Projected covariances**. [B, C, N, 2, 2]
    """
    warnings.warn(
        "persp_proj is deprecated and will be removed in a future release. "
        "Use proj with ortho=False instead.",
        DeprecationWarning,
    )
    return proj(means, covars, Ks, width, height, ortho=False)


def proj(
    means: Tensor,  # [..., C, N, 3]
    covars: Tensor,  # [..., C, N, 3, 3]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
) -> Tuple[Tensor, Tensor]:
    """Projection of Gaussians (perspective or orthographic).

    Args:
        means: Gaussian means. [..., C, N, 3]
        covars: Gaussian covariances. [..., C, N, 3, 3]
        Ks: Camera intrinsics. [..., C, 3, 3]
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **Projected means**. [..., C, N, 2]
        - **Projected covariances**. [..., C, N, 2, 2]
    """
    batch_dims = means.shape[:-3]
    C, N = means.shape[-3:-1]
    assert means.shape == batch_dims + (C, N, 3), means.size()
    assert covars.shape == batch_dims + (C, N, 3, 3), covars.size()
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.size()
    means = means.contiguous()
    covars = covars.contiguous()
    Ks = Ks.contiguous()
    return _Proj.apply(means, covars, Ks, width, height, camera_model)


def fully_fused_projection(
    means: Tensor,  # [..., N, 3]
    covars: Optional[Tensor],  # [..., N, 6] or None
    quats: Optional[Tensor],  # [..., N, 4] or None
    scales: Optional[Tensor],  # [..., N, 3] or None
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    opacities: Optional[Tensor] = None,  # [..., N] or None
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D.

    This function fuse the process of computing covariances
    (:func:`quat_scale_to_covar_preci()`), transforming to camera space (:func:`world_to_cam()`),
    and projection (:func:`proj()`).

    .. note::

        During projection, we ignore the Gaussians that are outside of the camera frustum.
        So not all the elements in the output tensors are valid. The output `radii` could serve as
        an indicator, in which zero radii means the corresponding elements are invalid in
        the output tensors and will be ignored in the next rasterization process. If `packed=True`,
        the output tensors will be packed into a flattened tensor, in which all elements are valid.
        In this case, a `camera_ids` tensor and `gaussian_ids` tensor will be returned to indicate the
        row (camera) and column (Gaussian) indices of the packed flattened tensor, which is essentially
        following the COO sparse tensor format.

    .. note::

        This functions supports projecting Gaussians with either covariances or {quaternions, scales},
        which will be converted to covariances internally in a fused CUDA kernel. Either `covars` or
        {`quats`, `scales`} should be provided.

    Args:
        means: Gaussian means. [..., N, 3]
        covars: Gaussian covariances (flattened upper triangle). [..., N, 6] Optional.
        quats: Quaternions (No need to be normalized). [..., N, 4] Optional.
        scales: Scales. [..., N, 3] Optional.
        viewmats: World-to-camera matrices. [..., C, 4, 4]
        Ks: Camera intrinsics. [..., C, 3, 3]
        width: Image width.
        height: Image height.
        eps2d: A epsilon added to the 2D covariance for numerical stability. Default: 0.3.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 1e10.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        packed: If True, the output tensors will be packed into a flattened tensor. Default: False.
        sparse_grad: This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.
        calc_compensations: If True, a view-dependent opacity compensation factor will be computed, which
          is useful for anti-aliasing. Default: False.
        opacities: Gaussian opacities in range [0, 1]. If provided, will use it to compute a tighter bounds.
            [..., N] or None. Default: None.

    Returns:
        A tuple:

        If `packed` is True:

        - **batch_ids**. The row indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **camera_ids**. The row indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **gaussian_ids**. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [nnz, 2].
        - **means**. Projected Gaussian means in 2D. [nnz, 2]
        - **depths**. The z-depth of the projected Gaussians. [nnz]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [nnz, 3]
        - **compensations**. The view-dependent opacity compensation factor. [nnz]

        If `packed` is False:

        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [..., C, N, 2].
        - **means**. Projected Gaussian means in 2D. [..., C, N, 2]
        - **depths**. The z-depth of the projected Gaussians. [..., C, N]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [..., C, N, 3]
        - **compensations**. The view-dependent opacity compensation factor. [..., C, N]
    """
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.size() == batch_dims + (N, 3), means.size()
    assert viewmats.size() == batch_dims + (C, 4, 4), viewmats.size()
    assert Ks.size() == batch_dims + (C, 3, 3), Ks.size()
    means = means.contiguous()
    if covars is not None:
        assert covars.size() == batch_dims + (N, 6), covars.size()
        covars = covars.contiguous()
    else:
        assert quats is not None, "covars or quats is required"
        assert scales is not None, "covars or scales is required"
        assert quats.size() == batch_dims + (N, 4), quats.size()
        assert scales.size() == batch_dims + (N, 3), scales.size()
        quats = quats.contiguous()
        scales = scales.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"
    if opacities is not None:
        assert opacities.size() == batch_dims + (N,), opacities.size()
        opacities = opacities.contiguous()

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()
    if packed:
        return _FullyFusedProjectionPacked.apply(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            sparse_grad,
            calc_compensations,
            camera_model,
            opacities,
        )
    else:
        return _FullyFusedProjection.apply(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model,
            opacities,
        )


@torch.no_grad()
def isect_tiles(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    radii: Tensor,  # [..., N, 2] or [nnz, 2]
    depths: Tensor,  # [..., N] or [nnz]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    segmented: bool = False,
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles.

    Args:
        means2d: Projected Gaussian means. [B, C, N, 2] if packed is False, [nnz, 2] if packed is True.
        radii: Maximum radii of the projected Gaussians. [B, C, N, 2] if packed is False, [nnz, 2] if packed is True.
        depths: Z-depth of the projected Gaussians. [B, C, N] if packed is False, [nnz] if packed is True.
        tile_size: Tile size.
        tile_width: Tile width.
        tile_height: Tile height.
        sort: If True, the returned intersections will be sorted by the intersection ids. Default: True.
        segmented: If True, segmented radix sort will be used to sort the intersections. Default: False.
        packed: If True, the input tensors are packed. Default: False.
        n_images: Number of images. Required if packed is True.
        image_ids: The image indices of the projected Gaussians. Required if packed is True.
        gaussian_ids: The column indices of the projected Gaussians. Required if packed is True.

    Returns:
        A tuple:

        - **Tiles per Gaussian**. The number of tiles intersected by each Gaussian.
          Int32 [..., N] if packed is False, Int32 [nnz] if packed is True.
        - **Intersection ids**. Each id is an 64-bit integer with the following
          information: image_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
          Xc and Xt are the maximum number of bits required to represent the image and
          tile ids, respectively. Int64 [n_isects]
        - **Flatten ids**. The global flatten indices in [I * N] or [nnz] (packed). [n_isects]
    """
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.size()
        assert radii.shape == (nnz, 2), radii.size()
        assert depths.shape == (nnz,), depths.size()
        assert image_ids is not None, "image_ids is required if packed is True"
        assert gaussian_ids is not None, "gaussian_ids is required if packed is True"
        assert n_images is not None, "n_images is required if packed is True"
        image_ids = image_ids.contiguous()
        gaussian_ids = gaussian_ids.contiguous()
        I = n_images

    else:
        image_dims = means2d.shape[:-2]
        I = math.prod(image_dims)
        N = means2d.shape[-2]
        assert means2d.shape == image_dims + (N, 2), means2d.size()
        assert radii.shape == image_dims + (N, 2), radii.size()
        assert depths.shape == image_dims + (N,), depths.size()

    tiles_per_gauss, isect_ids, flatten_ids = _make_lazy_cuda_func("intersect_tile")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        image_ids,
        gaussian_ids,
        I,
        tile_size,
        tile_width,
        tile_height,
        sort,
        segmented,
    )
    return tiles_per_gauss, isect_ids, flatten_ids


@torch.no_grad()
def isect_offset_encode(
    isect_ids: Tensor,
    n_images: int,
    tile_width: int,
    tile_height: int,
) -> Tensor:
    """Encodes intersection ids to offsets.

    Args:
        isect_ids: Intersection ids. [n_isects]
        n_images: Number of images.
        tile_width: Tile width.
        tile_height: Tile height.

    Returns:
        Offsets. [I, tile_height, tile_width]
    """
    return _make_lazy_cuda_func("intersect_offset")(
        isect_ids.contiguous(), n_images, tile_width, tile_height
    )


def rasterize_to_pixels(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    colors: Tensor,  # [..., N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., N] or [nnz]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., channels]
    masks: Optional[Tensor] = None,  # [..., tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [..., N, 3] if packed is False, [nnz, 3] if packed is True.
        colors: Gaussian colors or ND features. [..., N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [..., N] if packed is False, [nnz] if packed is True.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [..., tile_height, tile_width]
        flatten_ids: The global flatten indices in [I * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [..., channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [..., tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**. [..., image_height, image_width, channels]
        - **Rendered alphas**. [..., image_height, image_width, 1]
    """

    image_dims = means2d.shape[:-2]
    channels = colors.shape[-1]
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(-2)
        assert means2d.shape == image_dims + (N, 2), means2d.shape
        assert conics.shape == image_dims + (N, 3), conics.shape
        assert colors.shape == image_dims + (N, channels), colors.shape
        assert opacities.shape == image_dims + (N,), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == image_dims + (channels,), backgrounds.shape
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()

    # Pad the channels to the nearest supported number if necessary
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.zeros(*colors.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[-2:]
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
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
    )

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]
    return render_colors, render_alphas


@torch.no_grad()
def rasterize_to_indices_in_range(
    range_start: int,
    range_end: int,
    transmittances: Tensor,  # [..., image_height, image_width]
    means2d: Tensor,  # [..., N, 2]
    conics: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Rasterizes a batch of Gaussians to images but only returns the indices.

    .. note::

        This function supports iterative rasterization, in which each call of this function
        will rasterize a batch of Gaussians from near to far, defined by `[range_start, range_end)`.
        If a one-step full rasterization is desired, set `range_start` to 0 and `range_end` to a really
        large number, e.g, 1e10.

    Args:
        range_start: The start batch of Gaussians to be rasterized (inclusive).
        range_end: The end batch of Gaussians to be rasterized (exclusive).
        transmittances: Currently transmittances. [..., image_height, image_width]
        means2d: Projected Gaussian means. [..., N, 2]
        conics: Inverse of the projected covariances with only upper triangle values. [..., N, 3]
        opacities: Gaussian opacities that support per-view values. [..., N]
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [..., tile_height, tile_width]
        flatten_ids: The global flatten indices in [I * N] from  `isect_tiles()`. [n_isects]

    Returns:
        A tuple:

        - **Gaussian ids**. Gaussian ids for the pixel intersection. A flattened list of shape [M].
        - **Pixel ids**. pixel indices (row-major). A flattened list of shape [M].
        - **Image ids**. image indices. A flattened list of shape [M].
    """

    image_dims = means2d.shape[:-2]
    tile_height, tile_width = isect_offsets.shape[-2:]
    N = means2d.shape[-2]
    assert conics.shape == image_dims + (N, 3), conics.shape
    assert opacities.shape == image_dims + (N,), opacities.shape
    assert isect_offsets.shape == image_dims + (
        tile_height,
        tile_width,
    ), isect_offsets.shape
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    out_gauss_ids, out_indices = _make_lazy_cuda_func("rasterize_to_indices_3dgs")(
        range_start,
        range_end,
        transmittances.contiguous(),
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )
    out_pixel_ids = out_indices % (image_width * image_height)
    out_image_ids = out_indices // (image_width * image_height)
    return out_gauss_ids, out_pixel_ids, out_image_ids


class _QuatScaleToCovarPreci(torch.autograd.Function):
    """Converts quaternions and scales to covariance and precision matrices."""

    @staticmethod
    def forward(
        ctx,
        quats: Tensor,  # [..., 4],
        scales: Tensor,  # [..., 3],
        compute_covar: bool = True,
        compute_preci: bool = True,
        triu: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        covars, precis = _make_lazy_cuda_func("quat_scale_to_covar_preci_fwd")(
            quats, scales, compute_covar, compute_preci, triu
        )
        ctx.save_for_backward(quats, scales)
        ctx.compute_covar = compute_covar
        ctx.compute_preci = compute_preci
        ctx.triu = triu
        return covars, precis

    @staticmethod
    def backward(ctx, v_covars: Tensor, v_precis: Tensor):
        quats, scales = ctx.saved_tensors
        compute_covar = ctx.compute_covar
        compute_preci = ctx.compute_preci
        triu = ctx.triu
        if compute_covar and v_covars.is_sparse:
            v_covars = v_covars.to_dense()
        if compute_preci and v_precis.is_sparse:
            v_precis = v_precis.to_dense()
        v_quats, v_scales = _make_lazy_cuda_func("quat_scale_to_covar_preci_bwd")(
            quats,
            scales,
            triu,
            v_covars.contiguous() if compute_covar else None,
            v_precis.contiguous() if compute_preci else None,
        )
        return v_quats, v_scales, None, None, None


class _Proj(torch.autograd.Function):
    """Perspective fully_fused_projection on Gaussians."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [..., C, N, 3]
        covars: Tensor,  # [..., C, N, 3, 3]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    ) -> Tuple[Tensor, Tensor]:
        camera_model_type = _make_lazy_cuda_obj(
            f"CameraModelType.{camera_model.upper()}"
        )

        means2d, covars2d = _make_lazy_cuda_func("projection_ewa_simple_fwd")(
            means,
            covars,
            Ks,
            width,
            height,
            camera_model_type,
        )
        ctx.save_for_backward(means, covars, Ks)
        ctx.width = width
        ctx.height = height
        ctx.camera_model_type = camera_model_type
        return means2d, covars2d

    @staticmethod
    def backward(ctx, v_means2d: Tensor, v_covars2d: Tensor):
        means, covars, Ks = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        camera_model_type = ctx.camera_model_type
        v_means, v_covars = _make_lazy_cuda_func("projection_ewa_simple_bwd")(
            means,
            covars,
            Ks,
            width,
            height,
            camera_model_type,
            v_means2d.contiguous(),
            v_covars2d.contiguous(),
        )
        return v_means, v_covars, None, None, None, None


class _FullyFusedProjection(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [..., N, 3]
        covars: Tensor,  # [..., N, 6] or None
        quats: Tensor,  # [..., N, 4] or None
        scales: Tensor,  # [..., N, 3] or None
        viewmats: Tensor,  # [..., C, 4, 4]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        calc_compensations: bool,
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
        opacities: Optional[Tensor] = None,  # [..., N] or None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        camera_model_type = _make_lazy_cuda_obj(
            f"CameraModelType.{camera_model.upper()}"
        )

        # "covars" and {"quats", "scales"} are mutually exclusive
        radii, means2d, depths, conics, compensations = _make_lazy_cuda_func(
            "projection_ewa_3dgs_fused_fwd"
        )(
            means,
            covars,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model_type,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            means, covars, quats, scales, viewmats, Ks, radii, conics, compensations
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.camera_model_type = camera_model_type

        return radii, means2d, depths, conics, compensations

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_conics, v_compensations):
        (
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            conics,
            compensations,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        camera_model_type = ctx.camera_model_type
        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "projection_ewa_3dgs_fused_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            camera_model_type,
            radii,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            ctx.needs_input_grad[4],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_quats = None
        if not ctx.needs_input_grad[3]:
            v_scales = None
        if not ctx.needs_input_grad[4]:
            v_viewmats = None
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _RasterizeToPixels(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [..., N, 2]
        conics: Tensor,  # [..., N, 3]
        colors: Tensor,  # [..., N, D]
        opacities: Tensor,  # [..., N]
        backgrounds: Tensor,  # [..., D], Optional
        masks: Tensor,  # [..., tile_height, tile_width], Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [..., tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        absgrad: bool,
    ) -> Tuple[Tensor, Tensor]:
        render_colors, render_alphas, last_ids = _make_lazy_cuda_func(
            "rasterize_to_pixels_3dgs_fwd"
        )(
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
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad

        # double to float
        render_alphas = render_alphas.float()
        return render_colors, render_alphas

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [..., H, W, 3]
        v_render_alphas: Tensor,  # [..., H, W, 1]
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad

        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(
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
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            absgrad,
        )

        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[4]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(-3, -2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FullyFusedProjectionPacked(torch.autograd.Function):
    """Projects Gaussians to 2D. Return packed tensors."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [..., N, 3]
        covars: Tensor,  # [..., N, 6] or None
        quats: Tensor,  # [..., N, 4] or None
        scales: Tensor,  # [..., N, 3] or None
        viewmats: Tensor,  # [..., C, 4, 4]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        sparse_grad: bool,
        calc_compensations: bool,
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
        opacities: Optional[Tensor] = None,  # [..., N] or None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        camera_model_type = _make_lazy_cuda_obj(
            f"CameraModelType.{camera_model.upper()}"
        )

        (
            indptr,
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = _make_lazy_cuda_func("projection_ewa_3dgs_packed_fwd")(
            means,
            covars,  # optional
            quats,  # optional
            scales,  # optional
            opacities,  # optional
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model_type,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            batch_ids,
            camera_ids,
            gaussian_ids,
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            conics,
            compensations,
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.sparse_grad = sparse_grad
        ctx.camera_model_type = camera_model_type

        return (
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        )

    @staticmethod
    def backward(
        ctx,
        v_batch_ids,
        v_camera_ids,
        v_gaussian_ids,
        v_radii,
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
    ):
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            conics,
            compensations,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        sparse_grad = ctx.sparse_grad
        camera_model_type = ctx.camera_model_type

        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "projection_ewa_3dgs_packed_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            camera_model_type,
            batch_ids,
            camera_ids,
            gaussian_ids,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            ctx.needs_input_grad[4],  # viewmats_requires_grad
            sparse_grad,
        )

        if sparse_grad:
            batch_dims = means.shape[:-2]
            B = math.prod(batch_dims)
            N = means.shape[-2]
        if not ctx.needs_input_grad[0]:
            v_means = None
        else:
            if sparse_grad:
                # TODO: gaussian_ids is duplicated so not ideal.
                # An idea is to directly set the attribute (e.g., .sparse_grad) of
                # the tensor but this requires the tensor to be leaf node only. And
                # a customized optimizer would be needed in this case.
                v_means = torch.sparse_coo_tensor(
                    indices=torch.stack([batch_ids, gaussian_ids]),
                    values=v_means,  # [nnz, 3]
                    size=(B, N, 3),
                    is_coalesced=len(viewmats) == 1,
                )
                v_means = v_means.reshape(batch_dims + (N, 3))
        if not ctx.needs_input_grad[1]:
            v_covars = None
        else:
            if sparse_grad:
                v_covars = torch.sparse_coo_tensor(
                    indices=torch.stack([batch_ids, gaussian_ids]),
                    values=v_covars,  # [nnz, 6]
                    size=(B, N, 6),
                    is_coalesced=len(viewmats) == 1,
                )
                v_covars = v_covars.reshape(batch_dims + (N, 6))
        if not ctx.needs_input_grad[2]:
            v_quats = None
        else:
            if sparse_grad:
                v_quats = torch.sparse_coo_tensor(
                    indices=torch.stack([batch_ids, gaussian_ids]),
                    values=v_quats,  # [nnz, 4]
                    size=(B, N, 4),
                    is_coalesced=len(viewmats) == 1,
                )
                v_quats = v_quats.reshape(batch_dims + (N, 4))
        if not ctx.needs_input_grad[3]:
            v_scales = None
        else:
            if sparse_grad:
                v_scales = torch.sparse_coo_tensor(
                    indices=torch.stack([batch_ids, gaussian_ids]),
                    values=v_scales,  # [nnz, 3]
                    size=(B, N, 3),
                    is_coalesced=len(viewmats) == 1,
                )
                v_scales = v_scales.reshape(batch_dims + (N, 3))
        if not ctx.needs_input_grad[4]:
            v_viewmats = None

        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _SphericalHarmonics(torch.autograd.Function):
    """Spherical Harmonics"""

    @staticmethod
    def forward(
        ctx, sh_degree: int, dirs: Tensor, coeffs: Tensor, masks: Tensor
    ) -> Tensor:
        colors = _make_lazy_cuda_func("spherical_harmonics_fwd")(
            sh_degree, dirs, coeffs, masks
        )
        ctx.save_for_backward(dirs, coeffs, masks)
        ctx.sh_degree = sh_degree
        ctx.num_bases = coeffs.shape[-2]
        return colors

    @staticmethod
    def backward(ctx, v_colors: Tensor):
        dirs, coeffs, masks = ctx.saved_tensors
        sh_degree = ctx.sh_degree
        num_bases = ctx.num_bases
        compute_v_dirs = ctx.needs_input_grad[1]
        v_coeffs, v_dirs = _make_lazy_cuda_func("spherical_harmonics_bwd")(
            num_bases,
            sh_degree,
            dirs,
            coeffs,
            masks,
            v_colors.contiguous(),
            compute_v_dirs,
        )
        if not compute_v_dirs:
            v_dirs = None
        return None, v_dirs, v_coeffs, None


###### 2DGS ######
def fully_fused_projection_2dgs(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Prepare Gaussians for rasterization

    This function prepares ray-splat intersection matrices, computes
    per splat bounding box and 2D means in image space.

    Args:
        means: Gaussian means. [..., N, 3]
        quats: Quaternions (No need to be normalized). [..., N, 4].
        scales: Scales. [..., N, 3].
        viewmats: World-to-camera matrices. [..., C, 4, 4]
        Ks: Camera intrinsics. [..., C, 3, 3]
        width: Image width.
        height: Image height.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 200.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        packed: If True, the output tensors will be packed into a flattened tensor. Default: False.
        sparse_grad (Experimental): This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.

    Returns:
        A tuple:

        If `packed` is True:

        - **camera_ids**. The row indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **gaussian_ids**. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [nnz, 2].
        - **means**. Projected Gaussian means in 2D. [nnz, 2]
        - **depths**. The z-depth of the projected Gaussians. [nnz]
        - **ray_transforms**. transformation matrices that transforms xy-planes in pixel spaces into splat coordinates (WH)^T in equation (9) in paper [nnz, 3, 3]
        - **normals**. The normals in camera spaces. [nnz, 3]

        If `packed` is False:

        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [..., C, N, 2].
        - **means**. Projected Gaussian means in 2D. [..., C, N, 2]
        - **depths**. The z-depth of the projected Gaussians. [..., C, N]
        - **ray_transforms**. transformation matrices that transforms xy-planes in pixel spaces into splat coordinates.
        - **normals**. The normals in camera spaces. [..., C, N, 3]

    """
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    means = means.contiguous()
    assert quats is not None, "quats is required"
    assert scales is not None, "scales is required"
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    quats = quats.contiguous()
    scales = scales.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()
    if packed:
        return _FullyFusedProjectionPacked2DGS.apply(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            near_plane,
            far_plane,
            radius_clip,
            sparse_grad,
        )
    else:
        return _FullyFusedProjection2DGS.apply(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
        )


class _FullyFusedProjection2DGS(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [..., N, 3]
        quats: Tensor,  # [..., N, 4]
        scales: Tensor,  # [..., N, 3]
        viewmats: Tensor,  # [..., C, 4, 4]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        radii, means2d, depths, ray_transforms, normals = _make_lazy_cuda_func(
            "projection_2dgs_fused_fwd"
        )(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
        )
        ctx.save_for_backward(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            ray_transforms,
            normals,
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d

        return radii, means2d, depths, ray_transforms, normals

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_ray_transforms, v_normals):
        (
            means,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            ray_transforms,
            normals,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        v_means, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "projection_2dgs_fused_bwd"
        )(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            radii,
            ray_transforms,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_normals.contiguous(),
            v_ray_transforms.contiguous(),
            ctx.needs_input_grad[3],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_quats = None
        if not ctx.needs_input_grad[2]:
            v_scales = None
        if not ctx.needs_input_grad[3]:
            v_viewmats = None

        return (
            v_means,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FullyFusedProjectionPacked2DGS(torch.autograd.Function):
    """Projects Gaussians to 2D. Return packed tensors."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        quats: Tensor,  # [N, 4]
        scales: Tensor,  # [N, 3]
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        sparse_grad: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (
            indptr,
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        ) = _make_lazy_cuda_func("projection_2dgs_packed_fwd")(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            near_plane,
            far_plane,
            radius_clip,
        )
        ctx.save_for_backward(
            batch_ids,
            camera_ids,
            gaussian_ids,
            means,
            quats,
            scales,
            viewmats,
            Ks,
            ray_transforms,
        )
        ctx.width = width
        ctx.height = height
        ctx.sparse_grad = sparse_grad

        return (
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        )

    @staticmethod
    def backward(
        ctx,
        v_batch_ids,
        v_camera_ids,
        v_gaussian_ids,
        v_radii,
        v_means2d,
        v_depths,
        v_ray_transforms,
        v_normals,
    ):
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            means,
            quats,
            scales,
            viewmats,
            Ks,
            ray_transforms,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        sparse_grad = ctx.sparse_grad

        v_means, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "projection_2dgs_packed_bwd"
        )(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            batch_ids,
            camera_ids,
            gaussian_ids,
            ray_transforms,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_ray_transforms.contiguous(),
            v_normals.contiguous(),
            ctx.needs_input_grad[3],  # viewmats_requires_grad
            sparse_grad,
        )

        if not ctx.needs_input_grad[0]:
            v_means = None
        else:
            if sparse_grad:
                # TODO: gaussian_ids is duplicated so not ideal.
                # An idea is to directly set the attribute (e.g., .sparse_grad) of
                # the tensor but this requires the tensor to be leaf node only. And
                # a customized optimizer would be needed in this case.
                v_means = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_means,  # [nnz, 3]
                    size=means.size(),  # [N, 3]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[1]:
            v_quats = None
        else:
            if sparse_grad:
                v_quats = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_quats,  # [nnz, 4]
                    size=quats.size(),  # [N, 4]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[2]:
            v_scales = None
        else:
            if sparse_grad:
                v_scales = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_scales,  # [nnz, 3]
                    size=scales.size(),  # [N, 3]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[3]:
            v_viewmats = None

        return (
            v_means,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rasterize_to_pixels_2dgs(
    means2d: Tensor,  # [B, C, N, 2]
    ray_transforms: Tensor,  # [B, C, N, 3, 3]
    colors: Tensor,  # [B, C, N, channels]
    opacities: Tensor,  # [B, C, N]
    normals: Tensor,  # [B, C, N, 3]
    densify: Tensor,  # [B, C, N, 2]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [B, C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [B, C, channels]
    masks: Optional[Tensor] = None,  # [B, C, tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterize Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [B, C, N, 2] if packed is False, [nnz, 2] if packed is True.
        ray_transforms: transformation matrices that transforms xy-planes in pixel spaces into splat coordinates. [B, C, N, 3, 3] if packed is False, [nnz, channels] if packed is True.
        colors: Gaussian colors or ND features. [B, C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [B, C, N] if packed is False, [nnz] if packed is True.
        normals: The normals in camera space. [B, C, N, 3] if packed is False, [nnz, 3] if packed is True.
        densify: Dummy variable to keep track of gradient for densification. [B, C, N, 2] if packed, [nnz, 3] if packed is True.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [B, C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [B * C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [B, C, channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [B, C, tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**.      [B, C, image_height, image_width, channels]
        - **Rendered alphas**.      [B, C, image_height, image_width, 1]
        - **Rendered normals**.     [B, C, image_height, image_width, 3]
        - **Rendered distortion**.  [B, C, image_height, image_width, 1]
        - **Rendered median depth**.[B, C, image_height, image_width, 1]


    """
    B, C = isect_offsets.shape[:2]
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert ray_transforms.shape == (nnz, 3, 3), ray_transforms.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(2)
        assert means2d.shape == (B, C, N, 2), means2d.shape
        assert ray_transforms.shape == (B, C, N, 3, 3), ray_transforms.shape
        assert colors.shape[:3] == (B, C, N), colors.shape
        assert opacities.shape == (B, C, N), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (B, C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[-1]
    if channels > 512 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        # Make sure the depth (last channel if present) remains in the last channel after padding (for depth distortion and median depth in CUDA kernel)
        colors = torch.cat(
            [
                colors[..., :-1],
                torch.empty(*colors.shape[:-1], padded_channels, device=device),
                colors[..., -1:],
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0
    tile_height, tile_width = isect_offsets.shape[2:4]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
    ) = _RasterizeToPixels2DGS.apply(
        means2d.contiguous(),
        ray_transforms.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        normals.contiguous(),
        densify.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
        distloss,
    )

    if padded_channels > 0:
        render_colors = torch.cat(
            [render_colors[..., : -padded_channels - 1], render_colors[..., -1:]],
            dim=-1,
        )

    return render_colors, render_alphas, render_normals, render_distort, render_median


@torch.no_grad()
def rasterize_to_indices_in_range_2dgs(
    range_start: int,
    range_end: int,
    transmittances: Tensor,  # [B, C, image_height, image_width]
    means2d: Tensor,  # [B, C, N, 2]
    ray_transforms: Tensor,  # [B, C, N, 3, 3]
    opacities: Tensor,  # [B, C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Rasterizes a batch of Gaussians to images but only returns the indices.

    .. note::

        This function supports iterative rasterization, in which each call of this function
        will rasterize a batch of Gaussians from near to far, defined by `[range_start, range_end)`.
        If a one-step full rasterization is desired, set `range_start` to 0 and `range_end` to a really
        large number, e.g, 1e10.

    Args:
        range_start: The start batch of Gaussians to be rasterized (inclusive).
        range_end: The end batch of Gaussians to be rasterized (exclusive).
        transmittances: Currently transmittances. [B, C, image_height, image_width]
        means2d: Projected Gaussian means. [B, C, N, 2]
        ray_transforms: transformation matrices that transforms xy-planes in pixel spaces into splat coordinates. [B, C, N, 3, 3]
        opacities: Gaussian opacities that support per-view values. [B, C, N]
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [B, C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [B * C * N] from  `isect_tiles()`. [n_isects]

    Returns:
        A tuple:

        - **Gaussian ids**. Gaussian ids for the pixel intersection. A flattened list of shape [M].
        - **Pixel ids**. pixel indices (row-major). A flattened list of shape [M].
        - **Camera ids**. Camera indices. A flattened list of shape [M].
        - **Batch ids**. Batch indices. A flattened list of shape [M].
    """

    B, C, N, _ = means2d.shape
    assert ray_transforms.shape == (B, C, N, 3, 3), ray_transforms.shape
    assert opacities.shape == (B, C, N), opacities.shape
    assert isect_offsets.shape[:2] == (B, C), isect_offsets.shape

    tile_height, tile_width = isect_offsets.shape[2:4]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    out_gauss_ids, out_indices = _make_lazy_cuda_func("rasterize_to_indices_2dgs")(
        range_start,
        range_end,
        transmittances.contiguous(),
        means2d.contiguous(),
        ray_transforms.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )
    out_pixel_ids = out_indices % (image_width * image_height)
    out_camera_ids = (out_indices // (image_width * image_height)) % C
    out_batch_ids = (out_indices // (image_width * image_height)) // C
    return out_gauss_ids, out_pixel_ids, out_camera_ids, out_batch_ids


class _RasterizeToPixels2DGS(torch.autograd.Function):
    """Rasterize gaussians 2DGS"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        ray_transforms: Tensor,
        colors: Tensor,
        opacities: Tensor,
        normals: Tensor,
        densify: Tensor,
        backgrounds: Tensor,
        masks: Tensor,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        absgrad: bool,
        distloss: bool,
    ) -> Tuple[Tensor, Tensor]:
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

        ctx.save_for_backward(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.distloss = distloss

        # double to float
        render_alphas = render_alphas.float()
        return (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
        )

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,
        v_render_alphas: Tensor,
        v_render_normals: Tensor,
        v_render_distort: Tensor,
        v_render_median: Tensor,
    ):

        (
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad

        (
            v_means2d_abs,
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_2dgs_bwd")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous(),
            v_render_distort.contiguous(),
            v_render_median.contiguous(),
            absgrad,
        )
        torch.cuda.synchronize()
        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[6]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(2, 3)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
