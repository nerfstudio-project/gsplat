import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from .cuda._wrapper import (
    isect_offset_encode,
    isect_tiles,
    projection,
    rasterize_to_pixels,
    spherical_harmonics,
)


def rasterization(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [N, D] or [N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    compute_means2d_absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize a set of Gaussians to pixels.

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The normalized quaternions of the Gaussians. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        colors: The colors of the Gaussians. [N, D] or [N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [N, K, 3] SH coefficients,
            else the `colors` should [N, D] per-Gaussian color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values not tested)
        backgrounds: The background colors. [C, 3]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the RGB image, "D" renders the depth image, and
            "ED" renders the expected depth image. Default is "RGB".

    Returns:
        A tuple of:
        **render_colors**: The rendered colors. [C, width, height, X]. X depends on the
            `render_mode`.
        **render_alphas**: The rendered alphas. [C, width, height, 1].
        **meta**: A dictionary of intermediate results.
    """

    N = means.shape[0]
    C = viewmats.shape[0]
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    if sh_degree is None:
        # treat colors as post-activation values
        assert colors.dim() == 2 and colors.shape[0] == N, colors.shape
    else:
        # treat colors as SH coefficients. Allowing for activating partial SH bands
        assert (
            colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[1], colors.shape

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = projection(
        means,
        None,  # covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
    )

    if packed:
        # The results are packed into shape [nnz, ...]. All elements are valid.
        rindices, cindices, radii, means2d, depths, conics, compensations = proj_results
        opacities = opacities[cindices.long()]  # [nnz]
    else:
        # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations = proj_results
        opacities = opacities.repeat(C, 1)  # [C, N]
        rindices, cindices = None, None

    if compensations is not None:
        opacities = opacities * compensations

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, gauss_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_cameras=C,
        rindices=rindices,
        cindices=cindices,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    # Compute the per-view colors
    colors = (
        colors[cindices.long()] if packed else colors.expand(C, *([-1] * colors.dim()))
    )  # [nnz, D] or [C, N, 3]
    if sh_degree is not None:  # SH coefficients
        camtoworlds = torch.inverse(viewmats)
        if packed:
            dirs = means[cindices.long(), :] - camtoworlds[rindices.long(), :3, 3]
        else:
            dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]
        # dirs = F.normalize(dirs, dim=-1).detach()  # [nnz, 3] or [C, N, 3]
        colors = spherical_harmonics(
            sh_degree, dirs, colors, masks=radii > 0
        )  # [nnz, D] or [C, N, 3]
        # colors = torch.clamp_min(colors + 0.5, 0.0)

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)
    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
    else:  # RGB
        pass
    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        gauss_ids,
        backgrounds=backgrounds,
        packed=packed,
        compute_means2d_absgrad=compute_means2d_absgrad,
    )
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )

    meta = {
        "rindices": rindices,
        "cindices": cindices,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": opacities,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "gauss_ids": gauss_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
    }
    return render_colors, render_alphas, meta
