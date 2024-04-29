import math

import torch

from ._torch_impl import (
    _isect_offset_encode,
    _isect_tiles,
    _persp_proj,
    _projection,
    _quat_scale_to_covar_perci,
    _world_to_cam,
    accumulate,
)
from ._warpper import (
    isect_offset_encode,
    isect_tiles,
    persp_proj,
    projection,
    quat_scale_to_covar_perci,
    rasterize_to_indices_iter,
    rasterize_to_pixels,
    world_to_cam,
)


def rendering(means, quats, scales, opacities, colors, viewmats, Ks, width, height):
    tile_size = 16
    C = len(viewmats)
    covars, _ = quat_scale_to_covar_perci(quats, scales, compute_perci=False, triu=True)
    radii, means2d, depths, conics = projection(
        means, covars, viewmats, Ks, width, height
    )
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    tiles_per_gauss, isect_ids, gauss_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
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
    )
    return render_colors, render_alphas


def _rendering(means, quats, scales, opacities, colors, viewmats, Ks, width, height):
    batch_per_iter = 100
    tile_size = 16
    C = len(viewmats)
    device = means.device

    covars, _ = quat_scale_to_covar_perci(quats, scales, compute_perci=False, triu=True)
    radii, means2d, depths, conics = projection(
        means, covars, viewmats, Ks, width, height
    )
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    tiles_per_gauss, isect_ids, gauss_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    # iteratively accumulate the colors and opacities via torch auto-grad rasterization
    render_colors = torch.zeros((C, height, width, colors.shape[-1]), device=device)
    render_alphas = torch.zeros(
        (C, height, width, 1), device=device, dtype=torch.float64
    )

    block_size = tile_size * tile_size
    max_range = (isect_offsets[1:] - isect_offsets[:-1]).max().item()
    num_batches = (max_range + block_size - 1) // block_size
    for step in range(0, num_batches, batch_per_iter):
        transmittances = 1.0 - render_alphas[..., 0]
        gs_ids, indices = rasterize_to_indices_iter(
            step,
            step + batch_per_iter,
            transmittances,
            means2d,
            conics,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            gauss_ids,
        )
        pixel_ids = indices % (width * height)
        camera_ids = indices // (width * height)
        if len(gs_ids) == 0:
            break

        renders_step, accs_step = accumulate(
            means2d,
            conics,
            opacities,
            colors,
            gs_ids,
            pixel_ids,
            camera_ids,
            width,
            height,
            prefix_trans=transmittances.float(),
        )
        render_colors = render_colors + renders_step
        render_alphas = render_alphas + accs_step

    return render_colors, render_alphas.float()


def _rendering_gsplat(
    means, quats, scales, opacities, colors, viewmats, Ks, width, height
):
    from gsplat import project_gaussians, rasterize_gaussians

    tile_size = 16
    bkgd = torch.zeros(3, device=means.device)
    render_colors, render_alphas = [], []
    for viewmat, K, color in zip(viewmats, Ks, colors):
        cx, cy, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]
        means2d, depths, radii, conics, _, num_tiles_hit, _ = project_gaussians(
            means, scales, 1.0, quats, viewmat, fx, fy, cx, cy, height, width, tile_size
        )
        _render_colors, _render_alphas = rasterize_gaussians(
            means2d,
            depths,
            radii,
            conics,
            num_tiles_hit,
            color,
            opacities[..., None],
            height,
            width,
            tile_size,
            background=bkgd,
            return_alpha=True,
        )
        render_colors.append(_render_colors)
        render_alphas.append(_render_alphas[..., None])
    return torch.stack(render_colors), torch.stack(render_alphas)
