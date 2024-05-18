import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def load_test_data(
    data_path: str = "assets/test_garden.npz",
    device="cuda",
    scene_crop: Tuple[float, float, float, float, float, float] = (-2, -2, -2, 2, 2, 2),
    scene_grid: int = 1,
):
    """Load the test data."""
    assert scene_grid % 2 == 1, "scene_grid must be odd"

    data = np.load(os.path.join(os.path.dirname(__file__), "..", data_path))
    height, width = data["height"].item(), data["width"].item()
    viewmats = torch.from_numpy(data["viewmats"]).float().to(device)
    Ks = torch.from_numpy(data["Ks"]).float().to(device)
    means = torch.from_numpy(data["means3d"]).float().to(device)
    colors = torch.from_numpy(data["colors"] / 255.0).float().to(device)
    C = len(viewmats)

    # crop
    aabb = torch.tensor(scene_crop, device=device)
    edges = aabb[3:] - aabb[:3]
    sel = ((means >= aabb[:3]) & (means <= aabb[3:])).all(dim=-1)
    sel = torch.where(sel)[0]
    means, colors = means[sel], colors[sel]

    # repeat the scene into a grid (to mimic a large-scale setting)
    repeats = scene_grid
    gridx, gridy = torch.meshgrid(
        [
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
        ],
        indexing="ij",
    )
    grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(-1, 3)
    means = means[None, :, :] + grid[:, None, :] * edges[None, None, :]
    means = means.reshape(-1, 3)
    colors = colors.repeat(repeats**2, 1)

    # create gaussian attributes
    N = len(means)
    scales = torch.rand((N, 3), device=device) * 0.02
    quats = F.normalize(torch.randn((N, 4), device=device), dim=-1)
    opacities = torch.rand((N,), device=device)

    return means, quats, scales, opacities, colors, viewmats, Ks, width, height


def rasterization_legacy_wrapper(
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
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor, Dict]:
    """Wrapper for old version gsplat.

    Note: This function exists for comparision purpose only. So we skip collecting
    the intermidiate variables, and only return an empty dict.
    """
    from gsplat.cuda_legacy._wrapper import (
        project_gaussians,
        rasterize_gaussians,
        spherical_harmonics,
    )

    assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
    C = len(viewmats)

    render_colors, render_alphas = [], []
    for cid in range(C):
        fx, fy = Ks[cid, 0, 0], Ks[cid, 1, 1]
        cx, cy = Ks[cid, 0, 2], Ks[cid, 1, 2]
        viewmat = viewmats[cid]
        background = (
            backgrounds[cid]
            if backgrounds is not None
            else torch.zeros(3, device=means.device)
        )

        means2d, depths, radii, conics, _, num_tiles_hit, _ = project_gaussians(
            means3d=means,
            scales=scales,
            glob_scale=1.0,
            quats=quats,
            viewmat=viewmat,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            img_height=height,
            img_width=width,
            block_width=tile_size,
            clip_thresh=near_plane,
        )

        if colors.dim() == 3:
            c2w = viewmat.inverse()
            viewdirs = means - c2w[:3, 3]
            # viewdirs = F.normalize(viewdirs, dim=-1).detach()
            if sh_degree is None:
                sh_degree = int(math.sqrt(colors.shape[1]) - 1)
            colors = spherical_harmonics(sh_degree, viewdirs, colors)  # [N, 3]

        render_colors_, render_alphas_ = rasterize_gaussians(
            xys=means2d,
            depths=depths,
            radii=radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=colors,
            opacity=opacities[..., None],
            img_height=height,
            img_width=width,
            block_width=tile_size,
            background=background,
            return_alpha=True,
        )
        render_colors.append(render_colors_)
        render_alphas.append(render_alphas_[..., None])
    render_colors = torch.stack(render_colors, dim=0)
    render_alphas = torch.stack(render_alphas, dim=0)
    return render_colors, render_alphas, {}


def rasterization_inria_wrapper(
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
    far_plane: float = 100.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    backgrounds: Optional[Tensor] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor, Dict]:
    """Wrapper for Inria's rasterization backend.

    Note: This function exists for comparision purpose only. Only rendered image is
    returned. Also, Inria's implementation will apply a
    `torch.clamp(colors + 0.5, min=0.0)` after spherical harmonics calculation, which is
    different from the behavior of gsplat. Use with caution!

    Inria's CUDA backend has its own LICENSE, so this function should be used with
    the respect to the original LICENSE at:
    https://github.com/graphdeco-inria/diff-gaussian-rasterization
    """
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    def _getProjectionMatrix(znear, zfar, fovX, fovY, device="cuda"):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4, device=device)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    assert colors.shape[-1] == 3, "Only RGB colors are supported"
    assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
    C = len(viewmats)
    device = means.device

    render_colors = []
    for cid in range(C):
        FoVx = 2 * math.atan(width / (2 * Ks[cid, 0, 0].item()))
        FoVy = 2 * math.atan(height / (2 * Ks[cid, 1, 1].item()))
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)

        world_view_transform = viewmats[cid].transpose(0, 1)
        projection_matrix = _getProjectionMatrix(
            znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        background = (
            backgrounds[cid]
            if backgrounds is not None
            else torch.zeros(3, device=device)
        )

        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=0 if sh_degree is None else sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = torch.zeros_like(means, requires_grad=True, device=device)
        # Note: This implementation will apply a
        # torch.clamp(colors + 0.5, min=0.0) after spherical_harmonics, which is
        # different from the behavior of gsplat. Use with caution!
        render_colors_, radii = rasterizer(
            means3D=means,
            means2D=means2D,
            shs=colors if colors.dim() == 3 else None,
            colors_precomp=colors if colors.dim() == 2 else None,
            opacities=opacities[:, None],
            scales=scales,
            rotations=quats,
            cov3D_precomp=None,
        )
        render_colors_ = render_colors_.permute(1, 2, 0)  # [H, W, 3]

        render_colors.append(render_colors_)
    render_colors = torch.stack(render_colors, dim=0)
    return render_colors, None, {}
