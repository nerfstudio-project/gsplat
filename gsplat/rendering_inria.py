import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .util.normal_utils import depth_to_normal
from .util.camera_utils import getProjectionMatrix


def rasterization_2dgs_inria_wrapper(
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
    """Wrapper for 2DGS's rasterization backend which is based on Inria's backend.

    Install the 2DGS rasterization backend from
        https://github.com/hbb1/diff-surfel-rasterization
    """
    from diff_surfel_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
    C = len(viewmats)
    device = means.device
    channels = colors.shape[-1]

    # rasterization from inria does not do normalization internally
    quats = F.normalize(quats, dim=-1)  # [N, 4]
    scales = scales[:, :2]  # [N, 2]

    render_colors = []
    for cid in range(C):
        FoVx = 2 * math.atan(width / (2 * Ks[cid, 0, 0].item()))
        FoVy = 2 * math.atan(height / (2 * Ks[cid, 1, 1].item()))
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)

        world_view_transform = viewmats[cid].transpose(0, 1)
        projection_matrix = getProjectionMatrix(
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

        render_colors_ = []
        for i in range(0, channels, 3):
            _colors = colors[..., i : i + 3]
            if _colors.shape[-1] < 3:
                pad = torch.zeros(
                    _colors.shape[0], 3 - _colors.shape[-1], device=device
                )
                _colors = torch.cat([_colors, pad], dim=-1)
            _render_colors_, _, allmap = rasterizer(
                means3D=means,
                means2D=means2D,
                shs=_colors if colors.dim() == 3 else None,
                colors_precomp=_colors if colors.dim() == 2 else None,
                opacities=opacities[:, None],
                scales=scales,
                rotations=quats,
                cov3D_precomp=None,
            )
            if _colors.shape[-1] < 3:
                _render_colors_ = _render_colors_[:, :, : _colors.shape[-1]]
            render_colors_.append(_render_colors_)
        render_colors_ = torch.cat(render_colors_, dim=-1)

        render_colors_ = render_colors_.permute(1, 2, 0)  # [H, W, 3]
        render_colors.append(render_colors_)
    render_colors = torch.stack(render_colors, dim=0)

    # additional maps
    allmap = allmap.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, C]
    render_depth_expected = allmap[..., 0:1]
    render_alphas = allmap[..., 1:2]
    render_normal = allmap[..., 2:5]
    render_depth_median = allmap[..., 5:6]
    render_dist = allmap[..., 6:7]

    render_normal = render_normal @ (world_view_transform[:3, :3].T)
    render_depth_expected = render_depth_expected / render_alphas
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # render_depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ratio = 0, to reduce disk aliasing.
    depth_ratio = 0
    render_depth = (
        render_depth_expected * (1 - depth_ratio) + (depth_ratio) * render_depth_median
    )

    normals_surf = depth_to_normal(
        render_depth,
        viewmats,
        Ks,
        near_plane=near_plane,
        far_plane=far_plane,
    )
    normals_surf = normals_surf * (render_alphas).detach()

    render_colors = torch.cat([render_colors, render_depth], dim=-1)
    meta = {
        "normals_rend": render_normal,
        "normals_surf": normals_surf,
        "render_distloss": render_dist,
    }
    return render_colors, render_alphas, meta
