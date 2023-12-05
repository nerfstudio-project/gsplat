"""Python bindings for custom Cuda functions"""

from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C
from .bin_and_sort_gaussians import bin_and_sort_gaussians
from .compute_cumulative_intersects import compute_cumulative_intersects


class NDRasterizeGaussians(Function):
    """Rasterizes 2D gaussians by sorting and binning gaussian intersections for each tile and returns an N-dimensional output using alpha-compositing.

    Args:
        xys (Tensor): xy coords of 2D gaussians.
        depths (Tensor): depths of 2D gaussians.
        radii (Tensor): radii of 2D gaussians
        conics (Tensor): conics (inverse of covariance) of 2D gaussians in upper triangular format
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        colors (Tensor): N-dimensional features associated with the gaussians.
        opacity (Tensor): opacity associated with the gaussians.
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        background (Tensor): background color

    Returns:
        A Tensor:

        - **out_img** (Tensor): N-dimensional rendered output image.
    """

    @staticmethod
    def forward(
        ctx,
        xys: Float[Tensor, "*batch 2"],
        depths: Float[Tensor, "*batch 1"],
        radii: Float[Tensor, "*batch 1"],
        conics: Float[Tensor, "*batch 3"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        colors: Float[Tensor, "*batch channels"],
        opacity: Float[Tensor, "*batch 1"],
        img_height: int,
        img_width: int,
        background: Optional[Float[Tensor, "channels"]] = None,
    ) -> Tensor:
        if colors.dtype == torch.uint8:
            # make sure colors are float [0,1]
            colors = colors.float() / 255

        if background is not None:
            assert (
                background.shape[0] == colors.shape[-1]
            ), f"incorrect shape of background color tensor, expected shape {colors.shape[-1]}"
        else:
            background = torch.ones(3, dtype=torch.float32, device=colors.device)

        if xys.ndimension() != 2 or xys.size(1) != 2:
            raise ValueError("xys must have dimensions (N, 2)")

        if colors.ndimension() != 2 or colors.size(1) != 3:
            raise ValueError("colors must have dimensions (N, 3)")

        channels = colors.size(1)
        num_points = xys.size(0)
        BLOCK_X, BLOCK_Y = 16, 16
        tile_bounds = (
            (img_width + BLOCK_X - 1) // BLOCK_X,
            (img_height + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        block = (BLOCK_X, BLOCK_Y, 1)
        img_size = (img_width, img_height, 1)
        num_tiles = tile_bounds[0] * tile_bounds[1]

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(
            num_points, num_tiles_hit
        )

        (
            isect_ids_unsorted,
            gaussian_ids_unsorted,
            isect_ids_sorted,
            gaussian_ids_sorted,
            tile_bins,
        ) = bin_and_sort_gaussians(
            num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds
        )

        out_img, final_Ts, final_idx = _C.nd_rasterize_forward(
            tile_bounds,
            block,
            img_size,
            gaussian_ids_sorted.contiguous(),
            tile_bins,
            xys.contiguous(),
            conics.contiguous(),
            colors.contiguous(),
            opacity.contiguous(),
            background.contiguous(),
        )

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.save_for_backward(
            gaussian_ids_sorted,
            tile_bins,
            xys,
            conics,
            colors,
            opacity,
            background,
            final_Ts,
            final_idx,
        )

        return out_img

    @staticmethod
    def backward(ctx, v_out_img):
        img_height = ctx.img_height
        img_width = ctx.img_width

        (
            gaussian_ids_sorted,
            tile_bins,
            xys,
            conics,
            colors,
            opacity,
            background,
            final_Ts,
            final_idx,
        ) = ctx.saved_tensors

        v_xy, v_conic, v_colors, v_opacity = _C.nd_rasterize_backward(
            img_height,
            img_width,
            gaussian_ids_sorted.contiguous().cuda(),
            tile_bins,
            xys.contiguous().cuda(),
            conics.contiguous().cuda(),
            colors.contiguous().cuda(),
            opacity.contiguous().cuda(),
            background.contiguous().cuda(),
            final_Ts.contiguous().cuda(),
            final_idx.contiguous().cuda(),
            v_out_img.contiguous().cuda(),
        )

        return (
            v_xy,  # xys
            None,  # depths
            None,  # radii
            v_conic,  # conics
            None,  # num_tiles_hit
            v_colors,  # colors
            v_opacity,  # opacity
            None,  # img_height
            None,  # img_width
            None,  # background
        )
