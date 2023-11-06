"""Python bindings for custom Cuda functions"""

from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C


class RasterizeGaussians(Function):
    """Rasterizes 2D gaussians by sorting and binning gaussian intersections for each tile and returns an output image using alpha-compositing.

    Args:
        xys (Tensor): xy coords of 2D gaussians.
        depths (Tensor): depths of 2D gaussians.
        radii (Tensor): radii of 2D gaussians
        conics (Tensor): conics (inverse of covariance) of 2D gaussians in upper triangular format
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        colors (Tensor): colors associated with the gaussians.
        opacity (Tensor): opacity associated with the gaussians.
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        background (Tensor): background color

    Returns:
        A Tensor:

        - **out_img** (Tensor): 3-channel RGB rendered output image.
        - **out_alpha** (Tensor): Alpha channel of the rendered output image.
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
    ):
        if colors.dtype == torch.uint8:
            # make sure colors are float [0,1]
            colors = colors.float() / 255

        if background is not None:
            assert (
                background.shape[0] == colors.shape[-1]
            ), f"incorrect shape of background color tensor, expected shape {colors.shape[-1]}"
        else:
            background = torch.ones(3, dtype=torch.float32)
        (
            out_img,
            final_Ts,
            final_idx,
            tile_bins,
            gaussian_ids_sorted,
            isect_ids_sorted,
            gaussian_ids_unsorted,
            isect_ids_unsorted,
        ) = _C.rasterize_forward(
            xys.contiguous().cuda(),
            depths.contiguous().cuda(),
            radii.contiguous().cuda(),
            conics.contiguous().cuda(),
            num_tiles_hit.contiguous().cuda(),
            colors.contiguous().cuda(),
            opacity.contiguous().cuda(),
            img_height,
            img_width,
            background.contiguous().cuda(),
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
        out_alpha = 1 - final_Ts

        return out_img, out_alpha

    @staticmethod
    def backward(ctx, v_out_img, v_out_alpha=None):
        img_height = ctx.img_height
        img_width = ctx.img_width

        if v_out_alpha is None:
            v_out_alpha = torch.zeros_like(v_out_img[..., 0])

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

        v_xy, v_conic, v_colors, v_opacity = _C.rasterize_backward(
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
            v_out_alpha.contiguous().cuda(),
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
