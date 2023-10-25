"""Python bindings for forward rasterization"""

from typing import Tuple, Any, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C


class RasterizeForwardKernel(Function):
    """Kernel function for rasterizing and alpha-composing each tile.

    Args:
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).
        block (Tuple): block dimensions as a len 3 tuple (block.x , block.y, 1).
        img_size (Tuple): image dimensions as a len 3 tuple (img.x , img.y, 1).
        gaussian_ids_sorted (Tensor): tensor that maps isect_ids back to cum_tiles_hit, sorted in ascending order.
        tile_bins (Tensor): range of gaussians IDs hit per tile.
        xys (Tensor): x,y locations of 2D gaussian projections.
        conics (Tensor): conics (inverse of covariance) of 2D gaussians in upper triangular format.
        colors (Tensor): colors associated with the gaussians.
        opacities (Tensor): opacity associated with the gaussians.
        background (Tensor): background color

    Returns:
        A tuple of {Tensor, Tensor, Tensor}:

        - **out_img** (Tensor): the rendered output image.
        - **final_Ts** (Tensor): the final transmittance values.
        - **final_idx** (Tensor): the final gaussian IDs.
    """

    @staticmethod
    def forward(
        ctx,
        tile_bounds: Tuple[int, int, int],
        block: Tuple[int, int, int],
        img_size: Tuple[int, int, int],
        gaussian_ids_sorted: Int[Tensor, "num_intersects 1"],
        tile_bins: Int[Tensor, "num_intersects 2"],
        xys: Float[Tensor, "batch 2"],
        conics: Float[Tensor, "*batch 3"],
        colors: Float[Tensor, "*batch channels"],
        opacities: Float[Tensor, "*batch 1"],
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

        (out_img, final_Ts, final_idx,) = _C.rasterize_forward_kernel(
            tile_bounds,
            block,
            img_size,
            gaussian_ids_sorted.contiguous().cuda(),
            tile_bins.contiguous().cuda(),
            xys.contiguous().cuda(),
            conics.contiguous().cuda(),
            colors.contiguous().cuda(),
            opacities.contiguous().cuda(),
            background.contiguous().cuda(),
        )
        return (
            out_img,
            final_Ts,
            final_idx,
        )

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError
