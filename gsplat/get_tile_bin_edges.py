"""Python bindings for computing tile bins"""

from typing import Any

from jaxtyping import Int
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C
import warnings


class GetTileBinEdges(Function):
    """Deprecated version of get_tile_bin_edges()"""

    @staticmethod
    def forward(
        ctx, num_intersects: int, isect_ids_sorted: Int[Tensor, "num_intersects 1"]
    ) -> Int[Tensor, "num_intersects 2"]:
        warnings.warn(
            "GetTileBinEdges is deprecated. Use get_tile_bin_edges() instead.",
            DeprecationWarning,
        )
        return get_tile_bin_edges(num_intersects, isect_ids_sorted)
    
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


def get_tile_bin_edges(
    num_intersects: int, isect_ids_sorted: Int[Tensor, "num_intersects 1"]
) -> Int[Tensor, "num_intersects 2"]:
    """Map sorted intersection IDs to tile bins which give the range of unique gaussian IDs belonging to each tile.

    Expects that intersection IDs are sorted by increasing tile ID.

    Indexing into tile_bins[tile_idx] returns the range (lower,upper) of gaussian IDs that hit tile_idx.

    Note:
        This function is not differentiable to any input.

    Args:
        num_intersects (int): total number of gaussian intersects.
        isect_ids_sorted (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).

    Returns:
        A Tensor:

        - **tile_bins** (Tensor): range of gaussians IDs hit per tile.
    """
    return _C.get_tile_bin_edges(num_intersects, isect_ids_sorted.contiguous())
