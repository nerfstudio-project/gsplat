"""Python bindings for computing tile bins"""

from typing import Any

from jaxtyping import Int
from torch import Tensor
from torch.autograd import Function

import diff_rast.cuda as _C


class GetTileBinEdges(Function):
    """Function to map sorted intersection IDs to tile bins which give the range of unique gaussian IDs belonging to each tile.

    Expects that intersection IDs are sorted by increasing tile ID.

    Indexing into tile_bins[tile_idx] returns the range (lower,upper) of gaussian IDs that hit tile_idx.

    Args:
        num_intersects (int): total number of gaussian intersects.
        isect_ids_sorted (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).

    Returns:
        A Tensor:

        - **tile_bins** (Tensor): range of gaussians IDs hit per tile.
    """

    @staticmethod
    def forward(
        ctx, num_intersects: int, isect_ids_sorted: Int[Tensor, "num_intersects 1"]
    ) -> Int[Tensor, "num_intersects 2"]:

        tile_bins = _C.get_tile_bin_edges(num_intersects, isect_ids_sorted)
        return tile_bins

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError
