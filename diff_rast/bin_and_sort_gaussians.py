"""Python bindings for binning and sorting gaussians"""

from typing import Tuple, Any

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import diff_rast.cuda as _C


class BinAndSortGaussians(Function):
    """Function for mapping gaussians to sorted unique intersection IDs and tile bins used for fast rasterization.

    We return both sorted and unsorted versions of intersect IDs and gaussian IDs for testing purposes.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): cumulative number of total gaussian intersections
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        isect_ids_unsorted (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        gaussian_ids_unsorted (Tensor): Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        isect_ids_sorted (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        gaussian_ids_sorted (Tensor): sorted Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        tile_bins (Tensor): range of gaussians hit per tile.
    """

    @staticmethod
    def forward(
        ctx,
        num_points: int,
        num_intersects: int,
        xys: Float[Tensor, "batch 2"],
        depths: Float[Tensor, "batch 1"],
        radii: Float[Tensor, "batch 1"],
        cum_tiles_hit: Float[Tensor, "batch 1"],
        tile_bounds: Tuple[int, int, int],
    ) -> Tuple[
        Float[Tensor, "num_intersects 1"],
        Float[Tensor, "num_intersects 1"],
        Float[Tensor, "num_intersects 1"],
        Float[Tensor, "num_intersects 1"],
        Float[Tensor, "num_intersects 2"],
    ]:

        (
            isect_ids_unsorted,
            gaussian_ids_unsorted,
            isect_ids_sorted,
            gaussian_ids_sorted,
            tile_bins,
        ) = _C.bin_and_sort_gaussians(
            num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds
        )

        return (
            isect_ids_unsorted,
            gaussian_ids_unsorted,
            isect_ids_sorted,
            gaussian_ids_sorted,
            tile_bins,
        )

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError
