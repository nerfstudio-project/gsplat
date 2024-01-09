"""Python bindings for mapping gaussians to interset IDs"""

from typing import Tuple, Any

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C
import warnings

class MapGaussiansToIntersects(Function):
    """Deprecated version of map_gaussian_to_intersects()"""

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
    ) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
        warnings.warn(
            "MapGaussiansToIntersects is deprecated. Use map_gaussian_to_intersects() instead.",
            DeprecationWarning,
        )
        return map_gaussian_to_intersects(
            num_points,
            num_intersects,
            xys,
            depths,
            radii,
            cum_tiles_hit,
            tile_bounds,
        )

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


def map_gaussian_to_intersects(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
    """Map each gaussian intersection to a unique tile ID and depth value for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): total number of tile intersections.
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor}:

        - **isect_ids** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit.
    """
    isect_ids, gaussian_ids = _C.map_gaussian_to_intersects(
        num_points, num_intersects, xys.contiguous(), depths.contiguous(), radii.contiguous(), cum_tiles_hit.contiguous(), tile_bounds
    )
    return (isect_ids, gaussian_ids)
