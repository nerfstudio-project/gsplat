"""Python bindings for mapping gaussians to interset IDs"""

from typing import Tuple, Any

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import diff_rast.cuda as _C


class MapGaussiansToIntersects(Function):
    """Function to map each gaussian intersection to a unique tile ID and depth value for sorting.

    Args:
        num_points (int): number of gaussians.
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

    @staticmethod
    def forward(
        ctx,
        num_points: int,
        xys: Float[Tensor, "batch 2"],
        depths: Float[Tensor, "batch 1"],
        radii: Float[Tensor, "batch 1"],
        cum_tiles_hit: Float[Tensor, "batch 1"],
        tile_bounds: Tuple[int, int, int],
    ) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:

        isect_ids, gaussian_ids = _C.map_gaussian_to_intersects(
            num_points, xys, depths, radii, cum_tiles_hit, tile_bounds
        )
        return (isect_ids, gaussian_ids)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError
