"""Python bindings for computing cumulative intersects"""

from typing import Tuple, Any

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C


class ComputeCumulativeIntersects(Function):
    """Computes cumulative intersections of gaussians. This is useful for creating unique gaussian IDs and for sorting.

    Args:
        num_points (int): number of gaussians.
        num_tiles_hit (Tensor): number of intersected tiles per gaussian.

    Returns:
        A tuple of {int, Tensor}:

        - **num_intersects** (int): total number of tile intersections.
        - **cum_tiles_hit** (Tensor): a tensor of cumulated intersections (used for sorting).
    """

    @staticmethod
    def forward(
        ctx, num_points: int, num_tiles_hit: Float[Tensor, "batch 1"]
    ) -> Tuple[int, Float[Tensor, "batch 1"]]:

        num_intersects, cum_tiles_hit = _C.compute_cumulative_intersects(
            num_points, num_tiles_hit
        )

        return (num_intersects, cum_tiles_hit)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


def compute_cumulative_intersects(
    num_points: int, num_tiles_hit: Float[Tensor, "batch 1"]
) -> Tuple[int, Float[Tensor, "batch 1"]]:
    """Computes cumulative intersections of gaussians. This is useful for creating unique gaussian IDs and for sorting.

    Args:
        num_points (int): number of gaussians.
        num_tiles_hit (Tensor): number of intersected tiles per gaussian.

    Returns:
        A tuple of {int, Tensor}:

        - **num_intersects** (int): total number of tile intersections.
        - **cum_tiles_hit** (Tensor): a tensor of cumulated intersections (used for sorting).
    """
    return _C.compute_cumulative_intersects(num_points, num_tiles_hit.contiguous())