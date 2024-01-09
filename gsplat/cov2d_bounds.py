"""Python bindings for 2D covariance bounds"""

from typing import Any, Tuple

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C
import warnings


class ComputeCov2dBounds(Function):
    """Deprecated version of compute_cov2d_bounds()"""

    @staticmethod
    def forward(
        ctx, cov2d: Float[Tensor, "batch 3"]
    ) -> Tuple[Float[Tensor, "batch_conics 3"], Float[Tensor, "batch_radii 1"]]:
        warnings.warn(
            "ComputeCov2dBounds is deprecated. Use compute_cov2d_bounds() instead.",
            DeprecationWarning,
        )
        return compute_cov2d_bounds(cov2d)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError


def compute_cov2d_bounds(
    cov2d: Float[Tensor, "batch 3"]
) -> Tuple[Float[Tensor, "batch_conics 3"], Float[Tensor, "batch_radii 1"]]:
    """Computes bounds of 2D covariance matrix

    Args:
        cov2d (Tensor): input cov2d of size  (batch, 3) of upper triangular 2D covariance values

    Returns:
        A tuple of {Tensor, Tensor}:

        - **conic** (Tensor): conic parameters for 2D gaussian.
        - **radii** (Tensor): radii of 2D gaussian projections.
    """
    assert (
        cov2d.shape[-1] == 3
    ), f"Expected input cov2d to be of shape (*batch, 3) (upper triangular values), but got {tuple(cov2d.shape)}"
    num_pts = cov2d.shape[0]
    assert num_pts > 0
    return _C.compute_cov2d_bounds(num_pts, cov2d.contiguous())
