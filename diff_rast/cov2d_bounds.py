"""Python bindings for 2D covariance bounds"""

from typing import Any, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import diff_rast.cuda as _C


class compute_cov2d_bounds(Function):
    """Computes bounds of 2D covariance matrix

    Args:
        cov2d (Tensor): input cov2d of size  (batch, 3) of upper triangular 2D covariance values

    Returns:
        conics (batch, 3) and radii (batch, 1)
    """

    @staticmethod
    def forward(
        ctx, cov2d: Float[Tensor, "batch 3"]
    ) -> Tuple[Float[Tensor, "batch_conics 3"], Float[Tensor, "batch_radii 1"]]:
        assert (
            cov2d.shape[-1] == 3
        ), f"Expected input cov2d to be of shape (*batch, 3) (upper triangular values), but got {tuple(cov2d.shape)}"
        num_pts = cov2d.shape[0]
        assert num_pts > 0
        conic, radius = _C.compute_cov2d_bounds_forward(num_pts, cov2d.contiguous().cuda())
        return (conic, radius)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError
