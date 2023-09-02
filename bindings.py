"""Python bindings for custom Cuda functions"""

from typing import Tuple

import torch
import diff_rast  # make sure to import torch before diff_rast
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function


class compute_cov2d_bounds(Function):
    """Computes bounds of 2D covariance matrix

    expects input cov2d to size (batch, 3) upper triangular values

    Returns: tuple of conics (batch, 3) and radii (batch, 1)
    """

    @staticmethod
    def forward(
        ctx, cov2d: Float[Tensor, "batch 3"]
    ) -> Tuple[Float[Tensor, "batch_conics 3"], Float[Tensor, "batch_radii 1"]]:
        num_pts = A.shape[0]
        assert num_pts > 0

        output = diff_rast.compute_cov2d_bounds_forward(num_pts, A)
        return output


if __name__ == "__main__":
    """Test bindings"""
    # cov2d bounds
    f = compute_cov2d_bounds()
    A = torch.rand(4, 3, device="cuda:0").requires_grad_()
    conics, radii = f.apply(A)
    print(A)
    print(conics, radii)
