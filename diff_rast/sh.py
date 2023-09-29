"""Python bindings for SH"""

import torch
import cuda_lib

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function


def num_sh_bases(degree: int):
    if degree == 0:
        return 1
    if degree == 1:
        return 4
    if degree == 2:
        return 9
    if degree == 3:
        return 16
    return 25


class SphericalHarmonics(Function):
    """Compute spherical harmonics

    Args:
        degree (int): degree of SHs.
        viewdirs (Tensor): viewing directions.
        coeffs (Tensor): harmonic coefficients.
    """

    @staticmethod
    def forward(
        ctx,
        degree: int,
        viewdirs: Float[Tensor, "*batch 3"],
        coeffs: Float[Tensor, "*batch D C"],
    ):
        num_points = coeffs.shape[0]
        assert coeffs.shape[-2] == num_sh_bases(degree)
        ctx.degree = degree
        ctx.save_for_backward(viewdirs)
        return cuda_lib.compute_sh_forward(num_points, degree, viewdirs, coeffs)

    @staticmethod
    def backward(ctx, v_colors: Float[Tensor, "*batch 3"]):
        degree = ctx.degree
        viewdirs = ctx.saved_tensors[0]
        num_points = v_colors.shape[0]
        return (
            None,
            None,
            cuda_lib.compute_sh_backward(num_points, degree, viewdirs, v_colors),
        )
