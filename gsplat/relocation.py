import math

import torch
from torch import Tensor

from .cuda._wrapper import _make_lazy_cuda_func

# N_MAX = 51
# BINOMS = torch.zeros((N_MAX, N_MAX)).float().cuda()  # TODO
# for n in range(N_MAX):
#     for k in range(n + 1):
#         BINOMS[n, k] = math.comb(n, k)


def compute_relocation(
    opacities: Tensor,  # [N]
    scales: Tensor,  # [N, 3]
    ratios: Tensor,  # [N]
) -> tuple[Tensor, Tensor]:
    """Compute new Gaussians from a set of old Gaussians.

    This function interprets the Gaussians as samples from a likelihood distribution.
    It uses the old opacities and scales to compute the new opacities and scales.
    This is an implementation of the paper
    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/pdf/2404.09591>`_,

    Args:
        opacities: The opacities of the Gaussians. [N]
        scales: The scales of the Gaussians. [N, 3]
        ratios: The relative frequencies for each of the Gaussians. [N]

    Returns:
        A tuple:

        **new_opacities**: The opacities of the new Gaussians. [N]
        **new_scales**: The scales of the Gaussians. [N, 3]
    """

    N_MAX = 51
    BINOMS = torch.zeros((N_MAX, N_MAX), device=opacities.device)
    for n in range(N_MAX):
        for k in range(n + 1):
            BINOMS[n, k] = math.comb(n, k)

    N = opacities.shape[0]
    assert scales.shape == (N, 3), scales.shape
    assert ratios.shape == (N,), ratios.shape
    opacities = opacities.contiguous()
    scales = scales.contiguous()
    ratios.clamp_(min=1, max=N_MAX)
    ratios = ratios.int().contiguous()

    new_opacities, new_scales = _make_lazy_cuda_func("compute_relocation")(
        opacities, scales, ratios, BINOMS, N_MAX
    )
    return new_opacities, new_scales
