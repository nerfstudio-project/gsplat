import math
import torch
from torch import Tensor
from .cuda._wrapper import _make_lazy_cuda_func

N_MAX = 51
BINOMS = torch.zeros((N_MAX, N_MAX)).float().cuda()
for n in range(N_MAX):
    for k in range(n + 1):
        BINOMS[n, k] = math.comb(n, k)
BINOMS = BINOMS.contiguous()


def compute_relocation(
    old_opacities: Tensor,  # [N]
    old_scales: Tensor,  # [N, 3]
    ratios: Tensor,  # [N]
) -> tuple[Tensor, Tensor]:
    """Compute new Gaussians from a set of old Gaussians.

    This function interprets the Gaussians as samples from a likelihood distribution.
    It uses the old opacities and scales to compute the new opacities and scales.
    This is an implementation of the paper
    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/pdf/2404.09591>`_,

    Args:
        old_opacities: The opacities of the Gaussians. [N]
        old_scales: The scales of the Gaussians. [N, 3]
        ratios: The relative frequencies for each of the Gaussians. [N]

    Returns:
        A tuple:

        **new_opacities**: The opacities of the new Gaussians. [N]
        **new_scales**: The scales of the Gaussians. [N, 3]
    """
    N = old_opacities.shape[0]
    assert old_scales.shape == (N, 3), old_scales.shape
    assert ratios.shape == (N,), ratios.shape
    old_opacities = old_opacities.contiguous()
    old_scales = old_scales.contiguous()
    ratios = ratios.int().contiguous()

    new_opacities, new_scales = _make_lazy_cuda_func("compute_relocation")(
        old_opacities, old_scales, ratios, BINOMS, N_MAX
    )
    new_opacities = torch.clamp(
        new_opacities, max=1.0 - torch.finfo(torch.float32).eps, min=0.005
    )
    new_opacities = torch.logit(new_opacities)
    new_scales = torch.log(new_scales.reshape(-1, 3))
    return new_opacities, new_scales
