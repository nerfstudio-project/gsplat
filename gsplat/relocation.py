import math
import torch
from torch import Tensor
from .cuda._wrapper import _make_lazy_cuda_func

N_MAX = 51
BINOMS = torch.zeros((N_MAX, N_MAX)).float().cuda()
for n in range(N_MAX):
    for k in range(n + 1):
        BINOMS[n, k] = math.comb(n, k)


def compute_relocation(
    old_opacities: Tensor,  # [N]
    old_scales: Tensor,  # [N, 3]
    N: Tensor,  # [N]
) -> tuple[Tensor, Tensor]:
    """Compute new Gaussians from a set of old Gaussians.

    This function interprets the Gaussians as samples from a likelihood distribution.
    It uses the old opacities, scales, and weights to compute the new opacities and scales.
    This is an implementation of the paper
    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/pdf/2404.09591>`_,

    Args:
        old_opacities: The opacities of the Gaussians. [N]
        old_scales: The scales of the Gaussians. [N, 3]
        N: Weights for each of the Gaussians. [N]

    Returns:
        A tuple:

        **new_opacities**: The opacities of the new Gaussians. [N]

        **new_scales**: The scales of the Gaussians. [N, 3]
    """
    new_opacities, new_scales = _make_lazy_cuda_func("compute_relocation")(
        old_opacities, old_scales, N.int(), BINOMS, N_MAX
    )
    return new_opacities, new_scales


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L
