import math
import torch
from .cuda._wrapper import _make_lazy_cuda_func

N_max = 51
binoms = torch.zeros((N_max, N_max)).float().cuda()
for n in range(N_max):
    for k in range(n + 1):
        binoms[n, k] = math.comb(n, k)


def compute_relocation_cuda(opacity_old, scale_old, N):
    new_opacity, new_scale = _make_lazy_cuda_func("compute_relocation")(
        opacity_old, scale_old, N.int(), binoms, N_max
    )
    return new_opacity, new_scale


def sample_alives(probs, num, alive_indices=None):
    probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
    sampled_idxs = torch.multinomial(probs, num, replacement=True)
    if alive_indices is not None:
        sampled_idxs = alive_indices[sampled_idxs]
    ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
    return sampled_idxs, ratio


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
