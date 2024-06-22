import torch
import torch.nn.functional as F
from torch import Tensor

PI = 3.14159265358979323846


def gaussian(
    x: Tensor,  # [..., 3]
    mu: Tensor,  # [3]
    covar: Tensor,  # [3, 3]
) -> Tensor:
    d = x - mu
    perci = torch.inverse(covar)
    scalar = torch.sqrt((2 * PI) ** 3 * torch.det(covar))
    return torch.exp(-0.5 * torch.einsum("...i,ij,...j->...", d, perci, d))


torch.manual_seed(43)
mu = torch.randn(3)
temp = torch.randn(3, 3)
covar = torch.mm(temp, temp.t())

rays_o = torch.randn(3)
rays_d = mu - rays_o
rays_d = rays_d / torch.norm(rays_d)

# integral (numerical)
t = torch.linspace(-10, 10, 10000)
x = rays_o + t[:, None] * rays_d
p = gaussian(x, mu, covar)
delta = t[1] - t[0]
integral = torch.sum(p * delta)
print(integral)

# integral (analytical)
a = torch.einsum("...i,...ij,...j->...", rays_d, torch.inverse(covar), rays_d)
integral = torch.sqrt(2 * PI / a)
print(integral)

rays_o = torch.randn(3)
rays_d = mu - rays_o
rays_d = rays_d / torch.norm(rays_d)
a = torch.einsum("...i,...ij,...j->...", rays_d, torch.inverse(covar), rays_d)
integral = torch.sqrt(2 * PI / a)
print(integral)


def density_to_opacity(
    means: Tensor,  # [N, 3]
    density: Tensor,  # [N]
    perci: Tensor,  # [N, 3, 3]
    camtoworlds: Tensor,  # [C, 4, 4]
):
    assert camtoworlds.shape[0] == 1
    rays_o = camtoworlds[0, :3, 3]  # [3]
    rays_d = F.normalize(means - rays_o, dim=-1)  # [N, 3]
    a = torch.einsum("ni,nij,nj->n", rays_d, perci, rays_d)  # [N]
    integral = torch.sqrt(2 * PI / a)  # [N]
    return 1.0 - torch.exp(-integral * density)
