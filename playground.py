import torch
from nerfacc import accumulate_along_rays, exclusive_prod, exclusive_sum
from torch import Tensor


def distortion(
    weights: Tensor, t_mids: Tensor, ray_indices: Tensor, n_rays: int
) -> Tensor:
    """Distortion Regularization proposed in Mip-NeRF 360."""
    loss_bi_0 = weights * t_mids * exclusive_sum(weights, indices=ray_indices)
    loss_bi_1 = weights * exclusive_sum(weights * t_mids, indices=ray_indices)
    loss_bi = 2 * (loss_bi_0 - loss_bi_1)
    loss = accumulate_along_rays(loss_bi, None, ray_indices, n_rays)
    return loss


device = "cuda"
alphas = torch.tensor([0.3, 0.2, 0.5, 0.4], device=device, requires_grad=True)
weights = alphas * exclusive_prod(1 - alphas)
# print("T", exclusive_prod(1 - alphas))
# print("weights", weights)
t_mids = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device, requires_grad=True)
ray_indices = torch.tensor([0, 0, 0, 0], device=device)
n_rays = 1

loss = distortion(weights, t_mids, ray_indices, n_rays)
# print("loss", loss)
v_alphas, v_t = torch.autograd.grad(loss, (alphas, t_mids))
print("v_alphas", v_alphas, "v_t", v_t)

accum_depths = (t_mids * weights).sum()
accum_weights = weights.sum()


@torch.no_grad()
def backward(alphas: Tensor, t_mids: Tensor, accum_d: Tensor, accum_w: Tensor):
    v_alphas = torch.zeros_like(alphas)
    v_t = torch.zeros_like(t_mids)

    n = len(alphas)
    # kernel
    accum_d_buffer = accum_d.clone()
    accum_w_buffer = accum_w.clone()
    distort_buffer = 0.0
    T = 1.0 - accum_w  # the T beyond last sample

    # from back to front
    for i, (a, t) in enumerate(zip(alphas.flip(0), t_mids.flip(0))):
        ra = 1.0 / (1.0 - a)
        T *= ra
        w = a * T
        dl_dw = 2.0 * (
            2 * (t * accum_w_buffer - accum_d_buffer) + (accum_d - t * accum_w)
        )
        v_alphas[n - 1 - i] = dl_dw * T - distort_buffer / (1 - a)
        accum_d_buffer -= w * t
        accum_w_buffer -= w
        distort_buffer += dl_dw * w

        v_t[n - 1 - i] = 2 * w * (2 - 2 * T - accum_w + w)
    return v_alphas, v_t


v_alphas, v_t = backward(alphas, t_mids, accum_depths, accum_weights)
print("v_alphas", v_alphas, "v_t", v_t)
