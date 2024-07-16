# inline __device__ T add_blur(const T eps2d, mat3<T> &covar, T &compensation) {
#     T det_orig = glm::determinant(covar);
#     covar[0][0] += eps2d;
#     covar[1][1] += eps2d;
#     covar[2][2] += eps2d;
#     T det_blur = glm::determinant(covar);
#     compensation = sqrt(max(0.f, det_orig / det_blur));
#     return det_blur;
# }

import torch
from torch import Tensor


def add_blur(eps2d: float, covar: Tensor):
    det_orig = covar.det()
    covar = covar + torch.eye(3) * eps2d
    det_blur = covar.det()
    compensation = det_orig / det_blur
    return compensation


def add_blur_vjp(
    eps2d: float, covar: Tensor, compensation: Tensor, v_compensation: Tensor
):
    M = covar
    MaI = M + torch.eye(3) * eps2d

    conic_blur = torch.inverse(MaI)
    det_conic_blur = conic_blur.det()
    v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6)
    one_minus_sqr_comp = 1.0 - compensation * compensation

    v_covar = v_sqr_comp * (
        one_minus_sqr_comp * conic_blur - eps2d * det_conic_blur * torch.eye(3)
    )

    v_covar = (
        v_compensation
        * (M.inverse().t() - MaI.inverse().t())
        # * M.det()
        # / (MaI.det())
        * compensation
    )
    return v_covar


tmp = torch.randn(3, 3)
covar = tmp @ tmp.t()
covar.requires_grad = True

eps2d = 0.3

compensation = add_blur(eps2d, covar)
print(compensation)

v_compensation = torch.randn_like(compensation)

v_covar = torch.autograd.grad(compensation, covar, v_compensation, create_graph=True)[0]
print(v_covar)

_v_covar = add_blur_vjp(eps2d, covar, compensation, v_compensation)
print(_v_covar)
