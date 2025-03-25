import torch
from torch import Tensor

# __forceinline__ __device__ float evaluate_opacity_factor3D_geometric_(
#     const glm::vec3 &o_minus_mu,
#     const glm::vec3 &raydir,
#     const glm::mat3 &giscl_rot,
#     glm::vec3 &grd,
#     glm::vec3 &gro
# ) {
#     gro = giscl_rot * o_minus_mu;
#     grd = safe_normalize(giscl_rot * raydir);

#     const glm::vec3 gcrod = glm::cross(grd, gro);
#     const float grayDist = glm::dot(gcrod, gcrod);
#     return -0.5f * grayDist;

# __forceinline__ __device__ glm::vec3 safe_normalize(glm::vec3 v) {
#     const float l = v.x * v.x + v.y * v.y + v.z * v.z;
#     return l > 0.0f ? (v * rsqrtf(l)) : v;
# }

def safe_normalize(v: Tensor, dl_do: Tensor) -> Tensor:
    l = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
    if l > 0.0:
        il = 1.0 / l.sqrt()
        o = v * il
        dl_dv = dl_do * il - (dl_do * v).sum() * il ** 3 * v
    else:
        o = v
        dl_dv = torch.zeros_like(v)
    return o, dl_dv

def evaluate_opacity_factor3D_geometric_(
    o_minus_mu: Tensor,
    raydir: Tensor,
    giscl_rot: Tensor,
    dl_do: Tensor
):
    gro = giscl_rot @ o_minus_mu
    grd = giscl_rot @ raydir

    # safe normalize
    l = grd[0] ** 2 + grd[1] ** 2 + grd[2] ** 2
    if l > 0.0:
        il = 1.0 / l.sqrt()
        grdn = grd * il
    else:
        grdn = grd
    
    gcrod = torch.cross(grdn, gro, dim=-1)
    grayDist = torch.dot(gcrod, gcrod)
    print ("grayDist", grayDist)
    o = -0.5 * grayDist

    # backprop
    dl_dgrayDist = -0.5 * dl_do
    dl_dgcrod = dl_dgrayDist * 2.0 * gcrod
    dl_dgrdn = - torch.cross(dl_dgcrod, gro, dim=-1)
    dl_dgro = torch.cross(dl_dgcrod, grdn, dim=-1)

    if l > 0.0:
        dl_dgrd = dl_dgrdn * il - (dl_dgrdn * grd).sum() * il ** 3 * grd
    else:
        dl_dgrd = dl_dgrdn

    dl_dgiscl_rot = torch.ger(dl_dgrd, raydir) + torch.ger(dl_dgro, o_minus_mu)
    dl_do_minus_mu = giscl_rot.t() @ dl_dgro

    return o, dl_dgiscl_rot, dl_do_minus_mu

def evaluate_opacity_factor3D_geometric__(
    e: Tensor, # o_minus_mu
    d: Tensor,
    giscl_rot: Tensor,
    dl_do: Tensor
):
    M = giscl_rot.t() @ giscl_rot # (3, 3)

    eM = e.t() @ M
    Md = M @ d

    eMe = eM @ e
    eMd = eM @ d
    dMd = d.t() @ Md

    eMd_div_dMd = eMd / dMd

    # \sigma = (o - \mu)^T  M (o - \mu) - \frac{[(o - \mu)^T M d]^2}{d^T M d}
    sigma = eMe - eMd * eMd_div_dMd
    o = -0.5 * sigma    

    # backprop
    dl_dsigma = -0.5 * dl_do
    dl_dmu = dl_dsigma * 2 * (- M @ e + eMd_div_dMd * Md)
    dl_dM = dl_dsigma * (
        torch.ger(e, e) 
        - 2 * eMd_div_dMd * torch.ger(e, d) 
        + eMd_div_dMd ** 2 * torch.ger(d, d)
    )

    dl_dgiscl_rot = giscl_rot @ (dl_dM + dl_dM.t())
    
    return o, dl_dgiscl_rot, - dl_dmu


o_minus_mu = torch.randn(3, requires_grad=True)
raydir = torch.randn(3, requires_grad=False)
giscl_rot = torch.randn(3, 3, requires_grad=True)
dl_do = torch.randn(1, requires_grad=False)

o, dl_dgiscl_rot, dl_do_minus_mu = evaluate_opacity_factor3D_geometric_(
    o_minus_mu,
    raydir,
    giscl_rot,
    dl_do
)

dl_do_minus_mu_, dl_dgiscl_rot_ = torch.autograd.grad(
    (o * dl_do).sum(),
    (o_minus_mu, giscl_rot),
)

# print ("dl_do_minus_mu", dl_do_minus_mu, dl_do_minus_mu_)
print ("dl_dgiscl_rot", dl_dgiscl_rot, dl_dgiscl_rot_)

evaluate_opacity_factor3D_geometric__(
    o_minus_mu,
    raydir,
    giscl_rot,
    dl_do
)