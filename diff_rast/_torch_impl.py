"""Pure PyTorch implementation"""

from jaxtyping import Float
import torch
import torch.nn.functional as F
from torch import Tensor


def quat_to_rotmat(quat: Tensor) -> Tensor:
    assert quat.shape[-1] == 4, quat.shape
    x, y, z, w = torch.unbind(F.normalize(quat, dim=-1), dim=-1)
    return torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * (y**2 + z**2),
                    2 * (x * y - w * z),
                    2 * (x * z + w * y),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (x * y + w * z),
                    1 - 2 * (x**2 + z**2),
                    2 * (y * z - w * x),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (x * z - w * y),
                    2 * (y * z + w * x),
                    1 - 2 * (x**2 + y**2),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )


def scale_rot_to_cov3d(scale: Tensor, glob_scale: float, quat: Tensor) -> Tensor:
    assert scale.shape[-1] == 3, scale.shape
    assert quat.shape[-1] == 4, quat.shape
    assert scale.shape[:-1] == quat.shape[:-1], (scale.shape, quat.shape)
    R = quat_to_rotmat(quat)  # (..., 3, 3)
    M = R * glob_scale * scale[..., None, :]  # (..., 3, 3)
    # TODO: save upper right because symmetric
    return M @ M.transpose(-1, -2)  # (..., 3, 3)


def project_cov3d_ewa(
    mean3d: Tensor, cov3d: Tensor, viewmat: Tensor, fx: float, fy: float
) -> Tensor:
    assert mean3d.shape[-1] == 3, mean3d.shape
    assert cov3d.shape[-2:] == (3, 3), cov3d.shape
    assert viewmat.shape[-2:] == (4, 4), viewmat.shape
    W = viewmat[..., :3, :3]  # (..., 3, 3)
    p = viewmat[..., :3, 3]  # (..., 3)
    t = torch.matmul(W, mean3d[..., None])[..., 0] + p  # (..., 3)
    rz = 1.0 / t[..., 2]  # (...,)
    rz2 = rz**2  # (...,)
    J = torch.stack(
        [
            torch.stack([fx * rz, torch.zeros_like(rz), -fx * t[..., 0] * rz2], dim=-1),
            torch.stack([torch.zeros_like(rz), fy * rz, -fy * t[..., 1] * rz2], dim=-1),
        ],
        dim=-2,
    )  # (..., 2, 3)
    T = J @ W  # (..., 2, 3)
    cov2d = T @ cov3d @ T.transpose(-1, -2)  # (..., 2, 2)
    # add a little blur along axes and (TODO save upper triangular elements)
    cov2d[..., 0, 0] = cov2d[..., 0, 0] + 0.1
    cov2d[..., 1, 1] = cov2d[..., 1, 1] + 0.1
    return cov2d


def compute_cov2d_bounds(cov2d: Tensor, eps=1e-6):
    det = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] ** 2
    det = torch.clamp(det, min=eps)
    conic = torch.stack(
        [
            cov2d[..., 1, 1] / det,
            -cov2d[..., 0, 1] / det,
            cov2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # (..., 3)
    b = (cov2d[..., 0, 0] + cov2d[..., 1, 1]) / 2  # (...,)
    v1 = b + torch.sqrt(torch.clamp(b**2 - det, min=0.1))  # (...,)
    v2 = b - torch.sqrt(torch.clamp(b**2 - det, min=0.1))  # (...,)
    radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))  # (...,)
    return conic, radius, det > eps


def ndc2pix(x, W):
    return 0.5 * ((x + 1.0) * W - 1.0)


def project_pix(mat, p, img_size, eps=1e-6):
    p_hom = F.pad(p, (0, 1), value=1.0)
    p_hom = torch.einsum("...ij,...j->...i", mat, p_hom)
    rw = 1.0 / torch.clamp(p_hom[..., 3], min=eps)
    p_proj = p_hom[..., :3] * rw[..., None]
    u = ndc2pix(p_proj[..., 0], img_size[0])
    v = ndc2pix(p_proj[..., 1], img_size[1])
    return torch.stack([u, v], dim=-1)


def clip_near_plane(p, viewmat, thresh=0.1):
    R = viewmat[..., :3, :3]
    T = viewmat[..., :3, 3]
    p_view = torch.matmul(R, p[..., None])[..., 0] + T
    return p_view, p_view[..., 2] < thresh


def get_tile_bbox(pix_center, pix_radius, tile_bounds, BLOCK_X=16, BLOCK_Y=16):
    tile_size = torch.tensor(
        [BLOCK_X, BLOCK_Y], dtype=torch.float32, device=pix_center.device
    )
    tile_center = pix_center / tile_size
    tile_radius = pix_radius[..., None] / tile_size

    top_left = (tile_center - tile_radius).to(torch.int32)
    bottom_right = (tile_center + tile_radius).to(torch.int32) + 1
    tile_min = torch.stack(
        [
            torch.clamp(top_left[..., 0], 0, tile_bounds[0]),
            torch.clamp(top_left[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    tile_max = torch.stack(
        [
            torch.clamp(bottom_right[..., 0], 0, tile_bounds[0]),
            torch.clamp(bottom_right[..., 1], 0, tile_bounds[1]),
        ],
        -1,
    )
    return tile_min, tile_max


def project_gaussians_forward(
    means3d,
    scales,
    glob_scale,
    quats,
    viewmat,
    projmat,
    fx,
    fy,
    img_size,
    tile_bounds,
):
    p_view, is_close = clip_near_plane(means3d, viewmat)
    cov3d = scale_rot_to_cov3d(scales, glob_scale, quats)
    cov2d = project_cov3d_ewa(means3d, cov3d, viewmat, fx, fy)
    conic, radius, det_valid = compute_cov2d_bounds(cov2d)
    center = project_pix(projmat, means3d, img_size)
    tile_min, tile_max = get_tile_bbox(center, radius, tile_bounds)
    tile_area = (tile_max[..., 0] - tile_min[..., 0]) * (
        tile_max[..., 1] - tile_min[..., 1]
    )
    mask = (tile_area > 0) & (~is_close) & det_valid

    num_tiles_hit = tile_area
    depths = p_view[..., 2]
    radii = radius.to(torch.int32)
    xys = center
    conics = conic

    return cov3d, xys, depths, radii, conics, num_tiles_hit, mask


def compute_sh_color(
    viewdirs: Float[Tensor, "*batch 3"], sh_coeffs: Float[Tensor, "*batch D C"]
):
    """
    :param viewdirs (*, C)
    :param sh_coeffs (*, D, C) sh coefficients for each color channel
    return colors (*, C)
    """
    *dims, dim_sh, C = sh_coeffs.shape
    bases = eval_sh_bases(dim_sh, viewdirs)  # (*, dim_sh)
    return (bases[..., None] * sh_coeffs).sum(dim=-2)


"""
Taken from https://github.com/sxyu/svox2
"""

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

MAX_SH_BASIS = 10


def eval_sh_bases(basis_dim: int, dirs: torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y
        result[..., 2] = SH_C1 * z
        result[..., 3] = -SH_C1 * x
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy
            result[..., 5] = SH_C2[1] * yz
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = SH_C2[3] * xz
            result[..., 8] = SH_C2[4] * (xx - yy)

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy)
                result[..., 10] = SH_C3[1] * xy * z
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
                result[..., 14] = SH_C3[5] * z * (xx - yy)
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy)

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy)
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
                    result[..., 24] = SH_C4[8] * (
                        xx * (xx - 3 * yy) - yy * (3 * xx - yy)
                    )
    return result
