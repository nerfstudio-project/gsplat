"""Pure PyTorch implementation"""

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


def scale_rot_to_cov3d(
    scale: Tensor, glob_scale: float, quat: Tensor
) -> Tensor:
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
            torch.stack(
                [fx * rz, torch.zeros_like(rz), -fx * t[..., 0] * rz2], dim=-1
            ),
            torch.stack(
                [torch.zeros_like(rz), fy * rz, -fy * t[..., 1] * rz2], dim=-1
            ),
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
