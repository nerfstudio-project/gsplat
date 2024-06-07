"""Pure PyTorch implementations of various functions"""

import struct

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from typing import Tuple, Literal, Optional


def compute_sh_color(
    viewdirs: Float[Tensor, "*batch 3"],
    sh_coeffs: Float[Tensor, "*batch D C"],
    method: Literal["poly", "fast"] = "fast",
):
    """
    :param viewdirs (*, C)
    :param sh_coeffs (*, D, C) sh coefficients for each color channel
    return colors (*, C)
    """
    *dims, dim_sh, C = sh_coeffs.shape
    if method == "poly":
        bases = eval_sh_bases(dim_sh, viewdirs)  # (*, dim_sh)
    elif method == "fast":
        bases = eval_sh_bases_fast(dim_sh, viewdirs)  # (*, dim_sh)
    else:
        raise RuntimeError(f"Unknown mode: {method} for compute sh color.")
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


def eval_sh_bases_fast(basis_dim: int, dirs: torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit direction for high orders
    using approach described by
    Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
    https://jcgt.org/published/0002/02/06/


    :param basis_dim: int SH basis dim. Currently, only 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)

    See reference C++ code in https://jcgt.org/published/0002/02/06/code.zip
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )

    result[..., 0] = 0.2820947917738781

    if basis_dim <= 1:
        return result

    x, y, z = dirs.unbind(-1)

    fTmpA = -0.48860251190292
    result[..., 2] = 0.4886025119029199 * z
    result[..., 3] = fTmpA * x
    result[..., 1] = fTmpA * y

    if basis_dim <= 4:
        return result

    z2 = z * z
    fTmpB = -1.092548430592079 * z
    fTmpA = 0.5462742152960395
    fC1 = x * x - y * y
    fS1 = 2 * x * y
    result[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    result[..., 7] = fTmpB * x
    result[..., 5] = fTmpB * y
    result[..., 8] = fTmpA * fC1
    result[..., 4] = fTmpA * fS1

    if basis_dim <= 9:
        return result

    fTmpC = -2.285228997322329 * z2 + 0.4570457994644658
    fTmpB = 1.445305721320277 * z
    fTmpA = -0.5900435899266435
    fC2 = x * fC1 - y * fS1
    fS2 = x * fS1 + y * fC1
    result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    result[..., 13] = fTmpC * x
    result[..., 11] = fTmpC * y
    result[..., 14] = fTmpB * fC1
    result[..., 10] = fTmpB * fS1
    result[..., 15] = fTmpA * fC2
    result[..., 9] = fTmpA * fS2

    if basis_dim <= 16:
        return result

    fTmpD = z * (-4.683325804901025 * z2 + 2.007139630671868)
    fTmpC = 3.31161143515146 * z2 - 0.47308734787878
    fTmpB = -1.770130769779931 * z
    fTmpA = 0.6258357354491763
    fC3 = x * fC2 - y * fS2
    fS3 = x * fS2 + y * fC2
    result[..., 20] = (
        1.984313483298443 * z * result[..., 12] + -1.006230589874905 * result[..., 6]
    )
    result[..., 21] = fTmpD * x
    result[..., 19] = fTmpD * y
    result[..., 22] = fTmpC * fC1
    result[..., 18] = fTmpC * fS1
    result[..., 23] = fTmpB * fC2
    result[..., 17] = fTmpB * fS2
    result[..., 24] = fTmpA * fC3
    result[..., 16] = fTmpA * fS3
    return result


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def quat_to_rotmat(quat: Tensor) -> Tensor:
    assert quat.shape[-1] == 4, quat.shape
    return normalized_quat_to_rotmat(F.normalize(quat, dim=-1))


def scale_rot_to_cov3d(scale: Tensor, glob_scale: float, quat: Tensor) -> Tensor:
    assert scale.shape[-1] == 3, scale.shape
    assert quat.shape[-1] == 4, quat.shape
    assert scale.shape[:-1] == quat.shape[:-1], (scale.shape, quat.shape)
    R = normalized_quat_to_rotmat(quat)  # (..., 3, 3)
    M = R * glob_scale * scale[..., None, :]  # (..., 3, 3)
    # TODO: save upper right because symmetric
    return M @ M.transpose(-1, -2)  # (..., 3, 3)


def project_cov3d_ewa(
    mean3d: Tensor,
    cov3d: Tensor,
    viewmat: Tensor,
    fx: float,
    fy: float,
    tan_fovx: float,
    tan_fovy: float,
    is_valid: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    assert mean3d.shape[-1] == 3, mean3d.shape
    assert cov3d.shape[-2:] == (3, 3), cov3d.shape
    assert viewmat.shape[-2:] == (4, 4), viewmat.shape

    if is_valid is None:
        is_valid = torch.ones(mean3d.shape[:-1], dtype=torch.bool, device=mean3d.device)

    W = viewmat[..., :3, :3]  # (..., 3, 3)
    p = viewmat[..., :3, 3]  # (..., 3)
    t = torch.einsum("...ij,...j->...i", W, mean3d[is_valid, ...]) + p  # (..., 3)

    rz = 1.0 / t[..., 2]  # (...,)
    rz2 = rz**2  # (...,)

    lim_x = 1.3 * torch.tensor([tan_fovx], device=mean3d.device)
    lim_y = 1.3 * torch.tensor([tan_fovy], device=mean3d.device)
    x_clamp = t[..., 2] * torch.clamp(t[..., 0] * rz, min=-lim_x, max=lim_x)
    y_clamp = t[..., 2] * torch.clamp(t[..., 1] * rz, min=-lim_y, max=lim_y)
    t = torch.stack([x_clamp, y_clamp, t[..., 2]], dim=-1)

    O = torch.zeros_like(rz)
    J = torch.stack(
        [fx * rz, O, -fx * t[..., 0] * rz2, O, fy * rz, -fy * t[..., 1] * rz2],
        dim=-1,
    ).reshape(*rz.shape, 2, 3)
    T = torch.matmul(J, W)  # (..., 2, 3)
    cov2d = torch.einsum(
        "...ij,...jk,...kl->...il", T, cov3d[is_valid, ...], T.transpose(-1, -2)
    )

    # add a little blur along axes and (TODO save upper triangular elements)
    det_orig = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 0, 1]
    cov2d_blurred = cov2d * 1
    cov2d_blurred[..., 0, 0] = cov2d[..., 0, 0] + 0.3
    cov2d_blurred[..., 1, 1] = cov2d[..., 1, 1] + 0.3
    det_blur = (
        cov2d_blurred[..., 0, 0] * cov2d_blurred[..., 1, 1]
        - cov2d_blurred[..., 0, 1] * cov2d_blurred[..., 0, 1]
    )
    # note: sqrt(x) is not differentiable at x=0
    compensation = torch.sqrt(torch.clamp(det_orig / det_blur, min=1e-10))

    cov2d_all = torch.zeros(cov3d.shape[0], 2, 2, device=cov2d.device)
    compensation_all = torch.zeros(cov3d.shape[0], device=cov2d.device)

    cov2d_all[is_valid, ...] = cov2d_blurred
    compensation_all[is_valid] = compensation
    return cov2d_all, compensation_all


def compute_compensation(cov2d_mat: Tensor):
    """
    params: cov2d matrix (*, 2, 2)
    returns: compensation factor as calculated in project_cov3d_ewa
    """
    det_denom = cov2d_mat[..., 0, 0] * cov2d_mat[..., 1, 1] - cov2d_mat[..., 0, 1] ** 2
    det_nomin = (cov2d_mat[..., 0, 0] - 0.3) * (cov2d_mat[..., 1, 1] - 0.3) - cov2d_mat[
        ..., 0, 1
    ] ** 2
    return torch.sqrt(torch.clamp(det_nomin / det_denom, min=0))


def compute_cov2d_bounds(cov2d_mat: Tensor, cov_valid: Optional[Tensor] = None):
    """
    param: cov2d matrix (*, 2, 2)
    returns: conic parameters (*, 3)
    """
    det_all = cov2d_mat[..., 0, 0] * cov2d_mat[..., 1, 1] - cov2d_mat[..., 0, 1] ** 2
    valid = det_all != 0
    if cov_valid is not None:
        valid = valid & cov_valid
    # det = torch.clamp(det, min=eps)
    det = det_all[valid]
    cov2d = cov2d_mat[valid]
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
    radius_all = torch.zeros(*cov2d_mat.shape[:-2], device=cov2d_mat.device)
    conic_all = torch.zeros(*cov2d_mat.shape[:-2], 3, device=cov2d_mat.device)
    radius_all[valid] = radius
    conic_all[valid] = conic
    return conic_all, radius_all, valid


def project_pix(fxfy, p_view, center, eps=1e-6):
    fx, fy = fxfy
    cx, cy = center

    rw = 1.0 / (p_view[..., 2] + eps)
    p_proj = (p_view[..., 0] * rw, p_view[..., 1] * rw)
    u, v = (p_proj[0] * fx + cx, p_proj[1] * fy + cy)
    return torch.stack([u, v], dim=-1)


def clip_near_plane(p, viewmat, clip_thresh=0.01):
    R = viewmat[:3, :3]
    T = viewmat[:3, 3]
    p_view = torch.einsum("ij,nj->ni", R, p) + T[None]
    return p_view, p_view[..., 2] < clip_thresh


def get_tile_bbox(pix_center, pix_radius, tile_bounds, block_width):
    tile_size = torch.tensor(
        [block_width, block_width], dtype=torch.float32, device=pix_center.device
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
    intrins,
    img_size,
    block_width,
    clip_thresh=0.01,
):
    tile_bounds = (
        (img_size[0] + block_width - 1) // block_width,
        (img_size[1] + block_width - 1) // block_width,
        1,
    )
    fx, fy, cx, cy = intrins
    tan_fovx = 0.5 * img_size[0] / fx
    tan_fovy = 0.5 * img_size[1] / fy
    p_view, is_close = clip_near_plane(means3d, viewmat, clip_thresh)
    cov3d = scale_rot_to_cov3d(scales, glob_scale, quats)
    cov2d, compensation = project_cov3d_ewa(
        means3d, cov3d, viewmat, fx, fy, tan_fovx, tan_fovy, ~is_close
    )
    conic, radius, det_valid = compute_cov2d_bounds(cov2d, ~is_close)
    xys = project_pix((fx, fy), p_view, (cx, cy))
    tile_min, tile_max = get_tile_bbox(xys, radius, tile_bounds, block_width)
    tile_area = (tile_max[..., 0] - tile_min[..., 0]) * (
        tile_max[..., 1] - tile_min[..., 1]
    )
    mask = (tile_area > 0) & (~is_close) & det_valid

    num_tiles_hit = tile_area
    depths = p_view[..., 2]
    radii = radius.to(torch.int32)

    radii = torch.where(~mask, 0, radii)
    conic = torch.where(~mask[..., None], 0, conic)
    xys = torch.where(~mask[..., None], 0, xys)
    cov3d = torch.where(~mask[..., None, None], 0, cov3d)
    cov2d = torch.where(~mask[..., None, None], 0, cov2d)
    compensation = torch.where(~mask, 0, compensation)
    num_tiles_hit = torch.where(~mask, 0, num_tiles_hit)
    depths = torch.where(~mask, 0, depths)

    i, j = torch.triu_indices(3, 3)
    cov3d_triu = cov3d[..., i, j]
    i, j = torch.triu_indices(2, 2)
    cov2d_triu = cov2d[..., i, j]
    return (
        cov3d_triu,
        cov2d_triu,
        xys,
        depths,
        radii,
        conic,
        compensation,
        num_tiles_hit,
        mask,
    )


def map_gaussian_to_intersects(
    num_points, xys, depths, radii, cum_tiles_hit, tile_bounds, block_width
):
    num_intersects = cum_tiles_hit[-1]
    isect_ids = torch.zeros(num_intersects, dtype=torch.int64, device=xys.device)
    gaussian_ids = torch.zeros(num_intersects, dtype=torch.int32, device=xys.device)

    for idx in range(num_points):
        if radii[idx] <= 0:
            break

        tile_min, tile_max = get_tile_bbox(
            xys[idx], radii[idx], tile_bounds, block_width
        )

        cur_idx = 0 if idx == 0 else cum_tiles_hit[idx - 1].item()

        # Get raw byte representation of the float value at the given index
        raw_bytes = struct.pack("f", depths[idx])

        # Interpret those bytes as an int32_t
        depth_id_n = struct.unpack("i", raw_bytes)[0]

        for i in range(tile_min[1], tile_max[1]):
            for j in range(tile_min[0], tile_max[0]):
                tile_id = i * tile_bounds[0] + j
                isect_ids[cur_idx] = (tile_id << 32) | depth_id_n
                gaussian_ids[cur_idx] = idx
                cur_idx += 1

    return isect_ids, gaussian_ids


def get_tile_bin_edges(num_intersects, isect_ids_sorted, tile_bounds):
    tile_bins = torch.zeros(
        (tile_bounds[0] * tile_bounds[1], 2),
        dtype=torch.int32,
        device=isect_ids_sorted.device,
    )

    for idx in range(num_intersects):
        cur_tile_idx = isect_ids_sorted[idx] >> 32

        if idx == 0:
            tile_bins[cur_tile_idx, 0] = 0
            continue

        if idx == num_intersects - 1:
            tile_bins[cur_tile_idx, 1] = num_intersects
            break

        prev_tile_idx = isect_ids_sorted[idx - 1] >> 32

        if cur_tile_idx != prev_tile_idx:
            tile_bins[prev_tile_idx, 1] = idx
            tile_bins[cur_tile_idx, 0] = idx

    return tile_bins


def rasterize_forward(
    tile_bounds,
    block,
    img_size,
    gaussian_ids_sorted,
    tile_bins,
    xys,
    conics,
    colors,
    opacities,
    background,
):
    channels = colors.shape[1]
    out_img = torch.zeros(
        (img_size[1], img_size[0], channels), dtype=torch.float32, device=xys.device
    )
    final_Ts = torch.zeros(
        (img_size[1], img_size[0]), dtype=torch.float32, device=xys.device
    )
    final_idx = torch.zeros(
        (img_size[1], img_size[0]), dtype=torch.int32, device=xys.device
    )
    for i in range(img_size[1]):
        for j in range(img_size[0]):
            tile_id = (i // block[0]) * tile_bounds[0] + (j // block[1])
            tile_bin_start = tile_bins[tile_id, 0]
            tile_bin_end = tile_bins[tile_id, 1]
            T = 1.0

            for idx in range(tile_bin_start, tile_bin_end):
                gaussian_id = gaussian_ids_sorted[idx]
                conic = conics[gaussian_id]
                center = xys[gaussian_id]
                delta = center - torch.tensor(
                    [j, i], dtype=torch.float32, device=xys.device
                )

                sigma = (
                    0.5
                    * (conic[0] * delta[0] * delta[0] + conic[2] * delta[1] * delta[1])
                    + conic[1] * delta[0] * delta[1]
                )

                if sigma < 0:
                    continue

                opac = opacities[gaussian_id]
                alpha = min(0.999, opac * torch.exp(-sigma))

                if alpha < 1 / 255:
                    continue

                next_T = T * (1 - alpha)

                if next_T <= 1e-4:
                    idx -= 1
                    break

                vis = alpha * T

                out_img[i, j] += vis * colors[gaussian_id]
                T = next_T

            final_Ts[i, j] = T
            final_idx[i, j] = idx
            out_img[i, j] += T * background

    return out_img, final_Ts, final_idx
