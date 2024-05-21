import struct
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def _quat_scale_to_covar_preci(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
):
    """PyTorch implementation."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack(
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

    R = R.reshape(quats.shape[:-1] + (3, 3))  # (..., 3, 3)
    # R.register_hook(lambda grad: print("grad R", grad))

    if compute_covar:
        M = R * scales[..., None, :]  # (..., 3, 3)
        covars = torch.bmm(M, M.transpose(-1, -2))  # (..., 3, 3)
        if triu:
            covars = covars.reshape(covars.shape[:-2] + (9,))  # (..., 9)
            covars = (
                covars[..., [0, 1, 2, 4, 5, 8]] + covars[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0  # (..., 6)
    if compute_preci:
        P = R * (1 / scales[..., None, :])  # (..., 3, 3)
        precis = torch.bmm(P, P.transpose(-1, -2))  # (..., 3, 3)
        if triu:
            precis = precis.reshape(precis.shape[:-2] + (9,))
            precis = (
                precis[..., [0, 1, 2, 4, 5, 8]] + precis[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0

    return covars if compute_covar else None, precis if compute_preci else None


def _persp_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """PyTorch implementation."""
    C, N, _ = means.shape

    tx, ty, tz = torch.unbind(means, dim=-1)  # [C, N]
    tz2 = tz**2  # [C, N]

    fx = Ks[..., 0, 0, None]  # [C, 1]
    fy = Ks[..., 1, 1, None]  # [C, 1]
    tan_fovx = 0.5 * width / fx  # [C, 1]
    tan_fovy = 0.5 * height / fy  # [C, 1]

    lim_x = 1.3 * tan_fovx
    lim_y = 1.3 * tan_fovy
    tx = tz * torch.clamp(tx / tz, min=-lim_x, max=lim_x)
    ty = tz * torch.clamp(ty / tz, min=-lim_y, max=lim_y)

    O = torch.zeros((C, N), device=means.device, dtype=means.dtype)
    J = torch.stack(
        [fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], dim=-1
    ).reshape(C, N, 2, 3)

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = torch.einsum("cij,cnj->cni", Ks[:, :2, :3], means)  # [C, N, 2]
    means2d = means2d / tz[..., None]  # [C, N, 2]
    return means2d, cov2d  # [C, N, 2], [C, N, 2, 2]


def _world_to_cam(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation."""
    R = viewmats[:, :3, :3]  # [C, 3, 3]
    t = viewmats[:, :3, 3]  # [C, 3]
    means_c = torch.einsum("cij,nj->cni", R, means) + t[:, None, :]  # (C, N, 3)
    covars_c = torch.einsum("cij,njk,clk->cnil", R, covars, R)  # [C, N, 3, 3]
    return means_c, covars_c


def _projection(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    calc_compensations: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """PyTorch implementation."""
    means_c, covars_c = _world_to_cam(means, covars, viewmats)
    means2d, covars2d = _persp_proj(means_c, covars_c, Ks, width, height)
    det_orig = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d

    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    det = det.clamp(min=1e-10)

    if calc_compensations:
        compensations = torch.sqrt(torch.clamp(det_orig / det, min=0.0))
    else:
        compensations = None

    conics = torch.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # [C, N, 3]

    depths = means_c[..., 2]  # [C, N]

    b = (covars2d[..., 0, 0] + covars2d[..., 1, 1]) / 2  # (...,)
    v1 = b + torch.sqrt(torch.clamp(b**2 - det, min=0.01))  # (...,)
    radius = torch.ceil(3.0 * torch.sqrt(v1))  # (...,)
    # v2 = b - torch.sqrt(torch.clamp(b**2 - det, min=0.01))  # (...,)
    # radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))  # (...,)

    valid = (det > 0) & (depths > near_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius > 0)
        & (means2d[..., 0] - radius < width)
        & (means2d[..., 1] + radius > 0)
        & (means2d[..., 1] - radius < height)
    )
    radius[~inside] = 0.0

    radii = radius.int()
    return radii, means2d, depths, conics, compensations


@torch.no_grad()
def _isect_tiles(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pytorch implementation"""
    C, N = means2d.shape[:2]
    device = means2d.device

    # compute tiles_per_gauss
    tile_means2d = means2d / tile_size
    tile_radii = radii / tile_size
    tile_mins = torch.floor(tile_means2d - tile_radii[..., None]).int()
    tile_maxs = torch.ceil(tile_means2d + tile_radii[..., None]).int()
    tile_mins[..., 0] = torch.clamp(tile_mins[..., 0], 0, tile_width)
    tile_mins[..., 1] = torch.clamp(tile_mins[..., 1], 0, tile_height)
    tile_maxs[..., 0] = torch.clamp(tile_maxs[..., 0], 0, tile_width)
    tile_maxs[..., 1] = torch.clamp(tile_maxs[..., 1], 0, tile_height)
    tiles_per_gauss = (tile_maxs - tile_mins).prod(dim=-1)  # [C, N]
    tiles_per_gauss *= radii > 0.0

    n_isects = tiles_per_gauss.sum().item()
    isect_ids = torch.empty(n_isects, dtype=torch.int64, device=device)
    gauss_ids = torch.empty(n_isects, dtype=torch.int32, device=device)

    cum_tiles_per_gauss = torch.cumsum(tiles_per_gauss.flatten(), dim=0)
    tile_n_bits = (tile_width * tile_height).bit_length()

    def binary(num):
        return "".join("{:0>8b}".format(c) for c in struct.pack("!f", num))

    def kernel(cam_id, gauss_id):
        if radii[cam_id, gauss_id] <= 0.0:
            return
        index = cam_id * N + gauss_id
        curr_idx = cum_tiles_per_gauss[index - 1] if index > 0 else 0

        depth_id = struct.unpack("i", struct.pack("f", depths[cam_id, gauss_id]))[0]

        tile_min = tile_mins[cam_id, gauss_id]
        tile_max = tile_maxs[cam_id, gauss_id]
        for y in range(tile_min[1], tile_max[1]):
            for x in range(tile_min[0], tile_max[0]):
                tile_id = y * tile_width + x
                isect_ids[curr_idx] = (
                    (cam_id << 32 << tile_n_bits) | (tile_id << 32) | depth_id
                )
                gauss_ids[curr_idx] = gauss_id
                curr_idx += 1

    for cam_id in range(C):
        for gauss_id in range(N):
            kernel(cam_id, gauss_id)

    if sort:
        isect_ids, sort_indices = torch.sort(isect_ids)
        gauss_ids = gauss_ids[sort_indices]

    return tiles_per_gauss.int(), isect_ids, gauss_ids


@torch.no_grad()
def _isect_offset_encode(
    isect_ids: Tensor, C: int, tile_width: int, tile_height: int
) -> Tensor:
    """Pytorch implementation"""
    tile_n_bits = (tile_width * tile_height).bit_length()

    n_isects = len(isect_ids)
    device = isect_ids.device
    tile_counts = torch.zeros(
        (C, tile_height, tile_width), dtype=torch.int64, device=device
    )

    isect_ids_uq, counts = torch.unique_consecutive(isect_ids >> 32, return_counts=True)

    cam_ids_uq = isect_ids_uq >> tile_n_bits
    tile_ids_uq = isect_ids_uq & ((1 << tile_n_bits) - 1)
    tile_ids_x_uq = tile_ids_uq % tile_width
    tile_ids_y_uq = tile_ids_uq // tile_width

    tile_counts[cam_ids_uq, tile_ids_y_uq, tile_ids_x_uq] = counts

    cum_tile_counts = torch.cumsum(tile_counts.flatten(), dim=0).reshape_as(tile_counts)
    offsets = cum_tile_counts - tile_counts
    return offsets.int()


def accumulate(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    opacities: Tensor,  # [C, N]
    colors: Tensor,  # [C, N, channels]
    gauss_ids: Tensor,  # [M]
    pixel_ids: Tensor,  # [M]
    camera_ids: Tensor,  # [M]
    image_width: int,
    image_height: int,
    prefix_trans: Tensor = None,  # [C, image_height, image_width]
    errmaps: Tensor = None,  # [C, image_height, image_width]
):
    try:
        from nerfacc import accumulate_along_rays, render_weight_from_alpha
    except ImportError:
        raise ImportError(
            "Please install nerfacc package: git+https://github.com/nerfstudio-project/nerfacc"
        )

    C, N = means2d.shape[:2]
    channels = colors.shape[-1]

    pixel_ids_x = pixel_ids % image_width
    pixel_ids_y = pixel_ids // image_width
    pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1) + 0.5  # [M, 2]
    deltas = pixel_coords - means2d[camera_ids, gauss_ids]  # [M, 2]
    c = conics[camera_ids, gauss_ids]  # [M, 3]
    sigmas = (
        0.5 * (c[:, 0] * deltas[:, 0] ** 2 + c[:, 2] * deltas[:, 1] ** 2)
        + c[:, 1] * deltas[:, 0] * deltas[:, 1]
    )  # [M]
    alphas = torch.clamp_max(
        opacities[camera_ids, gauss_ids] * torch.exp(-sigmas), 0.999
    )

    if prefix_trans is not None:
        prefix_trans = prefix_trans[camera_ids, pixel_ids_y, pixel_ids_x]
    indices = (camera_ids * image_height * image_width + pixel_ids).long()
    total_pixels = C * image_height * image_width

    weights, trans = render_weight_from_alpha(
        alphas, ray_indices=indices, n_rays=total_pixels, prefix_trans=prefix_trans
    )
    renders = accumulate_along_rays(
        weights, colors[camera_ids, gauss_ids], ray_indices=indices, n_rays=total_pixels
    ).reshape(C, image_height, image_width, channels)
    accs = accumulate_along_rays(
        weights, None, ray_indices=indices, n_rays=total_pixels
    ).reshape(C, image_height, image_width, 1)

    if errmaps is not None:
        errs_fl = errmaps[camera_ids, pixel_ids_y, pixel_ids_x]  # [M]
        errs = accumulate_along_rays(
            weights, errs_fl[:, None], ray_indices=gauss_ids, n_rays=N
        )
        ids = torch.where(gauss_ids == 1)[0]
        errs = errs.squeeze(-1)
        return renders, accs, errs
    else:
        return renders, accs, None


def _rasterize_to_pixels(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    colors: Tensor,  # [C, N, channels]
    opacities: Tensor,  # [C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    gauss_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    batch_per_iter: int = 100,
):
    from ._wrapper import rasterize_to_indices_iter

    C, N = means2d.shape[:2]
    device = means2d.device

    render_colors = torch.zeros(
        (C, image_height, image_width, colors.shape[-1]), device=device
    )
    render_alphas = torch.zeros((C, image_height, image_width, 1), device=device)

    # Split Gaussians into batches and iteratively accumulate the renderings
    block_size = tile_size * tile_size
    max_range = (isect_offsets[1:] - isect_offsets[:-1]).max().item()
    num_batches = (max_range + block_size - 1) // block_size
    for step in range(0, num_batches, batch_per_iter):
        transmittances = 1.0 - render_alphas[..., 0]

        # Find the M intersections between pixels and gaussians.
        # Each intersection corresponds to a tuple (gs_id, pixel_id, camera_id)
        gs_ids, indices = rasterize_to_indices_iter(
            step,
            step + batch_per_iter,
            transmittances,
            means2d,
            conics,
            opacities,
            image_width,
            image_height,
            tile_size,
            isect_offsets,
            gauss_ids,
        )  # [M], [M]
        pixel_ids = indices % (image_width * image_height)
        camera_ids = indices // (image_width * image_height)
        if len(gs_ids) == 0:
            break

        # Accumulate the renderings within this batch of Gaussians.
        renders_step, accs_step, errs_step = accumulate(
            means2d,
            conics,
            opacities,
            colors,
            gs_ids,
            pixel_ids,
            camera_ids,
            image_width,
            image_height,
            prefix_trans=transmittances,
        )
        render_colors = render_colors + renders_step
        render_alphas = render_alphas + accs_step

    render_alphas = render_alphas
    if backgrounds is not None:
        render_colors = render_colors + backgrounds[:, None, None, :] * (
            1.0 - render_alphas
        )

    return render_colors, render_alphas


def _eval_sh_bases_fast(basis_dim: int, dirs: torch.Tensor):
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
    result[..., 2] = -fTmpA * z
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


def _spherical_harmonics(
    degree: int,
    dirs: torch.Tensor,  # [..., 3]
    coeffs: torch.Tensor,  # [..., K, 3]
):
    num_bases = (degree + 1) ** 2
    bases = _eval_sh_bases_fast(num_bases, dirs)
    bases = torch.zeros_like(coeffs[..., 0])
    bases[..., :num_bases] = _eval_sh_bases_fast(num_bases, dirs)
    return (bases[..., None] * coeffs).sum(dim=-2)
