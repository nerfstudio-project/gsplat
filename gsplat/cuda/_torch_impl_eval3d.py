# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch reference implementations for eval3d rasterization functions.

This module contains ray-based 3D Gaussian evaluation functions that compute
Gaussian responses in 3D world space by casting rays from the camera through
each pixel.

These functions are separated from the main _torch_impl.py for better code
organization and maintainability.
"""

import math
from typing import Optional, Tuple
from torch import Tensor

import torch

from ._math import (
    _numerically_stable_norm2,
    _quat_inverse,
    _quat_to_rotmat,
    _safe_normalize,
    _quat_scale_to_preci_half,
)
from ._torch_cameras import (
    _viewmat_to_pose,
    _pose_camera_world_position,
    _PerfectPinholeCameraModel,
    _interpolate_shutter_pose,
)
from ._wrapper import RollingShutterType
from ._constants import (
    ALPHA_THRESHOLD,
    TRANSMITTANCE_THRESHOLD,
    MAX_KERNEL_DENSITY_CUTOFF,
)


def _generate_rays_from_pixels(
    pixel_coords: Tensor,  # [..., N, 2]
    cam_centers: Tensor,  # [..., N, 3]
    R_cam_to_world: Tensor,  # [..., N, 3, 3]
    Ks: Tensor,  # [..., N, 3, 3]
    means_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """
    Generate ray origins and directions from pixel coordinates.

    Args:
        pixel_coords: Pixel coordinates [..., N, 2] as (x, y) - already at pixel centers
        cam_centers: Camera centers for each ray [..., N, 3]
        R_cam_to_world: Rotation matrices for each ray [..., N, 3, 3]
        Ks: Camera intrinsics for each ray [..., N, 3, 3]
        means_dtype: Data type for computation

    Returns:
        ray_o: [..., N, 3] - Ray origins
        ray_d: [..., N, 3] - Normalized ray directions
    """
    # Extract camera intrinsics
    fx = Ks[..., 0, 0]
    fy = Ks[..., 1, 1]
    cx = Ks[..., 0, 2]
    cy = Ks[..., 1, 2]

    # Extract pixel coordinates
    px = pixel_coords[..., 0]
    py = pixel_coords[..., 1]

    # Compute ray directions in camera space
    ray_d_cam = torch.stack(
        [
            (px - cx) / fx,
            (py - cy) / fy,
            torch.ones_like(px, dtype=means_dtype),
        ],
        dim=-1,
    )  # [..., N, 3]

    # Normalize ray directions
    ray_d_cam = _safe_normalize(ray_d_cam)  # [..., N, 3]

    # Transform rays to world space: R @ v
    ray_d = torch.matmul(R_cam_to_world, ray_d_cam[..., None]).squeeze(-1)  # [..., 3]

    return cam_centers, ray_d


def _compute_gaussian_transform(
    quats_flat: Tensor,  # [..., N, 4]
    scales_flat: Tensor,  # [..., N, 3]
) -> Tensor:
    """
    Compute Gaussian transformation matrices for ray-based evaluation.

    Transforms Gaussians from (rotation, scale) representation to
    inverse scale-rotation matrices (M^T) used for efficient ray-Gaussian
    distance computation in world space.

    Args:
        quats_flat: Gaussian rotations [..., N, 4]
        scales_flat: Gaussian scales [..., N, 3]

    Returns:
        iscl_rot: [..., N, 3, 3] - Transposed inverse scale-rotation matrices
    """
    # Validate scales
    min_scale = scales_flat.min()
    assert min_scale > 0, f"Non-positive scale detected: {min_scale}"

    # Compute M = R * (1/scales)
    M_preci_half = _quat_scale_to_preci_half(quats_flat, scales_flat)
    assert torch.all(torch.isfinite(M_preci_half)), "Non-finite values in M_preci_half"

    # Transpose: iscl_rot = M^T
    iscl_rot = M_preci_half.transpose(-2, -1)

    return iscl_rot


def _compute_ray_gaussian_distance(
    ray_o: Tensor,  # [..., N, 3]
    ray_d: Tensor,  # [..., N, 3]
    xyz: Tensor,  # [..., N, 3]
    iscl_rot: Tensor,  # [..., N, 3, 3]
    scale: Tensor,  # [..., N, 3]
) -> Tuple[Tensor, Tensor]:
    """
    Compute squared distance from ray to Gaussian center in transformed space,
    and hit distance in camera space.

    Uses the inverse scale-rotation matrix to transform the ray into
    Gaussian-local space, then computes the minimum distance from the
    ray to the Gaussian center.

    Args:
        ray_o: Ray origins [..., N, 3]
        ray_d: Ray directions [..., N, 3] (normalized)
        xyz: Gaussian centers [..., N, 3]
        iscl_rot: Inverse scale-rotation matrices [..., N, 3, 3]

    Returns:
        grayDist: [..., N] - Squared distance from ray to Gaussian center
        hitDist: [..., N] - Hit distance in camera space
    """
    # Transform ray origin and direction to Gaussian space
    gro = torch.matmul(iscl_rot, (ray_o - xyz)[..., None]).squeeze(-1)  # [..., 3]
    grd = torch.matmul(iscl_rot, ray_d[..., None]).squeeze(-1)  # [..., 3]

    # Safe normalization
    grd = _safe_normalize(grd)  # [..., 3]

    # Compute distance via cross product
    gcrod = torch.linalg.cross(grd, gro)  # [..., 3]
    grayDist = torch.sum(gcrod * gcrod, dim=-1)  # [...]

    # Compute hit distance (matches CUDA: hit_t = dot(grd, -gro), grds = scale * grd * hit_t)
    hit_t = torch.sum(grd * (-gro), dim=-1)  # [...]
    grds = scale * (grd * hit_t[..., None])  # [..., 3]
    hitDist = torch.linalg.vector_norm(grds, dim=-1)  # [...]

    return grayDist, hitDist


def _compute_gaussian_alphas(
    grayDist: Tensor,  # [..., M]
    opac: Tensor,  # [..., M]
    trans_threshold: float,
) -> Tensor:
    """
    Compute Gaussian alpha values from ray-Gaussian distances.

    Converts ray-Gaussian distances to alpha values using the Gaussian
    exponential function with clamping to prevent numerical issues.

    Args:
        grayDist: Squared distances from rays to Gaussian centers [..., M]
        opac: Gaussian opacities [..., M]
        trans_threshold: Transmittance threshold for early termination

    Returns:
        alphas: [..., M] - Alpha values
    """
    # Alpha clamp: max_alpha = 1 - sqrt(trans_threshold)
    # Prevents numerical issues:
    # 1. Avoids alpha=1.0 which would make T=0 and kill gradient flow
    # 2. Prevents gradient singularities in prod(1-alpha) computation
    # 3. Ensures transmittance must reach max_alpha once before termination
    max_alpha = 1.0 - math.sqrt(trans_threshold)

    # Gaussian response
    power = -0.5 * grayDist
    max_response = torch.exp(power)
    alphas = torch.clamp(opac * max_response, max=max_alpha)

    assert torch.all(
        (alphas >= 0) & (alphas <= 1.0)
    ), f"Invalid alphas: range=[{alphas.min()}, {alphas.max()}]"

    return alphas, max_response


def accumulate_eval3d(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,  # [..., N, channels]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    gaussian_ids: Tensor,  # [M]
    pixel_ids: Tensor,  # [M]
    image_ids: Tensor,  # [M]
    image_width: int,
    image_height: int,
    flatten_idx: Tensor,  # [M] - index in original flatten_ids
    rs_type: RollingShutterType = RollingShutterType.GLOBAL,  # Rolling shutter type
    rays: Optional[Tensor] = None,  # [..., C, P, 6]
    viewmats_rs: Optional[Tensor] = None,  # [..., 4, 4] - optional for rolling shutter
    base_transmittance: Optional[
        Tensor
    ] = None,  # [I, image_height, image_width] - base transmittance for batched accumulation
    use_hit_distance: bool = False,
    return_normals: bool = False,  # Whether to compute normals
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Alpha compositing with ray-based 3D Gaussian evaluation in Pure PyTorch.

    Similar to accumulate(), but computes Gaussian responses in 3D world space
    by casting rays from the camera through each pixel.

    .. warning::
        This function requires the `nerfacc` package to be installed.

    Args:
        means: Gaussian means in 3D world space. [..., N, 3]
        quats: Gaussian rotations as quaternions. [..., N, 4]
        scales: Gaussian scales. [..., N, 3]
        opacities: Gaussian opacities. [..., N]
        colors: Gaussian colors. [..., N, channels]
        viewmats: Camera view matrices (start pose for rolling shutter). [..., 4, 4]
        Ks: Camera intrinsics. [..., 3, 3]
        gaussian_ids: Gaussian indices for intersections. [M]
        pixel_ids: Pixel indices (row-major) for intersections. [M]
        image_ids: Image indices for intersections. [M]
        image_width: Image width.
        image_height: Image height.
        flatten_idx: Index in original flatten_ids [M]
        rs_type: Rolling shutter type
        viewmats_rs: Optional end pose for rolling shutter. [..., 4, 4]
        base_transmittance: Optional base transmittance for batched accumulation.
                           Shape: [I, image_height, image_width]. If provided,
                           this is used as the starting transmittance for filtering.

    Returns:
        - **renders**: Accumulated colors. [..., image_height, image_width, channels]
        - **alphas**: Accumulated opacities. [..., image_height, image_width, 1]
        - **last_ids**: Last flatten_idx per pixel. [..., image_height, image_width]
        - **sample_counts**: Number of samples per pixel. [..., image_height, image_width]
        - **normals**: Accumulated normals if return_normals=True, else None. [..., image_height, image_width, 3]
    """
    try:
        from nerfacc import accumulate_along_rays, render_weight_from_alpha
    except ImportError:
        raise ImportError("Please install nerfacc package: pip install nerfacc")

    # Get dimensions
    # Note: means/quats/scales are SHARED across cameras (batch_dims, N, 3/4/3)
    #       colors/opacities are PER-IMAGE [I, N, ...] where I = B*C
    B = math.prod(means.shape[:-2])  # batch dims
    N = means.shape[-2]  # number of Gaussians
    P = image_width * image_height
    channels = colors.shape[-1]
    device = means.device

    # Extract I from colors shape (it's already [I, N, channels])
    I = colors.shape[0]  # Number of images (I = B * C)
    C = I // B  # Cameras per batch

    # 1. Flatten batch dimensions for Gaussian parameters (shared across cameras)
    means_flat = means.reshape(B, N, 3)
    quats_flat = quats.reshape(B, N, 4)
    scales_flat = scales.reshape(B, N, 3)

    # 2. Get pixel coordinates for all M intersections
    pixel_coords = torch.stack(
        [
            (pixel_ids % image_width).float() + 0.5,  # pixel center x
            (pixel_ids // image_width).float() + 0.5,  # pixel center y
        ],
        dim=-1,
    )  # [M, 2]

    if rays is None:
        # Create focal_lengths and principal_points tensors out of Ks[image_ids]
        # Ks[image_ids]: [M, 3, 3]
        Ks_selected = Ks[image_ids]  # [M, 3, 3]
        focal_lengths = Ks_selected[:, [0, 1], [0, 1]]  # [M, 2], fx and fy
        principal_points = Ks_selected[:, [0, 1], [2, 2]]  # [M, 2], cx and cy
        camera = _PerfectPinholeCameraModel(
            focal_lengths, principal_points, image_width, image_height, rs_type
        )

        pose_start = _viewmat_to_pose(viewmats[image_ids])
        if viewmats_rs is None:
            pose = pose_start
        else:
            relative_time = camera.shutter_relative_frame_time(pixel_coords)  # [M, 1]
            pose_end = _viewmat_to_pose(viewmats_rs[image_ids])
            pose = _interpolate_shutter_pose(pose_start, pose_end, relative_time)

        pose_q = pose[..., 3:]

        # Extract rotation and camera world position from pose
        R_cam_to_world = _quat_to_rotmat(_quat_inverse(pose_q))
        cam_centers = _pose_camera_world_position(pose)

        # 4. Generate rays from pixels
        ray_o, ray_d = _generate_rays_from_pixels(
            pixel_coords, cam_centers, R_cam_to_world, Ks[image_ids], means.dtype
        )  # [M, 3, 1]
    else:
        ray_indices = image_ids * image_height * image_width + pixel_ids
        rays_flat = rays.reshape(I * P, 6)
        ray_o = rays_flat[ray_indices, :3]
        ray_d = rays_flat[ray_indices, 3:]

    # 5. Compute Gaussian transform to map Gaussians to world space
    iscl_rot = _compute_gaussian_transform(quats_flat, scales_flat)  # [B, N, 3, 3]

    # 6. Get Gaussian parameters for all M intersections
    # Extract batch index from image_ids (image_ids = batch_id * C + camera_id)
    batch_ids = image_ids // C
    xyz = means_flat[batch_ids, gaussian_ids]  # [M, 3]
    iscl_rot = iscl_rot[batch_ids, gaussian_ids]  # [M, 3, 3]
    scale_per_gauss = scales_flat[batch_ids, gaussian_ids]  # [M, 3]
    opac = opacities[image_ids, gaussian_ids]
    gauss_colors = colors[image_ids, gaussian_ids]

    # 6b. Compute normals if requested
    # Normal computation uses the normal (0,0,1) transformed by rotation
    gauss_normals = None
    if return_normals:
        quats_per_gauss = quats_flat[batch_ids, gaussian_ids]  # [M, 4]
        R = _quat_to_rotmat(quats_per_gauss)  # [M, 3, 3]

        # Canonical normal in Gaussian space: (0, 0, 1)
        # Transform to world space: R @ [0, 0, 1]^T = R[:, 2] (third column)
        gauss_normals = R[:, :, 2]  # [M, 3]

        # Direction resolution: flip if facing away from ray
        ray_d_normalized = _safe_normalize(ray_d)  # [M, 3]
        dot_product = torch.sum(
            gauss_normals * ray_d_normalized, dim=-1, keepdim=True
        )  # [M, 1]
        gauss_normals = torch.where(
            dot_product > 0, -gauss_normals, gauss_normals
        )  # [M, 3]

        # Normalize (should already be unit length, but ensure numerical stability)
        gauss_normals = _safe_normalize(gauss_normals)  # [M, 3]

    # 7. Compute ray-Gaussian distances
    grayDist, hitDist = _compute_ray_gaussian_distance(
        ray_o, ray_d, xyz, iscl_rot, scale_per_gauss
    )

    # 7b. Replace last channel with hit distance if requested (matches CUDA behavior)
    # CUDA: const float value = (k == CDIM - 1) ? hit_distance : c_ptr[k];
    if use_hit_distance:
        gauss_colors = torch.cat([gauss_colors[..., :-1], hitDist[..., None]], dim=-1)

    # 8. Compute Gaussian alphas
    alphas, max_response = _compute_gaussian_alphas(
        grayDist, opac, TRANSMITTANCE_THRESHOLD
    )

    # 9. Filter out low-contribution Gaussians (explicit masking)
    # CUDA: if (alpha < 1.f / 255.f) continue;
    valid_mask = (alphas >= ALPHA_THRESHOLD) & (
        max_response > MAX_KERNEL_DENSITY_CUTOFF
    )

    # Apply filter to all arrays early to reduce memory usage
    alphas = alphas[valid_mask]
    gauss_colors = gauss_colors[valid_mask]
    if gauss_normals is not None:
        gauss_normals = gauss_normals[valid_mask]
    image_ids = image_ids[valid_mask]
    pixel_ids = pixel_ids[valid_mask]
    flatten_idx = flatten_idx[valid_mask]

    # 10. Create ray indices for nerfacc (after filtering to reduce array size)
    ray_indices = image_ids * image_height * image_width + pixel_ids
    total_pixels = I * image_height * image_width

    # CRITICAL: Verify ray indices are sorted (required for packed_info)
    assert torch.all(
        ray_indices[1:] >= ray_indices[:-1]
    ), "indices must be sorted for packed_info"

    # Create packed_info for nerfacc (more memory efficient than ray_indices)
    from nerfacc import pack_info

    # Use nerfacc with packed_info
    weights, trans = render_weight_from_alpha(
        alphas, packed_info=pack_info(ray_indices, total_pixels)
    )

    # Apply base_transmittance if provided (for batched accumulation)
    if base_transmittance is not None:
        # Get per-sample base transmittance by indexing with pixel/image IDs
        base_trans_flat = base_transmittance.reshape(I * image_height * image_width)
        base_trans_per_sample = base_trans_flat[ray_indices]

        # Multiply trans by base transmittance
        trans = trans * base_trans_per_sample

    # Filter by transmittance to match CUDA early termination
    # CUDA: next_T = T * (1-alpha); if (next_T <= TRANSMITTANCE_THRESHOLD) break;
    # Keep samples where next_T > threshold (will be processed before break)
    next_T = trans * (1.0 - alphas)
    valid_mask = next_T > TRANSMITTANCE_THRESHOLD

    weights = weights[valid_mask]
    gauss_colors = gauss_colors[valid_mask]
    if gauss_normals is not None:
        gauss_normals = gauss_normals[valid_mask]
    ray_indices = ray_indices[valid_mask]
    flatten_idx = flatten_idx[valid_mask]

    renders = accumulate_along_rays(
        weights, gauss_colors, ray_indices=ray_indices, n_rays=total_pixels
    )  # [total_pixels, channels]
    alphas = accumulate_along_rays(
        weights, None, ray_indices=ray_indices, n_rays=total_pixels
    )  # [total_pixels, 1]

    # Accumulate normals if computed
    if gauss_normals is not None:
        normals = accumulate_along_rays(
            weights, gauss_normals, ray_indices=ray_indices, n_rays=total_pixels
        )  # [total_pixels, 3]
    else:
        normals = None

    # Compute last flatten_idx per pixel (vectorized using packed_info)
    # CUDA stores: last_ids[pix_id] = cur_idx (index in flatten_ids)
    # PyTorch stores: last_ids[ray_id] = flatten_idx (same indexing as CUDA)

    # Create packed_info from final filtered indices
    from nerfacc import pack_info

    chunk_starts, chunk_cnts = pack_info(ray_indices, total_pixels).unbind(
        dim=-1
    )  # [total_pixels] each

    # Last sample position = start + count - 1 (for rays with samples)
    has_samples = chunk_cnts > 0
    last_positions = chunk_starts + chunk_cnts - 1  # [total_pixels]

    # Extract flatten_idx at last positions
    last_ids = torch.full((total_pixels,), -1, dtype=torch.int32, device=device)
    last_ids[has_samples] = flatten_idx[last_positions[has_samples]].int()

    # Compute sample counts per pixel (from packed_info)
    sample_counts = chunk_cnts.int()  # [total_pixels] - number of samples per ray

    # Reshape: total_pixels = I * H * W, so reshape as [I, H, W, ...]
    renders = renders.reshape(I, image_height, image_width, channels)
    alphas = alphas.reshape(I, image_height, image_width, 1)
    last_ids = last_ids.reshape(I, image_height, image_width)
    sample_counts = sample_counts.reshape(I, image_height, image_width)
    if normals is not None:
        normals = normals.reshape(I, image_height, image_width, 3)

    return renders, alphas, last_ids, sample_counts, normals


# Only supports PerfectPinholeCameraModel.
# Since we have comprehensive camera model tests, we don't need to
# add support for more camera models in this pytorch implementation.
def _rasterize_to_pixels_eval3d(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    colors: Tensor,  # [..., C, N, channels]
    opacities: Tensor,  # [..., C, N]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., C, channels]
    batch_per_iter: int = 200,
    rs_type: RollingShutterType = RollingShutterType.GLOBAL,  # Rolling shutter type
    rays: Optional[Tensor] = None,  # [..., C, H, W, 6]
    viewmats_rs: Optional[
        Tensor
    ] = None,  # [..., C, 4, 4] - optional for rolling shutter
    return_last_ids: bool = False,
    return_sample_counts: bool = False,
    use_hit_distance: bool = False,
    return_normals: bool = False,  # Whether to compute normals
):
    """PyTorch reference implementation of rasterize_to_pixels_eval3d().

    Returns (colors, alphas) by default. Optionally returns last_ids and/or sample_counts
    if return_last_ids and/or return_sample_counts are True.

    This function computes Gaussian responses in 3D world space by casting rays
    from the camera through each pixel and computing the distance from the ray
    to each Gaussian center in the Gaussian's transformed space.

    Uses vectorized evaluation via nerfacc for significant performance improvement,
    similar to _rasterize_to_pixels.

    .. note::
        This is a reference implementation for testing purposes. The CUDA implementation
        is much faster.

    .. note::
        This function relies on Pytorch's autograd for backpropagation and nerfacc
        for efficient accumulation.

    .. warning::
        This function requires the `nerfacc` package to be installed.

    Args:
        tile_size: Size of tiles for culling (e.g., 16)
        isect_offsets: Tile intersection offsets from isect_offset_encode
        flatten_ids: Flattened Gaussian IDs from isect_tiles
        batch_per_iter: Batch size for iterative processing
        viewmats_rs: Optional end pose for rolling shutter [..., C, 4, 4]
        rs_type: Rolling shutter type
        return_last_ids: If True, return the index of the last Gaussian contributing
            to each pixel. Default: False.
        return_sample_counts: If True, return the number of samples (Gaussians)
            evaluated per pixel. Default: False.
        return_normals: If True, compute and return accumulated normals per pixel.
            Normals are computed from Gaussian quaternions (canonical normal = (0,0,1)
            transformed by rotation, flipped if facing away from ray). Default: False.

    Returns:
        A tuple with variable length depending on parameters:
        - (colors, alphas) if all optional returns are False
        - (colors, alphas, last_ids) if return_last_ids=True
        - (colors, alphas, sample_counts) if return_sample_counts=True
        - (colors, alphas, normals) if return_normals=True
        - Various combinations if multiple flags are True
        The order is always: colors, alphas, [last_ids], [sample_counts], [normals]
    """
    from ._wrapper import (
        rasterize_to_indices_in_range,
        fully_fused_projection,
        quat_scale_to_covar_preci,
    )

    # Get dimensions - treat [..., C, N, ...] as flattened images
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    channels = colors.shape[-1]
    device = means.device

    # Reshape to treat batch×cameras as separate images
    # NOTE: means/quats/scales are SHARED across cameras, so don't expand them!
    # Only colors/opacities/viewmats/Ks are per-camera
    B = math.prod(batch_dims)
    I = B * C

    # DON'T expand means/quats/scales - they're shared across cameras
    # Just pass them as-is to accumulate_eval3d

    # Reshape per-camera data
    colors_exp = colors.reshape(I, N, channels)
    opacities_exp = opacities.reshape(I, N)
    viewmats_exp = viewmats.reshape(I, 4, 4)
    viewmats_rs_exp = viewmats_rs.reshape(I, 4, 4) if viewmats_rs is not None else None
    Ks_exp = Ks.reshape(I, 3, 3)
    isect_offsets_exp = isect_offsets.reshape(
        I, isect_offsets.shape[-2], isect_offsets.shape[-1]
    )
    if rays is not None:
        rays_exp = rays.reshape(I, image_height * image_width, 6)
    else:
        rays_exp = None

    # Decode flatten_ids and create properly ordered (gs_id, pix_id, img_id) lists for nerfacc
    # nerfacc requires: sorted by pixel_id first, then by depth within each pixel
    # flatten_ids is already sorted by depth within each tile

    tile_height = isect_offsets_exp.shape[-2]
    tile_width = isect_offsets_exp.shape[-1]
    n_isects = len(flatten_ids)

    # Flatten isect_offsets and append n_isects (matches CUDA pattern)
    isect_offsets_fl = torch.cat(
        [isect_offsets_exp.flatten(), torch.tensor([n_isects], device=device)]
    )

    # Calculate batching parameters
    # max_range = maximum number of Gaussians in any single tile
    max_range = (isect_offsets_fl[1:] - isect_offsets_fl[:-1]).max().item()
    # num_batches = number of iterations needed to process all Gaussians
    num_batches = (max_range + batch_per_iter - 1) // batch_per_iter

    # Initialize outputs
    render_colors = torch.zeros((I, image_height, image_width, channels), device=device)
    render_alphas = torch.zeros((I, image_height, image_width, 1), device=device)
    last_ids = torch.full(
        (I, image_height, image_width), -1, dtype=torch.int32, device=device
    )
    sample_counts = torch.zeros(
        (I, image_height, image_width), dtype=torch.int32, device=device
    )
    render_normals = (
        torch.zeros((I, image_height, image_width, 3), device=device)
        if return_normals
        else None
    )

    # Convert offsets to CPU for indexing
    isect_offsets_cpu = isect_offsets_fl.cpu().numpy()

    # Batch processing loop
    # =====================================================================
    # Batching Strategy:
    # - Each tile has tile_num_gauss Gaussians (sorted by depth)
    # - We process batch_per_iter Gaussians at a time per tile
    # - num_batches = ceil(max_range / batch_per_iter) iterations needed
    # - For each iteration 'step', we process Gaussians at indices
    #   [step*batch_per_iter : (step+1)*batch_per_iter] within each tile
    # =====================================================================
    for step in range(num_batches):
        # Compute current transmittance for early termination
        transmittances = 1.0 - render_alphas[..., 0]  # [I, H, W]

        # Build intersection lists for current batch
        # batch_start/end define which Gaussians to process (by position in tile's sorted list)
        batch_start = step * batch_per_iter
        batch_end = (step + 1) * batch_per_iter

        all_gs_ids = []
        all_pix_ids = []
        all_img_ids = []
        all_flatten_idx = []

        for img_id in range(I):
            for ty in range(tile_height):
                for tx in range(tile_width):
                    # Flattened tile index (matches CUDA)
                    tile_idx = img_id * tile_height * tile_width + ty * tile_width + tx

                    # Get start/end from flattened offsets (matches CUDA pattern)
                    start_idx = int(isect_offsets_cpu[tile_idx])

                    # Match CUDA logic for end index
                    tile_id_local = ty * tile_width + tx
                    is_last_tile_of_last_image = (img_id == I - 1) and (
                        tile_id_local == tile_width * tile_height - 1
                    )

                    if is_last_tile_of_last_image:
                        end_idx = n_isects
                    else:
                        end_idx = int(isect_offsets_cpu[tile_idx + 1])

                    tile_num_gauss = end_idx - start_idx
                    if tile_num_gauss == 0:
                        continue

                    # Filter by batch range (only include Gaussians in this batch)
                    # local_start/local_end are indices within the tile's Gaussian list
                    local_start = max(0, batch_start)
                    local_end = min(tile_num_gauss, batch_end)

                    if local_start >= local_end:
                        continue  # No Gaussians in this batch for this tile

                    # Get Gaussians for this batch slice
                    batch_flatten_ids = flatten_ids[
                        start_idx + local_start : start_idx + local_end
                    ]
                    batch_gauss_ids = batch_flatten_ids % N
                    # batch_num_gauss = number of Gaussians in this batch slice for this tile
                    batch_num_gauss = local_end - local_start

                    # Pixels in this tile (vectorized)
                    py_start = ty * tile_size
                    py_end = min(py_start + tile_size, image_height)
                    px_start = tx * tile_size
                    px_end = min(px_start + tile_size, image_width)

                    # Vectorized pixel ID generation
                    py_grid, px_grid = torch.meshgrid(
                        torch.arange(py_start, py_end, device=device),
                        torch.arange(px_start, px_end, device=device),
                        indexing="ij",
                    )
                    pix_ids_in_tile = (
                        py_grid.flatten() * image_width + px_grid.flatten()
                    )
                    num_pixels = len(pix_ids_in_tile)

                    # Create flatten_ids indices for this batch
                    batch_flatten_indices = torch.arange(
                        start_idx + local_start,
                        start_idx + local_end,
                        device=device,
                        dtype=torch.long,
                    )

                    # Expand and append
                    all_gs_ids.append(batch_gauss_ids.repeat(num_pixels))
                    all_pix_ids.append(
                        pix_ids_in_tile.repeat_interleave(batch_num_gauss)
                    )
                    all_img_ids.append(
                        torch.full(
                            (num_pixels * batch_num_gauss,),
                            img_id,
                            device=device,
                            dtype=torch.long,
                        )
                    )
                    all_flatten_idx.append(batch_flatten_indices.repeat(num_pixels))

        # Concatenate batch lists
        if not all_gs_ids:
            break  # Early termination - no more Gaussians to process

        batch_gs_ids = torch.cat(all_gs_ids)
        batch_pix_ids = torch.cat(all_pix_ids)
        batch_img_ids = torch.cat(all_img_ids)
        batch_flatten_idx = torch.cat(all_flatten_idx)

        # Sort by ray index (required by nerfacc)
        ray_sort_key = batch_img_ids * (image_height * image_width) + batch_pix_ids
        sort_indices = torch.argsort(ray_sort_key, stable=True)
        batch_gs_ids = batch_gs_ids[sort_indices]
        batch_pix_ids = batch_pix_ids[sort_indices]
        batch_img_ids = batch_img_ids[sort_indices]
        batch_flatten_idx = batch_flatten_idx[sort_indices]

        # Accumulate this batch with current transmittance as base
        (
            renders_step,
            alphas_step,
            last_ids_step,
            counts_step,
            normals_step,
        ) = accumulate_eval3d(
            means,
            quats,
            scales,
            opacities_exp,
            colors_exp,
            viewmats_exp,
            Ks_exp,
            batch_gs_ids,
            batch_pix_ids,
            batch_img_ids,
            image_width,
            image_height,
            batch_flatten_idx,
            rs_type,
            rays_exp,
            viewmats_rs_exp,
            base_transmittance=transmittances,  # Pass current transmittance
            use_hit_distance=use_hit_distance,
            return_normals=return_normals,
        )

        # Composite results using transmittance
        render_colors = render_colors + renders_step * transmittances[..., None]
        render_alphas = render_alphas + alphas_step * transmittances[..., None]

        # Composite normals (same accumulation pattern as colors)
        if normals_step is not None and render_normals is not None:
            render_normals = render_normals + normals_step * transmittances[..., None]

        # Update last_ids (keep most recent non-(-1) value)
        last_ids = torch.where(last_ids_step >= 0, last_ids_step, last_ids)

        # Accumulate sample counts
        sample_counts = sample_counts + counts_step

    # Reshape back to original dimensions
    render_colors = render_colors.reshape(
        batch_dims + (C, image_height, image_width, channels)
    )
    render_alphas = render_alphas.reshape(
        batch_dims + (C, image_height, image_width, 1)
    )
    last_ids = last_ids.reshape(batch_dims + (C, image_height, image_width))
    sample_counts = sample_counts.reshape(batch_dims + (C, image_height, image_width))
    if render_normals is not None:
        render_normals = render_normals.reshape(
            batch_dims + (C, image_height, image_width, 3)
        )

    # Add background
    if backgrounds is not None:
        render_colors = render_colors + backgrounds[..., None, None, :] * (
            1.0 - render_alphas
        )

    # Build return tuple based on requested outputs
    # Order: colors, alphas, [last_ids], [sample_counts], [normals]
    outputs: list[Tensor] = [render_colors, render_alphas]
    if return_last_ids:
        outputs.append(last_ids)
    if return_sample_counts:
        outputs.append(sample_counts)
    if return_normals:
        outputs.append(render_normals)

    return tuple(outputs)
