import torch
from torch import Tensor
import time
import numpy as np
import os
import shutil
from typing import Tuple, Optional

from gsplat.cuda._wrapper import quat_scale_to_covar_preci, proj
# world_to_cam is deprecated in CUDA backend, but we still need it for the PyTorch implementation. Check the implementation in _wrapper.py for more details.
from gsplat.cuda._torch_impl import _world_to_cam, _quat_scale_to_covar_preci, _persp_proj

def calc_pixel_size(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    viewmat: Tensor,  # [4, 4]
    K: Tensor,  # [3, 3]
    width: int, height: int,
    scale_modifier: float = 1.0, eps2d: float = 0.3, method: str = "cuda"
) -> Tensor:
    """
    Calculate the pixel size of a Gaussian in the image plane.

    Args:
        means: [N, 3] - Gaussian centers in world coordinates
        quats: [N, 4] - Gaussian quaternions
        scales: [N, 3] - Gaussian scales
        opacities: [N] - Gaussian opacities
        viewmat: [4, 4] - view matrix (world to camera)
        K: [3, 3] - camera intrinsic matrix
        width: int - image width
        height: int - image height
        scale_modifier: float - scale modifier applied to Gaussian scales
        eps2d: float - low-pass filter value added to 2D covariance diagonal (default: 0.3, not used in pixel size calculation)
        method: str - method to use for pixel size calculation ("cuda" or "torch")

    Returns:
        pixel_size: [N] - pixel size of each Gaussian in the image plane
    """
    if method == "cuda":
        return calc_pixel_size_torch_cuda(means, quats, scales, opacities, viewmat, K, width, height, scale_modifier, eps2d)
    elif method == "torch":
        return calc_pixel_size_torch_only(means, quats, scales, opacities, viewmat, K, width, height, scale_modifier, eps2d)
    else:
        raise ValueError(f"Invalid method: {method}")


def calc_pixel_size_torch_only(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    viewmat: Tensor,  # [4, 4]
    K: Tensor,  # [3, 3]
    width: int, 
    height: int,
    scale_modifier: float = 1.0,
    eps2d: float = 0.3
) -> Tensor:
    """
    PyTorch-only implementation of pixel size calculation.
    """
    
    N = means.shape[0]
    device = means.device
    
    # Apply scale modifier to scales (same as CUDA implementation)
    modified_scales = scales * scale_modifier
    
    # Add batch dimension consistently from the start
    means_batch = means.unsqueeze(0)  # [1, N, 3]
    quats_batch = quats.unsqueeze(0)  # [1, N, 4] 
    scales_batch = modified_scales.unsqueeze(0)  # [1, N, 3]
    
    # 1. Convert quaternions and scales to 3D covariance matrices with batched inputs
    covars_3d_batch, _ = _quat_scale_to_covar_preci(
        quats_batch, scales_batch, compute_covar=True, compute_preci=False, triu=False
    )  # [1, N, 3, 3]
    
    # 2. Transform to camera coordinate system
    viewmat_batch = viewmat.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4] - batch_dims + (C, 4, 4)
    
    means_cam_batch, covars_cam_batch = _world_to_cam(means_batch, covars_3d_batch, viewmat_batch) # Output: [1, 1, N, 3], [1, 1, N, 3, 3]
    
    # 3. Project to 2D screen space - keep batch dimensions
    K_batch = K.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    
    _, covars2d_batch = _persp_proj(means_cam_batch, covars_cam_batch, K_batch, width, height) # Output: [1, 1, N, 2], [1, 1, N, 2, 2]
    
    # 4. Remove all batch dimensions at once
    covars2d = covars2d_batch.squeeze(0).squeeze(0)  # [N, 2, 2]
    
    # 5. Apply low-pass filter (same as CUDA implementation), but not used in pixel size calculation. Refer to forward.cu in MS-GS for more details.
    # Add eps2d to diagonal elements to ensure Gaussians are at least one pixel wide/high
    covars2d_filtered = covars2d.clone()
    covars2d_filtered[..., 0, 0] += eps2d  # cov[0][0] += eps2d
    covars2d_filtered[..., 1, 1] += eps2d  # cov[1][1] += eps2d
    
    # 6. Compute determinant for original covariance (without low-pass filter) for pixel size calculation
    det_orig = (covars2d[..., 0, 0] * covars2d[..., 1, 1] - 
                covars2d[..., 0, 1] * covars2d[..., 1, 0])
    det_orig = torch.clamp(det_orig, min=1e-10)
    
    # Compute conic_ori (precision matrix without low-pass filter)
    conic_ori_xx = covars2d[..., 1, 1] / det_orig
    conic_ori_zz = covars2d[..., 0, 0] / det_orig
    
    # 7. Calculate pixel size using level set
    # level_set = -2 * log(1 / (255.0 * opacity))
    level_set = -2.0 * torch.log(1.0 / (255.0 * opacities))
    level_set = torch.clamp(level_set, min=0.0)  # negative level set when gaussian opacity is too low
    
    # Calculate dx and dy
    dx = torch.sqrt(level_set / conic_ori_xx)
    dy = torch.sqrt(level_set / conic_ori_zz)
    
    # Pixel size is the minimum of dx and dy
    pixel_size = torch.min(dx, dy)
    
    # Apply scale modifier division (same as CUDA implementation)
    pixel_size = pixel_size / scale_modifier
    
    return pixel_size


def calc_pixel_size_torch_cuda(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    viewmat: Tensor,  # [4, 4]
    K: Tensor,  # [3, 3]
    width: int, 
    height: int,
    scale_modifier: float = 1.0,
    eps2d: float = 0.3
) -> Tensor:
    """
    PyTorch + CUDA implementation of pixel size calculation.
    """
    
    N = means.shape[0]
    device = means.device
    
    # Apply scale modifier to scales (same as CUDA implementation)
    modified_scales = scales * scale_modifier
    
    # Add batch dimension consistently from the start
    means_batch = means.unsqueeze(0)  # [1, N, 3]
    quats_batch = quats.unsqueeze(0)  # [1, N, 4] 
    scales_batch = modified_scales.unsqueeze(0)  # [1, N, 3]
    
    # 1. Convert quaternions and scales to 3D covariance matrices using CUDA with batched inputs
    covars_3d_batch, _ = _quat_scale_to_covar_preci(
        quats_batch, scales_batch, compute_covar=True, compute_preci=False, triu=False
    )  # [1, N, 3, 3]
    
    # 2. Transform to camera coordinate system
    viewmat_batch = viewmat.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4] - batch_dims + (C, 4, 4)
    
    means_cam_batch, covars_cam_batch = _world_to_cam(means_batch, covars_3d_batch, viewmat_batch)
    # Output: [1, 1, N, 3], [1, 1, N, 3, 3]
    
    # 3. Project to 2D screen space using CUDA - keep batch dimensions
    K_batch = K.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    
    _, covars2d_batch = proj(means_cam_batch, covars_cam_batch, K_batch, width, height, camera_model="pinhole")
    # Output: [1, 1, N, 2], [1, 1, N, 2, 2]
    
    # 4. Remove all batch dimensions at once
    covars2d = covars2d_batch.squeeze(0).squeeze(0)  # [N, 2, 2]
    
    # 5. Apply low-pass filter (same as CUDA implementation), but not used in pixel size calculation. Refer to forward.cu in MS-GS for more details.
    # Add eps2d to diagonal elements to ensure Gaussians are at least one pixel wide/high
    covars2d_filtered = covars2d.clone()
    covars2d_filtered[..., 0, 0] += eps2d  # cov[0][0] += eps2d
    covars2d_filtered[..., 1, 1] += eps2d  # cov[1][1] += eps2d
    
    # 6. Compute determinant for original covariance (without low-pass filter) for pixel size calculation
    det_orig = (covars2d[..., 0, 0] * covars2d[..., 1, 1] - 
                covars2d[..., 0, 1] * covars2d[..., 1, 0])
    det_orig = torch.clamp(det_orig, min=1e-10)
    
    # Compute conic_ori (precision matrix without low-pass filter)
    conic_ori_xx = covars2d[..., 1, 1] / det_orig
    conic_ori_zz = covars2d[..., 0, 0] / det_orig
    
    # 7. Calculate pixel size using level set
    # level_set = -2 * log(1 / (255.0 * opacity))
    level_set = -2.0 * torch.log(1.0 / (255.0 * opacities))
    level_set = torch.clamp(level_set, min=0.0)  # negative level set when gaussian opacity is too low
    
    # Calculate dx and dy
    dx = torch.sqrt(level_set / conic_ori_xx)
    dy = torch.sqrt(level_set / conic_ori_zz)
    
    # Pixel size is the minimum of dx and dy
    pixel_size = torch.min(dx, dy)
    
    # Apply scale modifier division (same as CUDA implementation)
    pixel_size = pixel_size / scale_modifier
    
    return pixel_size

def distance_culling(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    viewmat: Tensor,  # [4, 4]
    K: Tensor,  # [3, 3]
    width: int, height: int,
    scale_modifier: float = 1.0,
    eps2d: float = 0.3, 
    pixel_threshold: float = 2.0,
    method: str = "cuda",
    return_pixel_sizes: bool = False,
    dump_file: str = "pixel_sizes_vs_distance.npz",
    use_timestamps: bool = True
) -> Tuple[Tensor, Optional[np.ndarray]]:
    """
    Perform distance culling on Gaussian splats.
    
    Args:
        means: [N, 3] - Gaussian centers in world coordinates
        quats: [N, 4] - Gaussian quaternions
        scales: [N, 3] - Gaussian scales
        opacities: [N] - Gaussian opacities
        viewmat: [4, 4] - view matrix (world to camera)
        K: [3, 3] - camera intrinsic matrix
        width: int - image width
        height: int - image height
        scale_modifier: float - scale modifier applied to Gaussian scales
        eps2d: float - low-pass filter value
        method: str - method to use for pixel size calculation
        return_pixel_sizes: bool - if True, return pixel sizes as numpy array
        dump_file: str - base file path to dump data (NPZ format), Pass None to disable dumping
        use_timestamps: bool - if True, append timestamp to filename to avoid overwriting
        
    Returns:
        mask: [N] - boolean mask for gaussians with pixel_size >= 2.0
    """
    # Calculate pixel sizes
    pixel_sizes = calc_pixel_size(means, quats, scales, opacities, viewmat, K, width, height, scale_modifier, eps2d, method)
    mask = pixel_sizes >= pixel_threshold  # 2.0 is the minimum pixel size to prevent aliasing
    if return_pixel_sizes:
        return mask, pixel_sizes.detach().cpu().numpy() # [N]
    else:
        return mask, None # [N]

def frustum_culling(
    means: Tensor,  # [N, 3]
    scales: Tensor,  # [N, 3]
    viewmat: Tensor,  # [4, 4]
    K: Tensor,  # [3, 3]
    width: int, height: int, near: float = 0.01, far: float = 100.0
) -> Tensor:
    """
    Perform frustum culling on Gaussian splats.
    
    Args:
        means: [N, 3] - Gaussian centers in world coordinates
        scales: [N, 3] - Gaussian scales
        viewmat: [4, 4] - view matrix (world to camera)
        K: [3, 3] - camera intrinsic matrix
        width: int - image width
        height: int - image height
        near: float - near clipping plane distance
        far: float - far clipping plane distance
    
    Returns:
        mask: [N] - boolean mask indicating which Gaussians are inside the frustum
    """
    device = means.device
    N = means.shape[0]
    
    # Transform Gaussian centers to camera coordinates
    means_homo = torch.cat([means, torch.ones(N, 1, device=device)], dim=1)  # [N, 4]
    means_cam = (viewmat @ means_homo.T).T  # [N, 4]
    means_cam_xyz = means_cam[:, :3]  # [N, 3]
    
    # Get maximum scale for each Gaussian (conservative estimate of extent)
    max_scales = torch.max(scales, dim=1)[0]  # [N]
    
    # Near/far plane culling (ENABLED)
    # Since +Z points into the screen, larger Z = farther from camera
    z_cam = means_cam_xyz[:, 2]  # Positive Z values represent depth into screen
    near_mask = z_cam > (near - max_scales)  # Points farther than near plane
    far_mask = z_cam < (far + max_scales)    # Points closer than far plane
    depth_mask = near_mask & far_mask
    
    # Project to screen coordinates
    # Note: +Z points into the screen, so we use positive Z for depth
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Avoid division by zero or negative depth
    z_depth = means_cam_xyz[:, 2]
    valid_depth_mask = z_depth > 1e-6  # Points must be in front of camera
    
    x_screen = (means_cam_xyz[:, 0] * fx / z_depth) + cx
    y_screen = (means_cam_xyz[:, 1] * fy / z_depth) + cy
    
    # Compute projected scale for conservative culling
    # Use the maximum scale and project it to screen space
    projected_scale_x = max_scales * fx / z_depth
    projected_scale_y = max_scales * fy / z_depth
    projected_scale = torch.max(projected_scale_x, projected_scale_y)
    
    # Screen space culling with margin for Gaussian extent
    margin = projected_scale
    left_mask = x_screen > (-margin)
    right_mask = x_screen < (width + margin)
    top_mask = y_screen > (-margin)
    bottom_mask = y_screen < (height + margin)
    
    # Combine all masks (screen-space + near/far + valid depth culling)
    screen_mask = left_mask & right_mask & top_mask & bottom_mask
    final_mask = depth_mask & screen_mask & valid_depth_mask
    
    return final_mask