import torch

def frustum_culling(means, scales, viewmat, K, width, height, near=0.01, far=100.0):
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