import math

import torch
from torch import Tensor


def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    """Convert normalized quaternion to rotation matrix.

    Args:
        quat: Normalized quaternion in wxyz convension. (..., 4)

    Returns:
        Rotation matrix (..., 3, 3)
    """
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

def depth_to_normal(
        depths: Tensor, 
        camtoworlds: Tensor, 
        Ks: Tensor, 
        near_plane: float, 
        far_plane: float
) -> Tensor: 
    """
    Convert depth to surface normal 

    Args:
        depths: Z-depth of the Gaussians.
        camtoworlds: camera to world transformation matrix.
        Ks: camera intrinsics.
        near_plane: Near plane distance.
        far_plane: Far plane distance.

    Returns:
        -**Surface normals**
    """
    height, width = depths.shape[1:3]
    viewmats = torch.linalg.inv(camtoworlds)  # [C, 4, 4]

    normals = []
    for cid, depth in enumerate(depths):
        FoVx = 2 * math.atan(width / (2 * Ks[cid, 0, 0].item()))
        FoVy = 2 * math.atan(height / (2 * Ks[cid, 1, 1].item()))
        world_view_transform = viewmats[cid].transpose(0, 1)
        projection_matrix = _get_projection_matrix(
            znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=depths.device
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        normal = _depth_to_normal(
            depth,
            world_view_transform,
            full_proj_transform,
            Ks[cid, 0, 0],
            Ks[cid, 1, 1],
        )
        normals.append(normal)
    normals = torch.stack(normals, dim=0)
    return normals

# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/61c7b417393d5e0c58b742ad5e2e5f9e9f240cc6/utils/point_utils.py#L26
def _depths_to_points(depthmap, world_view_transform, full_proj_transform, fx, fy) -> Tensor:
    c2w = (world_view_transform.T).inverse()
    H, W = depthmap.shape[:2]

    intrins = (
        torch.tensor([[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]])
        .float()
        .cuda()
    )

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device="cuda").float(),
        torch.arange(H, device="cuda").float(),
        indexing="xy",
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(
        -1, 3
    )
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def _depth_to_normal(depth, world_view_transform, full_proj_transform, fx, fy) -> Tensor:
    points = _depths_to_points(
        depth,
        world_view_transform,
        full_proj_transform,
        fx,
        fy,
    ).reshape(*depth.shape[:2], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def _get_projection_matrix(znear, zfar, fovX, fovY, device="cuda") -> Tensor:
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P
