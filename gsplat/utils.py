import math

import torch
import torch.nn.functional as F
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


def log_transform(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def inverse_log_transform(y):
    return torch.sign(y) * (torch.expm1(torch.abs(y)))


def depth_to_points(
    depths: Tensor, camtoworlds: Tensor, Ks: Tensor, z_depth: bool = True
) -> Tensor:
    """Convert depth maps to 3D points

    Args:
        depths: Depth maps [..., H, W, 1]
        camtoworlds: Camera-to-world transformation matrices [..., 4, 4]
        Ks: Camera intrinsics [..., 3, 3]
        z_depth: Whether the depth is in z-depth (True) or ray depth (False)

    Returns:
        points: 3D points in the world coordinate system [..., H, W, 3]
    """
    assert depths.shape[-1] == 1, f"Invalid depth shape: {depths.shape}"
    assert camtoworlds.shape[-2:] == (
        4,
        4,
    ), f"Invalid viewmats shape: {camtoworlds.shape}"
    assert Ks.shape[-2:] == (3, 3), f"Invalid Ks shape: {Ks.shape}"
    assert (
        depths.shape[:-3] == camtoworlds.shape[:-2] == Ks.shape[:-2]
    ), f"Shape mismatch! depths: {depths.shape}, viewmats: {camtoworlds.shape}, Ks: {Ks.shape}"

    device = depths.device
    height, width = depths.shape[-3:-1]

    x, y = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="xy",
    )  # [H, W]

    fx = Ks[..., 0, 0]  # [...]
    fy = Ks[..., 1, 1]  # [...]
    cx = Ks[..., 0, 2]  # [...]
    cy = Ks[..., 1, 2]  # [...]

    # camera directions in camera coordinates
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx[..., None, None] + 0.5) / fx[..., None, None],
                (y - cy[..., None, None] + 0.5) / fy[..., None, None],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [..., H, W, 3]

    # ray directions in world coordinates
    directions = torch.einsum(
        "...ij,...hwj->...hwi", camtoworlds[..., :3, :3], camera_dirs
    )  # [..., H, W, 3]
    origins = camtoworlds[..., :3, -1]  # [..., 3]

    if not z_depth:
        directions = F.normalize(directions, dim=-1)

    points = origins[..., None, None, :] + depths * directions
    return points


def depth_to_normal(
    depths: Tensor, camtoworlds: Tensor, Ks: Tensor, z_depth: bool = True
) -> Tensor:
    """Convert depth maps to surface normals

    Args:
        depths: Depth maps [..., H, W, 1]
        camtoworlds: Camera-to-world transformation matrices [..., 4, 4]
        Ks: Camera intrinsics [..., 3, 3]
        z_depth: Whether the depth is in z-depth (True) or ray depth (False)

    Returns:
        normals: Surface normals in the world coordinate system [..., H, W, 3]
    """
    points = depth_to_points(depths, camtoworlds, Ks, z_depth=z_depth)  # [..., H, W, 3]
    dx = torch.cat(
        [points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]], dim=-3
    )  # [..., H-2, W-2, 3]
    dy = torch.cat(
        [points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]], dim=-2
    )  # [..., H-2, W-2, 3]
    normals = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)  # [..., H-2, W-2, 3]
    normals = F.pad(normals, (0, 0, 1, 1, 1, 1), value=0.0)  # [..., H, W, 3]
    return normals


def get_projection_matrix(znear, zfar, fovX, fovY, device="cuda"):
    """Create OpenGL-style projection matrix"""
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
