import torch
import torch.nn.functional as F
import math

from .camera_utils import _getProjectionMatrix


def _depths_to_points(depthmap, world_view_transform, full_proj_transform):
    c2w = (world_view_transform.T).inverse()
    H, W = depthmap.shape[:2]
    ndc2pix = (
        torch.tensor([[W / 2, 0, 0, (W) / 2], [0, H / 2, 0, (H) / 2], [0, 0, 0, 1]])
        .float()
        .cuda()
        .T
    )
    projection_matrix = c2w.T @ full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

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


def _depth_to_normal(depth, world_view_transform, full_proj_transform):
    points = _depths_to_points(
        depth, world_view_transform, full_proj_transform
    ).reshape(*depth.shape[:2], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def depth_to_normal(depths, viewmats, Ks, near_plane, far_plane):
    height, width = depths.shape[1:3]

    normals = []
    for cid, depth in enumerate(depths):
        FoVx = 2 * math.atan(width / (2 * Ks[cid, 0, 0].item()))
        FoVy = 2 * math.atan(height / (2 * Ks[cid, 1, 1].item()))
        world_view_transform = viewmats[cid].transpose(0, 1)
        projection_matrix = _getProjectionMatrix(
            znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=depths.device
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        normal = _depth_to_normal(depth, world_view_transform, full_proj_transform)
        normals.append(normal)
    normals = torch.stack(normals, dim=0)
    return normals
