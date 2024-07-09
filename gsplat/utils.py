# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/61c7b417393d5e0c58b742ad5e2e5f9e9f240cc6/utils/point_utils.py#L26
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math
from torch import Tensor

import pdb

# def depths_to_points(
#     c2w: Tensor, 
#     intrinsic: Tensor, 
#     width: int,
#     height: int,
#     depthmap
# ) -> Tensor:
#     grid_x, grid_y = torch.meshgrid(torch.arange(width, device="cuda").float(), torch.arange(height, device="cuda").float(), indexing='xy')
#     points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
#     rays_d = points @ intrinsic.inverse().T @ c2w[:3, :3].T
#     rays_o = c2w[:3, 3]
#     points = depthmap.reshape(-1, 1) * rays_d + rays_o
#     return points 
    
# def depth_to_normal(
#     c2w: Tensor, 
#     intrinsic: Tensor, 
#     width: int,
#     height: int,
#     depth
# ):  
#     c2w = torch.linalg.inv(c2w)
#     points = depths_to_points(c2w, intrinsic, width, height, depth).reshape(height, width, 3)
#     output = torch.zeros_like(points)
#     dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
#     dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
#     normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
#     output[1:-1, 1:-1, :] = normal_map
#     return output
    

def _depths_to_points(depthmap, world_view_transform, full_proj_transform, fx, fy):
    c2w = (world_view_transform.T).inverse()
    # import pdb
    # pdb.set_trace()
    H, W = depthmap.shape[:2]
    # ndc2pix = (
    #     torch.tensor([[W / 2, 0, 0, (W) / 2], [0, H / 2, 0, (H) / 2], [0, 0, 0, 1]])
    #     .float()
    #     .cuda()
    #     .T
    # )
    # projection_matrix = c2w.T @ full_proj_transform
    # intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()

    import pdb
    # pdb.set_trace()

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


def _depth_to_normal(depth, world_view_transform, full_proj_transform, fx, fy):
    points = _depths_to_points(
        depth, world_view_transform, full_proj_transform, fx, fy,
    ).reshape(*depth.shape[:2], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def depth_to_normal(depths, camtoworlds, Ks, near_plane, far_plane):
    import pdb
    # pdb.set_trace()
    height, width = depths.shape[1:3]
    viewmats = torch.linalg.inv(camtoworlds)  # [C, 4, 4]

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
        normal = _depth_to_normal(depth, world_view_transform, full_proj_transform, Ks[cid, 0, 0], Ks[cid, 1, 1])
        normals.append(normal)
    normals = torch.stack(normals, dim=0)
    return normals


def _getProjectionMatrix(znear, zfar, fovX, fovY, device="cuda"):
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