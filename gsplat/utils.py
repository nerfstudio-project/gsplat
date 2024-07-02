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

def depths_to_points(
    c2w: Tensor, 
    intrinsic: Tensor, 
    width: int,
    height: int,
    depthmap
) -> Tensor:    
    c2w = torch.Tensor([[-8.6745e-01, -2.2594e-01,  4.4326e-01, -1.3165e+00],
        [ 2.3024e-01,  6.0748e-01,  7.6023e-01, -1.3669e+00],
        [-4.4104e-01,  7.6152e-01, -4.7494e-01,  2.9974e+00],
        [-1.5894e-09,  1.7377e-08, -8.6406e-09,  1.0000e+00]]).to("cuda")
    grid_x, grid_y = torch.meshgrid(torch.arange(width, device="cuda").float(), torch.arange(height, device="cuda").float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrinsic.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points 
    
def depth_to_normal(
    c2w: Tensor, 
    intrinsic: Tensor, 
    width: int,
    height: int,
    depth
):  
    c2w = torch.linalg.inv(c2w)
    points = depths_to_points(c2w, intrinsic, width, height, depth).reshape(height, width, 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output
    