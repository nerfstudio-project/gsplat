import json
import os
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F


def load_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Load checkpoint and return Gaussian parameters."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)["splats"]
    print(f"Checkpoint loaded from: {ckpt_path}")
    print(f"Checkpoint keys: {ckpt.keys()}")
    
    means = ckpt["means"]
    quats = F.normalize(ckpt["quats"], p=2, dim=-1)
    scales = torch.exp(ckpt["scales"])
    opacities = torch.sigmoid(ckpt["opacities"])
    sh0 = ckpt["sh0"]
    shN = ckpt["shN"]
    colors = torch.cat([sh0, shN], dim=-2)  # [N, K, 3]
    
    print(f"Number of Gaussians: {means.shape[0]}")
    return means, quats, scales, opacities, colors


def load_poses(poses_file: str) -> List[Dict]:
    """Load viewer poses from JSON file."""
    if not os.path.exists(poses_file):
        raise FileNotFoundError(f"Poses file not found: {poses_file}")
    
    with open(poses_file, 'r') as f:
        poses = json.load(f)
    
    print(f"Loaded {len(poses)} poses from {poses_file}")
    return poses