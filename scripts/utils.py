import json
import os
import torch
import torch.nn.functional as F
from examples.config import Config
from typing import Optional, Dict, List, Tuple

def extract_path_components(data_dir_path):
    """
    Extract Actor01, Sequence1, and resolution_1 from path like:
    /main/rajrup/Dropbox/Project/GsplatStream/gsplat/data/Actor01/Sequence1/0/resolution_1
    """
    parts = data_dir_path.strip('/').split('/')
    
    # Find the index of 'data' in the path
    try:
        data_index = parts.index('data')
        if data_index + 3 < len(parts):
            actor_name = parts[data_index + 1]  # Actor01
            sequence_name = parts[data_index + 2]  # Sequence1
            resolution_name = parts[data_index + 4]  # resolution_1 (skip frame_id at index 3)
            return actor_name, sequence_name, resolution_name
    except (ValueError, IndexError):
        pass
    
    return None, None, None

def set_result_dir(config: Config, exp_name: str, sub_exp_name: Optional[str] = None):
    data_dir = config.data_dir
    scene_id = config.scene_id
    actor, sequence, resolution = extract_path_components(data_dir)
    if sub_exp_name is not None:
        config.result_dir = f"./results/{exp_name}/{actor}/{sequence}/{resolution}/{scene_id}/{sub_exp_name}"
    else:
        config.result_dir = f"./results/{exp_name}/{actor}/{sequence}/{resolution}/{scene_id}"
        
def set_result_dir_voxelize(config: Config, exp_name: str, voxel_size: float, aggregate_method: str):
    data_dir = config.data_dir
    scene_id = config.scene_id
    actor, sequence, resolution = extract_path_components(data_dir)
    config.result_dir = f"./results/{exp_name}_to_voxel_size_{voxel_size}/{actor}/{sequence}/{resolution}/{scene_id}/{aggregate_method}/"

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