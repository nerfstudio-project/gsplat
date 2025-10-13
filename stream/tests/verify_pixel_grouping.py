#!/usr/bin/env python3
"""
Verify that pixel grouping and sorting (steps 1-2) match exactly between CUDA and PyTorch implementations.
This script tests the deterministic parts of the clustering algorithm before depth clustering.
"""

import torch
from torch import Tensor
import numpy as np
from typing import Dict, List, Tuple
import sys
import os
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.config import Config, load_config_from_toml, merge_config
from scripts.utils import set_result_dir, load_checkpoint, load_poses
from stream.merging import find_merge_candidates
from stream.clustering_cuda import cluster_center_in_pixel_cuda, cluster_center_in_pixel_torch
from stream.cuda._wrapper import _cluster_center_in_pixel_cuda, extract_pixel_groups_step2_cuda
from gsplat.cuda._wrapper import proj, world_to_cam

def load_test_data(pose_id: int = 80) -> Dict:
    """Load the same test data used in test_cuda_clustering.py"""
    cfg = Config()
    
    # Load template config
    template_path = "./configs/actorshq_stream.toml"
    config_from_file = load_config_from_toml(template_path)
    cfg = merge_config(cfg, config_from_file)
    
    # Set experiment parameters
    exp_name = f"actorshq_l1_{1.0 - cfg.ssim_lambda}_ssim_{cfg.ssim_lambda}"
    if cfg.masked_l1_loss:
        exp_name += f"_ml1_{cfg.masked_l1_lambda}"
    if cfg.masked_ssim_loss:
        exp_name += f"_mssim_{cfg.masked_ssim_lambda}"
    if cfg.alpha_loss:
        exp_name += f"_alpha_{cfg.alpha_lambda}"
    if cfg.scale_var_loss:
        exp_name += f"_svar_{cfg.scale_var_lambda}"
    if cfg.random_bkgd:
        exp_name += "_rbkgd"
    
    # Set data paths
    frame_id = 0
    cfg.data_dir = os.path.join(cfg.actorshq_data_dir, f"{frame_id}", f"resolution_{cfg.resolution}")
    cfg.exp_name = exp_name
    cfg.run_mode = "render"
    cfg.scene_id = frame_id
    
    set_result_dir(cfg, exp_name=exp_name)
    iter = cfg.max_steps
    ckpt = os.path.join(f"{cfg.result_dir}/ckpts/ckpt_{iter - 1}_rank0.pt")
    cfg.ckpt = ckpt
    
    log.info(f"Loading scene data for pose {pose_id}...")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    assert os.path.exists(cfg.ckpt), f"Checkpoint not found: {cfg.ckpt}"
    means, quats, scales, opacities, colors = load_checkpoint(cfg.ckpt, device)
    
    # Load poses
    poses_dir = os.path.join(cfg.result_dir, "viewer_poses")
    poses_file = os.path.join(poses_dir, "viewer_poses.json")
    assert os.path.exists(poses_file), f"Poses file not found: {poses_file}"
    
    poses = load_poses(poses_file)
    
    # Find the specific pose
    target_pose = None
    for pose_data in poses:
        if pose_data['pose_id'] == pose_id:
            target_pose = pose_data
            break
    
    assert target_pose is not None, f"Pose ID {pose_id} not found in poses file"
    
    # Extract camera parameters
    c2w_matrix = torch.tensor(target_pose["c2w_matrix"], device=device, dtype=torch.float32)
    K_matrix = torch.tensor(target_pose["K_matrix"], device=device, dtype=torch.float32)
    viewmat = c2w_matrix.inverse()
    
    # Use custom rendering resolution
    width = cfg.render_width
    height = cfg.render_height
    original_width = target_pose["width"]
    original_height = target_pose["height"]
    
    # Adjust intrinsics for the new resolution
    scale_x = width / original_width
    scale_y = height / original_height
    K_adjusted = K_matrix.clone()
    K_adjusted[0, 0] *= scale_x  # fx
    K_adjusted[1, 1] *= scale_y  # fy
    K_adjusted[0, 2] *= scale_x  # cx
    K_adjusted[1, 2] *= scale_y  # cy
    
    # Find merge candidates
    candidate_mask = find_merge_candidates(
        means, quats, scales, opacities, viewmat, K_adjusted, width, height,
        use_pixel_area=True, pixel_area_threshold=2.0, scale_modifier=1.0, eps2d=0.3, method="cuda"
    )
    
    candidate_indices = torch.where(candidate_mask)[0].cpu().numpy()
    candidate_means = means[candidate_mask]
    
    log.info(f"Loaded {len(candidate_indices)} candidates out of {means.shape[0]} Gaussians")
    
    return {
        'candidate_means': candidate_means,
        'candidate_indices': candidate_indices,
        'viewmat': viewmat,
        'K': K_adjusted,
        'width': width,
        'height': height
    }

def extract_torch_pixel_groups(
    candidate_means: Tensor,
    candidate_indices: np.ndarray,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int
) -> Dict[Tuple[int, int], List[Tuple[int, float, int]]]:
    """Extract pixel groups from PyTorch implementation (steps 1-2 only)"""
    
    log.info("Extracting PyTorch pixel groups...")
    
    device = candidate_means.device
    M = len(candidate_means)
    
    # 1. Transform to camera coordinates (same as cluster_center_in_pixel_torch)
    means_batch = candidate_means.unsqueeze(0)  # [1, M, 3]
    viewmat_batch = viewmat.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
    
    dummy_covars = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, M, 3, 3)
    means_cam_batch, _ = world_to_cam(means_batch, dummy_covars, viewmat_batch)
    means_cam = means_cam_batch.squeeze(0).squeeze(0)  # [M, 3]
    
    # 2. Project to 2D pixel coordinates
    means_cam_proj = means_cam.unsqueeze(0).unsqueeze(0)  # [1, 1, M, 3]
    K_batch = K.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    
    dummy_covars_cam = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, M, 3, 3)
    means2d_batch, _ = proj(means_cam_proj, dummy_covars_cam, K_batch, width, height)
    means2d = means2d_batch.squeeze(0).squeeze(0)  # [M, 2]
    
    # 3. Convert to discrete pixel coordinates
    pixel_coords = torch.floor(means2d).long()  # [M, 2]
    
    # Filter out points outside image bounds
    valid_mask = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height) &
        (means_cam[:, 2] > 0)  # Points must be in front of camera
    )
    
    if not valid_mask.any():
        return {}
    
    # Apply valid mask
    valid_indices = torch.where(valid_mask)[0]
    valid_pixel_coords = pixel_coords[valid_mask]  # [V, 2]
    valid_depths = means_cam[valid_mask, 2]  # [V] - camera Z coordinate (depth)
    valid_candidate_indices = candidate_indices[valid_indices.cpu().numpy()]
    
    # 4. Group by pixel coordinates (STEP 1)
    pixel_groups = {}
    for i, (pixel_coord, depth, orig_idx) in enumerate(zip(valid_pixel_coords, valid_depths, valid_candidate_indices)):
        pixel_key = (pixel_coord[0].item(), pixel_coord[1].item())
        if pixel_key not in pixel_groups:
            pixel_groups[pixel_key] = []
        pixel_groups[pixel_key].append((i, depth.item(), orig_idx))
    
    # 5. Sort each group by depth (STEP 2)
    for pixel_key in pixel_groups:
        pixel_groups[pixel_key].sort(key=lambda x: x[1])  # Sort by depth (x[1])
    
    log.info(f"PyTorch: Found {len(pixel_groups)} pixel groups")
    
    return pixel_groups

def extract_cuda_pixel_groups_step2(
    candidate_means: Tensor,
    candidate_indices: np.ndarray,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    depth_threshold: float = 0.1,
    min_cluster_size: int = 2
) -> Dict[Tuple[int, int], List[Tuple[int, float, int]]]:
    """Extract pixel groups using actual CUDA implementation after step 2 (grouping + sorting)"""
    
    log.info("Extracting CUDA pixel groups using actual CUDA function...")
    
    # Replicate the same preprocessing as CUDA implementation
    device = candidate_means.device
    M = len(candidate_means)
    
    # 1. Transform to camera coordinates (same preprocessing as CUDA)
    means_batch = candidate_means.unsqueeze(0)  # [1, M, 3]
    viewmat_batch = viewmat.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
    
    dummy_covars = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, M, 3, 3)
    means_cam_batch, _ = world_to_cam(means_batch, dummy_covars, viewmat_batch)
    means_cam = means_cam_batch.squeeze(0).squeeze(0)  # [M, 3]
    
    # 2. Project to 2D pixel coordinates
    means_cam_proj = means_cam.unsqueeze(0).unsqueeze(0)  # [1, 1, M, 3]
    K_batch = K.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    
    dummy_covars_cam = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, M, 3, 3)
    means2d_batch, _ = proj(means_cam_proj, dummy_covars_cam, K_batch, width, height)
    means2d = means2d_batch.squeeze(0).squeeze(0)  # [M, 2]
    
    # Convert to discrete pixel coordinates (same as CUDA implementation)
    pixel_coords = torch.floor(means2d).int()  # [M, 2] - discrete coordinates
    
    # Filter out points outside image bounds (same as CUDA implementation)
    valid_mask = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height) &
        (means_cam[:, 2] > 0)  # Points must be in front of camera
    )
    
    if not valid_mask.any():
        return {}
    
    # Apply valid mask (same as CUDA implementation)
    valid_indices = torch.where(valid_mask)[0]
    valid_pixel_coords = pixel_coords[valid_mask]  # [V, 2]
    valid_means_cam = means_cam[valid_mask]  # [V, 3]
    valid_candidate_indices = candidate_indices[valid_indices.cpu().numpy()]
    valid_candidate_indices_tensor = torch.from_numpy(valid_candidate_indices).to(device).int()
    
    # Call the actual CUDA debug function to get step 2 results
    cuda_result = extract_pixel_groups_step2_cuda(
        valid_means_cam, valid_pixel_coords, valid_candidate_indices_tensor,
        viewmat, K, width, height, depth_threshold, min_cluster_size
    )
    
    group_starts = cuda_result['group_starts'].cpu().numpy()
    group_sizes = cuda_result['group_sizes'].cpu().numpy()
    sorted_pixel_hashes = cuda_result['sorted_pixel_hashes'].cpu().numpy() 
    sorted_depths = cuda_result['sorted_depths'].cpu().numpy()
    sorted_indices = cuda_result['sorted_indices'].cpu().numpy()
    num_groups = cuda_result['num_groups']
    num_valid = cuda_result['num_valid']
    
    log.info(f"CUDA actual: Found {num_groups} pixel groups with {num_valid} valid candidates")
    
    # Convert CUDA results to the same format as PyTorch for comparison
    pixel_groups = {}
    
    # Extract pixel groups from CUDA results using the sorted pixel hashes
    for group_idx in range(num_groups):
        start = group_starts[group_idx]
        size = group_sizes[group_idx]
        
        if size > 0:
            # Get the pixel coordinates for this group from the first hash in the group
            first_hash = sorted_pixel_hashes[start]
            px = int(first_hash >> 32)
            py = int(first_hash & 0xFFFFFFFF)
            pixel_key = (px, py)
            
            # Extract group data
            group = []
            for i in range(size):
                local_idx = i  # local index within the group
                depth = sorted_depths[start + i]
                orig_idx = sorted_indices[start + i]
                group.append((local_idx, depth, orig_idx))
            
            pixel_groups[pixel_key] = group
    
    return pixel_groups

def compare_pixel_groups(torch_groups: Dict, cuda_groups: Dict) -> bool:
    """Compare pixel groups from both implementations"""
    
    log.info("\n" + "="*80)
    log.info("COMPARING PIXEL GROUPS (Steps 1-2: Grouping + Sorting)")
    log.info("="*80)
    
    # Basic statistics
    log.info(f"PyTorch groups: {len(torch_groups)}")
    log.info(f"CUDA groups: {len(cuda_groups)}")
    
    if len(torch_groups) != len(cuda_groups):
        log.error(f"❌ Different number of pixel groups!")
        return False
    
    # Compare pixel keys
    torch_keys = set(torch_groups.keys())
    cuda_keys = set(cuda_groups.keys())
    
    if torch_keys != cuda_keys:
        log.error(f"❌ Different pixel coordinates!")
        missing_in_cuda = torch_keys - cuda_keys
        missing_in_torch = cuda_keys - torch_keys
        if missing_in_cuda:
            log.error(f"  Missing in CUDA: {list(missing_in_cuda)[:5]}...")
        if missing_in_torch:
            log.error(f"  Missing in PyTorch: {list(missing_in_torch)[:5]}...")
        return False
    
    log.info("✅ Same pixel coordinates in both implementations")
    
    # Compare group contents and sorting
    all_match = True
    mismatched_groups = []
    
    for pixel_key in sorted(torch_keys):
        torch_group = torch_groups[pixel_key]
        cuda_group = cuda_groups[pixel_key]
        
        if len(torch_group) != len(cuda_group):
            log.error(f"❌ Different group sizes for pixel {pixel_key}: PyTorch={len(torch_group)}, CUDA={len(cuda_group)}")
            all_match = False
            mismatched_groups.append(pixel_key)
            continue
        
        # Compare sorted depths and indices
        torch_depths = [item[1] for item in torch_group]
        cuda_depths = [item[1] for item in cuda_group]
        
        torch_orig_indices = [item[2] for item in torch_group]
        cuda_orig_indices = [item[2] for item in cuda_group]
        
        # Check depth sorting
        if not np.allclose(torch_depths, cuda_depths, rtol=1e-12):
            log.error(f"❌ Different sorted depths for pixel {pixel_key}")
            log.error(f"  PyTorch depths: {torch_depths}")
            log.error(f"  CUDA depths: {cuda_depths}")
            all_match = False
            mismatched_groups.append(pixel_key)
            continue
        
        # Check original indices
        if torch_orig_indices != cuda_orig_indices:
            log.error(f"❌ Different sorted indices for pixel {pixel_key}")
            log.error(f"  PyTorch indices: {torch_orig_indices}")
            log.error(f"  CUDA indices: {cuda_orig_indices}")
            all_match = False
            mismatched_groups.append(pixel_key)
            continue
    
    if all_match:
        log.info("✅ ALL PIXEL GROUPS MATCH PERFECTLY!")
        log.info("  - Same pixel coordinates")
        log.info("  - Same group sizes")
        log.info("  - Same depth sorting")
        log.info("  - Same original indices")
        
        # Show sample groups for verification
        log.info("\nSample verification (first 3 groups):")
        for i, pixel_key in enumerate(sorted(torch_keys)[:3]):
            torch_group = torch_groups[pixel_key]
            log.info(f"  Pixel {pixel_key} (size: {len(torch_group)}):")
            log.info(f"    Depths: {[f'{item[1]:.3f}' for item in torch_group]}")
            log.info(f"    Indices: {[item[2] for item in torch_group]}")
        
        return True
    else:
        log.error(f"❌ {len(mismatched_groups)} groups have mismatches")
        log.error("This indicates a bug in the deterministic grouping/sorting steps!")
        return False

def main():
    """Main verification function"""
    log.info("Verifying pixel grouping and sorting consistency...")
    log.info("Testing deterministic steps 1-2 of clustering algorithm")
    
    # Load test data (same as test_cuda_clustering.py pose 80)
    data = load_test_data(pose_id=80)
    
    # Extract pixel groups using PyTorch approach (steps 1-2)
    torch_groups = extract_torch_pixel_groups(
        data['candidate_means'], data['candidate_indices'],
        data['viewmat'], data['K'], data['width'], data['height']
    )
    
    # Extract pixel groups using actual CUDA approach (steps 1-2)
    cuda_groups = extract_cuda_pixel_groups_step2(
        data['candidate_means'], data['candidate_indices'],
        data['viewmat'], data['K'], data['width'], data['height']
    )
    
    # Compare the results
    success = compare_pixel_groups(torch_groups, cuda_groups)
    
    log.info("\n" + "="*80)
    if success:
        log.info("🎉 VERIFICATION SUCCESSFUL!")
        log.info("Steps 1-2 (pixel grouping + sorting) are deterministically identical")
        log.info("Any differences in final clustering are due to step 3 (depth clustering)")
    else:
        log.error("💥 VERIFICATION FAILED!")
        log.error("Steps 1-2 should be deterministic - there's a bug to fix!")
    log.info("="*80)

if __name__ == "__main__":
    main()
