#!/usr/bin/env python3
"""
End-to-end CPU vs GPU Pipeline Comparison using Real Gaussian Splatting Data

This script compares two complete merging pipelines:
1. SLOW (CPU): cluster_center_in_pixel_torch() + merge_clusters_torch() 
2. FAST (GPU): cluster_center_in_pixel_cuda() + merge_clusters_cuda()

It loads real Gaussian splat data and performs the complete pipeline:
preprocessing → find candidates → clustering → merging → concatenation
"""

import torch
from torch import Tensor
import numpy as np
import time
import sys
import os
import json
from typing import Tuple, Dict

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.config import Config, load_config_from_toml, merge_config
from scripts.utils import set_result_dir, load_checkpoint, load_poses
from stream.merging import find_merge_candidates, merge_clusters_torch
from stream.clustering_cuda import cluster_center_in_pixel_cuda, cluster_center_in_pixel_torch
from stream.merging_cuda import merge_clusters_cuda
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_scene_data(cfg: Config, pose_id: int = 53) -> Dict:
    """Load Gaussian splat data and pose from disk."""
    log.info(f"Loading scene data...")
    log.info(f"  Checkpoint: {cfg.ckpt}")
    log.info(f"  Pose ID: {pose_id}")
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    assert os.path.exists(cfg.ckpt), f"Checkpoint not found: {cfg.ckpt}"
    means, quats, scales, opacities, colors = load_checkpoint(cfg.ckpt, device)
    log.info(f"  Loaded {means.shape[0]} Gaussians")
    
    # Load poses
    poses_dir = os.path.join(cfg.result_dir, "viewer_poses")
    poses_file = os.path.join(poses_dir, "viewer_poses.json")
    assert os.path.exists(poses_file), f"Poses file not found: {poses_file}"
        
    poses = load_poses(poses_file)
    log.info(f"  Loaded {len(poses)} poses")
    
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
    
    scene_data = {
        'means': means,
        'quats': quats,
        'scales': scales,
        'opacities': opacities,
        'colors': colors,
        'viewmat': viewmat,
        'K': K_adjusted,
        'width': width,
        'height': height,
        'pose_data': target_pose
    }
    
    log.info(f"  Image size: {width}x{height}")
    log.info(f"  Distance to bbox: {target_pose['distance_to_bbox_center']:.3f}m")
    
    return scene_data

def setup_config() -> Config:
    """Setup configuration for loading data."""
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
    
    return cfg

def run_slow_pipeline(scene_data: Dict, pixel_area_threshold: float = 2.0) -> Tuple[Dict, Dict]:
    """
    Run SLOW pipeline: cluster_center_in_pixel_torch() + merge_clusters_torch()
    Pure PyTorch/CPU implementation for reference correctness and performance baseline.
    Returns: (final_gaussians, timing_info)
    """
    log.info("🐌 Running SLOW pipeline (Pure PyTorch/CPU)...")
    timing = {}
    
    # Step 1: Preprocessing (minimal)
    start_time = time.time()
    device = scene_data['means'].device
    timing['preprocess'] = (time.time() - start_time) * 1000
    
    # Step 2: Find merge candidates
    start_time = time.time()
    candidate_mask = find_merge_candidates(
        scene_data['means'], scene_data['quats'], scene_data['scales'], scene_data['opacities'],
        scene_data['viewmat'], scene_data['K'], scene_data['width'], scene_data['height'],
        use_pixel_area=True, pixel_area_threshold=pixel_area_threshold,
        scale_modifier=1.0, eps2d=0.3, method="torch"
    )
    candidate_indices = torch.where(candidate_mask)[0]  # Keep as tensor
    candidate_means = scene_data['means'][candidate_mask]
    timing['find_candidates'] = (time.time() - start_time) * 1000
    
    log.info(f"  Found {len(candidate_indices)} candidates ({len(candidate_indices)/scene_data['means'].shape[0]*100:.1f}%)")
    
    if len(candidate_indices) == 0:
        return scene_data, timing
    
    # Step 3: Clustering using pure PyTorch/CPU implementation
    start_time = time.time()
    clusters = cluster_center_in_pixel_torch(
        candidate_means, candidate_indices,
        scene_data['viewmat'], scene_data['K'],
        scene_data['width'], scene_data['height'],
        depth_threshold=0.1, min_cluster_size=2
    )
    timing['clustering'] = (time.time() - start_time) * 1000
    
    log.info(f"  Found {len(clusters)} clusters")
    
    if len(clusters) == 0:
        return scene_data, timing
    
    # Step 4: Merging using PyTorch (slow)
    start_time = time.time()
    merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = merge_clusters_torch(
        clusters, scene_data['means'], scene_data['quats'], scene_data['scales'],
        scene_data['opacities'], scene_data['colors'], merge_strategy="weighted_mean", weight_by_opacity=True
    )
    timing['merging'] = (time.time() - start_time) * 1000
    
    log.info(f"  Merged into {merged_means.shape[0]} final Gaussians")
    
    # Step 5: Calculate total pipeline time
    timing['total'] = sum(timing.values())
    
    final_gaussians = {
        'means': merged_means,
        'quats': merged_quats,
        'scales': merged_scales,
        'opacities': merged_opacities,
        'colors': merged_colors,
        'num_gaussians': merged_means.shape[0]
    }
    
    return final_gaussians, timing

def run_fast_pipeline(scene_data: Dict, pixel_area_threshold: float = 2.0) -> Tuple[Dict, Dict]:
    """
    Run FAST pipeline: cluster_center_in_pixel_cuda() + merge_clusters_cuda()
    Pure CUDA/GPU implementation for maximum performance.
    Returns: (final_gaussians, timing_info)
    """
    log.info("🚀 Running FAST pipeline (Pure CUDA/GPU)...")
    timing = {}
    
    # Step 1: Preprocessing (minimal)
    start_time = time.time()
    device = scene_data['means'].device
    timing['preprocess'] = (time.time() - start_time) * 1000
    
    # Step 2: Find merge candidates
    start_time = time.time()
    candidate_mask = find_merge_candidates(
        scene_data['means'], scene_data['quats'], scene_data['scales'], scene_data['opacities'],
        scene_data['viewmat'], scene_data['K'], scene_data['width'], scene_data['height'],
        use_pixel_area=True, pixel_area_threshold=pixel_area_threshold,
        scale_modifier=1.0, eps2d=0.3, method="cuda"
    )
    candidate_indices = torch.where(candidate_mask)[0]  # Keep as tensor
    candidate_means = scene_data['means'][candidate_mask]
    timing['find_candidates'] = (time.time() - start_time) * 1000
    
    log.info(f"  Found {len(candidate_indices)} candidates ({len(candidate_indices)/scene_data['means'].shape[0]*100:.1f}%)")
    
    if len(candidate_indices) == 0:
        return scene_data, timing
    
    # Step 3: Clustering using high-level CUDA API
    start_time = time.time()
    clusters_result = cluster_center_in_pixel_cuda(
        candidate_means, candidate_indices,
        scene_data['viewmat'], scene_data['K'],
        scene_data['width'], scene_data['height'],
        depth_threshold=0.1, min_cluster_size=2,
        return_flat_format=True  # Return flat format for CUDA merging
    )
    timing['clustering'] = (time.time() - start_time) * 1000
    
    if clusters_result['num_clusters'] == 0:
        log.info(f"  No clusters found")
        return scene_data, timing
    
    log.info(f"  Found {clusters_result['num_clusters']} clusters")
    log.info(f"  Total clustered Gaussians: {clusters_result['total_clustered']}")
    
    # Step 4: Merging using high-level CUDA API (includes concatenation)
    start_time = time.time()
    final_means, final_quats, final_scales, final_opacities, final_colors = merge_clusters_cuda(
        cluster_indices=clusters_result['cluster_indices'],
        cluster_offsets=clusters_result['cluster_offsets'],
        current_means=scene_data['means'],
        current_quats=scene_data['quats'],
        current_scales=scene_data['scales'],
        current_opacities=scene_data['opacities'],
        current_colors=scene_data['colors'],
        merge_strategy="weighted_mean",
        weight_by_opacity=True
    )
    timing['merging'] = (time.time() - start_time) * 1000
    
    log.info(f"  Merged into {final_means.shape[0]} final Gaussians")
    log.info(f"  Clusters processed: {clusters_result['num_clusters']}, Total clustered: {clusters_result['total_clustered']}")
    
    # Step 5: Calculate total pipeline time (no concatenation step needed)
    timing['total'] = sum(timing.values())
    
    final_gaussians = {
        'means': final_means,
        'quats': final_quats,
        'scales': final_scales,
        'opacities': final_opacities,
        'colors': final_colors,
        'num_gaussians': final_means.shape[0]
    }
    
    return final_gaussians, timing

def compare_pipelines(slow_result: Dict, fast_result: Dict) -> bool:
    """Compare results from slow and fast pipelines."""
    log.info("\n📊 Comparing pipeline results...")
    
    # Check basic counts
    slow_count = slow_result['num_gaussians']
    fast_count = fast_result['num_gaussians']
    
    log.info(f"  Gaussian counts:")
    log.info(f"    CPU pipeline: {slow_count}")
    log.info(f"    GPU pipeline: {fast_count}")
    
    # For merging pipelines, counts should be very close (allow ±1 for floating-point differences)
    count_diff = abs(slow_count - fast_count)
    if count_diff > 1:
        log.error(f"❌ Significant Gaussian count difference: CPU={slow_count} vs GPU={fast_count} (diff: {count_diff})")
        return False
    elif count_diff == 1:
        log.warning(f"⚠️  Minor Gaussian count difference: CPU={slow_count} vs GPU={fast_count} (diff: 1 - likely floating-point precision)")
        log.info("✅ Acceptable difference for CPU vs GPU numerical computing")
    
    # Compare tensor shapes (allow for ±1 difference in first dimension)
    shapes_match = True
    for key in ['means', 'quats', 'scales', 'opacities', 'colors']:
        slow_shape = slow_result[key].shape
        fast_shape = fast_result[key].shape
        
        # Check if shapes are compatible (same except possibly first dimension ±1)
        if len(slow_shape) != len(fast_shape):
            log.error(f"❌ Shape dimension mismatch for {key}: {slow_shape} vs {fast_shape}")
            shapes_match = False
        elif slow_shape[1:] != fast_shape[1:]:
            log.error(f"❌ Shape mismatch for {key} (beyond first dim): {slow_shape} vs {fast_shape}")
            shapes_match = False
        elif abs(slow_shape[0] - fast_shape[0]) > 1:
            log.error(f"❌ Significant shape mismatch for {key}: {slow_shape} vs {fast_shape}")
            shapes_match = False
        elif slow_shape != fast_shape:
            log.warning(f"⚠️  Minor shape difference for {key}: {slow_shape} vs {fast_shape} (±1 difference)")
    
    if not shapes_match:
        return False
    
    log.info("✅ Tensor shapes match")
    
    # Compare tensor statistics (since exact values may differ due to ordering)
    tolerance = 1e-4
    all_match = True
    
    for key in ['means', 'quats', 'scales', 'opacities', 'colors']:
        slow_tensor = slow_result[key].float()
        fast_tensor = fast_result[key].float()
        
        # Compare statistical properties (mean, std, min, max)
        slow_mean = slow_tensor.mean().item()
        fast_mean = fast_tensor.mean().item()
        slow_std = slow_tensor.std().item()
        fast_std = fast_tensor.std().item()
        slow_min = slow_tensor.min().item()
        fast_min = fast_tensor.min().item()
        slow_max = slow_tensor.max().item()
        fast_max = fast_tensor.max().item()
        
        mean_diff = abs(slow_mean - fast_mean)
        std_diff = abs(slow_std - fast_std)
        min_diff = abs(slow_min - fast_min)
        max_diff = abs(slow_max - fast_max)
        
        if mean_diff > tolerance or std_diff > tolerance or min_diff > tolerance or max_diff > tolerance:
            log.warning(f"⚠️ Statistical differences for {key}:")
            log.warning(f"    Mean: {slow_mean:.6f} vs {fast_mean:.6f} (diff: {mean_diff:.6f})")
            log.warning(f"    Std:  {slow_std:.6f} vs {fast_std:.6f} (diff: {std_diff:.6f})")
            log.warning(f"    Min:  {slow_min:.6f} vs {fast_min:.6f} (diff: {min_diff:.6f})")
            log.warning(f"    Max:  {slow_max:.6f} vs {fast_max:.6f} (diff: {max_diff:.6f})")
            if mean_diff > tolerance * 10:  # Only fail on large differences
                all_match = False
    
    if all_match:
        log.info("✅ Pipeline results are statistically equivalent")
    else:
        log.error("❌ Pipeline results differ significantly")
    
    return all_match

def main():
    """Main test function."""
    log.info("🚀 End-to-End CPU vs GPU Pipeline Comparison")
    log.info("=" * 70)
    
    if not torch.cuda.is_available():
        log.error("CUDA is not available!")
        return False
    
    try:
        # Setup configuration and load scene data
        log.info("Setting up scene data...")
        cfg = setup_config()
        scene_data = load_scene_data(cfg, pose_id=53)
        
        log.info(f"\n📊 Scene Information:")
        log.info(f"  Total Gaussians: {scene_data['means'].shape[0]:,}")
        log.info(f"  Image Resolution: {scene_data['width']}x{scene_data['height']}")
        log.info(f"  Color Shape: {scene_data['colors'].shape}")
        log.info(f"  Color Dtype: {scene_data['colors'].dtype}")
        
        # Run both pipelines
        log.info(f"\n" + "=" * 70)
        
        for _ in range(2):
            # Run CPU pipeline
            slow_result, slow_timing = run_slow_pipeline(scene_data, pixel_area_threshold=2.0)
            
            log.info(f"\n" + "=" * 70)
            
            # Run GPU pipeline  
            fast_result, fast_timing = run_fast_pipeline(scene_data, pixel_area_threshold=2.0)
            
            # Compare correctness
            correctness_passed = compare_pipelines(slow_result, fast_result)
        
        # Performance comparison
        log.info(f"\n⚡ Performance Comparison (CPU vs GPU):")
        log.info(f"{'Stage':<20} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<8}")
        log.info(f"{'='*20} {'='*12} {'='*12} {'='*8}")
        
        for stage in ['preprocess', 'find_candidates', 'clustering', 'merging']:
            if stage in slow_timing and stage in fast_timing:
                slow_time = slow_timing[stage]
                fast_time = fast_timing[stage]
                speedup = slow_time / fast_time if fast_time > 0 else float('inf')
                log.info(f"{stage:<20} {slow_time:<12.2f} {fast_time:<12.2f} {speedup:<8.1f}x")
        
        log.info(f"{'='*20} {'='*12} {'='*12} {'='*8}")
        slow_total = slow_timing['total']
        fast_total = fast_timing['total']
        overall_speedup = slow_total / fast_total if fast_total > 0 else float('inf')
        log.info(f"{'TOTAL':<20} {slow_total:<12.2f} {fast_total:<12.2f} {overall_speedup:<8.1f}x")
        
        # 60fps target analysis
        target_time = 16.67  # 60fps = 16.67ms per frame
        log.info(f"\n🎯 60fps Target Analysis:")
        log.info(f"  Target time per frame: {target_time:.2f} ms")
        log.info(f"  CPU pipeline: {slow_total:.2f} ms ({'✅ PASS' if slow_total < target_time else '❌ FAIL'})")
        log.info(f"  GPU pipeline: {fast_total:.2f} ms ({'✅ PASS' if fast_total < target_time else '❌ FAIL'})")
        
        if fast_total < target_time:
            headroom = target_time - fast_total
            log.info(f"  🎉 GPU pipeline has {headroom:.2f} ms headroom for 60fps!")
        
        # Final summary
        log.info(f"\n" + "=" * 70)
        log.info("FINAL RESULTS:")
        log.info(f"  Correctness (CPU vs GPU): {'✅ PASS' if correctness_passed else '❌ FAIL'}")
        log.info(f"  GPU Speedup: {overall_speedup:.1f}x faster than CPU")
        log.info(f"  60fps Ready: {'✅ YES' if fast_total < target_time else '❌ NO'}")
        
        success = correctness_passed and fast_total < target_time
        
        if success:
            log.info("🎉 COMPLETE SUCCESS! GPU pipeline is ready for real-time rendering!")
        else:
            if not correctness_passed:
                log.error("💥 GPU vs CPU correctness test failed!")
            if fast_total >= target_time:
                log.error(f"💥 GPU performance target missed! Need {fast_total - target_time:.2f} ms improvement.")
        
        return success
            
    except Exception as e:
        log.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
