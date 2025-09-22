#!/usr/bin/env python3
"""
CUDA vs CPU Clustering Comparison using Real Gaussian Splatting Data

This script compares the performance and correctness of:
1. CPU implementation: _cluster_center_in_pixel() from merging.py
2. CUDA implementation: cluster_center_in_pixel_cuda_accelerated() 

It loads real Gaussian splat data from disk and uses a specific pose for testing.
"""

import torch
from torch import Tensor
import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Callable
import time
import sys
import os
import json

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.config import Config, load_config_from_toml, merge_config
from scripts.utils import set_result_dir, load_checkpoint, load_poses
from stream.culling import calc_pixel_area
from stream.merging import find_merge_candidates, _cluster_center_in_pixel
from stream.clustering_cuda import cluster_center_in_pixel_cuda_accelerated

def load_scene_data(cfg: Config, pose_id: int = 88) -> Dict:
    """Load Gaussian splat data and pose from disk."""
    print(f"Loading scene data...")
    print(f"  Checkpoint: {cfg.ckpt}")
    print(f"  Pose ID: {pose_id}")
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    assert os.path.exists(cfg.ckpt), f"Checkpoint not found: {cfg.ckpt}"
    means, quats, scales, opacities, colors = load_checkpoint(cfg.ckpt, device)
    print(f"  Loaded {means.shape[0]} Gaussians")
    
    # Load poses
    poses_dir = os.path.join(cfg.result_dir, "viewer_poses")
    poses_file = os.path.join(poses_dir, "viewer_poses.json")
    assert os.path.exists(poses_file), f"Poses file not found: {poses_file}"
        
    poses = load_poses(poses_file)
    print(f"  Loaded {len(poses)} poses")
    
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
    
    print(f"  Image size: {width}x{height}")
    print(f"  Distance to bbox: {target_pose['distance_to_bbox_center']:.3f}m")
    
    return scene_data


def find_merge_candidates_with_threshold(
    scene_data: Dict,
    pixel_area_threshold: float = 2.0,
    use_pixel_area: bool = True,
    scale_modifier: float = 1.0,
    eps2d: float = 0.3,
    method: str = "cuda"
) -> Tuple[np.ndarray, Tensor]:
    """Find merge candidates using pixel area threshold."""
    print(f"\nFinding merge candidates with threshold {pixel_area_threshold}...")
    
    # Use the same method as in merging.py
    candidate_mask = find_merge_candidates(
        scene_data['means'], scene_data['quats'], scene_data['scales'], scene_data['opacities'],
        scene_data['viewmat'], scene_data['K'], scene_data['width'], scene_data['height'],
        use_pixel_area=use_pixel_area, pixel_area_threshold=pixel_area_threshold,
        scale_modifier=scale_modifier, eps2d=eps2d, method=method
    )
    
    candidate_indices = torch.where(candidate_mask)[0].cpu().numpy()
    candidate_means = scene_data['means'][candidate_mask]
    
    print(f"  Found {len(candidate_indices)} candidates out of {scene_data['means'].shape[0]} Gaussians")
    print(f"  Candidate percentage: {len(candidate_indices)/scene_data['means'].shape[0]*100:.1f}%")
    
    return candidate_indices, candidate_means


def compare_clustering_implementations(
    candidate_indices: np.ndarray, 
    candidate_means: Tensor,
    clustering_params: Dict
) -> None:
    """Compare CPU and CUDA clustering implementations."""
    print(f"\n{'='*80}")
    print("CLUSTERING IMPLEMENTATION COMPARISON")
    print(f"{'='*80}")
    
    print(f"Testing parameters:")
    print(f"  Candidates: {len(candidate_indices)}")
    print(f"  Image size: {clustering_params['width']}x{clustering_params['height']}")
    print(f"  Depth threshold: {clustering_params['depth_threshold']}")
    print(f"  Min cluster size: {clustering_params['min_cluster_size']}")

     # Test CUDA implementation
    print(f"\n--- CUDA Implementation ---")
    cuda_clusters = None
    cuda_time_ms = None
    
    try:
        start_time = time.time()
        cuda_clusters = cluster_center_in_pixel_cuda_accelerated(
            candidate_means, candidate_indices,
            clustering_params['viewmat'], clustering_params['K'],
            clustering_params['width'], clustering_params['height'],
            depth_threshold=clustering_params['depth_threshold'],
            min_cluster_size=clustering_params['min_cluster_size']
        )
        cuda_time_ms = (time.time() - start_time) * 1000
        
        print(f"✅ CUDA clustering successful!")
        print(f"   Time: {cuda_time_ms:.2f} ms")
        print(f"   Clusters found: {len(cuda_clusters)}")
        
        if len(cuda_clusters) > 0:
            cluster_sizes = [len(cluster) for cluster in cuda_clusters]
            total_clustered = sum(cluster_sizes)
            print(f"   Total Gaussians clustered: {total_clustered}")
            print(f"   Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}")
            
            # Show first few clusters for verification  
            print(f"   First 3 clusters indices: {[cluster[:3].tolist() for cluster in cuda_clusters[:3]]}")
    except Exception as e:
        print(f"❌ CUDA clustering failed: {e}")
    
    # Test CPU implementation (reference)
    print(f"\n--- CPU Implementation (Reference) ---")
    try:
        start_time = time.time()
        cpu_clusters = _cluster_center_in_pixel(
            candidate_means, candidate_indices, **clustering_params
        )
        cpu_time_ms = (time.time() - start_time) * 1000
        
        print(f"✅ CPU clustering successful!")
        print(f"   Time: {cpu_time_ms:.2f} ms")
        print(f"   Clusters found: {len(cpu_clusters)}")
        
        if len(cpu_clusters) > 0:
            cluster_sizes = [len(cluster) for cluster in cpu_clusters]
            total_clustered = sum(cluster_sizes)
            print(f"   Total Gaussians clustered: {total_clustered}")
            print(f"   Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}")
            
            # Show first few clusters for verification
            print(f"   First 3 clusters indices: {[cluster[:3].tolist() for cluster in cpu_clusters[:3]]}")
        
    except Exception as e:
        print(f"❌ CPU clustering failed: {e}")
        cpu_clusters = None
        cpu_time_ms = None
    
    # Performance and correctness comparison
    print(f"\n--- COMPARISON RESULTS ---")
    assert cpu_clusters is not None and cuda_clusters is not None, "Both implementations failed"
    
    # Performance comparison
    speedup = cpu_time_ms / cuda_time_ms
    print(f"Performance:")
    print(f"  CPU time: {cpu_time_ms:.2f} ms")
    print(f"  CUDA time: {cuda_time_ms:.2f} ms")  
    print(f"  Speedup: {speedup:.2f}x")
    
    # Correctness comparison
    print(f"\nCorrectness:")
    print(f"  CPU clusters: {len(cpu_clusters)}")
    print(f"  CUDA clusters: {len(cuda_clusters)}")
    
    assert len(cpu_clusters) == len(cuda_clusters), f"Different number of clusters: CPU={len(cpu_clusters)}, CUDA={len(cuda_clusters)}"
        
    # Check if clusters contain same elements (order might differ)
    cpu_total = sum(len(cluster) for cluster in cpu_clusters)
    cuda_total = sum(len(cluster) for cluster in cuda_clusters)
    
    assert cpu_total == cuda_total, f"Different total number of clustered Gaussians: CPU={cpu_total}, CUDA={cuda_total}"
            
    # Convert clusters to sets for comparison
    cpu_clustered_indices = set()
    for cluster in cpu_clusters:
        cpu_clustered_indices.update(cluster)
                
    cuda_clustered_indices = set()
    for cluster in cuda_clusters:
        cuda_clustered_indices.update(cluster)
                
    if cpu_clustered_indices == cuda_clustered_indices:
        print(f"  ✅ Identical sets of clustered Gaussians")
    else:
        print(f"  ⚠️  Different sets of clustered Gaussians")
        diff1 = cpu_clustered_indices - cuda_clustered_indices
        diff2 = cuda_clustered_indices - cpu_clustered_indices
        print(f"       CPU only: {len(diff1)} indices")
        print(f"       CUDA only: {len(diff2)} indices")

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

def main():
    """Main function."""
    print("CUDA vs CPU Clustering Comparison")
    print("=" * 50)
    
    # Setup configuration
    cfg = setup_config()
    assert cfg is not None, "Configuration setup failed"
    
    # Load scene data
    scene_data = load_scene_data(cfg, pose_id=88)
    assert scene_data is not None, "Scene data loading failed"
    
    # Find merge candidates
    candidate_indices, candidate_means = find_merge_candidates_with_threshold(
        scene_data, pixel_area_threshold=2.0
    )
    assert len(candidate_indices) > 0, "No merge candidates found - try a higher threshold"
    assert len(candidate_means) > 0, "No candidate means found"
    assert isinstance(candidate_means, Tensor), "Candidate means must be a torch tensor"
    
    # Compare implementations
    clustering_params = {
        'viewmat': scene_data['viewmat'],
        'K': scene_data['K'],
        'width': scene_data['width'],
        'height': scene_data['height'],
        'depth_threshold': 0.1,
        'min_cluster_size': 2
    }
    compare_clustering_implementations(candidate_indices, candidate_means, clustering_params)
    
    print(f"\n{'='*50}")
    print("Comparison completed successfully!")

if __name__ == "__main__":
    main()


'''
# Debug using computer-sanitizer
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python3 -c "import torch; print(torch.__path__[0])")/lib compute-sanitizer --tool memcheck python stream/tests/test_cuda_clustering.py
'''