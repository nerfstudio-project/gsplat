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
from typing import Tuple, Dict
import time
import sys
import os
import json

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.config import Config, load_config_from_toml, merge_config
from scripts.utils import set_result_dir, load_checkpoint, load_poses
from stream.merging import find_merge_candidates
from stream.clustering_cuda import cluster_center_in_pixel_cuda, cluster_center_in_pixel_torch

import logging
log = logging.getLogger() # Use root logger
log.setLevel(logging.DEBUG)

# Create the handler for console output (stdout)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Log INFO and higher levels to console
console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s - %(funcName)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Create the handler for file output
file_handler = logging.FileHandler(f"{__name__}.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s - %(funcName)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add the handlers to the logger
log.addHandler(console_handler)
log.addHandler(file_handler)

def load_scene_data(cfg: Config, pose_id: int = 88) -> Dict:
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


def find_merge_candidates_with_threshold(
    scene_data: Dict,
    pixel_area_threshold: float = 2.0,
    use_pixel_area: bool = True,
    scale_modifier: float = 1.0,
    eps2d: float = 0.3,
    method: str = "cuda"
) -> Tuple[np.ndarray, Tensor]:
    """Find merge candidates using pixel area threshold."""
    log.info(f"\nFinding merge candidates with threshold {pixel_area_threshold}...")
    
    # Use the same method as in merging.py
    candidate_mask = find_merge_candidates(
        scene_data['means'], scene_data['quats'], scene_data['scales'], scene_data['opacities'],
        scene_data['viewmat'], scene_data['K'], scene_data['width'], scene_data['height'],
        use_pixel_area=use_pixel_area, pixel_area_threshold=pixel_area_threshold,
        scale_modifier=scale_modifier, eps2d=eps2d, method=method
    )
    candidate_indices = torch.where(candidate_mask)[0]  # Keep as tensor for optimized functions
    candidate_means = scene_data['means'][candidate_mask]
    
    log.info(f"  Found {len(candidate_indices)} candidates out of {scene_data['means'].shape[0]} Gaussians")
    log.info(f"  Candidate percentage: {len(candidate_indices)/scene_data['means'].shape[0]*100:.1f}%")
    
    return candidate_indices, candidate_means


def compare_clustering_implementations(
    candidate_indices: Tensor, 
    candidate_means: Tensor,
    clustering_params: Dict
) -> None:
    """Compare CPU and CUDA clustering implementations."""
    log.info(f"\n{'='*80}")
    log.info("CLUSTERING IMPLEMENTATION COMPARISON")
    log.info(f"{'='*80}")
    
    log.info(f"Testing parameters:")
    log.info(f"  Candidates: {len(candidate_indices)}")
    log.info(f"  Image size: {clustering_params['width']}x{clustering_params['height']}")
    log.info(f"  Depth threshold: {clustering_params['depth_threshold']}")
    log.info(f"  Min cluster size: {clustering_params['min_cluster_size']}")

    # Test CUDA implementation
    log.info(f"\n--- CUDA Implementation ---")
    cuda_clusters = None
    cuda_time_ms = None
    
    try:
        start_time = time.time()
        cuda_clusters = cluster_center_in_pixel_cuda(
            candidate_means, candidate_indices,
            clustering_params['viewmat'], clustering_params['K'],
            clustering_params['width'], clustering_params['height'],
            depth_threshold=clustering_params['depth_threshold'],
            min_cluster_size=clustering_params['min_cluster_size']
        )
        cuda_time_ms = (time.time() - start_time) * 1000
        log.debug(f"Total CUDA time: {cuda_time_ms:.2f} ms")
        
        log.info(f"✅ CUDA clustering successful!")
        log.info(f"   Time: {cuda_time_ms:.2f} ms")
        log.info(f"   Clusters found: {len(cuda_clusters)}")
        
        if len(cuda_clusters) > 0:
            # OPTIMIZED: Use vectorized tensor operations instead of Python loops
            cuda_cluster_sizes = torch.tensor([cluster.size(0) for cluster in cuda_clusters], device=cuda_clusters[0].device)
            total_clustered = cuda_cluster_sizes.sum().item()
            log.info(f"   Total Gaussians clustered: {total_clustered}")
            log.info(f"   Cluster sizes: min={cuda_cluster_sizes.min().item()}, max={cuda_cluster_sizes.max().item()}, mean={cuda_cluster_sizes.float().mean().item():.1f}")
            
            # Show first few clusters for verification (minimal GPU→CPU transfer)
            first_3_samples = [cluster[:3].cpu().tolist() for cluster in cuda_clusters[:3]]
            log.info(f"   First 3 clusters indices: {first_3_samples}")
    except Exception as e:
        log.info(f"❌ CUDA clustering failed: {e}")
    
    # Test CPU implementation (reference)
    log.info(f"\n--- CPU Implementation (Reference) ---")
    try:
        start_time = time.time()
        cpu_clusters = cluster_center_in_pixel_torch(
            candidate_means, candidate_indices, **clustering_params
        )
        cpu_time_ms = (time.time() - start_time) * 1000
        
        log.info(f"✅ CPU clustering successful!")
        log.info(f"   Time: {cpu_time_ms:.2f} ms")
        log.info(f"   Clusters found: {len(cpu_clusters)}")
        
        if len(cpu_clusters) > 0:
            # OPTIMIZED: Use vectorized tensor operations instead of Python loops
            cpu_cluster_sizes = torch.tensor([cluster.size(0) for cluster in cpu_clusters], device=cpu_clusters[0].device)
            total_clustered = cpu_cluster_sizes.sum().item()
            log.info(f"   Total Gaussians clustered: {total_clustered}")
            log.info(f"   Cluster sizes: min={cpu_cluster_sizes.min().item()}, max={cpu_cluster_sizes.max().item()}, mean={cpu_cluster_sizes.float().mean().item():.1f}")
            
            # Show first few clusters for verification (minimal GPU→CPU transfer)
            first_3_samples = [cluster[:3].cpu().tolist() for cluster in cpu_clusters[:3]]
            log.info(f"   First 3 clusters indices: {first_3_samples}")
        
    except Exception as e:
        log.info(f"❌ CPU clustering failed: {e}")
        cpu_clusters = None
        cpu_time_ms = None
    
    # Performance and correctness comparison
    log.info(f"\n--- COMPARISON RESULTS ---")
    assert cpu_clusters is not None and cuda_clusters is not None, "Both implementations failed"
    
    # Performance comparison
    speedup = cpu_time_ms / cuda_time_ms
    log.info(f"Performance:")
    log.info(f"  CPU time: {cpu_time_ms:.2f} ms")
    log.info(f"  CUDA time: {cuda_time_ms:.2f} ms")  
    log.info(f"  Speedup: {speedup:.2f}x")
    
    # Correctness comparison
    log.info(f"\nCorrectness:")
    log.info(f"  CPU clusters: {len(cpu_clusters)}")
    log.info(f"  CUDA clusters: {len(cuda_clusters)}")
    
    assert len(cpu_clusters) == len(cuda_clusters), f"Different number of clusters: CPU={len(cpu_clusters)}, CUDA={len(cuda_clusters)}"
    
    # OPTIMIZED: Compute cluster sizes using vectorized operations
    cpu_cluster_sizes = torch.tensor([cluster.size(0) for cluster in cpu_clusters], device=cpu_clusters[0].device)
    cuda_cluster_sizes = torch.tensor([cluster.size(0) for cluster in cuda_clusters], device=cuda_clusters[0].device)
    
    cpu_total = cpu_cluster_sizes.sum().item()
    cuda_total = cuda_cluster_sizes.sum().item()
    
    assert cpu_total == cuda_total, f"Different total number of clustered Gaussians: CPU={cpu_total}, CUDA={cuda_total}"
            
    # DETAILED CLUSTER-BY-CLUSTER COMPARISON
    log.info(f"\n--- DETAILED CLUSTER COMPARISON (First Few Clusters) ---")
    
    # Compare first few clusters individually
    num_clusters_to_check = min(5, len(cpu_clusters), len(cuda_clusters))
    log.info(f"  Comparing first {num_clusters_to_check} clusters individually...")
    
    individual_matches = 0
    for i in range(num_clusters_to_check):
        # OPTIMIZED: Use torch.isin instead of converting to Python sets
        cpu_cluster = cpu_clusters[i]
        cuda_cluster = cuda_clusters[i]
        
        # Check if clusters have same size and same elements (order-independent)
        if cpu_cluster.size(0) == cuda_cluster.size(0):
            # Use torch operations to check set equality efficiently
            cpu_sorted = torch.sort(cpu_cluster)[0]
            cuda_sorted = torch.sort(cuda_cluster)[0]
            is_match = torch.equal(cpu_sorted, cuda_sorted)
        else:
            is_match = False
        
        if is_match:
            individual_matches += 1
            log.info(f"    Cluster {i}: ✅ MATCH (size: {cpu_cluster.size(0)})")
            sample_indices = torch.sort(cpu_cluster)[0][:5].cpu().tolist()
            log.info(f"      Indices: {sample_indices}{'...' if cpu_cluster.size(0) > 5 else ''}")
        else:
            log.info(f"    Cluster {i}: ❌ DIFFERENT")
            cpu_sample = torch.sort(cpu_cluster)[0][:5].cpu().tolist()
            cuda_sample = torch.sort(cuda_cluster)[0][:5].cpu().tolist()
            log.info(f"      CPU  (size: {cpu_cluster.size(0)}): {cpu_sample}{'...' if cpu_cluster.size(0) > 5 else ''}")
            log.info(f"      CUDA (size: {cuda_cluster.size(0)}): {cuda_sample}{'...' if cuda_cluster.size(0) > 5 else ''}")
    
    log.info(f"  Individual cluster matches: {individual_matches}/{num_clusters_to_check}")
    
    # CLUSTER SIZE DISTRIBUTION ANALYSIS
    log.info(f"\n--- CLUSTER SIZE DISTRIBUTION COMPARISON ---")
    
    # OPTIMIZED: Use torch.unique instead of numpy
    cpu_unique_sizes, cpu_counts = torch.unique(cpu_cluster_sizes, return_counts=True)
    cuda_unique_sizes, cuda_counts = torch.unique(cuda_cluster_sizes, return_counts=True)
    
    # Convert to CPU for dictionary operations (minimal transfer)
    cpu_unique_sizes_list = cpu_unique_sizes.cpu().tolist()
    cpu_counts_list = cpu_counts.cpu().tolist()
    cuda_unique_sizes_list = cuda_unique_sizes.cpu().tolist()
    cuda_counts_list = cuda_counts.cpu().tolist()
    
    cpu_dist = dict(zip(cpu_unique_sizes_list, cpu_counts_list))
    cuda_dist = dict(zip(cuda_unique_sizes_list, cuda_counts_list))
    
    # Basic statistics comparison using torch
    log.info(f"  Statistical Comparison:")
    log.info(f"    CPU  - Min: {cpu_cluster_sizes.min().item()}, Max: {cpu_cluster_sizes.max().item()}, Mean: {cpu_cluster_sizes.float().mean().item():.1f}, Median: {cpu_cluster_sizes.float().median().item():.1f}")
    log.info(f"    CUDA - Min: {cuda_cluster_sizes.min().item()}, Max: {cuda_cluster_sizes.max().item()}, Mean: {cuda_cluster_sizes.float().mean().item():.1f}, Median: {cuda_cluster_sizes.float().median().item():.1f}")
    
    # Check if distributions are identical
    distributions_identical = cpu_dist == cuda_dist
    log.info(f"  Distribution Match: {'✅ IDENTICAL' if distributions_identical else '❌ DIFFERENT'}")
    
    if distributions_identical:
        log.info(f"    🎉 Perfect statistical equivalence!")
        log.info(f"    📊 Total cluster count: {len(cpu_clusters)}")
        log.info(f"    📊 Sample size distribution:")
        # Show first 10 size categories
        sample_sizes = sorted(cpu_dist.keys())[:10]
        for size in sample_sizes:
            log.info(f"        Size {size}: {cpu_dist[size]} clusters")
        if len(cpu_dist) > 10:
            log.info(f"        ... and {len(cpu_dist) - 10} more size categories")
    else:
        log.info(f"  Detailed Differences:")
        
        # Find all unique sizes across both
        all_sizes = set(cpu_dist.keys()) | set(cuda_dist.keys())
        differences = []
        
        for size in sorted(all_sizes):
            cpu_count = cpu_dist.get(size, 0)
            cuda_count = cuda_dist.get(size, 0)
            if cpu_count != cuda_count:
                differences.append((size, cpu_count, cuda_count))
        
        if differences:
            log.info(f"    Found {len(differences)} size categories with different counts:")
            for size, cpu_count, cuda_count in differences[:10]:  # Show first 10 differences
                log.info(f"      Size {size}: CPU={cpu_count}, CUDA={cuda_count}, diff={cuda_count-cpu_count}")
            if len(differences) > 10:
                log.info(f"      ... and {len(differences) - 10} more differences")
        else:
            log.info(f"    Unexpected: No differences found in individual size counts")
    
    # OVERALL SET COMPARISON - OPTIMIZED VERSION
    log.info(f"\n--- OVERALL SET COMPARISON ---")
    
    # ULTRA-OPTIMIZED: Use torch.cat to concatenate all clusters, then torch.unique
    if len(cpu_clusters) > 0:
        cpu_all_indices = torch.cat(cpu_clusters, dim=0)
        cpu_unique_indices = torch.unique(cpu_all_indices, sorted=True)
    else:
        cpu_unique_indices = torch.empty(0, dtype=torch.long, device=candidate_indices.device)
    
    if len(cuda_clusters) > 0:
        cuda_all_indices = torch.cat(cuda_clusters, dim=0)
        cuda_unique_indices = torch.unique(cuda_all_indices, sorted=True)
    else:
        cuda_unique_indices = torch.empty(0, dtype=torch.long, device=candidate_indices.device)
    
    # Check if the unique sets are identical using torch.equal
    sets_identical = torch.equal(cpu_unique_indices, cuda_unique_indices)
    
    if sets_identical:
        log.info(f"  ✅ Identical sets of clustered Gaussians")
        log.info(f"       Total unique indices: {len(cpu_unique_indices)}")
        sample_indices = cpu_unique_indices[:10].cpu().tolist()
        log.info(f"       Sample indices: {sample_indices}{'...' if len(cpu_unique_indices) > 10 else ''}")
    else:
        log.info(f"  ⚠️  Different sets of clustered Gaussians")
        
        # Find differences using torch operations
        cpu_only_mask = ~torch.isin(cpu_unique_indices, cuda_unique_indices)
        cuda_only_mask = ~torch.isin(cuda_unique_indices, cpu_unique_indices)
        
        cpu_only_indices = cpu_unique_indices[cpu_only_mask]
        cuda_only_indices = cuda_unique_indices[cuda_only_mask]
        
        log.info(f"       CPU only: {len(cpu_only_indices)} indices")
        log.info(f"       CUDA only: {len(cuda_only_indices)} indices")
        
        if len(cpu_only_indices) > 0:
            cpu_only_sample = cpu_only_indices[:5].cpu().tolist()
            log.info(f"       CPU only sample: {cpu_only_sample}{'...' if len(cpu_only_indices) > 5 else ''}")
        if len(cuda_only_indices) > 0:
            cuda_only_sample = cuda_only_indices[:5].cpu().tolist()
            log.info(f"       CUDA only sample: {cuda_only_sample}{'...' if len(cuda_only_indices) > 5 else ''}")
    
    # SUMMARY
    if individual_matches == num_clusters_to_check and sets_identical:
        log.info(f"\n  🎉 PERFECT MATCH: Individual clusters and overall sets are identical!")
    elif sets_identical:
        log.info(f"\n  ✅ EQUIVALENT RESULTS: Same overall clustering, different cluster ordering")
        log.info(f"  ⚡ {len(cpu_unique_indices)} clustered Gaussians with identical distributions")
    else:
        log.info(f"\n  ⚠️  DIFFERENT RESULTS: Clustering algorithms produced different outcomes")

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
    log.info("CUDA vs CPU Clustering Comparison")
    log.info("=" * 50)
    
    # Setup configuration
    cfg = setup_config()
    assert cfg is not None, "Configuration setup failed"
    
    # Load scene data
    scene_data = load_scene_data(cfg, pose_id=53)
    assert scene_data is not None, "Scene data loading failed"
    
    # Find merge candidates
    for _ in range(2):
        start_time = time.time()
        candidate_indices, candidate_means = find_merge_candidates_with_threshold(
            scene_data, pixel_area_threshold=2.0
        )
        end_time = time.time()
        log.info(f"Time taken to find merge candidates: {(end_time - start_time)*1000:.2f} ms")

    assert len(candidate_indices) > 0, "No merge candidates found - try a higher threshold"
    assert len(candidate_means) > 0, "No candidate means found"
    assert isinstance(candidate_means, Tensor), "Candidate means must be a torch tensor"
    assert isinstance(candidate_indices, Tensor), "Candidate indices must be a torch tensor"
    
    # Compare implementations
    clustering_params = {
        'viewmat': scene_data['viewmat'],
        'K': scene_data['K'],
        'width': scene_data['width'],
        'height': scene_data['height'],
        'depth_threshold': 0.1,
        'min_cluster_size': 2
    }

    for _ in range(2):
        compare_clustering_implementations(candidate_indices, candidate_means, clustering_params)
    
    log.info(f"\n{'='*50}")
    log.info("Comparison completed successfully!")

if __name__ == "__main__":
    main()


'''
# Debug using computer-sanitizer
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python3 -c "import torch; print(torch.__path__[0])")/lib compute-sanitizer --tool memcheck python stream/tests/test_cuda_clustering.py
'''