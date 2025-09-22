#!/usr/bin/env python3
"""
Debug script to systematically compare CPU and CUDA clustering implementations
to identify exactly where they produce different results.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
gsplat_root = os.path.dirname(current_dir)
sys.path.append(gsplat_root)

from stream.merging import _cluster_center_in_pixel
from stream.clustering_cuda import _cluster_center_in_pixel_cuda_impl
from gsplat.cuda._torch_impl import _world_to_cam
from gsplat.cuda._wrapper import proj


def trace_cpu_implementation(
    candidate_means: torch.Tensor,
    candidate_indices: np.ndarray,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    depth_threshold: float = 0.1,
    min_cluster_size: int = 2
) -> Dict:
    """Trace through CPU implementation step by step."""
    print("=== TRACING CPU IMPLEMENTATION ===")
    
    device = candidate_means.device
    M = len(candidate_means)
    print(f"Input: {M} candidate means")
    
    # Step 1: World to camera coordinates
    means_batch = candidate_means.unsqueeze(0)
    viewmat_batch = viewmat.unsqueeze(0).unsqueeze(0)
    dummy_covars = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, M, 3, 3)
    means_cam_batch, _ = _world_to_cam(means_batch, dummy_covars, viewmat_batch)
    means_cam = means_cam_batch.squeeze(0).squeeze(0)
    print(f"Step 1 - Camera coords: {means_cam.shape}, depth range: [{means_cam[:, 2].min():.3f}, {means_cam[:, 2].max():.3f}]")
    
    # Step 2: Project to 2D
    means_cam_proj = means_cam.unsqueeze(0).unsqueeze(0)
    K_batch = K.unsqueeze(0).unsqueeze(0)
    dummy_covars_cam = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, M, 3, 3)
    means2d_batch, _ = proj(means_cam_proj, dummy_covars_cam, K_batch, width, height)
    means2d = means2d_batch.squeeze(0).squeeze(0)
    print(f"Step 2 - 2D coords: {means2d.shape}, range: x[{means2d[:, 0].min():.1f}, {means2d[:, 0].max():.1f}], y[{means2d[:, 1].min():.1f}, {means2d[:, 1].max():.1f}]")
    
    # Step 3: Discretize
    pixel_coords = torch.floor(means2d).long()  # CPU uses .long()
    print(f"Step 3 - Discrete coords: {pixel_coords.dtype}, range: x[{pixel_coords[:, 0].min()}, {pixel_coords[:, 0].max()}], y[{pixel_coords[:, 1].min()}, {pixel_coords[:, 1].max()}]")
    
    # Step 4: Filter
    valid_mask = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height) &
        (means_cam[:, 2] > 0)
    )
    valid_count = valid_mask.sum().item()
    print(f"Step 4 - Valid filtering: {valid_count}/{M} candidates remain ({valid_count/M*100:.1f}%)")
    
    if valid_count == 0:
        return {"clusters": [], "debug_info": {"valid_count": 0}}
    
    # Step 5: Apply mask
    valid_indices = torch.where(valid_mask)[0]
    valid_pixel_coords = pixel_coords[valid_mask]
    valid_depths = means_cam[valid_mask, 2]
    valid_candidate_indices = candidate_indices[valid_indices.cpu().numpy()]
    print(f"Step 5 - After masking: {len(valid_candidate_indices)} valid candidates")
    
    # Step 6: Group by pixel
    pixel_groups = {}
    for i, (pixel_coord, depth, orig_idx) in enumerate(zip(valid_pixel_coords, valid_depths, valid_candidate_indices)):
        pixel_key = (pixel_coord[0].item(), pixel_coord[1].item())
        if pixel_key not in pixel_groups:
            pixel_groups[pixel_key] = []
        pixel_groups[pixel_key].append((i, depth.item(), orig_idx))
    
    print(f"Step 6 - Pixel grouping: {len(pixel_groups)} unique pixels")
    pixel_group_sizes = [len(group) for group in pixel_groups.values()]
    print(f"         Group sizes: min={min(pixel_group_sizes)}, max={max(pixel_group_sizes)}, mean={np.mean(pixel_group_sizes):.1f}")
    
    # Step 7: Depth clustering
    clusters = []
    total_clustered = 0
    
    for pixel_key, pixel_group in pixel_groups.items():
        if len(pixel_group) < min_cluster_size:
            continue
            
        # Sort by depth
        pixel_group.sort(key=lambda x: x[1])
        
        # Depth clustering
        depth_clusters = []
        current_cluster = [pixel_group[0]]
        
        for j in range(1, len(pixel_group)):
            depth_diff = abs(pixel_group[j][1] - pixel_group[j-1][1])
            
            if depth_diff <= depth_threshold:
                current_cluster.append(pixel_group[j])
            else:
                if len(current_cluster) >= min_cluster_size:
                    depth_clusters.append(current_cluster)
                current_cluster = [pixel_group[j]]
        
        # Last cluster
        if len(current_cluster) >= min_cluster_size:
            depth_clusters.append(current_cluster)
        
        # Convert to original indices
        for depth_cluster in depth_clusters:
            if len(depth_cluster) >= min_cluster_size:
                original_cluster = np.array([item[2] for item in depth_cluster])
                clusters.append(original_cluster)
                total_clustered += len(original_cluster)
    
    print(f"Step 7 - Depth clustering: {len(clusters)} clusters, {total_clustered} total clustered")
    if len(clusters) > 0:
        cluster_sizes = [len(c) for c in clusters]
        print(f"         Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}")
    
    return {
        "clusters": clusters,
        "debug_info": {
            "valid_count": valid_count,
            "pixel_groups": len(pixel_groups),
            "total_clustered": total_clustered
        }
    }


def trace_cuda_preprocessing(
    candidate_means: torch.Tensor,
    candidate_indices: np.ndarray,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    depth_threshold: float = 0.1,
    min_cluster_size: int = 2
) -> Dict:
    """Trace CUDA preprocessing (before kernel call) to compare with CPU."""
    print("\n=== TRACING CUDA PREPROCESSING ===")
    
    device = candidate_means.device
    M = len(candidate_means)
    print(f"Input: {M} candidate means")
    
    # Step 1: World to camera (same as CPU)
    means_batch = candidate_means.unsqueeze(0)
    viewmat_batch = viewmat.unsqueeze(0).unsqueeze(0)
    dummy_covars = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, M, 3, 3)
    means_cam_batch, _ = _world_to_cam(means_batch, dummy_covars, viewmat_batch)
    means_cam = means_cam_batch.squeeze(0).squeeze(0)
    print(f"Step 1 - Camera coords: {means_cam.shape}, depth range: [{means_cam[:, 2].min():.3f}, {means_cam[:, 2].max():.3f}]")
    
    # Step 2: Project to 2D (same as CPU)
    means_cam_proj = means_cam.unsqueeze(0).unsqueeze(0)
    K_batch = K.unsqueeze(0).unsqueeze(0)
    dummy_covars_cam = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, M, 3, 3)
    means2d_batch, _ = proj(means_cam_proj, dummy_covars_cam, K_batch, width, height)
    means2d = means2d_batch.squeeze(0).squeeze(0)
    print(f"Step 2 - 2D coords: {means2d.shape}, range: x[{means2d[:, 0].min():.1f}, {means2d[:, 0].max():.1f}], y[{means2d[:, 1].min():.1f}, {means2d[:, 1].max():.1f}]")
    
    # Step 3: Discretize - DIFFERENCE HERE!
    pixel_coords = torch.floor(means2d).int()  # CUDA uses .int() instead of .long()
    print(f"Step 3 - Discrete coords: {pixel_coords.dtype}, range: x[{pixel_coords[:, 0].min()}, {pixel_coords[:, 0].max()}], y[{pixel_coords[:, 1].min()}, {pixel_coords[:, 1].max()}]")
    
    # Step 4: Filter (same as CPU)
    valid_mask = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height) &
        (means_cam[:, 2] > 0)
    )
    valid_count = valid_mask.sum().item()
    print(f"Step 4 - Valid filtering: {valid_count}/{M} candidates remain ({valid_count/M*100:.1f}%)")
    
    if valid_count == 0:
        return {"valid_count": 0}
    
    # Step 5: Apply mask (same as CPU)
    valid_indices = torch.where(valid_mask)[0]
    valid_pixel_coords = pixel_coords[valid_mask]
    valid_means_cam = means_cam[valid_mask]
    valid_candidate_indices = candidate_indices[valid_indices.cpu().numpy()]
    print(f"Step 5 - After masking: {len(valid_candidate_indices)} valid candidates")
    
    return {
        "valid_count": valid_count,
        "valid_pixel_coords": valid_pixel_coords,
        "valid_means_cam": valid_means_cam,
        "valid_candidate_indices": valid_candidate_indices
    }


def compare_implementations_step_by_step(test_size: int = 1000):
    """Compare implementations with controlled test data."""
    print(f"COMPARING IMPLEMENTATIONS WITH {test_size} CANDIDATES")
    print("=" * 80)
    
    # Create controlled test data
    torch.manual_seed(42)
    device = 'cuda'
    
    # Create data that will actually cluster
    candidate_means = torch.zeros(test_size, 3, device=device)
    candidate_means[:, 0] = torch.randn(test_size, device=device) * 0.5  # x spread
    candidate_means[:, 1] = torch.randn(test_size, device=device) * 0.5  # y spread  
    candidate_means[:, 2] = 5.0 + torch.randn(test_size, device=device) * 0.2  # depth spread
    
    candidate_indices = np.arange(test_size)
    viewmat = torch.eye(4, device=device)
    K = torch.eye(3, device=device) * 100
    width, height = 100, 100
    depth_threshold = 0.1
    min_cluster_size = 2
    
    # Trace both implementations
    cpu_result = trace_cpu_implementation(
        candidate_means, candidate_indices, viewmat, K, width, height, depth_threshold, min_cluster_size
    )
    
    cuda_preprocessing = trace_cuda_preprocessing(
        candidate_means, candidate_indices, viewmat, K, width, height, depth_threshold, min_cluster_size
    )
    
    print("\n=== COMPARISON RESULTS ===")
    print(f"CPU valid count: {cpu_result['debug_info']['valid_count']}")
    print(f"CUDA valid count: {cuda_preprocessing['valid_count']}")
    print(f"Valid counts match: {cpu_result['debug_info']['valid_count'] == cuda_preprocessing['valid_count']}")
    
    if cpu_result['debug_info']['valid_count'] > 0:
        # Now run actual CUDA clustering
        print("\n=== RUNNING ACTUAL CUDA CLUSTERING ===")
        cuda_clusters = _cluster_center_in_pixel_cuda_impl(
            candidate_means, candidate_indices, viewmat, K, width, height, depth_threshold, min_cluster_size
        )
        
        print(f"CPU clusters: {len(cpu_result['clusters'])}")
        print(f"CUDA clusters: {len(cuda_clusters)}")
        
        if len(cpu_result['clusters']) > 0:
            cpu_sizes = [len(c) for c in cpu_result['clusters']]
            print(f"CPU cluster sizes: min={min(cpu_sizes)}, max={max(cpu_sizes)}, mean={np.mean(cpu_sizes):.1f}")
            
        if len(cuda_clusters) > 0:
            cuda_sizes = [len(c) for c in cuda_clusters]
            print(f"CUDA cluster sizes: min={min(cuda_sizes)}, max={max(cuda_sizes)}, mean={np.mean(cuda_sizes):.1f}")


if __name__ == "__main__":
    compare_implementations_step_by_step(test_size=1000)
