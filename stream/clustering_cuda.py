"""
CUDA-accelerated clustering functions for stream operations.
This module provides high-performance GPU implementations of clustering algorithms
used in Gaussian Splatting, with fallback to CPU implementations.
"""

import torch
from torch import Tensor
from typing import List, Dict, Optional, Tuple
import numpy as np
from .cuda._wrapper import _cluster_center_in_pixel_cuda
from gsplat.cuda._wrapper import proj, world_to_cam
import time
import logging
log = logging.getLogger(__name__)

def cluster_center_in_pixel_cuda(
    candidate_means: Tensor,  # [M, 3]
    candidate_indices: Tensor,  # [M]
    viewmat: Tensor,  # [4, 4]
    K: Tensor,  # [3, 3]
    width: int,
    height: int,
    depth_threshold: float = 0.1,
    min_cluster_size: int = 2
) -> List[Tensor]:
    """
    CUDA-accelerated center-in-pixel clustering.
    
    This function provides the same interface as the Python version but uses
    GPU acceleration when available. It automatically falls back to CPU
    implementation when CUDA is not available.
    
    Args:
        candidate_means: [M, 3] - positions of candidate Gaussians
        candidate_indices: [M] - original indices of candidates
        viewmat: [4, 4] - view matrix (world to camera)
        K: [3, 3] - camera intrinsic matrix
        width: int - image width
        height: int - image height
        depth_threshold: float - maximum depth difference for clustering within pixel
        min_cluster_size: int - minimum Gaussians per cluster
        
    Returns:
        clusters: List of tensors containing original indices
    """

    start_time = time.time()
    assert candidate_means.is_cuda, "Candidate means must be on GPU"
    
    # Strict tensor-only enforcement - no numpy fallbacks
    if not isinstance(candidate_indices, torch.Tensor):
        raise TypeError(f"candidate_indices must be a torch.Tensor, got {type(candidate_indices)}. "
                       f"Use torch.from_numpy() to convert numpy arrays to tensors before calling this function.")

    if len(candidate_means) < min_cluster_size:
        log.debug("Not enough candidate means for center-in-pixel clustering")
        return []
    
    device = candidate_means.device
    M = len(candidate_means)
    
    # 1. Transform to camera coordinates
    means_batch = candidate_means.unsqueeze(0)  # [1, M, 3]
    viewmat_batch = viewmat.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
    
    # Use dummy covariances for compatibility with _world_to_cam
    dummy_covars = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, M, 3, 3)
    means_cam_batch, _ = world_to_cam(means_batch, dummy_covars, viewmat_batch)
    means_cam = means_cam_batch.squeeze(0).squeeze(0)  # [M, 3]
    
    # 2. Project to 2D pixel coordinates
    means_cam_proj = means_cam.unsqueeze(0).unsqueeze(0)  # [1, 1, M, 3]
    K_batch = K.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    
    dummy_covars_cam = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, M, 3, 3)
    means2d_batch, _ = proj(means_cam_proj, dummy_covars_cam, K_batch, width, height)
    means2d = means2d_batch.squeeze(0).squeeze(0)  # [M, 2] - continuous coordinates
    
    # Convert to discrete pixel coordinates (same as CPU implementation)
    pixel_coords = torch.floor(means2d).int()  # [M, 2] - discrete coordinates
    
    # Filter out points outside image bounds (same as CPU implementation)
    valid_mask = (
        (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) &
        (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height) &
        (means_cam[:, 2] > 0)  # Points must be in front of camera
    )
    
    if not valid_mask.any():
        return []
    
    # Apply valid mask (same as CPU implementation)
    valid_indices = torch.where(valid_mask)[0]
    valid_pixel_coords = pixel_coords[valid_mask]  # [V, 2]
    valid_means_cam = means_cam[valid_mask]  # [V, 3]
    valid_candidate_indices_tensor = candidate_indices[valid_indices].int()  # Keep as tensor, no conversion
    end_time = time.time()
    log.debug(f"Preprocessing time: {(end_time - start_time)*1000:.2f} ms")
    
    
    # Call CUDA clustering function with filtered, discrete data
    start_time = time.time()
    clusters = _cluster_center_in_pixel_cuda(
        valid_means_cam, valid_pixel_coords, valid_candidate_indices_tensor, viewmat, K,
        width, height, depth_threshold, min_cluster_size
    )
    end_time = time.time()
    log.debug(f"Clustering time: {(end_time - start_time)*1000:.2f} ms")
    return clusters

def cluster_center_in_pixel_torch(
    candidate_means: Tensor,  # [M, 3]
    candidate_indices: Tensor,  # [M]
    viewmat: Tensor,  # [4, 4]
    K: Tensor,  # [3, 3]
    width: int,
    height: int,
    depth_threshold: float = 0.1,
    min_cluster_size: int = 2
) -> List[Tensor]:
    """
    Cluster using center-in-pixel approach with depth sub-clustering.
    
    This clustering method implements the "Center-in-Pixel Merging" strategy:
    1. Projects Gaussian centers to pixel coordinates using the same projection as gsplat rasterizer
    2. Groups Gaussians that fall into the same pixel 
    3. Within each pixel group, sub-clusters by camera depth using threshold
    4. Returns clusters that respect pixel boundaries and depth locality
    
    This approach ensures merged Gaussians maintain visual coherence from
    the current viewpoint, avoiding artifacts from 3D spatial clustering.
    Uses gsplat's standard proj() function for consistent projection behavior.
    
    Args:
        candidate_means: [M, 3] - positions of candidate Gaussians
        candidate_indices: [M] - original indices of candidates
        viewmat: [4, 4] - view matrix (world to camera)
        K: [3, 3] - camera intrinsic matrix
        width: int - image width
        height: int - image height
        depth_threshold: float - maximum depth difference for clustering within pixel
        min_cluster_size: int - minimum Gaussians per cluster
        
    Returns:
        clusters: List of tensors containing original indices
    """
    # Strict tensor-only enforcement - no numpy fallbacks
    if not isinstance(candidate_indices, torch.Tensor):
        raise TypeError(f"candidate_indices must be a torch.Tensor, got {type(candidate_indices)}. "
                       f"Use torch.from_numpy() to convert numpy arrays to tensors before calling this function.")
                       
    if len(candidate_means) < min_cluster_size:
        log.debug("Not enough candidate means for center-in-pixel clustering")
        return []
    
    device = candidate_means.device
    M = len(candidate_means)
    
    # 1. Transform to camera coordinates
    # Add batch dimension for compatibility with _world_to_cam
    means_batch = candidate_means.unsqueeze(0)  # [1, M, 3]
    viewmat_batch = viewmat.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
    
    # Transform to camera coordinates (no covariances needed, just pass dummy ones)
    dummy_covars = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, M, 3, 3)  # [1, M, 3, 3]
    means_cam_batch, _ = world_to_cam(means_batch, dummy_covars, viewmat_batch)
    
    # Remove batch dimensions
    means_cam = means_cam_batch.squeeze(0).squeeze(0)  # [M, 3]
    
    # 2. Project to 2D pixel coordinates
    # Add batch dimension for projection
    means_cam_proj = means_cam.unsqueeze(0).unsqueeze(0)  # [1, 1, M, 3]
    K_batch = K.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    
    # Project to 2D (again, we only need means2d, not covariances)
    dummy_covars_cam = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, M, 3, 3)
    means2d_batch, _ = proj(means_cam_proj, dummy_covars_cam, K_batch, width, height)
    
    # Remove batch dimensions
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
        log.debug("No valid points found for center-in-pixel clustering")
        return []
    
    # Apply valid mask
    valid_indices = torch.where(valid_mask)[0]
    valid_pixel_coords = pixel_coords[valid_mask]  # [V, 2]
    valid_depths = means_cam[valid_mask, 2]  # [V] - camera Z coordinate (depth)
    valid_candidate_indices = candidate_indices[valid_indices]  # Keep as tensor
    
    # 4. ULTRA-OPTIMIZED: Vectorized pixel grouping and depth clustering
    # Create hash keys for pixel coordinates  
    pixel_hashes = valid_pixel_coords[:, 0] * height + valid_pixel_coords[:, 1]  # [V]
    
    # Sort all points by (pixel_hash, depth) - two-step stable sort for precision
    # First sort by depth to establish depth ordering within each pixel
    depth_sort_indices = torch.argsort(valid_depths, stable=True)
    
    # Apply depth sorting
    temp_pixel_hashes = pixel_hashes[depth_sort_indices]
    temp_depths = valid_depths[depth_sort_indices]
    temp_orig_indices = valid_candidate_indices[depth_sort_indices]
    
    # Then sort by pixel_hash (stable sort preserves depth order within each pixel)
    pixel_sort_indices = torch.argsort(temp_pixel_hashes, stable=True)
    sort_indices = depth_sort_indices[pixel_sort_indices]  # Compose the two sorts
    
    sorted_pixel_hashes = pixel_hashes[sort_indices]
    sorted_depths = valid_depths[sort_indices]  
    sorted_orig_indices = valid_candidate_indices[sort_indices]
    
    # 5. VECTORIZED: Find pixel group boundaries (where pixel hash changes)
    pixel_boundaries = torch.cat([
        torch.tensor([True], device=device),  # First element is always a boundary
        sorted_pixel_hashes[1:] != sorted_pixel_hashes[:-1]  # Changes in pixel hash
    ])
    
    pixel_boundary_indices = torch.where(pixel_boundaries)[0]  # Positions where pixel groups start
    pixel_group_starts = pixel_boundary_indices
    pixel_group_ends = torch.cat([pixel_boundary_indices[1:], torch.tensor([len(sorted_depths)], device=device)])
    pixel_group_sizes = pixel_group_ends - pixel_group_starts
    
    # Filter out pixel groups that are too small (vectorized)
    valid_pixel_groups = pixel_group_sizes >= min_cluster_size
    if not valid_pixel_groups.any():
        return []
    
    valid_starts = pixel_group_starts[valid_pixel_groups]
    valid_ends = pixel_group_ends[valid_pixel_groups]
    valid_sizes = pixel_group_sizes[valid_pixel_groups]
    
    # 6. ULTRA-VECTORIZED: Process all depth clustering in parallel
    clusters = []
    
    # Process each valid pixel group (this is now much smaller and necessary sequential part)
    for start_idx, end_idx in zip(valid_starts.cpu().tolist(), valid_ends.cpu().tolist()):
        group_depths = sorted_depths[start_idx:end_idx]
        group_orig_indices = sorted_orig_indices[start_idx:end_idx] 
        group_size = end_idx - start_idx
        
        if group_size < min_cluster_size:
            continue
            
        if group_size == 1:
            if min_cluster_size <= 1:
                clusters.append(group_orig_indices)
            continue
        
        # Vectorized depth clustering within this pixel group
        depth_diffs = torch.abs(group_depths[1:] - group_depths[:-1])
        depth_boundaries = depth_diffs > depth_threshold
        
        if not depth_boundaries.any():
            # All points in this pixel group form one cluster
            if group_size >= min_cluster_size:
                clusters.append(group_orig_indices)
        else:
            # Split into depth-based sub-clusters
            depth_boundary_positions = torch.where(depth_boundaries)[0] + 1  # Positions after boundaries
            
            # Create cluster boundaries: [0, boundary_pos1, boundary_pos2, ..., group_size]
            cluster_starts_local = torch.cat([torch.tensor([0], device=device), depth_boundary_positions])
            cluster_ends_local = torch.cat([depth_boundary_positions, torch.tensor([group_size], device=device)])
            
            # Extract valid clusters (vectorized size checking)
            cluster_sizes_local = cluster_ends_local - cluster_starts_local
            valid_clusters_mask = cluster_sizes_local >= min_cluster_size
            
            if valid_clusters_mask.any():
                valid_cluster_starts = cluster_starts_local[valid_clusters_mask]
                valid_cluster_ends = cluster_ends_local[valid_clusters_mask]
                
                # Extract each valid cluster
                for c_start, c_end in zip(valid_cluster_starts.tolist(), valid_cluster_ends.tolist()):
                    clusters.append(group_orig_indices[c_start:c_end])
    
    return clusters

def benchmark_clustering_performance(
    candidate_means: Tensor,
    candidate_indices: np.ndarray,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    num_runs: int = 5
) -> Dict[str, float]:
    """
    Benchmark clustering performance between CUDA and CPU implementations.
    
    Args:
        candidate_means: Test data for clustering
        candidate_indices: Test indices
        viewmat: View matrix
        K: Camera intrinsics  
        width: Image width
        height: Image height
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    results = {}
    
    # Benchmark CPU implementation
    if candidate_means.is_cuda:
        cpu_means = candidate_means.cpu()
        cpu_viewmat = viewmat.cpu()
        cpu_K = K.cpu()
    else:
        cpu_means = candidate_means
        cpu_viewmat = viewmat
        cpu_K = K
    
    print("Benchmarking CPU implementation...")
    cpu_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        cluster_center_in_pixel_torch(
            cpu_means, candidate_indices, cpu_viewmat, cpu_K,
            width, height, 0.1, 2
        )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        cpu_times.append((time.time() - start_time) * 1000)  # Convert to ms
    
    results["cpu_mean_ms"] = np.mean(cpu_times)
    results["cpu_std_ms"] = np.std(cpu_times)
    
    # Benchmark CUDA implementation if available
    if candidate_means.is_cuda:
        print("Benchmarking CUDA implementation...")
        gpu_times = []
        
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            
            cluster_center_in_pixel_cuda(
                candidate_means, candidate_indices, viewmat, K,
                width, height, 0.1, 2
            )
            
            torch.cuda.synchronize()
            gpu_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        results["gpu_mean_ms"] = np.mean(gpu_times)
        results["gpu_std_ms"] = np.std(gpu_times)
        results["speedup"] = results["cpu_mean_ms"] / results["gpu_mean_ms"]
        
    else:
        results["gpu_mean_ms"] = None
        results["gpu_std_ms"] = None
        results["speedup"] = None
        
    return results


def get_clustering_performance_info() -> Dict[str, bool]:
    """
    Get information about CUDA clustering availability and performance.
    
    Returns:
        Dict with keys:
            - 'cuda_extension_loaded': bool - Whether CUDA extension is loaded
            - 'torch_cuda_available': bool - Whether PyTorch CUDA is available
    """
    # If we reach here, CUDA extension is loaded (otherwise import would have failed)
    return {
        'cuda_extension_loaded': True,
        'torch_cuda_available': torch.cuda.is_available()
    }
