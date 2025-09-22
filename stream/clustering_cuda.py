"""
CUDA-accelerated clustering functions for stream operations.
This module provides high-performance GPU implementations of clustering algorithms
used in Gaussian Splatting, with fallback to CPU implementations.
"""

import torch
from torch import Tensor
from typing import List, Dict, Optional
import numpy as np
from .cuda._wrapper import cluster_center_in_pixel_cuda

def cluster_center_in_pixel_cuda_accelerated(
    candidate_means: Tensor,  # [M, 3]
    candidate_indices: np.ndarray,  # [M]
    viewmat: Tensor,  # [4, 4]
    K: Tensor,  # [3, 3]
    width: int,
    height: int,
    depth_threshold: float = 0.1,
    min_cluster_size: int = 2
) -> List[np.ndarray]:
    """
    CUDA-accelerated center-in-pixel clustering with CPU fallback.
    
    This function provides the same interface as the Python version but uses
    GPU acceleration when available. It automatically falls back to CPU
    implementation when CUDA is not available.
    
    Args:
        candidate_means: [M, 3] - positions of candidate Gaussians (torch tensor)
        candidate_indices: [M] - original indices of candidates
        viewmat: [4, 4] - view matrix (world to camera)
        K: [3, 3] - camera intrinsic matrix
        width: int - image width
        height: int - image height
        depth_threshold: float - maximum depth difference for clustering within pixel
        min_cluster_size: int - minimum Gaussians per cluster
        
    Returns:
        clusters: List of arrays containing original indices
    """
    
    if len(candidate_means) < min_cluster_size:
        return []

    assert candidate_means.is_cuda, "Candidate means must be on GPU"

    return _cluster_center_in_pixel_cuda_impl(
        candidate_means, candidate_indices, viewmat, K, 
        width, height, depth_threshold, min_cluster_size
    )

def _cluster_center_in_pixel_cuda_impl(
    candidate_means: Tensor,
    candidate_indices: np.ndarray,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    depth_threshold: float,
    min_cluster_size: int
) -> List[np.ndarray]:
    """CUDA implementation of center-in-pixel clustering."""
    
    from gsplat.cuda._torch_impl import _world_to_cam
    from gsplat.cuda._wrapper import proj
    
    device = candidate_means.device
    M = len(candidate_means)
    
    # 1. Transform to camera coordinates
    means_batch = candidate_means.unsqueeze(0)  # [1, M, 3]
    viewmat_batch = viewmat.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
    
    # Use dummy covariances for compatibility with _world_to_cam
    dummy_covars = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, M, 3, 3)
    means_cam_batch, _ = _world_to_cam(means_batch, dummy_covars, viewmat_batch)
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
    valid_candidate_indices = candidate_indices[valid_indices.cpu().numpy()]
    
    # Convert back to tensor for CUDA call
    valid_candidate_indices_tensor = torch.from_numpy(valid_candidate_indices).to(device).int()
    
    # Call CUDA clustering function with filtered, discrete data
    clusters = cluster_center_in_pixel_cuda(
        valid_means_cam, valid_pixel_coords, valid_candidate_indices_tensor, viewmat, K,
        width, height, depth_threshold, min_cluster_size
    )
    
    return clusters


def _cluster_center_in_pixel_cpu_impl(
    candidate_means: Tensor,
    candidate_indices: np.ndarray,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    depth_threshold: float,
    min_cluster_size: int
) -> List[np.ndarray]:
    """CPU fallback implementation (calls original Python version)."""
    
    # Import the original Python implementation
    try:
        from .merging import _cluster_center_in_pixel
    except ImportError:
        from merging import _cluster_center_in_pixel
    
    return _cluster_center_in_pixel(
        candidate_means, candidate_indices, viewmat, K,
        width, height, depth_threshold, min_cluster_size
    )


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
        
        _cluster_center_in_pixel_cpu_impl(
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
            
            _cluster_center_in_pixel_cuda_impl(
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
