"""
CUDA wrapper for stream clustering operations.
"""

import torch
from torch import Tensor
from typing import Tuple, List, Optional
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Try to import compiled extension
try:
    import stream_cuda_ext
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False
    stream_cuda_ext = None

def cluster_center_in_pixel_cuda(
    means_cam: Tensor,           # [M, 3] - candidate means in camera coordinates
    pixel_coords: Tensor,        # [M, 2] - candidate pixel coordinates  
    candidate_indices: Tensor,   # [M] - original indices of candidates
    viewmat: Tensor,             # [4, 4] - view matrix
    K: Tensor,                   # [3, 3] - camera intrinsic matrix
    width: int,
    height: int,
    depth_threshold: float = 0.1,
    min_cluster_size: int = 2
) -> List[np.ndarray]:
    """
    Perform center-in-pixel clustering using CUDA acceleration.
    
    This function implements the same algorithm as the Python version but uses
    GPU parallelization for improved performance on large datasets.
    
    Args:
        means_cam: [M, 3] - Gaussian centers in camera coordinates
        pixel_coords: [M, 2] - Projected pixel coordinates (discrete/integer)
        candidate_indices: [M] - Original indices of candidate Gaussians
        viewmat: [4, 4] - View matrix (world to camera)
        K: [3, 3] - Camera intrinsic matrix
        width: int - Image width in pixels
        height: int - Image height in pixels
        depth_threshold: float - Maximum depth difference for clustering within pixel
        min_cluster_size: int - Minimum Gaussians per cluster
        
    Returns:
        clusters: List of numpy arrays containing original indices for each cluster
    """
    if means_cam.shape[0] == 0:
        return []
    
    # Ensure inputs are on GPU and have correct dtypes
    means_cam = means_cam.cuda().contiguous().float()
    pixel_coords = pixel_coords.cuda().contiguous().int()  # Now expects discrete int32 coordinates
    candidate_indices = candidate_indices.cuda().contiguous().int()
    viewmat = viewmat.cuda().contiguous().float()
    K = K.cuda().contiguous().float()
    
    # Call CUDA extension
    cluster_indices, cluster_sizes, cluster_offsets, num_clusters, total_clustered = \
        stream_cuda_ext.cluster_center_in_pixel(
            means_cam, pixel_coords, candidate_indices, viewmat, K,
            width, height, depth_threshold, min_cluster_size
        )
    
    if num_clusters == 0:
        return []
    
    # Convert to list of numpy arrays (same format as Python version)
    clusters = []
    for i in range(num_clusters):
        start_idx = cluster_offsets[i].item()
        end_idx = cluster_offsets[i + 1].item()
        
        if end_idx > start_idx:  # Ensure cluster is not empty
            cluster_array = cluster_indices[start_idx:end_idx].cpu().numpy()
            clusters.append(cluster_array)
    
    return clusters

def is_cuda_available() -> bool:
    """Check if CUDA extension is available."""
    return _CUDA_AVAILABLE


