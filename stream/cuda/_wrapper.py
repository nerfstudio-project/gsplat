"""
CUDA wrapper for stream clustering operations.
"""

import torch
from torch import Tensor
from typing import Dict, List
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# Try to import compiled extension
try:
    import stream_cuda_ext
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False
    stream_cuda_ext = None

def _cluster_center_in_pixel_cuda(
    means_cam: Tensor,           # [M, 3] - candidate means in camera coordinates
    pixel_coords: Tensor,        # [M, 2] - candidate pixel coordinates  
    candidate_indices: Tensor,   # [M] - original indices of candidates
    viewmat: Tensor,             # [4, 4] - view matrix
    K: Tensor,                   # [3, 3] - camera intrinsic matrix
    width: int,
    height: int,
    depth_threshold: float = 0.1,
    min_cluster_size: int = 2
) -> List[Tensor]:
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
        clusters: List of tensors containing original indices for each cluster
    """
    start_time = time.time()
    if means_cam.shape[0] == 0:
        return []
    
    # Ensure inputs are on GPU and have correct dtypes
    means_cam = means_cam.cuda().contiguous().float()
    pixel_coords = pixel_coords.cuda().contiguous().int()  # Now expects discrete int32 coordinates
    candidate_indices = candidate_indices.cuda().contiguous().int()
    viewmat = viewmat.cuda().contiguous().float()
    K = K.cuda().contiguous().float()
    end_time = time.time()
    log.debug(f"Preproc Input time: {(end_time - start_time)*1000:.2f} ms")

    # Call CUDA extension
    start_time = time.time()
    cluster_indices, cluster_sizes, cluster_offsets, num_clusters, total_clustered = \
        stream_cuda_ext.cluster_center_in_pixel(
            means_cam, pixel_coords, candidate_indices, viewmat, K,
            width, height, depth_threshold, min_cluster_size
        )
    end_time = time.time()
    log.debug(f"Clustering CUDA wrapper time: {(end_time - start_time)*1000:.2f} ms")
    if num_clusters == 0:
        return []
    
    # PERFORMANCE TEST: Back to optimized torch.split() - measure actual cost
    start_time = time.time()
    
    if num_clusters == 0:
        end_time = time.time()
        log.debug(f"Postproc Output time: {(end_time - start_time)*1000:.2f} ms")
        return []
    
    # OPTIMIZED: Use torch.split() but with better memory management
    cluster_sizes = cluster_offsets[1:] - cluster_offsets[:-1]  # [num_clusters] on GPU
    
    # CRITICAL: Try to make torch.split() more efficient
    size_list = cluster_sizes.cpu().tolist()  # Single GPU→CPU transfer
    
    # Use torch.split - but profile exactly where the time goes
    clusters = list(torch.split(cluster_indices, size_list))
    
    end_time = time.time()
    log.debug(f"Postproc Output time: {(end_time - start_time)*1000:.2f} ms")
    log.debug(f"  - GPU→CPU transfer: {len(size_list)} values")
    log.debug(f"  - Tensor objects created: {len(clusters)}")
    return clusters

def extract_pixel_groups_step2_cuda(
    means_cam: Tensor,
    pixel_coords: Tensor, 
    candidate_indices: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    depth_threshold: float = 0.1,
    min_cluster_size: int = 2
) -> Dict:
    """
    Extract pixel groups after step 2 (grouping + sorting) for debugging.
    
    Returns the intermediate state of CUDA clustering after pixel grouping and depth sorting
    but before depth clustering.
    
    Returns:
        Dict with keys:
            - 'group_starts': [num_groups] start indices
            - 'group_sizes': [num_groups] group sizes  
            - 'sorted_depths': [num_valid] sorted depths
            - 'sorted_indices': [num_valid] sorted original indices
            - 'num_groups': number of pixel groups
            - 'num_valid': number of valid candidates
    """
    # Ensure inputs are on GPU and have correct dtypes
    means_cam = means_cam.cuda().contiguous().float()
    pixel_coords = pixel_coords.cuda().contiguous().int()
    candidate_indices = candidate_indices.cuda().contiguous().int()
    viewmat = viewmat.cuda().contiguous().float()
    K = K.cuda().contiguous().float()
    
    # Call CUDA extension
    group_starts, group_sizes, sorted_pixel_hashes, sorted_depths, sorted_indices, num_groups, num_valid = \
        stream_cuda_ext.extract_pixel_groups_step2(
            means_cam, pixel_coords, candidate_indices, viewmat, K,
            width, height, depth_threshold, min_cluster_size
        )
    
    return {
        'group_starts': group_starts,
        'group_sizes': group_sizes,
        'sorted_pixel_hashes': sorted_pixel_hashes,
        'sorted_depths': sorted_depths, 
        'sorted_indices': sorted_indices,
        'num_groups': num_groups,
        'num_valid': num_valid
    }

def extract_cluster_assignments_step7_cuda(
    means_cam: Tensor,
    pixel_coords: Tensor, 
    candidate_indices: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    depth_threshold: float = 0.1,
    min_cluster_size: int = 2
) -> Dict:
    """
    Extract cluster assignments after step 7 (depth clustering) for debugging.
    
    Returns the intermediate state of CUDA clustering after depth-based clustering
    but before result processing.
    
    Returns:
        Dict with keys:
            - 'group_starts': [num_groups] start indices
            - 'group_sizes': [num_groups] group sizes  
            - 'sorted_pixel_hashes': [num_valid] sorted pixel hashes
            - 'sorted_depths': [num_valid] sorted depths
            - 'sorted_indices': [num_valid] sorted original indices
            - 'cluster_assignments': [num_valid] cluster ID for each candidate (-1 if not clustered)
            - 'num_groups': number of pixel groups
            - 'num_valid': number of valid candidates
            - 'total_clusters': total number of clusters assigned
    """
    # Ensure inputs are on GPU and have correct dtypes
    means_cam = means_cam.cuda().contiguous().float()
    pixel_coords = pixel_coords.cuda().contiguous().int()
    candidate_indices = candidate_indices.cuda().contiguous().int()
    viewmat = viewmat.cuda().contiguous().float()
    K = K.cuda().contiguous().float()
    
    # Call CUDA extension
    group_starts, group_sizes, sorted_pixel_hashes, sorted_depths, sorted_indices, cluster_assignments, num_groups, num_valid, total_clusters = \
        stream_cuda_ext.extract_cluster_assignments_step7(
            means_cam, pixel_coords, candidate_indices, viewmat, K,
            width, height, depth_threshold, min_cluster_size
        )
    
    return {
        'group_starts': group_starts,
        'group_sizes': group_sizes,
        'sorted_pixel_hashes': sorted_pixel_hashes,
        'sorted_depths': sorted_depths, 
        'sorted_indices': sorted_indices,
        'cluster_assignments': cluster_assignments,
        'num_groups': num_groups,
        'num_valid': num_valid,
        'total_clusters': total_clusters
    }

def _is_cuda_available() -> bool:
    """Check if CUDA extension is available."""
    return _CUDA_AVAILABLE


