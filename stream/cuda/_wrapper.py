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
) -> Dict[str, Tensor]:
    """
    Perform center-in-pixel clustering using CUDA acceleration.
    
    This function implements the same algorithm as the Python version but uses
    GPU parallelization for improved performance on large datasets.
    
    Returns flat tensor format optimized for CUDA merging pipeline.
    
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
        Dict containing:
            'cluster_indices': [total_clustered] - Flat array of original indices
            'cluster_offsets': [num_clusters + 1] - Cluster boundary offsets  
            'num_clusters': int - Total number of clusters found
            'total_clustered': int - Total number of clustered Gaussians
    """
    start_time = time.time()
    if means_cam.shape[0] == 0:
        return {
            'cluster_indices': torch.empty(0, dtype=torch.int32, device=means_cam.device),
            'cluster_offsets': torch.zeros(1, dtype=torch.int32, device=means_cam.device),
            'num_clusters': 0,
            'total_clustered': 0
        }
    
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
    
    # ULTRA-FAST: Return native CUDA format - no postprocessing needed!
    start_time = time.time()
    
    result = {
        'cluster_indices': cluster_indices,     # [total_clustered] - already on GPU
        'cluster_offsets': cluster_offsets,     # [num_clusters + 1] - already on GPU  
        'num_clusters': num_clusters,           # int
        'total_clustered': total_clustered      # int
    }
    
    end_time = time.time()
    log.debug(f"Postproc Output time: {(end_time - start_time)*1000:.2f} ms")
    log.debug(f"  - No GPU→CPU transfers!")
    log.debug(f"  - No tensor object creation!")
    log.debug(f"  - Native CUDA format returned")
    
    return result

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

def _merge_clusters_cuda(
    cluster_indices: Tensor,        # [total_clustered] - flat array of original indices
    cluster_offsets: Tensor,        # [num_clusters + 1] - cluster boundaries  
    means: Tensor,                  # [N, 3] - all Gaussian centers
    quats: Tensor,                  # [N, 4] - all Gaussian quaternions
    scales: Tensor,                 # [N, 3] - all Gaussian scales
    opacities: Tensor,              # [N] - all Gaussian opacities
    colors: Tensor,                 # [N, color_dim] - all Gaussian colors
    strategy: str = "weighted_mean", # "weighted_mean" or "moment_matching"
    weight_by_opacity: bool = True,  # For weighted_mean strategy
    preserve_volume: bool = True     # For moment_matching strategy
) -> Dict[str, Tensor]:
    """
    CUDA accelerated cluster merging.
    
    Args:
        cluster_indices: [total_clustered] - flat array of original indices from clustering
        cluster_offsets: [num_clusters + 1] - cluster boundaries (from clustering)
        means: [N, 3] - all Gaussian centers
        quats: [N, 4] - all Gaussian quaternions  
        scales: [N, 3] - all Gaussian scales (linear space)
        opacities: [N] - all Gaussian opacities (linear space)
        colors: [N, color_dim] - all Gaussian colors
        strategy: merging strategy ("weighted_mean" or "moment_matching")
        weight_by_opacity: whether to weight by opacity (weighted_mean only)
        preserve_volume: whether to preserve volume (moment_matching only)
        
    Returns:
        Dict containing merged Gaussian parameters:
            'means': [num_clusters, 3] - merged centers
            'quats': [num_clusters, 4] - merged quaternions
            'scales': [num_clusters, 3] - merged scales 
            'opacities': [num_clusters] - merged opacities
            'colors': [num_clusters, color_dim] - merged colors
    """
    start_time = time.time()
    
    if cluster_indices.shape[0] == 0:
        # Return empty results
        device = means.device
        color_dim = colors.shape[1]
        return {
            'means': torch.empty(0, 3, device=device, dtype=torch.float32),
            'quats': torch.empty(0, 4, device=device, dtype=torch.float32),
            'scales': torch.empty(0, 3, device=device, dtype=torch.float32),
            'opacities': torch.empty(0, device=device, dtype=torch.float32),
            'colors': torch.empty(0, color_dim, device=device, dtype=torch.float32)
        }
    
    # Ensure inputs are on GPU and have correct dtypes
    cluster_indices = cluster_indices.cuda().contiguous().int()
    cluster_offsets = cluster_offsets.cuda().contiguous().int()
    means = means.cuda().contiguous().float()
    quats = quats.cuda().contiguous().float()
    scales = scales.cuda().contiguous().float()
    opacities = opacities.cuda().contiguous().float()
    colors = colors.cuda().contiguous().float()
    
    end_time = time.time()
    log.debug(f"Merging CUDA preproc time: {(end_time - start_time)*1000:.2f} ms")
    
    # Call CUDA extension
    start_time = time.time()
    merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = \
        stream_cuda_ext.merge_clusters_cuda(
            cluster_indices, cluster_offsets, means, quats, scales, opacities, colors,
            strategy, weight_by_opacity, preserve_volume
        )
    end_time = time.time()
    log.debug(f"Merging CUDA kernel time: {(end_time - start_time)*1000:.2f} ms")
    
    return {
        'means': merged_means,
        'quats': merged_quats, 
        'scales': merged_scales,
        'opacities': merged_opacities,
        'colors': merged_colors
    }

def _is_cuda_available() -> bool:
    """Check if CUDA extension is available."""
    return _CUDA_AVAILABLE


