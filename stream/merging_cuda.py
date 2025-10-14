"""
CUDA-accelerated merging functions for stream operations.
This module provides high-performance GPU implementations of merging algorithms
used in Gaussian Splatting.
"""

import torch
from torch import Tensor
from typing import Tuple, Dict
import time
from .cuda._wrapper import _merge_clusters_cuda
import logging

log = logging.getLogger(__name__)

def merge_clusters_cuda(
    cluster_indices: Tensor,        # [total_clustered] - flat array of original indices
    cluster_offsets: Tensor,        # [num_clusters + 1] - cluster boundaries
    current_means: Tensor,          # [N, 3] - all Gaussian centers
    current_quats: Tensor,          # [N, 4] - all Gaussian quaternions
    current_scales: Tensor,         # [N, 3] - all Gaussian scales
    current_opacities: Tensor,      # [N] - all Gaussian opacities
    current_colors: Tensor,         # [N, color_dim] - all Gaussian colors
    merge_strategy: str = "weighted_mean",
    **merge_kwargs
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    CUDA-accelerated cluster merging with concatenation.
    
    This function provides the same interface as merge_clusters_torch() but uses
    GPU acceleration. It merges clusters and concatenates with unmerged Gaussians.
    
    Args:
        cluster_indices: [total_clustered] - flat array of original indices from clustering
        cluster_offsets: [num_clusters + 1] - cluster boundaries (from clustering)
        current_means: [N, 3] - all Gaussian centers
        current_quats: [N, 4] - all Gaussian quaternions
        current_scales: [N, 3] - all Gaussian scales (linear space)
        current_opacities: [N] - all Gaussian opacities (linear space)
        current_colors: [N, color_dim] - all Gaussian colors
        merge_strategy: merging strategy ("weighted_mean" or "moment_matching")
        **merge_kwargs: additional arguments for merging strategies
        
    Returns:
        Tuple containing final Gaussian parameters:
            merged_means: [M, 3] - final Gaussian centers
            merged_quats: [M, 4] - final Gaussian quaternions
            merged_scales: [M, 3] - final Gaussian scales in linear space
            merged_opacities: [M] - final Gaussian opacities in linear space
            merged_colors: [M, color_dim] - final Gaussian colors
    """
    start_time = time.time()
    device = current_means.device
    
    # Handle empty input
    if cluster_indices.shape[0] == 0:
        log.info("No clusters to merge")
        return current_means, current_quats, current_scales, current_opacities, current_colors
    
    log.debug(f"Merging {cluster_offsets.shape[0] - 1} clusters with {cluster_indices.shape[0]} total Gaussians")
    
    # Handle colors: reshape from [N, K, 3] to [N, K*3] for CUDA compatibility if needed
    colors_was_3d = current_colors.dim() == 3
    if colors_was_3d:
        original_color_shape = current_colors.shape  # Store original shape for later
        current_colors_2d = current_colors.view(current_colors.shape[0], -1)  # [N, K*3]
        log.debug(f"Reshaped colors from {original_color_shape} to {current_colors_2d.shape} for CUDA compatibility")
    else:
        current_colors_2d = current_colors
    
    # Step 1: Merge clusters using CUDA (use reshaped colors)
    merge_start = time.time()
    merged_result = _merge_clusters_cuda(
        cluster_indices=cluster_indices,
        cluster_offsets=cluster_offsets,
        means=current_means,
        quats=current_quats,
        scales=current_scales,
        opacities=current_opacities,
        colors=current_colors_2d,  # Use reshaped 2D colors
        strategy=merge_strategy,
        **merge_kwargs
    )
    merge_time = (time.time() - merge_start) * 1000
    log.debug(f"CUDA merging time: {merge_time:.2f} ms")
    
    # Step 2: Concatenate merged Gaussians with unmerged ones
    concat_start = time.time()
    
    # Create mask for Gaussians that were NOT merged
    unmerged_mask = torch.ones(current_means.shape[0], dtype=torch.bool, device=device)
    unmerged_mask[cluster_indices] = False
    
    # Combine merged + unmerged Gaussians (use reshaped colors for concatenation)  
    final_means = torch.cat([current_means[unmerged_mask], merged_result['means']], dim=0)
    final_quats = torch.cat([current_quats[unmerged_mask], merged_result['quats']], dim=0)
    final_scales = torch.cat([current_scales[unmerged_mask], merged_result['scales']], dim=0)
    final_opacities = torch.cat([current_opacities[unmerged_mask], merged_result['opacities']], dim=0)
    final_colors = torch.cat([current_colors_2d[unmerged_mask], merged_result['colors']], dim=0)
    
    # Step 3: Reshape colors back to original format if input was 3D
    if colors_was_3d:
        # Reshape from [M, K*3] back to [M, K, 3]
        original_k = original_color_shape[1]
        original_feature_dim = original_color_shape[2]
        final_colors = final_colors.view(final_colors.shape[0], original_k, original_feature_dim)
        log.debug(f"Reshaped final colors back to {final_colors.shape} to match input format")
    
    concat_time = (time.time() - concat_start) * 1000
    log.debug(f"Concatenation time: {concat_time:.2f} ms")
    
    total_time = (time.time() - start_time) * 1000
    log.debug(f"Total merge_clusters_cuda time: {total_time:.2f} ms")
    
    num_unmerged = unmerged_mask.sum().item()
    num_merged_clusters = merged_result['means'].shape[0]
    log.debug(f"Result: {num_unmerged} unmerged + {num_merged_clusters} merged = {final_means.shape[0]} total Gaussians")
    
    return final_means, final_quats, final_scales, final_opacities, final_colors
