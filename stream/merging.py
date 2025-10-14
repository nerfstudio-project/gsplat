import torch
from torch import Tensor
from typing import Tuple, List, Dict, Optional, Callable
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import time

from .culling import calc_pixel_size, calc_pixel_area
from .clustering_cuda import cluster_center_in_pixel_cuda, cluster_center_in_pixel_torch
from .merging_cuda import merge_clusters_cuda
import logging
log = logging.getLogger(__name__)

def _cluster_knn(
    candidate_means: np.ndarray,  # [M, 3]
    candidate_indices: np.ndarray,  # [M]
    k_neighbors: int = 3,
    max_distance: float = 0.1,
    min_cluster_size: int = 1
) -> List[np.ndarray]:
    """
    Cluster using K-nearest neighbors approach.
    
    Args:
        candidate_means: [M, 3] - positions of candidate Gaussians
        candidate_indices: [M] - original indices of candidates
        k_neighbors: int - number of neighbors to consider
        max_distance: float - maximum distance for clustering
        min_cluster_size: int - minimum Gaussians per cluster
        
    Returns:
        clusters: List of arrays containing original indices
    """
    if len(candidate_means) < min_cluster_size:
        return []
    
    # Build KNN graph
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(candidate_means)))
    nbrs.fit(candidate_means)
    distances, indices = nbrs.kneighbors(candidate_means)
    
    # Create adjacency list based on distance threshold
    adjacency = {}
    for i in range(len(candidate_means)):
        adjacency[i] = []
        for j in range(1, len(indices[i])):  # Skip self (index 0)
            if distances[i][j] <= max_distance:
                adjacency[i].append(indices[i][j])
    
    # Find connected components (clusters)
    visited = set()
    clusters = []
    
    def dfs(node, current_cluster):
        if node in visited:
            return
        visited.add(node)
        current_cluster.append(node)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                dfs(neighbor, current_cluster)
    
    for i in range(len(candidate_means)):
        if i not in visited:
            cluster = []
            dfs(i, cluster)
            if len(cluster) >= min_cluster_size:
                # Convert back to original indices
                original_cluster = candidate_indices[cluster]
                clusters.append(original_cluster)
    
    return clusters

def _cluster_dbscan(
    candidate_means: np.ndarray,  # [M, 3]
    candidate_indices: np.ndarray,  # [M]
    eps: float = 0.1,
    min_samples: int = 2
) -> List[np.ndarray]:
    """
    Cluster using DBSCAN algorithm.
    
    Args:
        candidate_means: [M, 3] - positions of candidate Gaussians
        candidate_indices: [M] - original indices of candidates
        eps: float - maximum distance between samples in a neighborhood
        min_samples: int - minimum samples per cluster
        
    Returns:
        clusters: List of arrays containing original indices
    """
    if len(candidate_means) < min_samples:
        return []
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clustering.fit_predict(candidate_means)
    
    clusters = []
    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)  # Remove noise label
    
    for label in unique_labels:
        cluster_mask = cluster_labels == label
        cluster_indices = candidate_indices[cluster_mask]
        if len(cluster_indices) >= min_samples:
            clusters.append(cluster_indices)
    
    return clusters

def _cluster_distance_based(
    candidate_means: np.ndarray,  # [M, 3]
    candidate_indices: np.ndarray,  # [M]
    max_distance: float = 0.1,
    min_cluster_size: int = 2
) -> List[np.ndarray]:
    """
    Simple distance-based clustering.
    
    Args:
        candidate_means: [M, 3] - positions of candidate Gaussians
        candidate_indices: [M] - original indices of candidates
        max_distance: float - maximum distance for clustering
        min_cluster_size: int - minimum Gaussians per cluster
        
    Returns:
        clusters: List of arrays containing original indices
    """
    if len(candidate_means) < min_cluster_size:
        return []
    
    clusters = []
    used = set()
    
    for i in range(len(candidate_means)):
        if i in used:
            continue
            
        cluster = [i]
        used.add(i)
        
        # Find all points within distance
        for j in range(i + 1, len(candidate_means)):
            if j in used:
                continue
            
            distance = np.linalg.norm(candidate_means[i] - candidate_means[j])
            if distance <= max_distance:
                cluster.append(j)
                used.add(j)
        
        if len(cluster) >= min_cluster_size:
            original_cluster = candidate_indices[cluster]
            clusters.append(original_cluster)
    
    return clusters

def _merge_weighted_mean(
    cluster_means: Tensor,  # [C, 3]
    cluster_quats: Tensor,  # [C, 4]
    cluster_scales: Tensor,  # [C, 3] - in linear space
    cluster_opacities: Tensor,  # [C] - in linear space
    cluster_colors: Tensor,  # [C, K, 3] or [C, 3]
    weight_by_opacity: bool = True
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Merge cluster using weighted mean approach.
    
    Args:
        cluster_*: Cluster parameters (scales and opacities in linear space)
        weight_by_opacity: bool - whether to weight by opacity
        
    Returns:
        Merged parameters (scales and opacities in linear space)
    """
    # Determine weights (using linear opacities)
    if weight_by_opacity:
        weights = cluster_opacities
    else:
        weights = torch.ones_like(cluster_opacities)
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Merge mean (weighted average)
    merged_mean = (weights.unsqueeze(-1) * cluster_means).sum(dim=0)
    
    # Merge quaternions (weighted average with normalization)
    # Note: This is a simple approach; SLERP might be better for quaternions
    merged_quat = (weights.unsqueeze(-1) * cluster_quats).sum(dim=0)
    merged_quat = merged_quat / torch.norm(merged_quat)  # Normalize quaternion
    
    # Merge scales (weighted average in linear space)
    merged_scales = (weights.unsqueeze(-1) * cluster_scales).sum(dim=0)
    
    # Merge opacity (sum in linear space, clamped to [0, 1])
    merged_opacities = torch.clamp(cluster_opacities.sum(), 0.0, 1.0)
    
    # Merge colors (weighted average)
    if cluster_colors.dim() == 3:  # SH colors [C, K, 3]
        merged_color = (weights.unsqueeze(-1).unsqueeze(-1) * cluster_colors).sum(dim=0)
    else:  # RGB colors [C, 3]
        merged_color = (weights.unsqueeze(-1) * cluster_colors).sum(dim=0)
    
    return merged_mean, merged_quat, merged_scales, merged_opacities, merged_color

def _merge_moment_matching(
    cluster_means: Tensor,  # [C, 3]
    cluster_quats: Tensor,  # [C, 4]
    cluster_scales: Tensor,  # [C, 3] - in linear space
    cluster_opacities: Tensor,  # [C] - in linear space
    cluster_colors: Tensor,  # [C, K, 3] or [C, 3]
    preserve_volume: bool = True
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Merge cluster using moment matching approach (preserves statistical properties).
    
    Args:
        cluster_*: Cluster parameters (scales and opacities in linear space)
        preserve_volume: bool - whether to preserve total volume
        
    Returns:
        Merged parameters (scales and opacities in linear space)
    """
    # Use linear opacity as weights
    weights = cluster_opacities / cluster_opacities.sum()
    
    # First moment (mean)
    merged_mean = (weights.unsqueeze(-1) * cluster_means).sum(dim=0)
    
    # For scales and quaternions, we need to be more careful
    # Convert quaternions to rotation matrices for averaging
    # This is a simplified approach - could be improved with proper quaternion averaging
    merged_quat = (weights.unsqueeze(-1) * cluster_quats).sum(dim=0)
    merged_quat = merged_quat / torch.norm(merged_quat)
    
    # For scales, we can either average or preserve total volume (in linear space)
    if preserve_volume:
        # Calculate total volume and redistribute (using linear scales)
        volumes = cluster_scales.prod(dim=1)  # Volume of each Gaussian
        total_volume = (cluster_opacities * volumes).sum()
        # Create scales that preserve total volume in a spherical Gaussian
        avg_radius = (total_volume / (4/3 * math.pi)) ** (1/3)
        merged_scales = torch.full_like(cluster_scales[0], avg_radius)
    else:
        merged_scales = (weights.unsqueeze(-1) * cluster_scales).sum(dim=0)
    
    # Merge opacity (sum, clamped to [0, 1]) in linear space
    merged_opacities = torch.clamp(cluster_opacities.sum(), 0.0, 1.0)
    
    # Merge colors (weighted average)
    if cluster_colors.dim() == 3:  # SH colors [C, K, 3]
        merged_color = (weights.unsqueeze(-1).unsqueeze(-1) * cluster_colors).sum(dim=0)
    else:  # RGB colors [C, 3]
        merged_color = (weights.unsqueeze(-1) * cluster_colors).sum(dim=0)
    
    return merged_mean, merged_quat, merged_scales, merged_opacities, merged_color

def find_merge_candidates(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    viewmat: Tensor,  # [4, 4]
    K: Tensor,  # [3, 3]
    width: int,
    height: int,
    pixel_size_threshold: float = 2.0,
    use_pixel_area: bool = False,
    pixel_area_threshold: float = 2.0,
    scale_modifier: float = 1.0,
    eps2d: float = 0.3,
    method: str = "cuda"
) -> Tensor:
    """
    Find Gaussians that are candidates for merging based on their projected size.
    
    Args:
        means: [N, 3] - Gaussian centers in world coordinates
        quats: [N, 4] - Gaussian quaternions
        scales: [N, 3] - Gaussian scales
        opacities: [N] - Gaussian opacities
        viewmat: [4, 4] - view matrix (world to camera)
        K: [3, 3] - camera intrinsic matrix
        width: int - image width
        height: int - image height
        pixel_size_threshold: float - minimum pixel size for merging
        use_pixel_area: bool - whether to use pixel area instead of pixel size
        pixel_area_threshold: float - minimum pixel area for merging
        scale_modifier: float - scale modifier applied to Gaussian scales
        eps2d: float - low-pass filter value
        method: str - method to use for calculation ("cuda" or "torch")
        
    Returns:
        merge_candidates: [N] - boolean mask for candidates
    """
    if use_pixel_area:
        pixel_metrics = calc_pixel_area(
            means, quats, scales, opacities, viewmat, K, width, height,
            scale_modifier, eps2d, method
        )
        merge_candidates = pixel_metrics < pixel_area_threshold
    else:
        pixel_metrics = calc_pixel_size(
            means, quats, scales, opacities, viewmat, K, width, height,
            scale_modifier, eps2d, method
        )
        merge_candidates = pixel_metrics < pixel_size_threshold
    return merge_candidates


def cluster_gaussians(
    means: Tensor,  # [N, 3]
    candidate_mask: Tensor,  # [N] boolean
    clustering_method: str = "center_in_pixel",
    method: str = "cuda",
    **clustering_kwargs
) -> List[np.ndarray]:
    """
    Cluster candidate Gaussians for merging using various methods.
    
    Args:
        means: [N, 3] - Gaussian centers in world coordinates
        candidate_mask: [N] - boolean mask for merge candidates
        clustering_method: str - clustering method ("knn", "dbscan", "distance_based", "center_in_pixel")
        method: str - method to use for clustering ("cuda" or "torch")
        **clustering_kwargs: additional arguments for clustering methods
        
    Returns:
        clusters: List of arrays containing indices of Gaussians in each cluster
    """
    if not candidate_mask.any():
        return []
    
    # Keep candidate_indices as torch tensor for faster processing
    candidate_indices = torch.where(candidate_mask)[0]  # Keep as tensor on GPU
    candidate_means = means[candidate_mask]  # [M, 3] - keep as tensor for center_in_pixel
    
    if clustering_method == "knn":
        # Convert to numpy only for numpy-based methods
        candidate_indices_np = candidate_indices.cpu().numpy()
        return _cluster_knn(candidate_means.cpu().numpy(), candidate_indices_np, **clustering_kwargs)
    elif clustering_method == "dbscan":
        # Convert to numpy only for numpy-based methods
        candidate_indices_np = candidate_indices.cpu().numpy()
        return _cluster_dbscan(candidate_means.cpu().numpy(), candidate_indices_np, **clustering_kwargs)
    elif clustering_method == "distance_based":
        # Convert to numpy only for numpy-based methods
        candidate_indices_np = candidate_indices.cpu().numpy()
        return _cluster_distance_based(candidate_means.cpu().numpy(), candidate_indices_np, **clustering_kwargs)
    elif clustering_method == "center_in_pixel":
        if method == "cuda":
            return cluster_center_in_pixel_cuda(candidate_means, candidate_indices, **clustering_kwargs)
        else:
            return cluster_center_in_pixel_torch(candidate_means, candidate_indices, **clustering_kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")

def _merge_cluster_torch(
    cluster_indices: Tensor,  # [C] - indices of Gaussians in cluster
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3] - in linear space
    opacities: Tensor,  # [N] - in linear space
    colors: Tensor,  # [N, K, 3] or [N, 3]
    merge_strategy: str = "weighted_mean",
    **merge_kwargs
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Merge a cluster of Gaussians into a single Gaussian.
    
    Args:
        cluster_indices: [C] - indices of Gaussians to merge
        means: [N, 3] - all Gaussian centers
        quats: [N, 4] - all Gaussian quaternions
        scales: [N, 3] - all Gaussian scales in linear space
        opacities: [N] - all Gaussian opacities in linear space
        colors: [N, K, 3] or [N, 3] - all Gaussian colors
        merge_strategy: str - merging strategy ("weighted_mean", "moment_matching")
        **merge_kwargs: additional arguments for merging strategies
        
    Returns:
        merged_mean: [3] - merged Gaussian center
        merged_quat: [4] - merged Gaussian quaternion
        merged_scale: [3] - merged Gaussian scale in linear space
        merged_opacity: [] - merged Gaussian opacity in linear space
        merged_color: [K, 3] or [3] - merged Gaussian color
    """
    # Strict tensor-only enforcement - no numpy fallbacks
    if not isinstance(cluster_indices, torch.Tensor):
        raise TypeError(f"cluster_indices must be a torch.Tensor, got {type(cluster_indices)}. "
                       f"Use torch.from_numpy() to convert numpy arrays to tensors before calling this function.")
    
    cluster_means = means[cluster_indices]  # [C, 3]
    cluster_quats = quats[cluster_indices]  # [C, 4]
    cluster_scales = scales[cluster_indices]  # [C, 3]
    cluster_opacities = opacities[cluster_indices]  # [C]
    cluster_colors = colors[cluster_indices]  # [C, K, 3] or [C, 3]
    
    if merge_strategy == "weighted_mean":
        return _merge_weighted_mean(
            cluster_means, cluster_quats, cluster_scales, 
            cluster_opacities, cluster_colors, **merge_kwargs
        )
    elif merge_strategy == "moment_matching":
        return _merge_moment_matching(
            cluster_means, cluster_quats, cluster_scales,
            cluster_opacities, cluster_colors, **merge_kwargs
        )
    else:
        raise ValueError(f"Unknown merge strategy: {merge_strategy}")

def merge_clusters_torch(
    clusters: List[Tensor],
    current_means: Tensor,
    current_quats: Tensor,
    current_scales: Tensor,
    current_opacities: Tensor,
    current_colors: Tensor,
    merge_strategy: str = "weighted_mean",
    **merge_kwargs
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Merge clusters using PyTorch.

    Args:
        clusters (List[Tensor]): List of cluster indices
        current_means (Tensor): Current means
        current_quats (Tensor): Current quaternions
        current_scales (Tensor): Current scales
        current_opacities (Tensor): Current opacities
        current_colors (Tensor): Current colors
        merge_strategy (str, optional): Merging strategy ("weighted_mean", "moment_matching"). Defaults to "weighted_mean".
        **merge_kwargs: Additional arguments for merging strategies

    Returns:
        merged_means: [M, 3] - merged Gaussian centers
        merged_quats: [M, 4] - merged Gaussian quaternions
        merged_scales: [M, 3] - merged Gaussian scales in linear space
        merged_opacities: [M] - merged Gaussian opacities in linear space
        merged_colors: [M, K, 3] or [M, 3] - merged Gaussian colors
    """
    # Prepare for merging
    device = current_means.device
    merged_gaussians = []
    cluster_indices_to_remove = []  # Collect tensor indices for vectorized removal

    # Merge each cluster
    for cluster_indices in clusters:
        # Strict tensor-only enforcement - no numpy fallbacks
        if not isinstance(cluster_indices, torch.Tensor):
            raise TypeError(f"All cluster_indices must be torch.Tensor, got {type(cluster_indices)} in clusters list. "
                           f"Use torch.from_numpy() to convert numpy arrays to tensors before calling this function.")
        
        # Merge the cluster
        merged_mean, merged_quat, merged_scale, merged_opacity, merged_color = _merge_cluster_torch(
            cluster_indices, current_means, current_quats, current_scales,
            current_opacities, current_colors, merge_strategy, **merge_kwargs
        )
        
        merged_gaussians.append({
            "mean": merged_mean,
            "quat": merged_quat,
            "scale": merged_scale,
            "opacity": merged_opacity,
            "color": merged_color
        })
        
        # Collect tensor indices for vectorized removal - no CPU conversion!
        cluster_indices_to_remove.append(cluster_indices)
    
    if not merged_gaussians:
        log.info(f"No successful merges")
        return current_means, current_quats, current_scales, current_opacities, current_colors
    
    # Create new parameter tensors using vectorized tensor operations
    # Keep non-merged Gaussians - vectorized approach, no CPU operations
    keep_mask = torch.ones(current_means.shape[0], dtype=torch.bool, device=device)
    
    if cluster_indices_to_remove:
        # Concatenate all cluster indices into a single tensor - fully vectorized
        all_indices_to_remove = torch.cat(cluster_indices_to_remove, dim=0)  # Stay on GPU
        
        # Vectorized mask update - much faster than Python loop
        keep_mask[all_indices_to_remove] = False
    
    kept_means = current_means[keep_mask]
    kept_quats = current_quats[keep_mask]
    kept_scales = current_scales[keep_mask]
    kept_opacities = current_opacities[keep_mask]
    kept_colors = current_colors[keep_mask]
    
    # Add merged Gaussians
    if merged_gaussians:
        merged_means_list = [g["mean"] for g in merged_gaussians]
        merged_quats_list = [g["quat"] for g in merged_gaussians]
        merged_scales_list = [g["scale"] for g in merged_gaussians]
        merged_opacities_list = [g["opacity"] for g in merged_gaussians]
        merged_colors_list = [g["color"] for g in merged_gaussians]
        
        new_merged_means = torch.stack(merged_means_list)
        new_merged_quats = torch.stack(merged_quats_list)
        new_merged_scales = torch.stack(merged_scales_list)
        new_merged_opacities = torch.stack(merged_opacities_list)
        new_merged_colors = torch.stack(merged_colors_list)
        
        # Combine kept and merged
        merged_means = torch.cat([kept_means, new_merged_means], dim=0)
        merged_quats = torch.cat([kept_quats, new_merged_quats], dim=0)
        merged_scales = torch.cat([kept_scales, new_merged_scales], dim=0)
        merged_opacities = torch.cat([kept_opacities, new_merged_opacities], dim=0)
        merged_colors = torch.cat([kept_colors, new_merged_colors], dim=0)
    else:
        merged_means = kept_means
        merged_quats = kept_quats
        merged_scales = kept_scales
        merged_opacities = kept_opacities
        merged_colors = kept_colors
    
    return merged_means, merged_quats, merged_scales, merged_opacities, merged_colors

def merge_gaussians_torch(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [N, K, 3] or [N, 3]
    viewmat: Tensor,  # [4, 4]
    K: Tensor,  # [3, 3]
    width: int,
    height: int,
    pixel_size_threshold: float = 2.0,
    pixel_area_threshold: float = 4.0,
    use_pixel_area: bool = False,
    clustering_method: str = "knn",
    merge_strategy: str = "weighted_mean",
    scale_modifier: float = 1.0,
    eps2d: float = 0.3,
    accelerated: bool = True,
    clustering_kwargs: Optional[Dict] = None,
    merge_kwargs: Optional[Dict] = None,
    max_iterations: int = 1,
    min_reduction_ratio: float = 0.01
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
    """
    Main function to merge small Gaussians.
    
    Args:
        means: [N, 3] - Gaussian centers in world coordinates
        quats: [N, 4] - Gaussian quaternions
        scales: [N, 3] - Gaussian scales in linear space
        opacities: [N] - Gaussian opacities in linear space
        colors: [N, K, 3] or [N, 3] - Gaussian colors (SH coefficients or RGB)
        viewmat: [4, 4] - view matrix (world to camera)
        K: [3, 3] - camera intrinsic matrix
        width: int - image width
        height: int - image height
        pixel_size_threshold: float - minimum pixel size for merging
        pixel_area_threshold: float - minimum pixel area for merging
        use_pixel_area: bool - whether to use pixel area instead of pixel size
        clustering_method: str - clustering method ("knn", "dbscan", "distance_based")
        merge_strategy: str - merging strategy ("weighted_mean", "moment_matching")
        scale_modifier: float - scale modifier applied to Gaussian scales
        eps2d: float - low-pass filter value
        accelerated: bool - whether to use accelerated clustering (True for CUDA, False for PyTorch)
        clustering_kwargs: Dict - additional arguments for clustering
        merge_kwargs: Dict - additional arguments for merging
        max_iterations: int - maximum number of merging iterations
        min_reduction_ratio: float - minimum reduction ratio to continue iterating
        
    Returns:
        merged_means: [M, 3] - merged Gaussian centers
        merged_quats: [M, 4] - merged Gaussian quaternions
        merged_scales: [M, 3] - merged Gaussian scales in linear space
        merged_opacities: [M] - merged Gaussian opacities in linear space
        merged_colors: [M, K, 3] or [M, 3] - merged Gaussian colors
        merge_info: Dict - information about the merging process
    """

    start_time = time.time()
    if clustering_kwargs is None:
        raise ValueError("clustering_kwargs must be provided and cannot be None.")
    if merge_kwargs is None:
        raise ValueError("merge_kwargs must be provided and cannot be None.")
    
    # # Set default clustering parameters
    # if clustering_method == "knn" and not clustering_kwargs:
    #     clustering_kwargs = {"k_neighbors": 3, "max_distance": 0.1, "min_cluster_size": 2}
    # elif clustering_method == "dbscan" and not clustering_kwargs:
    #     clustering_kwargs = {"eps": 0.1, "min_samples": 2}
    # elif clustering_method == "distance_based" and not clustering_kwargs:
    #     clustering_kwargs = {"max_distance": 0.1, "min_cluster_size": 2}
    
    original_count = means.shape[0]
    current_means = means.clone()
    current_quats = quats.clone()
    current_scales = scales.clone()
    current_opacities = opacities.clone()
    current_colors = colors.clone()

    merge_info = {
        "original_count": original_count,
        "iterations": 0,
        "total_merged": 0,
        "final_count": 0,
        "reduction_ratio": 0.0,
        "clusters_per_iteration": []
    }
    
    end_time = time.time()
    log.info(f"Preprocess time: {(end_time - start_time)*1000:.2f} ms")

    for iteration in range(max_iterations):
        # breakpoint()
        # Find merge candidates
        start_time = time.time()
        candidate_mask = find_merge_candidates(
            current_means, current_quats, current_scales, current_opacities,
            viewmat, K, width, height, pixel_size_threshold, use_pixel_area, pixel_area_threshold,
            scale_modifier, eps2d, method="cuda" if accelerated else "torch"
        )
        
        if not candidate_mask.any():
            log.warning(f"No merge candidates found at iteration {iteration}")
            break
        
        end_time = time.time()
        log.info(f"Find Merge Candidates time: {(end_time - start_time)*1000:.2f} ms")

        # Cluster candidates
        start_time = time.time()
        if clustering_method == "center_in_pixel":
            # Add camera parameters for center_in_pixel clustering
            clustering_kwargs_with_camera = clustering_kwargs.copy()
            clustering_kwargs_with_camera.update({
                "viewmat": viewmat,
                "K": K,
                "width": width,
                "height": height
            })
            clusters = cluster_gaussians(
                current_means, candidate_mask, clustering_method,
                method="cuda" if accelerated else "torch",
                **clustering_kwargs_with_camera
            )
        else:
            clusters = cluster_gaussians(
                current_means, candidate_mask, clustering_method, method="torch", **clustering_kwargs
            )
        
        if not clusters:
            log.warning(f"No clusters found at iteration {iteration}")
            break
        
        merge_info["clusters_per_iteration"].append(len(clusters))

        end_time = time.time()
        log.info(f"Cluster Candidates time: {(end_time - start_time)*1000:.2f} ms")
        
        start_time = time.time()
        merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = merge_clusters_torch(
            clusters, current_means, current_quats, current_scales, current_opacities, current_colors, merge_strategy, **merge_kwargs
        )

        current_means = merged_means
        current_quats = merged_quats
        current_scales = merged_scales
        current_opacities = merged_opacities
        current_colors = merged_colors

        end_time = time.time()
        log.info(f"Merge Clusters time: {(end_time - start_time)*1000:.2f} ms")

        # Update merge info
        current_count = merged_means.shape[0]
        merged_this_iteration = original_count - current_count
        merge_info["total_merged"] += merged_this_iteration
        merge_info["iterations"] += 1
        
        # Check if we should continue iterating
        if iteration > 0:
            reduction_this_iteration = merged_this_iteration / original_count
            if reduction_this_iteration < min_reduction_ratio:
                log.info(f"Reduction ratio {reduction_this_iteration:.4f} below threshold {min_reduction_ratio}, stopping")
                break
        
        log.info(f"Iteration {iteration}: {len(clusters)} clusters, "
              f"{original_count} → {current_count} "
              f"(net reduction: {merged_this_iteration}), "
              f"total count: {current_count}")
    
    # Final merge info
    merge_info["final_count"] = current_count
    merge_info["reduction_ratio"] = (original_count - current_count) / original_count
    
    log.info(f"Merging complete: {original_count} → {merge_info['final_count']} "
          f"({merge_info['reduction_ratio']:.1%} reduction)")
    
    return current_means, current_quats, current_scales, current_opacities, current_colors, merge_info

def merge_gaussians_cuda(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [N, K, 3] or [N, 3]
    viewmat: Tensor,  # [4, 4]
    K: Tensor,  # [3, 3]
    width: int,
    height: int,
    pixel_size_threshold: float = 2.0,
    pixel_area_threshold: float = 4.0,
    use_pixel_area: bool = False,
    clustering_method: str = "center_in_pixel",
    merge_strategy: str = "weighted_mean",
    scale_modifier: float = 1.0,
    eps2d: float = 0.3,
    clustering_kwargs: Optional[Dict] = None,
    merge_kwargs: Optional[Dict] = None,
    max_iterations: int = 1,
    min_reduction_ratio: float = 0.01
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
    """
    CUDA-accelerated function to merge small Gaussians.
    
    This function implements the high-performance GPU pipeline for Gaussian merging.
    Currently optimized for center_in_pixel clustering method.
    
    Args:
        means: [N, 3] - Gaussian centers in world coordinates
        quats: [N, 4] - Gaussian quaternions
        scales: [N, 3] - Gaussian scales in linear space
        opacities: [N] - Gaussian opacities in linear space
        colors: [N, K, 3] or [N, 3] - Gaussian colors (SH coefficients or RGB)
        viewmat: [4, 4] - view matrix (world to camera)
        K: [3, 3] - camera intrinsic matrix
        width: int - image width
        height: int - image height
        pixel_size_threshold: float - minimum pixel size for merging
        pixel_area_threshold: float - minimum pixel area for merging
        use_pixel_area: bool - whether to use pixel area instead of pixel size
        clustering_method: str - clustering method (currently only "center_in_pixel" supported for CUDA)
        merge_strategy: str - merging strategy ("weighted_mean", "moment_matching")
        scale_modifier: float - scale modifier applied to Gaussian scales
        eps2d: float - low-pass filter value
        clustering_kwargs: Dict - additional arguments for clustering
        merge_kwargs: Dict - additional arguments for merging
        max_iterations: int - maximum number of merging iterations
        min_reduction_ratio: float - minimum reduction ratio to continue iterating
        
    Returns:
        merged_means: [M, 3] - merged Gaussian centers
        merged_quats: [M, 4] - merged Gaussian quaternions
        merged_scales: [M, 3] - merged Gaussian scales in linear space
        merged_opacities: [M] - merged Gaussian opacities in linear space
        merged_colors: [M, K, 3] or [M, 3] - merged Gaussian colors
        merge_info: Dict - information about the merging process
    """
    
    start_time = time.time()
    if clustering_kwargs is None:
        raise ValueError("clustering_kwargs must be provided and cannot be None.")
    if merge_kwargs is None:
        raise ValueError("merge_kwargs must be provided and cannot be None.")
    
    # Currently only center_in_pixel clustering is supported for CUDA acceleration
    if clustering_method != "center_in_pixel":
        raise ValueError(f"CUDA acceleration currently only supports 'center_in_pixel' clustering method, got '{clustering_method}'")
    
    original_count = means.shape[0]
    current_means = means.clone()
    current_quats = quats.clone()
    current_scales = scales.clone()
    current_opacities = opacities.clone()
    current_colors = colors.clone()  # Color format conversion will be handled by merge_clusters_cuda()

    merge_info = {
        "original_count": original_count,
        "iterations": 0,
        "total_merged": 0,
        "final_count": 0,
        "reduction_ratio": 0.0,
        "clusters_per_iteration": []
    }
    
    end_time = time.time()
    log.info(f"Preprocess time: {(end_time - start_time)*1000:.2f} ms")

    for iteration in range(max_iterations):
        # Find merge candidates
        start_time = time.time()
        candidate_mask = find_merge_candidates(
            current_means, current_quats, current_scales, current_opacities,
            viewmat, K, width, height, pixel_size_threshold, use_pixel_area, pixel_area_threshold,
            scale_modifier, eps2d, method="cuda"
        )
        
        if not candidate_mask.any():
            log.warning(f"No merge candidates found at iteration {iteration}")
            break
        
        candidate_indices = torch.where(candidate_mask)[0]  # Keep as tensor on GPU
        candidate_means = current_means[candidate_mask]
        
        end_time = time.time()
        log.info(f"Find Merge Candidates time: {(end_time - start_time)*1000:.2f} ms")

        # CUDA-accelerated clustering
        start_time = time.time()
        clusters_result = cluster_center_in_pixel_cuda(
            candidate_means, candidate_indices,
            viewmat, K, width, height,
            return_flat_format=True,  # Return flat format for CUDA merging
            **clustering_kwargs
        )
        
        if clusters_result['num_clusters'] == 0:
            log.warning(f"No clusters found at iteration {iteration}")
            break
        
        merge_info["clusters_per_iteration"].append(clusters_result['num_clusters'])
        
        end_time = time.time()
        log.info(f"Cluster Candidates time: {(end_time - start_time)*1000:.2f} ms")
        
        # CUDA-accelerated merging
        start_time = time.time()
        current_means, current_quats, current_scales, current_opacities, current_colors = merge_clusters_cuda(
            cluster_indices=clusters_result['cluster_indices'],
            cluster_offsets=clusters_result['cluster_offsets'],
            current_means=current_means,
            current_quats=current_quats,
            current_scales=current_scales,
            current_opacities=current_opacities,
            current_colors=current_colors,
            merge_strategy=merge_strategy,
            **merge_kwargs
        )

        end_time = time.time()
        log.info(f"Merge Clusters time: {(end_time - start_time)*1000:.2f} ms")

        # Update merge info
        current_count = current_means.shape[0]
        merged_this_iteration = original_count - current_count
        merge_info["total_merged"] += merged_this_iteration
        merge_info["iterations"] += 1
        
        # Check if we should continue iterating
        if iteration > 0:
            reduction_this_iteration = merged_this_iteration / original_count
            if reduction_this_iteration < min_reduction_ratio:
                log.info(f"Reduction ratio {reduction_this_iteration:.4f} below threshold {min_reduction_ratio}, stopping")
                break
        
        log.info(f"Iteration {iteration}: {clusters_result['num_clusters']} clusters, "
              f"{original_count} → {current_count} "
              f"(net reduction: {merged_this_iteration}), "
              f"total count: {current_count}")
    
    # Final merge info
    merge_info["final_count"] = current_count
    merge_info["reduction_ratio"] = (original_count - current_count) / original_count
    
    log.info(f"CUDA merging complete: {original_count} → {merge_info['final_count']} "
          f"({merge_info['reduction_ratio']:.1%} reduction)")
    
    return current_means, current_quats, current_scales, current_opacities, current_colors, merge_info

def merge_gaussians(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [N, K, 3] or [N, 3]
    viewmat: Tensor,  # [4, 4]
    K: Tensor,  # [3, 3]
    width: int,
    height: int,
    pixel_size_threshold: float = 2.0,
    pixel_area_threshold: float = 4.0,
    use_pixel_area: bool = False,
    clustering_method: str = "knn",
    merge_strategy: str = "weighted_mean",
    scale_modifier: float = 1.0,
    eps2d: float = 0.3,
    accelerated: bool = True,
    clustering_kwargs: Optional[Dict] = None,
    merge_kwargs: Optional[Dict] = None,
    max_iterations: int = 1,
    min_reduction_ratio: float = 0.01
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
    """
    Main dispatcher function to merge small Gaussians using either CPU or GPU implementation.
    
    This function automatically dispatches to the appropriate implementation based on the
    `accelerated` parameter and clustering method:
    - When accelerated=True and clustering_method="center_in_pixel": Uses CUDA pipeline
    - Otherwise: Uses PyTorch CPU pipeline
    
    Args:
        means: [N, 3] - Gaussian centers in world coordinates
        quats: [N, 4] - Gaussian quaternions
        scales: [N, 3] - Gaussian scales in linear space
        opacities: [N] - Gaussian opacities in linear space
        colors: [N, K, 3] or [N, 3] - Gaussian colors (SH coefficients or RGB)
        viewmat: [4, 4] - view matrix (world to camera)
        K: [3, 3] - camera intrinsic matrix
        width: int - image width
        height: int - image height
        pixel_size_threshold: float - minimum pixel size for merging
        pixel_area_threshold: float - minimum pixel area for merging
        use_pixel_area: bool - whether to use pixel area instead of pixel size
        clustering_method: str - clustering method ("knn", "dbscan", "distance_based", "center_in_pixel")
        merge_strategy: str - merging strategy ("weighted_mean", "moment_matching")
        scale_modifier: float - scale modifier applied to Gaussian scales
        eps2d: float - low-pass filter value
        accelerated: bool - whether to use GPU acceleration (True for CUDA, False for CPU)
        clustering_kwargs: Dict - additional arguments for clustering
        merge_kwargs: Dict - additional arguments for merging
        max_iterations: int - maximum number of merging iterations
        min_reduction_ratio: float - minimum reduction ratio to continue iterating
        
    Returns:
        merged_means: [M, 3] - merged Gaussian centers
        merged_quats: [M, 4] - merged Gaussian quaternions
        merged_scales: [M, 3] - merged Gaussian scales in linear space
        merged_opacities: [M] - merged Gaussian opacities in linear space
        merged_colors: [M, K, 3] or [M, 3] - merged Gaussian colors
        merge_info: Dict - information about the merging process
    """
    
    # Dispatch to appropriate implementation
    if accelerated and clustering_method == "center_in_pixel":
        log.info("Using CUDA-accelerated merging pipeline")
        return merge_gaussians_cuda(
            means, quats, scales, opacities, colors, viewmat, K, width, height,
            pixel_size_threshold, pixel_area_threshold, use_pixel_area,
            clustering_method, merge_strategy, scale_modifier, eps2d,
            clustering_kwargs, merge_kwargs, max_iterations, min_reduction_ratio
        )
    else:
        if accelerated and clustering_method != "center_in_pixel":
            log.warning(f"CUDA acceleration only supports 'center_in_pixel' clustering. "
                       f"Falling back to PyTorch implementation for '{clustering_method}'")
        log.info("Using PyTorch CPU merging pipeline")
        return merge_gaussians_torch(
            means, quats, scales, opacities, colors, viewmat, K, width, height,
            pixel_size_threshold, pixel_area_threshold, use_pixel_area,
            clustering_method, merge_strategy, scale_modifier, eps2d,
            accelerated, clustering_kwargs, merge_kwargs, max_iterations, min_reduction_ratio
        )
