import torch
from torch import Tensor
from typing import Tuple, List, Dict, Optional, Union, Callable
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from .culling import calc_pixel_size, calc_pixel_area
from gsplat.cuda._torch_impl import _world_to_cam
from gsplat.cuda._wrapper import proj


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
    clustering_method: str = "knn",
    **clustering_kwargs
) -> List[np.ndarray]:
    """
    Cluster candidate Gaussians for merging using various methods.
    
    Args:
        means: [N, 3] - Gaussian centers in world coordinates
        candidate_mask: [N] - boolean mask for merge candidates
        clustering_method: str - clustering method ("knn", "dbscan", "distance_based", "center_in_pixel")
        **clustering_kwargs: additional arguments for clustering methods
        
    Returns:
        clusters: List of arrays containing indices of Gaussians in each cluster
    """
    if not candidate_mask.any():
        return []
    
    # TODO: If we use the python version of the clustering, we can avoid the conversion to numpy and back
    candidate_indices = torch.where(candidate_mask)[0].cpu().numpy()
    candidate_means = means[candidate_mask]  # [M, 3] - keep as tensor for center_in_pixel
    
    if clustering_method == "knn":
        return _cluster_knn(candidate_means.cpu().numpy(), candidate_indices, **clustering_kwargs)
    elif clustering_method == "dbscan":
        return _cluster_dbscan(candidate_means.cpu().numpy(), candidate_indices, **clustering_kwargs)
    elif clustering_method == "distance_based":
        return _cluster_distance_based(candidate_means.cpu().numpy(), candidate_indices, **clustering_kwargs)
    elif clustering_method == "center_in_pixel":
        return _cluster_center_in_pixel(candidate_means, candidate_indices, **clustering_kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")


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


def _cluster_center_in_pixel(
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
        print("Not enough candidate means for center-in-pixel clustering")
        return []
    
    device = candidate_means.device
    M = len(candidate_means)
    
    # 1. Transform to camera coordinates
    # Add batch dimension for compatibility with _world_to_cam
    means_batch = candidate_means.unsqueeze(0)  # [1, M, 3]
    viewmat_batch = viewmat.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
    
    # Transform to camera coordinates (no covariances needed, just pass dummy ones)
    dummy_covars = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, M, 3, 3)  # [1, M, 3, 3]
    means_cam_batch, _ = _world_to_cam(means_batch, dummy_covars, viewmat_batch)
    
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
        print("No valid points found for center-in-pixel clustering")
        return []
    
    # Apply valid mask
    valid_indices = torch.where(valid_mask)[0]
    valid_pixel_coords = pixel_coords[valid_mask]  # [V, 2]
    valid_depths = means_cam[valid_mask, 2]  # [V] - camera Z coordinate (depth)
    valid_candidate_indices = candidate_indices[valid_indices.cpu().numpy()]
    
    # 4. Group by pixel coordinates
    pixel_groups = {}
    for i, (pixel_coord, depth, orig_idx) in enumerate(zip(valid_pixel_coords, valid_depths, valid_candidate_indices)):
        pixel_key = (pixel_coord[0].item(), pixel_coord[1].item())
        if pixel_key not in pixel_groups:
            pixel_groups[pixel_key] = []
        pixel_groups[pixel_key].append((i, depth.item(), orig_idx))
    
    # 5. Sub-cluster by depth within each pixel
    clusters = []
    for pixel_key, pixel_group in pixel_groups.items():
        if len(pixel_group) < min_cluster_size:
            continue
        
        # Sort by depth
        pixel_group.sort(key=lambda x: x[1])  # Sort by depth (x[1])
        
        # Perform depth clustering using simple distance threshold
        depth_clusters = []
        current_cluster = [pixel_group[0]]
        
        for j in range(1, len(pixel_group)):
            depth_diff = abs(pixel_group[j][1] - pixel_group[j-1][1])
            
            if depth_diff <= depth_threshold:
                # Add to current cluster
                current_cluster.append(pixel_group[j])
            else:
                # If the current cluster is valid, add it to the depth clusters, otherwise discard it
                if len(current_cluster) >= min_cluster_size:
                    depth_clusters.append(current_cluster)
                # Start new cluster
                current_cluster = [pixel_group[j]]
        
        # Don't forget the last cluster
        if len(current_cluster) >= min_cluster_size:
            depth_clusters.append(current_cluster)
        
        # Convert depth clusters to original indices
        for depth_cluster in depth_clusters:
            if len(depth_cluster) >= min_cluster_size:
                original_cluster = np.array([item[2] for item in depth_cluster])  # Extract original indices
                clusters.append(original_cluster)
    
    return clusters


def merge_cluster(
    cluster_indices: np.ndarray,  # [C] - indices of Gaussians in cluster
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
    cluster_indices = torch.from_numpy(cluster_indices).to(means.device)
    
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
    method: str = "cuda",
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
        method: str - method to use for calculation ("cuda" or "torch")
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
    
    device = means.device
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
    
    for iteration in range(max_iterations):
        # breakpoint()
        # Find merge candidates
        candidate_mask = find_merge_candidates(
            current_means, current_quats, current_scales, current_opacities,
            viewmat, K, width, height, pixel_size_threshold, use_pixel_area, pixel_area_threshold,
            scale_modifier, eps2d, method
        )
        
        if not candidate_mask.any():
            print(f"No merge candidates found at iteration {iteration}")
            break
        
        # Cluster candidates
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
                current_means, candidate_mask, clustering_method, **clustering_kwargs_with_camera
            )
        else:
            clusters = cluster_gaussians(
                current_means, candidate_mask, clustering_method, **clustering_kwargs
            )
        
        if not clusters:
            print(f"No clusters found at iteration {iteration}")
            break
        
        merge_info["clusters_per_iteration"].append(len(clusters))
        
        # Prepare for merging
        merged_gaussians = []
        indices_to_remove = set()
        
        # Merge each cluster
        for cluster_indices in clusters:
            # Merge the cluster
            merged_mean, merged_quat, merged_scale, merged_opacity, merged_color = merge_cluster(
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
            
            # Mark original Gaussians for removal
            indices_to_remove.update(cluster_indices)
        
        if not merged_gaussians:
            print(f"No successful merges at iteration {iteration}")
            break
        
        # Create new parameter tensors
        # Keep non-merged Gaussians
        keep_mask = torch.ones(current_means.shape[0], dtype=torch.bool, device=device)
        for idx in indices_to_remove:
            keep_mask[idx] = False
        
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
            current_means = torch.cat([kept_means, new_merged_means], dim=0)
            current_quats = torch.cat([kept_quats, new_merged_quats], dim=0)
            current_scales = torch.cat([kept_scales, new_merged_scales], dim=0)
            current_opacities = torch.cat([kept_opacities, new_merged_opacities], dim=0)
            current_colors = torch.cat([kept_colors, new_merged_colors], dim=0)
        else:
            current_means = kept_means
            current_quats = kept_quats
            current_scales = kept_scales
            current_opacities = kept_opacities
            current_colors = kept_colors
        
        # Update merge info
        current_count = current_means.shape[0]
        merged_this_iteration = len(indices_to_remove) - len(merged_gaussians)
        merge_info["total_merged"] += merged_this_iteration
        merge_info["iterations"] += 1
        
        # Check if we should continue iterating
        if iteration > 0:
            reduction_this_iteration = merged_this_iteration / original_count
            if reduction_this_iteration < min_reduction_ratio:
                print(f"Reduction ratio {reduction_this_iteration:.4f} below threshold {min_reduction_ratio}, stopping")
                break
        
        print(f"Iteration {iteration}: {len(clusters)} clusters, "
              f"{len(indices_to_remove)} → {len(merged_gaussians)} "
              f"(net reduction: {merged_this_iteration}), "
              f"total count: {current_count}")
    
    # Final merge info
    merge_info["final_count"] = current_means.shape[0]
    merge_info["reduction_ratio"] = (original_count - merge_info["final_count"]) / original_count
    
    print(f"Merging complete: {original_count} → {merge_info['final_count']} "
          f"({merge_info['reduction_ratio']:.1%} reduction)")
    
    return current_means, current_quats, current_scales, current_opacities, current_colors, merge_info
