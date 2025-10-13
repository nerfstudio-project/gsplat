from .culling import frustum_culling, distance_culling, calc_pixel_size, distance_culling_area, calc_pixel_area
from .merging import merge_gaussians, find_merge_candidates, cluster_gaussians

# CUDA-accelerated clustering (optional)
try:
    from .clustering_cuda import (
        cluster_center_in_pixel_cuda,
        cluster_center_in_pixel_torch,
        get_clustering_performance_info,
        benchmark_clustering_performance
    )
    _CUDA_CLUSTERING_AVAILABLE = True
except ImportError:
    _CUDA_CLUSTERING_AVAILABLE = False

__all__ = [
    # Culling functions
    'frustum_culling', 'distance_culling', 'calc_pixel_size', 'distance_culling_area', 'calc_pixel_area',
    # Merging functions  
    'merge_gaussians', 'find_merge_candidates', 'cluster_gaussians',
]

# Add CUDA functions to exports if available
if _CUDA_CLUSTERING_AVAILABLE:
    __all__.extend([
        'cluster_center_in_pixel_cuda',
        'cluster_center_in_pixel_torch',
        'get_clustering_performance_info',
        'benchmark_clustering_performance'
    ])