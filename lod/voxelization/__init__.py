"""
Voxelization module for 3D Gaussian Splatting.

This module provides functionality to voxelize 3D Gaussian splat models
using various aggregation methods.
"""

from .voxelization import Voxelization
from .voxel_methods import AggregationMethod, Voxel
from .barycenter import wasserstein_barycenter_gaussians, wasserstein_barycenter_gaussians_orig

__all__ = ['Voxelization', 'AggregationMethod', 'Voxel', 'wasserstein_barycenter_gaussians', 'wasserstein_barycenter_gaussians_orig'] 