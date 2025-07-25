'''
Other methods for aggregating gaussians in a voxel. I will evaluate these later.
'''

import numpy as np
from dataclasses import dataclass
from typing import ClassVar

@dataclass(frozen=True)
class AggregationMethod:
    # New improved methods
    geometric: ClassVar[str] = "geometric"
    adaptive: ClassVar[str] = "adaptive"
    covariance_weighted: ClassVar[str] = "covariance_weighted"
    gaussian_mixture: ClassVar[str] = "gaussian_mixture"
    volume_weighted: ClassVar[str] = "volume_weighted"
    opacity_preserved: ClassVar[str] = "opacity_preserved"
    hybrid_dominant: ClassVar[str] = "hybrid_dominant"
    
class Voxel:
    def __init__(self, voxel_id, voxel_size, voxel_center):
        self.voxel_id = voxel_id
        self.voxel_size = voxel_size
        self.voxel_center = voxel_center
        self.gaussian_ids = []
        self.aggregated_mean = None
        self.aggregated_opacity = None
        self.aggregated_quat = None
        self.aggregated_scale = None
        self.aggregated_sh0 = None
        self.aggregated_shN = None

    def add_gaussian(self, gaussian_id):
        self.gaussian_ids.append(gaussian_id)

    def agg_attr_weighted_geometric(self, means, opacities, quats, scales, sh0, shN):
        """Use weighted geometric mean for better preservation of spatial relationships"""
        # Convert to post-activation space for weighting  
        sigmoid_opacities = 1.0 / (1.0 + np.exp(-opacities))
        weights = sigmoid_opacities / np.sum(sigmoid_opacities)
        
        # For means, use weighted average
        self.aggregated_mean = np.sum(means * weights[:, np.newaxis], axis=0)
        
        # For opacity, use maximum in logit space
        self.aggregated_opacity = np.max(opacities)
        
        # For quaternions, use weighted average and normalize
        self.aggregated_quat = np.sum(quats * weights[:, np.newaxis], axis=0)
        self.aggregated_quat = self.aggregated_quat / np.linalg.norm(self.aggregated_quat)
        
        # For scales, weighted average in log space (geometric mean)
        self.aggregated_scale = np.sum(scales * weights[:, np.newaxis], axis=0)
        
        # For spherical harmonics, use weighted average
        self.aggregated_sh0 = np.sum(sh0 * weights[:, np.newaxis, np.newaxis], axis=0)
        self.aggregated_shN = np.sum(shN * weights[:, np.newaxis, np.newaxis], axis=0)

    def agg_attr_adaptive(self, means, opacities, quats, scales, sh0, shN):
        """Adaptive aggregation based on spatial distribution and opacity"""
        # Convert to post-activation space for weighting
        sigmoid_opacities = 1.0 / (1.0 + np.exp(-opacities))
        weights = sigmoid_opacities / np.sum(sigmoid_opacities)
        
        # Calculate spatial variance
        mean_center = np.sum(means * weights[:, np.newaxis], axis=0)
        spatial_variance = np.sum(np.square(means - mean_center) * weights[:, np.newaxis], axis=0)
        
        # If gaussians are well-separated, use dominant gaussian
        if np.any(spatial_variance > self.voxel_size * 0.5):
            self.agg_attr_dominant(means, opacities, quats, scales, sh0, shN)
        else:
            # If gaussians are close, use weighted geometric mean
            self.agg_attr_weighted_geometric(means, opacities, quats, scales, sh0, shN)

    def agg_attr_dominant(self, means, opacities, quats, scales, sh0, shN):
        """Use the gaussian with highest opacity as the representative"""
        # Find dominant based on logit opacity (pre-activation)
        dominant_idx = np.argmax(opacities)
        self.aggregated_mean = means[dominant_idx]
        self.aggregated_opacity = opacities[dominant_idx]  # Keep in logit space
        self.aggregated_quat = quats[dominant_idx]
        self.aggregated_scale = scales[dominant_idx]  # Keep in log space
        self.aggregated_sh0 = sh0[dominant_idx]
        self.aggregated_shN = shN[dominant_idx]
        
    def agg_attr_mean(self, means, opacities, quats, scales, sh0, shN):
        """Use the mean of the gaussians as the representative"""
        self.aggregated_mean = np.mean(means, axis=0)
        self.aggregated_opacity = np.mean(opacities)  # Mean in logit space
        self.aggregated_quat = np.mean(quats, axis=0)
        self.aggregated_quat = self.aggregated_quat / np.linalg.norm(self.aggregated_quat)
        self.aggregated_scale = np.mean(scales, axis=0)  # Mean in log space
        self.aggregated_sh0 = np.mean(sh0, axis=0)
        self.aggregated_shN = np.mean(shN, axis=0)

    def _quat_to_rotation_matrix(self, quat):
        """Convert quaternion to rotation matrix"""
        w, x, y, z = quat
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])

    def _rotation_matrix_to_quat(self, R):
        """Convert rotation matrix to quaternion"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2,1] - R[1,2]) / s
            y = (R[0,2] - R[2,0]) / s
            z = (R[1,0] - R[0,1]) / s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                w = (R[2,1] - R[1,2]) / s
                x = 0.25 * s
                y = (R[0,1] + R[1,0]) / s
                z = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                w = (R[0,2] - R[2,0]) / s
                x = (R[0,1] + R[1,0]) / s
                y = 0.25 * s
                z = (R[1,2] + R[2,1]) / s
            else:
                s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                w = (R[1,0] - R[0,1]) / s
                x = (R[0,2] + R[2,0]) / s
                y = (R[1,2] + R[2,1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z])

    def _get_covariance_matrix(self, scale, quat):
        """Get 3D covariance matrix from scale and rotation"""
        R = self._quat_to_rotation_matrix(quat)
        S = np.diag(scale)
        return R @ S @ S @ R.T

    def _covariance_to_scale_rotation(self, cov):
        """Decompose covariance matrix back to scale and rotation"""
        eigenvalues, eigenvectors = np.linalg.eigh(cov.astype(np.float64))  # Use float64 for numerical stability
        
        # Ensure positive eigenvalues
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        scales = np.sqrt(eigenvalues)
        
        # Ensure proper rotation matrix (det = 1)
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 0] *= -1
        
        quat = self._rotation_matrix_to_quat(eigenvectors)
        
        # Convert back to float32
        return scales.astype(np.float32), quat.astype(np.float32)

    def agg_attr_covariance_weighted(self, means, opacities, quats, scales, sh0, shN):
        """Aggregate using proper covariance matrix combination"""
        # Convert from pre-activation to post-activation space
        sigmoid_opacities = 1.0 / (1.0 + np.exp(-opacities))  # logit -> probability
        exp_scales = np.exp(scales)  # log -> linear scales
        
        weights = sigmoid_opacities / np.sum(sigmoid_opacities)
        
        # Weighted mean position
        self.aggregated_mean = np.sum(means * weights[:, np.newaxis], axis=0).astype(np.float32)
        
        # Aggregate covariance matrices using linear scales
        total_cov = np.zeros((3, 3), dtype=np.float32)
        for i, (exp_scale, quat, weight) in enumerate(zip(exp_scales, quats, weights)):
            cov = self._get_covariance_matrix(exp_scale, quat).astype(np.float32)
            total_cov += weight * cov
        
        # Decompose back to scale and rotation
        linear_scales, self.aggregated_quat = self._covariance_to_scale_rotation(total_cov)
        
        # Convert back to log space for scales
        self.aggregated_scale = np.log(np.maximum(linear_scales, 1e-8)).astype(np.float32)
        self.aggregated_quat = self.aggregated_quat.astype(np.float32)
        
        # Opacity: combine sigmoid opacities then convert back to logit
        combined_sigmoid_opacity = np.sum(sigmoid_opacities * weights)
        combined_sigmoid_opacity = np.clip(combined_sigmoid_opacity, 1e-8, 1-1e-8)
        self.aggregated_opacity = np.log(combined_sigmoid_opacity / (1 - combined_sigmoid_opacity)).astype(np.float32)
        
        # Spherical harmonics: weighted average
        self.aggregated_sh0 = np.sum(sh0 * weights[:, np.newaxis, np.newaxis], axis=0).astype(np.float32)
        self.aggregated_shN = np.sum(shN * weights[:, np.newaxis, np.newaxis], axis=0).astype(np.float32)

    def agg_attr_volume_weighted(self, means, opacities, quats, scales, sh0, shN):
        """Weight by Gaussian volume (determinant of covariance)"""
        # Convert to post-activation space
        sigmoid_opacities = 1.0 / (1.0 + np.exp(-opacities))
        exp_scales = np.exp(scales)
        
        # Calculate volumes (determinant of scale matrix)
        volumes = np.prod(exp_scales, axis=1)
        
        # Combine opacity and volume for weighting
        volume_opacity_weights = sigmoid_opacities * volumes
        weights = volume_opacity_weights / np.sum(volume_opacity_weights)
        
        self.aggregated_mean = np.sum(means * weights[:, np.newaxis], axis=0).astype(np.float32)
        
        # Use covariance combination with linear scales
        total_cov = np.zeros((3, 3), dtype=np.float32)
        for i, (exp_scale, quat, weight) in enumerate(zip(exp_scales, quats, weights)):
            cov = self._get_covariance_matrix(exp_scale, quat).astype(np.float32)
            total_cov += weight * cov
        
        linear_scales, self.aggregated_quat = self._covariance_to_scale_rotation(total_cov)
        self.aggregated_scale = np.log(np.maximum(linear_scales, 1e-8)).astype(np.float32)
        
        # Opacity: combine and convert back to logit
        combined_sigmoid_opacity = np.sum(sigmoid_opacities * weights)
        combined_sigmoid_opacity = np.clip(combined_sigmoid_opacity, 1e-8, 1-1e-8)
        self.aggregated_opacity = np.log(combined_sigmoid_opacity / (1 - combined_sigmoid_opacity)).astype(np.float32)
        
        self.aggregated_sh0 = np.sum(sh0 * weights[:, np.newaxis, np.newaxis], axis=0).astype(np.float32)
        self.aggregated_shN = np.sum(shN * weights[:, np.newaxis, np.newaxis], axis=0).astype(np.float32)

    def agg_attr_opacity_preserved(self, means, opacities, quats, scales, sh0, shN):
        """Focus on preserving opacity distribution and color information"""
        # Convert to post-activation space
        sigmoid_opacities = 1.0 / (1.0 + np.exp(-opacities))
        exp_scales = np.exp(scales)
        
        weights = sigmoid_opacities / np.sum(sigmoid_opacities)
        
        # Weighted position
        self.aggregated_mean = np.sum(means * weights[:, np.newaxis], axis=0).astype(np.float32)
        
        # Preserve total opacity - use sum of sigmoid opacities
        total_sigmoid_opacity = np.mean(sigmoid_opacities) * 1.2  # Boost to compensate
        total_sigmoid_opacity = np.clip(total_sigmoid_opacity, 1e-8, 1-1e-8)
        self.aggregated_opacity = np.log(total_sigmoid_opacity / (1 - total_sigmoid_opacity)).astype(np.float32)
        
        # Weight-based quaternion interpolation (SLERP approximation)
        self.aggregated_quat = np.sum(quats * weights[:, np.newaxis], axis=0)
        self.aggregated_quat = (self.aggregated_quat / np.linalg.norm(self.aggregated_quat)).astype(np.float32)
        
        # Scale: geometric mean in linear space, then convert back to log
        weighted_linear_scales = np.exp(np.sum(scales * weights[:, np.newaxis], axis=0))  # geometric mean
        self.aggregated_scale = np.log(np.maximum(weighted_linear_scales, 1e-8)).astype(np.float32)
        
        # Preserve color information better
        safe_weights = np.maximum(weights, 1e-8)
        weighted_power = safe_weights**0.8
        
        self.aggregated_sh0 = np.sum(sh0 * weighted_power[:, np.newaxis, np.newaxis], axis=0).astype(np.float32)
        self.aggregated_shN = np.sum(shN * weighted_power[:, np.newaxis, np.newaxis], axis=0).astype(np.float32)

    def agg_attr_hybrid_dominant(self, means, opacities, quats, scales, sh0, shN):
        """Hybrid approach: dominant for geometry, weighted for appearance"""
        # Convert to post-activation space
        sigmoid_opacities = 1.0 / (1.0 + np.exp(-opacities))
        
        # Find dominant Gaussian based on sigmoid opacity
        dominant_idx = np.argmax(sigmoid_opacities)
        weights = sigmoid_opacities / np.sum(sigmoid_opacities)
        
        # Use dominant for position and basic geometry
        self.aggregated_mean = means[dominant_idx].astype(np.float32)
        self.aggregated_quat = quats[dominant_idx].astype(np.float32)
        
        # Use dominant scale but slightly increase (in log space)
        self.aggregated_scale = (scales[dominant_idx] + np.log(1.1)).astype(np.float32)
        
        # Combine sigmoid opacities from all gaussians
        dominant_sigmoid = sigmoid_opacities[dominant_idx]
        other_sigmoid = np.sum(sigmoid_opacities[np.arange(len(sigmoid_opacities)) != dominant_idx])
        combined_sigmoid = dominant_sigmoid + 0.3 * other_sigmoid
        combined_sigmoid = np.clip(combined_sigmoid, 1e-8, 1-1e-8)
        self.aggregated_opacity = np.log(combined_sigmoid / (1 - combined_sigmoid)).astype(np.float32)
        
        # Weight-combine appearance (SH coefficients)
        self.aggregated_sh0 = np.sum(sh0 * weights[:, np.newaxis, np.newaxis], axis=0).astype(np.float32)
        self.aggregated_shN = np.sum(shN * weights[:, np.newaxis, np.newaxis], axis=0).astype(np.float32)

    def agg_attr_gaussian_mixture(self, means, opacities, quats, scales, sh0, shN):
        """Approximate multiple Gaussians with single Gaussian using moment matching"""
        # Convert to post-activation space
        sigmoid_opacities = 1.0 / (1.0 + np.exp(-opacities))
        exp_scales = np.exp(scales)
        
        weights = sigmoid_opacities / np.sum(sigmoid_opacities)
        
        # First moment (mean)
        mu = np.sum(means * weights[:, np.newaxis], axis=0)
        self.aggregated_mean = mu.astype(np.float32)
        
        # Second moment (covariance) using linear scales
        total_cov = np.zeros((3, 3), dtype=np.float32)
        
        # Add individual covariances
        for i, (exp_scale, quat, weight) in enumerate(zip(exp_scales, quats, weights)):
            individual_cov = self._get_covariance_matrix(exp_scale, quat).astype(np.float32)
            total_cov += weight * individual_cov
        
        # Add inter-gaussian covariance (spread between means)
        for i, (mean_i, weight_i) in enumerate(zip(means, weights)):
            diff = (mean_i - mu).reshape(-1, 1)
            total_cov += weight_i * (diff @ diff.T)
        
        # Decompose to get scale and rotation
        linear_scales, self.aggregated_quat = self._covariance_to_scale_rotation(total_cov)
        self.aggregated_scale = np.log(np.maximum(linear_scales, 1e-8)).astype(np.float32)
        
        # Opacity: sum of weighted sigmoid opacities with boost
        combined_sigmoid_opacity = np.sum(sigmoid_opacities * weights) * 1.1
        combined_sigmoid_opacity = np.clip(combined_sigmoid_opacity, 1e-8, 1-1e-8)
        self.aggregated_opacity = np.log(combined_sigmoid_opacity / (1 - combined_sigmoid_opacity)).astype(np.float32)
        
        # Appearance: weighted combination
        self.aggregated_sh0 = np.sum(sh0 * weights[:, np.newaxis, np.newaxis], axis=0).astype(np.float32)
        self.aggregated_shN = np.sum(shN * weights[:, np.newaxis, np.newaxis], axis=0).astype(np.float32)

    def aggregate_attributes(self, trained_model, aggregate_method=AggregationMethod.gaussian_mixture):
        if not self.gaussian_ids:
            return
        
        # Get all gaussians in this voxel
        means = trained_model["splats"]["means"][self.gaussian_ids]
        opacities = trained_model["splats"]["opacities"][self.gaussian_ids]
        quats = trained_model["splats"]["quats"][self.gaussian_ids]
        scales = trained_model["splats"]["scales"][self.gaussian_ids]
        sh0 = trained_model["splats"]["sh0"][self.gaussian_ids]
        shN = trained_model["splats"]["shN"][self.gaussian_ids]
        
        # Select aggregation method
        if aggregate_method == AggregationMethod.geometric:
            method = self.agg_attr_weighted_geometric
        elif aggregate_method == AggregationMethod.adaptive:
            method = self.agg_attr_adaptive
        elif aggregate_method == AggregationMethod.covariance_weighted:
            method = self.agg_attr_covariance_weighted
        elif aggregate_method == AggregationMethod.volume_weighted:
            method = self.agg_attr_volume_weighted
        elif aggregate_method == AggregationMethod.opacity_preserved:
            method = self.agg_attr_opacity_preserved
        elif aggregate_method == AggregationMethod.hybrid_dominant:
            method = self.agg_attr_hybrid_dominant
        elif aggregate_method == AggregationMethod.gaussian_mixture:
            method = self.agg_attr_gaussian_mixture
        else:
            raise ValueError(f"Invalid aggregate method: {aggregate_method}")
        
        method(means, opacities, quats, scales, sh0, shN)