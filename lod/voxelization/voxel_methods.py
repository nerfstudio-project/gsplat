import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from .barycenter import wasserstein_barycenter_gaussians, wasserstein_barycenter_gaussians_orig

# -----------------------------------------------------------------------------
# Helper routines --------------------------------------------------------------
# -----------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid used for converting log‑opacity → mixture weight."""
    # Clamp to avoid overflow in exp for very negative numbers
    # return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))
    return 1.0 / (1.0 + np.exp(-x))

def _logit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Stable logit used for converting mixture weight → log‑opacity."""
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))

def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) → 3×3 rotation matrix.
    Assumes *unnormalised* input and normalises internally.
    """
    x, y, z, w = q / np.linalg.norm(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz),       2.0 * (xy - wz),         2.0 * (xz + wy)],
        [      2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz),         2.0 * (yz - wx)],
        [      2.0 * (xz - wy),       2.0 * (yz + wx),   1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float32)

def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → quaternion (x, y, z, w)."""
    m00, m11, m22 = np.diag(R)
    trace = m00 + m11 + m22

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if m00 > m11 and m00 > m22:
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    return q / np.linalg.norm(q)

def _surface_area_ellipsoid(a, b, c):
    """Surface area of an ellipsoid with semi-axes a, b, c.
    
    Args:
        a, b, c: Can be scalars or numpy arrays of the same shape
        
    Returns:
        Surface area(s) as scalar or array matching input shape
    """
    return a * b + a * c + b * c


@dataclass(frozen=True)
class AggregationMethod:
    sum: ClassVar[str] = "sum"
    dominant: ClassVar[str] = "dominant"
    mean: ClassVar[str] = "mean"
    kl_moment: ClassVar[str] = "kl_moment"  # KL(p||q) first‑/second‑moment match
    kl_moment_hybrid_mean_moment: ClassVar[str] = "kl_moment_hybrid_mean_moment"    # KL(p||q) for 1st moment
    h3dgs: ClassVar[str] = "h3dgs"  # H3DGS aggregation method
    h3dgs_hybrid_mean_moment: ClassVar[str] = "h3dgs_hybrid_mean_moment"            # H3DGS for 1st moment
    w2: ClassVar[str] = "w2"  # Wasserstein-2 distance aggregation method
    w2_h3dgs: ClassVar[str] = "w2_h3dgs"  # Wasserstein-2 distance aggregation method
    scales_voxel_width: ClassVar[str] = "scales_voxel_width"  # Scales voxel width aggregation method
    
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

    # -------------------------------------------------------------------------
    # CRUD helpers -------------------------------------------------------------
    # -------------------------------------------------------------------------

    def add_gaussian(self, gaussian_id):
        self.gaussian_ids.append(gaussian_id)

    # -------------------------------------------------------------------------
    # Baseline aggregation strategies -----------------------------------------
    # (existing methods kept unchanged) ---------------------------------------
    # -------------------------------------------------------------------------

    def agg_attr_sum(self, means, opacities, quats, scales, sh0, shN):
        """Simple weighted sum of attributes based on opacity"""
        # Convert to post-activation space for weighting
        sigmoid_opacities = 1.0 / (1.0 + np.exp(-opacities))
        weights = sigmoid_opacities / np.sum(sigmoid_opacities)
        
        self.aggregated_mean = np.sum(means * weights[:, np.newaxis], axis=0)
        
        # Use max opacity in logit space (most conservative)
        self.aggregated_opacity = np.max(opacities)
        
        self.aggregated_quat = np.sum(quats * weights[:, np.newaxis], axis=0)
        self.aggregated_quat = self.aggregated_quat / np.linalg.norm(self.aggregated_quat)
        
        # Weighted sum of scales in log space (geometric mean)
        self.aggregated_scale = np.sum(scales * weights[:, np.newaxis], axis=0)
        
        self.aggregated_sh0 = np.sum(sh0 * weights[:, np.newaxis, np.newaxis], axis=0)
        self.aggregated_shN = np.sum(shN * weights[:, np.newaxis, np.newaxis], axis=0)

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

    # -------------------------------------------------------------------------
    # New: Moment‑matching aggregation ----------------------------------------
    # Minimise KL(p‖q) between the voxel mixture and *one* Gaussian.
    # -------------------------------------------------------------------------

    def agg_attr_kl_moment(self, means, opacities, quats, scales, sh0, shN):
        """KL(p‖q) projection of the voxel mixture onto *one* Gaussian.

        Keeps 0‑th (mass), 1‑st (centroid) and 2‑nd (covariance) raw moments.
        """
        # --- mixture weights --------------------------------------------------
        w = _sigmoid(opacities)  # convert logit → [0,1]
        W = np.sum(w)
        # if W < 1e-8:
        #     # Fallback to uniform tiny mass to avoid /0
        #     w = np.ones_like(w)
        #     W = float(len(w))
        weights = w / W

        # --- first moment (centroid) -----------------------------------------
        mu_star = np.sum(means * weights[:, None], axis=0)  # (3,)

        # --- second moment (covariance) --------------------------------------
        cov_star = np.zeros((3, 3), dtype=np.float32)
        for wi, mi, qi, si in zip(w, means, quats, scales):
            # Convert to world‑space covariance Σ_i
            Ri = _quat_to_rot(qi)
            sigma = np.exp(si)  # scales are in log‑σ space
            Sigma_i = Ri @ np.diag(sigma ** 2) @ Ri.T  # (3,3)
            diff = (mi - mu_star).astype(np.float32)
            cov_star += wi * (Sigma_i + np.outer(diff, diff))
        cov_star /= W
        
        # print(f"Cov_star: {cov_star}")

        # Numerical regularisation -------------------------------------------
        # covariance is real and symmetric, so eigvecs are real and orthonormal
        # https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix
        eigvals, eigvecs = np.linalg.eigh(cov_star) 
        # eigvals, eigvecs = np.linalg.eig(cov_star)
        
        # Clamp eigenvalues to avoid degeneracy for nearly coincident points.
        # eigvals = np.clip(eigvals, (0.5 * self.voxel_size) ** 2, None)
        
        if np.linalg.det(eigvecs) < 0: # Det of eigvecs should be either -1 or 1
            # print("Det of eigvecs is negative")
            # Determinant of rotation matrix is always 1. Eigen vectors can be scaled. 
            eigvecs[:, 0] *= -1 
            # print(f"Det of eigvecs: {np.linalg.det(eigvecs)}")

        # Derive representative scale & orientation ---------------------------
        sigma_star = np.sqrt(eigvals)
        quat_star = _rot_to_quat(eigvecs)

        # --- colour / SH & opacity -------------------------------------------
        sh0_star = np.sum(sh0 * weights[:, None, None], axis=0)
        shN_star = np.sum(shN * weights[:, None, None], axis=0)

        # Compose into class members -----------------------------------------
        self.aggregated_mean = mu_star.astype(np.float32)
        self.aggregated_quat = quat_star.astype(np.float32)
        self.aggregated_scale = np.log(sigma_star).astype(np.float32)  # back to log‑σ
        self.aggregated_sh0 = sh0_star.astype(np.float32)
        self.aggregated_shN = shN_star.astype(np.float32)

        # Following the original *sum* strategy, stay conservative on opacity
        # by keeping the max logit within the voxel.
        self.aggregated_opacity = np.max(opacities).astype(np.float32)
        
        # Use the total mass of the voxel as the opacity
        # self.aggregated_opacity = _logit(W).astype(np.float32)
        
    def agg_attr_kl_moment_hybrid_mean_moment(self, means, opacities, quats, scales, sh0, shN):
        """KL(p‖q) projection of the voxel mixture onto *one* Gaussian.

        Keeps 0‑th (mass), 1‑st (centroid) and 2‑nd (covariance) raw moments.
        """
        # --- mixture weights --------------------------------------------------
        w = _sigmoid(opacities)  # convert logit → [0,1]
        W = np.sum(w)
        # assert W > 1e-8, "Total mass of the voxel is 0"
        weights = w / W
        dominant_idx = np.argmax(opacities)

        # --- first moment (centroid) -----------------------------------------
        mu_star = np.sum(means * weights[:, None], axis=0)  # (3,)

        # --- colour / SH & opacity -------------------------------------------
        sh0_star = np.sum(sh0 * weights[:, None, None], axis=0)
        shN_star = np.sum(shN * weights[:, None, None], axis=0)

        # Compose into class members -----------------------------------------
        self.aggregated_mean = mu_star.astype(np.float32)
        self.aggregated_quat = quats[dominant_idx].astype(np.float32)
        self.aggregated_scale = scales[dominant_idx].astype(np.float32)
        self.aggregated_opacity = opacities[dominant_idx].astype(np.float32)
        self.aggregated_sh0 = sh0_star.astype(np.float32)
        self.aggregated_shN = shN_star.astype(np.float32)
        
        # Use the total mass of the voxel as the opacity
        # self.aggregated_opacity = _logit(W).astype(np.float32)

    def agg_attr_h3dgs(self, means, opacities, quats, scales, sh0, shN):
        """H3DGS aggregation method
        https://arxiv.org/abs/2406.12080
        """
        
        # --- mixture weights --------------------------------------------------
        real_opacities = _sigmoid(opacities)    # convert logit → [0,1]
        real_scales = np.exp(scales)            # scales are in log‑σ space
        
        # Calculate weights for each Gaussian (vectorized)
        surface_areas = _surface_area_ellipsoid(real_scales[:, 0], real_scales[:, 1], real_scales[:, 2])
        w = real_opacities * surface_areas
        W = np.sum(w)
        weights = w / W
        
        # --- first moment (centroid) -----------------------------------------
        mu_star = np.sum(means * weights[:, None], axis=0)  # (3,)
        
        # --- second moment (covariance) --------------------------------------
        cov_star = np.zeros((3, 3), dtype=np.float32)
        for wi, mi, qi, si in zip(weights, means, quats, real_scales):
            # Convert to world‑space covariance Σ_i
            Ri = _quat_to_rot(qi)
            Sigma_i = Ri @ np.diag(si ** 2) @ Ri.T  # (3,3)
            diff = (mi - mu_star).astype(np.float32)
            cov_star += wi * (Sigma_i + np.outer(diff, diff))
            
        eigvals, eigvecs = np.linalg.eigh(cov_star)
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 0] *= -1
        sigma_star = np.sqrt(eigvals)
        quat_star = _rot_to_quat(eigvecs)
        
        # --- colour / SH & opacity -------------------------------------------
        sh0_star = np.sum(sh0 * weights[:, None, None], axis=0)
        shN_star = np.sum(shN * weights[:, None, None], axis=0)
        opacity_star = W / _surface_area_ellipsoid(sigma_star[0], sigma_star[1], sigma_star[2])
        
        # Compose into class members -----------------------------------------
        self.aggregated_mean = mu_star.astype(np.float32)
        self.aggregated_quat = quat_star.astype(np.float32)
        self.aggregated_scale = np.log(sigma_star).astype(np.float32)  # back to log‑σ
        self.aggregated_opacity = _logit(opacity_star).astype(np.float32)
        self.aggregated_sh0 = sh0_star.astype(np.float32)
        self.aggregated_shN = shN_star.astype(np.float32)
        
    def agg_attr_h3dgs_hybrid_mean_moment(self, means, opacities, quats, scales, sh0, shN):
        """H3DGS aggregation method
        https://arxiv.org/abs/2406.12080
        """
        
        # --- mixture weights --------------------------------------------------
        real_opacities = _sigmoid(opacities)    # convert logit → [0,1]
        real_scales = np.exp(scales)            # scales are in log‑σ space
        
        # Calculate weights for each Gaussian (vectorized)
        surface_areas = _surface_area_ellipsoid(real_scales[:, 0], real_scales[:, 1], real_scales[:, 2])
        w = real_opacities * surface_areas
        W = np.sum(w)
        # assert W > 1e-8, "Total mass of the voxel is 0"
        weights = w / W
        dominant_idx = np.argmax(opacities)
        
        # --- first moment (centroid) -----------------------------------------
        mu_star = np.sum(means * weights[:, None], axis=0)  # (3,)
        
        # --- colour / SH & opacity -------------------------------------------
        sh0_star = np.sum(sh0 * weights[:, None, None], axis=0)
        shN_star = np.sum(shN * weights[:, None, None], axis=0)
        
        # Compose into class members -----------------------------------------
        self.aggregated_mean = mu_star.astype(np.float32)
        self.aggregated_quat = quats[dominant_idx].astype(np.float32)
        self.aggregated_scale = scales[dominant_idx].astype(np.float32)
        self.aggregated_opacity = opacities[dominant_idx].astype(np.float32)
        self.aggregated_sh0 = sh0_star.astype(np.float32)
        self.aggregated_shN = shN_star.astype(np.float32)
        
    def agg_attr_w2(self, means, opacities, quats, scales, sh0, shN):
        """Wasserstein-2 distance aggregation method using barycenter computation.
        
        Uses the Wasserstein-2 distance to find the optimal representative Gaussian
        that minimizes the average W2 distance to all Gaussians in the voxel.
        """
        # Convert log-opacities to weights
        w = _sigmoid(opacities)  # convert logit → [0,1]
        W = np.sum(w)
        weights = w / W

        # Convert quaternions and scales to covariance matrices
        covariances_list = []
        means_list = []
        for mi, qi, si in zip(means, quats, scales):
            Ri = _quat_to_rot(qi)
            sigma = np.exp(si)  # scales are in log‑σ space
            Sigma_i = Ri @ np.diag(sigma ** 2) @ Ri.T  # (3,3)
            covariances_list.append(Sigma_i.astype(np.float64))
            means_list.append(mi.astype(np.float64))
        
        # Compute Wasserstein barycenter
        bary_mean, bary_cov = wasserstein_barycenter_gaussians_orig(means_list, covariances_list, weights)
        bary_mean = bary_mean
        bary_cov = bary_cov
        
        # Convert barycenter covariance back to quaternion and scale
        eigvals, eigvecs = np.linalg.eigh(bary_cov)
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 0] *= -1
            
        # Check for negative eigenvalues
        if np.any(eigvals < 0):
            print(f"Warning: Negative eigenvalues encountered: {eigvals}")
            print(f"Barycenter covariance matrix:\n{bary_cov}")
            # Clip negative eigenvalues to small positive value
            eigvals = np.clip(eigvals, 1e-8, None)
            
        sigma_star = np.sqrt(eigvals)
        quat_star = _rot_to_quat(eigvecs)
        
        # Compute weighted average of SH coefficients
        sh0_star = np.sum(sh0 * weights[:, None, None], axis=0)
        shN_star = np.sum(shN * weights[:, None, None], axis=0)
        
        # Store results
        self.aggregated_mean = bary_mean.astype(np.float32)
        self.aggregated_quat = quat_star.astype(np.float32)
        self.aggregated_scale = np.log(sigma_star).astype(np.float32)  # back to log‑σ
        self.aggregated_sh0 = sh0_star.astype(np.float32)
        self.aggregated_shN = shN_star.astype(np.float32)
        
        # Use max opacity in logit space (most conservative)
        self.aggregated_opacity = np.max(opacities).astype(np.float32)
        
        # Use the total mass of the voxel as the opacity
        # self.aggregated_opacity = _logit(W).astype(np.float32)
    
    def agg_attr_w2_h3dgs(self, means, opacities, quats, scales, sh0, shN):
        """Wasserstein-2 distance aggregation method using barycenter computation.
        
        Uses the Wasserstein-2 distance to find the optimal representative Gaussian
        that minimizes the average W2 distance to all Gaussians in the voxel.
        """
        # --- mixture weights --------------------------------------------------
        real_opacities = _sigmoid(opacities)    # convert logit → [0,1]
        real_scales = np.exp(scales)            # scales are in log‑σ space
        
        # Calculate weights for each Gaussian (vectorized)
        surface_areas = _surface_area_ellipsoid(real_scales[:, 0], real_scales[:, 1], real_scales[:, 2])
        w = real_opacities * surface_areas
        W = np.sum(w)
        weights = w / W

        # Convert quaternions and scales to covariance matrices
        covariances_list = []
        means_list = []
        for mi, qi, si in zip(means, quats, scales):
            Ri = _quat_to_rot(qi)
            sigma = np.exp(si)  # scales are in log‑σ space
            Sigma_i = Ri @ np.diag(sigma ** 2) @ Ri.T  # (3,3)
            covariances_list.append(Sigma_i.astype(np.float64))
            means_list.append(mi.astype(np.float64))
        
        # Compute Wasserstein barycenter
        bary_mean, bary_cov = wasserstein_barycenter_gaussians_orig(means_list, covariances_list, weights)
        bary_mean = bary_mean
        bary_cov = bary_cov
        
        # Convert barycenter covariance back to quaternion and scale
        eigvals, eigvecs = np.linalg.eigh(bary_cov)
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 0] *= -1
            
        # Check for negative eigenvalues
        if np.any(eigvals < 0):
            print(f"Warning: Negative eigenvalues encountered: {eigvals}")
            print(f"Barycenter covariance matrix:\n{bary_cov}")
            # Clip negative eigenvalues to small positive value
            eigvals = np.clip(eigvals, 1e-8, None)
            
        sigma_star = np.sqrt(eigvals)
        quat_star = _rot_to_quat(eigvecs)
        
        # Compute weighted average of SH coefficients
        sh0_star = np.sum(sh0 * weights[:, None, None], axis=0)
        shN_star = np.sum(shN * weights[:, None, None], axis=0)
        opacity_star = W / _surface_area_ellipsoid(sigma_star[0], sigma_star[1], sigma_star[2])
        
        # Store results
        self.aggregated_mean = bary_mean.astype(np.float32)
        self.aggregated_quat = quat_star.astype(np.float32)
        self.aggregated_scale = np.log(sigma_star).astype(np.float32)  # back to log‑σ
        self.aggregated_sh0 = sh0_star.astype(np.float32)
        self.aggregated_shN = shN_star.astype(np.float32)
        
        # Use max opacity in logit space (most conservative)
        # self.aggregated_opacity = np.max(opacities).astype(np.float32)
        self.aggregated_opacity = _logit(opacity_star).astype(np.float32)
        
        # Use the total mass of the voxel as the opacity
        # self.aggregated_opacity = _logit(W).astype(np.float32)

    def agg_attr_scales_voxel_width(self, means, opacities, quats, scales, sh0, shN):
        """Scales voxel width aggregation method.
        
        Uses the voxel width to define the scales while keeping other
        attributes (mean, opacity, SH coefficients) from the dominant Gaussian.
        """
        # Find dominant Gaussian based on opacity
        dominant_idx = np.argmax(opacities)
        
        # Use half voxel width as standard deviation to properly cover the voxel
        sigma_star = np.array([self.voxel_size/2.0, self.voxel_size/2.0, self.voxel_size/2.0])
        
        # Store results
        self.aggregated_mean = means[dominant_idx].astype(np.float32)
        self.aggregated_opacity = opacities[dominant_idx].astype(np.float32)
        self.aggregated_quat = quats[dominant_idx].astype(np.float32)
        self.aggregated_scale = np.log(sigma_star).astype(np.float32)  # back to log‑σ
        self.aggregated_sh0 = sh0[dominant_idx].astype(np.float32)
        self.aggregated_shN = shN[dominant_idx].astype(np.float32)

    def aggregate_attributes(self, trained_model, aggregate_method=AggregationMethod.sum):
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
        if aggregate_method == AggregationMethod.sum:
            method = self.agg_attr_sum
        elif aggregate_method == AggregationMethod.dominant:
            method = self.agg_attr_dominant
        elif aggregate_method == AggregationMethod.mean:
            method = self.agg_attr_mean
        elif aggregate_method == AggregationMethod.kl_moment:
            method = self.agg_attr_kl_moment
        elif aggregate_method == AggregationMethod.h3dgs:
            method = self.agg_attr_h3dgs
        elif aggregate_method == AggregationMethod.kl_moment_hybrid_mean_moment:
            method = self.agg_attr_kl_moment_hybrid_mean_moment
        elif aggregate_method == AggregationMethod.h3dgs_hybrid_mean_moment:
            method = self.agg_attr_h3dgs_hybrid_mean_moment
        elif aggregate_method == AggregationMethod.w2:
            method = self.agg_attr_w2
        elif aggregate_method == AggregationMethod.w2_h3dgs:
            method = self.agg_attr_w2_h3dgs
        elif aggregate_method == AggregationMethod.scales_voxel_width:
            method = self.agg_attr_scales_voxel_width
        else:
            raise ValueError(f"Invalid aggregate method: {aggregate_method}")
        
        method(means, opacities, quats, scales, sh0, shN)