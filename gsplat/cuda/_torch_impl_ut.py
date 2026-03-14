# SPDX-FileCopyrightText: Copyright 2025-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch reference implementation for Unscented Transform (UT) projection.

This module contains the reference implementation of fully_fused_projection_with_ut
that uses the Unscented Transform to handle non-linear camera projections (distortion,
fisheye, rolling shutter, etc.).

The Unscented Transform approximates the projection of a Gaussian through a non-linear
transformation by:
1. Generating sigma points from the input Gaussian
2. Transforming each sigma point through the non-linear function
3. Computing the mean and covariance of the transformed points

References:
- "The unscented Kalman filter for nonlinear estimation" - Wan and van der Merwe 2000
- "Some Relations Between Extended and Unscented Kalman Filters" - Gustafsson and Hendeby 2012
"""

import math
from typing import Literal, Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat._helper import assert_shape

from ._wrapper import (
    CameraModel,
    RollingShutterType,
    UnscentedTransformParameters,
    FThetaCameraDistortionParameters,
)
from ._math import (
    _quat_to_rotmat,
    _rotmat_to_quat,
    _quat_inverse,
    _safe_normalize,
)
from ._torch_cameras import (
    _viewmat_to_pose,
    _BaseCameraModel,
    _interpolate_shutter_pose,
)

from ._torch_lidars import (
    _RowOffsetStructuredSpinningLidarModel,
)
from ._lidar import (
    RowOffsetStructuredSpinningLidarModelParametersExt,
)


def _compute_ut_weights(
    ut_params: UnscentedTransformParameters,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    """Compute Unscented Transform weights for sigma points.

    For a 3D Gaussian, we generate 2*D+1 = 7 sigma points.

    Args:
        ut_params: UT parameters (alpha, beta, kappa)
        device: Device for output tensors
        dtype: Data type for output tensors

    Returns:
        weights_mean: [7] weights for mean computation
        weights_cov: [7] weights for covariance computation
    """
    D = 3  # Dimensionality of input space
    alpha = ut_params.alpha
    beta = ut_params.beta
    kappa = ut_params.kappa

    # Compute lambda parameter
    lambda_ = alpha * alpha * (D + kappa) - D

    # Compute weights (autograd-friendly construction)
    weight_center_mean = lambda_ / (D + lambda_)
    weight_center_cov = lambda_ / (D + lambda_) + (1.0 - alpha * alpha + beta)
    weight_others = 1.0 / (2.0 * (D + lambda_))

    # Build weight tensors (7 elements: 1 center + 6 others)
    weights_mean = torch.tensor(
        [weight_center_mean] + [weight_others] * 6, device=device, dtype=dtype
    )
    weights_cov = torch.tensor(
        [weight_center_cov] + [weight_others] * 6, device=device, dtype=dtype
    )

    return weights_mean, weights_cov


def _world_gaussian_sigma_points(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    ut_params: UnscentedTransformParameters,
) -> Tensor:
    """Generate sigma points for 3D Gaussians.

    For each Gaussian, generate 7 sigma points:
    - Point 0: mean
    - Points 1-3: mean + √(D+λ) * scale[i] * R[:, i]  (i=0,1,2)
    - Points 4-6: mean - √(D+λ) * scale[i] * R[:, i]

    where R = quat_to_rotmat(quat), D=3, λ = α²(D+κ) - D

    Args:
        means: Gaussian centers [..., N, 3]
        quats: Gaussian rotations [..., N, 4] (w, x, y, z)
        scales: Gaussian scales [..., N, 3]
        ut_params: UT parameters

    Returns:
        sigma_points: [..., N, 7, 3] - 7 sigma points per Gaussian
    """
    D = 3
    alpha = ut_params.alpha
    kappa = ut_params.kappa
    lambda_ = alpha * alpha * (D + kappa) - D

    # Normalize quaternions and convert to rotation matrix
    quats_norm = _safe_normalize(quats, dim=-1)
    # R transforms from Gaussian's local frame to world frame
    R = _quat_to_rotmat(quats_norm)  # [..., N, 3, 3] - local-to-world rotation

    # Vectorized sigma point construction (autograd-friendly, no loops)
    # Scale each column of R by corresponding scale value:
    #   scales.unsqueeze(-2): [..., N, 1, 3] broadcasts with R: [..., N, 3, 3]
    #   Element-wise: result[:, i, j] = R[:, i, j] * scales[:, j]
    #   This gives: column j = R[:, j] * scales[j] (j-th basis vector scaled)
    #   Result: 3x3 matrix with 3 scaled basis vectors as columns
    # Then transpose to get deltas as rows for easier concatenation
    deltas = (
        math.sqrt(D + lambda_) * R * scales.unsqueeze(-2)
    )  # [..., N, 3, 3], 3 deltas as columns
    deltas = deltas.transpose(-2, -1)  # [..., N, 3, 3], 3 deltas as rows

    # Build sigma points: [center, +deltas, -deltas]
    means_expanded = means.unsqueeze(-2)  # [..., N, 1, 3]

    # Concatenate: point 0 (center), points 1-3 (+deltas), points 4-6 (-deltas)
    sigma_points_world = torch.cat(
        [
            means_expanded,  # [..., N, 1, 3] - point 0
            means_expanded + deltas,  # [..., N, 3, 3] - points 1-3
            means_expanded - deltas,  # [..., N, 3, 3] - points 4-6
        ],
        dim=-2,
    )  # [..., N, 7, 3]

    return sigma_points_world


def _world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
    camera: _BaseCameraModel,
    means: Tensor,  # [B, N, 3]
    quats: Tensor,  # [B, N, 4]
    scales: Tensor,  # [B, N, 3]
    pose_start: Tensor,  # [B, C, 7]
    pose_end: Tensor,  # [B, C, 7]
    ut_params: UnscentedTransformParameters,
) -> Tuple[Tensor, Tensor]:
    """Project a 3D Gaussian to 2D using the Unscented Transform to handle non-linear
    camera models (distortion, fisheye, rolling shutter)."""
    B = means.shape[:-2]
    N = means.shape[-2]
    C = pose_start.shape[-2]
    assert_shape("means", means, B + (N, 3))
    assert_shape("quats", quats, B + (N, 4))
    assert_shape("scales", scales, B + (N, 3))
    assert_shape("pose_start", pose_start, B + (C, 7))
    assert_shape("pose_end", pose_end, B + (C, 7))

    dtype = means.dtype
    device = means.device

    # Generate sigma points for the Gaussian
    sigma_points_world = _world_gaussian_sigma_points(means, quats, scales, ut_params)

    # Compute UT weights (shared for all Gaussians)
    weights_mean, weights_cov = _compute_ut_weights(ut_params, device, dtype)

    # Project sigma points using camera model

    # Expand sigma points for each camera: [..., N, 7, 3] -> [..., C, N, 7, 3]
    sigma_points_world_exp = sigma_points_world.unsqueeze(-4).expand(B + (C, N, 7, 3))

    # Flatten only N and sigma points, keep C separate: [B, C, N, 7, 3] -> [B, C, N*7, 3]
    # Camera model is batched over C, so each camera processes its own N*7 points
    sigma_points_world_flat = sigma_points_world_exp.reshape(B + (C, N * 7, 3))

    # Project using camera model.
    points_2d_flat, valid_flat = camera.world_point_to_image_point_shutter_pose(
        world_points=sigma_points_world_flat,  # [..., C, N*7, 3] world points per camera
        shutter_pose_start=pose_start,  # [..., C, 7]
        shutter_pose_end=pose_end,  # [..., C, 7]
        margin_factor=ut_params.in_image_margin_factor,
    )

    # Reshape back: [B, C, N*7, 2] -> [B, C, N, 7, 2]
    points_2d = points_2d_flat.reshape(B + (C, N, 7, 2))
    valid_points = valid_flat.reshape(B + (C, N, 7))

    # Compute weighted mean and covariance for each Gaussian

    if ut_params.require_all_sigma_points_valid:
        # CUDA early-exits on first invalid point, using partial sum
        # Simulate this with cumulative validity mask
        # cumulative_valid[..., i] = True if points 0..i are ALL valid
        cumulative_valid = torch.cumprod(valid_points.to(torch.float32), dim=-1).to(
            torch.bool
        )

        # All sigma points must be valid for Gaussian to be valid
        assert_shape("cumulative_valid", cumulative_valid, B + (C, N, 7))
        valid_gaussian = cumulative_valid[..., -1]

        # Mask weights: only use points up to first invalid
        # effective_mask: [B, C, N, 7]
        effective_mask = cumulative_valid.to(dtype)

        # Broadcast weights with mask: weights_mean [7] -> [B, C, N, 7]
        # Use proper broadcasting: expand weights_mean to match effective_mask shape
        weights_mean_expanded = weights_mean.view(
            [1] * len(B) + [1, 1, 7]
        )  # [B, 1, 1, 7]
        weights_mean_eff = weights_mean_expanded * effective_mask  # [B, C, N, 7]

        weights_cov_expanded = weights_cov.view([1] * len(B) + [1, 1, 7])
        weights_cov_eff = weights_cov_expanded * effective_mask  # [B, C, N, 7]

        # Compute weighted mean: mean_2d = Σ w_mean[i] * points_2d[i]
        # Use masked weights (early-exit simulation)
        # weights_mean_eff: [B, C, N, 7], points_2d: [B, C, N, 7, 2]
        mean_2d = torch.sum(
            weights_mean_eff[..., None] * points_2d, dim=-2
        )  # [B, C, N, 2]
    else:
        # At least one sigma point must be valid - use all points
        valid_gaussian = torch.any(valid_points, dim=-1)  # [B, C, N]
        # Use all points
        mean_2d = torch.einsum(
            "i,...nij->...nj", weights_mean, points_2d
        )  # [B, C, N, 2]

    # Compute weighted covariance
    delta_2d = points_2d - mean_2d.unsqueeze(-2)  # [B, C, N, 7, 2]
    outer_products = torch.einsum(
        "...i,...j->...ij", delta_2d, delta_2d
    )  # [B, C, N, 7, 2, 2]

    if ut_params.require_all_sigma_points_valid:
        # Use masked weights: [B, C, N, 7] with [B, C, N, 7, 2, 2]
        cov_2d = torch.sum(
            weights_cov_eff[..., None, None] * outer_products, dim=-3
        )  # [..., C, N, 2, 2]
    else:
        cov_2d = torch.einsum("i,...nijk->...njk", weights_cov, outer_products)

    return mean_2d, cov_2d, valid_gaussian


def _add_blur(
    cov_2d: Tensor,  # [B, C, N, 2, 2]
    eps2d: float,
) -> Tensor:
    B = cov_2d.shape[:-4]
    C = cov_2d.shape[-4]
    N = cov_2d.shape[-3]
    assert_shape("cov_2d", cov_2d, B + (C, N, 2, 2))

    """Add eps2d to the covariance matrix for numerical stability."""
    det_orig = torch.linalg.det(cov_2d)  # [B, C, N]

    cov_2d = cov_2d + eps2d * torch.eye(2, dtype=cov_2d.dtype, device=cov_2d.device)
    det_blur = torch.linalg.det(cov_2d)
    compensation = torch.sqrt(torch.clamp(det_orig / det_blur, min=0.0))  # [B, C, N]

    assert_shape("det_blur", det_blur, B + (C, N))
    assert_shape("cov_2d", cov_2d, B + (C, N, 2, 2))
    assert_shape("compensation", compensation, B + (C, N))
    return det_blur, cov_2d, compensation


def _fully_fused_projection_with_ut(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Optional[Tensor],  # [..., N]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    calc_compensations: bool = False,
    camera_model: CameraModel = "pinhole",
    ut_params: Optional[UnscentedTransformParameters] = None,
    radial_coeffs: Optional[Tensor] = None,
    tangential_coeffs: Optional[Tensor] = None,
    thin_prism_coeffs: Optional[Tensor] = None,
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
    global_z_order: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """PyTorch reference implementation of fully_fused_projection_with_ut().

    Projects 3D Gaussians to 2D using the Unscented Transform to handle non-linear
    camera models (distortion, fisheye, rolling shutter).

    .. note::
        This is a reference implementation for testing purposes. The CUDA implementation
        is much faster.

    .. note::
        Currently supports:
        - Pinhole camera model
        - Radial distortion
        - Rolling shutter

    Args:
        means: Gaussian centers [..., N, 3]
        quats: Gaussian rotations [..., N, 4] (w, x, y, z)
        scales: Gaussian scales [..., N, 3]
        opacities: Gaussian opacities [..., N] (optional, used for tighter bounds)
        viewmats: Start camera view matrices [..., C, 4, 4]
        Ks: Camera intrinsics [..., C, 3, 3]
        width: Image width
        height: Image height
        eps2d: Epsilon added to 2D covariance for numerical stability
        near_plane: Near plane distance
        far_plane: Far plane distance
        radius_clip: Gaussians with projected radii smaller than this are culled
        calc_compensations: If True, compute opacity compensation
        camera_model: Camera model ("pinhole", "fisheye", "ftheta" - ortho not supported in UT)
        ut_params: Unscented Transform parameters
        radial_coeffs: [..., C, 4] or [..., C, 6] radial distortion coefficients (pinhole/fisheye)
        tangential_coeffs: [..., C, 2] tangential distortion coefficients (pinhole only)
        thin_prism_coeffs: [..., C, 4] thin prism distortion coefficients (pinhole only)
        ftheta_coeffs: F-theta camera parameters
        rolling_shutter: Rolling shutter type
        viewmats_rs: End camera view matrices for rolling shutter [..., C, 4, 4]

    Returns:
        radii: Projected radii [..., C, N, 2] (radius_x, radius_y)
        means2d: Projected 2D means [..., C, N, 2]
        depths: Depths (z-coordinate in camera space) [..., C, N]
        conics: Inverse covariances (conics) [..., C, N, 3] (xx, xy, yy)
        compensations: Opacity compensation factors [..., C, N] (or None)
    """
    if ut_params is None:
        ut_params = UnscentedTransformParameters()

    # Validate inputs
    B = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]

    assert_shape("means", means, B + (N, 3))
    assert_shape("quats", quats, B + (N, 4))
    assert_shape("scales", scales, B + (N, 3))
    assert_shape("viewmats", viewmats, B + (C, 4, 4))
    assert_shape("Ks", Ks, B + (C, 3, 3))

    device = means.device
    dtype = means.dtype

    assert (
        dtype == torch.float32
    ), f"CUDA uses float32, but got {dtype}. This will cause large divergences!"
    assert (
        viewmats.dtype == torch.float32
    ), f"viewmats must be float32, got {viewmats.dtype}"
    assert Ks.dtype == torch.float32, f"Ks must be float32, got {Ks.dtype}"

    # Validate camera model support
    if camera_model not in ["pinhole", "fisheye", "ftheta", "lidar"]:
        raise ValueError(
            f"Camera model '{camera_model}' not supported in UT projection. "
            f"UT supports: pinhole, fisheye, ftheta. "
            f"For ortho, use non-UT projection (with_ut=False)."
        )

    # Extract focal lengths and principal points from K matrix
    # K is [B, C, 3, 3] with structure [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    focal_lengths = torch.stack(
        [
            Ks[..., 0, 0],  # fx
            Ks[..., 1, 1],  # fy
        ],
        dim=-1,
    )  # [B, C, 2]
    principal_points = Ks[..., :2, 2]  # [B, C, 2] - extract [cx, cy]

    # Create camera model
    if camera_model == "lidar":
        camera = _RowOffsetStructuredSpinningLidarModel(lidar_coeffs)
    else:
        assert lidar_coeffs is None
        camera = _BaseCameraModel.create(
            width=width,
            height=height,
            camera_model=camera_model,
            principal_points=principal_points,
            focal_lengths=focal_lengths,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            ftheta_coeffs=ftheta_coeffs,
            rs_type=rolling_shutter,
        )

    # Create pose tensors for rolling shutter
    pose_start = _viewmat_to_pose(viewmats)  # [B, C, 7]
    if rolling_shutter != RollingShutterType.GLOBAL:
        pose_end = _viewmat_to_pose(viewmats_rs)
    else:
        pose_end = pose_start  # Same pose for global shutter

    if rolling_shutter == RollingShutterType.GLOBAL:
        R_cam = viewmats[..., :3, :3]
        t_cam = viewmats[..., :3, 3]
    else:
        # Interpolate at t=0.5 for Gaussian center
        relative_time = torch.full(B + (C,), 0.5, device=device, dtype=dtype)
        pose_interp = _interpolate_shutter_pose(pose_start, pose_end, relative_time)

        # Extract world-to-camera transform from interpolated pose
        R_cam = _quat_to_rotmat(pose_interp[..., 3:])  # [B, C, 3, 3]
        t_cam = pose_interp[..., :3]  # [..., C, 3]

    # Transform ONLY Gaussian centers to camera space for frustum culling
    # Full sigma points are transformed inside projection function.

    # means: [B, N, 3], R_cam: [B, C, 3, 3], t_cam: [B, C, 3]
    means_cam = (
        torch.einsum("...cij,...nj->...cni", R_cam, means)  # [B, C, 3, 3]  # [B, N, 3]
        + t_cam[..., None, :]
    )  # [B, C, N, 3]

    # Check if Gaussian center is within frustum.
    # Use transformed center point (means_cam) for depth check
    center_z = means_cam[..., 2]  # [B, C, N]
    in_frustum = (center_z >= near_plane) & (center_z <= far_plane)

    # Projection using unscented transform
    (
        mean_2d,
        cov_2d,
        valid_gaussian,
    ) = _world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
        camera=camera,
        means=means,
        quats=quats,
        scales=scales,
        pose_start=pose_start,
        pose_end=pose_end,
        ut_params=ut_params,
    )

    # Combine with frustum check
    valid_gaussian = valid_gaussian & in_frustum

    (
        det,  # [B, C, N]
        cov_2d,  # [B, C, N, 2, 2]
        compensations,  # [B, C, N]
    ) = _add_blur(cov_2d, eps2d)

    valid_gaussian = valid_gaussian & (det > 0.0)

    # Compute conics (inverse of 2D covariance)
    # This is more robust than manual formula, especially for non-symmetric matrices
    # (numerical errors in weighted sum can break exact symmetry)
    cov_2d_inv = torch.linalg.inv(cov_2d)  # [B, C, N, 2, 2]

    # Apply opacity-based culling
    # Reference: https://arxiv.org/pdf/2402.00525 Section B.2
    ALPHA_THRESHOLD: float = float(1.0) / float(255.0)  # Minimum visible opacity

    # Default extend
    extend = torch.full(B + (C, N), 3.33, dtype=dtype, device=device)

    if opacities is not None:
        # Apply compensation to get effective opacity
        opacity = opacities[..., None, :] * compensations  # [B, C, N]

        # Discard Gaussians that are too transparent
        valid_gaussian = valid_gaussian & (opacity >= ALPHA_THRESHOLD)

        # Compute opacity-aware extent and radii with eigenvalue-based tight bounding box
        # Reference: https://arxiv.org/pdf/2402.00525 Section B.2
        extend = torch.minimum(
            extend,
            # Clamp to avoid sqrt(negative).
            # Opacities < ALPHA_THRESHOLD are discarded already, no harm done.
            torch.sqrt(
                2.0 * torch.log(torch.clamp(opacity / ALPHA_THRESHOLD, min=1.0))
            ),
        )  # [B, C, N]

    # Compute radii with eigenvalue-based tight bounding box
    # Compute larger eigenvalue: λ₁ = (trace + sqrt(max(0.01, trace² - 4*det))) / 2
    cov_diag = torch.diagonal(cov_2d, dim1=-2, dim2=-1)  # [B, C, N, 2]
    trace = cov_diag.sum(dim=-1)  # [B, C, N]
    b = 0.5 * trace
    tmp = torch.sqrt(torch.clamp(b * b - det, min=0.01))
    v1 = b + tmp

    # Radius bound: r_i = min(extend * sqrt(cov[i][i]), extend * sqrt(λ_max))
    r1 = extend * torch.sqrt(v1)  # [B, C, N]

    # Compute radii for both x and y axes
    radius = torch.ceil(
        torch.minimum(extend[..., None] * torch.sqrt(cov_diag), r1[..., None])
    )  # [B, C, N, 2]

    # Apply radius clipping and image bounds culling

    # Radius clipping: cull sub-pixel Gaussians
    # If both x and y radii <= radius_clip, the Gaussian is culled
    valid_gaussian = valid_gaussian & (torch.max(radius, dim=-1)[0] > radius_clip)

    # Image bounds culling: cull Gaussians outside image
    if camera_model == "lidar":
        # Culling against fov was already done above.
        pass
    else:
        # Check if bounding box overlaps with image: (center ± radius) overlaps [0, width/height)
        image_bounds = torch.tensor(
            [width, height], dtype=radius.dtype, device=radius.device
        )
        in_image = torch.all(
            (mean_2d + radius > 0) & (mean_2d - radius < image_bounds), dim=-1
        )
        valid_gaussian = valid_gaussian & in_image

    # Set outputs for valid Gaussians only
    # For invalid Gaussians, radii should be 0 (default)
    radii_computed = radius.to(torch.int32)  # [B, C, N, 2]
    radii = torch.where(
        valid_gaussian[..., None], radii_computed, torch.zeros_like(radii_computed)
    )

    means2d = torch.where(valid_gaussian[..., None], mean_2d, torch.zeros_like(mean_2d))

    if global_z_order:
        depth = center_z
    else:
        depth = torch.norm(means_cam, dim=-1)  # [B, C, N]

    depths = torch.where(valid_gaussian, depth, torch.zeros_like(depth))

    # Extract conics as [xx, xy, yy] (symmetric representation)
    conics_computed = torch.stack(
        [
            cov_2d_inv[..., 0, 0],  # xx
            cov_2d_inv[..., 0, 1],  # xy
            cov_2d_inv[..., 1, 1],  # yy
        ],
        dim=-1,
    )  # [B, C, N, 3]
    conics = torch.where(
        valid_gaussian[..., None], conics_computed, torch.zeros_like(conics_computed)
    )

    # Mask compensation output if requested, otherwise return zeros
    if calc_compensations:
        compensations = torch.where(
            valid_gaussian, compensations, torch.zeros_like(compensations)
        )
    else:
        compensations = None

    return radii, means2d, depths, conics, compensations
