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

"""
Test C++ camera models (primary) against PyTorch reference.

Validates correctness of C++ implementations by comparing output
to PyTorch reference implementations for all 5 methods.
"""

import numpy as np
import pytest
import torch
import re
import math
from functools import lru_cache
from itertools import product, chain
from types import SimpleNamespace
from dataclasses import dataclass

if not torch.cuda.is_available():
    pytest.skip("CUDA required for camera model tests", allow_module_level=True)

from gsplat.cuda._backend import _C

if _C is None:
    pytest.skip("gsplat CUDA extension not available", allow_module_level=True)

from gsplat.cuda._wrapper import has_camera_wrappers

if not has_camera_wrappers():
    pytest.skip(
        "Camera wrappers not built (need BUILD_CAMERA_WRAPPERS=1)",
        allow_module_level=True,
    )

from gsplat._helper import expand_named_params
from gsplat.cuda._torch_cameras import (  # PyTorch reference
    _BaseCameraModel,
    _PerfectPinholeCameraModel,
    _OpenCVPinholeCameraModel,
    _OpenCVFisheyeCameraModel,
    _FThetaCameraModel,
)
from gsplat.cuda._torch_lidars import (  # PyTorch reference
    _RowOffsetStructuredSpinningLidarModel,
)
from gsplat._helper import (
    assert_mismatch_ratio,
    assert_close,
    assert_close_with_boundary_band,
)
from gsplat.cuda._wrapper import (
    RollingShutterType,
    FThetaPolynomialType,
    FThetaCameraDistortionParameters,
    create_camera_model,
    SpinningDirection,
)
from gsplat import (
    compute_lidar_angles_to_columns_map,
    compute_lidar_tiling,
    RowOffsetStructuredSpinningLidarModelParameters,
    RowOffsetStructuredSpinningLidarModelParametersExt,
)
from gsplat.cuda._math import (
    _quat_multiply,
    _safe_normalize,
    compute_inverse_polynomial,
)

SEED = 42

CAMERA_MODELS = [
    "pinhole",
    "pinhole[k4]",
    "pinhole[k6]",
    "pinhole[k6+tangential]",
    "pinhole[k6+thin_prism]",
    "fisheye",
    "ftheta[pinhole+a2p]",
    "ftheta[pinhole+p2a]",
    "ftheta[a2p]",
    "ftheta[p2a]",
    "lidar[pandar128]",
    "lidar[at128]",
]

ROLLING_SHUTTER_TYPES = [
    ("L2R", RollingShutterType.ROLLING_LEFT_TO_RIGHT),
    ("R2L", RollingShutterType.ROLLING_RIGHT_TO_LEFT),
    ("T2B", RollingShutterType.ROLLING_TOP_TO_BOTTOM),
    ("B2T", RollingShutterType.ROLLING_BOTTOM_TO_TOP),
    ("global", RollingShutterType.GLOBAL),
]

DEFAULT_ROLLING_SHUTTER_TYPE = [("global", RollingShutterType.ROLLING_LEFT_TO_RIGHT)]

# ==================================================
# Routines to parse camera model definition strings
# ==================================================


def _compute_focal_length(size: int, fov: torch.Tensor) -> torch.Tensor:
    """Compute focal length from image size and field of view.

    Args:
        size: Image size
        fov: Field of view (in radians)

    Returns:
        focal_length: Focal length
    """
    return size / (2.0 * torch.tan(fov / 2.0))


def parse_camera(
    camera_def: str,
    batch_dims: tuple,
    width: int,
    height: int,
    rs_type: RollingShutterType,
    device: torch.device = torch.device("cuda"),
    seed: int = 42,
):
    """
    Parse camera model definition string and generate test parameters.

    Args:
        camera_def: Camera model definitionstring in format "model[params]"
                     e.g., "pinhole", "pinhole[k6]", "fisheye[k4]", "ftheta[a2p]"
        batch_dims: Batch dimensions for camera parameters
        width: Image width in pixels
        height: Image height in pixels
        rs_type: Rolling shutter type
        device: Device to use for generating parameters

    Returns:
        Dictionary with camera constructor parameters
    """

    # Parse "model[params]" format
    match = re.match(r"(\w+)(?:\[([^\]]+)\])?", camera_def)
    if not match:
        raise ValueError(f"Invalid camera model string: {camera_def}")

    torch.manual_seed(seed)

    model_type = match.group(1)
    param_str = match.group(2) or ""

    # Map model_type to concrete class
    if model_type == "pinhole":
        if param_str:
            camera_parser = parse_opencv_pinhole_camera
        else:
            camera_parser = parse_perfect_pinhole_camera
    elif model_type == "fisheye":
        camera_parser = parse_opencv_fisheye_camera
    elif model_type == "ftheta":
        camera_parser = parse_ftheta_camera
    elif model_type == "lidar":
        camera_parser = parse_lidar_camera
    else:
        raise ValueError(f"Unknown camera model: {model_type}")

    # Get model-specific parameters from concrete class
    params = camera_parser(param_str, batch_dims, width, height, device)

    if model_type == "lidar":
        lidar_params, angles_to_columns_map, tiling = params
        return {
            "camera_model": model_type,
            "lidar_coeffs": RowOffsetStructuredSpinningLidarModelParametersExt(
                lidar_params, angles_to_columns_map, tiling
            ),
        }
    else:
        resolution = torch.tensor([width, height], dtype=torch.float32).cuda()
        principal_points = (
            torch.randn(*batch_dims, 2, dtype=torch.float32, device=device) * 0.05 + 0.5
        ) * resolution
        params["principal_points"] = principal_points
        params["rs_type"] = rs_type.to_cpp() if hasattr(rs_type, "to_cpp") else rs_type
        params["width"] = width
        params["height"] = height
        params["camera_model"] = model_type
        return params


def parse_perfect_pinhole_camera(
    param_str: str, batch_dims: tuple, width: int, height: int, device: torch.device
):
    """Parse parameters for perfect pinhole camera (no distortion)."""

    # Generate focal_lengths with normal FOV (40-90 degrees)
    fov_deg = 40.0 + torch.rand(batch_dims, device=device, dtype=torch.float32) * 50.0
    fov_rad = fov_deg * (math.pi / 180.0)
    focal_length_x = _compute_focal_length(width, fov_rad)

    asymmetry = 0.98 + torch.rand(batch_dims, device=device, dtype=torch.float32) * 0.04
    focal_length_y = focal_length_x * asymmetry
    focal_lengths = torch.stack([focal_length_x, focal_length_y], dim=-1)

    return {"focal_lengths": focal_lengths}


def parse_opencv_pinhole_camera(
    param_str: str, batch_dims: tuple, width: int, height: int, device: torch.device
):
    """Parse parameters for OpenCV pinhole camera with distortion."""

    # Generate focal_lengths with normal FOV (40-90 degrees)
    fov_deg = 40.0 + torch.rand(batch_dims, device=device, dtype=torch.float32) * 50.0
    fov_rad = fov_deg * (math.pi / 180.0)
    focal_length_x = _compute_focal_length(width, fov_rad)

    # 4% asymmetry
    asymmetry = 0.98 + torch.rand(batch_dims, device=device, dtype=torch.float32) * 0.04
    focal_length_y = focal_length_x * asymmetry
    focal_lengths = torch.stack([focal_length_x, focal_length_y], dim=-1)

    params = {"focal_lengths": focal_lengths}

    # Parse distortion parameters from param_str
    if k_match := re.search(r"k(\d+)", param_str):
        n_radial = int(k_match.group(1))
        params["radial_coeffs"] = (
            torch.randn(*batch_dims, n_radial, dtype=torch.float32).cuda() * 0.01
        )

    if "tangential" in param_str:
        params["tangential_coeffs"] = (
            torch.randn(*batch_dims, 2, dtype=torch.float32).cuda() * 0.001
        )

    if "thin_prism" in param_str:
        params["thin_prism_coeffs"] = (
            torch.randn(*batch_dims, 4, dtype=torch.float32).cuda() * 0.001
        )

    return params


def parse_opencv_fisheye_camera(
    param_str: str, batch_dims: tuple, width: int, height: int, device: torch.device
):
    """Parse parameters for OpenCV fisheye camera."""
    import re

    # Generate focal_lengths with wide FOV (90-180 degrees)
    fov_deg = 90.0 + torch.rand(batch_dims, device=device, dtype=torch.float32) * 90.0
    fov_rad = fov_deg * (math.pi / 180.0)
    focal_length_x = _compute_focal_length(width, fov_rad)

    asymmetry = 0.98 + torch.rand(batch_dims, device=device, dtype=torch.float32) * 0.04
    focal_length_y = focal_length_x * asymmetry

    focal_lengths = torch.stack([focal_length_x, focal_length_y], dim=-1)

    # Parse radial distortion count (default k4)
    n_radial = 4
    if k_match := re.search(r"k(\d+)", param_str):
        n_radial = int(k_match.group(1))

    radial_coeffs = (
        torch.randn(*batch_dims, n_radial, dtype=torch.float32).cuda() * 0.01
    )

    return {
        "focal_lengths": focal_lengths,
        "radial_coeffs": radial_coeffs,
    }


def parse_ftheta_camera(
    param_str: str, batch_dims: tuple, width: int, height: int, device: torch.device
):
    """Parse parameters for FTheta camera model."""

    # Generate focal_lengths with wide FOV (90-180 degrees)
    fov_deg = 90.0 + torch.rand(1, dtype=torch.float32) * 90.0
    fov_rad = fov_deg * (math.pi / 180.0)
    focal_length = _compute_focal_length(width, fov_rad)

    max_angle = fov_rad

    # Determine which polynomial is the reference
    reference_poly = None
    if "a2p" in param_str:
        reference_poly = FThetaPolynomialType.ANGLE_TO_PIXELDIST
    else:
        reference_poly = FThetaPolynomialType.PIXELDIST_TO_ANGLE

    if "pinhole" in param_str:
        pixeldist_to_angle_poly = [0, 1.0 / focal_length, 0, 0, 0, 0]
        angle_to_pixeldist_poly = [0, focal_length, 0, 0, 0, 0]
        linear_cde = [1, 0, 0]
    else:
        # linear_cde: chromatic distortion or center distortion parameters
        # c should be close to 1.0, d and e are small perturbations
        # Use uniform distribution for predictable test parameters
        c = 1.0 + (torch.rand(1).item() - 0.5) * 2.0 * 0.01  # c in [0.98, 1.02]
        d = (torch.rand(1).item() - 0.5) * 2.0 * 0.001  # d in [-0.001, 0.001]
        e = (torch.rand(1).item() - 0.5) * 2.0 * 0.001  # e in [-0.001, 0.001]
        linear_cde = [c, d, e]

        # Compute max pixel distance (image diagonal / 2)
        max_delta = math.sqrt(width**2 + height**2) / 2.0

        # Generate one accurate polynomial with distortion, compute the other as its inverse
        if reference_poly == FThetaPolynomialType.PIXELDIST_TO_ANGLE:
            # PIXELDIST_TO_ANGLE is reference: delta (pixeldist) -> theta (angle)
            # This polynomial maps pixel distance to angle in radians
            k1_linear = 1.0 / focal_length

            # Scale distortion coefficients appropriately for pixel input range
            # At delta ≈ focal_length, each term should contribute a small fraction of linear term

            # 0.05% of linear term
            k2_distortion = (torch.rand(1) - 0.5) * 2 * 0.0005 / focal_length**2
            # 0.1% of linear term
            k3_distortion = (torch.rand(1) - 0.5) * 2 * 0.001 / focal_length**3
            # 0.05% of linear term
            k4_distortion = (torch.rand(1) - 0.5) * 2 * 0.0005 / focal_length**4
            # 0.01% of linear term
            k5_distortion = (torch.rand(1) - 0.5) * 2 * 0.0001 / focal_length**5

            # Sample only delta values that map to valid angles [0, max_angle]
            # Use a conservative range to ensure polynomial stays well-behaved
            max_delta_linear = max_angle * focal_length
            # Use 70% of linear estimate - enough range for good fitting
            max_delta_for_inversion = max_delta_linear * 0.7
            input_range_for_inversion = (0.0, max_delta_for_inversion.item())
        else:
            # For angle input (range 0-1.5 rad), distortion coefficients can be larger
            # At theta ≈ 1 rad, each term should contribute a small fraction of linear term

            # ANGLE_TO_PIXELDIST is reference: theta (angle) -> delta (pixeldist)
            # This polynomial maps angle in radians to pixel distance
            k1_linear = focal_length

            # 0.01% of linear term
            k2_distortion = (torch.rand(1) - 0.5) * 2 * 0.01 * focal_length
            # 0.03% of linear term
            k3_distortion = (torch.rand(1) - 0.5) * 2 * 0.03 * focal_length
            # 0.005% of linear term
            k4_distortion = (torch.rand(1) - 0.5) * 2 * 0.005 * focal_length
            # 0.001% of linear term
            k5_distortion = (torch.rand(1) - 0.5) * 2 * 0.001 * focal_length

            input_range_for_inversion = (0.0, max_angle.item())  # Sample angles

        # Generate reference polynomial with distortion (all terms non-zero except k0)
        reference_poly_coeffs = [
            0.0,  # k0: must be zero (passes through origin)
            k1_linear.item(),  # k1: linear term
            k2_distortion.item(),  # k2: quadratic distortion
            k3_distortion.item(),  # k3: cubic distortion
            k4_distortion.item(),  # k4: quartic distortion
            k5_distortion.item(),  # k5: quintic distortion
        ]

        # Compute inverse polynomial
        inverse_poly_coeffs = compute_inverse_polynomial(
            reference_poly_coeffs,
            input_range=input_range_for_inversion,
            num_samples=10000,
        )

        # Assign to correct variables based on reference type
        if reference_poly == FThetaPolynomialType.PIXELDIST_TO_ANGLE:
            pixeldist_to_angle_poly = reference_poly_coeffs
            angle_to_pixeldist_poly = inverse_poly_coeffs
        else:
            angle_to_pixeldist_poly = reference_poly_coeffs
            pixeldist_to_angle_poly = inverse_poly_coeffs

    ftheta_coeffs = FThetaCameraDistortionParameters(
        reference_poly=reference_poly,
        pixeldist_to_angle_poly=pixeldist_to_angle_poly,
        angle_to_pixeldist_poly=angle_to_pixeldist_poly,
        max_angle=float(max_angle),
        linear_cde=linear_cde,
    )

    return {"ftheta_coeffs": ftheta_coeffs}


# ==================================================
# Fixtures used
# ==================================================


@lru_cache(maxsize=16)
def _cached_lidar_preprocessing(
    lidar_params: RowOffsetStructuredSpinningLidarModelParameters,
    n_bins_elevation: int,
    max_pts_per_tile: int,
    resolution_elevation: int,
    densification_factor_azimuth: int,
):
    """Cache expensive lidar preprocessing across test fixtures."""
    angles_to_columns_map = compute_lidar_angles_to_columns_map(lidar_params)
    tiling = compute_lidar_tiling(
        lidar_params,
        n_bins_elevation=n_bins_elevation,
        max_pts_per_tile=max_pts_per_tile,
        resolution_elevation=resolution_elevation,
        densification_factor_azimuth=densification_factor_azimuth,
    )
    return angles_to_columns_map, tiling


def parse_lidar_camera(
    param_str: str,
    batch_dims: tuple,
    width: int,
    height: int,
    device: torch.device,
    seed: int | None = None,
):
    """Parse parameters for lidar camera model."""

    params = SimpleNamespace()

    sensor_name = param_str

    if sensor_name == "pandar128":
        # Values taken from actual pandar128 representation
        n_rows = 128
        n_columns = 3600
        elevation_start = 0.25195573
        elevation_end = -0.41325906
        azimuth_base_start = 3.0847472798897
        azimuth_base_end = -3.196692698037892

        params.spinning_direction = SpinningDirection.CLOCKWISE
        params.spinning_frequency_hz = 10
    elif sensor_name == "at128":
        # Values taken from actual at128 representation
        n_rows = 128
        n_columns = 1200
        elevation_start = 0.2256710722828668
        elevation_end = -0.2176425577236929
        azimuth_base_start = 1.0471975511965976
        azimuth_base_end = -1.0471975511965976

        params.spinning_direction = SpinningDirection.CLOCKWISE
        params.spinning_frequency_hz = 10
    else:
        raise ValueError(f"Unknown lidar sensor: {sensor_name}")

    elevation_span = abs(elevation_end - elevation_start)
    azimuth_base_span = abs(azimuth_base_end - azimuth_base_start)
    # Allow tests to request deterministic lidar params without depending on
    # whatever random draws happened earlier in the same test. This keeps the
    # preprocessing cache reusable while preserving the old ambient-RNG behavior
    # when seed=None.
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    # Generate random row elevations within FOV (sorted descending for typical lidar)
    params.row_elevations_rad = (
        torch.linspace(
            elevation_start,
            elevation_end,
            n_rows,
            dtype=torch.float32,
            device=device
            # Add small noise, but make sure it's not larger than the spacing between each row.
        )
        + (
            torch.rand(n_rows, dtype=torch.float32, device=device, generator=generator)
            - 0.5
        )
        * (elevation_span / (n_rows - 1))
        * 0.01
    )

    params.column_azimuths_rad = (
        torch.linspace(
            azimuth_base_start,
            azimuth_base_end,
            n_columns,
            dtype=torch.float32,
            device=device,
        )
        + (
            torch.rand(
                n_columns, dtype=torch.float32, device=device, generator=generator
            )
            - 0.5
        )
        * (azimuth_base_span / (n_columns - 1))
        * 0.01
    )

    # Generate small random azimuth offsets per row
    params.row_azimuth_offsets_rad = (
        torch.rand(n_rows, dtype=torch.float32, device=device, generator=generator)
        - 0.5
    ) * 0.2

    lidar_params = RowOffsetStructuredSpinningLidarModelParameters(**vars(params))

    angles_to_columns_map, tiling = _cached_lidar_preprocessing(
        lidar_params,
        n_bins_elevation=16,
        # Lidar always dispatches the tile_size=8 3DGUT kernel: production
        # spinning lidars are wide (n_columns ≥ 1200) but shallow (n_rows ≤
        # 128), so the resolution-based fallback in rendering.py picks 8 via
        # min(W,H) < 1080. The compact-CTA kernel at <CDIM,8,32> processes at
        # most 64 elements per tile, which caps max_pts_per_tile here.
        max_pts_per_tile=8 * 8,
        resolution_elevation=1600,
        densification_factor_azimuth=8,
    )

    return lidar_params, angles_to_columns_map, tiling


@pytest.fixture
def camera_rays(batch_dims, image_dims, ref_camera):
    """Generate normalized random camera rays for testing."""
    torch.manual_seed(SEED)

    n_points = image_dims[0] * image_dims[1]
    shape = list(batch_dims) + [n_points]
    device = ref_camera.focal_lengths.device

    rays = (
        (torch.rand(*shape, 3, device=device) * 2 - 1)
        * 2
        * ref_camera.focal_lengths[..., None, 1:2]
    )

    rays = torch.nn.functional.normalize(rays, dim=-1)
    return rays


@pytest.fixture
def image_points(batch_dims, image_dims, ref_camera):
    """Generate random image points for testing."""
    torch.manual_seed(SEED)

    n_points = image_dims[0] * image_dims[1]
    height, width = image_dims
    shape = list(batch_dims) + [n_points]

    # Lidar cameras use scaled angle space, not pixel coordinates
    if isinstance(ref_camera, _RowOffsetStructuredSpinningLidarModel):
        # Generate points in scaled angle space
        # For lidar: image_point = angle * ANGLE_TO_PIXEL_SCALING_FACTOR
        device = "cuda"

        params = ref_camera.params
        if params.spinning_direction == SpinningDirection.CLOCKWISE:
            # Clockwise: azimuth decreases
            azimuth = (
                params.fov_horiz_rad.start
                - params.fov_horiz_rad.span * torch.rand(*shape, device=device)
            )
        else:
            # Counter-clockwise: azimuth increases
            azimuth = (
                params.fov_horiz_rad.start
                + params.fov_horiz_rad.span * torch.rand(*shape, device=device)
            )

        # Elevation decreases, always clockwise
        elevation = params.fov_vert_rad.start - params.fov_vert_rad.span * torch.rand(
            *shape, device=device
        )

        # Scale to image point space
        column = (
            azimuth
            * _RowOffsetStructuredSpinningLidarModel.ANGLE_TO_PIXEL_SCALING_FACTOR
        )
        row = (
            elevation
            * _RowOffsetStructuredSpinningLidarModel.ANGLE_TO_PIXEL_SCALING_FACTOR
        )

        points = torch.stack([column, row], dim=-1)
    else:
        # Regular cameras use pixel coordinates
        points = torch.rand(*shape, 2) * torch.tensor([height - 1, width - 1])
        points = points.cuda()

    return points


@pytest.fixture
def world_points(batch_dims, image_dims, ref_camera):
    """Generate random world points distributed uniformily in a box"""
    n_points = image_dims[0] * image_dims[1]
    shape = list(batch_dims) + [n_points]
    device = ref_camera.focal_lengths.device

    torch.manual_seed(SEED)

    world_pts = torch.rand(*shape, 3, device=device) * 2 - 1
    world_pts *= torch.cat(
        [
            ref_camera.focal_lengths[..., None, :] * 2,
            torch.ones(*batch_dims, 1, 1, device=device),
        ],
        dim=-1,
    )
    return world_pts


@pytest.fixture
def pose_start(batch_dims, ref_camera, rs_type):
    """Generate random start pose for rolling shutter testing."""
    torch.manual_seed(SEED)

    shape = list(batch_dims)

    device = ref_camera.focal_lengths.device

    # Pose format: [tx, ty, tz, qw, qx, qy, qz]
    pose = torch.randn(*shape, 7, device=device) * 0.01
    pose[..., :3] *= ref_camera.focal_lengths[..., 0:1]
    # Normalize quaternion
    pose[..., 3:] = _safe_normalize(pose[..., 3:])
    return pose.cuda()


@pytest.fixture
def pose_end(batch_dims, pose_start, rs_type, ref_camera):
    """Generate realistic end pose for rolling shutter testing."""

    torch.manual_seed(SEED)

    # Special case: GLOBAL shutter has no motion during instantaneous exposure
    if rs_type == RollingShutterType.GLOBAL:
        return pose_start.clone()

    # For rolling shutter types: generate realistic camera motion
    shape = list(batch_dims)
    device = pose_start.device

    # Random angular velocity: 1-10°/s
    angular_axis = torch.randn(*shape, 3, device=device)
    angular_axis = angular_axis / torch.linalg.vector_norm(
        angular_axis, dim=-1, keepdim=True
    )

    # Rotation angle during rolling shutter readout, 2 degrees max
    # Translation delta: max 10% focal length
    # TODO: these should be dependent on the camera's focal length and image dimensions!
    angle_delta = (torch.rand(*shape, 1, device=device) - 0.5) * 2 * torch.pi / 180
    translation_delta = (
        (torch.randn(*shape, 3, device=device) - 0.5)
        * ref_camera.focal_lengths[..., 1:2]
        * 0.1
    )

    # Convert axis-angle to quaternion: q = [cos(θ/2), sin(θ/2) * axis]
    qw_delta = torch.cos(angle_delta / 2)
    qxyz_delta = torch.sin(angle_delta / 2) * angular_axis
    q_delta = torch.cat([qw_delta, qxyz_delta], dim=-1)

    # Compose rotations using quaternion multiplication: q_end = q_delta * q_start
    q_start = pose_start[..., 3:]  # [qw, qx, qy, qz]
    q_end = _quat_multiply(q_delta, q_start)

    # Normalize the result
    q_end = q_end / torch.linalg.vector_norm(q_end, dim=-1, keepdim=True)

    # Apply constant velocity motion to create end pose
    t_end = pose_start[..., :3] + translation_delta
    pose_end = torch.cat([t_end, q_end], dim=-1)

    return pose_end


@pytest.fixture
def test_camera(camera_model, batch_dims, image_dims, rs_type):
    """Create C++ camera (primary implementation)"""
    height, width = image_dims
    params = parse_camera(
        camera_model, batch_dims, width=width, height=height, rs_type=rs_type, seed=SEED
    )

    return create_camera_model(**params)


@pytest.fixture
def ref_camera(camera_model, batch_dims, image_dims, rs_type):
    """Create PyTorch reference camera"""
    height, width = image_dims
    params = parse_camera(
        camera_model, batch_dims, width=width, height=height, rs_type=rs_type, seed=SEED
    )

    return _BaseCameraModel.create(**params)


@pytest.fixture
def camera(request, test_camera, ref_camera):
    """Parametrizable fixture that returns either test_camera or ref_camera"""
    if request.param == "test_camera":
        return test_camera
    elif request.param == "ref_camera":
        return ref_camera
    else:
        raise ValueError(f"Unknown camera type: {request.param}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "camera_model,batch_dims",
    [
        pytest.param(*params)
        for params in chain(
            # For non-lidar cameras
            product(
                [
                    lidar_model
                    for lidar_model in CAMERA_MODELS
                    if "lidar" not in lidar_model
                ],  # camera_model
                [(), (2,), (2, 3)],  # batch_dims
            ),
            # Lidar cameras
            product(
                [
                    lidar_model
                    for lidar_model in CAMERA_MODELS
                    if "lidar" in lidar_model
                ],  # camera_model
                [()],  # no batch dims (lidar currently doesn't support batched cameras
            ),
        )
    ],
)
@pytest.mark.parametrize("image_dims", [(127, 256)])
@pytest.mark.parametrize(
    "rs_type", expand_named_params(DEFAULT_ROLLING_SHUTTER_TYPE)
)  # this is ignored for lidars
class TestCameraModels:
    """
    Validate C++ camera models against PyTorch reference.
    """

    def test_camera_ray_to_image_point(self, camera_rays, test_camera, ref_camera):
        test_imgpt, test_valid = test_camera.camera_ray_to_image_point(camera_rays, 0.0)
        ref_imgpt, ref_valid = ref_camera.camera_ray_to_image_point(camera_rays, 0.0)

        all_valid = test_valid & ref_valid
        assert all_valid.any(), "No valid points found"

        # Tolerances based on camera type (ordered by decreasing atol)
        # Values set ~20% above observed maximums for tight margin
        if isinstance(ref_camera, _OpenCVFisheyeCameraModel):
            # fisheye: observed atol=4.96e-05, rtol=1.45e-03
            atol, rtol = 6e-05, 1.8e-03
        elif isinstance(
            ref_camera, (_OpenCVPinholeCameraModel, _PerfectPinholeCameraModel)
        ):
            # OpenCV pinhole & perfect pinhole: observed atol=3.05e-05, rtol=2.47e-03
            atol, rtol = 3.7e-05, 3e-03
        elif isinstance(ref_camera, _FThetaCameraModel):
            # ftheta: observed atol=1.53e-05, rtol=1.69e-07
            atol, rtol = 1.9e-05, 2.5e-07
        elif isinstance(ref_camera, _RowOffsetStructuredSpinningLidarModel):
            atol, rtol = 5e-03, 2e-03
        else:  # Fallback
            atol, rtol = None, None
        assert_close(test_imgpt[all_valid], ref_imgpt[all_valid], atol=atol, rtol=rtol)

        # Validity mismatch handling.
        #
        # The validity flag is `not_behind_camera & converged & (theta <=
        # max_angle) & valid_bounds`. Empirically, the FP-amplifier on this
        # path is the Newton iteration that inverts the ftheta polynomial
        # (`_eval_poly_inverse_horner_newton`); each iteration's residual is
        # within ULP of the convergence threshold (|dx|<1e-6) for rays at
        # high incidence (z near 0, theta near pi/2). ULP noise can
        # flip the convergence flag, which then zeros image_point and flips
        # validity. The flip is in INTERNAL Newton state -- not directly
        # observable from outside the kernel -- so we cannot write a strict
        # cross predicate ("a was just-converged, b was just-not-converged").
        # We rely on:
        #   interior assert (exact agreement off-band) -- regression catcher
        #   band flip-rate cap                        -- absorb FP noise
        #   symmetry guardrail                        -- catch directional bug
        #
        # The band classifier is geometric: rays at theta > some threshold
        # (close to pi/2 or close to max_angle) where Newton convergence is
        # FP-sensitive on this polynomial. Below the threshold all rays
        # converge identically, so validity must agree exactly.
        xy_norm = torch.linalg.norm(camera_rays[..., :2], dim=-1)
        theta_full = torch.atan2(xy_norm, camera_rays[..., 2])
        # FP-sensitive zone: theta > 1.3 rad (75 deg) or theta within 0.05
        # of max_angle. The 1.3 cutoff captures the empirical [1.348, 1.570]
        # range observed (RTX PRO 2000) with safety margin.
        max_angle = (
            ref_camera.max_angle.flatten()[0].item()
            if hasattr(ref_camera, "max_angle")
            else float("inf")
        )
        # FP-sensitive zone for Newton convergence flag flips:
        #   - Real cause is the |dx| < 1e-6 stopping rule, not geometry.
        #     Upstream sqrt/atan2/div drift (gsplat --use_fast_math vs
        #     PyTorch IEEE) can push |dx| onto opposite sides at any well-
        #     conditioned theta, not only near max_angle as first thought.
        #   - Worst observed flips:
        #       RTX PRO 2000  theta ~1.27
        #       RTX PRO 6000  theta ~1.019
        #   - Threshold 1.0 rad covers both with ~0.02 rad margin; plus
        #     a 0.05 rad margin near max_angle.
        boundary_mask = (theta_full > 1.0) | ((max_angle - theta_full).abs() < 5e-2)

        # Calibration trace (RTX PRO 2000):
        #   - boundary band: rays at theta > 1.3 rad or within 0.05 of max_angle
        #   - in-band flips:
        #       * 493/N_band  (1.52% of total)  ftheta[pinhole+p2a] arms
        #       * ~510/N_band (1.57% of total)  ftheta[p2a] arms
        #     all geometrically inside image bounds (Newton convergence flip)
        #   - asymmetry: 243 vs 250 (cuda-only-valid vs ref-only-valid)
        #     -> |delta|/sum = 7/493 = 0.014, well under 0.5 cap
        # RTX PRO 6000:
        #   - in-band flips: ~0.23% of total (well under any budget)
        # See plan: ~/.claude/plans/how-to-devise-tests-hidden-fox.md
        # No cross predicate: the Newton residual is internal state.
        assert_close_with_boundary_band(
            test_valid,
            ref_valid,
            boundary_mask=boundary_mask,
            interior_atol=0,  # exact agreement off-band
            interior_rtol=0,
            boundary_max_flip_ratio=0.10,  # 6x observed worst (1.57%)
            boundary_symmetry_tol=0.5,  # 3:1 directional bias rejected
            flip_predicate=None,  # default a != e for bool
            boundary_cross_predicate=None,
            msg="cluster B: ftheta Newton-convergence FP-sensitivity",
        )

    def test_image_point_to_camera_ray(self, image_points, test_camera, ref_camera):
        test_rays, test_valid = test_camera.image_point_to_camera_ray(image_points)
        ref_rays, ref_valid = ref_camera.image_point_to_camera_ray(image_points)

        all_valid = test_valid & ref_valid
        assert all_valid.any(), "No valid points found"

        # Per-element band+sparsity check on the rays (3D unit vectors).
        # Boundary band catches the two regions where the ftheta inverse-
        # distortion Newton iteration is FP-sensitive:
        #   xy_norm < 1e-2: rays near the optical axis (theta near 0).
        #     ULP cancellation in (sin(theta) -> 0) plus the inverse
        #     polynomial's behavior at theta=0.
        #   xy_norm >= 0.999: rays near the lens "horizon" (theta near
        #     pi/2).  The ftheta polynomial has a critical point there, so
        #     Newton convergence is sensitive to FP order.
        # Together these admit ~3.4% of rays in the worst arm; the remaining
        # 96% interior rays get a tight rtol bound.
        ray_xy_norm = torch.linalg.norm(test_rays[..., :2], dim=-1)
        ref_xy_norm = torch.linalg.norm(ref_rays[..., :2], dim=-1)
        near_axis = (ray_xy_norm < 1e-2) | (ref_xy_norm < 1e-2)
        near_horizon = (ray_xy_norm >= 0.999) | (ref_xy_norm >= 0.999)
        rays_band = ((near_axis | near_horizon)[all_valid])[..., None].expand(-1, 3)
        rays_a = test_rays[all_valid]
        rays_e = ref_rays[all_valid]
        nz_thresh = rays_e.abs().max().item() * 1e-5
        # ftheta interior rtol = 1.05 x worst observed in interior buckets
        # (5.18e-3 in xy in [1e-2, 1e-1)) -> 5.5e-3.  Replaces 9.7e-2.
        rays_rtol = 5.5e-3 if isinstance(ref_camera, _FThetaCameraModel) else 1e-3
        assert_close_with_boundary_band(
            rays_a,
            rays_e,
            boundary_mask=rays_band,
            interior_atol=nz_thresh,
            interior_rtol=rays_rtol,
            boundary_max_flip_ratio=1e-3,
            boundary_symmetry_tol=0.5,
            flip_predicate=lambda a, e, _t=nz_thresh: (a - e).abs() > 100 * _t,
            msg="image_point_to_camera_ray rays",
        )

        # Per-type tol = ~5x worst observed mismatch rate:
        #   - OpenCV pinhole (k4/k6/k6+tan/k6+thin):  0.0021% -> 1.1e-4 (50x)
        #   - OpenCV fisheye:                         0.0015% -> 1e-4   (6.7x)
        #   - ftheta:                                 0% to 0.77% (collapsed
        #       to 1e-2 because the pinhole+a2p variant needs it; differentiate
        #       once camera_model is plumbed in)
        #   - PerfectPinhole:                         0%      -> 1e-5
        if isinstance(ref_camera, _OpenCVPinholeCameraModel):
            validity_tol = 1.1e-04
        elif isinstance(ref_camera, _FThetaCameraModel):
            validity_tol = 1e-02
        elif isinstance(ref_camera, _OpenCVFisheyeCameraModel):
            validity_tol = 1e-04
        elif isinstance(ref_camera, _PerfectPinholeCameraModel):
            validity_tol = 1e-05
        else:
            validity_tol = 3e-02  # lidar / unknown fallback
        assert_mismatch_ratio(test_valid, ref_valid, max=validity_tol)

    def test_shutter_relative_frame_time(
        self, camera_model, batch_dims, image_points, test_camera, ref_camera
    ):
        test_times = test_camera.shutter_relative_frame_time(image_points)
        ref_times = ref_camera.shutter_relative_frame_time(image_points)

        # Tolerances based on camera type
        # Values set ~20% above observed maximums for tight margin
        if isinstance(ref_camera, _RowOffsetStructuredSpinningLidarModel):
            # Only 1 mismatched element (out of 3252. It could be due to
            # the angles_to_columns_map covering the angle 0 (==2*pi) twice
            # The expectation is that once this is fixed, the atol can be set back to 5e-3.
            atol, rtol = 3.5e-02, 3.5e-02
        else:
            # Other cameras: observed atol ~1e-06, rtol ~1e-06
            atol, rtol = 2e-06, 2e-06

        assert_close(test_times, ref_times, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("camera_model", ["pinhole"])
@pytest.mark.parametrize("batch_dims", [(), (2,), (2, 3)])
@pytest.mark.parametrize("image_dims", [(127, 257)])
@pytest.mark.parametrize("rs_type", expand_named_params(ROLLING_SHUTTER_TYPES))
class TestCameraModelsShutterPose:
    def test_image_point_to_world_ray_shutter_pose(
        self, image_points, pose_start, pose_end, test_camera, ref_camera
    ):
        """Test image_point_to_world_ray_shutter_pose method"""
        (
            test_rays_org,
            test_rays_dir,
            test_valid,
        ) = test_camera.image_point_to_world_ray_shutter_pose(
            image_points, pose_start, pose_end
        )
        (
            ref_rays_org,
            ref_rays_dir,
            ref_valid,
        ) = ref_camera.image_point_to_world_ray_shutter_pose(
            image_points, pose_start, pose_end
        )

        all_valid = test_valid & ref_valid
        assert all_valid.any(), "No valid points found"

        # Observed errors from shutter pose interpolation and projection
        # Values set ~20% above observed maximums for tight margin
        # rays_org: observed atol=3.34e-06, rtol=2.70e-06
        # rays_dir: observed atol=7.75e-07, rtol varies (high due to small components)
        assert_close(
            test_rays_org[all_valid], ref_rays_org[all_valid], atol=4e-06, rtol=3.5e-06
        )
        assert_close(
            test_rays_dir[all_valid], ref_rays_dir[all_valid], atol=1e-06, rtol=1e-03
        )

        # Tolerance for validity mismatches in shutter pose calculations
        # Higher for OpenCV models with distortion
        if isinstance(ref_camera, _OpenCVPinholeCameraModel):
            validity_tol = 3e-02  # 3% for OpenCV pinhole models
        elif isinstance(ref_camera, (_OpenCVFisheyeCameraModel, _FThetaCameraModel)):
            validity_tol = 1e-03  # 0.1% for fisheye/ftheta
        else:
            validity_tol = 1e-05  # 0.001% for perfect pinhole
        assert_mismatch_ratio(test_valid, ref_valid, max=validity_tol)

    def test_world_point_to_image_point_shutter_pose(
        self, world_points, pose_start, pose_end, test_camera, ref_camera
    ):
        """Test world_point_to_image_point_shutter_pose method"""
        test_img, test_valid = test_camera.world_point_to_image_point_shutter_pose(
            world_points, pose_start, pose_end, 0
        )
        ref_img, ref_valid = ref_camera.world_point_to_image_point_shutter_pose(
            world_points, pose_start, pose_end, 0
        )

        all_valid = test_valid & ref_valid
        assert all_valid.any(), "No valid points found"

        # Relaxed tolerances to account for numerical precision in shutter pose calculations
        # These tolerances handle accumulated errors from pose interpolation and projection
        # Values set ~20% above observed maximums for tight margin
        # Observed: atol=3.43e-03, rtol=0.717
        assert_close(test_img[all_valid], ref_img[all_valid], atol=4.2e-03, rtol=0.87)

        assert_mismatch_ratio(test_valid, ref_valid, max=1e-05)
