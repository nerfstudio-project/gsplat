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

"""Tests for opt-in projection validation and projection-family dispatch."""

from __future__ import annotations

import math

import pytest
import torch

from gsplat.sensors.kernels.cameras import (
    FISHEYE_MAX_FORWARD_POLY_TERMS,
    FTHETA_MAX_POLYNOMIAL_TERMS,
    FThetaProjection,
    OpenCVFisheyeProjection,
    validate_camera_projection,
)
from gsplat.sensors.kernels.cameras.ops import (
    _CameraRaysToImagePoints,
    _CameraRaysToImagePointsBivariateWindshield,
    _projection_on_device_any,
    _select_op,
)


class _UnknownProjection:
    pass


def _make_valid_ftheta_components(device: torch.device, dtype=torch.float32):
    """Linear-polynomial FTheta components forming a valid baseline projection."""
    k = 100.0
    fw_poly = torch.zeros(FTHETA_MAX_POLYNOMIAL_TERMS, device=device, dtype=dtype)
    fw_poly[1] = k
    bw_poly = torch.zeros(FTHETA_MAX_POLYNOMIAL_TERMS, device=device, dtype=dtype)
    bw_poly[1] = 1.0 / k
    return {
        "principal_point": torch.tensor([50.0, 40.0], device=device, dtype=dtype),
        "fw_poly": fw_poly,
        "bw_poly": bw_poly,
        "A": torch.tensor([1.0, 0.0, 0.0, 1.0], device=device, dtype=dtype),
        "resolution": (100, 80),
        "reference_polynomial": 0,
        "fw_poly_degree": 1,
        "bw_poly_degree": 1,
        "newton_iterations": 10,
        "max_angle": 1.4,
        "min_2d_norm": 1e-6,
    }


def _build_projection(components: dict) -> FThetaProjection:
    return FThetaProjection(
        principal_point=components["principal_point"],
        fw_poly=components["fw_poly"],
        bw_poly=components["bw_poly"],
        A=components["A"],
        resolution=components["resolution"],
        reference_polynomial=int(components["reference_polynomial"]),
        fw_poly_degree=int(components["fw_poly_degree"]),
        bw_poly_degree=int(components["bw_poly_degree"]),
        newton_iterations=int(components["newton_iterations"]),
        max_angle=float(components["max_angle"]),
        min_2d_norm=float(components["min_2d_norm"]),
    )


def _build_fisheye_projection(
    sensor_device: torch.device, focal_length: tuple[float, float] = (100.0, 100.0)
) -> OpenCVFisheyeProjection:
    return OpenCVFisheyeProjection(
        principal_point=torch.tensor([50.0, 40.0], device=sensor_device),
        focal_length=torch.tensor(focal_length, device=sensor_device),
        forward_poly=torch.zeros(FISHEYE_MAX_FORWARD_POLY_TERMS, device=sensor_device),
        approx_backward_factor=torch.tensor([1.0], device=sensor_device),
        resolution=(100, 80),
        newton_iterations=10,
        max_angle=1.8,
        min_2d_norm=1e-6,
    )


def test_cpp_max_polynomial_terms_is_python_source_of_truth():
    assert (
        FTHETA_MAX_POLYNOMIAL_TERMS
        == torch.classes.gsplat_sensors.FThetaProjection.get_max_polynomial_terms()
    )


def test_cpp_fisheye_forward_poly_terms_is_python_source_of_truth():
    assert (
        FISHEYE_MAX_FORWARD_POLY_TERMS
        == torch.classes.gsplat_sensors.OpenCVFisheyeProjection.get_max_forward_poly_terms()
    )


def test_validate_camera_projection_accepts_valid_ftheta_components(sensor_device):
    """Verify that a well-formed FTheta projection passes validation without raising."""
    components = _make_valid_ftheta_components(sensor_device)
    projection = _build_projection(components)
    validate_camera_projection(projection)


def test_validate_camera_projection_accepts_valid_fisheye_components(sensor_device):
    """Verify that a well-formed OpenCV fisheye projection passes validation."""
    validate_camera_projection(_build_fisheye_projection(sensor_device))


def test_validate_camera_projection_noops_for_pinhole(ideal_projection):
    validate_camera_projection(ideal_projection)


def test_validate_camera_projection_rejects_unknown_projection():
    with pytest.raises(TypeError, match="Unknown camera projection class"):
        validate_camera_projection(_UnknownProjection())


@pytest.mark.parametrize(
    "field,index,bad_value",
    [
        ("principal_point", 0, float("nan")),
        ("principal_point", 1, float("inf")),
        ("fw_poly", 1, float("nan")),
        ("bw_poly", 2, float("nan")),
        ("A", 0, float("inf")),
        ("A", 3, float("-inf")),
    ],
)
def test_validate_camera_projection_rejects_non_finite_ftheta_component_value(
    sensor_device, field, index, bad_value
):
    """Verify that nan/inf in any component tensor raises with a field-scoped message."""
    components = _make_valid_ftheta_components(sensor_device)
    components[field] = components[field].clone()
    components[field][index] = bad_value
    projection = _build_projection(components)
    with pytest.raises(ValueError, match=f"{field}\\["):
        validate_camera_projection(projection)


@pytest.mark.parametrize("focal_length", [(0.0, 100.0), (100.0, -1.0)])
def test_validate_camera_projection_rejects_non_positive_fisheye_focal_length(
    sensor_device, focal_length
):
    """Verify that invalid fisheye focal lengths are rejected before CUDA division."""
    projection = _build_fisheye_projection(sensor_device, focal_length)
    with pytest.raises(ValueError, match=r"focal_length\[[01]\] must be > 0"):
        validate_camera_projection(projection)


def test_validate_camera_projection_rejects_nonzero_fw_poly_constant(sensor_device):
    """Verify that non-zero fw_poly[0] is rejected so on-axis rays land on the principal point."""
    components = _make_valid_ftheta_components(sensor_device)
    components["fw_poly"] = components["fw_poly"].clone()
    components["fw_poly"][0] = 1e-3
    projection = _build_projection(components)
    with pytest.raises(ValueError, match=r"fw_poly\[0\] must be 0"):
        validate_camera_projection(projection)


def test_validate_camera_projection_rejects_nonzero_bw_poly_constant(sensor_device):
    """Verify that non-zero bw_poly[0] is rejected so on-axis rays land on the principal point."""
    components = _make_valid_ftheta_components(sensor_device)
    components["bw_poly"] = components["bw_poly"].clone()
    components["bw_poly"][0] = -1e-3
    projection = _build_projection(components)
    with pytest.raises(ValueError, match=r"bw_poly\[0\] must be 0"):
        validate_camera_projection(projection)


def test_validate_camera_projection_rejects_singular_A(sensor_device):
    """Verify that a singular A (det == 0) is rejected so the on-demand inverse is well-defined."""
    components = _make_valid_ftheta_components(sensor_device)
    components["A"] = torch.tensor(
        [1.0, 1.0, 1.0, 1.0], device=sensor_device, dtype=torch.float32
    )
    projection = _build_projection(components)
    with pytest.raises(ValueError, match=r"A must be non-singular"):
        validate_camera_projection(projection)


def test_native_rejects_max_angle_above_pi(sensor_device):
    """Verify that max_angle > pi is rejected by the C++ constructor."""
    components = _make_valid_ftheta_components(sensor_device)
    components["max_angle"] = math.pi + 1e-3
    with pytest.raises(RuntimeError, match=r"max_angle must be in \[0, pi\]"):
        _build_projection(components)


def test_native_rejects_max_angle_nan(sensor_device):
    """Verify that a non-finite max_angle is rejected by the C++ constructor."""
    components = _make_valid_ftheta_components(sensor_device)
    components["max_angle"] = float("nan")
    with pytest.raises(RuntimeError, match=r"max_angle must be in \[0, pi\]"):
        _build_projection(components)


def test_native_rejects_min_2d_norm_inf(sensor_device):
    """Verify that a non-finite min_2d_norm is rejected by the C++ constructor."""
    components = _make_valid_ftheta_components(sensor_device)
    components["min_2d_norm"] = float("inf")
    with pytest.raises(RuntimeError, match=r"min_2d_norm must be finite"):
        _build_projection(components)


def test_projection_on_device_any_rejects_unknown_projection(sensor_device):
    with pytest.raises(TypeError, match="Unknown camera projection class"):
        _projection_on_device_any(
            _UnknownProjection(),
            sensor_device,
            torch.float32,
            allow_device_transfer=True,
        )


def test_select_op_rejects_unknown_projection(sensor_device, no_external):
    with pytest.raises(TypeError, match="Unknown camera projection class"):
        _select_op(
            _UnknownProjection(),
            "camera_rays_to_image_points",
            _CameraRaysToImagePoints,
            _CameraRaysToImagePointsBivariateWindshield,
            no_external,
            sensor_device,
            torch.float32,
            allow_device_transfer=True,
        )
