# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Layer 1 functional spinning-LiDAR wrappers.

Verifies that each of the five functional ops returns the correct dataclass with
the correct field shapes, that boolean ``return_*`` flags gate optional fields,
that ``allow_device_transfer`` is honored, and that the inverse op is
dtype-polymorphic (float32->float32, float64->float64) while generate / elements
force float32.
"""

import pytest
import torch

from gsplat_sensors import functional as F
from gsplat_sensors.functional.return_types import (
    SensorAnglesReturn,
    SensorRayReturn,
    WorldPointsToSensorAnglesReturn,
    WorldRaysReturn,
)
from gsplat_sensors.kernels.common import DynamicPose, Pose


def _pose(device, dtype=torch.float32):
    """A mild moving DynamicPose (0.05 m translation, identity rotation, wxyz)."""
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
    return DynamicPose(
        start_pose=Pose(
            translation=torch.zeros(3, device=device, dtype=dtype),
            rotation=q.clone(),
        ),
        end_pose=Pose(
            translation=torch.tensor([0.05, 0.02, -0.03], device=device, dtype=dtype),
            rotation=q.clone(),
        ),
    )


def _elements(device):
    """A small in-bounds (row, col) element set for the generic sensor."""
    return torch.tensor([[0, 0], [1, 5], [2, 10]], device=device, dtype=torch.int32)


def _world_points(device, dtype=torch.float32):
    """A few forward-facing world points for inverse-projection shape checks."""
    return torch.tensor(
        [[5.0, 0.0, 0.0], [4.0, 1.0, -0.5], [6.0, -1.0, 0.3]],
        device=device,
        dtype=dtype,
    )


def test_sensor_rays_to_sensor_angles_returns_dataclass(sensor_device):
    """Verify sensor_rays_to_sensor_angles returns a SensorAnglesReturn (valid_flag None)."""
    rays = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=sensor_device)
    result = F.sensor_rays_to_sensor_angles(rays, allow_device_transfer=True)
    assert isinstance(result, SensorAnglesReturn)
    assert result.sensor_angles.shape == (2, 2)
    assert result.valid_flag is None


def test_sensor_rays_to_sensor_angles_zero_ray_no_nan(sensor_device):
    """Verify a zero-length ray normalizes safely to angles [0, 0] without NaN."""
    rays = torch.zeros((1, 3), device=sensor_device)
    result = F.sensor_rays_to_sensor_angles(rays, allow_device_transfer=True)
    assert not torch.isnan(result.sensor_angles).any()
    assert torch.allclose(
        result.sensor_angles, torch.zeros((1, 2), device=sensor_device)
    )


def test_sensor_angles_to_sensor_rays_returns_dataclass(sensor_device):
    """Verify sensor_angles_to_sensor_rays returns a SensorRayReturn (valid_flag None)."""
    angles = torch.tensor([[0.0, 0.0], [0.1, 0.5]], device=sensor_device)
    result = F.sensor_angles_to_sensor_rays(angles, allow_device_transfer=True)
    assert isinstance(result, SensorRayReturn)
    assert result.sensor_rays.shape == (2, 3)
    assert result.valid_flag is None


def test_elements_to_sensor_angles_valid_flag_remap(
    lidar_projection_from_json, sensor_device
):
    """Verify return_valid_flag gates the boolean valid_flag field."""
    projection = lidar_projection_from_json("generic")
    elements = _elements(sensor_device)

    no_flag = F.elements_to_sensor_angles(
        elements, projection, allow_device_transfer=True
    )
    assert isinstance(no_flag, SensorAnglesReturn)
    assert no_flag.valid_flag is None

    with_flag = F.elements_to_sensor_angles(
        elements, projection, return_valid_flag=True, allow_device_transfer=True
    )
    assert with_flag.valid_flag is not None
    assert with_flag.valid_flag.dtype == torch.bool
    assert with_flag.valid_flag.shape == (elements.shape[0],)


def test_generate_returns_world_rays_with_optional_fields(
    lidar_projection_from_json, sensor_device
):
    """Verify generate_spinning_lidar_rays returns WorldRaysReturn and gates pose / timestamp fields."""
    projection = lidar_projection_from_json("generic")
    pose = _pose(sensor_device)
    elements = _elements(sensor_device)

    bare = F.generate_spinning_lidar_rays(projection, elements, pose)
    assert isinstance(bare, WorldRaysReturn)
    assert bare.world_rays.shape == (elements.shape[0], 6)
    assert bare.T_sensor_world is None
    assert bare.timestamps_us is None

    full = F.generate_spinning_lidar_rays(
        projection,
        elements,
        pose,
        start_timestamp_us=0,
        end_timestamp_us=100000,
        return_T_sensor_world=True,
        return_timestamps=True,
    )
    assert full.T_sensor_world.shape == (elements.shape[0], 4, 4)
    assert full.timestamps_us.dtype == torch.int64
    assert full.timestamps_us.shape == (elements.shape[0],)


def test_inverse_returns_world_points_to_sensor_angles_with_optional_fields(
    lidar_projection_from_json, sensor_device
):
    """Verify inverse_project_spinning_lidar returns the dataclass and gates optional fields."""
    projection = lidar_projection_from_json("generic")
    pose = _pose(sensor_device)
    world_points = _world_points(sensor_device)

    bare = F.inverse_project_spinning_lidar(projection, world_points, pose)
    assert isinstance(bare, WorldPointsToSensorAnglesReturn)
    assert bare.sensor_angles.shape == (world_points.shape[0], 2)
    assert bare.valid_flag is None
    assert bare.valid_indices is None
    assert bare.timestamps_us is None
    assert bare.T_sensor_world is None

    full = F.inverse_project_spinning_lidar(
        projection,
        world_points,
        pose,
        start_timestamp_us=0,
        end_timestamp_us=100000,
        return_T_sensor_world=True,
        return_valid_flag=True,
        return_valid_indices=True,
        return_timestamps=True,
    )
    assert full.T_sensor_world.shape == (world_points.shape[0], 4, 4)
    assert full.valid_flag.dtype == torch.bool
    assert full.valid_indices.dtype == torch.int64
    assert full.timestamps_us.dtype == torch.int64

    indices_only = F.inverse_project_spinning_lidar(
        projection,
        world_points,
        pose,
        return_valid_indices=True,
    )
    assert indices_only.valid_flag is None
    assert indices_only.valid_indices.dtype == torch.int64


def test_sensor_rays_to_sensor_angles_raises_without_device_transfer(sensor_device):
    """Verify a CPU input raises when allow_device_transfer is False."""
    rays = torch.tensor([[1.0, 0.0, 0.0]], device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="allow_device_transfer"):
        F.sensor_rays_to_sensor_angles(rays)


def test_inverse_dtype_polymorphism(lidar_projection_from_json, sensor_device):
    """Verify inverse sensor_angles dtype follows world_points (float32->float32, float64->float64).

    The public functional wrapper keeps the registered projection POD float32
    while passing explicit double tables/poses to the dtype-templated op.
    """
    pose32 = _pose(sensor_device)
    world_points32 = _world_points(sensor_device)

    proj32 = lidar_projection_from_json("generic")
    out32 = F.inverse_project_spinning_lidar(proj32, world_points32, pose32)
    assert out32.sensor_angles.dtype == torch.float32

    pose64 = _pose(sensor_device, torch.float64)
    out64 = F.inverse_project_spinning_lidar(
        proj32,
        world_points32.double(),
        pose64,
        start_timestamp_us=0,
        end_timestamp_us=100000,
        allow_device_transfer=True,
    )
    assert out64.sensor_angles.dtype == torch.float64
    assert proj32.row_elevations_rad.dtype == torch.float32


def test_generate_forces_float32_output(lidar_projection_from_json, sensor_device):
    """Verify generate forces float32 world_rays even with a float64 pose."""
    projection = lidar_projection_from_json("generic")
    pose64 = _pose(sensor_device, torch.float64)
    elements = _elements(sensor_device)
    result = F.generate_spinning_lidar_rays(
        projection, elements, pose64, allow_device_transfer=True
    )
    assert result.world_rays.dtype == torch.float32
