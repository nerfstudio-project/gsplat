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

# Row mapping: design-tests-opencvpinhole.md §6 rows 24-30 are covered here.

import pytest
import torch

from gsplat.sensors import functional as F
from gsplat.sensors.functional.return_types import (
    ImagePointsReturn,
    WorldPointsToImagePointsReturn,
    WorldRaysReturn,
)
from gsplat.sensors.kernels.cameras import ops as kernel_ops
from gsplat.sensors.kernels.cameras import ShutterType


def test_camera_rays_to_image_points_returns_image_points_return(
    ideal_projection, no_external
):
    """Verifies that camera_rays_to_image_points returns an ImagePointsReturn with correct shapes."""
    rays = torch.tensor([[0.0, 0.0, 1.0]], device=ideal_projection.focal_length.device)
    result = F.camera_rays_to_image_points(
        rays,
        ideal_projection,
        no_external,
        allow_device_transfer=True,
    )
    assert isinstance(result, ImagePointsReturn)
    assert result.image_points.shape == (1, 2)
    assert result.valid_flag.shape == (1,)
    assert result.jacobians is None


def test_image_points_to_camera_rays_returns_bare_tensor(ideal_projection, no_external):
    """Verifies that image_points_to_camera_rays returns a bare (N, 3) Tensor."""
    rays = F.image_points_to_camera_rays(
        ideal_projection.principal_point.reshape(1, 2),
        ideal_projection,
        no_external,
        allow_device_transfer=True,
    )
    assert isinstance(rays, torch.Tensor)
    assert rays.shape == (1, 3)


def test_generate_image_points_returns_bare_tensor(sensor_device):
    """Verifies that generate_image_points produces a bare (H, W, 2) Tensor for a given grid size."""
    pts = F.generate_image_points(
        (2, 2), device=sensor_device, allow_device_transfer=True
    )
    assert isinstance(pts, torch.Tensor)
    assert pts.shape == (2, 2, 2)


def test_project_world_points_mean_pose_return_flag_matrix(
    ideal_projection, no_external, dynamic_pose
):
    """Verifies that project_world_points_mean_pose returns a WorldPointsToImagePointsReturn with correct dtypes and shapes."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0]], device=ideal_projection.focal_length.device
    )
    result = F.project_world_points_mean_pose(
        world_points,
        ideal_projection,
        no_external,
        (100, 80),
        dynamic_pose,
        start_timestamp_us=0,
        end_timestamp_us=10,
        return_T_sensor_world=True,
        return_valid_flag=True,
        return_valid_indices=True,
        return_timestamps=True,
        allow_device_transfer=True,
    )
    assert isinstance(result, WorldPointsToImagePointsReturn)
    assert result.T_sensor_world.shape == (1, 4, 4)
    assert result.valid_indices.dtype == torch.int64
    assert result.timestamps_us.dtype == torch.int64


def test_project_world_points_shutter_pose_return_flag_matrix(
    ideal_projection, no_external, dynamic_pose
):
    """Verifies that project_world_points_shutter_pose populates T_sensor_world and timestamps with correct shapes."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0]], device=ideal_projection.focal_length.device
    )
    result = F.project_world_points_shutter_pose(
        world_points,
        ideal_projection,
        no_external,
        (100, 80),
        ShutterType.ROLLING_TOP_TO_BOTTOM,
        dynamic_pose,
        start_timestamp_us=0,
        end_timestamp_us=10,
        return_T_sensor_world=True,
        return_valid_flag=True,
        return_valid_indices=True,
        return_timestamps=True,
        allow_device_transfer=True,
    )
    assert result.T_sensor_world.shape == (1, 4, 4)
    assert result.timestamps_us.shape == (1,)


def test_allow_device_transfer_false_raises_on_cpu_input(ideal_projection, no_external):
    """Verifies that passing CPU tensors with allow_device_transfer=False raises RuntimeError."""
    rays = torch.tensor([[0.0, 0.0, 1.0]], device="cpu")
    with pytest.raises(
        RuntimeError, match="CPU inputs require allow_device_transfer=True"
    ):
        F.camera_rays_to_image_points(rays, ideal_projection, no_external)


def test_consistency_with_kernel_layer(ideal_projection, no_external):
    """Verifies that functional camera_rays_to_image_points matches the Layer 0 kernel output exactly."""
    rays = torch.tensor([[0.1, 0.2, 1.0]], device=ideal_projection.focal_length.device)
    functional = F.camera_rays_to_image_points(
        rays, ideal_projection, no_external, allow_device_transfer=True
    )
    kernel_image_points, kernel_valid = kernel_ops.camera_rays_to_image_points(
        rays, ideal_projection, no_external, allow_device_transfer=True
    )
    assert torch.equal(functional.image_points, kernel_image_points)
    assert torch.equal(functional.valid_flag, kernel_valid)


def test_image_points_to_world_rays_static_return(
    ideal_projection, no_external, static_pose
):
    """Verifies that image_points_to_world_rays_static_pose returns a WorldRaysReturn with correct shapes and timestamp value."""
    result = F.image_points_to_world_rays_static_pose(
        ideal_projection.principal_point.reshape(1, 2),
        ideal_projection,
        no_external,
        static_pose,
        timestamp_us=2,
        return_T_sensor_world=True,
        return_timestamps=True,
        allow_device_transfer=True,
    )
    assert isinstance(result, WorldRaysReturn)
    assert result.world_rays.shape == (1, 6)
    assert result.T_sensor_world.shape == (1, 4, 4)
    assert result.timestamps_us.item() == 2
