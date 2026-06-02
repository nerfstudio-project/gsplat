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

# Row mapping: design-tests-opencvpinhole.md §6 rows 31-40 are covered here.

import pytest
import torch

from gsplat_sensors.functional.return_types import ImagePointsReturn
from gsplat_sensors.kernels.cameras import ShutterType
from gsplat_sensors.models import CameraModel


def test_camera_model_initialization(pinhole_model, ideal_projection, no_external):
    """Assert CameraModel stores projection, external distortion, resolution, and shutter type correctly."""
    assert pinhole_model.projection is ideal_projection
    assert pinhole_model.external_distortion is no_external
    assert pinhole_model.resolution == (100, 80)
    assert pinhole_model.shutter_type == ShutterType.GLOBAL


def test_camera_rays_to_image_points_basic(pinhole_model):
    """Assert an optical-axis ray projects to the principal point with a valid flag."""
    rays = torch.tensor(
        [[0.0, 0.0, 1.0]], device=pinhole_model.projection.focal_length.device
    )
    result = pinhole_model.camera_rays_to_image_points(rays, return_jacobians=True)
    assert isinstance(result, ImagePointsReturn)
    assert torch.allclose(
        result.image_points, pinhole_model.projection.principal_point.reshape(1, 2)
    )
    assert result.valid_flag.item()
    assert result.jacobians.shape == (1, 2, 3)


def test_camera_rays_to_image_points_jacobian_reference(pinhole_model):
    """Assert the analytic Jacobian matches the hand-computed reference values."""
    rays = torch.tensor(
        [[0.1, 0.2, 1.0]], device=pinhole_model.projection.focal_length.device
    )
    result = pinhole_model.camera_rays_to_image_points(rays, return_jacobians=True)
    expected = torch.tensor(
        [[[100.0, 0.0, -10.0], [0.0, 120.0, -24.0]]],
        device=rays.device,
    )
    assert torch.allclose(result.jacobians, expected, atol=1e-4)


def test_camera_rays_to_image_points_jacobians_returns_detached_value(pinhole_model):
    """Assert image_points has no grad_fn even when input rays require grad."""
    rays = torch.tensor(
        [[0.1, 0.2, 1.0]],
        device=pinhole_model.projection.focal_length.device,
        requires_grad=True,
    )
    result = pinhole_model.camera_rays_to_image_points(rays, return_jacobians=True)
    assert result.image_points.requires_grad is False
    assert result.image_points.grad_fn is None


def test_image_points_to_camera_rays_roundtrip(pinhole_model):
    """Assert image_points -> camera_rays -> image_points round-trips within 1e-4 tolerance."""
    image_points = torch.tensor(
        [[50.0, 40.0], [60.0, 52.0]],
        device=pinhole_model.projection.focal_length.device,
    )
    rays = pinhole_model.image_points_to_camera_rays(image_points)
    roundtrip = pinhole_model.camera_rays_to_image_points(rays).image_points
    assert torch.allclose(roundtrip, image_points, atol=1e-4)


def test_world_points_to_image_points_static_pose(pinhole_model, static_pose):
    """Assert static-pose projection filters invalid points and returns optional T_sensor_world and timestamps."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        device=pinhole_model.projection.focal_length.device,
    )
    result = pinhole_model.world_points_to_image_points_static_pose(
        world_points,
        static_pose,
        timestamp_us=11,
        return_T_sensor_world=True,
        return_valid_flag=True,
        return_valid_indices=True,
        return_timestamps=True,
    )
    assert result.image_points.shape == (1, 2)
    assert result.valid_flag.tolist() == [True, False]
    assert result.valid_indices.tolist() == [0]
    assert result.timestamps_us.tolist() == [11]
    assert result.T_sensor_world.shape == (1, 4, 4)


def test_world_points_to_image_points_shutter_pose(
    ideal_projection, no_external, dynamic_pose
):
    """Assert rolling-shutter projection produces per-point image_points, valid_flag, and timestamps."""
    model = CameraModel(
        ideal_projection, no_external, (100, 80), ShutterType.ROLLING_TOP_TO_BOTTOM
    )
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0]], device=ideal_projection.focal_length.device
    )
    result = model.world_points_to_image_points_shutter_pose(
        world_points,
        dynamic_pose,
        start_timestamp_us=0,
        end_timestamp_us=10,
        return_valid_flag=True,
        return_timestamps=True,
    )
    assert result.image_points.shape == (1, 2)
    assert result.valid_flag.shape == (1,)
    assert result.timestamps_us.shape == (1,)


@pytest.mark.parametrize(
    "shutter_type",
    [
        ShutterType.ROLLING_TOP_TO_BOTTOM,
        ShutterType.ROLLING_LEFT_TO_RIGHT,
        ShutterType.ROLLING_BOTTOM_TO_TOP,
        ShutterType.ROLLING_RIGHT_TO_LEFT,
        ShutterType.GLOBAL,
    ],
)
def test_image_points_relative_frame_times(ideal_projection, no_external, shutter_type):
    """Assert relative frame times are in [0, 1] for all supported shutter types."""
    model = CameraModel(ideal_projection, no_external, (100, 80), shutter_type)
    image_points = torch.tensor(
        [[0.5, 0.5], [99.5, 79.5]], device=ideal_projection.focal_length.device
    )
    times = model.image_points_relative_frame_times(image_points)
    assert times.shape == (2,)
    assert torch.all((times >= 0) & (times <= 1))


def test_pixels_image_points_conversions(pinhole_model):
    """Assert pixels_to_image_points adds 0.5 and image_points_to_pixels floors to int32."""
    pixels = torch.tensor([[0, 1]], device=pinhole_model.projection.focal_length.device)
    image_points = pinhole_model.pixels_to_image_points(pixels)
    assert torch.equal(
        image_points, torch.tensor([[0.5, 1.5]], device=image_points.device)
    )
    assert torch.equal(
        pinhole_model.image_points_to_pixels(image_points), pixels.to(torch.int32)
    )


def test_transform_scale_offset_anisotropic(pinhole_model):
    """Assert transform applies scale to focal length, offset to principal point, and anisotropic scale to resolution."""
    scaled = pinhole_model.transform(0.5)
    assert torch.allclose(
        scaled.projection.focal_length, pinhole_model.projection.focal_length * 0.5
    )
    offset = pinhole_model.transform(1.0, (1.0, 2.0))
    assert torch.allclose(
        offset.projection.principal_point,
        pinhole_model.projection.principal_point
        - torch.tensor([1.0, 2.0], device=offset.projection.principal_point.device),
    )
    anisotropic = pinhole_model.transform((0.5, 0.25))
    assert anisotropic.resolution == (50, 20)


def test_transform_returns_new_object_no_in_place_mutation(pinhole_model):
    """Assert transform returns a new CameraModel without mutating the original focal length."""
    original = pinhole_model.projection.focal_length.clone()
    transformed = pinhole_model.transform(0.5)
    assert transformed is not pinhole_model
    assert torch.equal(pinhole_model.projection.focal_length, original)


def test_transform_preserves_grad(pinhole_model):
    """Assert gradients flow back through transform to the original focal length with correct scale."""
    pinhole_model.projection.focal_length.requires_grad_(True)
    transformed = pinhole_model.transform(0.5)
    transformed.projection.focal_length.sum().backward()
    assert torch.allclose(
        pinhole_model.projection.focal_length.grad,
        torch.full_like(pinhole_model.projection.focal_length, 0.5),
    )
