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

"""Tests for the Layer 2 stateful LidarModel.

Verifies nn.Module behavior, device moves carrying the projection tables,
FOV validity with the fov_eps_factor slack, relative-frame-times in [0, 1],
agreement between model methods and the functional ops, and the
NotImplementedError forward contract.
"""

import pytest
import torch
from torch import nn

from gsplat_sensors import functional as F
from gsplat_sensors.kernels.common import DynamicPose, Pose
from gsplat_sensors.models import (
    LidarModel,
    RowOffsetStructuredSpinningLidarProjection,
    SpinningDirection,
)


def _model_pose(device, dtype=torch.float32):
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


def test_lidar_model_is_nn_module(lidar_model):
    """Verify LidarModel is an nn.Module and the projection is not registered as a buffer."""
    assert isinstance(lidar_model, nn.Module)
    buffer_names = dict(lidar_model.named_buffers())
    assert not any("projection" in name for name in buffer_names)


def test_lidar_model_to_moves_projection_tables(lidar_model):
    """Verify .cpu()/.to() moves the projection angle tables and returns a LidarModel."""
    original_device = lidar_model.projection.row_elevations_rad.device
    moved = lidar_model.cpu()
    assert isinstance(moved, LidarModel)
    assert moved.projection.row_elevations_rad.device.type == "cpu"
    assert moved.projection.column_azimuths_rad.device.type == "cpu"
    back = moved.to(original_device)
    assert back.projection.row_elevations_rad.device == original_device
    assert back.projection.column_azimuths_rad.device == original_device


def test_lidar_projection_types_reexported_from_models():
    """Verify public LiDAR projection types are available from gsplat_sensors.models."""
    assert RowOffsetStructuredSpinningLidarProjection is not None
    assert SpinningDirection.CLOCKWISE.value == 0


def test_lidar_model_requires_projection():
    """Verify constructing a LidarModel with projection=None raises TypeError."""
    with pytest.raises(TypeError, match="projection=None"):
        LidarModel(projection=None)


def test_forward_raises_not_implemented(lidar_model):
    """Verify LidarModel.forward() raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        lidar_model.forward()


def test_valid_sensor_angles_returns_bool_within_fov(lidar_model):
    """Verify valid_sensor_angles returns a bool mask and accepts an in-FOV angle."""
    angles = torch.tensor(
        [[0.0, 0.0]], device=lidar_model.projection.row_elevations_rad.device
    )
    valid = lidar_model.valid_sensor_angles(angles)
    assert valid.dtype == torch.bool
    assert valid.shape == (1,)
    assert bool(valid[0].item())


def test_valid_sensor_angles_rejects_above_vertical_fov(lidar_model):
    """Verify an elevation far above the vertical FOV is marked invalid."""
    above = lidar_model.projection.fov_vert_start_rad + 0.5
    angles = torch.tensor(
        [[above, 0.0]], device=lidar_model.projection.row_elevations_rad.device
    )
    assert not bool(lidar_model.valid_sensor_angles(angles)[0].item())


def test_valid_sensor_angles_does_not_wrap_vertical_fov(lidar_model):
    """Verify an elevation one full turn below the vertical FOV stays invalid."""
    bottom = (
        lidar_model.projection.fov_vert_start_rad
        - lidar_model.projection.fov_vert_span_rad
    )
    below = bottom - 2.0 * torch.pi
    angles = torch.tensor(
        [[below, 0.0]], device=lidar_model.projection.row_elevations_rad.device
    )
    assert not bool(lidar_model.valid_sensor_angles(angles)[0].item())


def test_valid_sensor_angles_fov_eps_slack(lidar_projection_from_json):
    """Verify the fov_eps_factor admits an angle exactly at the vertical FOV boundary.

    A larger fov_eps_factor widens the boundary slack; an angle placed just past
    the nominal boundary (by less than the slack) is admitted only with the
    larger factor.
    """
    projection = lidar_projection_from_json("generic")
    bottom = projection.fov_vert_start_rad - projection.fov_vert_span_rad
    device = projection.row_elevations_rad.device
    # one float32 ULP below the bottom boundary.
    just_below = torch.tensor(bottom, device=device, dtype=torch.float32)
    just_below = torch.nextafter(
        just_below, torch.tensor(-1e9, device=device, dtype=torch.float32)
    )
    angles = torch.stack([just_below, torch.zeros((), device=device)]).reshape(1, 2)

    tight = LidarModel(projection=projection, fov_eps_factor=0.0)
    loose = LidarModel(projection=projection, fov_eps_factor=1e6)
    assert not bool(tight.valid_sensor_angles(angles)[0].item())
    assert bool(loose.valid_sensor_angles(angles)[0].item())


def test_relative_frame_times_in_unit_interval(lidar_model):
    """Verify sensor_angles_relative_frame_times returns values in [0, 1]."""
    n_cols = lidar_model.n_columns
    cols = torch.arange(0, n_cols, max(1, n_cols // 50), device="cuda")
    azimuths = lidar_model.projection.column_azimuths_rad[cols]
    elevations = torch.full_like(azimuths, lidar_model.projection.row_elevations_rad[0])
    angles = torch.stack([elevations, azimuths], dim=1)
    times = lidar_model.sensor_angles_relative_frame_times(angles)
    assert times.shape == (angles.shape[0],)
    assert bool((times >= 0.0).all())
    assert bool((times <= 1.0).all())


def test_model_methods_agree_with_functional(lidar_projection_from_json, sensor_device):
    """Verify model ray/angle conversions agree with the functional ops."""
    projection = lidar_projection_from_json("generic")
    model = LidarModel(projection=projection)
    world_points = torch.tensor(
        [[5.0, 0.0, 0.0], [4.0, 1.0, -0.5], [6.0, -1.0, 0.3]],
        device=sensor_device,
    )
    pose = _model_pose(sensor_device)

    model_out = model.world_points_to_sensor_angles_shutter_pose(
        world_points, pose, start_timestamp_us=0, end_timestamp_us=100000
    )
    func_out = F.inverse_project_spinning_lidar(
        projection,
        world_points,
        pose,
        start_timestamp_us=0,
        end_timestamp_us=100000,
        return_valid_flag=True,
    )
    assert model_out.sensor_angles.shape[0] == world_points.shape[0]
    assert torch.allclose(
        model_out.sensor_angles, func_out.sensor_angles[func_out.valid_flag], atol=1e-6
    )


def test_world_points_to_sensor_angles_filters_invalid_by_default(
    lidar_projection_from_json, sensor_device
):
    """Verify model inverse projection filters invalid points unless all projections are requested."""
    projection = lidar_projection_from_json("generic")
    model = LidarModel(projection=projection)
    # A mix of in-FOV and out-of-FOV (straight up) points so some are filtered.
    world_points = torch.tensor(
        [[5.0, 0.0, 0.0], [4.0, 1.0, -0.5], [0.0, 0.0, 5.0]],
        device=sensor_device,
    )
    pose = _model_pose(sensor_device)

    all_out = model.world_points_to_sensor_angles_shutter_pose(
        world_points,
        pose,
        start_timestamp_us=0,
        end_timestamp_us=100000,
        return_valid_flag=True,
        return_all_projections=True,
    )
    filtered = model.world_points_to_sensor_angles_shutter_pose(
        world_points,
        pose,
        start_timestamp_us=0,
        end_timestamp_us=100000,
    )

    assert all_out.sensor_angles.shape[0] == world_points.shape[0]
    assert int(all_out.valid_flag.sum().item()) < world_points.shape[0]
    assert filtered.sensor_angles.shape[0] == int(all_out.valid_flag.sum().item())
    assert torch.allclose(
        filtered.sensor_angles, all_out.sensor_angles[all_out.valid_flag], atol=1e-6
    )


def test_model_sensor_ray_angle_round_trip(lidar_model):
    """Verify model angle->ray->angle round-trips through the model methods."""
    angles = torch.tensor(
        [[0.1, 0.5], [-0.2, -1.0]],
        device=lidar_model.projection.row_elevations_rad.device,
    )
    rays = lidar_model.sensor_angles_to_sensor_rays(angles).sensor_rays
    recovered = lidar_model.sensor_rays_to_sensor_angles(rays).sensor_angles
    assert torch.allclose(recovered, angles, atol=1e-5)


def test_model_sensor_rays_to_sensor_angles_zero_ray_no_nan(lidar_model):
    """A zero-length ray through the normalized=False path must not yield NaN."""
    device = lidar_model.projection.row_elevations_rad.device
    rays = torch.zeros((1, 3), device=device)
    # normalized=False routes through the guarded F.normalize denominator.
    out = lidar_model.sensor_rays_to_sensor_angles(rays, normalized=False)
    assert not torch.isnan(out.sensor_angles).any()


def test_spinning_direction_property(lidar_model):
    """Verify the spinning_direction property returns a SpinningDirection enum."""
    assert isinstance(lidar_model.spinning_direction, SpinningDirection)
