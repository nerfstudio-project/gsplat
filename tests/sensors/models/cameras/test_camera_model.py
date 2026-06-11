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

# Row mapping: design-tests-opencvpinhole.md §6 rows 41-45 are covered here.

import io

import torch
from torch import nn

from gsplat.sensors.models import (
    CameraModel,
    ImageFrame,
)


def test_camera_is_nn_module_and_to_device(pinhole_model):
    """Assert CameraModel is an nn.Module and .to() returns the same model type."""
    assert isinstance(pinhole_model, nn.Module)
    moved = pinhole_model.to(pinhole_model.projection.focal_length.device)
    assert isinstance(moved, CameraModel)
    assert (
        moved.projection.focal_length.device
        == pinhole_model.projection.focal_length.device
    )


def test_camera_model_state_dict_and_pickle(pinhole_model):
    """Assert state_dict is a dict and the model round-trips through torch.save/load."""
    assert isinstance(pinhole_model.state_dict(), dict)
    buf = io.BytesIO()
    torch.save(pinhole_model, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)
    assert loaded.projection.resolution == pinhole_model.projection.resolution


def test_image_frame_image_buffer_registered(pinhole_model, static_pose):
    """Assert the image tensor is registered as an nn.Module buffer on ImageFrame."""
    frame = ImageFrame("cam", pinhole_model, static_pose, 0, 0, torch.zeros(80, 100, 3))
    assert "image" in dict(frame.named_buffers())


def test_image_frame_to_device_moves_image_and_pose(pinhole_model, static_pose):
    """Assert .to(device) moves both the image buffer and the pose tensors."""
    frame = ImageFrame("cam", pinhole_model, static_pose, 0, 0, torch.zeros(80, 100, 3))
    frame = frame.to(frame.image.device)
    assert frame.pose.translation.device == frame.image.device
    assert frame.camera_model.projection.focal_length.device == frame.image.device


def test_image_frame_height_width_channels(pinhole_model, static_pose):
    """Assert height, width, and channels properties reflect the image tensor shape."""
    frame = ImageFrame("cam", pinhole_model, static_pose, 0, 0, torch.zeros(80, 100, 3))
    assert (frame.height, frame.width, frame.channels) == (80, 100, 3)


def test_world_points_to_pixels_respects_return_valid_flag(pinhole_model, static_pose):
    """Assert valid_flag and valid_indices are None when return_valid_flag=False, populated otherwise."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0]], device=pinhole_model.projection.focal_length.device
    )
    result = pinhole_model.world_points_to_pixels_static_pose(
        world_points, static_pose, return_valid_flag=False
    )
    assert result.valid_flag is None
    assert result.valid_indices is None

    result = pinhole_model.world_points_to_pixels_static_pose(
        world_points, static_pose, return_valid_flag=True, return_valid_indices=True
    )
    assert result.valid_flag is not None
    assert result.valid_indices is not None


def test_world_points_to_pixels_static_pose_filters_invalid_once(
    pinhole_model, static_pose
):
    """Assert behind-camera points are excluded from pixels output and marked False in valid_flag."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        device=pinhole_model.projection.focal_length.device,
    )

    result = pinhole_model.world_points_to_pixels_static_pose(
        world_points,
        static_pose,
        return_valid_flag=True,
        return_valid_indices=True,
    )

    assert result.pixels.shape == (1, 2)
    assert result.valid_flag.tolist() == [True, False]
    assert result.valid_indices.tolist() == [0]


def test_world_points_to_pixels_static_pose_return_all_keeps_input_alignment(
    pinhole_model, static_pose
):
    """Assert return_all_projections=True preserves input-point count even for invalid points."""
    world_points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        device=pinhole_model.projection.focal_length.device,
    )

    result = pinhole_model.world_points_to_pixels_static_pose(
        world_points,
        static_pose,
        return_valid_flag=True,
        return_valid_indices=True,
        return_all_projections=True,
    )

    assert result.pixels.shape == (2, 2)
    assert result.valid_flag.tolist() == [True, False]
    assert result.valid_indices.tolist() == [0]
