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

"""Tests for the LidarFrame observation container.

Verifies the dense ``(H, W, R)`` vs sparse ``(N, R)`` distinction (decided by
``model_element``), NaN/0.0 invalid markers, optional-property buffer
registration, the per-point-timestamp buffer, and that ``lidar_model`` is held
as a plain attribute rather than a buffer. ``forward`` raises NotImplementedError.
"""

import pytest
import torch

from gsplat_sensors.kernels.common import Pose
from gsplat_sensors.models.lidars import LidarFrame


def _static_pose(device):
    return Pose(
        translation=torch.zeros(3, device=device),
        rotation=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
    )


def test_dense_frame_shapes_and_flags(lidar_model):
    """Verify a dense (H, W, R) frame reports is_dense and the right point/return counts."""
    device = lidar_model.projection.row_elevations_rad.device
    h, w, r = 4, 6, 2
    frame = LidarFrame(
        frame_id="lidar",
        lidar_model=lidar_model,
        pose=_static_pose(device),
        timestamp_start_us=0,
        timestamp_end_us=100,
        distance_m=torch.zeros(h, w, r, device=device),
        intensity=torch.zeros(h, w, r, device=device),
        timestamp_us=torch.zeros(h, w, dtype=torch.int64, device=device),
    )
    assert frame.is_dense
    assert not frame.is_sparse
    assert frame.n_points == h * w
    assert frame.max_returns == r


def test_lidar_frame_rejects_invalid_dense_shapes(lidar_model):
    """Verify dense observations are validated before buffer registration."""
    device = lidar_model.projection.row_elevations_rad.device
    with pytest.raises(ValueError, match="dense distance_m"):
        LidarFrame(
            frame_id="lidar",
            lidar_model=lidar_model,
            pose=_static_pose(device),
            timestamp_start_us=0,
            timestamp_end_us=100,
            distance_m=torch.zeros(4, 2, device=device),
            intensity=torch.zeros(4, 2, device=device),
        )

    with pytest.raises(ValueError, match="dense timestamp_us"):
        LidarFrame(
            frame_id="lidar",
            lidar_model=lidar_model,
            pose=_static_pose(device),
            timestamp_start_us=0,
            timestamp_end_us=100,
            distance_m=torch.zeros(4, 6, 2, device=device),
            intensity=torch.zeros(4, 6, 2, device=device),
            timestamp_us=torch.zeros(24, dtype=torch.int64, device=device),
        )


def test_sparse_frame_shapes_and_flags(lidar_model):
    """Verify a sparse (N, R) frame with model_element reports is_sparse and N points."""
    device = lidar_model.projection.row_elevations_rad.device
    n, r = 5, 3
    frame = LidarFrame(
        frame_id="lidar",
        lidar_model=lidar_model,
        pose=_static_pose(device),
        timestamp_start_us=0,
        timestamp_end_us=100,
        distance_m=torch.zeros(n, r, device=device),
        intensity=torch.zeros(n, r, device=device),
        model_element=torch.zeros(n, 2, dtype=torch.int32, device=device),
    )
    assert frame.is_sparse
    assert not frame.is_dense
    assert frame.n_points == n
    assert frame.max_returns == r


def test_lidar_frame_rejects_invalid_sparse_shapes(lidar_model):
    """Verify sparse observations validate distance, element, and timestamp shapes."""
    device = lidar_model.projection.row_elevations_rad.device
    with pytest.raises(ValueError, match="sparse distance_m"):
        LidarFrame(
            frame_id="lidar",
            lidar_model=lidar_model,
            pose=_static_pose(device),
            timestamp_start_us=0,
            timestamp_end_us=100,
            distance_m=torch.zeros(5, 3, 1, device=device),
            intensity=torch.zeros(5, 3, 1, device=device),
            model_element=torch.zeros(5, 2, dtype=torch.int32, device=device),
        )

    with pytest.raises(ValueError, match="model_element"):
        LidarFrame(
            frame_id="lidar",
            lidar_model=lidar_model,
            pose=_static_pose(device),
            timestamp_start_us=0,
            timestamp_end_us=100,
            distance_m=torch.zeros(5, 3, device=device),
            intensity=torch.zeros(5, 3, device=device),
            model_element=torch.zeros(5, 3, dtype=torch.int32, device=device),
        )

    with pytest.raises(ValueError, match="sparse timestamp_us"):
        LidarFrame(
            frame_id="lidar",
            lidar_model=lidar_model,
            pose=_static_pose(device),
            timestamp_start_us=0,
            timestamp_end_us=100,
            distance_m=torch.zeros(5, 3, device=device),
            intensity=torch.zeros(5, 3, device=device),
            model_element=torch.zeros(5, 2, dtype=torch.int32, device=device),
            timestamp_us=torch.zeros(5, 1, dtype=torch.int64, device=device),
        )


def test_lidar_frame_rejects_intensity_shape_mismatch(lidar_model):
    """Verify intensity must match distance_m shape."""
    device = lidar_model.projection.row_elevations_rad.device
    with pytest.raises(ValueError, match="intensity"):
        LidarFrame(
            frame_id="lidar",
            lidar_model=lidar_model,
            pose=_static_pose(device),
            timestamp_start_us=0,
            timestamp_end_us=100,
            distance_m=torch.zeros(4, 6, 2, device=device),
            intensity=torch.zeros(4, 6, 1, device=device),
        )


def test_invalid_returns_use_nan_and_zero_markers(lidar_model):
    """Verify NaN distance and 0.0 intensity survive as the invalid-return markers."""
    device = lidar_model.projection.row_elevations_rad.device
    distance = torch.tensor([[1.0, float("nan")]], device=device)
    intensity = torch.tensor([[0.5, 0.0]], device=device)
    frame = LidarFrame(
        frame_id="lidar",
        lidar_model=lidar_model,
        pose=_static_pose(device),
        timestamp_start_us=0,
        timestamp_end_us=100,
        distance_m=distance,
        intensity=intensity,
        model_element=torch.zeros(1, 2, dtype=torch.int32, device=device),
    )
    assert torch.isnan(frame.distance_m[0, 1])
    assert frame.intensity[0, 1].item() == 0.0


def test_distance_intensity_and_optional_properties_are_buffers(lidar_model):
    """Verify distance_m, intensity, timestamp_us and optional properties register as buffers."""
    device = lidar_model.projection.row_elevations_rad.device
    n, r = 3, 1
    frame = LidarFrame(
        frame_id="lidar",
        lidar_model=lidar_model,
        pose=_static_pose(device),
        timestamp_start_us=0,
        timestamp_end_us=100,
        distance_m=torch.zeros(n, r, device=device),
        intensity=torch.zeros(n, r, device=device),
        model_element=torch.zeros(n, 2, dtype=torch.int32, device=device),
        timestamp_us=torch.zeros(n, dtype=torch.int64, device=device),
        optional_properties={"elongation": torch.zeros(n, r, device=device)},
    )
    buffers = dict(frame.named_buffers())
    assert "distance_m" in buffers
    assert "intensity" in buffers
    assert "timestamp_us" in buffers
    assert "elongation" in buffers
    assert "elongation" in frame.optional_properties


def test_optional_property_colliding_key_raises(lidar_model):
    """Verify an optional property whose key shadows an existing buffer raises ValueError."""
    device = lidar_model.projection.row_elevations_rad.device
    n, r = 3, 1
    with pytest.raises(ValueError, match="distance_m"):
        LidarFrame(
            frame_id="lidar",
            lidar_model=lidar_model,
            pose=_static_pose(device),
            timestamp_start_us=0,
            timestamp_end_us=100,
            distance_m=torch.zeros(n, r, device=device),
            intensity=torch.zeros(n, r, device=device),
            model_element=torch.zeros(n, 2, dtype=torch.int32, device=device),
            optional_properties={"distance_m": torch.zeros(n, r, device=device)},
        )


def test_lidar_model_is_not_a_buffer(lidar_model):
    """Verify lidar_model is held as a plain attribute, not a registered buffer."""
    device = lidar_model.projection.row_elevations_rad.device
    frame = LidarFrame(
        frame_id="lidar",
        lidar_model=lidar_model,
        pose=_static_pose(device),
        timestamp_start_us=0,
        timestamp_end_us=100,
        distance_m=torch.zeros(2, 1, device=device),
        intensity=torch.zeros(2, 1, device=device),
        model_element=torch.zeros(2, 2, dtype=torch.int32, device=device),
    )
    buffer_names = dict(frame.named_buffers())
    assert not any("lidar_model" in name for name in buffer_names)
    assert frame.lidar_model is lidar_model


def test_optional_timestamp_absent_by_default(lidar_model):
    """Verify timestamp_us is None when not provided and absent from the buffer set."""
    device = lidar_model.projection.row_elevations_rad.device
    frame = LidarFrame(
        frame_id="lidar",
        lidar_model=lidar_model,
        pose=_static_pose(device),
        timestamp_start_us=0,
        timestamp_end_us=100,
        distance_m=torch.zeros(2, 1, 1, device=device),
        intensity=torch.zeros(2, 1, 1, device=device),
    )
    assert frame.timestamp_us is None
    assert "timestamp_us" not in dict(frame.named_buffers())


def test_frame_forward_raises_not_implemented(lidar_model):
    """Verify LidarFrame.forward() raises NotImplementedError."""
    device = lidar_model.projection.row_elevations_rad.device
    frame = LidarFrame(
        frame_id="lidar",
        lidar_model=lidar_model,
        pose=_static_pose(device),
        timestamp_start_us=0,
        timestamp_end_us=100,
        distance_m=torch.zeros(2, 1, 1, device=device),
        intensity=torch.zeros(2, 1, 1, device=device),
    )
    with pytest.raises(NotImplementedError):
        frame.forward()
