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

"""Tests for opt-in value-level validation of spinning-LiDAR projections."""

from __future__ import annotations

import pytest
import torch

from gsplat.sensors.kernels.lidars import validate_lidar_projection
from gsplat.sensors.kernels.lidars.types import (
    RowOffsetStructuredSpinningLidarProjection,
)


class _UnknownProjection:
    pass


def _build_projection(
    row_elevations: torch.Tensor,
    column_azimuths: torch.Tensor,
    row_offsets: torch.Tensor | None,
) -> RowOffsetStructuredSpinningLidarProjection:
    has_offsets = row_offsets is not None
    offsets = (
        row_offsets
        if has_offsets
        else torch.zeros((0,), device=row_elevations.device, dtype=torch.float32)
    )
    # FOV scalars are fixed finite constants (not derived from the tables) so a
    # non-finite table entry reaches validate_lidar_projection rather than the
    # C++ constructor's scalar finiteness check.
    return RowOffsetStructuredSpinningLidarProjection(
        row_elevations,
        column_azimuths,
        offsets,
        0.2,
        0.4,
        -3.14159,
        6.28318,
        0,
        has_offsets,
    )


def test_validate_lidar_projection_accepts_finite_tables(lidar_projection_from_json):
    """Verify a well-formed projection passes validation without raising."""
    validate_lidar_projection(lidar_projection_from_json("generic"))


def test_validate_lidar_projection_rejects_unknown_projection():
    with pytest.raises(TypeError, match="Unknown LiDAR projection class"):
        validate_lidar_projection(_UnknownProjection())


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_validate_lidar_projection_rejects_non_finite_row_elevations(
    sensor_device, bad_value
):
    """Verify nan/inf in row_elevations_rad raises with a field-scoped message."""
    row_elevations = torch.tensor([-0.1, 0.0, 0.1], device=sensor_device)
    row_elevations[1] = bad_value
    column_azimuths = torch.tensor([-0.2, 0.0, 0.2], device=sensor_device)
    projection = _build_projection(row_elevations, column_azimuths, None)
    with pytest.raises(ValueError, match=r"row_elevations_rad\["):
        validate_lidar_projection(projection)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_validate_lidar_projection_rejects_non_finite_column_azimuths(
    sensor_device, bad_value
):
    """Verify nan/inf in column_azimuths_rad raises with a field-scoped message."""
    row_elevations = torch.tensor([-0.1, 0.0, 0.1], device=sensor_device)
    column_azimuths = torch.tensor([-0.2, 0.0, 0.2], device=sensor_device)
    column_azimuths[2] = bad_value
    projection = _build_projection(row_elevations, column_azimuths, None)
    with pytest.raises(ValueError, match=r"column_azimuths_rad\["):
        validate_lidar_projection(projection)


def test_validate_lidar_projection_rejects_non_finite_row_offsets(sensor_device):
    """Verify nan in row_azimuth_offsets_rad raises only when offsets are present."""
    row_elevations = torch.tensor([-0.1, 0.0, 0.1], device=sensor_device)
    column_azimuths = torch.tensor([-0.2, 0.0, 0.2], device=sensor_device)
    row_offsets = torch.tensor([0.01, float("nan"), 0.03], device=sensor_device)
    projection = _build_projection(row_elevations, column_azimuths, row_offsets)
    with pytest.raises(ValueError, match=r"row_azimuth_offsets_rad\["):
        validate_lidar_projection(projection)
