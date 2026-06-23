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

"""Conformance tests for the single-key spinning-LiDAR dispatch.

The LiDAR dispatch is the sibling of the camera ``projective_sensor_ops``
Cartesian-product dispatch: keyed on a single projection type rather than a
``(projection, distortion)`` pair. These tests verify that every dispatch table
resolves all registered LiDAR projections for all five ops and that an
unregistered projection raises, without touching the camera conformance test.
"""

import pytest
import torch

from gsplat_sensors.kernels.lidars import dispatch as lidar_dispatch
from gsplat_sensors.kernels.lidars.types import REGISTERED_LIDAR_PROJECTIONS

_EXPECTED_OPS = {
    "sensor_rays_to_sensor_angles",
    "sensor_angles_to_sensor_rays",
    "elements_to_sensor_angles",
    "generate_spinning_lidar_rays",
    "inverse_project_spinning_lidar",
}


def test_dispatch_tables_cover_all_five_ops():
    """Ensure the LiDAR dispatch exposes exactly the five spinning-LiDAR ops."""
    assert set(lidar_dispatch._DISPATCH_TABLES) == _EXPECTED_OPS


def test_each_table_keys_match_registered_projections():
    """Ensure every dispatch table is keyed by exactly the registered projections."""
    expected = set(REGISTERED_LIDAR_PROJECTIONS)
    for table in lidar_dispatch._DISPATCH_TABLES.values():
        assert set(table.keys()) == expected
        assert all(callable(value) for value in table.values())


def test_lookup_resolves_a_projection_instance(lidar_projection_from_json):
    """Ensure ``_lookup`` returns a callable for a real projection instance on every table."""
    projection = lidar_projection_from_json("generic")
    for table in lidar_dispatch._DISPATCH_TABLES.values():
        backend = lidar_dispatch._lookup(table, projection)
        assert callable(backend)


def test_lookup_raises_on_unregistered_projection():
    """Confirm an unknown projection type raises TypeError with a helpful message."""

    class _NotALidarProjection:
        pass

    table = lidar_dispatch._DISPATCH_TABLES["sensor_rays_to_sensor_angles"]
    with pytest.raises(TypeError, match="Unsupported LiDAR projection"):
        lidar_dispatch._lookup(table, _NotALidarProjection())


def test_public_wrappers_listed_in_all():
    """Ensure every op has a public wrapper exported from the dispatch module."""
    for op in _EXPECTED_OPS:
        assert op in lidar_dispatch.__all__
        assert callable(getattr(lidar_dispatch, op))


def test_wrapper_routes_through_lookup_to_backend(lidar_projection_from_json):
    """Ensure a public wrapper resolves and invokes the backend for a real projection."""
    projection = lidar_projection_from_json("generic")
    rays = torch.tensor([[1.0, 0.0, 0.0]], device="cuda")
    angles = lidar_dispatch.sensor_rays_to_sensor_angles(
        rays, projection, allow_device_transfer=True
    )
    assert angles.shape == (1, 2)
    assert torch.allclose(angles, torch.zeros_like(angles), atol=1e-6)
