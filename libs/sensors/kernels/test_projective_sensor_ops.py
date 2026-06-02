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

import itertools

import pytest
import torch

from gsplat_sensors.kernels.cameras import (
    REGISTERED_CAMERA_PROJECTIONS,
    REGISTERED_DISTORTIONS,
)
from gsplat_sensors.kernels import projective_sensor_ops


def test_dispatch_table_keys_match_registered_pairs():
    """Ensure every dispatch table covers exactly the Cartesian product of registered projections and distortions."""
    expected = set(
        itertools.product(REGISTERED_CAMERA_PROJECTIONS, REGISTERED_DISTORTIONS)
    )
    for table in projective_sensor_ops._DISPATCH_TABLES.values():
        assert set(table.keys()) == expected
        assert all(callable(value) for value in table.values())


def test_dispatch_raises_on_unregistered_pair(ideal_projection):
    """Confirm that passing external_distortion=None raises TypeError with a helpful message."""
    with pytest.raises(TypeError, match="NoExternalDistortion"):
        projective_sensor_ops.camera_rays_to_image_points(
            ideal_projection.focal_length.reshape(1, 2),
            ideal_projection,
            None,
        )


def test_dispatch_accepts_bivariate_windshield_pair(
    ideal_projection, windshield_distortion
):
    """Verify that camera_rays_to_image_points dispatches successfully for the bivariate windshield pair."""
    rays = torch.tensor([[0.0, 0.0, 1.0]], device=ideal_projection.focal_length.device)
    image_points, valid = projective_sensor_ops.camera_rays_to_image_points(
        rays,
        ideal_projection,
        windshield_distortion,
        allow_device_transfer=True,
    )
    assert image_points.shape == (1, 2)
    assert valid.item()
