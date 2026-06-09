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

"""Layer 0 spinning-LiDAR kernel bindings.

Exposes the differentiable LiDAR ops and the projection types / enums.
"""

from ._projection_validate import validate_lidar_projection
from .ops import (
    elements_to_sensor_angles,
    generate_spinning_lidar_rays,
    inverse_project_spinning_lidar,
    sensor_angles_to_sensor_rays,
    sensor_rays_to_sensor_angles,
)
from .types import (
    REGISTERED_LIDAR_PROJECTIONS,
    REGISTERED_LIDAR_PROJECTION_NAMES,
    RowOffsetStructuredSpinningLidarProjection,
    SpinningDirection,
    script_class_name,
)

__all__ = [
    "REGISTERED_LIDAR_PROJECTIONS",
    "REGISTERED_LIDAR_PROJECTION_NAMES",
    "RowOffsetStructuredSpinningLidarProjection",
    "SpinningDirection",
    "elements_to_sensor_angles",
    "generate_spinning_lidar_rays",
    "inverse_project_spinning_lidar",
    "script_class_name",
    "sensor_angles_to_sensor_rays",
    "sensor_rays_to_sensor_angles",
    "validate_lidar_projection",
]
