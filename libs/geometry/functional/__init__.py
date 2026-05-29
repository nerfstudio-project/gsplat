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

"""Public geometry API: quaternion, pose, and trajectory operators."""

from gsplat_geometry.functional.pose import (
    frame_transform_poses_tquat,
    se3pose_from_matrix,
    se3pose_inverse_transform_direction,
    se3pose_inverse_transform_point,
    se3pose_to_inverse_matrix,
    se3pose_to_matrix,
    se3pose_transform_direction,
    se3pose_transform_point,
    trajectory_get_rotation_2poses,
    trajectory_transform_point_1pose,
    trajectory_transform_point_2poses,
)
from gsplat_geometry.functional.quaternion import (
    quat_angular_distance,
    quat_conjugate,
    quat_from_axis_angle,
    quat_identity,
    quat_inverse,
    quat_lerp,
    quat_manifold_interp,
    quat_multiply,
    quat_normalize_safe,
    quat_rotate_vector,
    quat_slerp,
    quat_to_matrix,
)

__all__ = [
    "frame_transform_poses_tquat",
    "se3pose_from_matrix",
    "se3pose_inverse_transform_direction",
    "se3pose_inverse_transform_point",
    "se3pose_to_inverse_matrix",
    "se3pose_to_matrix",
    "se3pose_transform_direction",
    "se3pose_transform_point",
    "trajectory_get_rotation_2poses",
    "trajectory_transform_point_1pose",
    "trajectory_transform_point_2poses",
    "quat_angular_distance",
    "quat_conjugate",
    "quat_from_axis_angle",
    "quat_identity",
    "quat_inverse",
    "quat_lerp",
    "quat_manifold_interp",
    "quat_multiply",
    "quat_normalize_safe",
    "quat_rotate_vector",
    "quat_slerp",
    "quat_to_matrix",
]
