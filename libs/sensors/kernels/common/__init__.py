# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common types shared between camera and sensor kernels.

Exports pose dataclasses (Pose, Trajectory, DynamicPose) and cross-layer
utility helpers (quaternion swizzles, SE(3) matrix construction, valid-index
extraction).
"""

from .pose import DynamicPose, Pose, Trajectory
from .utils import (
    poses_to_matrix,
    valid_flags_to_indices,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

__all__ = [
    "DynamicPose",
    "Pose",
    "Trajectory",
    "poses_to_matrix",
    "valid_flags_to_indices",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]
