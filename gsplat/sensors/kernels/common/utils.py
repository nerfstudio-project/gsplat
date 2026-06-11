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

"""Cross-layer tensor utilities shared by camera and sensor kernel bindings.

Provides quaternion-convention swizzles, SE(3) matrix construction, and
valid-ray index extraction used across the sensorlib kernel layer.
"""

from __future__ import annotations

import torch
from gsplat.geometry.functional import pose as _geom_pose
from torch import Tensor


def wxyz_to_xyzw(quat: Tensor) -> Tensor:
    """Reorder a wxyz quaternion to xyzw convention.

    Args:
        quat: (..., 4) quaternion tensor in [qw, qx, qy, qz] order.

    Returns:
        (..., 4) quaternion tensor in [qx, qy, qz, qw] order.
    """
    return torch.cat([quat[..., 1:], quat[..., :1]], dim=-1)


def xyzw_to_wxyz(quat: Tensor) -> Tensor:
    """Reorder an xyzw quaternion to wxyz convention.

    Args:
        quat: (..., 4) quaternion tensor in [qx, qy, qz, qw] order.

    Returns:
        (..., 4) quaternion tensor in [qw, qx, qy, qz] order.
    """
    return torch.cat([quat[..., 3:], quat[..., :3]], dim=-1)


def poses_to_matrix(
    translations: Tensor | None, rotations: Tensor | None
) -> Tensor | None:
    """Build SE(3) matrices from translations and wxyz quaternions.

    Args:
        translations: (N, 3) translation vectors, or None.
        rotations: (N, 4) quaternions in wxyz order, or None.

    Returns:
        (N, 4, 4) homogeneous SE(3) matrices, or None if either input is None.
    """
    if translations is None or rotations is None:
        return None
    return _geom_pose.se3pose_to_matrix(translations, wxyz_to_xyzw(rotations))


def valid_flags_to_indices(valid_flags: Tensor | None) -> Tensor | None:
    """Convert a boolean valid-ray mask to a flat index tensor.

    Args:
        valid_flags: (N,) boolean tensor marking valid rays, or None.

    Returns:
        (M,) int64 tensor of indices where valid_flags is True, or None if the
        input is None.
    """
    if valid_flags is None:
        return None
    return torch.nonzero(valid_flags, as_tuple=False).reshape(-1).to(torch.int64)


__all__ = [
    "poses_to_matrix",
    "valid_flags_to_indices",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]
