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

"""Differentiable two-pose interpolation helpers shared by sensor kernel layers.

LERP translation + SLERP rotation with wxyz quaternion storage, used to unpack
a :class:`DynamicPose` for kernel dispatch.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .pose import DynamicPose
from .tensor_ops import to_dev


def unpack_dynamic_pose_components(
    dynamic_pose: DynamicPose,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Extract a DynamicPose into its four component tensors for kernel dispatch.

    Args:
        dynamic_pose: Time-varying dynamic pose holding start and end Pose objects.
        device: Target CUDA device.
        dtype: Target floating-point dtype.
        allow_device_transfer: Passed through to ``to_dev``; see its docstring.

    Returns:
        Four-tuple of (start_translation (3,), start_rotation (4,) wxyz,
        end_translation (3,), end_rotation (4,) wxyz) each on the target device.
    """
    return (
        to_dev(
            dynamic_pose.start_pose.translation, device, dtype, allow_device_transfer
        ),
        to_dev(dynamic_pose.start_pose.rotation, device, dtype, allow_device_transfer),
        to_dev(dynamic_pose.end_pose.translation, device, dtype, allow_device_transfer),
        to_dev(dynamic_pose.end_pose.rotation, device, dtype, allow_device_transfer),
    )


__all__ = [
    "unpack_dynamic_pose_components",
]
