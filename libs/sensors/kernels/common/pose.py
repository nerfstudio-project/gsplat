# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python dataclasses for pose and trajectory types.

These dataclasses provide the Python-side representation for pose data passed
to the sensorlib CUDA kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Pose:
    """Static SE(3) pose with translation and quaternion rotation.

    Attributes:
        translation: (3,) translation vector [x, y, z].
        rotation: (4,) quaternion in wxyz format [qw, qx, qy, qz].
    """

    translation: Tensor
    rotation: Tensor


@dataclass
class Trajectory:
    """Piecewise pose trajectory defined by control poses at normalized times.

    Attributes:
        control_poses: Sequence of Pose objects defining the trajectory keyframes.
        control_count: Number of control poses (must equal len(control_poses)).
        control_times: (N,) tensor of normalized times in [0, 1] for each control pose.
    """

    control_poses: tuple[Pose, ...]
    control_count: int
    control_times: Tensor


@dataclass
class DynamicPose:
    """Time-varying pose over normalized frame time [0, 1].

    Represents linear interpolation between exactly two poses: start (t=0)
    and end (t=1). Used for rolling-shutter compensation and motion-during-exposure.

    Attributes:
        start_pose: Pose at t=0 (start of exposure or scan).
        end_pose: Pose at t=1 (end of exposure or scan).
    """

    start_pose: Pose
    end_pose: Pose

    @staticmethod
    def from_static_pose(pose: Pose) -> "DynamicPose":
        """Create a DynamicPose from a static (constant) pose.

        ``start_pose`` aliases the input; ``end_pose`` is a new ``Pose`` holding
        cloned translation and rotation tensors. The two slots are therefore
        distinct ``Pose`` instances, so kernels that interpolate from start to
        end no longer see the same Python object at both endpoints.

        Args:
            pose: Static pose to replicate at both t=0 and t=1.

        Returns:
            DynamicPose with ``start_pose=pose`` (aliased) and ``end_pose``
            holding cloned tensors.
        """
        end = Pose(
            translation=pose.translation.clone(),
            rotation=pose.rotation.clone(),
        )
        return DynamicPose(start_pose=pose, end_pose=end)

    def to_trajectory(self) -> Trajectory:
        """Convert to a two-keyframe Trajectory for kernel consumption.

        Returns:
            Trajectory with control poses at t=0 and t=1.
        """
        device = self.start_pose.translation.device
        dtype = self.start_pose.translation.dtype
        control_times = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        return Trajectory(
            control_poses=(self.start_pose, self.end_pose),
            control_count=2,
            control_times=control_times,
        )


__all__ = ["DynamicPose", "Pose", "Trajectory"]
