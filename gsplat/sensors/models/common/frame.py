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

"""Base Frame class for trainable sensor frames with learnable pose.

This module provides the abstract base class for all sensor frame types,
containing common properties shared by cameras and LiDARs.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, TypeAlias

from torch import nn

from ...kernels.common.pose import DynamicPose, Pose

FrameId: TypeAlias = str


class Frame(nn.Module):
    """Base class for trainable sensor frames with learnable pose.

    Common frame properties shared by all sensor types (cameras, LiDARs).
    Subclasses add sensor-specific models and observation data.

    Use Cases:
    - Static pose: Fixed sensor position, global shutter, or when motion is negligible
    - Dynamic pose: Rolling shutter sensors, moving sensors, or temporal pose variation

    Attributes:
        frame_id: Unique string identifier for this frame. Must be unique within a
            frame group (e.g., ImageFrameGroup) to allow fast lookup and access.
        pose: Learnable pose or dynamic pose (T_sensor_world or T_world_sensor).
            Use ``Pose`` for static transformations and ``DynamicPose`` for
            time-varying transformations (e.g., rolling shutter).
        timestamp_start_us: Frame start timestamp in microseconds (int64).
        timestamp_end_us: Frame end timestamp in microseconds (int64).
            For global shutter: ``start == end``.
            For rolling shutter: ``start < end`` (capture duration).
        metadata: Flexible metadata dictionary for sensor-specific extras.
    """

    def __init__(
        self,
        frame_id: FrameId,
        pose: Pose | DynamicPose,
        timestamp_start_us: int,
        timestamp_end_us: int,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize base frame with common properties.

        Args:
            frame_id: Unique identifier for this frame.
            pose: Learnable pose (Pose) or dynamic pose (DynamicPose) object.
            timestamp_start_us: Frame start timestamp in microseconds.
            timestamp_end_us: Frame end timestamp in microseconds.
            metadata: Optional metadata dictionary.
        """
        super().__init__()
        self.frame_id = str(frame_id)
        self.pose = pose
        self.timestamp_start_us = timestamp_start_us
        self.timestamp_end_us = timestamp_end_us
        self.metadata = metadata or {}

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass — behavior depends on use case (e.g., projection, rendering).

        Must be implemented by subclasses for their specific use case.
        """
        raise NotImplementedError("Frame.forward() must be implemented by subclasses")

    def _apply(self, fn):
        # Propagate device/dtype transfers to the non-Parameter Pose tensors.
        super()._apply(fn)
        if isinstance(self.pose, Pose):
            self.pose = Pose(
                translation=fn(self.pose.translation),
                rotation=fn(self.pose.rotation),
            )
        elif isinstance(self.pose, DynamicPose):
            self.pose = DynamicPose(
                start_pose=Pose(
                    translation=fn(self.pose.start_pose.translation),
                    rotation=fn(self.pose.start_pose.rotation),
                ),
                end_pose=Pose(
                    translation=fn(self.pose.end_pose.translation),
                    rotation=fn(self.pose.end_pose.rotation),
                ),
            )
        return self

    @property
    def is_rolling_shutter(self) -> bool:
        """Return True if start and end timestamps differ (rolling shutter)."""
        return self.timestamp_start_us != self.timestamp_end_us

    @property
    def frame_duration_us(self) -> int:
        """Get frame capture duration in microseconds."""
        return self.timestamp_end_us - self.timestamp_start_us


__all__ = ["Frame", "FrameId"]
