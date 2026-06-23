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

"""ImageFrame and ImageFrameGroup for camera observations.

This module provides the ImageFrame class that extends Frame with a camera model
and an image tensor registered as an nn.Module buffer.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from ...kernels.common.pose import DynamicPose, Pose
from ..common.frame import Frame, FrameId
from .camera_model import CameraModel


class ImageFrame(Frame):
    """A trainable camera frame that pairs a CameraModel with an image observation.

    Extends Frame with camera-specific model and image data.  The image tensor
    is registered as an nn.Module buffer so it participates in .to() / .cuda()
    and state_dict serialization alongside learnable pose parameters.

    Attributes:
        camera_model: CameraModel used to capture this frame.
        image: Image tensor of shape (H, W, C), registered as a buffer.

    Example:
        frame = ImageFrame(
            frame_id="front_cam_0",
            camera_model=camera,
            pose=static_pose,
            timestamp_start_us=1_000_000,
            timestamp_end_us=1_033_333,
            image=image_tensor,  # (H, W, 3)
        )
        print(frame.height, frame.width, frame.channels)
    """

    # Declared here so type checkers know the buffer attribute exists.
    image: Tensor

    def __init__(
        self,
        frame_id: FrameId,
        camera_model: CameraModel,
        pose: Pose | DynamicPose,
        timestamp_start_us: int,
        timestamp_end_us: int,
        image: Tensor,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize the image frame.

        Args:
            frame_id: Unique identifier for this frame.
            camera_model: Camera model used to capture this frame.
            pose: Static pose (Pose) or interpolated pose (DynamicPose).
            timestamp_start_us: Frame start timestamp in microseconds.
            timestamp_end_us: Frame end timestamp in microseconds.
            image: Image tensor of shape (H, W, C).
            metadata: Optional key-value metadata dictionary.
        """
        if image.ndim != 3:
            raise ValueError(f"image must be (H, W, C), got {tuple(image.shape)}")
        super().__init__(frame_id, pose, timestamp_start_us, timestamp_end_us, metadata)
        self.camera_model = camera_model
        self.register_buffer("image", image)

    def forward(self, *args, **kwargs):
        """Not implemented; ImageFrame is a data container, not a callable model."""
        raise NotImplementedError("ImageFrame.forward() is not implemented")

    @property
    def height(self) -> int:
        """Get image height in pixels."""
        return self.image.shape[0]

    @property
    def width(self) -> int:
        """Get image width in pixels."""
        return self.image.shape[1]

    @property
    def channels(self) -> int:
        """Get number of image channels."""
        return self.image.shape[2]


# Type alias: a keyed collection of ImageFrame objects indexed by FrameId.
ImageFrameGroup = dict[FrameId, ImageFrame]

__all__ = ["ImageFrame", "ImageFrameGroup"]
