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

"""Trainable LiDAR frame extending the base ``Frame`` with observation data.

Adds LiDAR-specific model and observation data (distance, intensity, optional
per-point timestamps, and arbitrary optional per-ray properties) on top of the
common ``Frame`` state (frame id, pose, timestamps, metadata).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import Tensor

from ...kernels.common.pose import DynamicPose, Pose
from ..common.frame import Frame, FrameId

if TYPE_CHECKING:
    from .lidar_model import LidarModel


def _validate_lidar_observations(
    distance_m: Tensor,
    intensity: Tensor,
    model_element: Tensor | None,
    timestamp_us: Tensor | None,
) -> None:
    if intensity.shape != distance_m.shape:
        raise ValueError(
            "intensity must have the same shape as distance_m, got "
            f"{tuple(intensity.shape)} and {tuple(distance_m.shape)}"
        )

    if model_element is None:
        if distance_m.ndim != 3:
            raise ValueError(
                f"dense distance_m must be (H, W, R), got {tuple(distance_m.shape)}"
            )
        if timestamp_us is not None and tuple(timestamp_us.shape) != tuple(
            distance_m.shape[:2]
        ):
            raise ValueError(
                "dense timestamp_us must be (H, W), got "
                f"{tuple(timestamp_us.shape)} for distance_m {tuple(distance_m.shape)}"
            )
        return

    if distance_m.ndim != 2:
        raise ValueError(
            f"sparse distance_m must be (N, R), got {tuple(distance_m.shape)}"
        )
    if model_element.ndim != 2 or model_element.shape[1] != 2:
        raise ValueError(
            f"model_element must be (N, 2), got {tuple(model_element.shape)}"
        )
    if model_element.shape[0] != distance_m.shape[0]:
        raise ValueError(
            "model_element first dimension must match sparse distance_m, got "
            f"{model_element.shape[0]} and {distance_m.shape[0]}"
        )
    if timestamp_us is not None and tuple(timestamp_us.shape) != (distance_m.shape[0],):
        raise ValueError(
            "sparse timestamp_us must be (N,), got "
            f"{tuple(timestamp_us.shape)} for distance_m {tuple(distance_m.shape)}"
        )


class LidarFrame(Frame):
    """A trainable LiDAR frame extending :class:`Frame`.

    Adds the LiDAR model and observation data. Inherits the common frame
    properties (``frame_id``, ``pose``, timestamps, ``metadata``) from
    :class:`Frame`.

    Data Format:
        - Dense: ``distance_m`` shape ``(H, W, R)`` — full range image with ``R``
          returns per ray; element indices are implicit grid indices.
        - Sparse: ``distance_m`` shape ``(N, R)`` — filtered measurements with
          explicit ``model_element`` ``(N, 2)`` indices.
        - ``R``: maximum number of returns per ray.
        - Invalid returns are marked ``NaN`` (``distance_m``) or ``0.0``
          (``intensity``).

    The dense-vs-sparse distinction is decided solely by whether ``model_element``
    is provided.

    Frame-level vs point-level timestamps:
        - ``timestamp_start_us`` / ``timestamp_end_us`` (from ``Frame``): the frame
          capture interval.
        - ``timestamp_us`` (LiDAR-specific): optional per-point timestamps within
          the frame interval, for spinning LiDARs with rolling shutter.

    Attributes:
        lidar_model: The LiDAR model used to capture this frame (Layer 2). Held as
            a plain attribute, not a buffer.
        distance_m: Distance measurements in meters (float32), ``(H, W, R)`` or
            ``(N, R)``. Invalid returns are ``NaN``.
        intensity: Intensity values (float32), same shape as ``distance_m``.
            Invalid returns are ``0.0``.
        model_element: ``(N, 2)`` int ``[row, col]`` element indices for the sparse
            format, or ``None`` for the dense format.
        timestamp_us: Optional per-point timestamps (int64), or ``None``.
        optional_properties: Dict of optional per-ray properties (e.g.
            ``elongation``), each registered as a buffer.
    """

    distance_m: Tensor
    intensity: Tensor
    model_element: Tensor | None
    timestamp_us: Tensor | None

    def __init__(
        self,
        frame_id: FrameId,
        lidar_model: "LidarModel",
        pose: Pose | DynamicPose,
        timestamp_start_us: int,
        timestamp_end_us: int,
        distance_m: Tensor,
        intensity: Tensor,
        model_element: Tensor | None = None,
        timestamp_us: Tensor | None = None,
        optional_properties: dict[str, Tensor] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize the LiDAR frame with model and observation data.

        Args:
            frame_id: Unique identifier for this frame.
            lidar_model: The LiDAR model used to capture this frame.
            pose: Learnable ``Pose`` or ``DynamicPose`` object.
            timestamp_start_us: Frame start timestamp in microseconds.
            timestamp_end_us: Frame end timestamp in microseconds.
            distance_m: Distance measurements, ``(H, W, R)`` (dense) or ``(N, R)``
                (sparse). Invalid returns are ``NaN``.
            intensity: Intensity values, same shape as ``distance_m``. Invalid
                returns are ``0.0``.
            model_element: ``(N, 2)`` element indices selecting the sparse format,
                or ``None`` for the dense format.
            timestamp_us: Optional per-point timestamps.
            optional_properties: Dict of optional per-ray properties, each
                registered as a buffer.
            metadata: Optional metadata dictionary.
        """
        _validate_lidar_observations(distance_m, intensity, model_element, timestamp_us)
        super().__init__(
            frame_id=frame_id,
            pose=pose,
            timestamp_start_us=timestamp_start_us,
            timestamp_end_us=timestamp_end_us,
            metadata=metadata,
        )
        self.lidar_model = lidar_model

        self.register_buffer("distance_m", distance_m)
        self.register_buffer("intensity", intensity)

        if model_element is not None:
            self.register_buffer("model_element", model_element)
        else:
            self.model_element = None

        if timestamp_us is not None:
            self.register_buffer("timestamp_us", timestamp_us)
        else:
            self.timestamp_us = None

        self._optional_property_names: list[str] = []
        if optional_properties:
            for key, value in optional_properties.items():
                if hasattr(self, key):
                    raise ValueError(
                        f"optional property {key!r} collides with an existing "
                        "LidarFrame attribute or buffer; choose a different name."
                    )
                self.register_buffer(key, value)
                self._optional_property_names.append(key)

    def forward(self, *args, **kwargs):
        """Forward pass is not implemented for LiDAR frames.

        Specific use cases (point-cloud generation, pose optimization) are handled
        by callers or wrappers rather than a frame ``forward`` method.
        """
        raise NotImplementedError(
            "LidarFrame.forward() is not implemented. Specific use cases "
            "(e.g. point-cloud generation, pose optimization) are handled by callers."
        )

    @property
    def is_sparse(self) -> bool:
        """True if the frame uses the sparse format (explicit ``model_element``)."""
        return self.model_element is not None

    @property
    def is_dense(self) -> bool:
        """True if the frame uses the dense ``(H, W, R)`` format."""
        return self.model_element is None

    @property
    def n_points(self) -> int:
        """Number of measurement points (``N`` sparse, ``H * W`` dense)."""
        if self.is_sparse:
            return self.distance_m.shape[0]
        return self.distance_m.shape[0] * self.distance_m.shape[1]

    @property
    def max_returns(self) -> int:
        """Maximum number of returns per ray (``R``)."""
        return self.distance_m.shape[-1]

    @property
    def optional_properties(self) -> dict[str, Tensor]:
        """Dict of optional per-ray properties registered as buffers."""
        return {name: getattr(self, name) for name in self._optional_property_names}


LidarFrameSet = dict[FrameId, LidarFrame]


__all__ = ["LidarFrame", "LidarFrameSet"]
