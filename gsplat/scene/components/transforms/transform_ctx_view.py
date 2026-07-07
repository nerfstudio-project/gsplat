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

"""Component transform context view helpers."""

from __future__ import annotations

from collections.abc import KeysView, Mapping

import torch


class TransformCtxView:
    """Component-scoped view into a scene transform context buffer."""

    def __init__(
        self,
        ctx_buffer: Mapping[str, torch.Tensor],
        ranges: Mapping[str, tuple[int, int]],
    ) -> None:
        self._buf = ctx_buffer
        self._ranges = ranges

    def __getitem__(self, key: str) -> torch.Tensor:
        """Return a named transform-context tensor view for this component."""
        offset, count = self._ranges[key]
        return self._buf[key][offset : offset + count]

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        """Write into this component's context slice in place."""
        if not isinstance(value, torch.Tensor):
            raise TypeError("TransformCtxView values must be torch.Tensor instances")
        target = self[key]
        if value.shape != target.shape:
            raise ValueError(
                f"context value for {key!r} must have shape {tuple(target.shape)}, "
                f"got {tuple(value.shape)}"
            )
        if value.dtype != target.dtype:
            raise TypeError(
                f"context value for {key!r} dtype {value.dtype} must match "
                f"existing dtype {target.dtype}"
            )
        if value.device != target.device:
            raise ValueError(
                f"context value for {key!r} must be on device {target.device}, "
                f"got {value.device}"
            )

        replacement = value
        if key == "poses":
            replacement = self._normalize_pose_rows(value)
        elif key == "pose_times":
            self._validate_pose_times(value)

        with torch.no_grad():
            target.copy_(replacement)

    @staticmethod
    def _normalize_pose_rows(poses: torch.Tensor) -> torch.Tensor:
        """Return finite pose rows with unit-norm ``xyzw`` quaternions."""
        if poses.ndim != 2 or poses.shape[1] != 7:
            raise ValueError(
                "poses context must have shape (N, 7) with layout "
                "[tx, ty, tz, qx, qy, qz, qw]"
            )
        if not poses.dtype.is_floating_point:
            raise TypeError("poses context must have a floating dtype")
        if not bool(torch.isfinite(poses).all().item()):
            raise ValueError("pose rows must be finite")
        rotations = poses[:, 3:7]
        norms = torch.linalg.vector_norm(rotations, dim=-1, keepdim=True)
        eps = torch.finfo(poses.dtype).eps
        if bool((norms <= eps).any().item()):
            raise ValueError("pose quaternions must have non-zero norm")
        return torch.cat([poses[:, :3], rotations / norms], dim=-1)

    @staticmethod
    def _validate_pose_times(pose_times: torch.Tensor) -> None:
        """Validate finite, sorted pose times."""
        times = pose_times.reshape(-1)
        if times.dtype.is_floating_point and not bool(
            torch.isfinite(times).all().item()
        ):
            raise ValueError("pose_times must be finite")
        if times.numel() > 1 and bool(torch.any(times[1:] < times[:-1]).item()):
            raise ValueError("pose_times must be sorted non-decreasing")

    def __contains__(self, key: object) -> bool:
        return key in self._ranges

    def keys(self) -> KeysView[str]:
        """Return transform-context names available for this component."""
        return self._ranges.keys()

    def get(self, key: str, default: torch.Tensor | None = None) -> torch.Tensor | None:
        """Return a context tensor view or ``default`` if ``key`` is absent."""
        if key not in self._ranges:
            return default
        return self[key]
