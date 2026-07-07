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

"""Rigid-body transform op."""

from __future__ import annotations

import torch

from .transform_op import TransformOp


class RigidTransformOp(TransformOp):
    """Rigid local-to-world transform op backed by per-component SE(3) tracks.

    Context ``poses`` rows use ``[tx, ty, tz, qx, qy, qz, qw]`` with an
    ``xyzw`` quaternion. Gaussian ``splats["quats"]`` remains in renderer
    ``wxyz`` layout and is converted at the geometry boundary.
    """

    def validate_ctx(self, ctx: dict[str, torch.Tensor], count: int) -> None:
        """Validate a component's rigid pose-track context."""
        # `count` is the component's Gaussian row count. Rigid ctx is
        # per-component/keyframe (`N_c` pose rows), not per-Gaussian, so this
        # op validates poses against pose_times rather than ctx rows vs count.
        del count
        if "poses" not in ctx:
            raise KeyError("RigidTransformOp requires 'poses' in context")
        if "pose_times" not in ctx:
            raise KeyError("RigidTransformOp requires 'pose_times' in context")
        poses = ctx["poses"]
        pose_times = ctx["pose_times"]
        if poses.ndim == 1:
            if poses.shape != (7,):
                raise ValueError(
                    "poses must have shape (7,) with layout "
                    f"[tx, ty, tz, qx, qy, qz, qw], got {tuple(poses.shape)}"
                )
            n_poses = 1
        elif poses.ndim == 2:
            if poses.shape[1] != 7:
                raise ValueError(
                    "poses must have shape (N, 7) with layout "
                    f"[tx, ty, tz, qx, qy, qz, qw], got {tuple(poses.shape)}"
                )
            n_poses = poses.shape[0]
        else:
            raise ValueError(
                "poses must have shape (7,) or (N, 7) with layout "
                f"[tx, ty, tz, qx, qy, qz, qw], got {tuple(poses.shape)}"
            )
        if n_poses < 1:
            raise ValueError("RigidTransformOp requires at least one pose")
        if pose_times.reshape(-1).shape != (n_poses,):
            raise ValueError(
                f"pose_times must have {n_poses} rows to match poses, "
                f"got {tuple(pose_times.shape)}"
            )
        if n_poses > 1:
            times = pose_times.reshape(-1)
            if bool(torch.any(times[1:] < times[:-1]).item()):
                raise ValueError("pose_times must be sorted non-decreasing")

    def apply(self, collected: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply interpolated SE(3) tracks to scene tensors."""
        pose_t, pose_q_xyzw = self.interpolate_poses(collected)
        component_index = collected["component_index"].to(dtype=torch.long)
        per_gaussian_t = pose_t[component_index]
        per_gaussian_q_xyzw = pose_q_xyzw[component_index]

        transformed = dict(collected)
        transformed.update(
            self.apply_pose_to_splats(
                collected,
                per_gaussian_t,
                per_gaussian_q_xyzw,
            )
        )
        return transformed

    def interpolate_poses(
        self,
        collected: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-component translations and ``xyzw`` rotations."""
        from gsplat.geometry.functional import se3_interpolate_tracks

        return se3_interpolate_tracks(
            collected["poses"][:, :3],
            collected["poses"][:, 3:],
            collected["pose_times"],
            collected["pose_offsets"],
            collected["pose_counts"],
            collected["t"],
        )

    def apply_pose_to_splats(
        self,
        splats: dict[str, torch.Tensor],
        pose_t: torch.Tensor,
        pose_q_xyzw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply a resolved pose to Gaussian means and quaternions."""
        from gsplat.geometry.functional import (
            quat_multiply,
            se3pose_transform_point,
        )
        from gsplat.sensors.kernels.common.utils import wxyz_to_xyzw, xyzw_to_wxyz

        means = se3pose_transform_point(
            pose_t,
            pose_q_xyzw,
            splats["means"],
        )
        local_q_xyzw = wxyz_to_xyzw(splats["quats"])
        world_q_xyzw = quat_multiply(pose_q_xyzw, local_q_xyzw)
        return {
            "means": means,
            "quats": xyzw_to_wxyz(world_q_xyzw),
        }
