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

"""Python wrappers for native CUDA pose operations with PyTorch autograd support.

All functions expect batched torch tensors with shape (N, D) where N is the batch size.
For single inputs, add a batch dimension: tensor.reshape(1, -1)
"""

from typing import Any, Iterable

import torch

from . import _backend


def _expect_tensor(name: str, x: object) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    return x


def _require_cuda(name: str, t: torch.Tensor) -> None:
    if t.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor; got device {t.device}")


def _require_float32_or_float64(name: str, t: torch.Tensor) -> None:
    if t.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have dtype float32 or float64, got {t.dtype}")


def _require_float32(name: str, t: torch.Tensor) -> None:
    if t.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype float32, got {t.dtype}")


def _same_device(names_tensors: Iterable[tuple[str, torch.Tensor]]) -> torch.device:
    pairs = list(names_tensors)
    if not pairs:
        raise ValueError("internal error: empty tensor group")
    _, ref = pairs[0]
    dev = ref.device
    for name, t in pairs[1:]:
        if t.device != dev:
            raise ValueError(
                f"{name} must be on the same device as {pairs[0][0]} ({dev}); "
                f"got {t.device}"
            )
    return dev


def _validate_se3_vec3_inputs(
    translation: torch.Tensor,
    rotation: torch.Tensor,
    vec: torch.Tensor,
    vec_name: str,
) -> None:
    translation = _expect_tensor("translation", translation)
    rotation = _expect_tensor("rotation", rotation)
    vec = _expect_tensor(vec_name, vec)
    for n, t in (
        ("translation", translation),
        ("rotation", rotation),
        (vec_name, vec),
    ):
        _require_cuda(n, t)
    _require_float32_or_float64("translation", translation)
    if rotation.dtype != translation.dtype or vec.dtype != translation.dtype:
        raise TypeError(
            f"translation, rotation, and {vec_name} must have the same dtype; "
            f"got {translation.dtype}, {rotation.dtype}, {vec.dtype}"
        )
    _same_device(
        [("translation", translation), ("rotation", rotation), (vec_name, vec)]
    )
    if translation.dim() != 2 or translation.shape[1] != 3:
        raise ValueError(
            f"translation must have shape (N, 3); got {tuple(translation.shape)}"
        )
    if rotation.dim() != 2 or rotation.shape[1] != 4:
        raise ValueError(
            f"rotation must have shape (N, 4) xyzw; got {tuple(rotation.shape)}"
        )
    if vec.dim() != 2 or vec.shape[1] != 3:
        raise ValueError(f"{vec_name} must have shape (N, 3); got {tuple(vec.shape)}")
    n = translation.shape[0]
    if rotation.shape[0] != n or vec.shape[0] != n:
        raise ValueError(
            f"translation, rotation, and {vec_name} must share the same batch "
            f"size N; got N={n}, rotation rows={rotation.shape[0]}, "
            f"{vec_name} rows={vec.shape[0]}"
        )


def _validate_se3_pose_pair(translation: torch.Tensor, rotation: torch.Tensor) -> None:
    translation = _expect_tensor("translation", translation)
    rotation = _expect_tensor("rotation", rotation)
    _require_cuda("translation", translation)
    _require_cuda("rotation", rotation)
    _require_float32_or_float64("translation", translation)
    if rotation.dtype != translation.dtype:
        raise TypeError(
            "translation and rotation must have the same dtype; "
            f"got {translation.dtype} vs {rotation.dtype}"
        )
    if translation.device != rotation.device:
        raise ValueError(
            f"translation and rotation must be on the same device; "
            f"got {translation.device} vs {rotation.device}"
        )
    if translation.dim() != 2 or translation.shape[1] != 3:
        raise ValueError(
            f"translation must have shape (N, 3); got {tuple(translation.shape)}"
        )
    if rotation.dim() != 2 or rotation.shape[1] != 4:
        raise ValueError(
            f"rotation must have shape (N, 4) xyzw; got {tuple(rotation.shape)}"
        )
    if rotation.shape[0] != translation.shape[0]:
        raise ValueError(
            "translation and rotation must share the same batch size N; "
            f"got {translation.shape[0]} vs {rotation.shape[0]}"
        )


def _validate_trajectory_time_tensor(
    name: str, t: torch.Tensor, n: int, ref: torch.Tensor
) -> None:
    _expect_tensor(name, t)
    _require_cuda(name, t)
    _require_float32(name, t)
    if t.device != ref.device:
        raise ValueError(
            f"{name} must be on the same CUDA device as poses; got {t.device}"
        )
    if (t.dim() == 1 and t.shape[0] == n) or (t.dim() == 2 and t.shape == (n, 1)):
        return
    raise ValueError(
        f"{name} must have shape (N,) or (N, 1) with N={n}; got {tuple(t.shape)}"
    )


def _validate_trajectory_2poses_transform(
    trans0: torch.Tensor,
    rot0: torch.Tensor,
    time0: torch.Tensor,
    trans1: torch.Tensor,
    rot1: torch.Tensor,
    time1: torch.Tensor,
    point: torch.Tensor,
    query_time: torch.Tensor,
) -> None:
    trans0 = _expect_tensor("trans0", trans0)
    rot0 = _expect_tensor("rot0", rot0)
    trans1 = _expect_tensor("trans1", trans1)
    rot1 = _expect_tensor("rot1", rot1)
    point = _expect_tensor("point", point)
    for n, t in (
        ("trans0", trans0),
        ("rot0", rot0),
        ("trans1", trans1),
        ("rot1", rot1),
        ("point", point),
    ):
        _require_cuda(n, t)
        _require_float32(n, t)
    _same_device(
        [
            ("trans0", trans0),
            ("rot0", rot0),
            ("trans1", trans1),
            ("rot1", rot1),
            ("point", point),
        ]
    )
    n = point.shape[0]
    if trans0.dim() != 2 or trans0.shape != (n, 3):
        raise ValueError(f"trans0 must be (N, 3) with N={n}; got {tuple(trans0.shape)}")
    if rot0.dim() != 2 or rot0.shape != (n, 4):
        raise ValueError(f"rot0 must be (N, 4) with N={n}; got {tuple(rot0.shape)}")
    if trans1.dim() != 2 or trans1.shape != (n, 3):
        raise ValueError(f"trans1 must be (N, 3) with N={n}; got {tuple(trans1.shape)}")
    if rot1.dim() != 2 or rot1.shape != (n, 4):
        raise ValueError(f"rot1 must be (N, 4) with N={n}; got {tuple(rot1.shape)}")
    if point.dim() != 2 or point.shape[1] != 3:
        raise ValueError(f"point must be (N, 3); got {tuple(point.shape)}")
    time0 = _expect_tensor("time0", time0)
    time1 = _expect_tensor("time1", time1)
    query_time = _expect_tensor("query_time", query_time)
    _validate_trajectory_time_tensor("time0", time0, n, trans0)
    _validate_trajectory_time_tensor("time1", time1, n, trans0)
    _validate_trajectory_time_tensor("query_time", query_time, n, trans0)


def _validate_trajectory_2poses_rotation(
    trans0: torch.Tensor,
    rot0: torch.Tensor,
    time0: torch.Tensor,
    trans1: torch.Tensor,
    rot1: torch.Tensor,
    time1: torch.Tensor,
    query_time: torch.Tensor,
) -> None:
    trans0 = _expect_tensor("trans0", trans0)
    rot0 = _expect_tensor("rot0", rot0)
    trans1 = _expect_tensor("trans1", trans1)
    rot1 = _expect_tensor("rot1", rot1)
    for n, t in (
        ("trans0", trans0),
        ("rot0", rot0),
        ("trans1", trans1),
        ("rot1", rot1),
    ):
        _require_cuda(n, t)
        _require_float32(n, t)
    _same_device(
        [("trans0", trans0), ("rot0", rot0), ("trans1", trans1), ("rot1", rot1)]
    )
    n = trans0.shape[0]
    if trans0.dim() != 2 or trans0.shape != (n, 3):
        raise ValueError(f"trans0 must be (N, 3); got {tuple(trans0.shape)}")
    if rot0.dim() != 2 or rot0.shape != (n, 4):
        raise ValueError(f"rot0 must be (N, 4); got {tuple(rot0.shape)}")
    if trans1.dim() != 2 or trans1.shape != (n, 3):
        raise ValueError(f"trans1 must be (N, 3); got {tuple(trans1.shape)}")
    if rot1.dim() != 2 or rot1.shape != (n, 4):
        raise ValueError(f"rot1 must be (N, 4); got {tuple(rot1.shape)}")
    if rot0.shape[0] != n or trans1.shape[0] != n or rot1.shape[0] != n:
        raise ValueError("trans0, rot0, trans1, rot1 must share the same batch size")
    time0 = _expect_tensor("time0", time0)
    time1 = _expect_tensor("time1", time1)
    query_time = _expect_tensor("query_time", query_time)
    _validate_trajectory_time_tensor("time0", time0, n, trans0)
    _validate_trajectory_time_tensor("time1", time1, n, trans0)
    _validate_trajectory_time_tensor("query_time", query_time, n, trans0)


def _validate_trajectory_1pose(
    trans: torch.Tensor,
    rot: torch.Tensor,
    time: torch.Tensor,
    point: torch.Tensor,
    query_time: torch.Tensor,
) -> None:
    trans = _expect_tensor("trans", trans)
    rot = _expect_tensor("rot", rot)
    point = _expect_tensor("point", point)
    for n, t in (("trans", trans), ("rot", rot), ("point", point)):
        _require_cuda(n, t)
        _require_float32(n, t)
    _same_device([("trans", trans), ("rot", rot), ("point", point)])
    n = point.shape[0]
    if trans.dim() != 2 or trans.shape != (n, 3):
        raise ValueError(f"trans must be (N, 3) with N={n}; got {tuple(trans.shape)}")
    if rot.dim() != 2 or rot.shape != (n, 4):
        raise ValueError(f"rot must be (N, 4) with N={n}; got {tuple(rot.shape)}")
    if point.dim() != 2 or point.shape[1] != 3:
        raise ValueError(f"point must be (N, 3); got {tuple(point.shape)}")
    time = _expect_tensor("time", time)
    query_time = _expect_tensor("query_time", query_time)
    _validate_trajectory_time_tensor("time", time, n, trans)
    _validate_trajectory_time_tensor("query_time", query_time, n, trans)


# ============================================================================
# Autograd Function Wrappers for Differentiable Operations
# ============================================================================


class SE3PoseTransformPointFunction(torch.autograd.Function):
    """Differentiable SE3 point transformation."""

    @staticmethod
    def forward(
        ctx, translation: torch.Tensor, rotation: torch.Tensor, point: torch.Tensor
    ) -> torch.Tensor:
        translation = translation.contiguous()
        rotation = rotation.contiguous()
        point = point.contiguous()
        result = _backend._GEOMETRY_CUDA.se3pose_transform_point(
            translation, rotation, point
        )
        ctx.save_for_backward(translation, rotation, point, result)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        translation, rotation, point, _result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        (
            grad_translation,
            grad_rotation,
            grad_point,
        ) = _backend._GEOMETRY_CUDA.se3pose_transform_point_bwd(
            translation, rotation, point, grad_result
        )
        return grad_translation, grad_rotation, grad_point


class SE3PoseTransformDirectionFunction(torch.autograd.Function):
    """Differentiable SE3 direction transformation."""

    @staticmethod
    def forward(
        ctx, translation: torch.Tensor, rotation: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        translation = translation.contiguous()
        rotation = rotation.contiguous()
        direction = direction.contiguous()
        result = _backend._GEOMETRY_CUDA.se3pose_transform_direction(
            translation, rotation, direction
        )
        ctx.save_for_backward(translation, rotation, direction, result)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        translation, rotation, direction, _result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        (
            grad_translation,
            grad_rotation,
            grad_direction,
        ) = _backend._GEOMETRY_CUDA.se3pose_transform_direction_bwd(
            translation, rotation, direction, grad_result
        )
        return grad_translation, grad_rotation, grad_direction


class SE3PoseInverseTransformPointFunction(torch.autograd.Function):
    """Differentiable SE3 inverse point transformation."""

    @staticmethod
    def forward(
        ctx, translation: torch.Tensor, rotation: torch.Tensor, point: torch.Tensor
    ) -> torch.Tensor:
        translation = translation.contiguous()
        rotation = rotation.contiguous()
        point = point.contiguous()
        result = _backend._GEOMETRY_CUDA.se3pose_inverse_transform_point(
            translation, rotation, point
        )
        ctx.save_for_backward(translation, rotation, point, result)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        translation, rotation, point, _result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        (
            grad_translation,
            grad_rotation,
            grad_point,
        ) = _backend._GEOMETRY_CUDA.se3pose_inverse_transform_point_bwd(
            translation, rotation, point, grad_result
        )
        return grad_translation, grad_rotation, grad_point


class SE3PoseInverseTransformDirectionFunction(torch.autograd.Function):
    """Differentiable SE3 inverse direction transformation."""

    @staticmethod
    def forward(
        ctx, translation: torch.Tensor, rotation: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        translation = translation.contiguous()
        rotation = rotation.contiguous()
        direction = direction.contiguous()
        result = _backend._GEOMETRY_CUDA.se3pose_inverse_transform_direction(
            translation, rotation, direction
        )
        ctx.save_for_backward(translation, rotation, direction, result)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        translation, rotation, direction, _result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        (
            grad_translation,
            grad_rotation,
            grad_direction,
        ) = _backend._GEOMETRY_CUDA.se3pose_inverse_transform_direction_bwd(
            translation, rotation, direction, grad_result
        )
        return grad_translation, grad_rotation, grad_direction


class SE3PoseFromMatrixFunction(torch.autograd.Function):
    """Differentiable 4x4 matrix to SE3 pose (translation, rotation) conversion."""

    @staticmethod
    def forward(ctx, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        matrix = matrix.contiguous()
        N = matrix.shape[0]
        if N > 0:
            translation, rotation = _backend._GEOMETRY_CUDA.se3pose_from_matrix(matrix)
        else:
            translation = torch.zeros((N, 3), device=matrix.device, dtype=matrix.dtype)
            rotation = torch.zeros((N, 4), device=matrix.device, dtype=matrix.dtype)

        ctx.save_for_backward(matrix)
        ctx.N = N
        return translation, rotation

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_translation = grad_outputs[0]
        grad_rotation = grad_outputs[1]
        matrix = ctx.saved_tensors[0]
        grad_translation = grad_translation.contiguous()
        grad_rotation = grad_rotation.contiguous()
        if ctx.N > 0:
            grad_matrix = _backend._GEOMETRY_CUDA.se3pose_from_matrix_bwd(
                matrix, grad_translation, grad_rotation
            )
        else:
            grad_matrix = torch.zeros_like(matrix)
        return grad_matrix


class SE3PoseToMatrixFunction(torch.autograd.Function):
    """Differentiable SE3 pose to 4x4 matrix conversion."""

    @staticmethod
    def forward(ctx, translation: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        translation = translation.contiguous()
        rotation = rotation.contiguous()
        N = translation.shape[0]
        result = _backend._GEOMETRY_CUDA.se3pose_to_matrix(translation, rotation)
        ctx.save_for_backward(translation, rotation, result)
        ctx.N = N
        return result.reshape(N, 4, 4)

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        translation, rotation, result = ctx.saved_tensors
        grad_result = grad_result.reshape(ctx.N, 16).contiguous()
        grad_translation, grad_rotation = _backend._GEOMETRY_CUDA.se3pose_to_matrix_bwd(
            translation, rotation, grad_result
        )
        return grad_translation, grad_rotation


class SE3PoseToInverseMatrixFunction(torch.autograd.Function):
    """Differentiable SE3 pose to inverse 4x4 matrix conversion."""

    @staticmethod
    def forward(
        ctx,
        translation: torch.Tensor,
        rotation: torch.Tensor,
        wxyz_format: bool = False,
    ) -> torch.Tensor:
        translation = translation.contiguous()
        rotation = rotation.contiguous()
        N = translation.shape[0]
        result = _backend._GEOMETRY_CUDA.se3pose_to_inverse_matrix(
            translation, rotation, wxyz_format
        )
        ctx.save_for_backward(translation, rotation, result)
        ctx.N = N
        ctx.wxyz_format = wxyz_format
        return result.reshape(N, 4, 4)

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        translation, rotation, result = ctx.saved_tensors
        grad_result = grad_result.reshape(ctx.N, 16).contiguous()
        (
            grad_translation,
            grad_rotation,
        ) = _backend._GEOMETRY_CUDA.se3pose_to_inverse_matrix_bwd(
            translation, rotation, ctx.wxyz_format, grad_result
        )
        return grad_translation, grad_rotation, None


class TrajectoryTransformPoint2PosesFunction(torch.autograd.Function):
    """Differentiable 2-pose trajectory point transformation."""

    @staticmethod
    def forward(
        ctx,
        trans0: torch.Tensor,
        rot0: torch.Tensor,
        time0: torch.Tensor,
        trans1: torch.Tensor,
        rot1: torch.Tensor,
        time1: torch.Tensor,
        point: torch.Tensor,
        query_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trans0 = trans0.contiguous()
        rot0 = rot0.contiguous()
        time0 = time0.contiguous()
        trans1 = trans1.contiguous()
        rot1 = rot1.contiguous()
        time1 = time1.contiguous()
        point = point.contiguous()
        query_time = query_time.contiguous()
        (
            result_point,
            result_out_of_bounds,
        ) = _backend._GEOMETRY_CUDA.trajectory_transform_point_2poses(
            trans0,
            rot0,
            time0,
            trans1,
            rot1,
            time1,
            point,
            query_time,
        )
        ctx.save_for_backward(
            trans0,
            rot0,
            time0,
            trans1,
            rot1,
            time1,
            point,
            query_time,
            result_point,
            result_out_of_bounds,
        )
        return result_point, result_out_of_bounds

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result_point = grad_outputs[0]
        # grad_result_out_of_bounds = grad_outputs[1]  # out_of_bounds is no_diff
        (
            trans0,
            rot0,
            time0,
            trans1,
            rot1,
            time1,
            point,
            query_time,
            result_point,
            result_out_of_bounds,
        ) = ctx.saved_tensors
        grad_result_point = grad_result_point.contiguous()
        grad_trans0 = torch.empty_like(trans0)
        grad_rot0 = torch.empty_like(rot0)
        grad_time0 = torch.empty_like(time0)
        grad_trans1 = torch.empty_like(trans1)
        grad_rot1 = torch.empty_like(rot1)
        grad_time1 = torch.empty_like(time1)
        grad_point = torch.empty_like(point)
        grad_query_time = torch.empty_like(query_time)
        (
            grad_trans0,
            grad_rot0,
            grad_time0,
            grad_trans1,
            grad_rot1,
            grad_time1,
            grad_point,
            grad_query_time,
        ) = _backend._GEOMETRY_CUDA.trajectory_transform_point_2poses_bwd(
            trans0,
            rot0,
            time0,
            trans1,
            rot1,
            time1,
            point,
            query_time,
            grad_result_point,
        )
        return (
            grad_trans0,
            grad_rot0,
            grad_time0,
            grad_trans1,
            grad_rot1,
            grad_time1,
            grad_point,
            grad_query_time,
        )


class TrajectoryGetRotation2PosesFunction(torch.autograd.Function):
    """Differentiable 2-pose trajectory rotation query."""

    @staticmethod
    def forward(
        ctx,
        trans0: torch.Tensor,
        rot0: torch.Tensor,
        time0: torch.Tensor,
        trans1: torch.Tensor,
        rot1: torch.Tensor,
        time1: torch.Tensor,
        query_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trans0 = trans0.contiguous()
        rot0 = rot0.contiguous()
        time0 = time0.contiguous()
        trans1 = trans1.contiguous()
        rot1 = rot1.contiguous()
        time1 = time1.contiguous()
        query_time = query_time.contiguous()
        (
            result_quat,
            result_out_of_bounds,
        ) = _backend._GEOMETRY_CUDA.trajectory_get_rotation_2poses(
            trans0,
            rot0,
            time0,
            trans1,
            rot1,
            time1,
            query_time,
        )
        ctx.save_for_backward(
            trans0,
            rot0,
            time0,
            trans1,
            rot1,
            time1,
            query_time,
            result_quat,
            result_out_of_bounds,
        )
        return result_quat, result_out_of_bounds

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result_quat = grad_outputs[0]
        # grad_result_out_of_bounds = grad_outputs[1]  # out_of_bounds is no_diff
        (
            trans0,
            rot0,
            time0,
            trans1,
            rot1,
            time1,
            query_time,
            result_quat,
            result_out_of_bounds,
        ) = ctx.saved_tensors
        grad_result_quat = grad_result_quat.contiguous()
        grad_trans0 = torch.empty_like(trans0)
        grad_rot0 = torch.empty_like(rot0)
        grad_time0 = torch.empty_like(time0)
        grad_trans1 = torch.empty_like(trans1)
        grad_rot1 = torch.empty_like(rot1)
        grad_time1 = torch.empty_like(time1)
        grad_query_time = torch.empty_like(query_time)
        (
            grad_trans0,
            grad_rot0,
            grad_time0,
            grad_trans1,
            grad_rot1,
            grad_time1,
            grad_query_time,
        ) = _backend._GEOMETRY_CUDA.trajectory_get_rotation_2poses_bwd(
            trans0,
            rot0,
            time0,
            trans1,
            rot1,
            time1,
            query_time,
            grad_result_quat,
        )
        return (
            grad_trans0,
            grad_rot0,
            grad_time0,
            grad_trans1,
            grad_rot1,
            grad_time1,
            grad_query_time,
        )


class TrajectoryTransformPoint1PoseFunction(torch.autograd.Function):
    """Differentiable 1-pose trajectory point transformation."""

    @staticmethod
    def forward(
        ctx,
        trans: torch.Tensor,
        rot: torch.Tensor,
        time: torch.Tensor,
        point: torch.Tensor,
        query_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trans = trans.contiguous()
        rot = rot.contiguous()
        time = time.contiguous()
        point = point.contiguous()
        query_time = query_time.contiguous()
        (
            result_point,
            result_out_of_bounds,
        ) = _backend._GEOMETRY_CUDA.trajectory_transform_point_1pose(
            trans, rot, time, point, query_time
        )
        ctx.save_for_backward(
            trans, rot, time, point, query_time, result_point, result_out_of_bounds
        )
        return result_point, result_out_of_bounds

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result_point = grad_outputs[0]
        # grad_result_out_of_bounds = grad_outputs[1]  # out_of_bounds is no_diff
        (
            trans,
            rot,
            time,
            point,
            query_time,
            result_point,
            result_out_of_bounds,
        ) = ctx.saved_tensors
        grad_result_point = grad_result_point.contiguous()
        grad_trans = torch.empty_like(trans)
        grad_rot = torch.empty_like(rot)
        grad_time = torch.empty_like(time)
        grad_point = torch.empty_like(point)
        grad_query_time = torch.empty_like(query_time)
        (
            grad_trans,
            grad_rot,
            grad_time,
            grad_point,
            grad_query_time,
        ) = _backend._GEOMETRY_CUDA.trajectory_transform_point_1pose_bwd(
            trans,
            rot,
            time,
            point,
            query_time,
            grad_result_point,
        )
        return grad_trans, grad_rot, grad_time, grad_point, grad_query_time


# ============================================================================
# Public API - Differentiable Wrapper Functions
# ============================================================================


def se3pose_transform_point(
    translation: torch.Tensor, rotation: torch.Tensor, point: torch.Tensor
) -> torch.Tensor:
    """Transform points using SE3 poses (batched, GPU accelerated, differentiable).

    Args:
        translation: (N, 3) translation vectors
        rotation: (N, 4) quaternions in xyzw format
        point: (N, 3) points to transform

    Returns:
        (N, 3) transformed points
    """
    _validate_se3_vec3_inputs(translation, rotation, point, "point")
    return SE3PoseTransformPointFunction.apply(translation, rotation, point)


def se3pose_transform_direction(
    translation: torch.Tensor, rotation: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    """Transform directions using SE3 poses (rotation only, GPU accelerated, differentiable).

    Args:
        translation: (N, 3) translation vectors (not used for directions)
        rotation: (N, 4) quaternions in xyzw format
        direction: (N, 3) directions to transform

    Returns:
        (N, 3) transformed directions
    """
    _validate_se3_vec3_inputs(translation, rotation, direction, "direction")
    return SE3PoseTransformDirectionFunction.apply(translation, rotation, direction)


def se3pose_inverse_transform_point(
    translation: torch.Tensor, rotation: torch.Tensor, point: torch.Tensor
) -> torch.Tensor:
    """Apply inverse SE3 transformation to points (batched, GPU accelerated, differentiable).

    Args:
        translation: (N, 3) translation vectors
        rotation: (N, 4) quaternions in xyzw format
        point: (N, 3) points to transform

    Returns:
        (N, 3) inverse transformed points
    """
    _validate_se3_vec3_inputs(translation, rotation, point, "point")
    return SE3PoseInverseTransformPointFunction.apply(translation, rotation, point)


def se3pose_inverse_transform_direction(
    translation: torch.Tensor, rotation: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    """Apply inverse SE3 transformation to directions (batched, GPU accelerated, differentiable).

    Args:
        translation: (N, 3) translation vectors (not used for directions)
        rotation: (N, 4) quaternions in xyzw format
        direction: (N, 3) directions to transform

    Returns:
        (N, 3) inverse transformed directions
    """
    _validate_se3_vec3_inputs(translation, rotation, direction, "direction")
    return SE3PoseInverseTransformDirectionFunction.apply(
        translation, rotation, direction
    )


def se3pose_to_matrix(
    translation: torch.Tensor, rotation: torch.Tensor
) -> torch.Tensor:
    """Convert SE3 poses to 4x4 transformation matrices (batched, GPU accelerated, differentiable).

    Args:
        translation: (N, 3) translation vectors
        rotation: (N, 4) quaternions in xyzw format

    Returns:
        (N, 4, 4) transformation matrices
    """
    _validate_se3_pose_pair(translation, rotation)
    return SE3PoseToMatrixFunction.apply(translation, rotation)


def se3pose_from_matrix(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert 4x4 transformation matrices to SE3 poses (batched, GPU accelerated, differentiable).

    Args:
        matrix: (N, 4, 4) transformation matrices or flattened (N, 16) row-major
            matrices. Empty batches are valid and return empty outputs with the
            corresponding shape, dtype, and device.

    Returns:
        (N, 3) translation vectors
        (N, 4) quaternions in xyzw format

    Raises:
        ValueError: If ``matrix`` is not shaped as (N, 4, 4) or (N, 16).
    """
    matrix = _expect_tensor("matrix", matrix)
    _require_cuda("matrix", matrix)
    _require_float32_or_float64("matrix", matrix)
    if matrix.ndim == 3 and matrix.shape[1:] == (4, 4):
        # turn matrix from (N, 4, 4) to (N, 16)
        matrix = matrix.reshape(matrix.shape[0], -1)
    elif matrix.ndim != 2 or matrix.shape[1] != 16:
        raise ValueError(
            "matrix must have shape (N, 4, 4) or (N, 16); " f"got {tuple(matrix.shape)}"
        )
    return SE3PoseFromMatrixFunction.apply(matrix)


def se3pose_to_inverse_matrix(
    translation: torch.Tensor, rotation: torch.Tensor, wxyz_format: bool = False
) -> torch.Tensor:
    """Convert SE3 poses to inverse 4x4 transformation matrices (batched, GPU accelerated, differentiable).

    Args:
        translation: (N, 3) translation vectors
        rotation: (N, 4) quaternions in xyzw format
        wxyz_format: If true, input is (w,x,y,z) and will be converted to (x,y,z,w)
    """
    _validate_se3_pose_pair(translation, rotation)
    return SE3PoseToInverseMatrixFunction.apply(translation, rotation, wxyz_format)


def trajectory_transform_point_2poses(
    trans0: torch.Tensor,
    rot0: torch.Tensor,
    time0: torch.Tensor,
    trans1: torch.Tensor,
    rot1: torch.Tensor,
    time1: torch.Tensor,
    point: torch.Tensor,
    query_time: torch.Tensor,
) -> dict:
    """Transform points using 2-pose trajectories (with extrapolation, batched, GPU accelerated, differentiable).

    The two keyframe times are treated as an unordered span. If ``time0 > time1``,
    the implementation evaluates the same interpolation/extrapolation over
    ``[min(time0, time1), max(time0, time1)]`` while swapping the pose order
    internally. If ``time0 == time1``, the interpolation factor is defined as
    ``0`` to avoid division by zero, so the result uses pose 0 and time
    derivatives are zero.

    Args:
        trans0: (N, 3) translations of first poses
        rot0: (N, 4) rotations of first poses
        time0: (N,) times of first poses
        trans1: (N, 3) translations of second poses
        rot1: (N, 4) rotations of second poses
        time1: (N,) times of second poses
        point: (N, 3) points to transform
        query_time: (N,) times to query trajectories at

    Returns:
        dict with 'point' (N, 3) tensor and 'out_of_bounds' (N,) boolean tensor
    """
    _validate_trajectory_2poses_transform(
        trans0,
        rot0,
        time0,
        trans1,
        rot1,
        time1,
        point,
        query_time,
    )
    result_point, result_out_of_bounds = TrajectoryTransformPoint2PosesFunction.apply(
        trans0, rot0, time0, trans1, rot1, time1, point, query_time
    )
    return {"point": result_point, "out_of_bounds": result_out_of_bounds.bool()}


def trajectory_get_rotation_2poses(
    trans0: torch.Tensor,
    rot0: torch.Tensor,
    time0: torch.Tensor,
    trans1: torch.Tensor,
    rot1: torch.Tensor,
    time1: torch.Tensor,
    query_time: torch.Tensor,
) -> dict:
    """Get rotations from 2-pose trajectories (with extrapolation, batched, GPU accelerated, differentiable).

    The two keyframe times are treated as an unordered span. If ``time0 > time1``,
    the implementation evaluates interpolation/extrapolation over
    ``[min(time0, time1), max(time0, time1)]`` while swapping the pose order
    internally. If ``time0 == time1``, the interpolation factor is defined as
    ``0`` to avoid division by zero, so the result uses rotation 0 and time
    derivatives are zero.

    Args:
        trans0: (N, 3) translations of first poses
        rot0: (N, 4) rotations of first poses
        time0: (N,) times of first poses
        trans1: (N, 3) translations of second poses
        rot1: (N, 4) rotations of second poses
        time1: (N,) times of second poses
        query_time: (N,) times to query trajectories at

    Returns:
        dict with 'quat' (N, 4) tensor and 'out_of_bounds' (N,) boolean tensor
    """
    _validate_trajectory_2poses_rotation(
        trans0, rot0, time0, trans1, rot1, time1, query_time
    )
    result_quat, result_out_of_bounds = TrajectoryGetRotation2PosesFunction.apply(
        trans0, rot0, time0, trans1, rot1, time1, query_time
    )
    return {"quat": result_quat, "out_of_bounds": result_out_of_bounds.bool()}


def trajectory_transform_point_1pose(
    trans: torch.Tensor,
    rot: torch.Tensor,
    time: torch.Tensor,
    point: torch.Tensor,
    query_time: torch.Tensor,
) -> dict:
    """Transform points using 1-pose trajectories (batched, GPU accelerated, differentiable).

    With only one pose, always returns that pose (can't interpolate/extrapolate).

    Args:
        trans: (N, 3) translations of poses
        rot: (N, 4) rotations of poses
        time: (N,) times of poses
        point: (N, 3) points to transform
        query_time: (N,) times to query trajectories at

    Returns:
        dict with 'point' (N, 3) tensor and 'out_of_bounds' (N,) boolean tensor
    """
    _validate_trajectory_1pose(trans, rot, time, point, query_time)
    result_point, result_out_of_bounds = TrajectoryTransformPoint1PoseFunction.apply(
        trans, rot, time, point, query_time
    )
    return {"point": result_point, "out_of_bounds": result_out_of_bounds.bool()}


def frame_transform_poses_tquat(
    tquat_poses: torch.Tensor,
    rotation: tuple[float, float, float, float],
    translation: tuple[float, float, float],
    scale: float,
) -> torch.Tensor:
    tquat_poses = _expect_tensor("tquat_poses", tquat_poses)
    _require_cuda("tquat_poses", tquat_poses)
    _require_float32("tquat_poses", tquat_poses)
    if tquat_poses.dim() != 2 or tquat_poses.shape[1] != 7:
        raise ValueError(
            "tquat_poses must have shape (N, 7); " f"got {tuple(tquat_poses.shape)}"
        )
    tquat_poses = tquat_poses.contiguous()
    qx, qy, qz, qw = rotation
    tx, ty, tz = translation
    return _backend._GEOMETRY_CUDA.frame_transform_poses_tquat(
        tquat_poses,
        float(qx),
        float(qy),
        float(qz),
        float(qw),
        float(tx),
        float(ty),
        float(tz),
        float(scale),
    )
