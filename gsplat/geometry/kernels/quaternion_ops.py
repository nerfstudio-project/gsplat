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

"""Python wrappers for native CUDA quaternion operations with PyTorch autograd support.

All functions follow the same conventions:
- Quaternion format: float4 in xyzw order (x, y, z, w)
- Unit quaternions represent rotations in 3D space
- All operations require the `gsplat_geometry_cuda` extension on CUDA tensors
- Paired quaternion/vector inputs do not broadcast implicitly; matching batch shapes
  are required unless a function explicitly documents a narrower exception
"""

from typing import Any, Sequence

import torch

from torch import Tensor

from . import _backend


def _expect_tensor(name: str, x: object) -> Tensor:
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    return x


def _require_cuda(name: str, t: Tensor) -> None:
    if t.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor; got device {t.device}")


def _require_float32_or_float64(name: str, t: Tensor) -> None:
    if t.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have dtype float32 or float64, got {t.dtype}")


def _require_last_dim(name: str, t: Tensor, d: int) -> None:
    if t.dim() < 1:
        raise ValueError(f"{name} must have at least one dimension, got {t.dim()}")
    if t.size(-1) != d:
        raise ValueError(
            f"{name} must have last dimension {d}, got shape {tuple(t.shape)}"
        )


def _require_same_device_dtype(pairs: Sequence[tuple[str, Tensor]]) -> None:
    if not pairs:
        return
    ref_name, ref = pairs[0]
    for name, t in pairs[1:]:
        if t.device != ref.device:
            raise ValueError(
                f"{name} must be on the same device as {ref_name} "
                f"({ref.device}); got {t.device}"
            )
        if t.dtype != ref.dtype:
            raise TypeError(
                f"{name} must have the same dtype as {ref_name} ({ref.dtype}); "
                f"got {t.dtype}"
            )


def _validate_cuda_quat_tensor(name: str, q: Tensor) -> None:
    _expect_tensor(name, q)
    _require_cuda(name, q)
    _require_float32_or_float64(name, q)
    _require_last_dim(name, q, 4)


def _validate_cuda_vec3_tensor(name: str, v: Tensor) -> None:
    _expect_tensor(name, v)
    _require_cuda(name, v)
    _require_float32_or_float64(name, v)
    _require_last_dim(name, v, 3)


def _quat_blend_t_tensor(q1_flat: Tensor, t: float | Tensor) -> Tensor:
    """Build contiguous (N, 1) interpolation parameters on ``q1_flat`` device/dtype.

    Shared normalization for :func:`quat_slerp` and
    :func:`quat_manifold_interp`: Python ``int``/``float`` broadcast to batch,
    and tensor ``t`` must be floating-point with flattened size ``1`` or ``N``.
    """
    N = q1_flat.shape[0]
    if isinstance(t, (int, float)):
        return q1_flat.new_full((N, 1), float(t))
    t = _expect_tensor("t", t)
    if not t.dtype.is_floating_point:
        raise TypeError(
            f"t must be a floating-point tensor when tensor-valued, got {t.dtype}"
        )
    t = t.to(device=q1_flat.device, dtype=q1_flat.dtype)
    t_flat = t.reshape(-1, 1)
    if t_flat.shape[0] == 1:
        return t_flat.expand(N, 1).contiguous()
    if t_flat.shape[0] != N:
        raise ValueError(
            f"t must have batch size 1 or {N} when tensor-valued; "
            f"got {t_flat.shape[0]}"
        )
    return t_flat.contiguous()


def _quat_normalize_safe_eps_dtype(dtype: torch.dtype) -> float:
    """Match QuatNormEps in cuda/csrc/quaternion.cu."""
    return 1e-12 if dtype == torch.float64 else 1e-7


def _quat_conjugate_backward_reference(grad_out: Tensor) -> Tensor:
    """Backward for conjugate c = (-x,-y,-z,w): grad_q = (-gx,-gy,-gz,gw) w.r.t. grad upstream on c."""
    g = grad_out
    out = torch.empty_like(g)
    out[..., 0] = -g[..., 0]
    out[..., 1] = -g[..., 1]
    out[..., 2] = -g[..., 2]
    out[..., 3] = g[..., 3]
    return out


def _quat_normalize_safe_backward_reference(quat: Tensor, grad_out: Tensor) -> Tensor:
    """Reference backward for normalizeSafe, kept for tests."""
    norm_sq = (quat * quat).sum(dim=-1, keepdim=True)
    eps = _quat_normalize_safe_eps_dtype(quat.dtype)
    small = norm_sq.abs() < eps
    safe_norm_sq = torch.where(small, torch.ones_like(norm_sq), norm_sq)
    inv_s = torch.rsqrt(safe_norm_sq)
    q_dot_g = (quat * grad_out).sum(dim=-1, keepdim=True)
    inv_s3 = inv_s * inv_s * inv_s
    grad = grad_out * inv_s - quat * q_dot_g * inv_s3
    return torch.where(small.expand_as(grad), torch.zeros_like(grad), grad)


def _quat_angular_distance_forward_torch(q1: Tensor, q2: Tensor) -> Tensor:
    """Forward angular distance as (N, 1); kept as a test reference."""
    n1 = q1 / q1.norm(dim=-1, keepdim=True)
    n2 = q2 / q2.norm(dim=-1, keepdim=True)
    d = (n1 * n2).sum(dim=-1, keepdim=True)
    c = d.abs().clamp(min=0.0, max=1.0)
    return 2.0 * torch.acos(c)


def _quat_lerp_backward_reference(
    q1: Tensor, q2: Tensor, result: Tensor, grad_out: Tensor, t: float
) -> tuple[Tensor, Tensor]:
    """Reference backward for hemisphere lerp + normalize."""
    d = (q1 * q2).sum(dim=-1, keepdim=True)
    s = torch.where(d < 0, -torch.ones_like(d), torch.ones_like(d))
    q2e = s * q2
    om = 1.0 - t
    r = om * q1 + t * q2e
    n = r.norm(dim=-1, keepdim=True)
    ydotg = (result * grad_out).sum(dim=-1, keepdim=True)
    scale = 1.0 / n
    grad_r = (grad_out - result * ydotg) * scale
    grad_q1 = om * grad_r
    grad_q2 = s * t * grad_r
    return grad_q1, grad_q2


def _quat_slerp_batched_forward_reference(q1: Tensor, q2: Tensor, t: Tensor) -> Tensor:
    """Differentiable batched SLERP reference matching CUDA hemisphere and lerp-fallback semantics."""
    d = (q1 * q2).sum(dim=-1, keepdim=True)
    q2e = torch.where(d < 0, -q2, q2)
    c_raw = (q1 * q2e).sum(dim=-1, keepdim=True)
    c = c_raw.clamp(min=-1.0, max=1.0)
    use_lerp = c > 0.9995
    om = 1.0 - t
    r = om * q1 + t * q2e
    y_lerp = r / r.norm(dim=-1, keepdim=True)
    theta = torch.acos(c)
    sin_theta = torch.sin(theta)
    w1 = torch.sin((1.0 - t) * theta) / sin_theta
    w2 = torch.sin(t * theta) / sin_theta
    y_slerp = w1 * q1 + w2 * q2e
    return torch.where(use_lerp.expand_as(q1), y_lerp, y_slerp)


def _quat_slerp_batched_backward_reference(
    q1: Tensor, q2: Tensor, t: Tensor, grad_out: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Reference backward for batched SLERP, kept for tests."""
    with torch.enable_grad():
        q1n = q1.detach().requires_grad_(True)
        q2n = q2.detach().requires_grad_(True)
        tn = t.detach().requires_grad_(True)
        y = _quat_slerp_batched_forward_reference(q1n, q2n, tn)
        grad_q1, grad_q2, grad_t = torch.autograd.grad(
            y,
            (q1n, q2n, tn),
            grad_outputs=grad_out,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
        )
    return grad_q1, grad_q2, grad_t


class QuatNormalizeSafeFunction(torch.autograd.Function):
    """Normalize quaternion(s) with safety checks."""

    @staticmethod
    def forward(ctx, quat: Tensor) -> Tensor:
        quat = quat.contiguous()
        result = _backend._GEOMETRY_CUDA.quat_normalize_safe(quat)
        ctx.save_for_backward(quat)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        (quat,) = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_quat = _backend._GEOMETRY_CUDA.quat_normalize_safe_bwd(quat, grad_result)
        return grad_quat


class QuatConjugateFunction(torch.autograd.Function):
    """Compute quaternion conjugate."""

    @staticmethod
    def forward(ctx, quat: Tensor) -> Tensor:
        quat = quat.contiguous()
        result = _backend._GEOMETRY_CUDA.quat_conjugate(quat)
        ctx.save_for_backward(quat)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        (quat,) = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_quat = _backend._GEOMETRY_CUDA.quat_conjugate_bwd(quat, grad_result)
        return grad_quat


class QuatMultiplyFunction(torch.autograd.Function):
    """Multiply two quaternions using Hamilton product."""

    @staticmethod
    def forward(ctx, q1: Tensor, q2: Tensor) -> Tensor:
        q1 = q1.contiguous()
        q2 = q2.contiguous()
        result = _backend._GEOMETRY_CUDA.quat_multiply(q1, q2)
        ctx.save_for_backward(q1, q2, result)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        q1, q2, _result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_q1, grad_q2 = _backend._GEOMETRY_CUDA.quat_multiply_bwd(
            q1, q2, grad_result
        )
        return grad_q1, grad_q2


class QuatRotateVectorFunction(torch.autograd.Function):
    """Apply quaternion rotation to vector(s)."""

    @staticmethod
    def forward(ctx, quat: Tensor, vec: Tensor) -> Tensor:
        quat = quat.contiguous()
        vec = vec.contiguous()
        result = _backend._GEOMETRY_CUDA.quat_rotate_vector(quat, vec)
        ctx.save_for_backward(quat, vec, result)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        quat, vec, _result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_quat, grad_vec = _backend._GEOMETRY_CUDA.quat_rotate_vector_bwd(
            quat, vec, grad_result
        )
        return grad_quat, grad_vec


class QuatToMatrixFunction(torch.autograd.Function):
    """Convert quaternion(s) to 3x3 rotation matrix/matrices."""

    @staticmethod
    def forward(ctx, quat: Tensor) -> Tensor:
        quat = quat.contiguous()
        count = quat.shape[0]
        result_flat = _backend._GEOMETRY_CUDA.quat_to_matrix(quat)
        result = result_flat.reshape(count, 3, 3)
        ctx.save_for_backward(quat, result_flat)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        quat, result_flat = ctx.saved_tensors
        grad_result_flat = grad_result.reshape(result_flat.shape[0], 9).contiguous()
        grad_quat = _backend._GEOMETRY_CUDA.quat_to_matrix_bwd(quat, grad_result_flat)
        return grad_quat


class QuatSlerpBatchedFunction(torch.autograd.Function):
    """Batched spherical linear interpolation with per-element t values."""

    @staticmethod
    def forward(ctx, q1: Tensor, q2: Tensor, t: Tensor) -> Tensor:
        q1 = q1.contiguous()
        q2 = q2.contiguous()
        t = t.contiguous()
        result = _backend._GEOMETRY_CUDA.quat_slerp_batched(q1, q2, t)
        ctx.save_for_backward(q1, q2, t, result)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        q1, q2, t, result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        return _backend._GEOMETRY_CUDA.quat_slerp_batched_bwd(
            q1, q2, t, result, grad_result
        )


class QuatLerpFunction(torch.autograd.Function):
    """LERP with hemisphere fix and normalization."""

    @staticmethod
    def forward(ctx, q1: Tensor, q2: Tensor, t: float) -> Tensor:
        q1 = q1.contiguous()
        q2 = q2.contiguous()
        result = _backend._GEOMETRY_CUDA.quat_lerp(q1, q2, float(t))
        ctx.save_for_backward(q1, q2, result)
        ctx.t = t
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        q1, q2, result = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_q1, grad_q2 = _backend._GEOMETRY_CUDA.quat_lerp_bwd(
            q1, q2, result, grad_result, float(ctx.t)
        )
        return grad_q1, grad_q2, None


class QuatFromAxisAngleFunction(torch.autograd.Function):
    """Convert axis-angle representation to quaternion."""

    @staticmethod
    def forward(ctx, axis: Tensor, angle: Tensor) -> Tensor:
        axis = axis.contiguous()
        angle = angle.contiguous()
        quat = _backend._GEOMETRY_CUDA.quat_from_axis_angle(axis, angle)
        ctx.save_for_backward(axis, angle, quat)
        return quat

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_quat = grad_outputs[0]
        axis, angle, quat = ctx.saved_tensors
        grad_quat = grad_quat.contiguous()
        grad_axis, grad_angle = _backend._GEOMETRY_CUDA.quat_from_axis_angle_bwd(
            axis, angle, grad_quat
        )
        return grad_axis, grad_angle


class QuatAngularDistanceFunction(torch.autograd.Function):
    """Compute angular distance between quaternions."""

    @staticmethod
    def forward(ctx, q1: Tensor, q2: Tensor) -> Tensor:
        q1 = q1.contiguous()
        q2 = q2.contiguous()
        distance = _backend._GEOMETRY_CUDA.quat_angular_distance(q1, q2)
        ctx.save_for_backward(q1, q2)
        return distance.squeeze(-1)

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_distance = grad_outputs[0].unsqueeze(-1).contiguous()
        q1, q2 = ctx.saved_tensors
        grad_q1, grad_q2 = _backend._GEOMETRY_CUDA.quat_angular_distance_bwd(
            q1, q2, grad_distance
        )
        return grad_q1, grad_q2


class QuatManifoldInterpFunction(torch.autograd.Function):
    """Manifold interpolation using SO(3) operations."""

    @staticmethod
    def forward(ctx, q1: Tensor, q2: Tensor, t: Tensor) -> Tensor:
        q1 = q1.contiguous()
        q2 = q2.contiguous()
        t = t.contiguous()
        result = _backend._GEOMETRY_CUDA.quat_manifold_interp(q1, q2, t)
        ctx.save_for_backward(q1, q2, t)
        return result

    @staticmethod
    def backward(ctx, *grad_outputs: Any):
        grad_result = grad_outputs[0]
        q1, q2, t = ctx.saved_tensors
        grad_result = grad_result.contiguous()
        grad_q1, grad_q2, grad_t = _backend._GEOMETRY_CUDA.quat_manifold_interp_bwd(
            q1, q2, t, grad_result
        )
        return grad_q1, grad_q2, grad_t


# ============================================================================
# Public API - User-Friendly Functions with Batch Shape Support
# ============================================================================


def quat_normalize_safe(quat: Tensor) -> Tensor:
    """Normalize quaternion(s) with safety checks (GPU accelerated, differentiable).

    Handles near-zero quaternions by returning identity quaternion.

    Args:
        quat: (..., 4) quaternion(s) in xyzw format

    Returns:
        (..., 4) normalized quaternion(s)
    """
    quat = _expect_tensor("quat", quat)
    _validate_cuda_quat_tensor("quat", quat)
    original_shape = quat.shape
    quat_flat = quat.reshape(-1, 4)
    result_flat = QuatNormalizeSafeFunction.apply(quat_flat)
    return result_flat.reshape(original_shape)


def quat_conjugate(q: Tensor) -> Tensor:
    """Compute quaternion conjugate (GPU accelerated, differentiable).

    Args:
        q: (..., 4) quaternion(s) in xyzw format

    Returns:
        (..., 4) conjugate quaternion(s)
    """
    q = _expect_tensor("q", q)
    _validate_cuda_quat_tensor("q", q)
    original_shape = q.shape
    q_flat = q.reshape(-1, 4)
    result_flat = QuatConjugateFunction.apply(q_flat)
    return result_flat.reshape(original_shape)


def quat_inverse(q: Tensor) -> Tensor:
    """Compute quaternion inverse (GPU accelerated, differentiable).

    For unit quaternions, inverse equals conjugate; this API implements that case
    by delegating to :func:`quat_conjugate` (no separate native kernel).

    Args:
        q: (..., 4) quaternion(s) in xyzw format (assumed normalized)

    Returns:
        (..., 4) inverse quaternion(s)
    """
    return quat_conjugate(q)


def quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Multiply two quaternions using Hamilton product (GPU accelerated, differentiable).

    Result is q1 * q2 (apply q2 first, then q1).
    ``q1`` and ``q2`` must have exactly the same shape; this wrapper does not
    broadcast paired quaternion inputs.

    Args:
        q1: (..., 4) first quaternion(s) in xyzw format
        q2: (..., 4) second quaternion(s) in xyzw format

    Returns:
        (..., 4) product quaternion(s)
    """
    q1 = _expect_tensor("q1", q1)
    q2 = _expect_tensor("q2", q2)
    _validate_cuda_quat_tensor("q1", q1)
    _validate_cuda_quat_tensor("q2", q2)
    _require_same_device_dtype([("q1", q1), ("q2", q2)])
    if q1.shape != q2.shape:
        raise ValueError(
            f"q1 and q2 must have the same shape; got {tuple(q1.shape)} vs {tuple(q2.shape)}"
        )

    original_shape = q1.shape
    q1_flat = q1.reshape(-1, 4)
    q2_flat = q2.reshape(-1, 4)
    result_flat = QuatMultiplyFunction.apply(q1_flat, q2_flat)
    return result_flat.reshape(original_shape)


def quat_rotate_vector(q: Tensor, v: Tensor) -> Tensor:
    """Apply quaternion rotation to vector(s) (GPU accelerated, differentiable).

    Uses the efficient formula: q * v * q^(-1)
    ``q`` and ``v`` must have the same leading batch dimensions; this wrapper
    rejects otherwise-broadcastable shapes such as ``(1, 4)`` with ``(N, 3)``.

    Args:
        q: (..., 4) quaternion(s) in xyzw format
        v: (..., 3) vector(s) to rotate

    Returns:
        (..., 3) rotated vector(s)
    """
    q = _expect_tensor("q", q)
    v = _expect_tensor("v", v)
    _validate_cuda_quat_tensor("q", q)
    _validate_cuda_vec3_tensor("v", v)
    _require_same_device_dtype([("q", q), ("v", v)])
    if q.shape[:-1] != v.shape[:-1]:
        raise ValueError(
            f"q and v must share the same batch dimensions; got leading shape "
            f"{tuple(q.shape[:-1])} vs {tuple(v.shape[:-1])}"
        )

    original_shape = v.shape
    q_flat = q.reshape(-1, 4)
    v_flat = v.reshape(-1, 3)
    result_flat = QuatRotateVectorFunction.apply(q_flat, v_flat)
    return result_flat.reshape(original_shape)


def quat_to_matrix(quat: Tensor) -> Tensor:
    """Convert quaternion(s) to 3x3 rotation matrix/matrices (GPU accelerated, differentiable).

    Args:
        quat: (..., 4) quaternion(s) in xyzw format

    Returns:
        (..., 3, 3) rotation matrix/matrices
    """
    quat = _expect_tensor("quat", quat)
    _validate_cuda_quat_tensor("quat", quat)
    original_shape = quat.shape[:-1]
    quat_flat = quat.reshape(-1, 4)
    result = QuatToMatrixFunction.apply(quat_flat)
    return result.reshape(*original_shape, 3, 3)


def quat_slerp(q1: Tensor, q2: Tensor, t: float | Tensor) -> Tensor:
    """Spherical linear interpolation between quaternions (GPU accelerated, differentiable).

    Uses the batched kernel for all cases; no CPU/GPU sync. ``q1`` and ``q2``
    must have exactly the same shape. ``t`` is the only parameter with limited
    broadcasting support: a Python scalar or 0-dim tensor applies to the whole
    batch, and a tensor-valued ``t`` must flatten to either length ``1`` or the
    full flattened batch size ``prod(q1.shape[:-1])``.

    Args:
        q1: (..., 4) start quaternion(s) in xyzw format
        q2: (..., 4) end quaternion(s) in xyzw format
        t: Interpolation parameter(s) [0, 1]: Python scalar, 0-dim tensor,
            or tensor with flattened size 1 or ``prod(q1.shape[:-1])``

    Returns:
        (..., 4) interpolated quaternion(s)
    """
    q1 = _expect_tensor("q1", q1)
    q2 = _expect_tensor("q2", q2)
    _validate_cuda_quat_tensor("q1", q1)
    _validate_cuda_quat_tensor("q2", q2)
    _require_same_device_dtype([("q1", q1), ("q2", q2)])
    if q1.shape != q2.shape:
        raise ValueError(
            f"q1 and q2 must have the same shape; got {tuple(q1.shape)} vs {tuple(q2.shape)}"
        )
    original_shape = q1.shape
    q1_flat = q1.reshape(-1, 4)
    q2_flat = q2.reshape(-1, 4)
    t_tensor = _quat_blend_t_tensor(q1_flat, t)

    result_flat = QuatSlerpBatchedFunction.apply(q1_flat, q2_flat, t_tensor)
    return result_flat.reshape(original_shape)


def quat_lerp(q1: Tensor, q2: Tensor, t: float) -> Tensor:
    """Linear interpolation between quaternions (GPU accelerated, differentiable).

    If the dot product of ``q1`` and ``q2`` is negative, ``q2`` is negated so the
    blend takes the short arc, then the result is normalized (unit quaternion when
    the blended vector is non-zero).
    ``q1`` and ``q2`` must have exactly the same shape; this wrapper does not
    broadcast paired quaternion inputs. ``t`` must be a Python scalar.

    Args:
        q1: (..., 4) start quaternion(s) in xyzw format
        q2: (..., 4) end quaternion(s) in xyzw format
        t: Interpolation parameter [0, 1]

    Returns:
        (..., 4) interpolated quaternion(s)
    """
    if not isinstance(t, (int, float)):
        raise TypeError(
            "quat_lerp expects a scalar Python float or int for t; "
            "use quat_slerp for tensor interpolation parameters"
        )
    q1 = _expect_tensor("q1", q1)
    q2 = _expect_tensor("q2", q2)
    _validate_cuda_quat_tensor("q1", q1)
    _validate_cuda_quat_tensor("q2", q2)
    _require_same_device_dtype([("q1", q1), ("q2", q2)])
    if q1.shape != q2.shape:
        raise ValueError(
            f"q1 and q2 must have the same shape; got {tuple(q1.shape)} vs {tuple(q2.shape)}"
        )
    original_shape = q1.shape
    q1_flat = q1.reshape(-1, 4)
    q2_flat = q2.reshape(-1, 4)
    result_flat = QuatLerpFunction.apply(q1_flat, q2_flat, float(t))
    return result_flat.reshape(original_shape)


def quat_from_axis_angle(axis: Tensor, angle: Tensor) -> Tensor:
    """Convert axis-angle representation to quaternion (GPU accelerated, differentiable).

    ``axis`` is flattened to rows of shape ``(3,)``. A scalar ``angle`` is only
    accepted when ``axis`` contains exactly one row; otherwise ``angle`` must
    flatten to the same row count as ``axis``.

    Args:
        axis: (..., 3) rotation axis (does not need to be normalized)
        angle: Rotation angle in radians; either a scalar for a single axis row
            or a tensor whose flattened size matches ``prod(axis.shape[:-1])``

    Returns:
        (..., 4) quaternion in xyzw format
    """
    axis = _expect_tensor("axis", axis)
    angle = _expect_tensor("angle", angle)
    _validate_cuda_vec3_tensor("axis", axis)
    _require_cuda("angle", angle)
    _require_float32_or_float64("angle", angle)
    _require_same_device_dtype([("axis", axis), ("angle", angle)])
    axis_flat = axis.reshape(-1, 3)
    n = axis_flat.shape[0]
    if angle.dim() == 0:
        if n != 1:
            raise ValueError(
                "axis-angle batch mismatch: scalar angle requires axis with exactly "
                f"one quaternion row, got {n} rows from axis"
            )
        angle_flat = angle.unsqueeze(0).unsqueeze(1)
    else:
        angle_flat = angle.reshape(-1, 1)
        if angle_flat.shape[0] != n:
            raise ValueError(
                "axis and angle batch mismatch: "
                f"{n} axis rows vs {angle_flat.shape[0]} angle rows"
            )
    original_shape = axis.shape[:-1]
    result_flat = QuatFromAxisAngleFunction.apply(axis_flat, angle_flat)
    return result_flat.reshape(*original_shape, 4)


def quat_angular_distance(q1: Tensor, q2: Tensor) -> Tensor:
    """Compute angular distance between quaternions (GPU accelerated, differentiable).

    Returns the geodesic distance on the unit quaternion manifold.
    ``q1`` and ``q2`` must have exactly the same shape; this wrapper does not
    broadcast paired quaternion inputs.

    Args:
        q1: (..., 4) quaternion(s) in xyzw format
        q2: (..., 4) quaternion(s) in xyzw format

    Returns:
        (...,) angular distance in radians
    """
    q1 = _expect_tensor("q1", q1)
    q2 = _expect_tensor("q2", q2)
    _validate_cuda_quat_tensor("q1", q1)
    _validate_cuda_quat_tensor("q2", q2)
    _require_same_device_dtype([("q1", q1), ("q2", q2)])
    if q1.shape != q2.shape:
        raise ValueError(
            f"q1 and q2 must have the same shape; got {tuple(q1.shape)} vs {tuple(q2.shape)}"
        )
    original_shape = q1.shape[:-1]
    q1_flat = q1.reshape(-1, 4)
    q2_flat = q2.reshape(-1, 4)
    result_flat = QuatAngularDistanceFunction.apply(q1_flat, q2_flat)
    return result_flat.reshape(original_shape)


def quat_identity(
    shape: tuple = (),
    dtype: torch.dtype = torch.float32,
    *,
    device: torch.device | str | int,
) -> Tensor:
    """Create identity quaternion(s).

    Args:
        shape: Shape of the batch dimensions
        dtype: Data type
        device: Device for the output tensor (required, keyword-only). Pass e.g.
            ``"cuda:0"``, ``torch.device("cpu")``, or a device index. There is
            no implicit default (including no CUDA default when omitted).

    Returns:
        (*shape, 4) identity quaternion(s) [0, 0, 0, 1]

    Raises:
        TypeError: If ``dtype`` is not floating-point, or if ``device`` is
            omitted (missing required keyword-only argument).
        ValueError: If ``device`` is explicitly ``None``.
    """
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be a floating-point dtype, got {dtype}")
    if device is None:
        raise ValueError(
            "quat_identity() requires an explicit `device` argument "
            "(e.g. device='cuda' or device=torch.device('cpu')); got device=None."
        )
    quat = torch.zeros(shape + (4,), dtype=dtype, device=device)
    quat[..., 3] = 1.0
    return quat


def quat_manifold_interp(q1: Tensor, q2: Tensor, t: float | Tensor) -> Tensor:
    """Manifold interpolation using SO(3) operations (GPU accelerated, differentiable).

    Composed operation: q1 * exp(t * log(q1^-1 * q2))
    Uses a single optimized kernel instead of multiple operations. ``q1`` and
    ``q2`` must have exactly the same shape; this wrapper does not broadcast
    paired quaternion inputs. Interpolation parameter ``t`` follows the same
    rules as :func:`quat_slerp` (Python scalar, 0-dim tensor, or tensor of size
    ``1`` or full batch).

    Args:
        q1: (..., 4) start quaternion(s) in xyzw format
        q2: (..., 4) end quaternion(s) in xyzw format
        t: Interpolation parameter(s) [0, 1]: Python scalar, 0-dim tensor,
            or tensor with flattened size ``1`` or ``prod(q1.shape[:-1])``

    Returns:
        (..., 4) interpolated quaternion(s)
    """
    q1 = _expect_tensor("q1", q1)
    q2 = _expect_tensor("q2", q2)
    _validate_cuda_quat_tensor("q1", q1)
    _validate_cuda_quat_tensor("q2", q2)
    _require_same_device_dtype([("q1", q1), ("q2", q2)])
    if q1.shape != q2.shape:
        raise ValueError(
            f"q1 and q2 must have the same shape; got {tuple(q1.shape)} vs {tuple(q2.shape)}"
        )
    original_shape = q1.shape
    q1_flat = q1.reshape(-1, 4)
    q2_flat = q2.reshape(-1, 4)
    t_tensor = _quat_blend_t_tensor(q1_flat, t)
    result_flat = QuatManifoldInterpFunction.apply(q1_flat, q2_flat, t_tensor)
    return result_flat.reshape(original_shape)
