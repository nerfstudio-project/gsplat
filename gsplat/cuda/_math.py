# SPDX-FileCopyrightText: Copyright 2025-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Mathematical utility functions for GSplat.

This module contains mathematical utilities used by camera models and other components,
including numerically stable computations and polynomial evaluation helpers.
"""

import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from torch import Tensor
from gsplat._helper import assert_shape


def _numerically_stable_norm2(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute 2-norm of [x, y] vectors in a numerically stable way.

    Avoids overflow/underflow for very large or very small values by
    normalizing by the maximum absolute value before computing the norm.

    Matches CUDA numerically_stable_norm2 implementation.

    Args:
        x: [B] x-components
        y: [B] y-components

    Returns:
        norm: [B] ‖(x, y)‖ = √(x² + y²)
    """
    # Preconditions
    B = x.shape
    assert_shape("x", x, B)
    assert_shape("y", y, B)

    abs_x = torch.abs(x)
    abs_y = torch.abs(y)
    min_val = torch.minimum(abs_x, abs_y)
    max_val = torch.maximum(abs_x, abs_y)

    # If max is zero, return zero (avoids division by zero)
    result = torch.zeros_like(max_val)
    nonzero_mask = max_val > 0.0

    # For non-zero max: compute norm = max · √(1 + (min/max)²)
    min_max_ratio = torch.where(
        nonzero_mask, min_val / max_val, torch.zeros_like(min_val)
    )
    result = torch.where(
        nonzero_mask, max_val * torch.sqrt(1.0 + min_max_ratio * min_max_ratio), result
    )

    # Postconditions
    assert_shape("result", result, B)

    return result


# ============================================================================
# Polynomial Helper Classes and Functions
# ============================================================================


class PolynomialProxy(ABC):
    """
    Base class for polynomial evaluation with type dispatch.

    Matches the CUDA PolynomialProxy template struct pattern.
    Subclasses implement specific polynomial types (full, even, odd).
    """

    def __init__(self, coeffs: Tensor):
        """
        Initialize polynomial proxy.

        Args:
            coeffs: [..., B, N] Tensor of polynomial coefficients where N is the number of coefficients.
        """
        self.coeffs = coeffs

    @abstractmethod
    def eval_horner(self, x: Tensor) -> Tensor:
        """
        Evaluate polynomial at x using Horner's method.

        Broadcasting behavior follows these rules:
            coeffs [B, N] + x [..., B, M] -> [..., B, M]
            coeffs [N] + x [..., M]       -> [..., M]
            coeffs [] + x [..., M]        -> [..., M]
            coeffs [] + x []              -> []
            coeffs [..., B, N] + x []     -> [..., B]
            coeffs [..., B, N] + x []     -> [..., B]
            coeffs [..., B, N] + x [M]    -> [..., B, M]

        Args:
            x: evaluation points (evaluation per column)

        Returns:
            y: polynomial values at x
        """
        ...


class FullPolynomialProxy(PolynomialProxy):
    """
    Full polynomial: y = c₀ + c₁·x + c₂·x² + c₃·x³ + ...
    """

    def eval_horner(self, x: Tensor) -> Tensor:
        # Preconditions
        B = self.coeffs.shape[:-1]
        N = self.coeffs.shape[-1]
        assert_shape("x", x, B + (1,))

        # Start with highest order coefficient, keeping trailing dim for broadcasting
        result = self.coeffs[..., N - 1 : N]

        # Horner's method: iterate backwards through remaining coefficients
        for i in range(N - 2, -1, -1):
            # Standard broadcasting: result * x + coeff_i
            result = result * x + self.coeffs[..., i : i + 1]

        # Postconditions
        assert_shape("result", result, B + (1,))

        return result


class OddPolynomialProxy(PolynomialProxy):
    """
    Odd-only polynomial: y = c₀·x + c₁·x³ + c₂·x⁵ + c₃·x⁷ + ...

    Matches CUDA PolynomialProxy<PolynomialType::ODD, N>
    """

    def eval_horner(self, x: Tensor) -> Tensor:
        """
        Evaluate odd-only polynomial using Horner's method.

        Factors out one x term and evaluates the x²-based polynomial.

        Args:
            x: [...] or [..., M] evaluation points

        Returns:
            y: [...] or [..., M] polynomial values at x
        """
        B = self.coeffs.shape[:-1]
        N = self.coeffs.shape[-1]
        assert_shape("x", x, B + (1,))

        # Factor out x: y = x · (c₀ + c₁·x² + c₂·x⁴ + ...)
        result = x * FullPolynomialProxy(self.coeffs).eval_horner(x * x)

        assert_shape("result", result, B + (1,))
        return result


class EvenPolynomialProxy(PolynomialProxy):
    """
    Even-only polynomial: y = c₀ + c₁·x² + c₂·x⁴ + c₃·x⁶ + ...
    Matches CUDA PolynomialProxy<PolynomialType::EVEN, N>
    """

    def eval_horner(self, x: Tensor) -> Tensor:
        """
        Evaluate even-only polynomial using Horner's method.
        Substitutes x² for x in standard Horner evaluation.

        Args:
            x: [...] or [..., M] evaluation points
        Returns:
            y: [...] or [..., M] polynomial values at x
        """
        B = self.coeffs.shape[:-1]
        N = self.coeffs.shape[-1]
        assert_shape("x", x, B + (1,))

        # Substitute x² for x: y = c₀ + c₁·x² + c₂·x⁴ + ...
        result = FullPolynomialProxy(self.coeffs).eval_horner(x * x)

        assert_shape("result", result, B + (1,))
        return result


def _eval_poly_inverse_horner_newton(
    poly: PolynomialProxy,
    dpoly: PolynomialProxy,
    inv_poly_approx: PolynomialProxy,
    y: Tensor,  # [..., M]
    n_iterations: int,
) -> Tuple[Tensor, Tensor]:
    """
    Evaluate inverse polynomial x = f⁻¹(y) using Newton's method.

    Given a polynomial y = f(x), finds x such that f(x) = y using:
    1. Initial approximation from inv_poly_approx
    2. Newton iterations: xₙ₊₁ = xₙ - (f(xₙ) - y) / f'(xₙ)

    Matches the CUDA eval_poly_inverse_horner_newton function pattern.

    Args:
        poly: PolynomialProxy for forward polynomial f(x)
        dpoly: PolynomialProxy for derivative f'(x)
        inv_poly_approx: PolynomialProxy for initial approximation
        y: Target values to invert
        n_iterations: Number of Newton iterations

    Returns:
        x: [...] inverted values
        converged: [...] convergence mask
    """
    # Preconditions
    B = poly.coeffs.shape[:-1]
    N = poly.coeffs.shape[-1]
    M = y.shape[-1]
    assert_shape("poly", poly.coeffs, B + (1,))
    assert_shape("dpoly", dpoly.coeffs, B + (1,))
    assert_shape("inv_poly_approx", inv_poly_approx.coeffs, B + (1,))
    assert_shape("y", y, B + (M,))

    # Get initial approximation x₀ = approx_f⁻¹(y)
    x = inv_poly_approx.eval_horner(y)

    converged = torch.zeros_like(x, dtype=torch.bool)

    # Newton iterations
    for _ in range(n_iterations):
        # Evaluate f(x) and f'(x)
        fx = poly.eval_horner(x)
        dfdx = dpoly.eval_horner(x)

        # Compute Newton step
        residual = fx - y
        dx = residual / dfdx

        # Check convergence
        newly_converged = torch.abs(dx) < 1e-6

        # Only update elements that haven't converged yet
        x = torch.where(~converged, x - dx, x)

        # If any element converged in this iteration, it won't be
        # updated in the next iterations.
        converged = converged | newly_converged

        # Early exit if all converged
        if torch.all(converged):
            break

    # Postconditions
    assert_shape("x", x, B + (M,))
    assert_shape("converged", converged, B + (M,))

    return x, converged


# ============================================================================
# Quaternion Operations
# ============================================================================


class SafeNormalize(torch.autograd.Function):
    """
    Safe normalize with custom backward matching CUDA implementation.

    Forward: normalized = v / ||v|| if ||v|| > 0 else v
    Backward: grad_v = (1/||v||) * grad_out - (1/||v||^3) * dot(grad_out, v) * v
              or grad_v = grad_out if ||v|| == 0
    """

    @staticmethod
    def forward(ctx, v: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
        """
        Args:
            v: Input tensor to normalize
            dim: Dimension along which to compute the norm
            keepdim: Whether to keep the normalized dimension
        """
        # Compute squared norm
        norm_sq = (v * v).sum(dim=dim, keepdim=True)

        # Compute 1/sqrt(norm_sq) where norm_sq > 0, else 0
        # Use where to avoid NaN in backward
        inv_norm = torch.where(
            norm_sq > 0.0, torch.rsqrt(norm_sq), torch.zeros_like(norm_sq)
        )

        # Normalized output
        normalized = v * inv_norm

        # Save for backward
        ctx.save_for_backward(v, norm_sq, inv_norm)
        ctx.dim = dim
        ctx.keepdim = keepdim

        if not keepdim and inv_norm.shape != v.shape:
            normalized = normalized.squeeze(dim)

        return normalized

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple:
        """
        Implements: grad_v = (1/||v||) * grad_out - (1/||v||^3) * dot(grad_out, v) * v
                    or grad_v = grad_out if ||v|| == 0
        """
        v, norm_sq, inv_norm = ctx.saved_tensors
        dim = ctx.dim
        keepdim = ctx.keepdim

        # If keepdim was False, grad_output is missing the dimension
        # We need to unsqueeze it to match v's shape
        if not keepdim and grad_output.dim() < v.dim():
            grad_output = grad_output.unsqueeze(dim)

        # Compute gradient
        # For zero vectors: grad_v = grad_out (CUDA behavior)
        # For non-zero vectors: grad_v = il * grad_out - il3 * dot(grad_out, v) * v

        # il = 1/||v||
        il = inv_norm

        # il3 = 1/(||v||^3) = (1/||v||) * (1/||v||^2)
        il3 = torch.where(norm_sq > 0.0, inv_norm / norm_sq, torch.zeros_like(inv_norm))

        # dot(grad_out, v)
        dot_product = (grad_output * v).sum(dim=dim, keepdim=True)

        # grad_v = il * grad_out - il3 * dot(grad_out, v) * v for non-zero
        # grad_v = grad_out for zero vectors
        grad_v_nonzero = il * grad_output - il3 * dot_product * v
        grad_v = torch.where(norm_sq > 0.0, grad_v_nonzero, grad_output)

        return grad_v, None, None


def _safe_normalize(
    v: Tensor,
    dim: int = -1,
    keepdim: bool = False,
) -> Tensor:
    """
    Safely normalize a vector, returning zero vector if input norm is zero.

    Uses custom backward pass that matches CUDA implementation.

    Args:
        v: Input tensor to normalize
        dim: Dimension along which to compute the norm
        keepdim: Whether to keep the normalized dimension

    Returns:
        Normalized tensor with same shape as input (or reduced if keepdim=False)
    """
    return SafeNormalize.apply(v, dim, keepdim)


def _rotmat_to_quat(R: Tensor) -> Tensor:
    """Convert rotation matrix to quaternion (w, x, y, z).

    Direct port of GLM's quat_cast with corrected indexing for row-major matrices.
    GLM uses column-major m[col][row], PyTorch uses row-major R[row, col].
    """
    B = R.shape[:-2]
    assert_shape("R", R, B + (3, 3))

    R_flat = R.reshape(-1, 3, 3)

    # Compute decision values: 4*(component^2 - 1)
    fourXSquaredMinus1 = R_flat[:, 0, 0] - R_flat[:, 1, 1] - R_flat[:, 2, 2]
    fourYSquaredMinus1 = R_flat[:, 1, 1] - R_flat[:, 0, 0] - R_flat[:, 2, 2]
    fourZSquaredMinus1 = R_flat[:, 2, 2] - R_flat[:, 0, 0] - R_flat[:, 1, 1]
    fourWSquaredMinus1 = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]  # trace

    # Find largest component
    fourBiggestSquaredMinus1 = torch.stack(
        [
            fourWSquaredMinus1,
            fourXSquaredMinus1,
            fourYSquaredMinus1,
            fourZSquaredMinus1,
        ],
        dim=1,
    )
    biggestIndex = torch.argmax(fourBiggestSquaredMinus1, dim=1)

    # Compute largest component value and multiplier
    biggestVal = (
        torch.sqrt(
            fourBiggestSquaredMinus1.gather(1, biggestIndex.unsqueeze(1)).squeeze(1)
            + 1.0
        )
        * 0.5
    )
    mult = 0.25 / biggestVal

    # Initialize quaternion
    quat = torch.zeros((R_flat.shape[0], 4), dtype=R.dtype, device=R.device)

    # Case 0: w is largest
    # GLM: (biggestVal, (m[1][2] - m[2][1]) * mult, (m[2][0] - m[0][2]) * mult, (m[0][1] - m[1][0]) * mult)
    # PyTorch: m[1][2] = R[2,1], m[2][1] = R[1,2], m[2][0] = R[0,2], m[0][2] = R[2,0], m[0][1] = R[1,0], m[1][0] = R[0,1]
    mask = biggestIndex == 0
    quat[mask, 0] = biggestVal[mask]
    quat[mask, 1] = (R_flat[mask, 2, 1] - R_flat[mask, 1, 2]) * mult[mask]
    quat[mask, 2] = (R_flat[mask, 0, 2] - R_flat[mask, 2, 0]) * mult[mask]
    quat[mask, 3] = (R_flat[mask, 1, 0] - R_flat[mask, 0, 1]) * mult[mask]

    # Case 1: x is largest
    # GLM: ((m[1][2] - m[2][1]) * mult, biggestVal, (m[0][1] + m[1][0]) * mult, (m[2][0] + m[0][2]) * mult)
    mask = biggestIndex == 1
    quat[mask, 0] = (R_flat[mask, 2, 1] - R_flat[mask, 1, 2]) * mult[mask]
    quat[mask, 1] = biggestVal[mask]
    quat[mask, 2] = (R_flat[mask, 1, 0] + R_flat[mask, 0, 1]) * mult[mask]
    quat[mask, 3] = (R_flat[mask, 0, 2] + R_flat[mask, 2, 0]) * mult[mask]

    # Case 2: y is largest
    # GLM: ((m[2][0] - m[0][2]) * mult, (m[0][1] + m[1][0]) * mult, biggestVal, (m[1][2] + m[2][1]) * mult)
    mask = biggestIndex == 2
    quat[mask, 0] = (R_flat[mask, 0, 2] - R_flat[mask, 2, 0]) * mult[mask]
    quat[mask, 1] = (R_flat[mask, 1, 0] + R_flat[mask, 0, 1]) * mult[mask]
    quat[mask, 2] = biggestVal[mask]
    quat[mask, 3] = (R_flat[mask, 2, 1] + R_flat[mask, 1, 2]) * mult[mask]

    # Case 3: z is largest
    # GLM: ((m[0][1] - m[1][0]) * mult, (m[2][0] + m[0][2]) * mult, (m[1][2] + m[2][1]) * mult, biggestVal)
    mask = biggestIndex == 3
    quat[mask, 0] = (R_flat[mask, 1, 0] - R_flat[mask, 0, 1]) * mult[mask]
    quat[mask, 1] = (R_flat[mask, 0, 2] + R_flat[mask, 2, 0]) * mult[mask]
    quat[mask, 2] = (R_flat[mask, 2, 1] + R_flat[mask, 1, 2]) * mult[mask]
    quat[mask, 3] = biggestVal[mask]

    quat = quat.reshape(B + (4,))

    assert_shape("quat", quat, B + (4,))
    return quat


def _quat_normalize_rotation(q: Tensor, dim: int = -1) -> Tensor:
    assert q.shape[dim] == 4, q.shape

    dim = dim if dim >= 0 else q.ndim + dim

    result = _safe_normalize(q, dim=dim)

    # If any quaternion is all zeros along the given dimension, return (1,0,0,0)
    ones = torch.ones_like(result.select(dim, 0))
    zeros = torch.zeros_like(result.select(dim, 0))
    identity = torch.stack([ones, zeros, zeros, zeros], dim=dim)
    result = torch.where(torch.all(q == 0, dim=dim, keepdim=True), identity, result)

    # Make "double-cover" into "single-cover": if w < 0, negate the quaternion
    w_negative = (result.select(dim, 0) < 0).unsqueeze(dim)
    result = torch.where(w_negative, -result, result)

    return result


def _quat_inverse(q: Tensor) -> Tensor:
    """Invert unit quaternion via conjugate.

    Matches glm::inverse(q) for unit quaternions.
    For unit quaternions: inverse = conjugate = (w, -x, -y, -z)

    Args:
        q: Unit quaternion [..., 4] in (w, x, y, z) format

    Returns:
        Inverted quaternion [..., 4]
    """
    # Preconditions
    B = q.shape[:-1]
    assert_shape("q", q, B + (4,))

    result = torch.stack(
        [
            q[..., 0],  # w stays same
            -q[..., 1],  # -x
            -q[..., 2],  # -y
            -q[..., 3],  # -z
        ],
        dim=-1,
    )

    # Postconditions
    assert_shape("result", result, B + (4,))
    return result


def _quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vector v by unit quaternion q.

    Matches glm::rotate(q, v) which computes: q * v * conjugate(q)
    Using quaternion multiplication: result = q * (0, v) * q_conj

    Args:
        q: Unit quaternion [..., 4] in (w, x, y, z) format
        v: Vector [..., 3]

    Returns:
        Rotated vector [..., 3]
    """
    # Preconditions
    B = q.shape[:-1]
    assert_shape("q", q, B + (4,))
    assert_shape("v", v, B + (3,))

    assert abs(q.ndim - v.ndim) <= 1, (q.ndim, v.ndim)

    q = _safe_normalize(q)

    w, x, y, z = torch.unbind(q, dim=-1)  # [..., N]
    vx, vy, vz = torch.unbind(v, dim=-1)  # [..., N]

    # Compute q * (0, v) * q_conjugate using quaternion multiplication formula
    # Result is: v + 2*cross(q.xyz, cross(q.xyz, v) + w*v)
    qvec = torch.stack([x, y, z], dim=-1)  # Imaginary part [..., N, 3]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    uv = uv * (2.0 * w[..., None])
    uuv = uuv * 2.0
    result = v + uv + uuv

    # Postconditions
    assert_shape("result", result, B + (3,))
    return result


def _quat_multiply(q1: Tensor, q2: Tensor, dim: int = -1) -> Tensor:
    """Multiply two quaternions (w, x, y, z format).

    Args:
        q1: First quaternion with 4 components along 'dim'
        q2: Second quaternion with 4 components along 'dim'
        dim: Dimension along which the quaternions are stored

    Returns:
        Product q1 * q2 as quaternion with 4 components along 'dim'
    """

    # Preconditions
    dim = dim + q1.ndim if dim < 0 else dim
    A = q1.shape[:dim]
    B = q1.shape[dim + 1 :]
    assert_shape("q1", q1, A + (4,) + B)
    assert_shape("q2", q2, A + (4,) + B)

    # Decompose the quaternions into scalar and vector parts
    w1 = torch.narrow(q1, dim, 0, 1)
    v1 = torch.narrow(q1, dim, 1, 3)
    w2 = torch.narrow(q2, dim, 0, 1)
    v2 = torch.narrow(q2, dim, 1, 3)

    w = w1 * w2 - torch.sum(v1 * v2, dim=dim, keepdim=True)
    v = w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=dim)

    result = torch.cat([w, v], dim=dim)

    # Postconditions
    assert_shape("result", result, A + (4,) + B)
    return result


def _quat_slerp(x: Tensor, y: Tensor, t: Tensor) -> Tensor:
    """Spherical linear interpolation between two quaternions.

    Matches glm::slerp implementation exactly.
    Assumes input quaternions are already normalized.

    Args:
        x: Start quaternion [..., 4] in (w, x, y, z) format (normalized)
        y: End quaternion [..., 4] in (w, x, y, z) format (normalized)
        t: Interpolation parameter [...] in [0, 1]

    Returns:
        Interpolated quaternion [..., 4]
    """
    # Preconditions
    B = x.shape[:-1]
    assert_shape("x", x, B + (4,))
    assert_shape("y", y, B + (4,))
    assert_shape("t", t, B)

    a = t[..., None]

    # Compute dot product (cosTheta)
    cosTheta = torch.sum(x * y, dim=-1, keepdim=True)

    # If cosTheta < 0, negate q1 to take short path
    z = torch.where(cosTheta < 0, -y, y)
    cosTheta = torch.abs(cosTheta)

    # Linear interpolation when close to 1 (threshold ≈ 1 - epsilon)
    threshold = 1.0 - 1e-6

    resultLerp = (1.0 - a) * x + a * z

    # Check if any quaternions need slerp (are not close)
    if (cosTheta <= threshold).any():
        theta = torch.acos(cosTheta)
        assert (
            theta.isfinite().all()
        ), f"cosTheta ∈ [{cosTheta.min().item(),cosTheta.max().item()}]"

        sinTheta = torch.sin(theta)
        resultSlerp = (
            torch.sin((1.0 - a) * theta) * x + torch.sin(a * theta) * z
        ) / sinTheta

        # Use linear interpolation where close, slerp where distant
        result = torch.where(cosTheta > threshold, resultLerp, resultSlerp)
    else:
        # All quaternions are close - use lerp for all
        result = resultLerp

    # Post conditions
    assert_shape("result", result, B + (4,))
    return result


def _quat_scale_to_preci_half(quats: Tensor, scales: Tensor) -> Tensor:
    """Compute M = R * (1/scales)"""
    R = _quat_to_rotmat(quats)
    M = R * (1.0 / scales[..., None, :])
    return M


def _quat_to_rotmat(quats: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return R.reshape(quats.shape[:-1] + (3, 3))


def _quat_scale_to_matrix(
    quats: Tensor,  # [..., 4],
    scales: Tensor,  # [..., 3],
) -> Tensor:
    """Convert quaternion and scale to a 3x3 matrix (R * S)."""
    batch_dims = quats.shape[:-1]
    assert quats.shape == batch_dims + (4,), quats.shape
    assert scales.shape == batch_dims + (3,), scales.shape
    R = _quat_to_rotmat(quats)  # [..., 3, 3]
    M = R * scales[..., None, :]  # [..., 3, 3]
    return M


def _quat_scale_to_covar_preci(
    quats: Tensor,  # [..., 4],
    scales: Tensor,  # [..., 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.quat_scale_to_covar_preci()`."""
    batch_dims = quats.shape[:-1]
    assert quats.shape == batch_dims + (4,), quats.shape
    assert scales.shape == batch_dims + (3,), scales.shape
    R = _quat_to_rotmat(quats)  # [..., 3, 3]

    if compute_covar:
        M = R * scales[..., None, :]  # [..., 3, 3]
        covars = torch.einsum("...ij,...kj -> ...ik", M, M)  # [..., 3, 3]
        if triu:
            covars = covars.reshape(batch_dims + (9,))  # [..., 9]
            covars = (
                covars[..., [0, 1, 2, 4, 5, 8]] + covars[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0  # [..., 6]
    if compute_preci:
        P = R * (1 / scales[..., None, :])  # [..., 3, 3]
        precis = torch.einsum("...ij,...kj -> ...ik", P, P)  # [..., 3, 3]
        if triu:
            precis = precis.reshape(batch_dims + (9,))  # [..., 9]
            precis = (
                precis[..., [0, 1, 2, 4, 5, 8]] + precis[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0  # [..., 6]

    return covars if compute_covar else None, precis if compute_preci else None


# ============================================================================
# Polynomial Utilities
# ============================================================================


def compute_inverse_polynomial(forward_poly_coeffs, input_range, num_samples=1000):
    """
    Compute the inverse polynomial coefficients using least squares fitting.

    Given a polynomial f(x) = c0 + c1*x + c2*x^2 + ... + c5*x^5,
    compute g(y) such that g(f(x)) ≈ x using least squares fitting.

    Args:
        forward_poly_coeffs: List or array of 6 coefficients [c0, c1, c2, c3, c4, c5]
        input_range: Tuple (min_val, max_val) for sampling input values
        num_samples: Number of sample points for fitting

    Returns:
        List of 6 inverse polynomial coefficients [k0, k1, k2, k3, k4, k5]

    Raises:
        ValueError: If forward polynomial produces invalid values or if inverse
                   accuracy is insufficient (max error > 0.1% of range)
    """
    # Sample uniformly across input range
    x_samples = np.linspace(
        input_range[0], input_range[1], num_samples, dtype=np.float64
    )

    # Evaluate forward polynomial: y = f(x)
    # np.polyval expects coefficients in descending order [c5, c4, c3, c2, c1, c0]
    forward_coeffs_desc = np.array(forward_poly_coeffs[::-1], dtype=np.float64)
    y_samples = np.polyval(forward_coeffs_desc, x_samples)

    # Check for numerical issues
    if np.any(np.isnan(y_samples)) or np.any(np.isinf(y_samples)):
        raise ValueError("Forward polynomial evaluation produced NaN or Inf values")

    # Fit inverse: x = g(y) with constraint k0=0
    # Build design matrix [y, y^2, y^3, y^4, y^5] (no constant term)
    A = np.column_stack([y_samples**i for i in range(1, 6)])

    # Solve least squares with Tikhonov regularization for numerical stability
    # This prevents rank-deficiency when polynomial is nearly linear
    lambda_reg = 1e-10
    AtA = A.T @ A + lambda_reg * np.eye(5)
    Atb = A.T @ x_samples
    inverse_coeffs_no_k0 = np.linalg.solve(AtA, Atb)

    # Prepend k0=0 and return in ascending order [k0, k1, k2, k3, k4, k5]
    inverse_coeffs = np.concatenate([[0.0], inverse_coeffs_no_k0])

    if np.any(np.isnan(inverse_coeffs)) or np.any(np.isinf(inverse_coeffs)):
        raise ValueError("Least squares solution produced NaN or Inf values")

    # Verify inverse accuracy: test that g(f(x)) ≈ x on a subset of samples
    test_indices = np.linspace(0, len(x_samples) - 1, 100, dtype=int)
    x_reconstructed = np.polyval(inverse_coeffs[::-1], y_samples[test_indices])
    errors = np.abs(x_reconstructed - x_samples[test_indices])
    max_error = float(np.max(errors))

    # Tolerance: 0.1% of range
    tolerance = (input_range[1] - input_range[0]) * 1e-3
    if max_error > tolerance:
        raise ValueError(
            f"Inverse polynomial accuracy insufficient: max_error={max_error:.6e} "
            f"exceeds tolerance={tolerance:.6e}. Try increasing num_samples."
        )

    return inverse_coeffs.astype(np.float32).tolist()
