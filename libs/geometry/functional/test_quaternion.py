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

"""Comprehensive unit tests for quaternion operations.

Exercises the public API via :mod:`gsplat_geometry.functional` and pulls
backend-only symbols from :mod:`gsplat_geometry.kernels.quaternion_ops` where
needed (autograd ``Function`` classes, reference helpers).
"""

import unittest

import torch

import gsplat_geometry.functional as quat


if not torch.cuda.is_available():
    raise unittest.SkipTest("Geometry quaternion tests require CUDA")


# Test configuration
device = torch.device("cuda")
TEST_SEED = 42

# Test parameters
NUM_QUATERNIONS = 128
ATOL = 1e-6


def random_quaternions(n: int, normalized: bool = True) -> torch.Tensor:
    """Generate random quaternions.

    Args:
        n: Number of quaternions
        normalized: Whether to normalize the quaternions

    Returns:
        (n, 4) tensor of quaternions in xyzw format
    """
    quats = torch.randn(n, 4, device=device)
    if normalized:
        quats = quat.quat_normalize_safe(quats)
    return quats


def random_vectors(n: int) -> torch.Tensor:
    """Generate random 3D vectors.

    Args:
        n: Number of vectors

    Returns:
        (n, 3) tensor of vectors
    """
    return torch.randn(n, 3, device=device)


def quat_to_matrix_pytorch(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix using pure PyTorch (ground truth).

    Args:
        q: (..., 4) quaternion(s) in xyzw format

    Returns:
        (..., 3, 3) rotation matrix/matrices
    """
    # Normalize first
    q = q / torch.norm(q, dim=-1, keepdim=True)

    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Precompute repeated terms
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, xw = x * y, x * z, x * w
    yz, yw, zw = y * z, y * w, z * w

    # Build rotation matrix
    shape = q.shape[:-1] + (3, 3)
    matrix = torch.zeros(shape, dtype=q.dtype, device=q.device)

    matrix[..., 0, 0] = 1.0 - 2.0 * (y2 + z2)
    matrix[..., 0, 1] = 2.0 * (xy - zw)
    matrix[..., 0, 2] = 2.0 * (xz + yw)

    matrix[..., 1, 0] = 2.0 * (xy + zw)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x2 + z2)
    matrix[..., 1, 2] = 2.0 * (yz - xw)

    matrix[..., 2, 0] = 2.0 * (xz - yw)
    matrix[..., 2, 1] = 2.0 * (yz + xw)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x2 + y2)

    return matrix


class TestQuaternionOperations(unittest.TestCase):
    """Test quaternion operations using the CUDA-backed public API."""

    def setUp(self):
        """Reset RNG state before each test for deterministic random inputs."""
        torch.manual_seed(TEST_SEED)
        torch.cuda.manual_seed_all(TEST_SEED)

    def test_normalize_safe(self):
        """Test quaternion normalization."""
        quats = torch.randn(NUM_QUATERNIONS, 4, device=device)
        normalized = quat.quat_normalize_safe(quats)

        # Check that result is unit quaternions
        norms = torch.norm(normalized, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

    def test_normalize_safe_degenerate(self):
        """Test normalization with near-zero quaternions."""
        quats = torch.zeros(10, 4, device=device)
        quats[5:] = 1e-10  # Very small values

        normalized = quat.quat_normalize_safe(quats)

        # Should return identity quaternion
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        self.assertTrue(
            torch.allclose(normalized, expected.expand_as(normalized), atol=ATOL)
        )

    def test_normalize_safe_autograd_gradcheck(self):
        """Backward for quat_normalize_safe matches numeric Jacobian (CUDA ext regression)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatNormalizeSafeFunction

        n = 8
        q = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: QuatNormalizeSafeFunction.apply(x),
                (q,),
                eps=1e-6,
                atol=ATOL,
                rtol=1e-3,
            )
        )

    def test_normalize_safe_degenerate_backward_zero(self):
        """Degenerate input (near-zero norm): gradient w.r.t. quat is zero."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatNormalizeSafeFunction

        q = torch.zeros(6, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        y = QuatNormalizeSafeFunction.apply(q)
        y.backward(torch.randn_like(y))
        self.assertTrue(torch.allclose(q.grad, torch.zeros_like(q), atol=ATOL))

    def test_conjugate(self):
        """Test quaternion conjugate."""
        quats = random_quaternions(NUM_QUATERNIONS)
        conj = quat.quat_conjugate(quats)

        # xyz should be negated, w should stay the same
        self.assertTrue(torch.allclose(conj[..., :3], -quats[..., :3], atol=ATOL))
        self.assertTrue(torch.allclose(conj[..., 3], quats[..., 3], atol=ATOL))

    def test_conjugate_autograd_gradcheck(self):
        """Backward for quat_conjugate matches numeric Jacobian (CUDA ext)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatConjugateFunction

        n = 8
        q = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: QuatConjugateFunction.apply(x),
                (q,),
                eps=1e-6,
                atol=ATOL,
                rtol=1e-3,
            )
        )

    def test_conjugate_backward_cuda_matches_reference(self):
        """Native CUDA backward matches analytical grad (-gx,-gy,-gz,gw)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import (
            QuatConjugateFunction,
            _quat_conjugate_backward_reference,
        )

        n = 32
        q = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        g_up = torch.randn(n, 4, device="cuda", dtype=torch.float64)
        y = QuatConjugateFunction.apply(q)
        y.backward(g_up)
        g_cuda = q.grad.detach()
        g_ref = _quat_conjugate_backward_reference(g_up)
        self.assertTrue(torch.allclose(g_cuda, g_ref, atol=ATOL, rtol=1e-12))

    def test_inverse(self):
        """Test quaternion inverse."""
        quats = random_quaternions(NUM_QUATERNIONS)
        inv = quat.quat_inverse(quats)

        # For unit quaternions, q * q^(-1) = identity
        product = quat.quat_multiply(quats, inv)
        identity = quat.quat_identity((NUM_QUATERNIONS,), device=device)

        self.assertTrue(torch.allclose(product, identity, atol=ATOL))

    def test_inverse_matches_conjugate(self):
        """Inverse API is an alias of conjugate for the supported unit-quaternion contract."""
        quats = random_quaternions(NUM_QUATERNIONS)
        self.assertTrue(
            torch.allclose(
                quat.quat_inverse(quats), quat.quat_conjugate(quats), atol=ATOL
            )
        )

    def test_inverse_autograd_gradcheck(self):
        """quat_inverse follows the migrated conjugate path; gradcheck on public API."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        n = 8
        q = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: quat.quat_inverse(x),
                (q,),
                eps=1e-6,
                atol=ATOL,
                rtol=1e-3,
            )
        )

    def test_multiply(self):
        """Test quaternion multiplication."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)
        q3 = random_quaternions(NUM_QUATERNIONS)

        # Test associativity: (q1 * q2) * q3 = q1 * (q2 * q3)
        result1 = quat.quat_multiply(quat.quat_multiply(q1, q2), q3)
        result2 = quat.quat_multiply(q1, quat.quat_multiply(q2, q3))

        self.assertTrue(torch.allclose(result1, result2, atol=ATOL))

    def test_multiply_identity(self):
        """Test multiplication with identity."""
        quats = random_quaternions(NUM_QUATERNIONS)
        identity = quat.quat_identity((NUM_QUATERNIONS,), device=device)

        result = quat.quat_multiply(quats, identity)
        self.assertTrue(torch.allclose(result, quats, atol=ATOL))

    def test_multiply_matches_rotation_matrix_product(self):
        """Quaternion composition should match rotation-matrix composition."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        quat_product = quat.quat_multiply(q1, q2)
        matrix_from_product = quat.quat_to_matrix(quat_product)
        matrix_product = torch.bmm(quat.quat_to_matrix(q1), quat.quat_to_matrix(q2))

        self.assertTrue(torch.allclose(matrix_from_product, matrix_product, atol=ATOL))

    def test_multiply_autograd_gradcheck(self):
        """Backward for quat multiply matches numeric Jacobian (regression for CUDA ext)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatMultiplyFunction

        n = 8
        q1 = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        q2 = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda a, b: QuatMultiplyFunction.apply(a, b),
                (q1, q2),
                eps=1e-6,
                atol=ATOL,
                rtol=1e-3,
            )
        )

    def test_rotate_vector_autograd_gradcheck(self):
        """Backward for quat_rotate_vector matches numeric Jacobian (regression for CUDA ext)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatRotateVectorFunction

        n = 8
        q = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        v = torch.randn(n, 3, device="cuda", dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda a, b: QuatRotateVectorFunction.apply(a, b),
                (q, v),
                eps=1e-6,
                atol=ATOL,
                rtol=1e-3,
            )
        )

    def test_to_matrix_autograd_gradcheck(self):
        """Backward for quat_to_matrix matches numeric Jacobian (regression for CUDA ext)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatToMatrixFunction

        n = 8
        q = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda a: QuatToMatrixFunction.apply(a),
                (q,),
                eps=1e-6,
                atol=ATOL,
                rtol=1e-3,
            )
        )

    def test_to_matrix(self):
        """Test quaternion to matrix conversion."""
        quats = random_quaternions(NUM_QUATERNIONS)
        matrices = quat.quat_to_matrix(quats)

        # Compare with PyTorch ground truth
        matrices_expected = quat_to_matrix_pytorch(quats)
        self.assertTrue(torch.allclose(matrices, matrices_expected, atol=ATOL))

    def test_matrix_is_rotation(self):
        """Test that quaternion to matrix produces valid rotation matrices."""
        quats = random_quaternions(NUM_QUATERNIONS)
        matrices = quat.quat_to_matrix(quats)

        # Check orthogonality: R^T * R = I
        identity = (
            torch.eye(3, device=device).unsqueeze(0).expand(NUM_QUATERNIONS, 3, 3)
        )
        product = torch.bmm(matrices.transpose(-2, -1), matrices)
        self.assertTrue(torch.allclose(product, identity, atol=ATOL))

        # Check determinant = 1
        det = torch.linalg.det(matrices)
        self.assertTrue(torch.allclose(det, torch.ones_like(det), atol=ATOL))

    def test_rotate_vector_vs_matrix(self):
        """Test that vector rotation matches matrix multiplication."""
        quats = random_quaternions(NUM_QUATERNIONS)
        vectors = random_vectors(NUM_QUATERNIONS)

        # Rotate using quaternion
        rotated_quat = quat.quat_rotate_vector(quats, vectors)

        # Rotate using matrix
        matrices = quat.quat_to_matrix(quats)
        rotated_matrix = torch.bmm(matrices, vectors.unsqueeze(-1)).squeeze(-1)

        # Should be the same
        self.assertTrue(torch.allclose(rotated_quat, rotated_matrix, atol=ATOL))

    def test_rotate_preserves_length(self):
        """Test that rotation preserves vector length."""
        quats = random_quaternions(NUM_QUATERNIONS)
        vectors = random_vectors(NUM_QUATERNIONS)

        original_norms = torch.norm(vectors, dim=-1)
        rotated = quat.quat_rotate_vector(quats, vectors)
        rotated_norms = torch.norm(rotated, dim=-1)

        self.assertTrue(torch.allclose(original_norms, rotated_norms, atol=ATOL))

    def test_from_axis_angle(self):
        """Test axis-angle to quaternion conversion."""
        # Generate random axes and angles
        axes = random_vectors(NUM_QUATERNIONS)
        axes = axes / torch.norm(axes, dim=-1, keepdim=True)  # Normalize
        angles = torch.rand(NUM_QUATERNIONS, device=device) * 2 * torch.pi

        # Convert to quaternion
        quats = quat.quat_from_axis_angle(axes, angles)

        # Check that result is unit quaternions
        norms = torch.norm(quats, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

    def test_from_axis_angle_autograd_gradcheck(self):
        """Backward for quat_from_axis_angle matches numeric Jacobian."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatFromAxisAngleFunction

        n = 8
        axis = torch.randn(n, 3, device="cuda", dtype=torch.float64, requires_grad=True)
        angle = torch.randn(
            n, 1, device="cuda", dtype=torch.float64, requires_grad=True
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                QuatFromAxisAngleFunction.apply,
                (axis, angle),
                eps=1e-6,
                atol=ATOL,
                rtol=1e-3,
            )
        )

    def test_from_axis_angle_backward_cuda_matches_pytorch_reference(self):
        """CUDA backward matches autograd on the PyTorch reference formula."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatFromAxisAngleFunction

        def axis_angle_to_quat_ref(a: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
            half = ang * 0.5
            s = torch.sin(half)
            c = torch.cos(half)
            return torch.cat([a * s, c], dim=-1)

        n = 24
        g_up = torch.randn(n, 4, device="cuda", dtype=torch.float64)
        axis = torch.randn(n, 3, device="cuda", dtype=torch.float64, requires_grad=True)
        angle = torch.randn(
            n, 1, device="cuda", dtype=torch.float64, requires_grad=True
        )

        y_ext = QuatFromAxisAngleFunction.apply(axis, angle)
        y_ref = axis_angle_to_quat_ref(axis, angle)
        self.assertTrue(
            torch.allclose(y_ext.detach(), y_ref.detach(), atol=ATOL, rtol=1e-7)
        )

        y_ext.backward(g_up.clone())
        ga_ext, gan_ext = axis.grad.clone(), angle.grad.clone()
        axis.grad = None
        angle.grad = None
        y_ref.backward(g_up)
        self.assertTrue(torch.allclose(ga_ext, axis.grad, atol=ATOL, rtol=1e-5))
        self.assertTrue(torch.allclose(gan_ext, angle.grad, atol=ATOL, rtol=1e-5))

    def test_lerp_endpoints(self):
        """Test linear interpolation endpoints."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        # Test endpoints
        result_0 = quat.quat_lerp(q1, q2, 0.0)
        result_1 = quat.quat_lerp(q1, q2, 1.0)

        # At t=0, should be close to q1 (or -q1) - element-wise check
        matches_q1 = (torch.abs(result_0 - q1) < ATOL).all(dim=1)
        matches_neg_q1 = (torch.abs(result_0 + q1) < ATOL).all(dim=1)
        all_match_q1 = (matches_q1 | matches_neg_q1).all()
        self.assertTrue(all_match_q1)

        # At t=1, should be close to q2 (or -q2) - element-wise check
        matches_q2 = (torch.abs(result_1 - q2) < ATOL).all(dim=1)
        matches_neg_q2 = (torch.abs(result_1 + q2) < ATOL).all(dim=1)
        all_match_q2 = (matches_q2 | matches_neg_q2).all()
        self.assertTrue(all_match_q2)

    def test_lerp_autograd_gradcheck(self):
        """Backward for quat_lerp matches numeric Jacobian."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatLerpFunction

        n = 8
        q1 = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        q2 = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        t = 0.35
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda a, b: QuatLerpFunction.apply(a, b, t),
                (q1, q2),
                eps=1e-6,
                atol=ATOL,
                rtol=1e-3,
            )
        )

    def test_lerp_backward_cuda_matches_pytorch_reference(self):
        """CUDA quat_lerp backward matches autograd on a PyTorch reference."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatLerpFunction

        def lerp_ref(q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
            d = (q1 * q2).sum(dim=-1, keepdim=True)
            q2e = torch.where(d < 0, -q2, q2)
            r = (1.0 - t) * q1 + t * q2e
            return r / r.norm(dim=-1, keepdim=True)

        n = 32
        t = 0.41
        g_up = torch.randn(n, 4, device="cuda", dtype=torch.float64)

        q1 = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        q2_raw = torch.randn(n, 4, device="cuda", dtype=torch.float64)
        q2_raw[:6] = -q2_raw[:6] * 0.9
        q2 = q2_raw.clone().requires_grad_(True)

        y_ext = QuatLerpFunction.apply(q1, q2, t)
        y_ref = lerp_ref(q1, q2, t)
        self.assertTrue(
            torch.allclose(y_ext.detach(), y_ref.detach(), atol=ATOL, rtol=1e-5)
        )

        y_ext.backward(g_up.clone())
        g1_ext, g2_ext = q1.grad.clone(), q2.grad.clone()
        q1.grad = None
        q2.grad = None
        y_ref.backward(g_up)
        self.assertTrue(torch.allclose(g1_ext, q1.grad, atol=ATOL, rtol=1e-4))
        self.assertTrue(torch.allclose(g2_ext, q2.grad, atol=ATOL, rtol=1e-4))

    def _slerp_batched_torch_reference(
        self, q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor
    ):
        """Reference SLERP with hemisphere handling, clamp, and lerp fallback."""
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

    def test_slerp_batched_cuda_matches_torch_reference(self):
        """CUDA forward matches differentiable torch reference (float64)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatSlerpBatchedFunction

        n = 32
        q1 = torch.randn(n, 4, device="cuda", dtype=torch.float64)
        q2 = torch.randn(n, 4, device="cuda", dtype=torch.float64)
        q2[:5] = -q2[:5]
        t = torch.rand(n, 1, device="cuda", dtype=torch.float64)
        out_ext = QuatSlerpBatchedFunction.apply(q1, q2, t)
        out_ref = self._slerp_batched_torch_reference(q1, q2, t)
        same = torch.allclose(out_ext, out_ref, atol=ATOL, rtol=1e-5)
        same_neg = torch.allclose(out_ext, -out_ref, atol=ATOL, rtol=1e-5)
        self.assertTrue(same or same_neg)

    def test_slerp_batched_gradcheck_cuda_ext(self):
        """Gradcheck for QuatSlerpBatchedFunction on native CUDA path."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatSlerpBatchedFunction

        n = 6
        q1 = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        q2 = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        t = torch.rand(n, 1, device="cuda", dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                QuatSlerpBatchedFunction.apply,
                (q1, q2, t),
                eps=1e-6,
                atol=ATOL,
                rtol=1e-3,
            )
        )

    def test_slerp_batched_backward_matches_torch_reference(self):
        """CUDA SLERP backward matches a differentiable PyTorch reference."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatSlerpBatchedFunction

        n = 6
        q1_data = torch.nn.functional.normalize(
            torch.randn(n, 4, device="cuda", dtype=torch.float32), dim=-1
        )
        q2_seed = torch.randn(n, 4, device="cuda", dtype=torch.float32)
        proj = (q2_seed * q1_data).sum(dim=-1, keepdim=True) * q1_data
        q2_orthogonal = torch.nn.functional.normalize(q2_seed - proj, dim=-1)
        # Avoid the nondifferentiable hemisphere boundary at dot(q1, q2) == 0.
        # The CUDA kernel and PyTorch reference both branch on the dot sign;
        # keeping a small positive margin makes this a stable backward parity
        # test instead of a comparison of two valid branch-side choices.
        q2_data = torch.nn.functional.normalize(q2_orthogonal + 0.25 * q1_data, dim=-1)
        q1 = q1_data.requires_grad_()
        q2 = q2_data.requires_grad_()
        t = (
            0.2 + 0.6 * torch.rand(n, 1, device="cuda", dtype=torch.float32)
        ).requires_grad_()
        g_up = torch.randn(n, 4, device="cuda", dtype=torch.float32)

        y = QuatSlerpBatchedFunction.apply(q1, q2, t)
        y.backward(g_up)
        gq1, gq2, gt = q1.grad.clone(), q2.grad.clone(), t.grad.clone()

        q1_ref = q1.detach().clone().requires_grad_(True)
        q2_ref = q2.detach().clone().requires_grad_(True)
        t_ref = t.detach().clone().requires_grad_(True)
        y_ref = self._slerp_batched_torch_reference(q1_ref, q2_ref, t_ref)
        y_ref.backward(g_up)
        self.assertTrue(torch.allclose(gq1, q1_ref.grad, atol=ATOL, rtol=2e-3))
        self.assertTrue(torch.allclose(gq2, q2_ref.grad, atol=ATOL, rtol=2e-3))
        self.assertTrue(torch.allclose(gt, t_ref.grad, atol=ATOL, rtol=2e-3))

    def test_slerp_endpoints(self):
        """Test spherical linear interpolation endpoints."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        # Test endpoints
        result_0 = quat.quat_slerp(q1, q2, 0.0)
        result_1 = quat.quat_slerp(q1, q2, 1.0)

        # At t=0, should be close to q1 (or -q1) - element-wise check
        matches_q1 = (torch.abs(result_0 - q1) < ATOL).all(dim=1)
        matches_neg_q1 = (torch.abs(result_0 + q1) < ATOL).all(dim=1)
        all_match_q1 = (matches_q1 | matches_neg_q1).all()
        self.assertTrue(all_match_q1)

        # At t=1, should be close to q2 (or -q2) - element-wise check
        matches_q2 = (torch.abs(result_1 - q2) < ATOL).all(dim=1)
        matches_neg_q2 = (torch.abs(result_1 + q2) < ATOL).all(dim=1)
        all_match_q2 = (matches_q2 | matches_neg_q2).all()
        self.assertTrue(all_match_q2)

    def test_slerp_unit_quaternions(self):
        """Test that SLERP produces unit quaternions."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        for t in t_values:
            result = quat.quat_slerp(q1, q2, t)
            norms = torch.norm(result, dim=-1)
            self.assertTrue(
                torch.allclose(norms, torch.ones_like(norms), atol=ATOL),
                f"Failed at t={t}",
            )

    def test_slerp_batched_endpoints(self):
        """Test batched SLERP with per-element t at endpoints."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        # Test t=0 for all
        t_zeros = torch.zeros(NUM_QUATERNIONS, device=device)
        result_0 = quat.quat_slerp(q1, q2, t_zeros)

        matches_q1 = (torch.abs(result_0 - q1) < ATOL).all(dim=1)
        matches_neg_q1 = (torch.abs(result_0 + q1) < ATOL).all(dim=1)
        self.assertTrue((matches_q1 | matches_neg_q1).all())

        # Test t=1 for all
        t_ones = torch.ones(NUM_QUATERNIONS, device=device)
        result_1 = quat.quat_slerp(q1, q2, t_ones)

        matches_q2 = (torch.abs(result_1 - q2) < ATOL).all(dim=1)
        matches_neg_q2 = (torch.abs(result_1 + q2) < ATOL).all(dim=1)
        self.assertTrue((matches_q2 | matches_neg_q2).all())

    def test_slerp_batched_per_element_t(self):
        """Test batched SLERP with varying per-element t values."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        # Random t values per element
        t = torch.rand(NUM_QUATERNIONS, device=device)
        result = quat.quat_slerp(q1, q2, t)

        # Result should be unit quaternions
        norms = torch.norm(result, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

    def test_slerp_batched_matches_scalar_slerp(self):
        """Test that batched SLERP with uniform t matches scalar SLERP."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            # Scalar slerp
            result_scalar = quat.quat_slerp(q1, q2, t_val)

            # Batched slerp with uniform t
            t_batched = torch.full((NUM_QUATERNIONS,), t_val, device=device)
            result_batched = quat.quat_slerp(q1, q2, t_batched)

            # Should match (accounting for sign ambiguity)
            matches = (torch.abs(result_scalar - result_batched) < ATOL).all(dim=1)
            matches_neg = (torch.abs(result_scalar + result_batched) < ATOL).all(dim=1)
            self.assertTrue((matches | matches_neg).all(), f"Mismatch at t={t_val}")

    def test_slerp_batched_unit_quaternions(self):
        """Test that batched SLERP produces unit quaternions for all t values."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        # Test with random t values
        t = torch.rand(NUM_QUATERNIONS, device=device)
        result = quat.quat_slerp(q1, q2, t)
        norms = torch.norm(result, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

    def test_manifold_interp_endpoints(self):
        """Test manifold interpolation endpoints."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        # Test endpoints
        result_0 = quat.quat_manifold_interp(q1, q2, 0.0)
        result_1 = quat.quat_manifold_interp(q1, q2, 1.0)

        # At t=0, should be close to q1 (or -q1) - element-wise check
        matches_q1 = (torch.abs(result_0 - q1) < ATOL).all(dim=1)
        matches_neg_q1 = (torch.abs(result_0 + q1) < ATOL).all(dim=1)
        all_match_q1 = (matches_q1 | matches_neg_q1).all()
        self.assertTrue(all_match_q1)

        # At t=1, should be close to q2 (or -q2) - element-wise check
        matches_q2 = (torch.abs(result_1 - q2) < ATOL).all(dim=1)
        matches_neg_q2 = (torch.abs(result_1 + q2) < ATOL).all(dim=1)
        all_match_q2 = (matches_q2 | matches_neg_q2).all()
        self.assertTrue(all_match_q2)

    def test_manifold_batched_endpoints(self):
        """Manifold interpolation with tensor t at 0 and 1 matches scalar endpoints."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        t_zeros = torch.zeros(NUM_QUATERNIONS, device=device)
        result_0 = quat.quat_manifold_interp(q1, q2, t_zeros)
        matches_q1 = (torch.abs(result_0 - q1) < ATOL).all(dim=1)
        matches_neg_q1 = (torch.abs(result_0 + q1) < ATOL).all(dim=1)
        self.assertTrue((matches_q1 | matches_neg_q1).all())

        t_ones = torch.ones(NUM_QUATERNIONS, device=device)
        result_1 = quat.quat_manifold_interp(q1, q2, t_ones)
        matches_q2 = (torch.abs(result_1 - q2) < ATOL).all(dim=1)
        matches_neg_q2 = (torch.abs(result_1 + q2) < ATOL).all(dim=1)
        self.assertTrue((matches_q2 | matches_neg_q2).all())

    def test_manifold_batched_per_element_t(self):
        """Per-row tensor t produces unit quaternions (random t per element)."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)
        t = torch.rand(NUM_QUATERNIONS, device=device)
        result = quat.quat_manifold_interp(q1, q2, t)
        norms = torch.norm(result, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

    def test_manifold_batched_matches_scalar_interp(self):
        """Uniform batched t matches Python scalar t (sign ambiguity allowed)."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result_scalar = quat.quat_manifold_interp(q1, q2, t_val)
            t_batched = torch.full((NUM_QUATERNIONS,), t_val, device=device)
            result_batched = quat.quat_manifold_interp(q1, q2, t_batched)
            matches = (torch.abs(result_scalar - result_batched) < ATOL).all(dim=1)
            matches_neg = (torch.abs(result_scalar + result_batched) < ATOL).all(dim=1)
            self.assertTrue((matches | matches_neg).all(), f"Mismatch at t={t_val}")

    def test_manifold_interp_autograd_gradcheck(self):
        """Backward for manifold interp matches numeric Jacobian (CUDA ext when loaded)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatManifoldInterpFunction

        n = 8
        for t_val in (0.1, 0.5, 0.9):
            q1 = torch.randn(
                n, 4, device="cuda", dtype=torch.float64, requires_grad=True
            )
            q2 = torch.randn(
                n, 4, device="cuda", dtype=torch.float64, requires_grad=True
            )
            t_fix = torch.full((n, 1), t_val, device="cuda", dtype=torch.float64)
            self.assertTrue(
                torch.autograd.gradcheck(
                    lambda a, b: QuatManifoldInterpFunction.apply(a, b, t_fix),
                    (q1, q2),
                    eps=1e-6,
                    atol=ATOL,
                    rtol=1e-3,
                ),
                f"gradcheck failed at t={t_val}",
            )

    def test_manifold_interp_autograd_gradcheck_includes_t(self):
        """gradcheck w.r.t. q1, q2, and per-row t."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatManifoldInterpFunction

        n = 4
        q1 = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        q2 = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        t = (
            0.15 + 0.65 * torch.rand(n, 1, device="cuda", dtype=torch.float64)
        ).requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda a, b, c: QuatManifoldInterpFunction.apply(a, b, c),
                (q1, q2, t),
                eps=1e-6,
                atol=ATOL,
                rtol=1e-3,
            )
        )

    def test_quat_manifold_interp_public_api_gradcheck_includes_t(self):
        """Public quat_manifold_interp autograd w.r.t. q1, q2, and tensor t."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        n = 4
        q1 = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        q2 = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        t = (
            0.15 + 0.65 * torch.rand(n, device="cuda", dtype=torch.float64)
        ).requires_grad_(True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda a, b, c: quat.quat_manifold_interp(a, b, c),
                (q1, q2, t),
                eps=1e-6,
                atol=ATOL,
                rtol=1e-3,
            )
        )

    def test_angular_distance(self):
        """Test angular distance computation."""
        quats = random_quaternions(NUM_QUATERNIONS)

        # Distance to itself should be ~0 (renormalize + acos near 1 needs slack vs exact 0)
        dist_self = quat.quat_angular_distance(quats, quats)
        self.assertTrue(
            torch.allclose(dist_self, torch.zeros_like(dist_self), atol=2e-3)
        )

        # Distance is symmetric
        q2 = random_quaternions(NUM_QUATERNIONS)
        dist_12 = quat.quat_angular_distance(quats, q2)
        dist_21 = quat.quat_angular_distance(q2, quats)
        self.assertTrue(torch.allclose(dist_12, dist_21, atol=ATOL))

        # Distance should be in [0, pi]
        self.assertTrue((dist_12 >= 0).all())
        self.assertTrue((dist_12 <= torch.pi).all())

    def _angular_distance_torch_reference(
        self, q1: torch.Tensor, q2: torch.Tensor
    ) -> torch.Tensor:
        """Reference angular distance using normalize, abs dot, clamp, and 2*acos."""
        n1 = q1 / q1.norm(dim=-1, keepdim=True)
        n2 = q2 / q2.norm(dim=-1, keepdim=True)
        d = (n1 * n2).sum(dim=-1, keepdim=True)
        c = d.abs().clamp(min=0.0, max=1.0)
        return 2.0 * torch.acos(c)

    def test_angular_distance_autograd_gradcheck(self):
        """Backward for quat_angular_distance matches numeric Jacobian."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatAngularDistanceFunction

        n = 8
        q1 = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        q2 = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda a, b: QuatAngularDistanceFunction.apply(a, b),
                (q1, q2),
                eps=1e-6,
                atol=ATOL,
                rtol=1e-3,
            )
        )

    def test_angular_distance_backward_cuda_matches_pytorch_reference(self):
        """CUDA angular_distance backward matches autograd on a PyTorch reference."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        from gsplat_geometry.kernels.quaternion_ops import QuatAngularDistanceFunction

        n = 32
        g_up = torch.randn(n, device="cuda", dtype=torch.float64)

        q1 = torch.randn(n, 4, device="cuda", dtype=torch.float64, requires_grad=True)
        q2_base = torch.randn(n, 4, device="cuda", dtype=torch.float64)
        q2_base[:8] = -q2_base[:8] * 1.02
        q2 = q2_base.clone().requires_grad_(True)

        y_ext = QuatAngularDistanceFunction.apply(q1, q2)
        y_ref = self._angular_distance_torch_reference(q1, q2).squeeze(-1)
        self.assertTrue(
            torch.allclose(y_ext.detach(), y_ref.detach(), atol=ATOL, rtol=1e-5)
        )

        y_ext.backward(g_up.clone())
        g1_ext, g2_ext = q1.grad.clone(), q2.grad.clone()
        q1.grad = None
        q2.grad = None
        y_ref.backward(g_up)
        self.assertTrue(torch.allclose(g1_ext, q1.grad, atol=ATOL, rtol=1e-4))
        self.assertTrue(torch.allclose(g2_ext, q2.grad, atol=ATOL, rtol=1e-4))

    def test_batch_shapes(self):
        """Test that functions work with various batch shapes."""
        # Test with different batch shapes
        shapes = [(5,), (3, 4), (2, 3, 4)]

        for shape in shapes:
            q1 = torch.randn(*shape, 4, device=device)
            q1 = quat.quat_normalize_safe(q1)

            q2 = torch.randn(*shape, 4, device=device)
            q2 = quat.quat_normalize_safe(q2)

            # Test various operations preserve shape
            result = quat.quat_conjugate(q1)
            self.assertEqual(result.shape, (*shape, 4))

            result = quat.quat_multiply(q1, q2)
            self.assertEqual(result.shape, (*shape, 4))

            result = quat.quat_to_matrix(q1)
            self.assertEqual(result.shape, (*shape, 3, 3))

            vectors = torch.randn(*shape, 3, device=device)
            result = quat.quat_rotate_vector(q1, vectors)
            self.assertEqual(result.shape, (*shape, 3))


class TestQuaternionPublicValidation(unittest.TestCase):
    """Python-level validation errors for kernel-backed quaternion APIs."""

    def test_non_tensor_input_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, "must be a torch.Tensor"):
            quat.quat_normalize_safe([1.0, 0.0, 0.0, 0.0])  # type: ignore[arg-type]

    def test_cpu_tensor_raises_value_error(self):
        q = torch.randn(2, 4, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            quat.quat_normalize_safe(q)

    def test_wrong_last_dim_raises_value_error(self):
        q = torch.randn(2, 3, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "last dimension 4"):
            quat.quat_normalize_safe(q)

    def test_quat_multiply_dtype_mismatch_raises(self):
        q1 = torch.randn(2, 4, device=device, dtype=torch.float32)
        q2 = torch.randn(2, 4, device=device, dtype=torch.float64)
        with self.assertRaisesRegex(TypeError, "same dtype"):
            quat.quat_multiply(q1, q2)

    def test_quat_multiply_shape_mismatch_raises(self):
        q1 = torch.randn(1, 4, device=device, dtype=torch.float32)
        q2 = torch.randn(3, 4, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "same shape"):
            quat.quat_multiply(q1, q2)

    def test_quat_rotate_vector_batch_mismatch_raises(self):
        q = torch.randn(2, 4, device=device, dtype=torch.float32)
        v = torch.randn(3, 3, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "batch dimensions"):
            quat.quat_rotate_vector(q, v)

    def test_quat_rotate_vector_rejects_broadcast_like_batch_shapes(self):
        q = torch.randn(1, 4, device=device, dtype=torch.float32)
        v = torch.randn(5, 3, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "batch dimensions"):
            quat.quat_rotate_vector(q, v)

    def test_quat_slerp_t_batch_mismatch_raises(self):
        q = torch.randn(4, 4, device=device, dtype=torch.float32)
        t = torch.tensor([0.0, 0.5], device=device, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "batch size 1 or 4"):
            quat.quat_slerp(q, q, t)

    def test_quat_slerp_accepts_tensor_t_matching_nontrivial_batch_shape(self):
        q1 = random_quaternions(6).reshape(2, 3, 4)
        q2 = random_quaternions(6).reshape(2, 3, 4)
        t = torch.tensor(
            [[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]],
            device=device,
            dtype=torch.float32,
        )

        result = quat.quat_slerp(q1, q2, t)

        self.assertEqual(result.shape, q1.shape)
        for i in range(q1.shape[0]):
            for j in range(q1.shape[1]):
                expected = quat.quat_slerp(
                    q1[i, j].unsqueeze(0), q2[i, j].unsqueeze(0), float(t[i, j].item())
                ).squeeze(0)
                self.assertTrue(
                    torch.allclose(result[i, j], expected, atol=ATOL)
                    or torch.allclose(result[i, j], -expected, atol=ATOL)
                )

    def test_quat_manifold_interp_t_batch_mismatch_raises(self):
        q = torch.randn(4, 4, device=device, dtype=torch.float32)
        t = torch.tensor([0.0, 0.5], device=device, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "batch size 1 or 4"):
            quat.quat_manifold_interp(q, q, t)

    def test_quat_manifold_interp_rejects_non_float_tensor_t(self):
        q = torch.randn(2, 4, device=device, dtype=torch.float32)
        t = torch.tensor([0, 1], device=device, dtype=torch.int32)
        with self.assertRaisesRegex(TypeError, "floating-point tensor"):
            quat.quat_manifold_interp(q, q, t)

    def test_quat_manifold_interp_accepts_tensor_t_matching_nontrivial_batch_shape(
        self,
    ):
        q1 = random_quaternions(6).reshape(2, 3, 4)
        q2 = random_quaternions(6).reshape(2, 3, 4)
        t = torch.tensor(
            [[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        result = quat.quat_manifold_interp(q1, q2, t)
        self.assertEqual(result.shape, q1.shape)
        for i in range(q1.shape[0]):
            for j in range(q1.shape[1]):
                expected = quat.quat_manifold_interp(
                    q1[i, j].unsqueeze(0),
                    q2[i, j].unsqueeze(0),
                    float(t[i, j].item()),
                ).squeeze(0)
                self.assertTrue(
                    torch.allclose(result[i, j], expected, atol=ATOL)
                    or torch.allclose(result[i, j], -expected, atol=ATOL)
                )

    def test_quat_manifold_interp_zero_dim_tensor_t_matches_scalar(self):
        """0-dim float tensor t matches Python scalar (same semantics as quat_slerp)."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)
        for t_val in (0.0, 0.37, 1.0):
            t0 = torch.tensor(t_val, device=device, dtype=torch.float32)
            self.assertEqual(t0.dim(), 0)
            r_s = quat.quat_manifold_interp(q1, q2, t_val)
            r_t = quat.quat_manifold_interp(q1, q2, t0)
            self.assertTrue(
                torch.allclose(r_s, r_t, atol=ATOL)
                or torch.allclose(r_s, -r_t, atol=ATOL)
            )

    def test_quat_manifold_interp_singleton_tensor_t_broadcasts(self):
        """Flattened size-1 tensor t broadcasts to full batch like quat_slerp."""
        q1 = random_quaternions(5)
        q2 = random_quaternions(5)
        t_val = 0.42
        r_scalar = quat.quat_manifold_interp(q1, q2, t_val)
        for t_singleton in (
            torch.tensor([[t_val]], device=device, dtype=torch.float32),
            torch.tensor([t_val], device=device, dtype=torch.float32),
        ):
            r_b = quat.quat_manifold_interp(q1, q2, t_singleton)
            self.assertTrue(
                torch.allclose(r_scalar, r_b, atol=ATOL)
                or torch.allclose(r_scalar, -r_b, atol=ATOL)
            )

    def test_quat_lerp_rejects_tensor_t(self):
        q = torch.randn(2, 4, device=device, dtype=torch.float32)
        t = torch.tensor(0.5, device=device)
        with self.assertRaisesRegex(TypeError, "scalar Python"):
            quat.quat_lerp(q, q, t)  # type: ignore[arg-type]

    def test_quat_lerp_shape_mismatch_raises(self):
        q1 = torch.randn(1, 4, device=device, dtype=torch.float32)
        q2 = torch.randn(2, 4, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "same shape"):
            quat.quat_lerp(q1, q2, 0.5)

    def test_quat_from_axis_angle_batch_mismatch_raises(self):
        axis = torch.randn(4, 3, device=device, dtype=torch.float32)
        angle = torch.randn(2, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "batch mismatch"):
            quat.quat_from_axis_angle(axis, angle)

    def test_quat_angular_distance_shape_mismatch_raises(self):
        q1 = torch.randn(1, 4, device=device, dtype=torch.float32)
        q2 = torch.randn(4, 4, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "same shape"):
            quat.quat_angular_distance(q1, q2)

    def test_quat_identity_requires_explicit_device(self):
        with self.assertRaisesRegex(
            TypeError, "required keyword-only argument: 'device'"
        ):
            quat.quat_identity()  # type: ignore[call-arg]
        with self.assertRaisesRegex(
            ValueError, r"requires an explicit `device` argument"
        ):
            quat.quat_identity(device=None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
