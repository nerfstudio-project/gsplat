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

"""Tests for mathematical operations in gsplat.cuda._math module."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_quat_normalize_rotation():
    """Test _quat_normalize_rotation handles sign and normalization correctly."""
    from gsplat.cuda._math import _quat_normalize_rotation

    # Test 1: Already normalized with w>0 (should be unchanged)
    q_norm_pos_w = torch.tensor([[0.8, 0.2, 0.2, 0.4]], device=device)
    q_norm_pos_w = F.normalize(q_norm_pos_w, p=2, dim=-1)
    n1 = _quat_normalize_rotation(q_norm_pos_w)
    # w>0, should match normalization only
    torch.testing.assert_close(n1, q_norm_pos_w)

    # Test 2: Already normalized with w<0 (should flip sign)
    q_norm_neg_w = torch.tensor([[-0.8, 0.2, 0.2, 0.4]], device=device)
    q_norm_neg_w = F.normalize(q_norm_neg_w, p=2, dim=-1)
    n2 = _quat_normalize_rotation(q_norm_neg_w)
    # The result should be -q2, and w>0
    torch.testing.assert_close(n2, -q_norm_neg_w)
    assert (n2[:, 0] > 0).all()

    # Test 3: Non-normalized, w<0
    q_unnorm_neg_w = torch.tensor([[-2.0, 0.8, 0.3, 0.6]], device=device)
    n3 = _quat_normalize_rotation(q_unnorm_neg_w)
    true_norm = torch.linalg.norm(q_unnorm_neg_w, dim=-1, keepdim=True)
    expected = -q_unnorm_neg_w / true_norm
    torch.testing.assert_close(n3, expected)
    assert (n3[:, 0] > 0).all()

    # Test 4: Many quats, mix of w>0 and w<0 and all axes
    q_batch_mixed = torch.randn(20, 4, device=device)
    # Induce some w < 0
    q_batch_mixed[:10, 0] *= -1
    nqs = _quat_normalize_rotation(q_batch_mixed)
    # Should be normalized
    l2 = torch.linalg.norm(nqs, dim=-1)
    torch.testing.assert_close(l2, torch.ones_like(l2))
    # All output w > 0
    assert (nqs[:, 0] > 0).all(), f"Output w components: {nqs[:,0]}"

    # Test 5: Edge case, zeros (should return identity quaternion [1,0,0,0])
    q_zeros = torch.zeros(1, 4, device=device)
    result = _quat_normalize_rotation(q_zeros)
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    torch.testing.assert_close(result, expected)

    # Gradient check - concatenate all forward test inputs
    q_grad = (
        torch.cat(
            [
                q_norm_pos_w,
                q_norm_neg_w,
                q_unnorm_neg_w,
                q_batch_mixed,
            ],
            dim=0,
        )
        .double()
        .requires_grad_(True)
    )
    assert torch.autograd.gradcheck(
        lambda x: _quat_normalize_rotation(F.normalize(x, p=2, dim=-1)),
        q_grad,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_quat_to_rotmat():
    """Test _quat_to_rotmat converts quaternions correctly."""
    from gsplat.cuda._math import _quat_to_rotmat

    # Test 1: Identity quaternion → identity matrix
    q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    R = _quat_to_rotmat(q_identity)
    I = torch.eye(3, device=device).unsqueeze(0)
    torch.testing.assert_close(R, I)

    # Test 2: 180-degree rotation around X-axis
    # q = (0, 1, 0, 0)
    q_x180 = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
    R = _quat_to_rotmat(q_x180)
    R_expected = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]], device=device
    )
    torch.testing.assert_close(R, R_expected)

    # Test 3: random
    q_rand = torch.randn(100, 4, device=device)
    q_rand = F.normalize(q_rand, p=2, dim=-1)
    R = _quat_to_rotmat(q_rand)

    # Check rotation matrix properties
    # 1. Orthogonal: R @ R^T = I
    I = torch.eye(3, device=device).unsqueeze(0).expand(100, 3, 3)
    RRt = torch.matmul(R, R.transpose(-2, -1))
    torch.testing.assert_close(RRt, I)

    # 2. Determinant = 1
    det = torch.linalg.det(R)
    torch.testing.assert_close(det, torch.ones_like(det))

    # Gradient check - concatenate all forward test inputs
    q_grad = (
        torch.cat([q_identity, q_x180, q_rand], dim=0).double().requires_grad_(True)
    )
    assert torch.autograd.gradcheck(
        lambda x: _quat_to_rotmat(F.normalize(x, p=2, dim=-1)),
        q_grad,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_rotmat_to_quat():
    """Test _rotmat_to_quat converts rotation matrices correctly."""
    from gsplat.cuda._math import _rotmat_to_quat
    import math as m

    # Test 1: Identity matrix → identity quaternion (w is largest, trace > 0)
    I = torch.eye(3, device=device).unsqueeze(0)
    q = _rotmat_to_quat(I)
    q_expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    torch.testing.assert_close(q, q_expected)

    # Test 2: Known rotation matrix (90° around Z)
    c, s = 0.0, 1.0  # cos(90°), sin(90°)
    R_z90 = torch.tensor([[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]], device=device)
    q = _rotmat_to_quat(R_z90)
    # Expected: (cos(45°), 0, 0, sin(45°)) for 90° around Z
    q_expected = torch.tensor(
        [[m.cos(m.pi / 4), 0.0, 0.0, m.sin(m.pi / 4)]], device=device
    )
    torch.testing.assert_close(q, q_expected)

    # Test 3: 180° rotation around X axis (x is largest component)
    # R[0,0] = 1, R[1,1] = R[2,2] = -1, trace = -1
    R_x180 = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]], device=device
    )
    q = _rotmat_to_quat(R_x180)
    # Expected: (0, 1, 0, 0) for 180° around X
    q_expected = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
    torch.testing.assert_close(q, q_expected, atol=1e-6, rtol=1e-6)

    # Test 4: 180° rotation around Y axis (y is largest component)
    # R[1,1] = 1, R[0,0] = R[2,2] = -1, trace = -1
    R_y180 = torch.tensor(
        [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]], device=device
    )
    q = _rotmat_to_quat(R_y180)
    # Expected: (0, 0, 1, 0) for 180° around Y
    q_expected = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device)
    torch.testing.assert_close(q, q_expected, atol=1e-6, rtol=1e-6)

    # Test 5: 180° rotation around Z axis (z is largest component)
    # R[2,2] = 1, R[0,0] = R[1,1] = -1, trace = -1
    R_z180 = torch.tensor(
        [[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]], device=device
    )
    q = _rotmat_to_quat(R_z180)
    # Expected: (0, 0, 0, 1) for 180° around Z
    q_expected = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    torch.testing.assert_close(q, q_expected, atol=1e-6, rtol=1e-6)

    # Gradient check - combine all specialized inputs into one tensor

    R_grad = (
        torch.cat([I, R_z90, R_x180, R_y180, R_z180], dim=0)
        .double()
        .requires_grad_(True)
    )
    assert torch.autograd.gradcheck(
        lambda x: _rotmat_to_quat(x),
        R_grad,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_rotmat_to_quat_roundtrip():
    """Test _rotmat_to_quat and _quat_to_rotmat are inverses."""
    from gsplat.cuda._math import (
        _rotmat_to_quat,
        _quat_to_rotmat,
        _quat_normalize_rotation,
    )

    # Start with random quaternions
    quats = torch.randn(100, 4, device=device)
    quats = F.normalize(quats, p=2, dim=-1)

    # Convert to rotation matrix
    R = _quat_to_rotmat(quats)

    # Convert back to quaternion
    quats_back = _rotmat_to_quat(R)

    torch.testing.assert_close(
        _quat_normalize_rotation(quats_back), _quat_normalize_rotation(quats)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_quat_inverse():
    """Test _quat_inverse produces correct inverse."""
    from gsplat.cuda._math import _quat_inverse, _quat_rotate

    # Generate random unit quaternions
    q_rand = torch.randn(100, 4, device=device)
    q_rand = F.normalize(q_rand, p=2, dim=-1)

    # Compute inverse
    q_rand_inv = _quat_inverse(q_rand)

    # Test 1: Inverse should also be unit quaternion
    lengths = torch.norm(q_rand_inv, dim=-1)
    torch.testing.assert_close(lengths, torch.ones_like(lengths))

    # Test 2: q * q_inv should be identity rotation
    # Quaternion multiplication: (w1, v1) * (w2, v2) = (w1*w2 - dot(v1,v2), w1*v2 + w2*v1 + cross(v1,v2))
    w1, v1 = q_rand[:, 0:1], q_rand[:, 1:]
    w2, v2 = q_rand_inv[:, 0:1], q_rand_inv[:, 1:]

    w_prod = w1 * w2 - torch.sum(v1 * v2, dim=-1, keepdim=True)
    # For q * q_inv, result should be (1, 0, 0, 0)
    torch.testing.assert_close(w_prod, torch.ones_like(w_prod))
    v_prod = w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=-1)
    torch.testing.assert_close(v_prod, torch.zeros_like(v_prod))

    # Test 3: Rotating a vector by q then q_inv should return original vector
    vecs = torch.randn(100, 3, device=device)
    vecs_rotated = _quat_rotate(q_rand, vecs)
    vecs_back = _quat_rotate(q_rand_inv, vecs_rotated)
    torch.testing.assert_close(vecs_back, vecs)

    # Gradient check
    q_grad = q_rand.double().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda x: _quat_inverse(F.normalize(x, p=2, dim=-1)),
        q_grad,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_quat_rotate():
    """Test _quat_rotate rotates vectors correctly."""
    from gsplat.cuda._math import _quat_rotate

    # Test 1: Identity quaternion (1, 0, 0, 0) should not rotate
    q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    v_rand = torch.randn(1, 3, device=device)
    v_rot = _quat_rotate(q_identity, v_rand)
    torch.testing.assert_close(v_rot, v_rand)

    # Test 2: 90-degree rotation around Z-axis
    # q = (cos(45°), 0, 0, sin(45°))
    import math as m

    # 90-degree rotations around X, Y, and Z axes
    q_rot_90deg = torch.tensor(
        [
            [m.cos(m.pi / 4), m.sin(m.pi / 4), 0.0, 0.0],  # 90 deg around X
            [m.cos(m.pi / 4), 0.0, m.sin(m.pi / 4), 0.0],  # 90 deg around Y
            [m.cos(m.pi / 4), 0.0, 0.0, m.sin(m.pi / 4)],  # 90 deg around Z
        ],
        device=device,
    )
    v_unit = torch.tensor(
        [
            [0.0, 1.0, 0.0],  # Y unit vector, will rotate to Z
            [1.0, 0.0, 0.0],  # X unit vector, will rotate to Z
            [1.0, 0.0, 0.0],  # X unit vector, will rotate to Y
        ],
        device=device,
    )
    v_expected = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # Y to Z after X-rot
            [0.0, 0.0, -1.0],  # X to -Z after Y-rot
            [0.0, 1.0, 0.0],  # X to Y after Z-rot
        ],
        device=device,
    )
    v_rot = _quat_rotate(q_rot_90deg, v_unit)
    torch.testing.assert_close(v_rot, v_expected)

    # Test 3: Rotation preserves vector length
    q_batch = torch.randn(50, 4, device=device)
    q_batch = F.normalize(q_batch, p=2, dim=-1)
    v_batch = torch.randn(50, 3, device=device)

    vecs_rotated = _quat_rotate(q_batch, v_batch)

    orig_lengths = torch.norm(v_batch, dim=-1)
    rot_lengths = torch.norm(vecs_rotated, dim=-1)
    torch.testing.assert_close(orig_lengths, rot_lengths)

    # Gradient check - concatenate all forward test inputs
    q_grad = (
        torch.cat([q_identity, q_rot_90deg, q_batch], dim=0)
        .double()
        .requires_grad_(True)
    )
    v_grad = torch.cat([v_rand, v_unit, v_batch], dim=0).double().requires_grad_(True)

    # Test gradients w.r.t. all inputs
    assert torch.autograd.gradcheck(
        lambda q, v: _quat_rotate(F.normalize(q, p=2, dim=-1), v),
        (q_grad, v_grad),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_quat_slerp():
    """Test _quat_slerp interpolation."""
    from gsplat.cuda._math import _quat_slerp, _quat_to_rotmat, _quat_normalize_rotation

    # Test 1: Interpolation at t=0 returns q0
    q0_rand = torch.randn(10, 4, device=device)
    q1_rand = torch.randn(10, 4, device=device)
    q0_rand = F.normalize(q0_rand, p=2, dim=-1)
    q1_rand = F.normalize(q1_rand, p=2, dim=-1)

    t_zero = torch.zeros(10, device=device)
    q_interp = _quat_slerp(q0_rand, q1_rand, t_zero)
    # At t=0, should give same rotation as q0
    torch.testing.assert_close(
        _quat_normalize_rotation(q_interp), _quat_normalize_rotation(q0_rand)
    )

    # Test 2: Interpolation at t=1 gives same rotation as q1
    t_one = torch.ones(10, device=device)
    q_interp = _quat_slerp(q0_rand, q1_rand, t_one)
    torch.testing.assert_close(
        _quat_normalize_rotation(q_interp), _quat_normalize_rotation(q1_rand)
    )

    # Test 3: Interpolation half-way between +45° and -45° around Z gives identity (no rotation)
    import math as m

    angle = m.pi / 4  # 45 degrees in radians
    q_p45 = torch.tensor(
        [[m.cos(angle / 2), 0.0, 0.0, m.sin(angle / 2)]], device=device
    )
    q_m45 = torch.tensor(
        [[m.cos(-angle / 2), 0.0, 0.0, m.sin(-angle / 2)]], device=device
    )

    t_half = torch.tensor([0.5], device=device)
    q_mid = _quat_slerp(q_p45, q_m45, t_half)  # Should produce identity rotation

    # Check normalization
    lengths = torch.norm(q_mid, dim=-1)
    torch.testing.assert_close(lengths, torch.ones_like(lengths))

    # Instead of comparing the quaternion directly, compare its rotation matrix to the identity matrix.
    torch.testing.assert_close(
        _quat_normalize_rotation(q_mid),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
    )

    # Test 4: Slerp produces smooth interpolation (rotation matrix check)
    # Convert to rotation matrices and check intermediate is "between" q0 and q1
    t_rand = torch.rand(10, device=device)
    q_slerp = _quat_slerp(q0_rand, q1_rand, t_rand)
    R_slerp = _quat_to_rotmat(q_mid)

    # All should be valid rotation matrices (det=1)
    det_slerp = torch.linalg.det(R_slerp)
    torch.testing.assert_close(det_slerp, torch.ones_like(det_slerp))

    # Test 5: Very close quaternions use linear interpolation
    q_close = q0_rand + torch.randn_like(q0_rand) * 1e-7  # Very close to q0
    q_close = F.normalize(q_close, p=2, dim=-1)
    q_interp_close = _quat_slerp(q0_rand, q_close, t_half)

    # Should still produce valid quaternion
    lengths = torch.norm(q_interp_close, dim=-1)
    torch.testing.assert_close(lengths, torch.ones_like(lengths))

    # Gradient check - test gradients for q0, q1, and t
    q0_grad = (
        torch.cat([q0_rand, q0_rand, q_p45, q0_rand, q0_rand], dim=0)
        .double()
        .requires_grad_(True)
    )
    q1_grad = (
        torch.cat([q1_rand, q1_rand, q_m45, q1_rand, q_close], dim=0)
        .double()
        .requires_grad_(True)
    )
    t_grad = (
        torch.cat([t_zero, t_one, t_half, t_rand, t_half.repeat(10)], dim=0)
        .double()
        .requires_grad_(True)
    )

    print(f"q0_grad: {q0_grad.shape}, q1_grad: {q1_grad.shape}, t_grad: {t_grad.shape}")

    assert torch.autograd.gradcheck(
        lambda q0, q1, t: _quat_slerp(
            F.normalize(q0, p=2, dim=-1), F.normalize(q1, p=2, dim=-1), t
        ),
        (q0_grad, q1_grad, t_grad),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_quat_multiply():
    """Test _quat_multiply function for various quaternion multiplication scenarios."""
    from gsplat.cuda._math import _quat_multiply

    # Create batch of basis quaternions: [1, i, j, k]
    basis = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0, 0.0],  # i
            [0.0, 0.0, 1.0, 0.0],  # j
            [0.0, 0.0, 0.0, 1.0],  # k
        ],
        dtype=torch.float32,
        device=device,
    )

    # Multiply basis with itself to get full Hamilton multiplication table
    # Each row i will contain: basis[i] * basis[j] for all j
    basis_expanded = basis.unsqueeze(1)  # (4, 1, 4)
    basis_repeated = basis.unsqueeze(0).expand(4, 4, 4)  # (4, 4, 4)
    result = _quat_multiply(basis_expanded, basis_repeated, dim=2)  # (4, 4, 4)

    # Expected Hamilton multiplication table:
    expected = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],  # 1 * 1 = 1
                [0.0, 1.0, 0.0, 0.0],  # 1 * i = i
                [0.0, 0.0, 1.0, 0.0],  # 1 * j = j
                [0.0, 0.0, 0.0, 1.0],
            ],  # 1 * k = k
            [
                [0.0, 1.0, 0.0, 0.0],  # i * 1 = i
                [-1.0, 0.0, 0.0, 0.0],  # i * i = -1
                [0.0, 0.0, 0.0, 1.0],  # i * j = k
                [0.0, 0.0, -1.0, 0.0],
            ],  # i * k = -j
            [
                [0.0, 0.0, 1.0, 0.0],  # j * 1 = j
                [0.0, 0.0, 0.0, -1.0],  # j * i = -k
                [-1.0, 0.0, 0.0, 0.0],  # j * j = -1
                [0.0, 1.0, 0.0, 0.0],
            ],  # j * k = i
            [
                [0.0, 0.0, 0.0, 1.0],  # k * 1 = k
                [0.0, 0.0, 1.0, 0.0],  # k * i = j
                [0.0, -1.0, 0.0, 0.0],  # k * j = -i
                [-1.0, 0.0, 0.0, 0.0],
            ],  # k * k = -1
        ],
        dtype=torch.float32,
        device=device,
    )

    torch.testing.assert_close(result, expected)

    # Gradient check - test gradients for all inputs
    q0_grad = basis_expanded.double().requires_grad_(True)
    q1_grad = basis_repeated.double().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda q0, q1: _quat_multiply(
            F.normalize(q0, p=2, dim=-1), F.normalize(q1, p=2, dim=-1), dim=2
        ),
        (q0_grad, q1_grad),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_safe_normalize():
    """Test _safe_normalize matches expected behavior."""
    from gsplat.cuda._math import _safe_normalize

    # Test normal vectors
    v_rand = torch.randn(100, 3, device=device)
    v_norm = _safe_normalize(v_rand)

    # Check all are unit length
    lengths = torch.norm(v_norm, dim=-1)
    torch.testing.assert_close(lengths, torch.ones_like(lengths))

    # Test zero vectors remain zero
    v_zero = torch.zeros(10, 3, device=device)
    v_zero_norm = _safe_normalize(v_zero)
    torch.testing.assert_close(v_zero_norm, v_zero, rtol=0, atol=0)

    # Test mixed zero and non-zero
    v_mixed = torch.randn(20, 3, device=device)
    v_mixed[::2] = 0  # Every other vector is zero
    v_mixed_norm = _safe_normalize(v_mixed)

    # Non-zero should be normalized
    torch.testing.assert_close(
        torch.norm(v_mixed_norm[1::2], dim=-1), torch.ones(10, device=device)
    )
    # Zero should remain zero
    torch.testing.assert_close(
        v_mixed_norm[::2], torch.zeros(10, 3, device=device), rtol=0, atol=0
    )

    # Gradients check - test with non-zero vectors only
    v_rand_grad = v_rand[v_rand.abs() != 0].double().requires_grad_(True)
    assert v_rand_grad.numel() > 0
    assert torch.autograd.gradcheck(
        lambda x: _safe_normalize(x),
        v_rand_grad,
    )

    # Gradients check - test with zero vectors
    v_zero_grad = v_zero.requires_grad_(True)
    out = _safe_normalize(v_zero_grad)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    # Must yield the same gradient as the input's
    torch.testing.assert_close(v_zero_grad.grad, grad_out)
