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

"""Unit tests for pose and trajectory operations.

Uses :mod:`gsplat_geometry.functional` for quaternion helpers and
:mod:`gsplat_geometry.kernels.pose_ops` for pose APIs and autograd internals.

Pose gradchecks live alongside the related API test classes: SE3 ops use
float64 on CUDA, while trajectory ops use float32-only native bindings with
relaxed tolerances. ``frame_transform_poses_tquat`` is not registered for
autograd (frame transform is passed as Python floats), so it has no gradcheck.
"""

import unittest
import warnings

import torch

import gsplat_geometry.functional as geom
from gsplat_geometry.kernels.pose_ops import (
    SE3PoseFromMatrixFunction,
    SE3PoseInverseTransformDirectionFunction,
    SE3PoseInverseTransformPointFunction,
    SE3PoseToInverseMatrixFunction,
    SE3PoseToMatrixFunction,
    SE3PoseTransformDirectionFunction,
    SE3PoseTransformPointFunction,
    TrajectoryGetRotation2PosesFunction,
    TrajectoryTransformPoint1PoseFunction,
    TrajectoryTransformPoint2PosesFunction,
    frame_transform_poses_tquat,
)

if not torch.cuda.is_available():
    raise unittest.SkipTest("Geometry pose tests require CUDA")

quat_multiply = geom.quat_multiply
quat_from_axis_angle = geom.quat_from_axis_angle
quat_rotate_vector = geom.quat_rotate_vector
quat_slerp = geom.quat_slerp
quat_to_matrix = geom.quat_to_matrix
se3pose_from_matrix = geom.se3pose_from_matrix
se3pose_inverse_transform_direction = geom.se3pose_inverse_transform_direction
se3pose_inverse_transform_point = geom.se3pose_inverse_transform_point
se3pose_to_inverse_matrix = geom.se3pose_to_inverse_matrix
se3pose_to_matrix = geom.se3pose_to_matrix
se3pose_transform_direction = geom.se3pose_transform_direction
se3pose_transform_point = geom.se3pose_transform_point
trajectory_get_rotation_2poses = geom.trajectory_get_rotation_2poses
trajectory_transform_point_1pose = geom.trajectory_transform_point_1pose
trajectory_transform_point_2poses = geom.trajectory_transform_point_2poses


device = torch.device("cuda")
TEST_SEED = 42

# Test parameters (align with quaternion_test.py style)
NUM_POSES = 10000
NUM_POSES_SMALL = 1000
NUM_POSES_MEDIUM = 5000
ATOL = 1e-5
ATOL_STRICT = 1e-6
RTOL = 1e-5

# --- autograd.gradcheck (SE3 uses float64; trajectory CUDA kernels are float32-only in ext.cpp)
GC_SE3_BATCH = 4
GC_SE3_EPS = 1e-6
GC_SE3_ATOL = 1e-4
GC_SE3_RTOL = 1e-3

GC_TRAJ_BATCH = 4
GC_TRAJ_EPS = 1e-3
GC_TRAJ_ATOL = 5e-2
GC_TRAJ_RTOL = 5e-2

# Shepperd branch selection in pose.cuh (must match CUDA exactly).
def _shepperd_branch_from_R(R: torch.Tensor) -> int:
    """Return branch index 0..3 for a single 3x3 rotation matrix (row-major rows)."""
    r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
    r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
    r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]
    trace = r00 + r11 + r22
    if trace > r00 and trace > r11 and trace > r22:
        return 3
    if r00 > r11 and r00 > r22:
        return 0
    if r11 > r22:
        return 1
    return 2


def _flat_se3_matrix_from_R(
    R: torch.Tensor, translation: torch.Tensor | None = None
) -> torch.Tensor:
    """Row-major 4x4 flattened (16,) matching CUDA layout; upper 3x3 = R, last row [0,0,0,1]."""
    if translation is None:
        translation = torch.zeros(3, device=R.device, dtype=R.dtype)
    flat = torch.zeros(16, device=R.device, dtype=R.dtype)
    flat[0] = R[0, 0]
    flat[1] = R[0, 1]
    flat[2] = R[0, 2]
    flat[3] = translation[0]
    flat[4] = R[1, 0]
    flat[5] = R[1, 1]
    flat[6] = R[1, 2]
    flat[7] = translation[1]
    flat[8] = R[2, 0]
    flat[9] = R[2, 1]
    flat[10] = R[2, 2]
    flat[11] = translation[2]
    flat[15] = 1.0
    return flat


def _se3_gradcheck_tensors(
    n: int, device: torch.device, dtype: torch.dtype = torch.float64
):
    """Build leaf tensors for SE3 gradcheck (translation, raw quaternion, auxiliary vec3)."""
    t = torch.randn(n, 3, device=device, dtype=dtype, requires_grad=True)
    q_raw = torch.randn(n, 4, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(n, 3, device=device, dtype=dtype, requires_grad=True)
    return t, q_raw, v


def _se3_normalized_quat(q_raw: torch.Tensor) -> torch.Tensor:
    return geom.quat_normalize_safe(q_raw)


class TestSE3Pose(unittest.TestCase):
    """Regression tests for SE3 pose API on CUDA (vec3 and matrix-to/from-pose ops may use geometry_cuda)."""

    def setUp(self):
        """Reset RNG state before each test for deterministic random inputs."""
        torch.manual_seed(TEST_SEED)
        torch.cuda.manual_seed_all(TEST_SEED)

    def test_se3pose_transform_point_gradcheck(self):
        n = GC_SE3_BATCH
        t, q_raw, p = _se3_gradcheck_tensors(n, device)

        def fn(tt, qq, pp):
            return se3pose_transform_point(tt, _se3_normalized_quat(qq), pp)

        self.assertTrue(
            torch.autograd.gradcheck(
                fn,
                (t, q_raw, p),
                eps=GC_SE3_EPS,
                atol=GC_SE3_ATOL,
                rtol=GC_SE3_RTOL,
            )
        )

    def test_se3pose_transform_direction_gradcheck(self):
        n = GC_SE3_BATCH
        t, q_raw, d = _se3_gradcheck_tensors(n, device)

        def fn(tt, qq, dd):
            return se3pose_transform_direction(tt, _se3_normalized_quat(qq), dd)

        self.assertTrue(
            torch.autograd.gradcheck(
                fn,
                (t, q_raw, d),
                eps=GC_SE3_EPS,
                atol=GC_SE3_ATOL,
                rtol=GC_SE3_RTOL,
            )
        )

    def test_se3pose_inverse_transform_point_gradcheck(self):
        n = GC_SE3_BATCH
        t, q_raw, p = _se3_gradcheck_tensors(n, device)

        def fn(tt, qq, pp):
            return se3pose_inverse_transform_point(tt, _se3_normalized_quat(qq), pp)

        self.assertTrue(
            torch.autograd.gradcheck(
                fn,
                (t, q_raw, p),
                eps=GC_SE3_EPS,
                atol=GC_SE3_ATOL,
                rtol=GC_SE3_RTOL,
            )
        )

    def test_se3pose_inverse_transform_direction_gradcheck(self):
        n = GC_SE3_BATCH
        t, q_raw, d = _se3_gradcheck_tensors(n, device)

        def fn(tt, qq, dd):
            return se3pose_inverse_transform_direction(tt, _se3_normalized_quat(qq), dd)

        self.assertTrue(
            torch.autograd.gradcheck(
                fn,
                (t, q_raw, d),
                eps=GC_SE3_EPS,
                atol=GC_SE3_ATOL,
                rtol=GC_SE3_RTOL,
            )
        )

    def test_se3pose_to_matrix_gradcheck(self):
        n = GC_SE3_BATCH
        t, q_raw, _ = _se3_gradcheck_tensors(n, device)

        def fn(tt, qq):
            return se3pose_to_matrix(tt, _se3_normalized_quat(qq)).reshape(n, 16)

        self.assertTrue(
            torch.autograd.gradcheck(
                fn,
                (t, q_raw),
                eps=GC_SE3_EPS,
                atol=GC_SE3_ATOL,
                rtol=GC_SE3_RTOL,
            )
        )

    def test_se3pose_to_inverse_matrix_gradcheck(self):
        n = GC_SE3_BATCH
        t, q_raw, _ = _se3_gradcheck_tensors(n, device)

        def fn(tt, qq):
            return se3pose_to_inverse_matrix(tt, _se3_normalized_quat(qq)).reshape(
                n, 16
            )

        self.assertTrue(
            torch.autograd.gradcheck(
                fn,
                (t, q_raw),
                eps=GC_SE3_EPS,
                atol=GC_SE3_ATOL,
                rtol=GC_SE3_RTOL,
            )
        )

    def test_se3pose_to_inverse_matrix_wxyz_gradcheck(self):
        n = GC_SE3_BATCH
        t, q_raw, _ = _se3_gradcheck_tensors(n, device)

        def fn(tt, qq):
            q = _se3_normalized_quat(qq)
            q_wxyz = torch.cat([q[:, 3:4], q[:, :3]], dim=-1)
            return se3pose_to_inverse_matrix(tt, q_wxyz, wxyz_format=True).reshape(
                n, 16
            )

        self.assertTrue(
            torch.autograd.gradcheck(
                fn,
                (t, q_raw),
                eps=GC_SE3_EPS,
                atol=GC_SE3_ATOL,
                rtol=GC_SE3_RTOL,
            )
        )

    def test_se3pose_from_matrix_gradcheck(self):
        n = GC_SE3_BATCH
        with torch.no_grad():
            t0 = torch.randn(n, 3, device=device, dtype=torch.float64)
            q0 = geom.quat_normalize_safe(
                torch.randn(n, 4, device=device, dtype=torch.float64)
            )
            mat0 = se3pose_to_matrix(t0, q0).reshape(n, 16)
        mat = mat0.clone().requires_grad_(True)

        def fn(m):
            trans, rot = se3pose_from_matrix(m)
            return torch.cat([trans, rot], dim=-1)

        self.assertTrue(
            torch.autograd.gradcheck(
                fn,
                (mat,),
                eps=GC_SE3_EPS,
                atol=GC_SE3_ATOL,
                rtol=GC_SE3_RTOL,
            )
        )

    def test_se3pose_from_matrix_shepperd_branch_boundaries_forward(self):
        """Forward round-trip on rotations near Shepperd branch seams (pose.cuh).

        Branch logic compares ``trace`` to diagonals, then ``r00`` vs ``r11`` vs ``r22``.
        We build valid rotation matrices via axis-angle, sweep angles that cross those
        decision surfaces, and require ``quat_to_matrix(se3pose_from_matrix(M))`` to
        recover the upper 3x3 block. float64 avoids unnecessary numerical noise.
        """
        import math

        dtype = torch.float64
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)
        pi = torch.tensor([[math.pi]], device=device, dtype=dtype)
        mats = []

        # trace vs r22 (z-axis): branch 3 when trace > r22; switches near ~90.05° in float64.
        for deg in (85.0, 89.0, 90.0, 90.05, 90.1, 95.0, 100.0):
            ang = torch.tensor([[math.radians(deg)]], device=device, dtype=dtype)
            q = quat_from_axis_angle(z_axis, ang)[0]
            R = quat_to_matrix(q).view(3, 3)
            mats.append(_flat_se3_matrix_from_R(R))

        # r00 vs r11 after trace branch fails: 180° in the xy plane, flip near 45°.
        for deg in (43.0, 44.5, 45.0, 45.05, 46.0, 47.0):
            phi = math.radians(deg)
            axis = torch.tensor(
                [[math.cos(phi), math.sin(phi), 0.0]], device=device, dtype=dtype
            )
            q = quat_from_axis_angle(axis, pi)[0]
            R = quat_to_matrix(q).view(3, 3)
            mats.append(_flat_se3_matrix_from_R(R))

        # r11 vs r22 (180° in the yz plane): flip near 45°.
        for deg in (43.0, 44.5, 45.0, 45.05, 46.0, 47.0):
            phi = math.radians(deg)
            axis = torch.tensor(
                [[0.0, math.cos(phi), math.sin(phi)]], device=device, dtype=dtype
            )
            q = quat_from_axis_angle(axis, pi)[0]
            R = quat_to_matrix(q).view(3, 3)
            mats.append(_flat_se3_matrix_from_R(R))

        M = torch.stack(mats, dim=0)
        _, q_out = se3pose_from_matrix(M)
        R_out = quat_to_matrix(q_out).view(-1, 3, 3)
        R_ref = torch.stack([m.view(4, 4)[:3, :3] for m in M.view(-1, 4, 4)])

        self.assertTrue(torch.allclose(R_out, R_ref, atol=1e-12, rtol=1e-12))

        # At least one sample per seam side (guards a vacuous all-branch-3 batch).
        branches_z = []
        for deg in (85.0, 95.0):
            ang = torch.tensor([[math.radians(deg)]], device=device, dtype=dtype)
            q = quat_from_axis_angle(z_axis, ang)[0]
            R = quat_to_matrix(q).view(3, 3)
            branches_z.append(_shepperd_branch_from_R(R))
        self.assertIn(3, branches_z)
        self.assertIn(2, branches_z)

        branches_xy = []
        for deg in (44.0, 46.0):
            phi = math.radians(deg)
            axis = torch.tensor(
                [[math.cos(phi), math.sin(phi), 0.0]], device=device, dtype=dtype
            )
            q = quat_from_axis_angle(axis, pi)[0]
            R = quat_to_matrix(q).view(3, 3)
            branches_xy.append(_shepperd_branch_from_R(R))
        self.assertIn(0, branches_xy)
        self.assertIn(1, branches_xy)

        branches_yz = []
        for deg in (44.0, 46.0):
            phi = math.radians(deg)
            axis = torch.tensor(
                [[0.0, math.cos(phi), math.sin(phi)]], device=device, dtype=dtype
            )
            q = quat_from_axis_angle(axis, pi)[0]
            R = quat_to_matrix(q).view(3, 3)
            branches_yz.append(_shepperd_branch_from_R(R))
        self.assertIn(1, branches_yz)
        self.assertIn(2, branches_yz)

    def test_se3pose_from_matrix_shepperd_branch_boundaries_backward_finite(self):
        """Gradients w.r.t. matrix stay finite on Shepperd boundary-adjacent rotations."""
        import math

        dtype = torch.float64
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)
        pi = torch.tensor([[math.pi]], device=device, dtype=dtype)
        mats = []

        for deg in (85.0, 90.0, 90.05, 95.0):
            ang = torch.tensor([[math.radians(deg)]], device=device, dtype=dtype)
            q = quat_from_axis_angle(z_axis, ang)[0]
            R = quat_to_matrix(q).view(3, 3)
            mats.append(_flat_se3_matrix_from_R(R))

        for deg in (44.5, 45.0, 45.05):
            phi = math.radians(deg)
            axis_xy = torch.tensor(
                [[math.cos(phi), math.sin(phi), 0.0]], device=device, dtype=dtype
            )
            q = quat_from_axis_angle(axis_xy, pi)[0]
            mats.append(_flat_se3_matrix_from_R(quat_to_matrix(q).view(3, 3)))
            axis_yz = torch.tensor(
                [[0.0, math.cos(phi), math.sin(phi)]], device=device, dtype=dtype
            )
            q = quat_from_axis_angle(axis_yz, pi)[0]
            mats.append(_flat_se3_matrix_from_R(quat_to_matrix(q).view(3, 3)))

        M = torch.stack(mats, dim=0).clone().requires_grad_(True)
        trans, rot = se3pose_from_matrix(M)
        loss = trans.sum() + rot.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(M.grad).all())

    def test_se3pose_from_matrix_shepperd_branch_boundaries_backward_finite_eps_sweep(
        self,
    ):
        """Backward remains finite on both sides of Shepperd seams under small perturbations."""
        import math

        dtype = torch.float64
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)
        pi = torch.tensor([[math.pi]], device=device, dtype=dtype)

        # Coarse pair confirms each construction actually straddles a branch seam.
        coarse_deg = 0.5
        branch_pairs = []
        for sign in (-1.0, 1.0):
            ang = torch.tensor(
                [[math.radians(90.05 + sign * coarse_deg)]], device=device, dtype=dtype
            )
            q = quat_from_axis_angle(z_axis, ang)[0]
            R = quat_to_matrix(q).view(3, 3)
            branch_pairs.append(_shepperd_branch_from_R(R))
        self.assertNotEqual(branch_pairs[0], branch_pairs[1])

        branch_pairs = []
        for sign in (-1.0, 1.0):
            phi = math.radians(45.0 + sign * coarse_deg)
            axis_xy = torch.tensor(
                [[math.cos(phi), math.sin(phi), 0.0]], device=device, dtype=dtype
            )
            q = quat_from_axis_angle(axis_xy, pi)[0]
            R = quat_to_matrix(q).view(3, 3)
            branch_pairs.append(_shepperd_branch_from_R(R))
        self.assertNotEqual(branch_pairs[0], branch_pairs[1])

        branch_pairs = []
        for sign in (-1.0, 1.0):
            phi = math.radians(45.0 + sign * coarse_deg)
            axis_yz = torch.tensor(
                [[0.0, math.cos(phi), math.sin(phi)]], device=device, dtype=dtype
            )
            q = quat_from_axis_angle(axis_yz, pi)[0]
            R = quat_to_matrix(q).view(3, 3)
            branch_pairs.append(_shepperd_branch_from_R(R))
        self.assertNotEqual(branch_pairs[0], branch_pairs[1])

        mats = []
        # Sweep tiny offsets around each seam; include both sides for every epsilon.
        for eps_deg in (1e-2, 1e-3, 1e-4):
            for sign in (-1.0, 1.0):
                ang = torch.tensor(
                    [[math.radians(90.05 + sign * eps_deg)]], device=device, dtype=dtype
                )
                q = quat_from_axis_angle(z_axis, ang)[0]
                mats.append(_flat_se3_matrix_from_R(quat_to_matrix(q).view(3, 3)))

                phi = math.radians(45.0 + sign * eps_deg)
                axis_xy = torch.tensor(
                    [[math.cos(phi), math.sin(phi), 0.0]], device=device, dtype=dtype
                )
                q = quat_from_axis_angle(axis_xy, pi)[0]
                mats.append(_flat_se3_matrix_from_R(quat_to_matrix(q).view(3, 3)))

                axis_yz = torch.tensor(
                    [[0.0, math.cos(phi), math.sin(phi)]], device=device, dtype=dtype
                )
                q = quat_from_axis_angle(axis_yz, pi)[0]
                mats.append(_flat_se3_matrix_from_R(quat_to_matrix(q).view(3, 3)))

        M = torch.stack(mats, dim=0).clone().requires_grad_(True)
        trans, rot = se3pose_from_matrix(M)
        # Keep scalar objective simple but non-trivial in both outputs.
        loss = trans.square().sum() + rot.square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(M.grad).all())

    def test_se3pose_from_matrix_shepperd_off_branch_gradcheck(self):
        """gradcheck away from branch surfaces; finite-difference steps can cross seams if too close."""
        import math

        n = 6
        dtype = torch.float64
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype)
        pi = torch.tensor([[math.pi]], device=device, dtype=dtype)
        mats = []

        for deg in (80.0, 100.0):
            ang = torch.tensor([[math.radians(deg)]], device=device, dtype=dtype)
            q = quat_from_axis_angle(z_axis, ang)[0]
            R = quat_to_matrix(q).view(3, 3)
            for _ in range(n // 2):
                mats.append(_flat_se3_matrix_from_R(R))

        for deg in (40.0, 50.0):
            phi = math.radians(deg)
            axis_xy = torch.tensor(
                [[math.cos(phi), math.sin(phi), 0.0]], device=device, dtype=dtype
            )
            q = quat_from_axis_angle(axis_xy, pi)[0]
            R = quat_to_matrix(q).view(3, 3)
            mats.append(_flat_se3_matrix_from_R(R))
            axis_yz = torch.tensor(
                [[0.0, math.cos(phi), math.sin(phi)]], device=device, dtype=dtype
            )
            q = quat_from_axis_angle(axis_yz, pi)[0]
            R = quat_to_matrix(q).view(3, 3)
            mats.append(_flat_se3_matrix_from_R(R))

        mat0 = torch.stack(mats, dim=0)
        mat = mat0.clone().requires_grad_(True)

        def fn(m):
            trans, rot = se3pose_from_matrix(m)
            return torch.cat([trans, rot], dim=-1)

        self.assertTrue(
            torch.autograd.gradcheck(
                fn,
                (mat,),
                eps=GC_SE3_EPS,
                atol=GC_SE3_ATOL,
                rtol=GC_SE3_RTOL,
            )
        )

    def test_se3_transform_point(self):
        """Test SE3 point transformation - should apply rotation then translation"""
        N = NUM_POSES

        # Random poses and points (use CUDA tensors)
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        points = torch.randn(N, 3, device=device)

        # Public CUDA-backed geometry API
        output = se3pose_transform_point(translations, rotations, points)

        # Compute ground truth: T * p = R(p) + t
        rotated = quat_rotate_vector(rotations, points)
        expected = rotated + translations

        self.assertTrue(torch.allclose(output, expected, atol=ATOL_STRICT))

    def test_se3_transform_direction(self):
        """Test SE3 direction transformation - should only apply rotation, not translation"""
        N = NUM_POSES

        # Random poses and directions
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        directions = torch.randn(N, 3, device=device)

        # Public CUDA-backed geometry API
        output = se3pose_transform_direction(translations, rotations, directions)

        # Compute ground truth: should only rotate, ignore translation
        expected = quat_rotate_vector(rotations, directions)

        self.assertTrue(torch.allclose(output, expected, atol=ATOL_STRICT))

    def test_se3_inverse_transform_point(self):
        """Test SE3 inverse point transformation - should invert the transformation"""
        N = NUM_POSES

        # Random poses and points
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        points = torch.randn(N, 3, device=device)

        # Forward then inverse transform
        forward = se3pose_transform_point(translations, rotations, points)
        inverse = se3pose_inverse_transform_point(translations, rotations, forward)

        # Should recover original points
        self.assertTrue(torch.allclose(inverse, points, atol=ATOL))

    def test_se3_inverse_transform_direction(self):
        """Test SE3 inverse direction transformation - should invert rotation only"""
        N = NUM_POSES

        # Random poses and directions
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        directions = torch.randn(N, 3, device=device)

        # Forward then inverse transform
        forward = se3pose_transform_direction(translations, rotations, directions)
        inverse = se3pose_inverse_transform_direction(translations, rotations, forward)

        # Should recover original directions
        self.assertTrue(torch.allclose(inverse, directions, atol=ATOL))

    def test_se3_to_matrix(self):
        """Test SE3 pose to 4x4 matrix conversion - should produce correct transformation matrix"""
        N = NUM_POSES

        # Random poses
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))

        matrices = se3pose_to_matrix(translations, rotations)

        # Verify structure
        # Upper-left 3x3 should be rotation matrix
        rot_matrices = quat_to_matrix(rotations)  # (N, 9)
        rot_matrices = rot_matrices.view(N, 3, 3)

        self.assertTrue(
            torch.allclose(matrices[:, :3, :3], rot_matrices, atol=ATOL_STRICT)
        )

        # Fourth column should be [t; 1]
        self.assertTrue(
            torch.allclose(matrices[:, :3, 3], translations, atol=ATOL_STRICT)
        )
        self.assertTrue(
            torch.allclose(
                matrices[:, 3, 3], torch.ones(N, device=device), atol=ATOL_STRICT
            )
        )

        # Bottom row (except last element) should be zeros
        self.assertTrue(
            torch.allclose(
                matrices[:, 3, :3], torch.zeros(N, 3, device=device), atol=ATOL_STRICT
            )
        )

    def test_se3_from_matrix(self):
        """Test SE3 matrix to pose conversion - should extract translation and rotation correctly"""
        N = NUM_POSES

        # Random poses
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))

        # Convert to matrices
        matrices = se3pose_to_matrix(translations, rotations)

        # Convert back to translation and rotation
        extracted_trans, extracted_rot = se3pose_from_matrix(matrices)

        # Check translation matches exactly
        self.assertTrue(torch.allclose(extracted_trans, translations, atol=ATOL_STRICT))

        # Check rotation correctness: verify rotations produce same transformation
        # by rotating test vectors and comparing results
        # Note: Quaternions q and -q represent the same rotation, so we can't check
        # quaternion values directly. We must check the actual rotation effect.
        test_vectors = torch.randn(N, 3, device=device)
        rotated_original = quat_rotate_vector(rotations, test_vectors)
        rotated_extracted = quat_rotate_vector(extracted_rot, test_vectors)

        self.assertTrue(torch.allclose(rotated_original, rotated_extracted, atol=ATOL))

    def test_se3_from_matrix_round_trip(self):
        """Test round-trip conversion: pose → matrix → pose should recover original pose"""
        N = NUM_POSES

        # Random poses
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))

        # Round-trip: pose → matrix → pose
        matrices = se3pose_to_matrix(translations, rotations)
        extracted_trans, extracted_rot = se3pose_from_matrix(matrices)

        # Test that extracted pose produces same transformations as original
        test_points = torch.randn(N, 3, device=device)

        transformed_original = se3pose_transform_point(
            translations, rotations, test_points
        )
        transformed_extracted = se3pose_transform_point(
            extracted_trans, extracted_rot, test_points
        )

        self.assertTrue(
            torch.allclose(transformed_original, transformed_extracted, atol=ATOL)
        )

    def test_se3_from_matrix_backward(self):
        """Test that gradients flow correctly through matrix to pose conversion"""
        N = NUM_POSES_SMALL

        # Random matrices with gradients enabled
        translations = torch.randn(N, 3, device=device, requires_grad=True)
        rotations = torch.nn.functional.normalize(
            torch.randn(N, 4, device=device, requires_grad=True)
        )
        rotations.retain_grad()

        # Create matrices
        matrices = se3pose_to_matrix(translations, rotations)

        # Extract pose (this is what we're testing gradients for)
        extracted_trans, extracted_rot = se3pose_from_matrix(matrices)

        # Create a loss from extracted values
        loss = extracted_trans.sum() + extracted_rot.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist and are non-zero
        self.assertIsNotNone(translations.grad)
        self.assertIsNotNone(rotations.grad)
        self.assertTrue(torch.any(translations.grad != 0))
        self.assertTrue(torch.any(rotations.grad != 0))

    def test_se3_to_inverse_matrix(self):
        """Test SE3 pose to inverse 4x4 matrix conversion - should produce correct inverse transformation matrix"""
        N = NUM_POSES

        # Random poses
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))

        inverse_matrices = se3pose_to_inverse_matrix(translations, rotations)

        # Get forward matrices
        forward_matrices = se3pose_to_matrix(translations, rotations)

        # Multiply forward and inverse - should get identity
        identity = torch.bmm(forward_matrices, inverse_matrices)
        expected_identity = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)

        self.assertTrue(torch.allclose(identity, expected_identity, atol=ATOL))

        # Also check that inverse * forward = identity
        identity2 = torch.bmm(inverse_matrices, forward_matrices)
        self.assertTrue(torch.allclose(identity2, expected_identity, atol=ATOL))

    def test_se3_to_inverse_matrix_structure(self):
        """Test SE3 inverse matrix has correct structure - R^T in upper-left, -R^T*t in translation"""
        N = NUM_POSES

        # Random poses
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))

        inverse_matrices = se3pose_to_inverse_matrix(translations, rotations)

        # Get rotation matrix and its transpose
        rot_matrices = quat_to_matrix(rotations).view(N, 3, 3)
        rot_matrices_T = rot_matrices.transpose(1, 2)

        # Upper-left 3x3 should be R^T
        self.assertTrue(
            torch.allclose(
                inverse_matrices[:, :3, :3], rot_matrices_T, atol=ATOL_STRICT
            )
        )

        # Translation part should be -R^T * t
        expected_trans = -torch.bmm(rot_matrices_T, translations.unsqueeze(-1)).squeeze(
            -1
        )
        self.assertTrue(
            torch.allclose(inverse_matrices[:, :3, 3], expected_trans, atol=ATOL_STRICT)
        )

        # Bottom row should be [0, 0, 0, 1]
        self.assertTrue(
            torch.allclose(
                inverse_matrices[:, 3, :3],
                torch.zeros(N, 3, device=device),
                atol=ATOL_STRICT,
            )
        )
        self.assertTrue(
            torch.allclose(
                inverse_matrices[:, 3, 3],
                torch.ones(N, device=device),
                atol=ATOL_STRICT,
            )
        )

    def test_se3_to_inverse_matrix_wxyz_format(self):
        """Test SE3 inverse matrix with wxyz quaternion format"""
        N = NUM_POSES

        # Random poses in xyzw format
        translations = torch.randn(N, 3, device=device)
        rotations_xyzw = torch.nn.functional.normalize(torch.randn(N, 4, device=device))

        # Convert to wxyz format (swap first and last components)
        rotations_wxyz = torch.cat(
            [rotations_xyzw[:, 3:4], rotations_xyzw[:, :3]], dim=1
        )

        # Compute inverse matrix using wxyz format flag
        inverse_matrices_wxyz = se3pose_to_inverse_matrix(
            translations, rotations_wxyz, wxyz_format=True
        )

        # Compute inverse matrix using xyzw format (default)
        inverse_matrices_xyzw = se3pose_to_inverse_matrix(
            translations, rotations_xyzw, wxyz_format=False
        )

        # Both should produce the same result
        self.assertTrue(
            torch.allclose(
                inverse_matrices_wxyz, inverse_matrices_xyzw, atol=ATOL_STRICT
            )
        )

    def test_se3_to_inverse_matrix_transform_point(self):
        """Test that inverse matrix correctly inverse-transforms points"""
        N = NUM_POSES

        # Random poses and points
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        points = torch.randn(N, 3, device=device)

        # Forward transform using SE3 function
        transformed_points = se3pose_transform_point(translations, rotations, points)

        # Get inverse matrix
        inverse_matrices = se3pose_to_inverse_matrix(translations, rotations)

        # Apply inverse matrix to transformed points
        points_homogeneous = torch.cat(
            [transformed_points, torch.ones(N, 1, device=device)], dim=1
        )
        recovered_points = torch.bmm(
            inverse_matrices, points_homogeneous.unsqueeze(-1)
        ).squeeze(-1)[:, :3]

        # Should recover original points
        self.assertTrue(torch.allclose(recovered_points, points, atol=ATOL))

    def test_se3_to_inverse_matrix_backward(self):
        """Test that gradients flow correctly through SE3 to inverse matrix conversion"""
        N = NUM_POSES_SMALL

        # Random poses with gradients enabled
        translations = torch.randn(N, 3, device=device, requires_grad=True)
        rotations = torch.nn.functional.normalize(
            torch.randn(N, 4, device=device, requires_grad=True)
        )
        rotations.retain_grad()  # Need this for non-leaf tensors

        # Forward pass
        inverse_matrices = se3pose_to_inverse_matrix(translations, rotations)

        # Backward pass
        grad_output = torch.randn_like(inverse_matrices)
        inverse_matrices.backward(grad_output)

        # Check that gradients exist and are non-zero
        self.assertIsNotNone(translations.grad)
        self.assertIsNotNone(rotations.grad)
        self.assertTrue(torch.any(translations.grad != 0))
        self.assertTrue(torch.any(rotations.grad != 0))

    def test_se3_to_inverse_matrix_backward_wxyz(self):
        """Test that gradients flow correctly with wxyz format"""
        N = NUM_POSES_SMALL

        # Random poses with gradients enabled (in wxyz format)
        translations = torch.randn(N, 3, device=device, requires_grad=True)
        rotations = torch.nn.functional.normalize(
            torch.randn(N, 4, device=device, requires_grad=True)
        )
        rotations.retain_grad()  # Need this for non-leaf tensors

        # Forward pass with wxyz format
        inverse_matrices = se3pose_to_inverse_matrix(
            translations, rotations, wxyz_format=True
        )

        # Backward pass
        grad_output = torch.randn_like(inverse_matrices)
        inverse_matrices.backward(grad_output)

        # Check that gradients exist and are non-zero
        self.assertIsNotNone(translations.grad)
        self.assertIsNotNone(rotations.grad)
        self.assertTrue(torch.any(translations.grad != 0))
        self.assertTrue(torch.any(rotations.grad != 0))

    def test_se3_identity(self):
        """Test that identity pose doesn't change points"""
        N = NUM_POSES

        # Identity pose
        translations = torch.zeros(N, 3, device=device)
        rotations = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(N, 1)
        points = torch.randn(N, 3, device=device)

        # Transform should be identity
        output = se3pose_transform_point(translations, rotations, points)

        self.assertTrue(torch.allclose(output, points, atol=ATOL_STRICT))

    def test_se3_composition_via_matrix(self):
        """Test that composing poses works correctly via matrix multiplication"""
        N = NUM_POSES_MEDIUM

        # Two random poses
        t1 = torch.randn(N, 3, device=device)
        r1 = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        t2 = torch.randn(N, 3, device=device)
        r2 = torch.nn.functional.normalize(torch.randn(N, 4, device=device))

        # Get matrices and compose
        m1 = se3pose_to_matrix(t1, r1)
        m2 = se3pose_to_matrix(t2, r2)

        # Compose via matrix multiplication
        m_composed = torch.bmm(m1, m2)

        # Test on random points
        points = torch.randn(N, 3, device=device)
        points_homogeneous = torch.cat([points, torch.ones(N, 1, device=device)], dim=1)

        # Apply composed transform
        result_matrix = torch.bmm(m_composed, points_homogeneous.unsqueeze(-1)).squeeze(
            -1
        )[:, :3]

        # Apply transforms sequentially through the public CUDA-backed SE3 API
        intermediate = se3pose_transform_point(t2, r2, points)
        result_sequential = se3pose_transform_point(t1, r1, intermediate)

        self.assertTrue(torch.allclose(result_matrix, result_sequential, atol=ATOL))

    def test_se3_transform_point_backward(self):
        """Test that gradients flow correctly through SE3 point transformation"""
        N = NUM_POSES_SMALL

        # Random poses and points with gradients enabled
        translations = torch.randn(N, 3, device=device, requires_grad=True)
        rotations = torch.nn.functional.normalize(
            torch.randn(N, 4, device=device, requires_grad=True)
        )
        rotations.retain_grad()  # Need this for non-leaf tensors
        points = torch.randn(N, 3, device=device, requires_grad=True)

        # Forward pass
        output = se3pose_transform_point(translations, rotations, points)

        # Backward pass
        grad_output = torch.randn_like(output)
        output.backward(grad_output)

        # Check that gradients exist and are non-zero
        self.assertIsNotNone(translations.grad)
        self.assertIsNotNone(rotations.grad)
        self.assertIsNotNone(points.grad)
        self.assertTrue(torch.any(translations.grad != 0))
        self.assertTrue(torch.any(rotations.grad != 0))
        self.assertTrue(torch.any(points.grad != 0))

    def test_se3_to_matrix_backward(self):
        """Test that gradients flow correctly through SE3 to matrix conversion"""
        N = NUM_POSES_SMALL

        # Random poses with gradients enabled
        translations = torch.randn(N, 3, device=device, requires_grad=True)
        rotations = torch.nn.functional.normalize(
            torch.randn(N, 4, device=device, requires_grad=True)
        )
        rotations.retain_grad()  # Need this for non-leaf tensors

        # Forward pass
        matrices = se3pose_to_matrix(translations, rotations)

        # Backward pass
        grad_output = torch.randn_like(matrices)
        matrices.backward(grad_output)

        # Check that gradients exist and are non-zero
        self.assertIsNotNone(translations.grad)
        self.assertIsNotNone(rotations.grad)
        self.assertTrue(torch.any(translations.grad != 0))
        self.assertTrue(torch.any(rotations.grad != 0))


class TestTrajectory2Poses(unittest.TestCase):
    """Tests for 2-pose trajectory operations."""

    def setUp(self):
        """Reset RNG state before each test for deterministic random inputs."""
        torch.manual_seed(TEST_SEED)
        torch.cuda.manual_seed_all(TEST_SEED)

    def test_trajectory_transform_point_2poses_gradcheck_float32(self):
        """Trajectory CUDA bindings accept float32 only; gradcheck uses float32 + relaxed tol."""
        n = GC_TRAJ_BATCH
        torch.manual_seed(TEST_SEED + 1)
        time0 = torch.zeros(n, device=device, dtype=torch.float32)
        time1 = torch.ones(n, device=device, dtype=torch.float32)
        trans0 = torch.randn(
            n, 3, device=device, dtype=torch.float32, requires_grad=True
        )
        qr0 = torch.randn(n, 4, device=device, dtype=torch.float32, requires_grad=True)
        trans1 = torch.randn(
            n, 3, device=device, dtype=torch.float32, requires_grad=True
        )
        qr1 = torch.randn(n, 4, device=device, dtype=torch.float32, requires_grad=True)
        point = torch.randn(
            n, 3, device=device, dtype=torch.float32, requires_grad=True
        )
        query_time = torch.full(
            (n,), 0.5, device=device, dtype=torch.float32, requires_grad=True
        )

        def fn(t0, q0r, t1, q1r, p, qt):
            r0 = torch.nn.functional.normalize(q0r, dim=-1)
            r1 = torch.nn.functional.normalize(q1r, dim=-1)
            out, _ = TrajectoryTransformPoint2PosesFunction.apply(
                t0, r0, time0, t1, r1, time1, p, qt
            )
            return out

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertTrue(
                torch.autograd.gradcheck(
                    fn,
                    (trans0, qr0, trans1, qr1, point, query_time),
                    eps=GC_TRAJ_EPS,
                    atol=GC_TRAJ_ATOL,
                    rtol=GC_TRAJ_RTOL,
                    fast_mode=True,
                )
            )

    def test_trajectory_get_rotation_2poses_gradcheck_float32(self):
        n = GC_TRAJ_BATCH
        torch.manual_seed(TEST_SEED + 2)
        time0 = torch.zeros(n, device=device, dtype=torch.float32)
        time1 = torch.ones(n, device=device, dtype=torch.float32)
        trans0 = torch.randn(
            n, 3, device=device, dtype=torch.float32, requires_grad=True
        )
        qr0 = torch.randn(n, 4, device=device, dtype=torch.float32, requires_grad=True)
        trans1 = torch.randn(
            n, 3, device=device, dtype=torch.float32, requires_grad=True
        )
        qr1 = torch.randn(n, 4, device=device, dtype=torch.float32, requires_grad=True)
        query_time = torch.full(
            (n,), 0.5, device=device, dtype=torch.float32, requires_grad=True
        )

        def fn(t0, q0r, t1, q1r, qt):
            r0 = torch.nn.functional.normalize(q0r, dim=-1)
            r1 = torch.nn.functional.normalize(q1r, dim=-1)
            quat, _ = TrajectoryGetRotation2PosesFunction.apply(
                t0, r0, time0, t1, r1, time1, qt
            )
            return quat

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertTrue(
                torch.autograd.gradcheck(
                    fn,
                    (trans0, qr0, trans1, qr1, query_time),
                    eps=GC_TRAJ_EPS,
                    atol=GC_TRAJ_ATOL,
                    rtol=GC_TRAJ_RTOL,
                    fast_mode=True,
                )
            )

    def test_trajectory_transform_point_interpolation(self):
        """Test that interpolation at t=0.5 produces correct midpoint"""
        N = NUM_POSES

        # Create two poses with identity rotation and different translations
        t0 = torch.zeros(N, 3, device=device)
        t1 = torch.full((N, 3), 2.0, device=device)
        r = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(
            N, 1
        )  # Identity rotation

        time0 = torch.zeros(N, device=device)
        time1 = torch.ones(N, device=device)
        query_time = torch.full((N,), 0.5, device=device)

        # Point at origin
        point = torch.zeros(N, 3, device=device)

        # Expected: at t=0.5, translation should be [1.0, 1.0, 1.0]
        expected_trans = torch.ones(N, 3, device=device)

        # Call kernel
        result = trajectory_transform_point_2poses(
            t0, r, time0, t1, r, time1, point, query_time
        )

        self.assertTrue(
            torch.allclose(result["point"], expected_trans, atol=ATOL_STRICT)
        )
        self.assertFalse(torch.any(result["out_of_bounds"]))

    def test_trajectory_transform_origin_matches_translation_lerp(self):
        """Interpolating the origin should recover the interpolated translation."""
        N = NUM_POSES_SMALL

        trans0 = torch.randn(N, 3, device=device)
        trans1 = torch.randn(N, 3, device=device)
        rot0 = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        rot1 = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        time0 = torch.zeros(N, device=device)
        time1 = torch.ones(N, device=device)
        query_time = torch.rand(N, device=device)
        origin = torch.zeros(N, 3, device=device)

        result = trajectory_transform_point_2poses(
            trans0, rot0, time0, trans1, rot1, time1, origin, query_time
        )

        alpha = query_time.unsqueeze(-1)
        expected_translation = (1.0 - alpha) * trans0 + alpha * trans1

        self.assertTrue(
            torch.allclose(result["point"], expected_translation, atol=ATOL_STRICT)
        )
        self.assertFalse(torch.any(result["out_of_bounds"]))

    def test_trajectory_transform_point_extrapolation(self):
        """Test that extrapolation beyond bounds works and sets out_of_bounds flag"""
        N = NUM_POSES

        # Create two poses with linear motion
        t0 = torch.zeros(N, 3, device=device)
        t1 = torch.ones(N, 3, device=device)
        r = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(
            N, 1
        )  # Identity rotation

        time0 = torch.zeros(N, device=device)
        time1 = torch.ones(N, device=device)
        query_time = torch.full((N,), 2.0, device=device)  # Beyond bounds

        # Point at origin
        point = torch.zeros(N, 3, device=device)

        # Expected: at t=2.0, translation should be [2.0, 2.0, 2.0] (extrapolated)
        expected_trans = torch.full((N, 3), 2.0, device=device)

        # Call kernel
        result = trajectory_transform_point_2poses(
            t0, r, time0, t1, r, time1, point, query_time
        )

        self.assertTrue(
            torch.allclose(result["point"], expected_trans, atol=ATOL_STRICT)
        )
        self.assertTrue(torch.all(result["out_of_bounds"]))

    def test_trajectory_get_rotation_interpolation(self):
        """Test rotation interpolation using SLERP"""
        N = NUM_POSES

        # Two rotations - identity and 90-degree around Z
        r0 = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(N, 1)
        # 90 degrees around Z: quat = [0, 0, sin(45°), cos(45°)] = [0, 0, 0.707, 0.707]
        r1 = torch.tensor([0.0, 0.0, 0.7071068, 0.7071068], device=device).repeat(N, 1)

        t0 = torch.zeros(N, 3, device=device)
        t1 = torch.zeros(N, 3, device=device)

        time0 = torch.zeros(N, device=device)
        time1 = torch.ones(N, device=device)
        query_time = torch.full((N,), 0.5, device=device)

        # Call kernel
        result = trajectory_get_rotation_2poses(
            t0, r0, time0, t1, r1, time1, query_time
        )

        # Expected: SLERP at t=0.5 between identity and 90° rotation
        expected = quat_slerp(r0, r1, 0.5)

        self.assertTrue(torch.allclose(result["quat"], expected, atol=ATOL_STRICT))
        self.assertFalse(torch.any(result["out_of_bounds"]))

    def test_trajectory_get_rotation_2poses_near_parallel_matches_reference(self):
        """Near-parallel rotations match the expected SLERP reference."""
        torch.manual_seed(91026)
        n = 256
        trans0 = torch.randn(n, 3, device=device, dtype=torch.float32)
        trans1 = torch.randn(n, 3, device=device, dtype=torch.float32)
        rot0 = torch.nn.functional.normalize(
            torch.randn(n, 4, device=device, dtype=torch.float32), dim=-1
        )
        rot1 = torch.nn.functional.normalize(
            rot0 + 1e-4 * torch.randn(n, 4, device=device, dtype=torch.float32), dim=-1
        )
        time0 = torch.zeros(n, device=device, dtype=torch.float32)
        time1 = torch.ones(n, device=device, dtype=torch.float32)
        query_time = torch.full((n,), 0.35, device=device, dtype=torch.float32)
        q_c, oob_c = TrajectoryGetRotation2PosesFunction.apply(
            trans0, rot0, time0, trans1, rot1, time1, query_time
        )
        q_ref = quat_slerp(rot0, rot1, query_time.unsqueeze(-1))
        same = torch.allclose(q_ref, q_c, atol=1e-5, rtol=1e-4)
        same_neg = torch.allclose(q_ref, -q_c, atol=1e-5, rtol=1e-4)
        self.assertTrue(same or same_neg)
        self.assertTrue(torch.all(oob_c == 0))

    def test_trajectory_transform_point_2poses_tiny_positive_dt_matches_reference(self):
        """Tiny positive trajectory intervals still interpolate with positive dt semantics."""
        torch.manual_seed(91027)
        n = 128
        trans0 = torch.randn(n, 3, device=device, dtype=torch.float32)
        trans1 = torch.randn(n, 3, device=device, dtype=torch.float32)
        rot0 = torch.nn.functional.normalize(
            torch.randn(n, 4, device=device, dtype=torch.float32), dim=-1
        )
        rot1 = torch.nn.functional.normalize(
            torch.randn(n, 4, device=device, dtype=torch.float32), dim=-1
        )
        point = torch.randn(n, 3, device=device, dtype=torch.float32)
        time0 = torch.zeros(n, device=device, dtype=torch.float32)
        time1 = torch.full((n,), 5e-13, device=device, dtype=torch.float32)
        query_time = torch.full((n,), 2.5e-13, device=device, dtype=torch.float32)
        pt_c, oob_c = TrajectoryTransformPoint2PosesFunction.apply(
            trans0, rot0, time0, trans1, rot1, time1, point, query_time
        )
        alpha = ((query_time - time0) / (time1 - time0)).unsqueeze(-1)
        trans_ref = (1.0 - alpha) * trans0 + alpha * trans1
        rot_ref = quat_slerp(rot0, rot1, alpha)
        point_ref = se3pose_transform_point(trans_ref, rot_ref, point)
        self.assertTrue(torch.allclose(pt_c, point_ref, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.all(oob_c == 0))

    def test_trajectory_transform_point_2poses_swapped_keyframes_match_forward_order(
        self,
    ):
        """Swapping both keyframes and their times should preserve the interpolated pose."""
        torch.manual_seed(91028)
        n = 128
        trans0 = torch.randn(n, 3, device=device, dtype=torch.float32)
        trans1 = torch.randn(n, 3, device=device, dtype=torch.float32)
        rot0 = torch.nn.functional.normalize(
            torch.randn(n, 4, device=device, dtype=torch.float32), dim=-1
        )
        rot1 = torch.nn.functional.normalize(
            torch.randn(n, 4, device=device, dtype=torch.float32), dim=-1
        )
        point = torch.randn(n, 3, device=device, dtype=torch.float32)
        time0 = torch.zeros(n, device=device, dtype=torch.float32)
        time1 = torch.ones(n, device=device, dtype=torch.float32)
        query_time = torch.full((n,), 0.25, device=device, dtype=torch.float32)

        forward = trajectory_transform_point_2poses(
            trans0, rot0, time0, trans1, rot1, time1, point, query_time
        )
        swapped_keyframes = trajectory_transform_point_2poses(
            trans1, rot1, time1, trans0, rot0, time0, point, query_time
        )

        self.assertTrue(
            torch.allclose(
                forward["point"], swapped_keyframes["point"], atol=1e-5, rtol=1e-4
            )
        )
        self.assertTrue(
            torch.equal(forward["out_of_bounds"], swapped_keyframes["out_of_bounds"])
        )

    def test_trajectory_transform_point_2poses_equal_times_use_pose0_and_zero_time_grads(
        self,
    ):
        """Equal keyframe times avoid division by zero by selecting pose 0 with zero time grads."""
        torch.manual_seed(91029)
        n = 64
        trans0 = torch.randn(
            n, 3, device=device, dtype=torch.float32, requires_grad=True
        )
        rot0 = torch.nn.functional.normalize(
            torch.randn(n, 4, device=device, dtype=torch.float32, requires_grad=True),
            dim=-1,
        )
        rot0.retain_grad()
        trans1 = torch.randn(
            n, 3, device=device, dtype=torch.float32, requires_grad=True
        )
        rot1 = torch.nn.functional.normalize(
            torch.randn(n, 4, device=device, dtype=torch.float32, requires_grad=True),
            dim=-1,
        )
        rot1.retain_grad()
        point = torch.randn(
            n, 3, device=device, dtype=torch.float32, requires_grad=True
        )
        time0 = torch.zeros(n, device=device, dtype=torch.float32, requires_grad=True)
        time1 = torch.zeros(n, device=device, dtype=torch.float32, requires_grad=True)
        query_time = torch.zeros(
            n, device=device, dtype=torch.float32, requires_grad=True
        )

        result = trajectory_transform_point_2poses(
            trans0, rot0, time0, trans1, rot1, time1, point, query_time
        )
        expected = se3pose_transform_point(trans0, rot0, point)

        self.assertTrue(torch.allclose(result["point"], expected, atol=1e-5, rtol=1e-4))
        self.assertFalse(torch.any(result["out_of_bounds"]))

        shifted_query = torch.full((n,), 0.5, device=device, dtype=torch.float32)
        shifted = trajectory_transform_point_2poses(
            trans0.detach(),
            rot0.detach(),
            torch.zeros(n, device=device, dtype=torch.float32),
            trans1.detach(),
            rot1.detach(),
            torch.zeros(n, device=device, dtype=torch.float32),
            point.detach(),
            shifted_query,
        )
        self.assertTrue(torch.all(shifted["out_of_bounds"]))

        grad_output = torch.randn_like(result["point"])
        result["point"].backward(grad_output)
        self.assertTrue(torch.all(time0.grad == 0))
        self.assertTrue(torch.all(time1.grad == 0))
        self.assertTrue(torch.all(query_time.grad == 0))

    def test_trajectory_transform_point_backward(self):
        """Test that gradients flow correctly through trajectory transformation"""
        N = NUM_POSES_SMALL

        # Random trajectory with gradients enabled
        # Use simple, independent time values to avoid gradient cancellation
        t0 = torch.randn(N, 3, device=device, requires_grad=True)
        r0 = torch.nn.functional.normalize(
            torch.randn(N, 4, device=device, requires_grad=True)
        )
        r0.retain_grad()  # Need this for non-leaf tensors
        time0 = torch.zeros(N, device=device)
        t1 = torch.randn(N, 3, device=device, requires_grad=True)
        r1 = torch.nn.functional.normalize(
            torch.randn(N, 4, device=device, requires_grad=True)
        )
        r1.retain_grad()  # Need this for non-leaf tensors
        time1 = torch.ones(N, device=device)
        points = torch.randn(N, 3, device=device, requires_grad=True)
        query_time = torch.full((N,), 0.5, device=device)  # Query at midpoint

        # Forward pass
        result = trajectory_transform_point_2poses(
            t0, r0, time0, t1, r1, time1, points, query_time
        )

        # Backward pass
        grad_output = torch.randn_like(result["point"])
        result["point"].backward(grad_output)

        # Check that gradients exist and are non-zero for the main parameters
        # (translations, rotations, points - these should always have meaningful gradients)
        self.assertIsNotNone(t0.grad)
        self.assertIsNotNone(r0.grad)
        self.assertIsNotNone(t1.grad)
        self.assertIsNotNone(r1.grad)
        self.assertIsNotNone(points.grad)
        self.assertTrue(torch.any(t0.grad != 0))
        self.assertTrue(torch.any(r0.grad != 0))
        self.assertTrue(torch.any(t1.grad != 0))
        self.assertTrue(torch.any(r1.grad != 0))
        self.assertTrue(torch.any(points.grad != 0))


class TestTrajectory1Pose(unittest.TestCase):
    """Tests for 1-pose trajectory operations."""

    def setUp(self):
        """Reset RNG state before each test for deterministic random inputs."""
        torch.manual_seed(TEST_SEED)
        torch.cuda.manual_seed_all(TEST_SEED)

    def test_trajectory_transform_point_1pose_gradcheck_float32(self):
        n = GC_TRAJ_BATCH
        torch.manual_seed(TEST_SEED + 3)
        time = torch.zeros(n, device=device, dtype=torch.float32)
        trans = torch.randn(
            n, 3, device=device, dtype=torch.float32, requires_grad=True
        )
        qr = torch.randn(n, 4, device=device, dtype=torch.float32, requires_grad=True)
        point = torch.randn(
            n, 3, device=device, dtype=torch.float32, requires_grad=True
        )
        query_time = torch.zeros(
            n, device=device, dtype=torch.float32, requires_grad=True
        )

        def fn(tt, qqr, p, qt):
            r = torch.nn.functional.normalize(qqr, dim=-1)
            out, _ = TrajectoryTransformPoint1PoseFunction.apply(tt, r, time, p, qt)
            return out

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertTrue(
                torch.autograd.gradcheck(
                    fn,
                    (trans, qr, point, query_time),
                    eps=GC_TRAJ_EPS,
                    atol=GC_TRAJ_ATOL,
                    rtol=GC_TRAJ_RTOL,
                    fast_mode=True,
                )
            )

    def test_trajectory_transform_point_1pose(self):
        """Test that 1-pose trajectory returns the pose transformation"""
        N = NUM_POSES

        # Single pose
        trans = torch.randn(N, 3, device=device)
        rot = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        time = torch.zeros(N, device=device)
        point = torch.randn(N, 3, device=device)
        query_time = torch.zeros(N, device=device)  # Query at same time

        # Call kernel
        result = trajectory_transform_point_1pose(trans, rot, time, point, query_time)

        # Expected: same as SE3 transformation
        expected = se3pose_transform_point(trans, rot, point)

        self.assertTrue(torch.allclose(result["point"], expected, atol=ATOL_STRICT))
        self.assertFalse(torch.any(result["out_of_bounds"]))

    def test_trajectory_transform_point_1pose_out_of_bounds(self):
        """Test that 1-pose trajectory sets out_of_bounds flag when query_time != time"""
        N = NUM_POSES

        # Single pose
        trans = torch.randn(N, 3, device=device)
        rot = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        time = torch.zeros(N, device=device)
        point = torch.randn(N, 3, device=device)
        query_time = torch.ones(N, device=device)  # Query at different time

        # Call kernel
        result = trajectory_transform_point_1pose(trans, rot, time, point, query_time)

        # Expected: same transformation but out_of_bounds should be true
        expected = se3pose_transform_point(trans, rot, point)

        self.assertTrue(torch.allclose(result["point"], expected, atol=ATOL_STRICT))
        self.assertTrue(torch.all(result["out_of_bounds"]))

    def test_trajectory_transform_point_1pose_backward(self):
        """Test that gradients flow correctly through 1-pose trajectory"""
        N = NUM_POSES_SMALL

        # Random trajectory with gradients enabled
        trans = torch.randn(N, 3, device=device, requires_grad=True)
        rot = torch.nn.functional.normalize(
            torch.randn(N, 4, device=device, requires_grad=True)
        )
        rot.retain_grad()  # Need this for non-leaf tensors
        time = torch.zeros(N, device=device)
        points = torch.randn(N, 3, device=device, requires_grad=True)
        query_time = torch.zeros(N, device=device)  # Query at same time as the pose

        # Forward pass
        result = trajectory_transform_point_1pose(trans, rot, time, points, query_time)

        # Backward pass
        grad_output = torch.randn_like(result["point"])
        result["point"].backward(grad_output)

        # Check that gradients exist and are non-zero for the main parameters
        # (translation, rotation, points - these should always have meaningful gradients)
        self.assertIsNotNone(trans.grad)
        self.assertIsNotNone(rot.grad)
        self.assertIsNotNone(points.grad)
        self.assertTrue(torch.any(trans.grad != 0))
        self.assertTrue(torch.any(rot.grad != 0))
        self.assertTrue(torch.any(points.grad != 0))


class TestFrameTransformPosesTQuat(unittest.TestCase):
    """Tests for frame transform poses tquat."""

    def setUp(self):
        """Reset RNG state before each test for deterministic random inputs."""
        torch.manual_seed(TEST_SEED)
        torch.cuda.manual_seed_all(TEST_SEED)

    def test_frame_transform_poses_tquat(self):
        """Test frame transform poses tquat with random transforms."""
        N = NUM_POSES_SMALL

        # Random input poses (normalize quaternion part)
        tquat_poses = torch.randn(N, 7, device=device)
        tquat_poses[..., 3:] = torch.nn.functional.normalize(
            tquat_poses[..., 3:], dim=-1
        )

        # Random frame transform (normalize quaternion)
        frame_quat = torch.randn(4)
        frame_quat = torch.nn.functional.normalize(frame_quat, dim=-1)
        rotation = tuple(frame_quat.tolist())  # (qx, qy, qz, qw)

        frame_trans = torch.randn(3)
        translation = tuple(frame_trans.tolist())

        scale = torch.rand(1).item() * 2.0 + 0.5  # random scale in [0.5, 2.5]

        # GPU kernel result
        result = frame_transform_poses_tquat(tquat_poses, rotation, translation, scale)

        frame_quat_tensor = torch.tensor(rotation, device=device).unsqueeze(0)  # (1, 4)
        frame_trans_tensor = torch.tensor(translation, device=device).unsqueeze(
            0
        )  # (1, 3)

        pose_t = tquat_poses[..., :3]  # (N, 3)
        pose_R = tquat_poses[..., 3:]  # (N, 4)

        expected_R = quat_multiply(frame_quat_tensor.expand_as(pose_R), pose_R)
        rotated_t = quat_rotate_vector(frame_quat_tensor.expand_as(pose_R), pose_t)
        expected_t = scale * (rotated_t + frame_trans_tensor)
        expected = torch.cat([expected_t, expected_R], dim=-1)

        torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


class TestPosePublicValidation(unittest.TestCase):
    """Python-level validation errors for kernel-backed pose APIs."""

    def test_se3_transform_point_cpu_raises(self):
        t = torch.randn(2, 3)
        r = torch.randn(2, 4)
        p = torch.randn(2, 3)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            se3pose_transform_point(t, r, p)

    def test_se3_transform_point_batch_mismatch_raises(self):
        t = torch.randn(2, 3, device=device, dtype=torch.float64)
        r = torch.randn(3, 4, device=device, dtype=torch.float64)
        p = torch.randn(2, 3, device=device, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "batch"):
            se3pose_transform_point(t, r, p)

    def test_se3_from_matrix_non_tensor_raises(self):
        with self.assertRaisesRegex(TypeError, "must be a torch.Tensor"):
            se3pose_from_matrix([[1.0] * 16, [1.0] * 16])  # type: ignore[arg-type]

    def test_se3_from_matrix_rejects_n_by_4_shape(self):
        matrix = torch.randn(3, 4, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(
            ValueError, "matrix must have shape \\(N, 4, 4\\) or \\(N, 16\\)"
        ):
            se3pose_from_matrix(matrix)

    def test_trajectory_time_float64_raises(self):
        n = 4
        trans0 = torch.randn(n, 3, device=device, dtype=torch.float32)
        rot0 = torch.randn(n, 4, device=device, dtype=torch.float32)
        trans1 = torch.randn(n, 3, device=device, dtype=torch.float32)
        rot1 = torch.randn(n, 4, device=device, dtype=torch.float32)
        point = torch.randn(n, 3, device=device, dtype=torch.float32)
        t0 = torch.randn(n, device=device, dtype=torch.float64)
        t1 = torch.randn(n, device=device, dtype=torch.float32)
        qt = torch.randn(n, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(TypeError, "float32"):
            trajectory_transform_point_2poses(
                trans0, rot0, t0, trans1, rot1, t1, point, qt
            )

    def test_frame_transform_not_float32_raises(self):
        poses = torch.randn(2, 7, device=device, dtype=torch.float64)
        with self.assertRaisesRegex(TypeError, "float32"):
            frame_transform_poses_tquat(
                poses, (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 1.0
            )


if __name__ == "__main__":
    unittest.main()
