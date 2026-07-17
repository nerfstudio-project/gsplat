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

"""Tests for the fused CUDA ground-gaussian distortion loss.

Compares the CUDA kernel against the pure-PyTorch fallback for the forward
value and the backward gradients w.r.t. positions and rotations.
"""

import pytest
import torch

from tests._cuda import cuda_is_available

from gsplat import has_losses
from gsplat.losses_fused import FusedGroundGaussiansLosses, ground_gaussians_loss

CUDA_AVAILABLE = cuda_is_available() and has_losses()

N = 2048
N_BINS = 16

# Bin parameters chosen so several bins capture many points: positions span
# roughly z in [-3, 27] (camera frame), bins live in [min_bias, min_bias +
# range_bias + grid_len].
MIN_BIAS = 0.0
RANGE_BIAS = 20.0
GRID_LEN = 3.0
ROTATION_LAMBDA = 0.1


def _normalize(q):
    return q / q.norm(dim=-1, keepdim=True)


def _make_inputs(device, n=N, n_bins=N_BINS, requires_grad=False, dtype=torch.float32):
    """Create random world-frame gaussians, a camera pose, and bin samples."""
    torch.manual_seed(7)
    # Spread points along a ground-like volume so bins are populated.
    positions = torch.empty(n, 3, device=device, dtype=dtype)
    positions[:, 0] = (torch.rand(n, device=device, dtype=dtype) - 0.5) * 8.0  # lateral
    positions[:, 1] = (torch.rand(n, device=device, dtype=dtype) - 0.5) * 1.0  # height
    positions[:, 2] = torch.rand(n, device=device, dtype=dtype) * 24.0  # depth
    rotations = _normalize(torch.randn(n, 4, device=device, dtype=dtype))

    # Camera-from-world pose: small translation + a modest rotation.
    cam_tquat = torch.tensor(
        [0.2, -0.1, 0.5, 0.05, 0.02, -0.03, 0.998],
        device=device,
        dtype=dtype,
    )
    cam_tquat = torch.cat([cam_tquat[:3], _normalize(cam_tquat[3:])])

    random_values = torch.rand(n_bins, device=device, dtype=dtype)

    if requires_grad:
        positions.requires_grad_(True)
        rotations.requires_grad_(True)
    return positions, rotations, cam_tquat, random_values


def _module(device):
    return FusedGroundGaussiansLosses(
        min_bias=MIN_BIAS,
        range_bias=RANGE_BIAS,
        grid_len=GRID_LEN,
        rotation_lambda=ROTATION_LAMBDA,
    )


def _assert_cuda_matches_reference(
    positions,
    rotations,
    cam_tquat,
    random_values,
    min_bias,
    range_bias,
    grid_len,
):
    p_ref = positions.detach().clone().requires_grad_(True)
    r_ref = rotations.detach().clone().requires_grad_(True)
    ref = ground_gaussians_loss(
        p_ref,
        r_ref,
        cam_tquat,
        random_values,
        min_bias,
        range_bias,
        grid_len,
        ROTATION_LAMBDA,
    )
    if ref.requires_grad:
        ref.backward()

    p_cuda = positions.detach().clone().requires_grad_(True)
    r_cuda = rotations.detach().clone().requires_grad_(True)
    module = FusedGroundGaussiansLosses(
        min_bias=min_bias,
        range_bias=range_bias,
        grid_len=grid_len,
        rotation_lambda=ROTATION_LAMBDA,
    )
    loss = module(p_cuda, r_cuda, cam_tquat, random_values)
    loss.backward()

    assert torch.allclose(loss, ref, rtol=1e-5, atol=1e-5)
    if ref.requires_grad:
        assert torch.allclose(p_cuda.grad, p_ref.grad, rtol=1e-4, atol=1e-4)
        assert torch.allclose(r_cuda.grad, r_ref.grad, rtol=1e-4, atol=1e-4)
    else:
        assert ref.item() == 0.0
        assert torch.count_nonzero(p_cuda.grad) == 0
        assert torch.count_nonzero(r_cuda.grad) == 0
    return loss


# ---------------------------------------------------------------------------
# CPU fallback tests — always runnable
# ---------------------------------------------------------------------------


class TestFusedGroundGaussiansLossesFallback:
    """Verify the module works in pure-PyTorch (CPU) fallback mode."""

    def test_forward_matches_reference(self):
        device = torch.device("cpu")
        positions, rotations, cam_tquat, random_values = _make_inputs(device)
        module = _module(device)

        loss = module(positions, rotations, cam_tquat, random_values)
        ref = ground_gaussians_loss(
            positions,
            rotations,
            cam_tquat,
            random_values,
            MIN_BIAS,
            RANGE_BIAS,
            GRID_LEN,
            ROTATION_LAMBDA,
        )
        assert torch.allclose(loss, ref)

    def test_scalar_shape(self):
        device = torch.device("cpu")
        positions, rotations, cam_tquat, random_values = _make_inputs(device, n=128)
        module = _module(device)
        loss = module(positions, rotations, cam_tquat, random_values)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_loss_is_positive(self):
        """With random gaussians the distortion loss is strictly positive."""
        device = torch.device("cpu")
        positions, rotations, cam_tquat, random_values = _make_inputs(device)
        module = _module(device)
        loss = module(positions, rotations, cam_tquat, random_values)
        assert loss.item() > 0.0

    def test_backward_runs(self):
        device = torch.device("cpu")
        positions, rotations, cam_tquat, random_values = _make_inputs(
            device, requires_grad=True
        )
        module = _module(device)
        loss = module(positions, rotations, cam_tquat, random_values)
        loss.backward()
        assert positions.grad is not None
        assert rotations.grad is not None
        assert torch.isfinite(positions.grad).all()
        assert torch.isfinite(rotations.grad).all()

    def test_empty_random_values(self):
        """Empty ``random_values`` (B == 0) must return a finite differentiable
        zero (not ``0 / 0 = NaN``), matching the CUDA ``n_bins == 0`` path, with
        zero gradients to positions and rotations."""
        device = torch.device("cpu")
        positions, rotations, cam_tquat, _ = _make_inputs(device, requires_grad=True)
        random_values = torch.empty(0, device=device, dtype=torch.float32)
        module = _module(device)
        loss = module(positions, rotations, cam_tquat, random_values)
        assert loss.shape == ()
        assert torch.isfinite(loss) and loss.item() == 0.0
        loss.backward()
        assert positions.grad is not None and rotations.grad is not None
        assert torch.count_nonzero(positions.grad) == 0
        assert torch.count_nonzero(rotations.grad) == 0


# ---------------------------------------------------------------------------
# CUDA tests — only run when GPU + compiled extension available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA or fused losses not available")
class TestFusedGroundGaussiansLossesCUDA:
    """Compare the CUDA fused kernel against the pure-PyTorch fallback."""

    def test_forward_matches_reference(self):
        device = torch.device("cuda")
        positions, rotations, cam_tquat, random_values = _make_inputs(device)
        module = _module(device)

        loss = module(positions, rotations, cam_tquat, random_values)
        ref = ground_gaussians_loss(
            positions,
            rotations,
            cam_tquat,
            random_values,
            MIN_BIAS,
            RANGE_BIAS,
            GRID_LEN,
            ROTATION_LAMBDA,
        )
        assert torch.allclose(
            loss, ref, rtol=1e-5, atol=1e-5
        ), f"forward max diff: {(loss - ref).abs().max()}"

    def test_backward_matches_reference(self):
        device = torch.device("cuda")

        # --- pure-PyTorch (fallback) gradients ---
        p1, r1, t1, rv1 = _make_inputs(device, requires_grad=True)
        ref = ground_gaussians_loss(
            p1, r1, t1, rv1, MIN_BIAS, RANGE_BIAS, GRID_LEN, ROTATION_LAMBDA
        )
        ref.backward()

        # --- CUDA gradients ---
        p2, r2, t2, rv2 = _make_inputs(device, requires_grad=True)
        module = _module(device)
        loss = module(p2, r2, t2, rv2)
        loss.backward()

        assert torch.allclose(
            p2.grad, p1.grad, rtol=1e-4, atol=1e-4
        ), f"positions grad max diff: {(p2.grad - p1.grad).abs().max()}"
        assert torch.allclose(
            r2.grad, r1.grad, rtol=1e-4, atol=1e-4
        ), f"rotations grad max diff: {(r2.grad - r1.grad).abs().max()}"

    def test_scalar_shape(self):
        device = torch.device("cuda")
        n = 512
        positions, rotations, cam_tquat, random_values = _make_inputs(device, n=n)
        module = _module(device)
        loss = module(positions, rotations, cam_tquat, random_values)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_empty_bins(self):
        """Bins placed entirely outside the point depth range yield zero loss
        and zero gradients (no bin has > 1 member)."""
        device = torch.device("cuda")
        positions, rotations, cam_tquat, _ = _make_inputs(device, requires_grad=True)
        # Bins far beyond the deepest point.
        random_values = torch.rand(N_BINS, device=device, dtype=torch.float32)
        module = FusedGroundGaussiansLosses(
            min_bias=1000.0,
            range_bias=10.0,
            grid_len=1.0,
            rotation_lambda=ROTATION_LAMBDA,
        )
        loss = module(positions, rotations, cam_tquat, random_values)
        assert loss.item() == 0.0
        loss.backward()
        assert torch.count_nonzero(positions.grad) == 0
        assert torch.count_nonzero(rotations.grad) == 0

    def test_empty_random_values(self):
        """Empty ``random_values`` (B == 0) on CUDA returns a finite zero with
        zero gradients — the native fwd/bwd launchers early-return for
        ``n_bins == 0`` — matching the pure-PyTorch fallback (both avoid 0/0)."""
        device = torch.device("cuda")
        positions, rotations, cam_tquat, _ = _make_inputs(device, requires_grad=True)
        random_values = torch.empty(0, device=device, dtype=torch.float32)
        module = _module(device)
        loss = module(positions, rotations, cam_tquat, random_values)
        assert torch.isfinite(loss) and loss.item() == 0.0
        loss.backward()
        assert torch.count_nonzero(positions.grad) == 0
        assert torch.count_nonzero(rotations.grad) == 0

    def test_production_scale(self):
        """Large, non-multiple-of-256 N exercises the grid-boundary guard and
        atomic accumulation at realistic point counts."""
        device = torch.device("cuda")
        n = 500_003
        positions, rotations, cam_tquat, random_values = _make_inputs(
            device, n=n, requires_grad=True
        )
        module = _module(device)
        loss = module(positions, rotations, cam_tquat, random_values)
        loss.backward()
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert positions.grad is not None
        assert rotations.grad is not None

    @pytest.mark.parametrize("n", [1, 31, 32, 33, 255, 257])
    def test_warp_and_block_boundaries_match_reference(self, n):
        """Tail lanes around warp/block boundaries preserve reference parity."""
        device = torch.device("cuda")
        positions, rotations, cam_tquat, _ = _make_inputs(device, n=n)
        random_values = torch.zeros(1, device=device, dtype=torch.float32)
        _assert_cuda_matches_reference(
            positions,
            rotations,
            cam_tquat,
            random_values,
            min_bias=-100.0,
            range_bias=0.0,
            grid_len=200.0,
        )

    def test_float64_partial_block_matches_reference(self):
        """The retained float64 path preserves forward and gradient parity."""
        device = torch.device("cuda")
        positions, rotations, cam_tquat, _ = _make_inputs(
            device, n=257, dtype=torch.float64
        )
        random_values = torch.zeros(1, device=device, dtype=torch.float64)
        _assert_cuda_matches_reference(
            positions,
            rotations,
            cam_tquat,
            random_values,
            min_bias=-100.0,
            range_bias=0.0,
            grid_len=200.0,
        )

    def test_partial_warp_non_leader_hits_match_reference(self):
        """A tail warp is reduced when only non-leader lanes hit the bin."""
        device = torch.device("cuda")
        n = 35
        positions = torch.zeros(n, 3, device=device, dtype=torch.float32)
        positions[:, 2] = 2.0
        positions[-2:, 1] = torch.tensor([0.0, 1.0], device=device)
        positions[-2:, 2] = 0.5
        rotations = torch.zeros(n, 4, device=device, dtype=torch.float32)
        rotations[:, 3] = 1.0
        cam_tquat = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            device=device,
            dtype=torch.float32,
        )
        random_values = torch.zeros(1, device=device, dtype=torch.float32)

        loss = _assert_cuda_matches_reference(
            positions,
            rotations,
            cam_tquat,
            random_values,
            min_bias=0.0,
            range_bias=0.0,
            grid_len=1.0,
        )
        assert loss.item() > 0.0

    def test_disable_cuda_uses_fallback(self):
        """When the module's CUDA flag is off, CUDA inputs route through the
        pure-PyTorch fallback and match it exactly."""
        device = torch.device("cuda")
        positions, rotations, cam_tquat, random_values = _make_inputs(device)

        module = _module(device)
        module._cuda_available = False  # force fallback
        loss = module(positions, rotations, cam_tquat, random_values)
        ref = ground_gaussians_loss(
            positions,
            rotations,
            cam_tquat,
            random_values,
            MIN_BIAS,
            RANGE_BIAS,
            GRID_LEN,
            ROTATION_LAMBDA,
        )
        assert torch.allclose(loss, ref)
