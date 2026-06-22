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

"""Tests for fused CUDA gaussian regularization losses.

Compares Tier 2 (CUDA) against Tier 1 (pure-PyTorch) implementations for
forward values and backward gradients.
"""

import pytest
import torch

from gsplat import has_losses
from gsplat.losses import (
    gaussian_density_reg,
    gaussian_scale_reg,
    gaussian_z_scale_reg,
    out_of_bound_loss,
)
from gsplat.losses_fused import FusedGaussianLosses

CUDA_AVAILABLE = torch.cuda.is_available() and has_losses()
N = 1024


def _make_inputs(device, n=N, requires_grad=False):
    """Create random gaussian reg inputs on *device*."""
    torch.manual_seed(42)
    scales = torch.rand(n, 3, device=device, dtype=torch.float32) * 2.0
    densities = torch.rand(n, device=device, dtype=torch.float32)
    z_scales = torch.rand(n, device=device, dtype=torch.float32) * 1.5
    positions = (torch.rand(n, 3, device=device, dtype=torch.float32) - 0.5) * 6.0
    cuboid_dims = torch.rand(n, 3, device=device, dtype=torch.float32) * 4.0 + 0.5
    if requires_grad:
        scales.requires_grad_(True)
        densities.requires_grad_(True)
        z_scales.requires_grad_(True)
        positions.requires_grad_(True)
    return scales, densities, z_scales, positions, cuboid_dims


# ---------------------------------------------------------------------------
# Tier 1 (CPU) fallback tests — always runnable
# ---------------------------------------------------------------------------


class TestFusedGaussianLossesFallback:
    """Verify the FusedGaussianLosses module works in fallback (CPU) mode."""

    def test_forward_matches_tier1(self):
        device = torch.device("cpu")
        scales, densities, z_scales, positions, cuboid_dims = _make_inputs(device)
        threshold = 0.5

        module = FusedGaussianLosses(z_scale_threshold=threshold)
        ls, ld, lz, lo = module(scales, densities, z_scales, positions, cuboid_dims)

        assert torch.allclose(ls, gaussian_scale_reg(scales))
        assert torch.allclose(ld, gaussian_density_reg(densities))
        assert torch.allclose(lz, gaussian_z_scale_reg(z_scales, threshold))
        assert torch.allclose(lo, out_of_bound_loss(positions, cuboid_dims))

    def test_with_visibility(self):
        device = torch.device("cpu")
        scales, densities, z_scales, positions, cuboid_dims = _make_inputs(device)
        visibility = torch.randint(0, 2, (N,), dtype=torch.float32, device=device)
        threshold = 0.5

        module = FusedGaussianLosses(z_scale_threshold=threshold)
        ls, ld, lz, lo = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            visibility=visibility,
        )

        assert torch.allclose(ls, gaussian_scale_reg(scales, visibility=visibility))
        assert torch.allclose(
            ld, gaussian_density_reg(densities, visibility=visibility)
        )
        # Verify visibility does not affect z_scale and oob losses (Tier 1 helpers
        # don't accept a visibility arg; module should pass through unchanged).
        assert torch.allclose(lz, gaussian_z_scale_reg(z_scales, threshold))
        assert torch.allclose(lo, out_of_bound_loss(positions, cuboid_dims))

    def test_shapes(self):
        device = torch.device("cpu")
        scales, densities, z_scales, positions, cuboid_dims = _make_inputs(device, n=64)
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        ls, ld, lz, lo = module(scales, densities, z_scales, positions, cuboid_dims)

        assert ls.shape == (64, 3)
        assert ld.shape == (64,)
        assert lz.shape == (64,)
        assert lo.shape == (64, 3)

    def test_backward_runs(self):
        device = torch.device("cpu")
        scales, densities, z_scales, positions, cuboid_dims = _make_inputs(
            device, requires_grad=True
        )
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        ls, ld, lz, lo = module(scales, densities, z_scales, positions, cuboid_dims)
        total = ls.sum() + ld.sum() + lz.sum() + lo.sum()
        total.backward()

        assert scales.grad is not None
        assert densities.grad is not None
        assert z_scales.grad is not None
        assert positions.grad is not None


# ---------------------------------------------------------------------------
# Tier 2 (CUDA) tests — only run when GPU + compiled extension available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA or fused losses not available")
class TestFusedGaussianLossesCUDA:
    """Compare CUDA fused kernel against pure-PyTorch Tier 1."""

    def test_forward_matches_tier1(self):
        device = torch.device("cuda")
        scales, densities, z_scales, positions, cuboid_dims = _make_inputs(device)
        threshold = 0.5

        module = FusedGaussianLosses(z_scale_threshold=threshold)
        ls, ld, lz, lo = module(scales, densities, z_scales, positions, cuboid_dims)

        ref_ls = gaussian_scale_reg(scales)
        ref_ld = gaussian_density_reg(densities)
        ref_lz = gaussian_z_scale_reg(z_scales, threshold)
        ref_lo = out_of_bound_loss(positions, cuboid_dims)

        # Pure element-wise fp32 ops, no transcendentals, no fast-math — expect
        # bit-exact parity with the Tier 1 reference. If this ever fails, a
        # precision path regressed and we want to see it.
        assert torch.equal(
            ls, ref_ls
        ), f"scale_reg max diff: {(ls - ref_ls).abs().max()}"
        assert torch.equal(
            ld, ref_ld
        ), f"density_reg max diff: {(ld - ref_ld).abs().max()}"
        assert torch.equal(
            lz, ref_lz
        ), f"z_scale_reg max diff: {(lz - ref_lz).abs().max()}"
        assert torch.equal(lo, ref_lo), f"oob max diff: {(lo - ref_lo).abs().max()}"

    def test_forward_with_visibility(self):
        device = torch.device("cuda")
        scales, densities, z_scales, positions, cuboid_dims = _make_inputs(device)
        visibility = torch.randint(0, 2, (N,), dtype=torch.float32, device=device)
        threshold = 0.3

        module = FusedGaussianLosses(z_scale_threshold=threshold)
        ls, ld, lz, lo = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            visibility=visibility,
        )

        ref_ls = gaussian_scale_reg(scales, visibility=visibility)
        ref_ld = gaussian_density_reg(densities, visibility=visibility)

        assert torch.equal(ls, ref_ls)
        assert torch.equal(ld, ref_ld)
        # Verify visibility does not affect z_scale and oob losses (Tier 1 helpers
        # don't accept a visibility arg; module should pass through unchanged).
        assert torch.equal(lz, gaussian_z_scale_reg(z_scales, threshold))
        assert torch.equal(lo, out_of_bound_loss(positions, cuboid_dims))

    def test_backward_matches_tier1(self):
        device = torch.device("cuda")
        threshold = 0.5

        # --- Tier 1 (Python) gradients ---
        s1, d1, z1, p1, c1 = _make_inputs(device, requires_grad=True)
        ref_ls = gaussian_scale_reg(s1)
        ref_ld = gaussian_density_reg(d1)
        ref_lz = gaussian_z_scale_reg(z1, threshold)
        ref_lo = out_of_bound_loss(p1, c1)
        (ref_ls.sum() + ref_ld.sum() + ref_lz.sum() + ref_lo.sum()).backward()

        # --- Tier 2 (CUDA) gradients ---
        s2, d2, z2, p2, c2 = _make_inputs(device, requires_grad=True)
        module = FusedGaussianLosses(z_scale_threshold=threshold)
        ls, ld, lz, lo = module(s2, d2, z2, p2, c2)
        (ls.sum() + ld.sum() + lz.sum() + lo.sum()).backward()

        # Backward is pure element-wise fp32 (relu/sign masks times upstream
        # times visibility). Expect bit-exact parity vs Tier 1.
        assert torch.equal(
            s2.grad, s1.grad
        ), f"scales grad max diff: {(s2.grad - s1.grad).abs().max()}"
        assert torch.equal(
            d2.grad, d1.grad
        ), f"densities grad max diff: {(d2.grad - d1.grad).abs().max()}"
        assert torch.equal(
            z2.grad, z1.grad
        ), f"z_scales grad max diff: {(z2.grad - z1.grad).abs().max()}"
        assert torch.equal(
            p2.grad, p1.grad
        ), f"positions grad max diff: {(p2.grad - p1.grad).abs().max()}"

    def test_zero_threshold_z_scale(self):
        device = torch.device("cuda")
        scales, densities, z_scales, positions, cuboid_dims = _make_inputs(device)
        module = FusedGaussianLosses(z_scale_threshold=0.0)
        _, _, lz, _ = module(scales, densities, z_scales, positions, cuboid_dims)

        # With threshold=0, z_scale_reg = relu(z_scales - 0) = z_scales (all >= 0)
        assert torch.equal(lz, z_scales)

    def test_shapes(self):
        device = torch.device("cuda")
        n = 256
        scales, densities, z_scales, positions, cuboid_dims = _make_inputs(device, n=n)
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        ls, ld, lz, lo = module(scales, densities, z_scales, positions, cuboid_dims)

        assert ls.shape == (n, 3)
        assert ld.shape == (n,)
        assert lz.shape == (n,)
        assert lo.shape == (n, 3)

    def test_empty_input(self):
        device = torch.device("cuda")
        scales = torch.empty(0, 3, device=device)
        densities = torch.empty(0, device=device)
        z_scales = torch.empty(0, device=device)
        positions = torch.empty(0, 3, device=device)
        cuboid_dims = torch.empty(0, 3, device=device)

        module = FusedGaussianLosses(z_scale_threshold=0.5)
        ls, ld, lz, lo = module(scales, densities, z_scales, positions, cuboid_dims)

        assert ls.shape == (0, 3)
        assert ld.shape == (0,)
        assert lz.shape == (0,)
        assert lo.shape == (0, 3)

    def test_production_scale(self):
        """Non-multiple-of-256 N exercises the `idx >= N` grid-boundary guard
        in both kernels and catches register-spill regressions at realistic N.
        """
        device = torch.device("cuda")
        n = 1_000_001  # production scale, also not a multiple of the block size
        scales, densities, z_scales, positions, cuboid_dims = _make_inputs(
            device, n=n, requires_grad=True
        )
        threshold = 0.5

        module = FusedGaussianLosses(z_scale_threshold=threshold)
        ls, ld, lz, lo = module(scales, densities, z_scales, positions, cuboid_dims)
        (ls.sum() + ld.sum() + lz.sum() + lo.sum()).backward()

        assert ls.shape == (n, 3)
        assert ld.shape == (n,)
        assert lz.shape == (n,)
        assert lo.shape == (n, 3)
        assert scales.grad is not None
        assert densities.grad is not None
        assert z_scales.grad is not None
        assert positions.grad is not None

    def test_visibility_none_matches_ones(self):
        """`visibility=None` must be bit-identical to an all-ones tensor —
        both in the forward values and in every backward gradient."""
        device = torch.device("cuda")
        threshold = 0.5

        s1, d1, z1, p1, c1 = _make_inputs(device, requires_grad=True)
        module = FusedGaussianLosses(z_scale_threshold=threshold)
        ls_none, ld_none, lz_none, lo_none = module(s1, d1, z1, p1, c1)
        (ls_none.sum() + ld_none.sum() + lz_none.sum() + lo_none.sum()).backward()

        s2, d2, z2, p2, c2 = _make_inputs(device, requires_grad=True)
        ones = torch.ones(s2.shape[0], device=device, dtype=torch.float32)
        ls_ones, ld_ones, lz_ones, lo_ones = module(s2, d2, z2, p2, c2, visibility=ones)
        (ls_ones.sum() + ld_ones.sum() + lz_ones.sum() + lo_ones.sum()).backward()

        assert torch.equal(ls_none, ls_ones)
        assert torch.equal(ld_none, ld_ones)
        assert torch.equal(lz_none, lz_ones)
        assert torch.equal(lo_none, lo_ones)
        assert torch.equal(s1.grad, s2.grad)
        assert torch.equal(d1.grad, d2.grad)
        assert torch.equal(z1.grad, z2.grad)
        assert torch.equal(p1.grad, p2.grad)

    def test_visibility_shape_N1(self):
        """Tier 1 accepts visibility as `[N]` or `[N, 1]`; the wrapper
        reshapes before dispatch so the CUDA path matches."""
        device = torch.device("cuda")
        scales, densities, z_scales, positions, cuboid_dims = _make_inputs(device)
        visibility_flat = torch.randint(0, 2, (N,), dtype=torch.float32, device=device)
        visibility_col = visibility_flat.unsqueeze(1)  # [N, 1]

        module = FusedGaussianLosses(z_scale_threshold=0.5)
        ls_flat, ld_flat, _, _ = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            visibility=visibility_flat,
        )
        ls_col, ld_col, _, _ = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            visibility=visibility_col,
        )

        assert torch.equal(ls_flat, ls_col)
        assert torch.equal(ld_flat, ld_col)
