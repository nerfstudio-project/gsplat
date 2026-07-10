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

"""Tests for fused CUDA loss modules.

Compares CUDA kernels against pure-PyTorch fallback implementations, including
forward values and backward gradients.
"""

import pytest
import torch

from gsplat._helper import (
    assert_grad_reference_close,
    expect_grad_reference_close,
    expect_group,
)
from gsplat import LossFlag, has_losses
from gsplat.losses import (
    gaussian_density_reg,
    gaussian_scale_reg,
    gaussian_z_scale_reg,
    out_of_bound_loss,
)
from gsplat.losses_fused import (
    FusedCameraLosses,
    FusedGaussianLosses,
    FusedLidarLosses,
    _cuda_camera_losses_available,
    _cuda_lidar_losses_available,
    _cuda_losses_available,
)

CUDA_AVAILABLE = torch.cuda.is_available() and has_losses()
N = 1024

# Per-ray flag bits (plain int) for building/masking the int32 ``flags``
# tensors, derived from the public ``LossFlag`` vocabulary rather than importing
# private module names. ``None`` when the loss extension is not built (the fused
# camera/LiDAR modules require it; flag-using tests are gated accordingly).
if LossFlag is not None:
    _FLAG_RGB_LABEL = int(LossFlag.RGB_LABEL)
    _FLAG_SKY_SEMANTIC = int(LossFlag.SKY_SEMANTIC)
    _FLAG_DROPPED = int(LossFlag.DROPPED)
    _FLAG_INVALID = int(LossFlag.INVALID)
    _FLAG_DIFIXED = int(LossFlag.DIFIXED)
    _FLAG_SYNTHETIC = int(LossFlag.SYNTHETIC)
else:
    _FLAG_RGB_LABEL = _FLAG_SKY_SEMANTIC = _FLAG_DROPPED = None
    _FLAG_INVALID = _FLAG_DIFIXED = _FLAG_SYNTHETIC = None

# Test-local shorthand for the modules' return tuples and factor args. The
# library code uses the full names; these terse aliases are confined to tests:
#   ls, ld, lz, lo  -> loss_scale, loss_density, loss_z_scale, loss_oob   (gaussian)
#   rl, bl          -> rgb_loss, bg_loss                                  (camera)
#   dl, il, rl, bl  -> distance_loss, intensity_loss, raydrop_loss, bg_loss (lidar)
#   rf, bf          -> rgb_factor, bg_factor                              (camera)
#   df, if_, rf, bf -> distance_factor, intensity_factor, raydrop_factor, bg_factor (lidar)


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
# CPU fallback tests — always runnable
# ---------------------------------------------------------------------------


class TestFusedGaussianLossesFallback:
    """Verify the FusedGaussianLosses module works in fallback (CPU) mode."""

    def test_forward_matches_pytorch(self):
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
        # Verify visibility does not affect z_scale and oob losses (the pure-PyTorch
        # helpers don't accept a visibility arg; module should pass through unchanged).
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
# CUDA tests — only run when GPU + compiled extension available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA or fused losses not available")
class TestFusedGaussianLossesCUDA:
    """Compare the CUDA fused kernel against the pure-PyTorch reference."""

    def test_forward_matches_pytorch(self):
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
        # bit-exact parity with the pure-PyTorch reference. If this ever fails, a
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
        # Verify visibility does not affect z_scale and oob losses (the pure-PyTorch
        # helpers don't accept a visibility arg; module should pass through unchanged).
        assert torch.equal(lz, gaussian_z_scale_reg(z_scales, threshold))
        assert torch.equal(lo, out_of_bound_loss(positions, cuboid_dims))

    def test_backward_matches_pytorch(self):
        device = torch.device("cuda")
        threshold = 0.5

        # --- pure-PyTorch (Python) gradients ---
        s1, d1, z1, p1, c1 = _make_inputs(device, requires_grad=True)
        ref_ls = gaussian_scale_reg(s1)
        ref_ld = gaussian_density_reg(d1)
        ref_lz = gaussian_z_scale_reg(z1, threshold)
        ref_lo = out_of_bound_loss(p1, c1)
        (ref_ls.sum() + ref_ld.sum() + ref_lz.sum() + ref_lo.sum()).backward()

        # --- CUDA gradients ---
        s2, d2, z2, p2, c2 = _make_inputs(device, requires_grad=True)
        module = FusedGaussianLosses(z_scale_threshold=threshold)
        ls, ld, lz, lo = module(s2, d2, z2, p2, c2)
        (ls.sum() + ld.sum() + lz.sum() + lo.sum()).backward()

        # Backward is pure element-wise fp32 (relu/sign masks times upstream
        # times visibility). Keep bit-exact parity vs the pure-PyTorch
        # path, but use the shared helper so failures report vector
        # diagnostics too.
        for name, actual, expected in (
            ("scales", s2.grad, s1.grad),
            ("densities", d2.grad, d1.grad),
            ("z_scales", z2.grad, z1.grad),
            ("positions", p2.grad, p1.grad),
        ):
            assert_grad_reference_close(
                actual,
                expected,
                rtol=0.0,
                atol=0.0,
                max_rel_l2=0.0,
                max_rel_l1=0.0,
                min_cosine=1.0 - 1e-15,
                max_signed_bias=0.0,
                msg=f"{name} grad",
            )

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
        with expect_group("visibility=None vs all-ones gradients"):
            for name, actual, expected in (
                ("scales", s1.grad, s2.grad),
                ("densities", d1.grad, d2.grad),
                ("z_scales", z1.grad, z2.grad),
                ("positions", p1.grad, p2.grad),
            ):
                expect_grad_reference_close(
                    actual,
                    expected,
                    rtol=0.0,
                    atol=0.0,
                    max_rel_l2=0.0,
                    max_rel_l1=0.0,
                    min_cosine=1.0 - 1e-15,
                    max_signed_bias=0.0,
                    msg=f"{name} visibility gradient",
                )

    def test_visibility_shape_N1(self):
        """The pure-PyTorch path accepts visibility as `[N]` or `[N, 1]`; the
        wrapper reshapes before dispatch so the CUDA path matches."""
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

    def test_out_of_bound_sign_symmetry(self):
        """``out_of_bound`` uses ``|position|``, so its forward is symmetric in
        the sign of the position and its gradient flips sign with it. Positions
        are the only signed gaussian input (scales/densities/z_scales are
        post-activation, ``>= 0`` by contract), so this is where the abs()/sign()
        forward+backward paths are exercised with negative values."""
        device = torch.device("cuda")
        torch.manual_seed(13)
        n = 512
        # |position| >> cuboid half-extent so relu(|p| - cuboid/2) is active.
        base = torch.rand(n, 3, device=device) * 2.0 + 3.0  # [3, 5)
        cuboid = torch.ones(n, 3, device=device)  # half-extent 0.5
        scales = torch.rand(n, 3, device=device)
        densities = torch.rand(n, device=device)
        z_scales = torch.rand(n, device=device)

        def _run(sign):
            pos = (base * sign).clone().requires_grad_(True)
            ls, ld, lz, lo = FusedGaussianLosses()(
                scales, densities, z_scales, pos, cuboid
            )
            (ls.sum() + ld.sum() + lz.sum() + lo.sum()).backward()
            return lo.detach(), pos.grad.detach()

        lo_pos, grad_pos = _run(1.0)
        lo_neg, grad_neg = _run(-1.0)
        # Forward is |p|-symmetric; the gradient flips sign with the position.
        assert torch.allclose(lo_pos, lo_neg, atol=1e-5)
        assert torch.allclose(grad_pos, -grad_neg, atol=1e-5)


# The public ``LossFlag`` vocabulary is sourced from the compiled extension
# (csrc/LossFlags.h via m.attr) — single source of truth, no Python literals.
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA or fused losses not available")
def test_loss_flags_sourced_from_extension():
    """LossFlag must be read from the compiled extension (csrc/LossFlags.h via
    m.attr), not hardcoded — verifies the binding + read-through path."""
    import gsplat.csrc as _C

    assert LossFlag is not None
    assert int(LossFlag.RGB_LABEL) == _C.LOSS_FLAG_RGB_LABEL
    assert int(LossFlag.SKY_SEMANTIC) == _C.LOSS_FLAG_SKY_SEMANTIC
    assert int(LossFlag.DROPPED) == _C.LOSS_FLAG_DROPPED
    assert int(LossFlag.INVALID) == _C.LOSS_FLAG_INVALID
    assert int(LossFlag.DIFIXED) == _C.LOSS_FLAG_DIFIXED
    assert int(LossFlag.SYNTHETIC) == _C.LOSS_FLAG_SYNTHETIC


def test_requires_extension(monkeypatch):
    """The fused camera/LiDAR modules require the compiled loss extension (the
    per-ray flag vocabulary). Without it they must raise a clear error at
    construction, not defer to a None-masking crash in the pure-PyTorch path."""
    import gsplat.losses_fused as lf

    monkeypatch.setattr(lf, "_FLAG_RGB_LABEL", None)
    with pytest.raises(RuntimeError, match="requires the compiled gsplat loss"):
        lf.FusedCameraLosses()
    with pytest.raises(RuntimeError, match="requires the compiled gsplat loss"):
        lf.FusedLidarLosses()


CAMERA_CUDA = torch.cuda.is_available() and _cuda_camera_losses_available()
LIDAR_CUDA = torch.cuda.is_available() and _cuda_lidar_losses_available()


def _make_camera_inputs(device, n=512, requires_grad=False):
    torch.manual_seed(42)
    # Mix of flag patterns: valid+rgb, valid+sky, invalid, difixed
    flags = torch.zeros(n, dtype=torch.int32, device=device)
    flags[: n // 4] = _FLAG_RGB_LABEL  # valid with RGB label
    flags[n // 4 : n // 2] = _FLAG_RGB_LABEL | _FLAG_SKY_SEMANTIC
    flags[n // 2 : 3 * n // 4] = _FLAG_INVALID
    flags[3 * n // 4 :] = _FLAG_DIFIXED

    rgb_pred = torch.rand(n, 3, device=device, dtype=torch.float32)
    rgb_gt = torch.rand(n, 3, device=device, dtype=torch.float32)
    bg_pred = (
        torch.rand(n, device=device, dtype=torch.float32) * 1.4 - 0.2
    )  # some out of [0,1]
    if requires_grad:
        rgb_pred.requires_grad_(True)
        bg_pred.requires_grad_(True)
    return flags, rgb_pred, rgb_gt, bg_pred


def _make_lidar_inputs(device, n=512, requires_grad=False):
    torch.manual_seed(42)
    flags = torch.zeros(n, dtype=torch.int32, device=device)
    flags[: n // 4] = 0  # valid, not dropped
    flags[n // 4 : n // 2] = _FLAG_DROPPED  # valid, dropped
    flags[n // 2 : 3 * n // 4] = _FLAG_SKY_SEMANTIC  # valid, sky
    flags[3 * n // 4 :] = _FLAG_INVALID

    distance_pred = torch.rand(n, device=device, dtype=torch.float32) * 50
    distance_gt = torch.rand(n, device=device, dtype=torch.float32) * 50
    intensity_pred = torch.rand(n, device=device, dtype=torch.float32)
    intensity_gt = torch.rand(n, device=device, dtype=torch.float32)
    raydrop_pred = torch.rand(n, device=device, dtype=torch.float32)
    raydrop_gt = torch.rand(n, device=device, dtype=torch.float32)
    bg_pred = torch.rand(n, device=device, dtype=torch.float32) * 1.4 - 0.2
    if requires_grad:
        distance_pred.requires_grad_(True)
        intensity_pred.requires_grad_(True)
        raydrop_pred.requires_grad_(True)
        bg_pred.requires_grad_(True)
    return (
        flags,
        distance_pred,
        distance_gt,
        intensity_pred,
        intensity_gt,
        raydrop_pred,
        raydrop_gt,
        bg_pred,
    )


# ---------------------------------------------------------------------------
# Camera tests — CPU fallback
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not has_losses(), reason="fused loss extension (flag vocabulary) not built"
)
class TestFusedCameraLossesFallback:
    def test_forward_shapes(self):
        device = torch.device("cpu")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        module = FusedCameraLosses()
        rl, bl = module(flags, rgb_pred, rgb_gt, bg_pred)
        assert rl.shape == (512,)
        assert bl.shape == (512,)

    def test_invalid_masked_out(self):
        device = torch.device("cpu")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        module = FusedCameraLosses()
        rl, bl = module(flags, rgb_pred, rgb_gt, bg_pred)
        # Pixels with INVALID flag should have zero loss
        invalid_mask = (flags & _FLAG_INVALID) != 0
        assert (rl[invalid_mask] == 0).all()
        assert (bl[invalid_mask] == 0).all()

    def test_difixed_bg_masked_out(self):
        device = torch.device("cpu")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        module = FusedCameraLosses()
        _, bl = module(flags, rgb_pred, rgb_gt, bg_pred)
        # BG loss masks DIFIXED rays (the fixture's last quarter).
        difixed = (flags & _FLAG_DIFIXED) != 0
        assert (bl[difixed] == 0).all()

    def test_backward_runs(self):
        device = torch.device("cpu")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(
            device, requires_grad=True
        )
        module = FusedCameraLosses()
        rl, bl = module(flags, rgb_pred, rgb_gt, bg_pred)
        (rl.sum() + bl.sum()).backward()
        assert rgb_pred.grad is not None
        assert bg_pred.grad is not None

    def test_target_requires_grad_rejected(self):
        device = torch.device("cpu")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        rgb_gt.requires_grad_(True)
        module = FusedCameraLosses()
        with pytest.raises(ValueError, match="rgb_gt"):
            module(flags, rgb_pred, rgb_gt, bg_pred)


# ---------------------------------------------------------------------------
# Camera tests — CUDA
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CAMERA_CUDA, reason="CUDA or fused camera losses not available")
class TestFusedCameraLossesCUDA:
    def test_forward_matches_fallback(self):
        device = torch.device("cuda")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        rf, bf = 0.5, 0.3

        # CUDA
        module = FusedCameraLosses()
        rl_cuda, bl_cuda = module(flags, rgb_pred, rgb_gt, bg_pred, rf, bf)

        # Reference (CPU fallback logic on CUDA tensors)
        module2 = FusedCameraLosses()
        module2._cuda_available = False
        rl_ref, bl_ref = module2(flags, rgb_pred, rgb_gt, bg_pred, rf, bf)

        assert torch.allclose(
            rl_cuda, rl_ref, atol=1e-5
        ), f"rgb_loss max diff: {(rl_cuda - rl_ref).abs().max()}"
        assert torch.allclose(
            bl_cuda, bl_ref, atol=1e-5
        ), f"bg_loss max diff: {(bl_cuda - bl_ref).abs().max()}"

    def test_backward_matches_fallback(self):
        device = torch.device("cuda")
        rf, bf = 0.5, 0.3

        # Reference gradients
        f1, rp1, rg1, bp1 = _make_camera_inputs(device, requires_grad=True)
        module_ref = FusedCameraLosses()
        module_ref._cuda_available = False
        rl1, bl1 = module_ref(f1, rp1, rg1, bp1, rf, bf)
        (rl1.sum() + bl1.sum()).backward()

        # CUDA gradients
        f2, rp2, rg2, bp2 = _make_camera_inputs(device, requires_grad=True)
        module_cuda = FusedCameraLosses()
        rl2, bl2 = module_cuda(f2, rp2, rg2, bp2, rf, bf)
        (rl2.sum() + bl2.sum()).backward()

        assert torch.allclose(
            rp2.grad, rp1.grad, atol=1e-4
        ), f"rgb_pred grad max diff: {(rp2.grad - rp1.grad).abs().max()}"
        assert torch.allclose(
            bp2.grad, bp1.grad, atol=1e-4
        ), f"bg_pred grad max diff: {(bp2.grad - bp1.grad).abs().max()}"

    def test_disabled_factors(self):
        device = torch.device("cuda")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        module = FusedCameraLosses()
        rl, bl = module(
            flags, rgb_pred, rgb_gt, bg_pred, rgb_factor=-1.0, bg_factor=-1.0
        )
        # factor < 0 means disabled — kernel writes zero
        assert (rl == 0).all()
        assert (bl == 0).all()

    def test_synthetic_rays(self):
        """SYNTHETIC rays are excluded from the background loss. Verify the
        zeroing plus CUDA/pure-PyTorch parity on forward and backward — the
        camera fixtures otherwise never set SYNTHETIC."""
        device = torch.device("cuda")
        n = 256
        flags = torch.zeros(n, dtype=torch.int32, device=device)
        flags[: n // 2] = _FLAG_RGB_LABEL
        flags[n // 2 :] = _FLAG_RGB_LABEL | _FLAG_SYNTHETIC
        synthetic = (flags & _FLAG_SYNTHETIC) != 0

        def _inputs():
            torch.manual_seed(7)
            rgb_pred = torch.rand(n, 3, device=device, requires_grad=True)
            rgb_gt = torch.rand(n, 3, device=device)
            bg_pred = torch.rand(n, device=device, requires_grad=True)
            return rgb_pred, rgb_gt, bg_pred

        rp1, rg1, bp1 = _inputs()
        ref = FusedCameraLosses()
        ref._cuda_available = False
        rl_ref, bl_ref = ref(flags, rp1, rg1, bp1, 0.5, 0.7)
        (rl_ref.sum() + bl_ref.sum()).backward()

        rp2, rg2, bp2 = _inputs()
        rl_cuda, bl_cuda = FusedCameraLosses()(flags, rp2, rg2, bp2, 0.5, 0.7)
        (rl_cuda.sum() + bl_cuda.sum()).backward()

        # Synthetic rays contribute no background loss, on both paths.
        assert (bl_ref[synthetic] == 0).all()
        assert (bl_cuda[synthetic] == 0).all()
        assert torch.allclose(rl_cuda, rl_ref, atol=1e-5)
        assert torch.allclose(bl_cuda, bl_ref, atol=1e-5)
        assert torch.allclose(bp2.grad, bp1.grad, atol=1e-4)
        assert torch.allclose(rp2.grad, rp1.grad, atol=1e-4)

    def test_empty_input(self):
        """Empty tensors exercise the autograd/module wrappers (kernels early
        return for N=0)."""
        device = torch.device("cuda")
        flags = torch.empty(0, dtype=torch.int32, device=device)
        rgb_pred = torch.empty(0, 3, device=device)
        rgb_gt = torch.empty(0, 3, device=device)
        bg_pred = torch.empty(0, device=device)
        rl, bl = FusedCameraLosses()(flags, rgb_pred, rgb_gt, bg_pred)
        assert rl.shape == (0,)
        assert bl.shape == (0,)


# ---------------------------------------------------------------------------
# LiDAR tests — CPU fallback
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not has_losses(), reason="fused loss extension (flag vocabulary) not built"
)
class TestFusedLidarLossesFallback:
    def test_forward_shapes(self):
        device = torch.device("cpu")
        inputs = _make_lidar_inputs(device)
        module = FusedLidarLosses()
        dl, il, rl, bl = module(*inputs)
        assert dl.shape == (512,)
        assert il.shape == (512,)
        assert rl.shape == (512,)
        assert bl.shape == (512,)

    def test_invalid_masked_out(self):
        device = torch.device("cpu")
        inputs = _make_lidar_inputs(device)
        flags = inputs[0]
        module = FusedLidarLosses()
        dl, il, rl, bl = module(*inputs)
        invalid = (flags & _FLAG_INVALID) != 0
        assert (dl[invalid] == 0).all()
        assert (il[invalid] == 0).all()
        assert (rl[invalid] == 0).all()
        assert (bl[invalid] == 0).all()

    def test_dropped_masking(self):
        device = torch.device("cpu")
        inputs = _make_lidar_inputs(device)
        flags = inputs[0]
        module = FusedLidarLosses()
        dl, il, rl, bl = module(*inputs)
        # Dropped rays: distance/intensity/bg should be zero, raydrop should NOT be
        dropped = ((flags & _FLAG_DROPPED) != 0) & ((flags & _FLAG_INVALID) == 0)
        assert (dl[dropped] == 0).all()
        assert (il[dropped] == 0).all()
        assert (bl[dropped] == 0).all()
        # raydrop does NOT mask DROPPED: dropped-but-valid rays still
        # contribute raydrop loss (random pred != gt in the fixture).
        assert (rl[dropped] != 0).any()

    def test_target_requires_grad_rejected(self):
        device = torch.device("cpu")
        inputs = _make_lidar_inputs(device)
        inputs[2].requires_grad_(True)  # distance_gt
        module = FusedLidarLosses()
        with pytest.raises(ValueError, match="distance_gt"):
            module(*inputs)


# ---------------------------------------------------------------------------
# LiDAR tests — CUDA
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not LIDAR_CUDA, reason="CUDA or fused lidar losses not available")
class TestFusedLidarLossesCUDA:
    def test_forward_matches_fallback(self):
        device = torch.device("cuda")
        inputs = _make_lidar_inputs(device)
        df, if_, rf, bf = 0.5, 0.3, 0.2, 0.1

        module = FusedLidarLosses()
        dl_cuda, il_cuda, rl_cuda, bl_cuda = module(*inputs, df, if_, rf, bf)

        module_ref = FusedLidarLosses()
        module_ref._cuda_available = False
        dl_ref, il_ref, rl_ref, bl_ref = module_ref(*inputs, df, if_, rf, bf)

        assert torch.allclose(
            dl_cuda, dl_ref, atol=1e-5
        ), f"distance max diff: {(dl_cuda - dl_ref).abs().max()}"
        assert torch.allclose(
            il_cuda, il_ref, atol=1e-5
        ), f"intensity max diff: {(il_cuda - il_ref).abs().max()}"
        assert torch.allclose(
            rl_cuda, rl_ref, atol=1e-5
        ), f"raydrop max diff: {(rl_cuda - rl_ref).abs().max()}"
        assert torch.allclose(
            bl_cuda, bl_ref, atol=1e-5
        ), f"bg max diff: {(bl_cuda - bl_ref).abs().max()}"

    def test_backward_matches_fallback(self):
        device = torch.device("cuda")
        df, if_, rf, bf = 0.5, 0.3, 0.2, 0.1

        # Reference
        inputs1 = _make_lidar_inputs(device, requires_grad=True)
        module_ref = FusedLidarLosses()
        module_ref._cuda_available = False
        dl1, il1, rl1, bl1 = module_ref(*inputs1, df, if_, rf, bf)
        (dl1.sum() + il1.sum() + rl1.sum() + bl1.sum()).backward()

        # CUDA
        inputs2 = _make_lidar_inputs(device, requires_grad=True)
        module_cuda = FusedLidarLosses()
        dl2, il2, rl2, bl2 = module_cuda(*inputs2, df, if_, rf, bf)
        (dl2.sum() + il2.sum() + rl2.sum() + bl2.sum()).backward()

        # inputs[1]=distance_pred, [3]=intensity_pred, [5]=raydrop_pred, [7]=bg_pred
        for i, name in [(1, "distance"), (3, "intensity"), (5, "raydrop"), (7, "bg")]:
            assert torch.allclose(
                inputs2[i].grad, inputs1[i].grad, atol=1e-4
            ), f"{name}_pred grad max diff: {(inputs2[i].grad - inputs1[i].grad).abs().max()}"

    def test_disabled_factors(self):
        """All four factors < 0 → every sub-loss is disabled and zeroed
        (LiDAR counterpart of the camera disabled-factors test)."""
        device = torch.device("cuda")
        inputs = _make_lidar_inputs(device)
        dl, il, rl, bl = FusedLidarLosses()(
            *inputs,
            distance_factor=-1.0,
            intensity_factor=-1.0,
            raydrop_factor=-1.0,
            bg_factor=-1.0,
        )
        assert (dl == 0).all()
        assert (il == 0).all()
        assert (rl == 0).all()
        assert (bl == 0).all()

    def test_empty_input(self):
        """Empty tensors exercise the autograd/module wrappers (kernels early
        return for N=0)."""
        device = torch.device("cuda")
        flags = torch.empty(0, dtype=torch.int32, device=device)
        z = torch.empty(0, device=device)
        dl, il, rl, bl = FusedLidarLosses()(flags, z, z, z, z, z, z, z)
        for t in (dl, il, rl, bl):
            assert t.shape == (0,)
