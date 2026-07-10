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

"""Tests for fused CUDA bg-grid losses.

Compares the CUDA kernels against the pure-PyTorch implementations for
forward values and backward gradients across five fused sub-losses:

- Sky env-map TV (planar and cubemap)
- Grid drift (camera + frame)
- Grid spatial TV (camera + frame)
"""

import pytest
import torch

from gsplat import has_losses
from gsplat.losses_fused import FusedBgGridLosses, _bg_grid_losses_pytorch

CUDA_AVAILABLE = torch.cuda.is_available() and has_losses()

# Small default shapes — plenty to exercise boundary conditions without
# being slow.
BG_B, BG_H, BG_W, BG_C = 2, 4, 5, 3
GC_B, GC_D, GC_H, GC_W = 2, 3, 4, 5  # camera grid shape (per-batch)
GF_B, GF_D, GF_H, GF_W = 1, 2, 3, 4  # frame grid shape (per-batch)


def _make_bg_tex(device, depth=1, *, requires_grad=False, seed=0):
    torch.manual_seed(seed)
    t = torch.randn(BG_B * depth, BG_H, BG_W, BG_C, device=device, dtype=torch.float32)
    if requires_grad:
        t.requires_grad_(True)
    return t


def _make_grids_camera(device, *, requires_grad=False, seed=1):
    torch.manual_seed(seed)
    t = torch.randn(GC_B * 12, GC_D, GC_H, GC_W, device=device, dtype=torch.float32)
    if requires_grad:
        t.requires_grad_(True)
    return t


def _make_grids_frame(device, *, requires_grad=False, seed=2):
    torch.manual_seed(seed)
    t = torch.randn(GF_B * 12, GF_D, GF_H, GF_W, device=device, dtype=torch.float32)
    if requires_grad:
        t.requires_grad_(True)
    return t


# ---------------------------------------------------------------------------
# CPU fallback tests — always runnable
# ---------------------------------------------------------------------------


class TestFusedBgGridLossesFallback:
    """Verify the pure-PyTorch fallback produces sensible shapes and values."""

    def test_all_inputs_planar_shapes(self):
        device = torch.device("cpu")
        bg = _make_bg_tex(device, depth=1)
        gc = _make_grids_camera(device)
        gf = _make_grids_frame(device)

        module = FusedBgGridLosses()
        bg_loss, drift_loss, tv_c, tv_f = module(bg, gc, gf)

        assert bg_loss.shape == (bg.numel(),)
        assert drift_loss.shape == (
            GC_B * GC_D * GC_H * GC_W + GF_B * GF_D * GF_H * GF_W,
        )
        assert tv_c.shape == (GC_B * GC_D * GC_H * GC_W,)
        assert tv_f.shape == (GF_B * GF_D * GF_H * GF_W,)
        assert torch.all(torch.isfinite(bg_loss))
        assert torch.all(drift_loss >= 0.0)

    def test_cubemap(self):
        device = torch.device("cpu")
        bg = _make_bg_tex(device, depth=6)

        module = FusedBgGridLosses()
        bg_loss, _, _, _ = module(
            bg,
            grids_camera=None,
            grids_frame=None,
            bg_tex_depth=6,
        )
        # Cubemap faces treated independently — we should see non-zero TV
        # contributions in both the D (across faces within a batch) and the
        # H/W directions.
        assert bg_loss.shape == (bg.numel(),)
        assert bg_loss.sum() > 0.0

    def test_partial_inputs(self):
        """Only grids_camera, no bg_tex / grids_frame."""
        device = torch.device("cpu")
        gc = _make_grids_camera(device)
        module = FusedBgGridLosses()
        bg_loss, drift_loss, tv_c, tv_f = module(
            bg_tex=None, grids_camera=gc, grids_frame=None
        )
        assert bg_loss.numel() == 0
        assert drift_loss.shape == (GC_B * GC_D * GC_H * GC_W,)
        assert tv_c.shape == (GC_B * GC_D * GC_H * GC_W,)
        assert tv_f.numel() == 0

    def test_disabled_sub_loss_via_negative_factor(self):
        """factor < 0 should produce a zero tensor of the correct shape."""
        device = torch.device("cpu")
        gc = _make_grids_camera(device)
        module = FusedBgGridLosses()
        _, drift_loss, tv_c, _ = module(
            grids_camera=gc,
            grid_drift_camera_factor=-1.0,  # disabled
            grid_camera_tv_factor=1.0,
        )
        assert drift_loss.shape == (GC_B * GC_D * GC_H * GC_W,)
        assert torch.all(drift_loss == 0.0)
        assert tv_c.sum() > 0.0

    def test_invalid_bg_tex_depth_rejected(self):
        device = torch.device("cpu")
        bg = _make_bg_tex(device, depth=1)  # bg_tex.shape[0] == BG_B == 2
        module = FusedBgGridLosses()
        with pytest.raises(ValueError, match="bg_tex_depth must be >= 1"):
            module(bg_tex=bg, bg_tex_depth=0)
        with pytest.raises(ValueError, match="divisible"):
            module(bg_tex=bg, bg_tex_depth=4)  # 2 % 4 != 0

    def test_backward_runs(self):
        device = torch.device("cpu")
        bg = _make_bg_tex(device, requires_grad=True)
        gc = _make_grids_camera(device, requires_grad=True)
        gf = _make_grids_frame(device, requires_grad=True)
        module = FusedBgGridLosses()
        bg_loss, drift_loss, tv_c, tv_f = module(bg, gc, gf)
        total = bg_loss.sum() + drift_loss.sum() + tv_c.sum() + tv_f.sum()
        total.backward()
        assert bg.grad is not None
        assert gc.grad is not None
        assert gf.grad is not None

    def test_disabled_output_with_present_input_is_differentiable(self):
        """A present-but-disabled sub-loss (negative factor) must return an
        output still connected to its input's autograd graph, so backward
        yields a zero gradient rather than raising on a detached leaf — matching
        the CUDA custom Function. Covers bg_tex, grid-drift and grid-TV."""
        device = torch.device("cpu")
        bg = _make_bg_tex(device, requires_grad=True)
        gc = _make_grids_camera(device, requires_grad=True)
        gf = _make_grids_frame(device, requires_grad=True)
        module = FusedBgGridLosses()
        # Every input present, but every sub-loss disabled via a negative factor.
        bg_loss, drift_loss, tv_c, tv_f = module(
            bg,
            gc,
            gf,
            bg_tex_factor=-1.0,
            grid_drift_camera_factor=-1.0,
            grid_drift_frame_factor=-1.0,
            grid_camera_tv_factor=-1.0,
            grid_frame_tv_factor=-1.0,
        )
        # Outputs are all-zero but must carry a grad_fn (not detached leaves).
        for out in (bg_loss, drift_loss, tv_c, tv_f):
            assert torch.all(out == 0.0)
            assert out.requires_grad
        (bg_loss.sum() + drift_loss.sum() + tv_c.sum() + tv_f.sum()).backward()
        for t in (bg, gc, gf):
            assert t.grad is not None
            assert torch.all(t.grad == 0.0)

    def test_non_contiguous_inputs_fallback(self):
        """The fallback must accept non-contiguous inputs (the CUDA path makes
        them contiguous first). A strided view with the documented logical shape
        must not raise and must match the contiguous-input result."""
        device = torch.device("cpu")

        def _non_contig(shape, seed):
            torch.manual_seed(seed)
            # Allocate an extra trailing dim and drop it, yielding a tensor with
            # the requested shape but non-contiguous storage.
            big = torch.randn(*shape, 2, device=device, dtype=torch.float32)
            t = big[..., 0]
            assert not t.is_contiguous()
            return t

        bg = _non_contig((BG_B, BG_H, BG_W, BG_C), seed=0)
        gc = _non_contig((GC_B * 12, GC_D, GC_H, GC_W), seed=1)
        gf = _non_contig((GF_B * 12, GF_D, GF_H, GF_W), seed=2)

        module = FusedBgGridLosses()
        # Must not raise on non-contiguous inputs...
        out_nc = module(bg, gc, gf)
        # ...and must equal the contiguous-input result.
        out_c = module(bg.contiguous(), gc.contiguous(), gf.contiguous())
        for a, b, name in zip(out_nc, out_c, ("bg", "drift", "tv_c", "tv_f")):
            assert torch.equal(a, b), f"{name} differs on non-contiguous input"

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_disabled_output_with_nonfinite_input_stays_zero(self, bad):
        """A present-but-disabled sub-loss whose input contains NaN/±Inf must
        still produce an exact-zero output and exact-zero input gradient. The
        disabled branch keeps the autograd edge via an empty-view sum, which
        never reads the input's values, so a non-finite disabled input cannot
        poison the output (``0.0 * NaN`` would be NaN)."""
        device = torch.device("cpu")
        bg = _make_bg_tex(device, requires_grad=True)
        with torch.no_grad():
            bg[0, 0, 0, 0] = bad
        gc = _make_grids_camera(device, requires_grad=True)
        gf = _make_grids_frame(device, requires_grad=True)
        module = FusedBgGridLosses()
        # bg_tex present but disabled (contains a non-finite value).
        bg_loss, drift_loss, tv_c, tv_f = module(
            bg,
            gc,
            gf,
            bg_tex_factor=-1.0,
            grid_drift_camera_factor=-1.0,
            grid_drift_frame_factor=-1.0,
            grid_camera_tv_factor=-1.0,
            grid_frame_tv_factor=-1.0,
        )
        for out in (bg_loss, drift_loss, tv_c, tv_f):
            assert torch.all(out == 0.0), "disabled output not exactly zero"
        (bg_loss.sum() + drift_loss.sum() + tv_c.sum() + tv_f.sum()).backward()
        for tns in (bg, gc, gf):
            assert tns.grad is not None
            assert torch.all(tns.grad == 0.0), "disabled input grad not exactly zero"


# ---------------------------------------------------------------------------
# CUDA tests — only run when GPU + compiled extension available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA or fused losses not available")
class TestFusedBgGridLossesCUDA:
    """Compare the CUDA fused kernel against the pure-PyTorch reference."""

    def test_forward_matches_pytorch_all_inputs(self):
        device = torch.device("cuda")
        bg = _make_bg_tex(device)
        gc = _make_grids_camera(device)
        gf = _make_grids_frame(device)

        module = FusedBgGridLosses()
        bg_c, drift_c, tv_c_cuda, tv_f_cuda = module(bg, gc, gf)

        ref_bg, ref_drift, ref_tv_c, ref_tv_f = _bg_grid_losses_pytorch(
            bg, 1, gc, gf, 1.0, 1.0, 1.0, 1.0, 1.0
        )

        assert torch.allclose(
            bg_c, ref_bg, atol=1e-5
        ), f"bg_tex max diff: {(bg_c - ref_bg).abs().max()}"
        assert torch.allclose(
            drift_c, ref_drift, atol=1e-5
        ), f"drift max diff: {(drift_c - ref_drift).abs().max()}"
        assert torch.allclose(
            tv_c_cuda, ref_tv_c, atol=1e-5
        ), f"tv_camera max diff: {(tv_c_cuda - ref_tv_c).abs().max()}"
        assert torch.allclose(
            tv_f_cuda, ref_tv_f, atol=1e-5
        ), f"tv_frame max diff: {(tv_f_cuda - ref_tv_f).abs().max()}"

    def test_forward_cubemap(self):
        device = torch.device("cuda")
        bg = _make_bg_tex(device, depth=6)

        module = FusedBgGridLosses()
        bg_c, _, _, _ = module(bg, bg_tex_depth=6)
        ref_bg, _, _, _ = _bg_grid_losses_pytorch(
            bg, 6, None, None, 1.0, 1.0, 1.0, 1.0, 1.0
        )
        assert torch.allclose(bg_c, ref_bg, atol=1e-5)

    def test_partial_inputs(self):
        device = torch.device("cuda")
        gc = _make_grids_camera(device)

        module = FusedBgGridLosses()
        bg_c, drift_c, tv_c_cuda, tv_f_cuda = module(
            bg_tex=None, grids_camera=gc, grids_frame=None
        )
        # Shapes match the all-inputs case for the populated sub-losses;
        # empty for the missing ones.
        assert bg_c.numel() == 0
        assert drift_c.shape == (GC_B * GC_D * GC_H * GC_W,)
        assert tv_c_cuda.shape == (GC_B * GC_D * GC_H * GC_W,)
        assert tv_f_cuda.numel() == 0

        ref_bg, ref_drift, ref_tv_c, ref_tv_f = _bg_grid_losses_pytorch(
            None, 1, gc, None, 1.0, 1.0, 1.0, 1.0, 1.0
        )
        assert torch.allclose(drift_c, ref_drift, atol=1e-5)
        assert torch.allclose(tv_c_cuda, ref_tv_c, atol=1e-5)

    def test_disabled_sub_loss_via_negative_factor(self):
        device = torch.device("cuda")
        gc = _make_grids_camera(device)
        module = FusedBgGridLosses()
        _, drift_loss, tv_c, _ = module(
            grids_camera=gc,
            grid_drift_camera_factor=-1.0,
            grid_camera_tv_factor=1.0,
        )
        assert torch.all(drift_loss == 0.0)
        assert tv_c.sum() > 0.0

    def test_nan_factor_propagates_matches_pytorch(self):
        # A NaN factor must propagate (not be silently skipped) on the CUDA
        # path, matching the pure-PyTorch reference which disables only on
        # factor < 0; the CUDA gates use the same negative-only disable test.
        device = torch.device("cuda")
        gc = _make_grids_camera(device)
        nan = float("nan")
        module = FusedBgGridLosses()
        _, _, tv_cuda, _ = module(grids_camera=gc, grid_camera_tv_factor=nan)
        module_ref = FusedBgGridLosses()
        module_ref._cuda_available = False
        _, _, tv_ref, _ = module_ref(grids_camera=gc, grid_camera_tv_factor=nan)
        assert torch.isnan(tv_cuda).any()
        assert torch.isnan(tv_ref).any()

    def test_backward_matches_pytorch(self):
        device = torch.device("cuda")

        bg1 = _make_bg_tex(device, requires_grad=True, seed=11)
        gc1 = _make_grids_camera(device, requires_grad=True, seed=12)
        gf1 = _make_grids_frame(device, requires_grad=True, seed=13)
        ref_bg, ref_drift, ref_tv_c, ref_tv_f = _bg_grid_losses_pytorch(
            bg1, 1, gc1, gf1, 1.0, 1.0, 1.0, 1.0, 1.0
        )
        (ref_bg.sum() + ref_drift.sum() + ref_tv_c.sum() + ref_tv_f.sum()).backward()

        bg2 = _make_bg_tex(device, requires_grad=True, seed=11)
        gc2 = _make_grids_camera(device, requires_grad=True, seed=12)
        gf2 = _make_grids_frame(device, requires_grad=True, seed=13)
        module = FusedBgGridLosses()
        bg_c, drift_c, tv_c_cuda, tv_f_cuda = module(bg2, gc2, gf2)
        (bg_c.sum() + drift_c.sum() + tv_c_cuda.sum() + tv_f_cuda.sum()).backward()

        assert torch.allclose(
            bg2.grad, bg1.grad, atol=1e-4
        ), f"bg grad max diff: {(bg2.grad - bg1.grad).abs().max()}"
        assert torch.allclose(
            gc2.grad, gc1.grad, atol=1e-4
        ), f"grids_camera grad max diff: {(gc2.grad - gc1.grad).abs().max()}"
        assert torch.allclose(
            gf2.grad, gf1.grad, atol=1e-4
        ), f"grids_frame grad max diff: {(gf2.grad - gf1.grad).abs().max()}"

    def test_float64_rejected(self):
        """The fused bg-grid op is float32-only; an fp64 CUDA input must raise
        at the native op rather than silently misbehaving."""
        device = torch.device("cuda")
        bg = _make_bg_tex(device).double()
        module = FusedBgGridLosses()
        with pytest.raises(RuntimeError):
            module(bg, bg_tex_depth=1)

    def test_drift_near_identity_deadband(self):
        """Near-identity grids fall in the drift backward dead-zone
        (sum_sq <= 1e-12): the kernel guards `sum_sq > 1e-12` before the
        1/sqrt drift gradient, and the reference mirrors it. Both must give a
        zero drift gradient (and a matching forward), so neither blows up."""
        device = torch.device("cuda")
        # 3x4 affine identity, broadcast to [GC_B*12, GC_D, GC_H, GC_W], then a
        # tiny uniform perturbation that keeps each cell's sum_sq < 1e-12.
        eye = torch.eye(3, 4, device=device).reshape(12)
        base = (
            eye.view(1, 12, 1, 1, 1)
            .expand(GC_B, 12, GC_D, GC_H, GC_W)
            .reshape(GC_B * 12, GC_D, GC_H, GC_W)
            .contiguous()
        )

        def _grid():
            return (base + 1e-7).clone().requires_grad_(True)

        # Drift only (TV disabled) so the gradient isolates the drift path.
        g1 = _grid()
        _, ref_drift, _, _ = _bg_grid_losses_pytorch(
            None, 1, g1, None, 1.0, 1.0, 1.0, -1.0, -1.0
        )
        ref_drift.sum().backward()

        g2 = _grid()
        _, drift_c, _, _ = FusedBgGridLosses()(
            grids_camera=g2,
            grid_drift_camera_factor=1.0,
            grid_camera_tv_factor=-1.0,
        )
        drift_c.sum().backward()

        # Forward is the exact (tiny) sqrt(sum_sq) on both paths.
        assert torch.allclose(drift_c, ref_drift, atol=1e-6)
        # Dead-zone: zero drift gradient, no 1/sqrt blow-up.
        assert torch.all(g1.grad == 0.0)
        assert torch.all(g2.grad == 0.0)

    def test_backward_cubemap_matches_pytorch(self):
        """Backward parity for the cubemap (depth=6) face-neighbour TV path."""
        device = torch.device("cuda")
        bg1 = _make_bg_tex(device, depth=6, requires_grad=True, seed=21)
        ref_bg, _, _, _ = _bg_grid_losses_pytorch(
            bg1, 6, None, None, 1.0, 1.0, 1.0, 1.0, 1.0
        )
        ref_bg.sum().backward()

        bg2 = _make_bg_tex(device, depth=6, requires_grad=True, seed=21)
        bg_c, _, _, _ = FusedBgGridLosses()(bg2, bg_tex_depth=6)
        bg_c.sum().backward()
        assert torch.allclose(
            bg2.grad, bg1.grad, atol=1e-4
        ), f"cubemap bg grad max diff: {(bg2.grad - bg1.grad).abs().max()}"

    def test_backward_partial_inputs_matches_pytorch(self):
        """Backward parity when only grids_camera is provided."""
        device = torch.device("cuda")
        gc1 = _make_grids_camera(device, requires_grad=True, seed=31)
        _, ref_drift, ref_tv_c, _ = _bg_grid_losses_pytorch(
            None, 1, gc1, None, 1.0, 1.0, 1.0, 1.0, 1.0
        )
        (ref_drift.sum() + ref_tv_c.sum()).backward()

        gc2 = _make_grids_camera(device, requires_grad=True, seed=31)
        _, drift_c, tv_c_cuda, _ = FusedBgGridLosses()(grids_camera=gc2)
        (drift_c.sum() + tv_c_cuda.sum()).backward()
        assert torch.allclose(
            gc2.grad, gc1.grad, atol=1e-4
        ), f"partial grids_camera grad max diff: {(gc2.grad - gc1.grad).abs().max()}"

    def test_backward_disabled_factors_matches_pytorch(self):
        """Backward parity with drift disabled (TV-only gradient) on both the
        camera and frame grids."""
        device = torch.device("cuda")
        gc1 = _make_grids_camera(device, requires_grad=True, seed=41)
        gf1 = _make_grids_frame(device, requires_grad=True, seed=42)
        _, _, ref_tv_c, ref_tv_f = _bg_grid_losses_pytorch(
            None, 1, gc1, gf1, 1.0, -1.0, -1.0, 1.0, 1.0
        )
        (ref_tv_c.sum() + ref_tv_f.sum()).backward()

        gc2 = _make_grids_camera(device, requires_grad=True, seed=41)
        gf2 = _make_grids_frame(device, requires_grad=True, seed=42)
        _, _, tv_c_cuda, tv_f_cuda = FusedBgGridLosses()(
            grids_camera=gc2,
            grids_frame=gf2,
            grid_drift_camera_factor=-1.0,
            grid_drift_frame_factor=-1.0,
            grid_camera_tv_factor=1.0,
            grid_frame_tv_factor=1.0,
        )
        (tv_c_cuda.sum() + tv_f_cuda.sum()).backward()
        assert torch.allclose(
            gc2.grad, gc1.grad, atol=1e-4
        ), f"camera grad max diff: {(gc2.grad - gc1.grad).abs().max()}"
        assert torch.allclose(
            gf2.grad, gf1.grad, atol=1e-4
        ), f"frame grad max diff: {(gf2.grad - gf1.grad).abs().max()}"

    def test_backward_nan_factor_on_unused_output_does_not_poison_sibling(self):
        """An unused output carrying a NaN factor must not contaminate a finite
        sibling's shared-input gradient on the fused path.

        ``grids_camera`` feeds both the drift output (NaN camera-drift factor) and
        the camera-TV output (finite). Backpropagating only the finite TV sibling
        must leave ``grids_camera.grad`` finite and matching the PyTorch fallback,
        which never traverses the unused NaN branch. ``set_materialize_grads(False)``
        plus the per-output factor-disable in ``backward`` is what prevents the fused
        path from evaluating ``NaN * 0`` for the unused drift output.
        """
        device = torch.device("cuda")
        # PyTorch reference: only the TV output is backpropagated; the NaN-weighted
        # drift output stays untraversed, so the grad is finite.
        gc1 = _make_grids_camera(device, requires_grad=True, seed=41)
        _, _, ref_tv_c, _ = _bg_grid_losses_pytorch(
            None, 1, gc1, None, 1.0, float("nan"), -1.0, 1.0, -1.0
        )
        ref_tv_c.sum().backward()
        assert torch.isfinite(gc1.grad).all()

        # Fused CUDA path: same setup, only the finite TV sibling is used.
        gc2 = _make_grids_camera(device, requires_grad=True, seed=41)
        _, _, tv_c_cuda, _ = FusedBgGridLosses()(
            grids_camera=gc2,
            grid_drift_camera_factor=float("nan"),  # NaN-weighted, unused output
            grid_drift_frame_factor=-1.0,
            grid_camera_tv_factor=1.0,  # finite sibling — the only output used
            grid_frame_tv_factor=-1.0,
        )
        tv_c_cuda.sum().backward()
        assert torch.isfinite(
            gc2.grad
        ).all(), "fused path leaked a NaN factor into the finite sibling's gradient"
        assert torch.allclose(
            gc2.grad, gc1.grad, atol=1e-4
        ), f"camera grad max diff: {(gc2.grad - gc1.grad).abs().max()}"

    def test_backward_nan_factor_on_unused_frame_output_does_not_poison_sibling(self):
        """Frame-path mirror of the camera case: a NaN frame-drift factor on an
        unused output must not poison the finite frame-TV sibling's shared-input
        gradient. Uses a *present* ``grids_frame`` (not pre-disabled), so the
        frame drift branch is actually exercised — the combined camera+frame
        drift output is untraversed (``None`` cotangent) and both drift factors
        are disabled for the native backward, so the NaN frame-drift factor
        never reaches ``NaN * 0``.
        """
        device = torch.device("cuda")
        # _bg_grid_losses_pytorch(bg_tex, depth, grids_camera, grids_frame,
        #   bg_factor, drift_cam, drift_frame, tv_cam, tv_frame): only the frame
        # TV output is backpropagated; the NaN-weighted frame drift stays unused.
        gf1 = _make_grids_frame(device, requires_grad=True, seed=52)
        _, _, _, ref_tv_f = _bg_grid_losses_pytorch(
            None, 1, None, gf1, 1.0, -1.0, float("nan"), -1.0, 1.0
        )
        ref_tv_f.sum().backward()
        assert torch.isfinite(gf1.grad).all()

        gf2 = _make_grids_frame(device, requires_grad=True, seed=52)
        _, _, _, tv_f_cuda = FusedBgGridLosses()(
            grids_frame=gf2,
            grid_drift_camera_factor=-1.0,
            grid_drift_frame_factor=float("nan"),  # NaN-weighted, unused output
            grid_camera_tv_factor=-1.0,
            grid_frame_tv_factor=1.0,  # finite sibling — the only output used
        )
        tv_f_cuda.sum().backward()
        assert torch.isfinite(
            gf2.grad
        ).all(), "fused path leaked a NaN frame-drift factor into the finite sibling"
        assert torch.allclose(
            gf2.grad, gf1.grad, atol=1e-4
        ), f"frame grad max diff: {(gf2.grad - gf1.grad).abs().max()}"
