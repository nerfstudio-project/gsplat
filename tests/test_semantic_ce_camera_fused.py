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

"""Tests for the semantic-CE member of the fused camera-loss dispatch.

The masked semantic cross-entropy rides along the fused camera dispatch: one
``camera_losses_fwd``/``camera_losses_bwd`` op call
launches the RGB/background kernels plus, when the optional ``semantic_*``
inputs are present, the semantic-CE kernels.

The pure-PyTorch reference (``semantic_cross_entropy_masked``) is first
validated against an independent ``torch.nn.functional.cross_entropy``
composition (CPU, always runnable); the grouped CUDA member is then compared
against that reference for the forward value and the backward gradients
w.r.t. the logits.

Denominator contract under test: every valid row
counts in the denominator, ignore_index rows contribute zero to the numerator
only, and the denominator is clamped to >= 1.
"""

import os
import subprocess
import sys

import pytest
import torch

import gsplat.losses as losses_module
from gsplat import LossFlag, has_losses
from gsplat.losses import semantic_cross_entropy_masked
from gsplat.losses_fused import FusedCameraLosses, _cuda_camera_losses_available

# The CE kernels ship inside the camera dispatch: op availability is the
# camera op's availability (there is no standalone semantic-CE op).
CAMERA_CUDA = torch.cuda.is_available() and _cuda_camera_losses_available()

N = 4096
C = 33
IGNORE_INDEX = -100

# Per-ray flag bits (plain int) for the camera fixtures, derived from the
# public ``LossFlag`` vocabulary (mirrors tests/test_losses_fused.py).
# ``None`` when the loss extension is not built; flag-using tests are gated
# on ``has_losses()`` / ``CAMERA_CUDA`` accordingly.
if LossFlag is not None:
    _FLAG_RGB_LABEL = int(LossFlag.RGB_LABEL)
    _FLAG_SKY_SEMANTIC = int(LossFlag.SKY_SEMANTIC)
    _FLAG_INVALID = int(LossFlag.INVALID)
    _FLAG_DIFIXED = int(LossFlag.DIFIXED)
else:
    _FLAG_RGB_LABEL = _FLAG_SKY_SEMANTIC = None
    _FLAG_INVALID = _FLAG_DIFIXED = None


def _make_semantic_inputs(
    device,
    n=N,
    c=C,
    dtype=torch.float32,
    target_dtype=torch.int64,
    n_ignored=0,
    requires_grad=False,
    seed=11,
):
    """Random logits, targets, and a ~70%-valid row mask.

    When ``n_ignored > 0``, that many *valid* rows get an ``IGNORE_INDEX``
    target (int64 targets only — uint8 cannot represent -100).
    """
    torch.manual_seed(seed)
    logits = torch.randn(n, c, device=device, dtype=dtype)
    targets = torch.randint(0, c, (n,), device=device, dtype=target_dtype)
    valid = torch.rand(n, device=device) < 0.7
    if n_ignored > 0:
        assert target_dtype == torch.int64, "-100 requires int64 targets"
        valid_rows = valid.nonzero(as_tuple=True)[0]
        assert valid_rows.numel() >= n_ignored
        targets[valid_rows[:n_ignored]] = IGNORE_INDEX
    if requires_grad:
        logits.requires_grad_(True)
    return logits, targets, valid


def _make_camera_inputs(device, n=512, requires_grad=False):
    """Random camera fixtures (mirrors tests/test_losses_fused.py)."""
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


def _reference_loss(logits, targets, valid, ignore_index=IGNORE_INDEX):
    """Independent reference: sum-reduced F.cross_entropy over contributing
    rows divided by the clamped count of *valid* rows (ignored rows count in
    the denominator but not the numerator)."""
    target_idx = targets.long()
    contributing = valid & (target_idx != ignore_index)
    n_valid = valid.sum().clamp(min=1).to(logits.dtype)
    ce_sum = torch.nn.functional.cross_entropy(
        logits[contributing], target_idx[contributing], reduction="sum"
    )
    return ce_sum / n_valid


def _noncontiguous_view(t):
    """The same values as ``t`` seen through a stride-doubled view."""
    view = torch.repeat_interleave(t.unsqueeze(-1), 2, dim=-1)[..., 0]
    assert not view.is_contiguous()
    return view


# ---------------------------------------------------------------------------
# Pure-PyTorch reference tests — always runnable, validate the reference itself
# ---------------------------------------------------------------------------


class TestSemanticCrossEntropyMaskedReference:
    """Verify the pure-PyTorch reference against an independent
    F.cross_entropy composition (the grouped CUDA member is later held to
    this reference)."""

    def test_forward_matches_reference(self):
        device = torch.device("cpu")
        logits, targets, valid = _make_semantic_inputs(device, n_ignored=16)

        loss = semantic_cross_entropy_masked(logits, targets, valid)
        ref = _reference_loss(logits, targets, valid)
        assert loss.shape == ()
        assert torch.allclose(loss, ref, rtol=1e-6, atol=1e-7)

    def test_backward_matches_reference(self):
        device = torch.device("cpu")
        logits, targets, valid = _make_semantic_inputs(
            device, n_ignored=16, requires_grad=True
        )

        loss = semantic_cross_entropy_masked(logits, targets, valid)
        loss.backward()

        logits_ref = logits.detach().clone().requires_grad_(True)
        ref = _reference_loss(logits_ref, targets, valid)
        ref.backward()

        assert torch.allclose(logits.grad, logits_ref.grad, rtol=1e-6, atol=1e-7)
        # Invalid rows must receive exact zeros (not merely small values).
        assert logits.grad[~valid].abs().max().item() == 0.0

    def test_ignore_index_denominator_contract(self):
        """A -100 target on a valid row: zero numerator contribution, but the
        row still counts in the denominator; its logits-grad row is exactly
        zero."""
        device = torch.device("cpu")
        torch.manual_seed(3)
        n, c = 8, 5
        logits = torch.randn(n, c, device=device, requires_grad=True)
        targets = torch.randint(0, c, (n,), device=device, dtype=torch.int64)
        valid = torch.ones(n, dtype=torch.bool, device=device)
        valid[3] = False  # masked-out row
        targets[5] = IGNORE_INDEX  # ignored-but-valid row

        loss = semantic_cross_entropy_masked(logits, targets, valid)

        contributing = valid.clone()
        contributing[5] = False
        n_valid = 7  # rows 0,1,2,4,5,6,7 — row 5 counts despite being ignored
        expected = (
            torch.nn.functional.cross_entropy(
                logits.detach()[contributing],
                targets[contributing],
                reduction="sum",
            )
            / n_valid
        )
        assert torch.allclose(loss, expected, rtol=1e-6, atol=1e-7)

        loss.backward()
        assert logits.grad[5].abs().max().item() == 0.0, "ignored row grad"
        assert logits.grad[3].abs().max().item() == 0.0, "invalid row grad"
        assert logits.grad[contributing].abs().max().item() > 0.0

    def test_uint8_targets_match_int64(self):
        device = torch.device("cpu")
        logits, targets, valid = _make_semantic_inputs(
            device, target_dtype=torch.uint8, requires_grad=True
        )

        loss_u8 = semantic_cross_entropy_masked(logits, targets, valid)
        loss_u8.backward()
        grad_u8 = logits.grad.clone()

        logits2 = logits.detach().clone().requires_grad_(True)
        loss_i64 = semantic_cross_entropy_masked(logits2, targets.long(), valid)
        loss_i64.backward()

        assert torch.equal(loss_u8, loss_i64)
        assert torch.equal(grad_u8, logits2.grad)

    def test_custom_ignore_index_uint8(self):
        """A non-negative ignore_index is expressible with uint8 targets and
        takes precedence over the class-index interpretation."""
        device = torch.device("cpu")
        torch.manual_seed(5)
        n, c = 64, 7
        logits = torch.randn(n, c, device=device, requires_grad=True)
        targets = torch.randint(0, c, (n,), device=device, dtype=torch.uint8)
        valid = torch.ones(n, dtype=torch.bool, device=device)
        targets[2] = 4  # ignored below

        loss = semantic_cross_entropy_masked(logits, targets, valid, ignore_index=4)
        ref = _reference_loss(logits.detach(), targets, valid, ignore_index=4)
        assert torch.allclose(loss, ref, rtol=1e-6, atol=1e-7)

        loss.backward()
        ignored = targets.long() == 4
        assert logits.grad[ignored].abs().max().item() == 0.0

    def test_all_invalid_mask(self):
        """All-invalid mask: the clamped denominator yields an exact zero loss
        and exact-zero gradients."""
        device = torch.device("cpu")
        logits, targets, _ = _make_semantic_inputs(device, requires_grad=True)
        valid = torch.zeros(N, dtype=torch.bool, device=device)

        loss = semantic_cross_entropy_masked(logits, targets, valid)
        assert loss.shape == ()
        assert loss.item() == 0.0
        loss.backward()
        assert logits.grad is not None
        assert torch.count_nonzero(logits.grad) == 0

    def test_empty_inputs(self):
        """N == 0 returns a finite, differentiable zero."""
        device = torch.device("cpu")
        logits = torch.empty(0, C, device=device, requires_grad=True)
        targets = torch.empty(0, dtype=torch.int64, device=device)
        valid = torch.empty(0, dtype=torch.bool, device=device)

        loss = semantic_cross_entropy_masked(logits, targets, valid)
        assert loss.shape == ()
        assert torch.isfinite(loss) and loss.item() == 0.0
        loss.backward()
        assert logits.grad is not None and logits.grad.shape == (0, C)

    def test_fp64(self):
        device = torch.device("cpu")
        logits, targets, valid = _make_semantic_inputs(
            device, dtype=torch.float64, n_ignored=8, requires_grad=True
        )

        loss = semantic_cross_entropy_masked(logits, targets, valid)
        ref = _reference_loss(logits.detach(), targets, valid)
        assert loss.dtype == torch.float64
        assert torch.allclose(loss, ref, rtol=1e-12, atol=1e-12)
        loss.backward()
        assert torch.isfinite(logits.grad).all()

    def test_structural_dtype_checks_are_unconditional(self, monkeypatch):
        """Structural shape/dtype checks run on every call — with
        ENFORCE_CONTRACTS off — and before any cast, so float targets are
        never silently truncated by ``.long()`` and non-bool masks never
        silently reinterpreted. The CUDA entry rejects the same inputs, so
        the two paths refuse identical calls."""
        monkeypatch.setattr(losses_module, "ENFORCE_CONTRACTS", False)
        device = torch.device("cpu")
        logits, targets, valid = _make_semantic_inputs(device, n=32)
        with pytest.raises(TypeError, match="targets must be uint8 or int64"):
            semantic_cross_entropy_masked(logits, targets.float(), valid)
        with pytest.raises(TypeError, match="targets must be uint8 or int64"):
            semantic_cross_entropy_masked(logits, targets.to(torch.int32), valid)
        with pytest.raises(TypeError, match="valid must be bool"):
            semantic_cross_entropy_masked(logits, targets, valid.to(torch.uint8))
        with pytest.raises(ValueError, match="logits must be"):
            semantic_cross_entropy_masked(logits.flatten(), targets, valid)

    def test_out_of_range_contributing_target_skips_deterministically(
        self, monkeypatch
    ):
        """A *valid* target outside [0, C) (and != ignore_index) is a caller
        bug; with ENFORCE_CONTRACTS off it is deterministically skipped
        exactly like an ignored row — zero numerator contribution, still
        counted in the denominator, exact-zero gradient row — matching the
        CUDA kernels on assert-free builds."""
        monkeypatch.setattr(losses_module, "ENFORCE_CONTRACTS", False)
        device = torch.device("cpu")
        torch.manual_seed(3)
        n, c = 8, 5
        logits = torch.randn(n, c, device=device, requires_grad=True)
        targets = torch.randint(0, c, (n,), device=device, dtype=torch.int64)
        valid = torch.ones(n, dtype=torch.bool, device=device)
        targets[2] = c + 7  # above the class range
        targets[6] = -3  # below the class range (and != ignore_index)

        loss = semantic_cross_entropy_masked(logits, targets, valid)

        keep = torch.ones(n, dtype=torch.bool, device=device)
        keep[2] = False
        keep[6] = False
        # Both bad rows still count in the denominator (all n rows are valid).
        expected = (
            torch.nn.functional.cross_entropy(
                logits.detach()[keep], targets[keep], reduction="sum"
            )
            / n
        )
        assert torch.allclose(loss, expected, rtol=1e-6, atol=1e-7)

        loss.backward()
        assert logits.grad[2].abs().max().item() == 0.0, "out-of-range row grad"
        assert logits.grad[6].abs().max().item() == 0.0, "out-of-range row grad"
        assert logits.grad[keep].abs().max().item() > 0.0

    def test_out_of_range_contributing_target_loud_under_contracts(self, monkeypatch):
        """With ENFORCE_CONTRACTS on, the same caller bug fails loudly
        (mirroring the CUDA kernels' device assert on assert-enabled
        builds) instead of being skipped."""
        monkeypatch.setattr(losses_module, "ENFORCE_CONTRACTS", True)
        device = torch.device("cpu")
        logits, targets, valid = _make_semantic_inputs(device, n=32)
        bad_row = valid.nonzero(as_tuple=True)[0][0]
        targets[bad_row] = C + 7
        with pytest.raises(AssertionError, match="ignore_index"):
            semantic_cross_entropy_masked(logits, targets, valid)


# ---------------------------------------------------------------------------
# Grouped member — CPU fallback path (always runnable with the extension)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not has_losses(), reason="fused loss extension (flag vocabulary) not built"
)
class TestFusedCameraSemanticGroupedFallback:
    """Grouped-mode behavior of FusedCameraLosses on the pure-PyTorch path."""

    def test_grouped_forward_matches_composition(self):
        device = torch.device("cpu")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        logits, targets, valid = _make_semantic_inputs(device, n=384, n_ignored=16)
        module = FusedCameraLosses()

        rl, bl, ce = module(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=logits,
            semantic_targets=targets,
            semantic_valid=valid,
        )
        rl_ref, bl_ref = module(flags, rgb_pred, rgb_gt, bg_pred)
        ce_ref = semantic_cross_entropy_masked(logits, targets, valid)

        assert ce.shape == ()
        assert torch.equal(rl, rl_ref)
        assert torch.equal(bl, bl_ref)
        assert torch.allclose(ce, ce_ref)

    def test_grouped_backward_routes_v_logits(self):
        device = torch.device("cpu")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(
            device, requires_grad=True
        )
        logits, targets, valid = _make_semantic_inputs(
            device, n=384, n_ignored=16, requires_grad=True
        )
        module = FusedCameraLosses()
        rl, bl, ce = module(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=logits,
            semantic_targets=targets,
            semantic_valid=valid,
        )
        (rl.sum() + bl.sum() + ce).backward()
        assert rgb_pred.grad is not None
        assert bg_pred.grad is not None
        assert logits.grad is not None

        # The CE member's gradient matches the standalone reference.
        logits_ref = logits.detach().clone().requires_grad_(True)
        semantic_cross_entropy_masked(logits_ref, targets, valid).backward()
        assert torch.allclose(logits.grad, logits_ref.grad, atol=1e-6)

    def test_partial_semantic_args_rejected(self):
        device = torch.device("cpu")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        logits, targets, valid = _make_semantic_inputs(device, n=384)
        module = FusedCameraLosses()
        with pytest.raises(ValueError, match="pass all three or none"):
            module(flags, rgb_pred, rgb_gt, bg_pred, semantic_logits=logits)
        with pytest.raises(ValueError, match="pass all three or none"):
            module(
                flags,
                rgb_pred,
                rgb_gt,
                bg_pred,
                semantic_targets=targets,
                semantic_valid=valid,
            )

    def test_absent_semantic_keeps_two_outputs(self):
        """Member-disabled case: omitting the semantic inputs keeps the
        pre-fold two-output behavior."""
        device = torch.device("cpu")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        outputs = FusedCameraLosses()(flags, rgb_pred, rgb_gt, bg_pred)
        assert len(outputs) == 2

    def test_noncontiguous_semantic_inputs_match_contiguous(self):
        """Non-contiguous semantic views reproduce the contiguous fallback
        call bitwise, forward and backward (CPU vs CPU)."""
        device = torch.device("cpu")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        logits, targets, valid = _make_semantic_inputs(device, n=384, n_ignored=16)
        module = FusedCameraLosses()

        base = logits.detach().clone().requires_grad_(True)
        *_, ce = module(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=base,
            semantic_targets=targets,
            semantic_valid=valid,
        )
        ce.backward()

        leaf = logits.detach().clone().requires_grad_(True)
        *_, nc_ce = module(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=_noncontiguous_view(leaf),
            semantic_targets=_noncontiguous_view(targets),
            semantic_valid=_noncontiguous_view(valid),
        )
        nc_ce.backward()

        assert torch.equal(nc_ce, ce)
        assert torch.equal(leaf.grad, base.grad)

    def test_nonfinite_noncontributing_rows_do_not_poison(self):
        """NaN/Inf logits confined to invalid or ignored rows must not leak
        into the loss or the gradients: those rows' logits are never read."""
        device = torch.device("cpu")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        logits, targets, valid = _make_semantic_inputs(device, n=384, n_ignored=8)
        clean = semantic_cross_entropy_masked(logits, targets, valid)

        invalid_rows = (~valid).nonzero(as_tuple=True)[0]
        ignored_rows = (valid & (targets == IGNORE_INDEX)).nonzero(as_tuple=True)[0]
        poisoned = logits.detach().clone().requires_grad_(True)
        with torch.no_grad():
            poisoned[invalid_rows[0]] = float("nan")
            poisoned[invalid_rows[1]] = float("inf")
            poisoned[ignored_rows[0]] = float("-inf")

        *_, ce = FusedCameraLosses()(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=poisoned,
            semantic_targets=targets,
            semantic_valid=valid,
        )
        assert torch.isfinite(ce)
        assert torch.allclose(ce, clean, rtol=1e-6, atol=1e-7)
        ce.backward()
        assert torch.isfinite(poisoned.grad).all()
        assert poisoned.grad[invalid_rows[0]].abs().max().item() == 0.0
        assert poisoned.grad[invalid_rows[1]].abs().max().item() == 0.0
        assert poisoned.grad[ignored_rows[0]].abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Grouped member — CUDA path (one op call launching camera + CE kernels)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CAMERA_CUDA, reason="CUDA or fused camera losses not available")
class TestFusedCameraSemanticGroupedCUDA:
    """The grouped dispatch must reproduce the ungrouped camera call
    bit-for-bit and hold the CE member to the pure-PyTorch reference
    (semantic_cross_entropy_masked) for both forward and backward."""

    @staticmethod
    def _tols(dtype, grad=False):
        """CE kernel vs torch-reference tolerances.

        The CUDA forward accumulates row losses into a scalar workspace via
        float atomicAdd (order nondeterministic), so fp32 parity with the
        reference is tight but not bitwise; fp64 atomics keep ~1e-12 slack.
        Gradients scale as 1/n_valid, so the fp32 atol must sit well below
        the typical gradient magnitude to stay meaningful.
        """
        if dtype == torch.float64:
            return {"rtol": 1e-10, "atol": 1e-12}
        return {"rtol": 1e-5, "atol": 1e-8} if grad else {"rtol": 1e-5, "atol": 1e-6}

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_grouped_forward_matches_pytorch(self, dtype):
        """The camera tensors stay fp32 while the CE member runs in
        ``dtype`` — the two domains dispatch independently."""
        device = torch.device("cuda")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        logits, targets, valid = _make_semantic_inputs(
            device, dtype=dtype, n_ignored=16
        )
        rf, bf = 0.5, 0.3

        rl, bl, ce = FusedCameraLosses()(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            rf,
            bf,
            semantic_logits=logits,
            semantic_targets=targets,
            semantic_valid=valid,
        )
        rl_ref, bl_ref = FusedCameraLosses()(flags, rgb_pred, rgb_gt, bg_pred, rf, bf)
        ce_ref = semantic_cross_entropy_masked(logits, targets, valid)

        # Camera members are untouched by the fold (same kernel, same launch).
        assert torch.equal(rl, rl_ref)
        assert torch.equal(bl, bl_ref)
        assert ce.shape == () and ce.dtype == dtype
        assert torch.allclose(
            ce, ce_ref, **self._tols(dtype)
        ), f"forward max diff: {(ce - ce_ref).abs().max()}"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_grouped_backward_matches_pytorch(self, dtype):
        device = torch.device("cuda")
        rf, bf = 0.5, 0.3

        # Grouped: one backward through the single dispatch op. Scale the
        # upstream gradient so v_semantic_loss != 1 is exercised.
        f1, rp1, rg1, bp1 = _make_camera_inputs(device, requires_grad=True)
        l1, tg, vd = _make_semantic_inputs(
            device, dtype=dtype, n_ignored=16, requires_grad=True
        )
        rl, bl, ce = FusedCameraLosses()(
            f1,
            rp1,
            rg1,
            bp1,
            rf,
            bf,
            semantic_logits=l1,
            semantic_targets=tg,
            semantic_valid=vd,
        )
        (rl.sum() + bl.sum() + 2.5 * ce).backward()

        # Ungrouped camera + the pure-PyTorch reference CE.
        f2, rp2, rg2, bp2 = _make_camera_inputs(device, requires_grad=True)
        l2 = l1.detach().clone().requires_grad_(True)
        rl2, bl2 = FusedCameraLosses()(f2, rp2, rg2, bp2, rf, bf)
        ce2 = semantic_cross_entropy_masked(l2, tg, vd)
        (rl2.sum() + bl2.sum() + 2.5 * ce2).backward()

        # Camera gradients are element-wise deterministic: bit-equal.
        assert torch.equal(rp1.grad, rp2.grad)
        assert torch.equal(bp1.grad, bp2.grad)
        tols = self._tols(dtype, grad=True)
        assert torch.allclose(
            l1.grad, l2.grad, **tols
        ), f"logits grad max diff: {(l1.grad - l2.grad).abs().max()}"
        # Invalid rows must be exactly zero on both paths.
        assert l1.grad[~vd].abs().max().item() == 0.0

    def test_uint8_vs_int64_target_parity(self):
        """uint8 and int64 targets through the grouped camera op agree.

        The uint8/int64 template instantiations may accumulate the workspace
        partials in a different order (ulp-level float differences via
        atomicAdd), so the parity tolerance is tight but not bitwise."""
        device = torch.device("cuda")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        logits, targets_u8, valid = _make_semantic_inputs(
            device, target_dtype=torch.uint8
        )
        module = FusedCameraLosses()

        logits_u8 = logits.detach().clone().requires_grad_(True)
        *_, loss_u8 = module(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=logits_u8,
            semantic_targets=targets_u8,
            semantic_valid=valid,
        )
        loss_u8.backward()

        logits_i64 = logits.detach().clone().requires_grad_(True)
        *_, loss_i64 = module(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=logits_i64,
            semantic_targets=targets_u8.long(),
            semantic_valid=valid,
        )
        loss_i64.backward()

        assert torch.allclose(loss_i64, loss_u8, rtol=1e-6, atol=1e-7)
        assert torch.allclose(logits_i64.grad, logits_u8.grad, rtol=1e-6, atol=1e-7)

    def test_camera_outputs_unchanged_by_semantic_member(self):
        """Member-disabled vs member-enabled: enabling the CE member must not
        perturb the camera members' outputs or gradients (the folded kernels
        share only the host entry), and omitting it keeps two outputs."""
        device = torch.device("cuda")
        f1, rp1, rg1, bp1 = _make_camera_inputs(device, requires_grad=True)
        logits, targets, valid = _make_semantic_inputs(device, n=384)
        rl, bl, _ = FusedCameraLosses()(
            f1,
            rp1,
            rg1,
            bp1,
            semantic_logits=logits,
            semantic_targets=targets,
            semantic_valid=valid,
        )
        (rl.sum() + bl.sum()).backward()

        f2, rp2, rg2, bp2 = _make_camera_inputs(device, requires_grad=True)
        outputs = FusedCameraLosses()(f2, rp2, rg2, bp2)
        assert len(outputs) == 2
        rl2, bl2 = outputs
        (rl2.sum() + bl2.sum()).backward()

        assert torch.equal(rl, rl2)
        assert torch.equal(bl, bl2)
        assert torch.equal(rp1.grad, rp2.grad)
        assert torch.equal(bp1.grad, bp2.grad)

    def test_ce_loss_unused_in_graph_contributes_no_grad(self):
        """Backward with only the camera losses in the graph: the CE member's
        cotangent is None, its backward launch is skipped, and the logits get
        no gradient contribution at all — even when they carry non-finite
        values (an unused member must never poison a finite sibling)."""
        device = torch.device("cuda")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(
            device, requires_grad=True
        )
        logits, targets, valid = _make_semantic_inputs(
            device, n=384, requires_grad=True
        )
        with torch.no_grad():
            logits[7] = float("nan")  # unused member: must stay inert
        rl, bl, _ = FusedCameraLosses()(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=logits,
            semantic_targets=targets,
            semantic_valid=valid,
        )
        (rl.sum() + bl.sum()).backward()
        assert logits.grad is None
        assert rgb_pred.grad is not None and torch.isfinite(rgb_pred.grad).all()
        assert bg_pred.grad is not None and torch.isfinite(bg_pred.grad).all()

    def test_mixed_enable_camera_disabled_semantic_active(self):
        """Camera factors < 0 disable the camera members while the CE member
        still runs from the same grouped entry."""
        device = torch.device("cuda")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        logits, targets, valid = _make_semantic_inputs(
            device, n=384, n_ignored=16, requires_grad=True
        )
        rl, bl, ce = FusedCameraLosses()(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            rgb_factor=-1.0,
            bg_factor=-1.0,
            semantic_logits=logits,
            semantic_targets=targets,
            semantic_valid=valid,
        )
        assert (rl == 0).all()
        assert (bl == 0).all()

        l2 = logits.detach().clone().requires_grad_(True)
        ce_ref = semantic_cross_entropy_masked(l2, targets, valid)
        assert torch.allclose(ce, ce_ref, atol=1e-6)

        ce.backward()
        ce_ref.backward()
        assert torch.allclose(logits.grad, l2.grad, rtol=1e-5, atol=1e-8)

    def test_ignore_index_denominator_contract(self):
        """-100 on a valid row through the grouped op (int64 only): zero
        numerator contribution, row counted in the denominator, exact-zero
        logits-grad row."""
        device = torch.device("cuda")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        torch.manual_seed(3)
        n, c = 64, C
        logits = torch.randn(n, c, device=device, requires_grad=True)
        targets = torch.randint(0, c, (n,), device=device, dtype=torch.int64)
        valid = torch.ones(n, dtype=torch.bool, device=device)
        valid[3] = False
        targets[5] = IGNORE_INDEX

        *_, ce = FusedCameraLosses()(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=logits,
            semantic_targets=targets,
            semantic_valid=valid,
        )

        contributing = valid.clone()
        contributing[5] = False
        n_valid = n - 1  # every valid row, including the ignored one
        expected = (
            torch.nn.functional.cross_entropy(
                logits.detach()[contributing],
                targets[contributing],
                reduction="sum",
            )
            / n_valid
        )
        assert torch.allclose(ce, expected, rtol=1e-5, atol=1e-6)

        ce.backward()
        assert logits.grad[5].abs().max().item() == 0.0, "ignored row grad"
        assert logits.grad[3].abs().max().item() == 0.0, "invalid row grad"

    @pytest.mark.parametrize(
        "ignore_index,target_dtype",
        [
            (4, torch.uint8),  # inside [0, C): also expressible as uint8
            (4, torch.int64),  # inside [0, C): ignore takes precedence
            (200, torch.int64),  # outside [0, C): pure sentinel value
        ],
    )
    def test_custom_ignore_index_matches_fallback(self, ignore_index, target_dtype):
        """Non-default ignore_index through the grouped CUDA op: parity with
        the pure-PyTorch fallback, forward and backward. Covers a sentinel
        inside [0, C) — which then names a class index, with ignore taking
        precedence over the class meaning — and one outside the range.
        Ignored rows get exact-zero gradient rows while still counting in the
        denominator (the reference parity pins the denominator)."""
        device = torch.device("cuda")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        logits, targets, valid = _make_semantic_inputs(
            device, n=384, target_dtype=target_dtype, requires_grad=True
        )
        valid_rows = valid.nonzero(as_tuple=True)[0]
        targets[valid_rows[:24]] = ignore_index

        *_, ce = FusedCameraLosses()(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=logits,
            semantic_targets=targets,
            semantic_valid=valid,
            semantic_ignore_index=ignore_index,
        )
        ce.backward()

        ref_logits = logits.detach().clone().requires_grad_(True)
        ce_ref = semantic_cross_entropy_masked(
            ref_logits, targets, valid, ignore_index=ignore_index
        )
        ce_ref.backward()

        assert torch.allclose(ce, ce_ref, **self._tols(torch.float32))
        assert torch.allclose(
            logits.grad, ref_logits.grad, **self._tols(torch.float32, grad=True)
        )
        ignored = valid & (targets.long() == ignore_index)
        assert int(ignored.sum()) >= 24
        assert logits.grad[ignored].abs().max().item() == 0.0

    def test_grouped_matches_fallback(self):
        """CUDA grouped path against the pure-PyTorch grouped composition
        (forward + backward), covering all three loss members at once."""
        device = torch.device("cuda")
        rf, bf = 0.5, 0.7

        f1, rp1, rg1, bp1 = _make_camera_inputs(device, requires_grad=True)
        l1, tg, vd = _make_semantic_inputs(
            device, n=384, n_ignored=16, requires_grad=True
        )
        rl, bl, ce = FusedCameraLosses()(
            f1,
            rp1,
            rg1,
            bp1,
            rf,
            bf,
            semantic_logits=l1,
            semantic_targets=tg,
            semantic_valid=vd,
        )
        (rl.sum() + bl.sum() + ce).backward()

        f2, rp2, rg2, bp2 = _make_camera_inputs(device, requires_grad=True)
        l2 = l1.detach().clone().requires_grad_(True)
        module_ref = FusedCameraLosses()
        module_ref._cuda_available = False  # force the pure-PyTorch path
        rl_ref, bl_ref, ce_ref = module_ref(
            f2,
            rp2,
            rg2,
            bp2,
            rf,
            bf,
            semantic_logits=l2,
            semantic_targets=tg,
            semantic_valid=vd,
        )
        (rl_ref.sum() + bl_ref.sum() + ce_ref).backward()

        assert torch.allclose(rl, rl_ref, atol=1e-5)
        assert torch.allclose(bl, bl_ref, atol=1e-5)
        assert torch.allclose(ce, ce_ref, atol=1e-5)
        assert torch.allclose(rp1.grad, rp2.grad, atol=1e-4)
        assert torch.allclose(bp1.grad, bp2.grad, atol=1e-4)
        assert torch.allclose(l1.grad, l2.grad, atol=1e-5)

    def test_semantic_rows_independent_of_camera_rays(self):
        """The CE domain is row-masked, not flag-masked: its row count need
        not match the camera ray count (including M == 0, which skips the
        accumulate kernel but still finalizes a zero loss)."""
        device = torch.device("cuda")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device, n=512)
        for m in (0, 1, 33, 1024):
            logits, targets, valid = _make_semantic_inputs(device, n=m)
            rl, bl, ce = FusedCameraLosses()(
                flags,
                rgb_pred,
                rgb_gt,
                bg_pred,
                semantic_logits=logits,
                semantic_targets=targets,
                semantic_valid=valid,
            )
            ce_ref = semantic_cross_entropy_masked(logits, targets, valid)
            assert rl.shape == (512,)
            assert bl.shape == (512,)
            assert torch.allclose(ce, ce_ref, atol=1e-6)

    def test_m_edge_cases_backward(self):
        """m in (0, 1) through the grouped op with backward: the gradient
        keeps the (m, C) shape; with valid_count == 0 (m == 0, or the single
        row masked out) the finalize kernel and the backward count guard
        yield an exact-zero loss and exact-zero gradients; a single
        contributing row matches the reference."""
        device = torch.device("cuda")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        module = FusedCameraLosses()

        def _grouped_ce(logits, targets, valid):
            *_, ce = module(
                flags,
                rgb_pred,
                rgb_gt,
                bg_pred,
                semantic_logits=logits,
                semantic_targets=targets,
                semantic_valid=valid,
            )
            return ce

        # m == 0: the accumulate kernel is skipped entirely, the finalize
        # kernel still writes an exact zero from the zeroed workspace, and
        # backward produces an empty (0, C) gradient.
        logits = torch.empty(0, C, device=device, requires_grad=True)
        targets = torch.empty(0, dtype=torch.int64, device=device)
        valid = torch.empty(0, dtype=torch.bool, device=device)
        ce = _grouped_ce(logits, targets, valid)
        assert ce.shape == () and ce.item() == 0.0
        ce.backward()
        assert logits.grad is not None and logits.grad.shape == (0, C)

        # m == 1 with the row masked out: the accumulate kernel runs but
        # flushes nothing (valid_count == 0), so the finalize kernel and the
        # backward count guard both take the count <= 0 path — exact-zero
        # loss and an exact-zero gradient row.
        torch.manual_seed(7)
        logits = torch.randn(1, C, device=device, requires_grad=True)
        targets = torch.randint(0, C, (1,), device=device, dtype=torch.int64)
        valid = torch.zeros(1, dtype=torch.bool, device=device)
        ce = _grouped_ce(logits, targets, valid)
        assert ce.item() == 0.0
        ce.backward()
        assert logits.grad.shape == (1, C)
        assert logits.grad.abs().max().item() == 0.0

        # m == 1 with a single contributing row: forward and backward parity
        # with the reference.
        logits = torch.randn(1, C, device=device, requires_grad=True)
        targets = torch.randint(0, C, (1,), device=device, dtype=torch.int64)
        valid = torch.ones(1, dtype=torch.bool, device=device)
        ce = _grouped_ce(logits, targets, valid)
        ce.backward()
        ref_logits = logits.detach().clone().requires_grad_(True)
        ce_ref = semantic_cross_entropy_masked(ref_logits, targets, valid)
        ce_ref.backward()
        assert torch.allclose(ce, ce_ref, **self._tols(torch.float32))
        assert logits.grad.shape == (1, C)
        assert torch.allclose(
            logits.grad, ref_logits.grad, **self._tols(torch.float32, grad=True)
        )

    def test_structural_dtype_checks_match_fallback(self):
        """The CUDA entry rejects the same structurally invalid inputs the
        fallback rejects unconditionally: float targets and non-bool valid
        masks fail loudly instead of being silently cast."""
        device = torch.device("cuda")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        logits, targets, valid = _make_semantic_inputs(device, n=64)
        module = FusedCameraLosses()
        with pytest.raises(RuntimeError, match="targets must be uint8 or int64"):
            module(
                flags,
                rgb_pred,
                rgb_gt,
                bg_pred,
                semantic_logits=logits,
                semantic_targets=targets.float(),
                semantic_valid=valid,
            )
        with pytest.raises(RuntimeError, match="valid must be"):
            module(
                flags,
                rgb_pred,
                rgb_gt,
                bg_pred,
                semantic_logits=logits,
                semantic_targets=targets,
                semantic_valid=valid.to(torch.uint8),
            )

    def test_out_of_range_target_cuda_contract(self):
        """Out-of-range contributing target through the grouped CUDA op, in a
        subprocess (a fired device-side assert is sticky for the whole CUDA
        context). Assert-enabled builds must fail loudly; assert-free builds
        must deterministically skip the row — numerator only, still counted
        in the denominator, exact-zero gradient row — matching the fallback.
        """
        script = r"""
import torch
from gsplat.losses import semantic_cross_entropy_masked
from gsplat.losses_fused import FusedCameraLosses

device = torch.device("cuda")
torch.manual_seed(0)
n, c = 8, 5
flags = torch.zeros(16, dtype=torch.int32, device=device)
rgb_pred = torch.rand(16, 3, device=device)
rgb_gt = torch.rand(16, 3, device=device)
bg_pred = torch.rand(16, device=device)
logits = torch.randn(n, c, device=device, requires_grad=True)
targets = torch.randint(0, c, (n,), device=device, dtype=torch.int64)
targets[2] = c + 7  # out-of-range on a valid row
valid = torch.ones(n, dtype=torch.bool, device=device)

*_, ce = FusedCameraLosses()(
    flags,
    rgb_pred,
    rgb_gt,
    bg_pred,
    semantic_logits=logits,
    semantic_targets=targets,
    semantic_valid=valid,
)
ce.backward()

# Skip contract: row 2 out of the numerator, still in the denominator.
keep = torch.ones(n, dtype=torch.bool, device=device)
keep[2] = False
expected = (
    torch.nn.functional.cross_entropy(
        logits.detach()[keep], targets[keep], reduction="sum"
    )
    / n
)
assert torch.allclose(ce.detach(), expected, rtol=1e-5, atol=1e-6)
assert logits.grad[2].abs().max().item() == 0.0
assert logits.grad[keep].abs().max().item() > 0.0
# The pure-PyTorch fallback agrees on the same inputs.
ref = semantic_cross_entropy_masked(
    logits.detach().cpu(), targets.cpu(), valid.cpu()
)
assert torch.allclose(ce.detach().cpu(), ref, rtol=1e-5, atol=1e-6)
print("SKIP-CONTRACT-OK")
"""
        env = dict(os.environ)
        # The child exercises the *default* (skip) fallback contract.
        env["GSPLAT_ENFORCE_CONTRACTS"] = "0"
        env.pop("PYTHONOPTIMIZE", None)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        output = completed.stdout + completed.stderr
        if completed.returncode == 0:
            # Assert-free build: the deterministic-skip contract holds.
            assert "SKIP-CONTRACT-OK" in completed.stdout, output
        else:
            # Assert-enabled build: the device assert fired loudly.
            assert "device-side assert" in output or "CUDA error" in output, output

    def test_all_invalid_mask(self):
        """All-invalid mask through the grouped op: clamped denominator gives
        exact zero loss and exact-zero gradients (count == 0 path in both
        kernels)."""
        device = torch.device("cuda")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        logits, targets, _ = _make_semantic_inputs(device, requires_grad=True)
        valid = torch.zeros(N, dtype=torch.bool, device=device)

        *_, ce = FusedCameraLosses()(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=logits,
            semantic_targets=targets,
            semantic_valid=valid,
        )
        assert ce.item() == 0.0
        ce.backward()
        assert torch.count_nonzero(logits.grad) == 0

    def test_noncontiguous_semantic_inputs_match_contiguous(self):
        """Non-contiguous semantic inputs are contiguified at the wrapper
        boundary: same loss and gradients as the contiguous call."""
        device = torch.device("cuda")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        logits, targets, valid = _make_semantic_inputs(device, n=384, n_ignored=16)
        module = FusedCameraLosses()

        base = logits.detach().clone().requires_grad_(True)
        *_, ce = module(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=base,
            semantic_targets=targets,
            semantic_valid=valid,
        )
        ce.backward()

        leaf = logits.detach().clone().requires_grad_(True)
        *_, nc_ce = module(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=_noncontiguous_view(leaf),
            semantic_targets=_noncontiguous_view(targets),
            semantic_valid=_noncontiguous_view(valid),
        )
        nc_ce.backward()

        assert torch.allclose(nc_ce, ce, **self._tols(torch.float32))
        assert torch.allclose(
            leaf.grad, base.grad, **self._tols(torch.float32, grad=True)
        )

    def test_nonfinite_noncontributing_rows_do_not_poison(self):
        """NaN/Inf logits confined to invalid or ignored rows must not leak
        into the CUDA loss or gradients (the kernels predicate every read on
        the contributing mask)."""
        device = torch.device("cuda")
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        logits, targets, valid = _make_semantic_inputs(device, n=384, n_ignored=8)
        clean = semantic_cross_entropy_masked(logits, targets, valid)

        invalid_rows = (~valid).nonzero(as_tuple=True)[0]
        ignored_rows = (valid & (targets == IGNORE_INDEX)).nonzero(as_tuple=True)[0]
        poisoned = logits.detach().clone().requires_grad_(True)
        with torch.no_grad():
            poisoned[invalid_rows[0]] = float("nan")
            poisoned[invalid_rows[1]] = float("inf")
            poisoned[ignored_rows[0]] = float("-inf")

        *_, ce = FusedCameraLosses()(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=poisoned,
            semantic_targets=targets,
            semantic_valid=valid,
        )
        assert torch.isfinite(ce)
        assert torch.allclose(ce, clean, **self._tols(torch.float32))
        ce.backward()
        assert torch.isfinite(poisoned.grad).all()
        assert poisoned.grad[invalid_rows[0]].abs().max().item() == 0.0
        assert poisoned.grad[invalid_rows[1]].abs().max().item() == 0.0
        assert poisoned.grad[ignored_rows[0]].abs().max().item() == 0.0

    def test_production_scale(self):
        """Large, non-multiple-of-256 M exercises the grid-stride loop and the
        block-reduced atomics at realistic row counts, with fwd+bwd parity
        against the reference through the grouped op."""
        device = torch.device("cuda")
        m = 500_003
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device, n=64)
        logits, targets, valid = _make_semantic_inputs(
            device, n=m, n_ignored=1000, requires_grad=True
        )
        *_, ce = FusedCameraLosses()(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            semantic_logits=logits,
            semantic_targets=targets,
            semantic_valid=valid,
        )
        ce.backward()

        ref_logits = logits.detach().clone().requires_grad_(True)
        ce_ref = semantic_cross_entropy_masked(ref_logits, targets, valid)
        ce_ref.backward()

        assert ce.shape == () and torch.isfinite(ce)
        assert torch.allclose(ce, ce_ref, rtol=1e-4, atol=1e-5)
        assert torch.allclose(logits.grad, ref_logits.grad, rtol=1e-4, atol=1e-7)


@pytest.mark.skipif(not CAMERA_CUDA, reason="CUDA or fused camera losses not available")
class TestWarpButterflyReductionsCUDA:
    """Pin the shared butterfly reductions (KernelUtils'
    ``warp_reduce_max_all_lanes`` / ``warp_reduce_sum_all_lanes``) through the
    CE kernels, which consume them as one warp per row with lanes striding
    the classes. The class-count sweep walks the lane boundary (idle lanes
    must contribute exactly the identity), and the planted-max /
    planted-target sweeps make each of the 32 lanes the sole owner of the
    reduced value — with 64 classes, on both of its strided classes. A lane
    the xor tree failed to reach fails loudly: a stale row max leaves an
    unshifted ``exp`` in the sum, and a lost target logit moves that row's
    loss by the plant's magnitude."""

    def _ce(self, logits, targets, valid):
        device = logits.device
        flags, rgb_pred, rgb_gt, bg_pred = _make_camera_inputs(device)
        _, _, ce = FusedCameraLosses()(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            rgb_factor=-1.0,
            bg_factor=-1.0,
            semantic_logits=logits,
            semantic_targets=targets,
            semantic_valid=valid,
        )
        return ce

    def _assert_matches_reference(self, logits, targets, valid):
        l_ref = logits.detach().clone().requires_grad_(True)
        ce = self._ce(logits, targets, valid)
        ce.backward()
        ce_ref = semantic_cross_entropy_masked(l_ref, targets, valid)
        ce_ref.backward()
        assert torch.allclose(
            ce, ce_ref, rtol=1e-5, atol=1e-6
        ), f"forward max diff: {(ce - ce_ref).abs().max()}"
        assert torch.allclose(
            logits.grad, l_ref.grad, rtol=1e-5, atol=1e-8
        ), f"grad max diff: {(logits.grad - l_ref.grad).abs().max()}"

    @pytest.mark.parametrize("n_classes", [1, 31, 32, 33, 63, 64, 65])
    def test_class_counts_across_lane_boundaries(self, n_classes):
        """257 rows x swept class counts: rows cross the warp-per-row block
        and grid-stride boundaries while classes cross the lane stride, so
        the idle lanes' identity contributions (0 for the sums, lowest() for
        the max) pass through the butterfly on every boundary shape."""
        device = torch.device("cuda")
        torch.manual_seed(100 + n_classes)
        n_rows = 257
        logits = torch.randn(n_rows, n_classes, device=device, requires_grad=True)
        targets = torch.randint(0, n_classes, (n_rows,), device=device)
        valid = torch.ones(n_rows, dtype=torch.bool, device=device)
        self._assert_matches_reference(logits, targets, valid)

    def test_row_max_owned_by_every_lane(self):
        """64 rows x 64 classes; row i plants a +60 spike at class i, so every
        lane owns the row max once per class stride (classes i and i + 32).
        A lane left with a stale max would exponentiate an unshifted logit
        and blow the row loss past any tolerance."""
        device = torch.device("cuda")
        torch.manual_seed(7)
        n = 64
        logits = torch.randn(n, n, device=device)
        logits[torch.arange(n), torch.arange(n)] += 60.0
        logits.requires_grad_(True)
        targets = torch.randint(0, n, (n,), device=device)
        valid = torch.ones(n, dtype=torch.bool, device=device)
        self._assert_matches_reference(logits, targets, valid)

    def test_target_logit_owned_by_every_lane(self):
        """64 rows x 64 classes with row i's target at class i: the "exactly
        one lane saw the target; summing broadcasts it" property is exercised
        with every lane as that owner, on both of its strided classes. A
        dropped broadcast zeroes that row's (deliberately large) target logit
        and shifts the row loss by its magnitude."""
        device = torch.device("cuda")
        torch.manual_seed(11)
        n = 64
        logits = torch.randn(n, n, device=device)
        logits[torch.arange(n), torch.arange(n)] += 12.0
        logits.requires_grad_(True)
        targets = torch.arange(n, device=device)
        valid = torch.ones(n, dtype=torch.bool, device=device)
        self._assert_matches_reference(logits, targets, valid)
