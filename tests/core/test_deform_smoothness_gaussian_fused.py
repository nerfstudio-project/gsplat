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

"""Tests for the deform-smoothness member of the fused gaussian-loss dispatch.

The deform-smoothness loss rides along the fused gaussian dispatch: one
``gaussian_losses_fwd``/``gaussian_losses_bwd`` op call
launches the four per-gaussian regularizer kernels plus, when the optional
``deformation``/``deform_mask`` inputs are present, the deform-smoothness
kernels.

The pure-PyTorch reference (``deform_smoothness_loss``) is first validated
against hand-computed values and closed-form gradient formulas (CPU, always
runnable); the grouped CUDA member is then compared against that reference
for the forward value and the backward gradients w.r.t. deformation and mask
on every branch (masked / broadcast-masked / scalar-masked / unmasked,
clamped / unclamped denominator).
"""

import pytest
import torch

from gsplat import has_losses
from gsplat.losses import (
    deform_smoothness_loss,
    gaussian_density_reg,
    gaussian_scale_reg,
    gaussian_z_scale_reg,
    out_of_bound_loss,
)
from gsplat.losses_fused import FusedGaussianLosses

CUDA_AVAILABLE = torch.cuda.is_available() and has_losses()

N = 1024

# Deform-smoothness fixture data: mixed
# signs in the deformation, a fractional/zero/greater-than-one per-point mask.
DEFORMATION_BASE = [
    [-1.4, -0.3, 0.2],
    [0.7, -0.9, 1.1],
    [2.0, -1.6, 0.4],
    [-0.8, 0.6, -2.3],
]
MASK_BASE = [[1.0], [0.25], [0.0], [1.5]]
# Hand-computed: numerator = 1.9 * 1.0 + 2.7 * 0.25 + 4.0 * 0.0 + 3.7 * 1.5,
# denominator = max(1.0 + 0.25 + 0.0 + 1.5, 1) = 2.75.
EXPECTED_MASKED_LOSS = 8.125 / 2.75


def _small_inputs(device, dtype=torch.float32, requires_grad=False):
    deformation = torch.tensor(DEFORMATION_BASE, device=device, dtype=dtype)
    mask = torch.tensor(MASK_BASE, device=device, dtype=dtype)
    if requires_grad:
        deformation.requires_grad_(True)
        mask.requires_grad_(True)
    return deformation, mask


def _random_deform_inputs(
    device,
    n=1024,
    d=3,
    dtype=torch.float32,
    mask_shape="broadcast",
    requires_grad=False,
):
    """Random deformation plus a mask with zeros, fractions, and values > 1."""
    torch.manual_seed(7)
    deformation = torch.randn(n, d, device=device, dtype=dtype)
    if mask_shape == "broadcast":
        mask = torch.rand(n, 1, device=device, dtype=dtype) * 1.5
    elif mask_shape == "full":
        mask = torch.rand(n, d, device=device, dtype=dtype) * 1.5
    elif mask_shape == "scalar":
        mask = torch.rand((), device=device, dtype=dtype) + 0.5
    elif mask_shape == "none":
        mask = None
    else:
        raise AssertionError(mask_shape)
    if mask is not None and mask.numel() > 1:
        # Exact zeros exercise masked-out elements.
        mask[::3] = 0.0
    if requires_grad:
        deformation.requires_grad_(True)
        if mask is not None:
            mask.requires_grad_(True)
    return deformation, mask


def _closed_form_grads(deformation, mask, upstream=1.0):
    """Closed-form gradients of the deform-smoothness loss.

    grad_deformation = up * mask~ * sign(d) / denom
    grad_mask        = sum-over-broadcast([mask~ != 0] * up * |d|) / denom
                       - [mask_sum >= 1] * up * loss_sum / denom**2
    with denom = max(mask_sum, 1) (mask_sum = numel without a mask).
    Masked-out elements (mask~ == 0) are fully inert: their |d| reaches
    neither the loss sum nor the mask-gradient numerator.
    """
    d = deformation.detach()
    if mask is None:
        denom = max(float(d.numel()), 1.0)
        return upstream * torch.sign(d) / denom, None
    m = mask.detach()
    mask_sum = m.sum()
    denom = mask_sum.clamp(min=1)
    mask_b = torch.broadcast_to(m, d.shape)
    inert = mask_b == 0
    g_d = upstream * mask_b * torch.sign(d) / denom
    g_d = torch.where(inert, torch.zeros_like(g_d), g_d)
    numerator_term = (
        upstream * torch.where(inert, torch.zeros_like(d), d.abs()) / denom
    ).sum_to_size(m.shape)
    loss_sum = torch.where(inert, torch.zeros_like(d), d.abs() * mask_b).sum()
    denominator_term = torch.where(
        mask_sum >= 1,
        -upstream * loss_sum / (denom * denom),
        torch.zeros((), device=d.device, dtype=d.dtype),
    )
    return g_d, numerator_term + denominator_term


def _noncontiguous_view(t):
    """The same values as ``t`` seen through a stride-doubled view."""
    view = torch.repeat_interleave(t.unsqueeze(-1), 2, dim=-1)[..., 0]
    assert not view.is_contiguous()
    return view


def _make_gaussian_inputs(device, n=N, requires_grad=False):
    """Random gaussian reg inputs (mirrors tests/test_losses_fused.py)."""
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


def _tols(dtype):
    """Kernel-vs-reference tolerances: the deform reduction accumulates its
    running sums via float atomicAdd (order nondeterministic), so fp32 parity
    is tight but not bitwise; fp64 atomics keep ~1e-12 slack."""
    return (
        {"rtol": 1e-5, "atol": 1e-6}
        if dtype == torch.float32
        else {"rtol": 1e-10, "atol": 1e-12}
    )


# ---------------------------------------------------------------------------
# Pure-PyTorch reference tests — always runnable, validate the reference itself
# ---------------------------------------------------------------------------


class TestDeformSmoothnessReference:
    """Validate the pure-PyTorch reference against hand-computed math."""

    def test_forward_matches_hand_computed(self):
        deformation, mask = _small_inputs(torch.device("cpu"))
        loss = deform_smoothness_loss(deformation, mask)
        assert loss.shape == ()
        assert torch.allclose(loss, torch.tensor(EXPECTED_MASKED_LOSS))

    def test_forward_no_mask_is_mean_abs(self):
        deformation, _ = _small_inputs(torch.device("cpu"))
        loss = deform_smoothness_loss(deformation)
        assert torch.allclose(loss, deformation.abs().mean())

    def test_backward_matches_closed_form(self):
        deformation, mask = _small_inputs(torch.device("cpu"), requires_grad=True)
        loss = deform_smoothness_loss(deformation, mask)
        loss.backward()
        expected_d, expected_m = _closed_form_grads(deformation, mask)
        assert torch.allclose(deformation.grad, expected_d)
        assert torch.allclose(mask.grad, expected_m)
        # Denominator (mask_sum = 2.75 >= 1) term must be active, while the
        # masked-out row's numerator is inert: the row with mask 0 receives
        # ONLY the denominator term -loss_sum / denom^2 (= -8.125/2.75**2),
        # not its own |d|.sum() / denom.
        expected_zero_mask_row = -8.125 / 2.75**2
        assert abs(mask.grad[2].item() - expected_zero_mask_row) < 1e-6

    def test_backward_no_mask_matches_closed_form(self):
        deformation, _ = _small_inputs(torch.device("cpu"), requires_grad=True)
        loss = deform_smoothness_loss(deformation)
        loss.backward()
        expected_d, _ = _closed_form_grads(deformation, None)
        assert torch.allclose(deformation.grad, expected_d)

    def test_zero_mask_clamps_denominator_and_suppresses_its_gradient(self):
        """All-zero mask: mask_sum = 0 < 1 clamps the denominator to 1, so the
        loss is an exact zero and deformation gradients vanish (weight 0). The
        mask gradient is exactly zero everywhere: masked-out lanes are inert
        in the numerator, and the clamp suppresses the denominator term."""
        deformation, _ = _small_inputs(torch.device("cpu"), requires_grad=True)
        mask = torch.zeros(4, 1, requires_grad=True)
        loss = deform_smoothness_loss(deformation, mask)
        assert loss.item() == 0.0
        loss.backward()
        assert torch.count_nonzero(deformation.grad) == 0
        assert torch.count_nonzero(mask.grad) == 0

    def test_masked_out_nonfinite_deformation_is_inert(self):
        """inf/nan residuals confined to masked-out (mask == 0) rows must
        reach neither the loss nor any gradient — masking a region is often
        exactly why its residuals are left as garbage. The results match the
        same inputs with finite values in the masked-out rows."""
        deformation, mask = _small_inputs(torch.device("cpu"), requires_grad=True)
        poisoned = deformation.detach().clone()
        poisoned[2, 0] = float("inf")  # row 2 carries mask 0
        poisoned[2, 1] = float("nan")
        poisoned.requires_grad_(True)

        loss = deform_smoothness_loss(poisoned, mask)
        assert torch.isfinite(loss)
        assert torch.allclose(
            loss, deform_smoothness_loss(deformation.detach(), mask.detach())
        )

        loss.backward()
        assert torch.isfinite(poisoned.grad).all()
        assert torch.count_nonzero(poisoned.grad[2]) == 0
        assert torch.isfinite(mask.grad).all()
        expected_d, expected_m = _closed_form_grads(poisoned, mask)
        assert torch.allclose(poisoned.grad, expected_d)
        assert torch.allclose(mask.grad, expected_m)

    def test_sub_unit_mask_sum_clamps(self):
        """A nonzero mask summing below 1 still clamps: denominator = 1 (the
        loss is the plain weighted sum) and the mask's denominator gradient
        term stays suppressed."""
        deformation, _ = _small_inputs(torch.device("cpu"), requires_grad=True)
        mask = torch.tensor([[0.2], [0.0], [0.3], [0.0]], requires_grad=True)
        loss = deform_smoothness_loss(deformation, mask)
        expected = 1.9 * 0.2 + 4.0 * 0.3  # numerator / 1
        assert torch.allclose(loss, torch.tensor(expected))
        loss.backward()
        expected_d, expected_m = _closed_form_grads(deformation, mask)
        assert torch.allclose(deformation.grad, expected_d)
        assert torch.allclose(mask.grad, expected_m)
        # Suppressed denominator term: the numerator-only gradient is >= 0.
        assert (mask.grad >= 0).all()

    def test_empty_inputs(self):
        """N == 0 must yield a finite differentiable zero (not 0/0 = NaN),
        with and without a mask."""
        deformation = torch.empty(0, 3, requires_grad=True)
        mask = torch.empty(0, 1, requires_grad=True)
        loss = deform_smoothness_loss(deformation, mask)
        assert torch.isfinite(loss) and loss.item() == 0.0
        loss.backward()
        assert deformation.grad.shape == (0, 3)
        assert mask.grad.shape == (0, 1)

        deformation2 = torch.empty(0, 3, requires_grad=True)
        loss2 = deform_smoothness_loss(deformation2)
        assert torch.isfinite(loss2) and loss2.item() == 0.0
        loss2.backward()
        assert deformation2.grad.shape == (0, 3)

    def test_mask_expanding_deformation_raises(self):
        """One-way broadcast contract: a mask that would expand deformation is
        rejected (it would change the numerator element count)."""
        deformation = torch.randn(4, 1)
        mask = torch.rand(4, 3)
        with pytest.raises(ValueError, match="broadcast"):
            deform_smoothness_loss(deformation, mask)


# ---------------------------------------------------------------------------
# Grouped member — CPU fallback path (always runnable)
# ---------------------------------------------------------------------------


class TestFusedGaussianDeformGroupedFallback:
    """Grouped-mode behavior of FusedGaussianLosses on the pure-PyTorch path."""

    def test_grouped_deform_matches_composed_references(self):
        """Grouped fallback = composing the existing torch references: the
        gaussian outputs are unchanged and the appended fifth output equals
        deform_smoothness_loss, forward and backward."""
        device = torch.device("cpu")
        scales, densities, z_scales, positions, cuboid_dims = _make_gaussian_inputs(
            device, requires_grad=True
        )
        deformation, mask = _random_deform_inputs(device, n=257, requires_grad=True)
        threshold = 0.5

        module = FusedGaussianLosses(z_scale_threshold=threshold)
        outputs = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            deformation=deformation,
            deform_mask=mask,
        )
        assert len(outputs) == 5
        ls, ld, lz, lo, dl = outputs

        assert torch.allclose(ls, gaussian_scale_reg(scales))
        assert torch.allclose(ld, gaussian_density_reg(densities))
        assert torch.allclose(lz, gaussian_z_scale_reg(z_scales, threshold))
        assert torch.allclose(lo, out_of_bound_loss(positions, cuboid_dims))
        assert dl.shape == ()
        assert torch.allclose(
            dl, deform_smoothness_loss(deformation.detach(), mask.detach())
        )

        (ls.sum() + ld.sum() + lz.sum() + lo.sum() + dl).backward()
        assert scales.grad is not None
        assert deformation.grad is not None
        assert mask.grad is not None

    def test_grouped_deform_without_mask(self):
        device = torch.device("cpu")
        scales, densities, z_scales, positions, cuboid_dims = _make_gaussian_inputs(
            device
        )
        deformation, _ = _random_deform_inputs(device, n=257, mask_shape="none")
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        *_, dl = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            deformation=deformation,
        )
        assert torch.allclose(dl, deform_smoothness_loss(deformation))

    def test_absent_deform_returns_four_tuple(self):
        """Member-disabled case: omitting the deform inputs keeps the
        pre-fold four-tuple behavior."""
        device = torch.device("cpu")
        scales, densities, z_scales, positions, cuboid_dims = _make_gaussian_inputs(
            device
        )
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        outputs = module(scales, densities, z_scales, positions, cuboid_dims)
        assert len(outputs) == 4

    def test_deform_mask_without_deformation_raises(self):
        device = torch.device("cpu")
        scales, densities, z_scales, positions, cuboid_dims = _make_gaussian_inputs(
            device
        )
        mask = torch.rand(8, 1, device=device)
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        with pytest.raises(ValueError, match="deform_mask requires deformation"):
            module(
                scales,
                densities,
                z_scales,
                positions,
                cuboid_dims,
                deform_mask=mask,
            )

    def test_non_2d_deformation_rejected(self):
        """The CUDA kernels require [N, D] while the pure-PyTorch reference is
        shape-agnostic; the module boundary validates dim == 2 before dispatch
        so both paths reject the same shapes with the same clear error."""
        device = torch.device("cpu")
        gaussians = _make_gaussian_inputs(device)
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        for bad_shape in ((12,), (4, 3, 2)):
            with pytest.raises(ValueError, match=r"deformation must be 2-D \[N, D\]"):
                module(*gaussians, deformation=torch.randn(*bad_shape, device=device))

    def test_noncontiguous_deform_inputs_match_contiguous(self):
        """Non-contiguous deformation/mask views reproduce the contiguous
        fallback call bitwise, forward and backward (CPU vs CPU)."""
        device = torch.device("cpu")
        gaussians = _make_gaussian_inputs(device)
        base_d, base_m = _random_deform_inputs(device, n=257, requires_grad=True)
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        *_, dl = module(*gaussians, deformation=base_d, deform_mask=base_m)
        dl.backward()

        leaf_d = base_d.detach().clone().requires_grad_(True)
        leaf_m = base_m.detach().clone().requires_grad_(True)
        *_, nc_dl = module(
            *gaussians,
            deformation=_noncontiguous_view(leaf_d),
            deform_mask=_noncontiguous_view(leaf_m),
        )
        nc_dl.backward()

        assert torch.equal(nc_dl, dl)
        assert torch.equal(leaf_d.grad, base_d.grad)
        assert torch.equal(leaf_m.grad, base_m.grad)

    def test_nonfinite_unused_deform_does_not_poison_gaussian_grads(self):
        """A present-but-unused deform member carrying non-finite values must
        not leak into the gaussian outputs' gradients, and must leave the
        deform inputs without a (NaN) gradient of their own."""
        device = torch.device("cpu")
        s1, d1, z1, p1, c1 = _make_gaussian_inputs(device, requires_grad=True)
        deformation = torch.full(
            (16, 3), float("nan"), device=device, requires_grad=True
        )
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        ls, ld, lz, lo, _ = module(s1, d1, z1, p1, c1, deformation=deformation)
        (ls.sum() + ld.sum() + lz.sum() + lo.sum()).backward()
        for grad in (s1.grad, d1.grad, z1.grad, p1.grad):
            assert grad is not None and torch.isfinite(grad).all()
        assert deformation.grad is None


# ---------------------------------------------------------------------------
# Grouped member — CUDA path (one op call launching gaussian + deform kernels)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA or fused losses not available")
class TestFusedGaussianDeformGroupedCUDA:
    """Grouped dispatch: the gaussian entry launches the
    deform-smoothness kernels when the optional inputs are present.

    Asserts the folded member reproduces the pure-PyTorch reference
    (deform_smoothness_loss) forward and backward, unchanged gaussian results
    when the member is enabled, and mixed-enable combinations (with/without
    mask, with/without visibility, empty gaussian work)."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("mask_shape", ["broadcast", "full", "scalar", "none"])
    def test_grouped_forward_matches_pytorch(self, dtype, mask_shape):
        """The gaussian tensors stay fp32 while the deform member runs in
        ``dtype`` — the two domains dispatch independently."""
        device = torch.device("cuda")
        scales, densities, z_scales, positions, cuboid_dims = _make_gaussian_inputs(
            device
        )
        deformation, mask = _random_deform_inputs(
            device, dtype=dtype, mask_shape=mask_shape
        )
        threshold = 0.5

        module = FusedGaussianLosses(z_scale_threshold=threshold)
        outputs = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            deformation=deformation,
            deform_mask=mask,
        )
        assert len(outputs) == 5
        ls, ld, lz, lo, dl = outputs

        # Folded member == the pure-PyTorch reference.
        ref_dl = deform_smoothness_loss(deformation, mask)
        assert dl.shape == () and dl.dtype == dtype
        assert torch.allclose(
            dl, ref_dl, **_tols(dtype)
        ), f"deform loss diff: {(dl - ref_dl).abs().max()}"

        # Original losses unchanged by the extra group member: bit-equal to an
        # ungrouped call of the same module on the same inputs.
        ref_ls, ref_ld, ref_lz, ref_lo = module(
            scales, densities, z_scales, positions, cuboid_dims
        )
        assert torch.equal(ls, ref_ls)
        assert torch.equal(ld, ref_ld)
        assert torch.equal(lz, ref_lz)
        assert torch.equal(lo, ref_lo)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("mask_shape", ["broadcast", "full", "scalar", "none"])
    @pytest.mark.parametrize("with_visibility", [False, True])
    def test_grouped_backward_matches_pytorch(self, dtype, mask_shape, with_visibility):
        device = torch.device("cuda")
        threshold = 0.5
        upstream = 1.7  # non-unit weight exercises the v_deform_loss scaling
        visibility = None
        if with_visibility:
            torch.manual_seed(3)
            visibility = torch.randint(0, 2, (N,), dtype=torch.float32, device=device)

        # --- grouped gradients ---
        s1, d1, z1, p1, c1 = _make_gaussian_inputs(device, requires_grad=True)
        def1, m1 = _random_deform_inputs(
            device, dtype=dtype, mask_shape=mask_shape, requires_grad=True
        )
        module = FusedGaussianLosses(z_scale_threshold=threshold)
        ls, ld, lz, lo, dl = module(
            s1,
            d1,
            z1,
            p1,
            c1,
            visibility=visibility,
            deformation=def1,
            deform_mask=m1,
        )
        (ls.sum() + ld.sum() + lz.sum() + lo.sum() + upstream * dl).backward()

        # --- ungrouped gaussian gradients (same module, no deform member) ---
        s2, d2, z2, p2, c2 = _make_gaussian_inputs(device, requires_grad=True)
        ls2, ld2, lz2, lo2 = module(s2, d2, z2, p2, c2, visibility=visibility)
        (ls2.sum() + ld2.sum() + lz2.sum() + lo2.sum()).backward()

        # --- pure-PyTorch reference deform gradients ---
        def2, m2 = _random_deform_inputs(
            device, dtype=dtype, mask_shape=mask_shape, requires_grad=True
        )
        (upstream * deform_smoothness_loss(def2, m2)).backward()

        # Gaussian gradients are element-wise deterministic: the deform member
        # must not perturb them at all.
        assert torch.equal(s1.grad, s2.grad)
        assert torch.equal(d1.grad, d2.grad)
        assert torch.equal(z1.grad, z2.grad)
        assert torch.equal(p1.grad, p2.grad)

        # Deform gradients: grouped == reference.
        tols = _tols(dtype)
        assert torch.allclose(
            def1.grad, def2.grad, **tols
        ), f"deformation grad max diff: {(def1.grad - def2.grad).abs().max()}"
        if mask_shape != "none":
            # fp64 mask grads can be ~0; atomic accumulation order differs
            # across GPU archs, so allow a looser absolute tolerance there.
            mask_tols = dict(tols, atol=1e-9) if dtype == torch.float64 else tols
            assert torch.allclose(
                m1.grad, m2.grad, **mask_tols
            ), f"mask grad max diff: {(m1.grad - m2.grad).abs().max()}"

    def test_zero_mask_clamps_denominator_and_suppresses_its_gradient(self):
        """All-zero mask through the gaussian op: exact-zero loss, zero
        deformation gradients, and an exactly-zero mask gradient (masked-out
        lanes are inert in the numerator, the clamp suppresses the
        denominator term) — parity with the reference on the clamped
        branch."""
        device = torch.device("cuda")
        scales, densities, z_scales, positions, cuboid_dims = _make_gaussian_inputs(
            device
        )
        deformation, _ = _small_inputs(device, requires_grad=True)
        mask = torch.zeros(4, 1, device=device, requires_grad=True)
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        *_, dl = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            deformation=deformation,
            deform_mask=mask,
        )
        assert dl.item() == 0.0
        dl.backward()
        assert torch.count_nonzero(deformation.grad) == 0
        assert torch.count_nonzero(mask.grad) == 0

    def test_masked_out_nonfinite_deformation_is_inert(self):
        """CUDA parity for the inert-lane contract: inf/nan residuals in
        masked-out rows reach neither the loss nor any gradient, matching the
        pure-PyTorch reference."""
        device = torch.device("cuda")
        scales, densities, z_scales, positions, cuboid_dims = _make_gaussian_inputs(
            device
        )
        deformation, mask = _small_inputs(device, requires_grad=True)
        poisoned = deformation.detach().clone()
        poisoned[2, 0] = float("inf")  # row 2 carries mask 0
        poisoned[2, 1] = float("nan")
        poisoned.requires_grad_(True)

        module = FusedGaussianLosses(z_scale_threshold=0.5)
        *_, dl = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            deformation=poisoned,
            deform_mask=mask,
        )
        assert torch.isfinite(dl)
        ref = deform_smoothness_loss(deformation.detach(), mask.detach())
        assert torch.allclose(dl, ref, **_tols(torch.float32))

        dl.backward()
        assert torch.isfinite(poisoned.grad).all()
        assert torch.count_nonzero(poisoned.grad[2]) == 0
        assert torch.isfinite(mask.grad).all()
        expected_d, expected_m = _closed_form_grads(poisoned, mask)
        assert torch.allclose(poisoned.grad, expected_d, rtol=1e-5, atol=1e-6)
        assert torch.allclose(mask.grad, expected_m, rtol=1e-5, atol=1e-6)

    def test_partial_mask_matches_hand_computed(self):
        """A partial mask (zeros, fractions, > 1) reproduces the
        hand-computed loss and the closed-form gradients through the grouped
        op."""
        device = torch.device("cuda")
        scales, densities, z_scales, positions, cuboid_dims = _make_gaussian_inputs(
            device
        )
        deformation, mask = _small_inputs(device, requires_grad=True)
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        *_, dl = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            deformation=deformation,
            deform_mask=mask,
        )
        assert torch.allclose(
            dl, torch.tensor(EXPECTED_MASKED_LOSS, device=device), rtol=1e-6
        )
        dl.backward()
        expected_d, expected_m = _closed_form_grads(deformation, mask)
        assert torch.allclose(deformation.grad, expected_d, rtol=1e-5, atol=1e-6)
        assert torch.allclose(mask.grad, expected_m, rtol=1e-5, atol=1e-6)

    def test_empty_gaussians_do_not_skip_deform(self):
        """Empty gaussian work must not skip later group
        members' launches — the deform loss must still be computed when the
        gaussian tensors are empty."""
        device = torch.device("cuda")
        scales = torch.empty(0, 3, device=device)
        densities = torch.empty(0, device=device)
        z_scales = torch.empty(0, device=device)
        positions = torch.empty(0, 3, device=device)
        cuboid_dims = torch.empty(0, 3, device=device)
        deformation, mask = _random_deform_inputs(device, n=257, requires_grad=True)

        module = FusedGaussianLosses(z_scale_threshold=0.5)
        ls, ld, lz, lo, dl = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            deformation=deformation,
            deform_mask=mask,
        )
        assert ls.shape == (0, 3)
        assert ld.shape == (0,)
        assert lz.shape == (0,)
        assert lo.shape == (0, 3)

        ref = deform_smoothness_loss(deformation.detach(), mask.detach())
        assert dl.item() != 0.0
        assert torch.allclose(dl, ref, **_tols(torch.float32))

        dl.backward()
        assert deformation.grad is not None and deformation.grad.shape == (257, 3)
        assert mask.grad is not None and mask.grad.shape == (257, 1)

    def test_empty_deformation_yields_zero_loss(self):
        """Empty deform work beside non-empty gaussian work: the deform
        launchers early-return and the caller-zeroed scalar stands, while the
        gaussian outputs are unaffected."""
        device = torch.device("cuda")
        scales, densities, z_scales, positions, cuboid_dims = _make_gaussian_inputs(
            device
        )
        deformation = torch.empty(0, 3, device=device, requires_grad=True)

        module = FusedGaussianLosses(z_scale_threshold=0.5)
        ls, ld, lz, lo, dl = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            deformation=deformation,
        )
        assert torch.isfinite(dl) and dl.item() == 0.0
        assert torch.equal(ls, gaussian_scale_reg(scales))
        dl.backward()
        assert deformation.grad.shape == (0, 3)

    def test_grouped_deform_only_gaussian_outputs_unused(self):
        """Backward with only the deform output in the graph: the gaussian
        upstream grads are normalized to zeros and the deform gradients match
        the pure-PyTorch reference."""
        device = torch.device("cuda")
        scales, densities, z_scales, positions, cuboid_dims = _make_gaussian_inputs(
            device
        )
        d1, m1 = _random_deform_inputs(device, n=257, requires_grad=True)

        module = FusedGaussianLosses(z_scale_threshold=0.5)
        *_, dl = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            deformation=d1,
            deform_mask=m1,
        )
        dl.backward()

        d2, m2 = _random_deform_inputs(device, n=257, requires_grad=True)
        deform_smoothness_loss(d2, m2).backward()

        tols = _tols(torch.float32)
        assert torch.allclose(d1.grad, d2.grad, **tols)
        assert torch.allclose(m1.grad, m2.grad, **tols)

    def test_production_scale(self):
        """Large, non-multiple-of-256 N exercises the grid-boundary guard and
        the block-reduced atomics at realistic point counts, with full
        fwd+bwd parity against the reference through the grouped op."""
        device = torch.device("cuda")
        n = 500_003
        scales, densities, z_scales, positions, cuboid_dims = _make_gaussian_inputs(
            device, n=64
        )
        d1, m1 = _random_deform_inputs(device, n=n, requires_grad=True)
        deform_smoothness_loss(d1, m1).backward()

        d2, m2 = _random_deform_inputs(device, n=n, requires_grad=True)
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        *_, dl = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            deformation=d2,
            deform_mask=m2,
        )
        dl.backward()

        ref = deform_smoothness_loss(d1.detach(), m1.detach())
        assert dl.shape == () and torch.isfinite(dl)
        assert torch.allclose(dl, ref, rtol=1e-4, atol=1e-5)
        assert torch.allclose(d2.grad, d1.grad, rtol=1e-4, atol=1e-6)
        assert torch.allclose(m2.grad, m1.grad, rtol=1e-4, atol=1e-6)

    def test_noncontiguous_deform_inputs_match_contiguous(self):
        """Non-contiguous deformation/mask are contiguified at the wrapper
        boundary: same loss and gradients as the contiguous call."""
        device = torch.device("cuda")
        gaussians = _make_gaussian_inputs(device)
        base_d, base_m = _random_deform_inputs(device, n=257, requires_grad=True)
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        *_, dl = module(*gaussians, deformation=base_d, deform_mask=base_m)
        dl.backward()

        leaf_d = base_d.detach().clone().requires_grad_(True)
        leaf_m = base_m.detach().clone().requires_grad_(True)
        *_, nc_dl = module(
            *gaussians,
            deformation=_noncontiguous_view(leaf_d),
            deform_mask=_noncontiguous_view(leaf_m),
        )
        nc_dl.backward()

        tols = _tols(torch.float32)
        assert torch.allclose(nc_dl, dl, **tols)
        assert torch.allclose(leaf_d.grad, base_d.grad, **tols)
        assert torch.allclose(leaf_m.grad, base_m.grad, **tols)

    def test_nonfinite_unused_deform_does_not_poison_gaussian_grads(self):
        """A present-but-unused deform member carrying non-finite values must
        not poison the gaussian gradients: its backward launch is skipped when
        its cotangent is absent, and the deform inputs keep no (NaN) gradient
        of their own. The reverse direction holds too: non-finite gaussian
        inputs never leak into the deform member (independent domains)."""
        device = torch.device("cuda")
        s1, d1, z1, p1, c1 = _make_gaussian_inputs(device, requires_grad=True)
        deformation = torch.full(
            (64, 3), float("nan"), device=device, requires_grad=True
        )
        mask = torch.rand(64, 1, device=device)
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        ls, ld, lz, lo, _ = module(
            s1, d1, z1, p1, c1, deformation=deformation, deform_mask=mask
        )
        (ls.sum() + ld.sum() + lz.sum() + lo.sum()).backward()
        for grad in (s1.grad, d1.grad, z1.grad, p1.grad):
            assert grad is not None and torch.isfinite(grad).all()
        assert deformation.grad is None

        # Non-finite gaussian inputs beside a finite, used deform member.
        s2, d2, z2, p2, c2 = _make_gaussian_inputs(device)
        s2 = s2 + float("nan")
        def2, m2 = _random_deform_inputs(device, n=257, requires_grad=True)
        *_, dl2 = module(s2, d2, z2, p2, c2, deformation=def2, deform_mask=m2)
        dl2.backward()
        def_ref, m_ref = _random_deform_inputs(device, n=257, requires_grad=True)
        deform_smoothness_loss(def_ref, m_ref).backward()
        assert torch.isfinite(dl2)
        tols = _tols(torch.float32)
        assert torch.allclose(def2.grad, def_ref.grad, **tols)
        assert torch.allclose(m2.grad, m_ref.grad, **tols)

    def test_non_2d_deformation_rejected_identically_on_both_paths(self):
        """A 1-D deformation must fail before dispatch with the same error on
        the CUDA and fallback paths — previously the fallback silently
        computed a result while the CUDA path raised deep in the native op."""
        device = torch.device("cuda")
        gaussians = _make_gaussian_inputs(device)
        bad = torch.randn(12, device=device)
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        with pytest.raises(
            ValueError, match=r"deformation must be 2-D \[N, D\]"
        ) as cuda_err:
            module(*gaussians, deformation=bad)
        module._cuda_available = False  # force fallback
        with pytest.raises(
            ValueError, match=r"deformation must be 2-D \[N, D\]"
        ) as fallback_err:
            module(*gaussians, deformation=bad)
        assert str(cuda_err.value) == str(fallback_err.value)

    def test_disable_cuda_uses_fallback(self):
        """When the module's CUDA flag is off, CUDA inputs route through the
        pure-PyTorch grouped fallback and match it exactly."""
        device = torch.device("cuda")
        scales, densities, z_scales, positions, cuboid_dims = _make_gaussian_inputs(
            device
        )
        deformation, mask = _random_deform_inputs(device, n=257)
        module = FusedGaussianLosses(z_scale_threshold=0.5)
        module._cuda_available = False  # force fallback
        *_, dl = module(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            deformation=deformation,
            deform_mask=mask,
        )
        ref = deform_smoothness_loss(deformation, mask)
        assert torch.equal(dl, ref)
