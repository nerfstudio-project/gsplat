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

"""Tests for per-member domains and pre-activation inputs of the fused
gaussian losses.

The four regularizer members of ``FusedGaussianLosses`` own independent row
counts (heterogeneous regularization domains): scales ``[N_scales, 3]``,
densities ``[N_densities]``, z_scales ``[N_z_scales]``, positions/cuboid_dims
``[N_oob, 3]`` — any of which may be zero. Pre-activation mode
(``preactivation=True``) switches to log-space member math: ``exp()`` fused
in-kernel for scales/z_scales (per-segment weights folded caller-side as
``+log(w)`` with an exact-zero branch for non-positive weights, see
``fold_log_space_weight``) and ``|.|`` for densities.

Every behavior is validated against an inline pure-torch reference on both
the fallback (CPU) and CUDA paths, including per-member backward gradients,
zero-length members, the folded-weight exact-zero/NaN-safety contract, the
legacy shared-cloud post-activation path (bit-identical, including through
pre-extension positional op calls), and composition with the grouped
deform-smoothness member in both modes.
"""

import pytest
import torch

from tests._cuda import cuda_is_available

from gsplat import has_losses
from gsplat.losses import (
    deform_smoothness_loss,
    fold_log_space_weight,
    gaussian_density_reg,
    gaussian_scale_reg,
    gaussian_z_scale_reg,
    out_of_bound_loss,
)
from gsplat.losses_fused import FusedGaussianLosses

CUDA_AVAILABLE = cuda_is_available() and has_losses()

# Distinct, non-multiple-of-warp member counts so a shared-index or
# shared-count bug cannot hide behind equal domains.
N_SCALES, N_DENSITIES, N_Z_SCALES, N_OOB = 129, 257, 65, 191

THRESHOLD = 0.5


def _make_domain_inputs(
    device,
    n_scales=N_SCALES,
    n_densities=N_DENSITIES,
    n_z_scales=N_Z_SCALES,
    n_oob=N_OOB,
    preactivation=False,
    dtype=torch.float32,
    requires_grad=False,
    seed=42,
):
    """Random per-member inputs. Pre-activation mode draws signed log-space
    scales/z_scales and signed densities (folded weights may flip signs)."""
    torch.manual_seed(seed)
    if preactivation:
        scales = torch.randn(n_scales, 3, device=device, dtype=dtype) * 1.5
        densities = torch.randn(n_densities, device=device, dtype=dtype)
        z_scales = torch.randn(n_z_scales, device=device, dtype=dtype) * 1.5
    else:
        scales = torch.rand(n_scales, 3, device=device, dtype=dtype) * 2.0
        densities = torch.rand(n_densities, device=device, dtype=dtype)
        z_scales = torch.rand(n_z_scales, device=device, dtype=dtype) * 1.5
    positions = (torch.rand(n_oob, 3, device=device, dtype=dtype) - 0.5) * 6.0
    cuboid_dims = torch.rand(n_oob, 3, device=device, dtype=dtype) * 4.0 + 0.5
    if requires_grad:
        for t in (scales, densities, z_scales, positions):
            t.requires_grad_(True)
    return scales, densities, z_scales, positions, cuboid_dims


def _make_visibility(
    device, n_scales=N_SCALES, n_densities=N_DENSITIES, dtype=torch.float32
):
    torch.manual_seed(3)
    n = max(n_scales, n_densities)
    return torch.randint(0, 2, (n,), device=device).to(dtype)


def _reference(
    scales,
    densities,
    z_scales,
    positions,
    cuboid_dims,
    visibility=None,
    preactivation=False,
    threshold=THRESHOLD,
):
    """Inline pure-torch reference for the four members with per-member
    domains (visibility spans the scale and density members)."""
    loss_scale = scales.exp() if preactivation else scales
    loss_density = densities.abs() if preactivation else densities
    if visibility is not None:
        flat = visibility.reshape(-1)
        loss_scale = loss_scale * flat[: scales.shape[0]].view(-1, 1)
        loss_density = loss_density * flat[: densities.shape[0]]
    z_post = z_scales.exp() if preactivation else z_scales
    loss_z_scale = torch.relu(z_post - threshold)
    loss_oob = torch.relu(positions.abs() - cuboid_dims / 2)
    return loss_scale, loss_density, loss_z_scale, loss_oob


def _tols(dtype, preactivation):
    """Post-activation members are pure element-wise arithmetic → bit-exact.
    Pre-activation fp32 goes through the extension's fast-math ``exp`` (a few
    ulp vs torch's); fp64 ``exp`` is unaffected by fast-math but still a
    correctly-rounded-vs-libm comparison, so keep a ~1 ulp allowance."""
    if not preactivation:
        return {"rtol": 0.0, "atol": 0.0}
    if dtype == torch.float32:
        return {"rtol": 1e-5, "atol": 1e-7}
    return {"rtol": 1e-14, "atol": 1e-16}


def _grads(inputs, outputs, upstreams):
    """Backward through ``sum(upstream * output)`` and collect input grads."""
    total = sum((u * o).sum() for u, o in zip(upstreams, outputs))
    total.backward()
    return [t.grad for t in inputs]


def _make_upstreams(outputs, seed=11):
    torch.manual_seed(seed)
    return [torch.rand_like(o) for o in outputs]


# ---------------------------------------------------------------------------
# fold_log_space_weight — always runnable
# ---------------------------------------------------------------------------


class TestFoldLogSpaceWeight:
    """The caller-side folding helper for pre-activation segment weights."""

    def test_positive_weight_matches_post_exp_multiplication(self):
        torch.manual_seed(0)
        x = torch.randn(64, 3, dtype=torch.float64)
        for weight in (0.25, 1.0, 2.0, 7.5):
            folded = fold_log_space_weight(x, weight)
            assert torch.allclose(folded.exp(), weight * x.exp(), rtol=1e-14)

    def test_unit_weight_is_identity(self):
        torch.manual_seed(0)
        x = torch.randn(64, 3)
        assert torch.equal(fold_log_space_weight(x, 1.0), x)

    @pytest.mark.parametrize("weight", [0.0, -1.0])
    def test_nonpositive_weight_takes_exact_zero_branch(self, weight):
        x = torch.randn(16, 3)
        folded = fold_log_space_weight(x, weight)
        assert torch.isinf(folded).all() and (folded < 0).all()
        assert torch.equal(folded.exp(), torch.zeros_like(x))

    def test_zero_weight_is_overflow_safe(self):
        """The reason the zero branch exists: weighting after exp() turns an
        overflowing pre-activation into 0 * inf = NaN; the -inf fill keeps an
        exact zero."""
        x = torch.full((8,), 1000.0)  # exp(1000) overflows to inf in fp32
        assert torch.isnan(0.0 * x.exp()).all()  # the hazard being avoided
        folded = fold_log_space_weight(x, 0.0)
        assert torch.equal(folded.exp(), torch.zeros(8))

    def test_fold_is_invalid_for_thresholded_members(self):
        """The fold contract: +log(w) is exact only for members LINEAR in
        exp(x). A post-exp threshold (the z-scale member) puts the folded
        weight inside the relu — relu(w * exp(z) - T) != w * relu(exp(z) - T)
        — so fractional weights must never be folded into z-scale
        pre-activations; only w == 1 and the exact-zero w <= 0 branch are
        valid there."""
        threshold = 8.0
        z = torch.tensor([10.0], dtype=torch.float64).log()  # exp(z) == 10
        for weight in (0.5, 2.0):
            folded = torch.relu(fold_log_space_weight(z, weight).exp() - threshold)
            intended = weight * torch.relu(z.exp() - threshold)
            assert not torch.allclose(folded, intended)
        # w = 0.5: relu(5 - 8) = 0, where the intended weighted loss is
        # 0.5 * relu(10 - 8) = 1.
        assert torch.relu(fold_log_space_weight(z, 0.5).exp() - threshold).item() == 0.0
        # The two cases that stay valid: identity and the exact-zero branch.
        assert torch.equal(
            torch.relu(fold_log_space_weight(z, 1.0).exp() - threshold),
            torch.relu(z.exp() - threshold),
        )
        assert torch.equal(
            torch.relu(fold_log_space_weight(z, 0.0).exp() - threshold),
            torch.zeros_like(z),
        )

    def test_gradients_scale_with_weight(self):
        x = torch.randn(32, requires_grad=True)
        fold_log_space_weight(x, 2.5).exp().sum().backward()
        assert torch.allclose(x.grad, 2.5 * x.detach().exp(), rtol=1e-6)

    def test_zero_weight_detaches_the_segment(self):
        """A disabled segment contributes no gradient: the fill is a constant
        with no autograd edge back to the pre-activations."""
        x = torch.randn(32, requires_grad=True)
        folded = fold_log_space_weight(x, 0.0)
        assert not folded.requires_grad


# ---------------------------------------------------------------------------
# Per-member domains — fallback (CPU) path, always runnable
# ---------------------------------------------------------------------------


class TestPerMemberDomainsFallback:
    """Independent member row counts through the pure-PyTorch path."""

    def test_distinct_counts_match_reference(self):
        device = torch.device("cpu")
        inputs = _make_domain_inputs(device)
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD)
        outputs = module(*inputs)
        for out, ref in zip(outputs, _reference(*inputs)):
            assert torch.equal(out, ref)
        assert outputs[0].shape == (N_SCALES, 3)
        assert outputs[1].shape == (N_DENSITIES,)
        assert outputs[2].shape == (N_Z_SCALES,)
        assert outputs[3].shape == (N_OOB, 3)

    def test_visibility_spans_scale_and_density_members(self):
        device = torch.device("cpu")
        inputs = _make_domain_inputs(device)
        visibility = _make_visibility(device)
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD)
        outputs = module(*inputs, visibility=visibility)
        for out, ref in zip(outputs, _reference(*inputs, visibility=visibility)):
            assert torch.equal(out, ref)

    def test_wrong_visibility_length_raises(self):
        device = torch.device("cpu")
        inputs = _make_domain_inputs(device)
        for bad_length in (N_SCALES, N_DENSITIES + 1):  # max is N_DENSITIES
            visibility = torch.ones(bad_length, device=device)
            with pytest.raises(ValueError, match="visibility"):
                FusedGaussianLosses(z_scale_threshold=THRESHOLD)(
                    *inputs, visibility=visibility
                )

    @pytest.mark.parametrize("bad", [-0.1, float("nan")])
    def test_invalid_z_scale_threshold_rejected(self, bad):
        """T >= 0 is the public contract: with T < 0, relu(0 - T) > 0, so a
        disabled (w <= 0 folded) or empty z-scale member would no longer be
        an exact zero — and NaN poisons that branch outright. The Python
        guards mirror the native ``T >= 0`` predicate (``not T >= 0`` also
        rejects NaN, which ``T < 0`` would let through), so the module
        rejects at construction (covering both dispatch paths) and the
        standalone functional rejects too; T == 0 (the default) stays
        valid."""
        with pytest.raises(ValueError, match="non-negative"):
            FusedGaussianLosses(z_scale_threshold=bad)
        with pytest.raises(ValueError, match="non-negative"):
            gaussian_z_scale_reg(torch.ones(4), threshold=bad)
        FusedGaussianLosses(z_scale_threshold=0.0)
        gaussian_z_scale_reg(torch.ones(4), threshold=0.0)

    @pytest.mark.parametrize("preactivation", [False, True])
    def test_zero_length_members(self, preactivation):
        """Each member may be empty independently, including all at once."""
        device = torch.device("cpu")
        module = FusedGaussianLosses(
            z_scale_threshold=THRESHOLD, preactivation=preactivation
        )
        for empty in ("scales", "densities", "z_scales", "oob", "all"):
            counts = dict(
                n_scales=N_SCALES,
                n_densities=N_DENSITIES,
                n_z_scales=N_Z_SCALES,
                n_oob=N_OOB,
            )
            if empty == "all":
                counts = {k: 0 for k in counts}
            else:
                counts[f"n_{empty}" if empty != "oob" else "n_oob"] = 0
            inputs = _make_domain_inputs(device, preactivation=preactivation, **counts)
            outputs = module(*inputs)
            for out, ref in zip(
                outputs, _reference(*inputs, preactivation=preactivation)
            ):
                assert out.shape == ref.shape
                assert torch.equal(out, ref)

    def test_preactivation_matches_reference_forward_and_backward(self):
        device = torch.device("cpu")
        s1, d1, z1, p1, c1 = _make_domain_inputs(
            device, preactivation=True, requires_grad=True
        )
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD, preactivation=True)
        outputs = module(s1, d1, z1, p1, c1)
        upstreams = _make_upstreams(outputs)
        grads = _grads((s1, d1, z1, p1), outputs, upstreams)

        s2, d2, z2, p2, c2 = _make_domain_inputs(
            device, preactivation=True, requires_grad=True
        )
        ref_outputs = _reference(s2, d2, z2, p2, c2, preactivation=True)
        ref_grads = _grads((s2, d2, z2, p2), ref_outputs, upstreams)

        for out, ref in zip(outputs, ref_outputs):
            assert torch.allclose(out, ref, rtol=1e-6, atol=1e-7)
        for grad, ref in zip(grads, ref_grads):
            assert torch.allclose(grad, ref, rtol=1e-6, atol=1e-7)

    def test_preactivation_zero_visibility_is_exact_zero_on_overflow(self):
        """Pre-activation + visibility: a zero-visibility lane takes the same
        exact-zero branch as a folded non-positive weight, even when the
        lane's pre-activations overflow exp() — multiplying the visibility
        after the exp() would yield inf * 0 = NaN for value and gradient."""
        device = torch.device("cpu")
        n = 64
        torch.manual_seed(5)
        live = torch.randn(32, 3, device=device)
        overflowing = torch.full((32, 3), 600.0, device=device)  # exp → inf
        assert torch.isnan(overflowing.exp() * 0.0).all()  # the naive hazard
        scales = torch.cat([live, overflowing]).requires_grad_(True)
        _, densities, z_scales, positions, cuboid_dims = _make_domain_inputs(
            device, n_scales=n, n_densities=n, preactivation=True
        )
        visibility = torch.ones(n, device=device)
        visibility[32:] = 0.0
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD, preactivation=True)
        loss_scale, *_ = module(
            scales, densities, z_scales, positions, cuboid_dims, visibility=visibility
        )
        assert torch.isfinite(loss_scale).all()
        assert torch.equal(loss_scale[32:], torch.zeros_like(loss_scale[32:]))
        assert torch.allclose(loss_scale[:32], live.exp(), rtol=1e-6, atol=1e-7)
        loss_scale.sum().backward()
        assert torch.isfinite(scales.grad).all()
        assert torch.equal(scales.grad[32:], torch.zeros_like(scales.grad[32:]))
        assert torch.allclose(scales.grad[:32], live.exp(), rtol=1e-6, atol=1e-7)

    def test_deform_member_composes_with_domains_and_preactivation(self):
        device = torch.device("cpu")
        for preactivation in (False, True):
            inputs = _make_domain_inputs(device, preactivation=preactivation)
            torch.manual_seed(7)
            deformation = torch.randn(33, 3, device=device)
            module = FusedGaussianLosses(
                z_scale_threshold=THRESHOLD, preactivation=preactivation
            )
            *gaussian_losses, dl = module(*inputs, deformation=deformation)
            for out, ref in zip(
                gaussian_losses, _reference(*inputs, preactivation=preactivation)
            ):
                assert torch.equal(out, ref)
            assert torch.allclose(dl, deform_smoothness_loss(deformation))


# ---------------------------------------------------------------------------
# Per-member domains — CUDA path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA or fused losses not available")
class TestPerMemberDomainsCUDA:
    """Independent member row counts through the fused CUDA kernel.

    Post-activation math is pure element-wise arithmetic, so parity with the
    inline torch reference is asserted bit-exactly."""

    @pytest.mark.parametrize("bad", [-0.1, float("nan")])
    def test_invalid_z_scale_threshold_rejected_by_native_op(self, bad):
        """The host launchers enforce the T >= 0 contract for direct op
        callers that bypass the module's constructor check; the ``>= 0``
        predicate rejects NaN identically to the Python guards."""
        device = torch.device("cuda")
        scales, densities, z_scales, positions, cuboid_dims = _make_domain_inputs(
            device
        )
        outs = [torch.empty_like(t) for t in (scales, densities, z_scales, positions)]
        with pytest.raises(RuntimeError, match="non-negative"):
            torch.ops.gsplat.gaussian_losses_fwd(
                scales,
                densities,
                z_scales,
                positions,
                cuboid_dims,
                None,
                bad,
                *outs,
            )

    @pytest.mark.parametrize("with_visibility", [False, True])
    def test_forward_matches_reference_bitwise(self, with_visibility):
        device = torch.device("cuda")
        inputs = _make_domain_inputs(device)
        visibility = _make_visibility(device) if with_visibility else None
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD)
        outputs = module(*inputs, visibility=visibility)
        refs = _reference(*inputs, visibility=visibility)
        for name, out, ref in zip(
            ("scale", "density", "z_scale", "oob"), outputs, refs
        ):
            assert torch.equal(out, ref), f"{name} max diff: {(out - ref).abs().max()}"

    @pytest.mark.parametrize("with_visibility", [False, True])
    def test_backward_matches_reference_bitwise(self, with_visibility):
        device = torch.device("cuda")
        visibility = _make_visibility(device) if with_visibility else None

        inputs_cuda = _make_domain_inputs(device, requires_grad=True)
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD)
        outputs = module(*inputs_cuda, visibility=visibility)
        upstreams = _make_upstreams(outputs)
        grads = _grads(inputs_cuda[:4], outputs, upstreams)

        inputs_ref = _make_domain_inputs(device, requires_grad=True)
        ref_outputs = _reference(*inputs_ref, visibility=visibility)
        ref_grads = _grads(inputs_ref[:4], ref_outputs, upstreams)

        for name, grad, ref in zip(
            ("scales", "densities", "z_scales", "positions"), grads, ref_grads
        ):
            assert torch.equal(
                grad, ref
            ), f"{name} grad max diff: {(grad - ref).abs().max()}"

    @pytest.mark.parametrize(
        "n_scales,n_densities",
        [(N_SCALES, N_DENSITIES), (N_DENSITIES, N_SCALES)],
        ids=["scales_shorter", "densities_shorter"],
    )
    def test_visibility_span_boundary_both_directions(self, n_scales, n_densities):
        """visibility is [max(N_scales, N_densities)], so the shorter member
        consumes only a prefix. Cover BOTH orderings — in the mirror
        direction (N_scales > N_densities) it is the DENSITY member that
        reads the visibility[:N_densities] prefix — with bitwise forward and
        backward parity against the reference."""
        device = torch.device("cuda")
        visibility = _make_visibility(
            device, n_scales=n_scales, n_densities=n_densities
        )
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD)

        inputs_cuda = _make_domain_inputs(
            device, n_scales=n_scales, n_densities=n_densities, requires_grad=True
        )
        outputs = module(*inputs_cuda, visibility=visibility)
        upstreams = _make_upstreams(outputs)
        grads = _grads(inputs_cuda[:4], outputs, upstreams)

        inputs_ref = _make_domain_inputs(
            device, n_scales=n_scales, n_densities=n_densities, requires_grad=True
        )
        ref_outputs = _reference(*inputs_ref, visibility=visibility)
        ref_grads = _grads(inputs_ref[:4], ref_outputs, upstreams)

        for name, out, ref in zip(
            ("scale", "density", "z_scale", "oob"), outputs, ref_outputs
        ):
            assert torch.equal(out, ref), f"{name} diverges from the reference"
        for name, grad, ref in zip(
            ("scales", "densities", "z_scales", "positions"), grads, ref_grads
        ):
            assert torch.equal(grad, ref), f"{name} grad diverges from the reference"

    def test_z_scales_can_be_the_largest_member(self):
        """The launch grid must cover the largest member, whichever it is —
        a grid sized off the scale member alone would leave z rows unwritten."""
        device = torch.device("cuda")
        inputs = _make_domain_inputs(
            device, n_scales=17, n_densities=33, n_z_scales=4099, n_oob=65
        )
        outputs = FusedGaussianLosses(z_scale_threshold=THRESHOLD)(*inputs)
        refs = _reference(*inputs)
        for out, ref in zip(outputs, refs):
            assert torch.equal(out, ref)

    @pytest.mark.parametrize("empty", ["scales", "densities", "z_scales", "oob", "all"])
    def test_zero_length_members(self, empty):
        device = torch.device("cuda")
        counts = dict(
            n_scales=N_SCALES,
            n_densities=N_DENSITIES,
            n_z_scales=N_Z_SCALES,
            n_oob=N_OOB,
        )
        if empty == "all":
            counts = {k: 0 for k in counts}
        else:
            counts[f"n_{empty}" if empty != "oob" else "n_oob"] = 0
        inputs = _make_domain_inputs(device, requires_grad=True, **counts)
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD)
        outputs = module(*inputs)
        inputs_ref = _make_domain_inputs(device, requires_grad=True, **counts)
        ref_outputs = _reference(*inputs_ref)
        for out, ref in zip(outputs, ref_outputs):
            assert out.shape == ref.shape
            assert torch.equal(out, ref)
        # Backward under a non-trivial (non-all-ones) upstream: with one
        # member zero-length, the non-empty siblings must receive bitwise
        # reference gradient VALUES — not just correctly-shaped finite ones —
        # so an indexing bug triggered only by an empty sibling cannot slip
        # past shape checks.
        upstreams = _make_upstreams(outputs)
        grads = _grads(inputs[:4], outputs, upstreams)
        ref_grads = _grads(inputs_ref[:4], ref_outputs, upstreams)
        for name, grad, ref in zip(
            ("scales", "densities", "z_scales", "positions"), grads, ref_grads
        ):
            assert grad is not None and grad.shape == ref.shape
            assert torch.equal(grad, ref), f"{name} grad diverges from the reference"

    def test_relu_gate_at_exact_boundary(self):
        """Values sitting exactly on the relu kink — z == threshold and
        |position| == cuboid_dim / 2 — must produce zero loss AND an
        exact-zero gradient: the kernels gate on a strict >, matching
        torch.relu's zero subgradient at 0, on the kernel and fallback paths
        alike."""
        device = torch.device("cuda")
        threshold = 1.0
        for use_cuda in (True, False):
            z = torch.tensor([threshold], device=device, requires_grad=True)
            positions = torch.tensor(
                [[1.0, -1.0, 0.5]], device=device, requires_grad=True
            )
            cuboid_dims = torch.tensor([[2.0, 2.0, 1.0]], device=device)
            scales, densities, _, _, _ = _make_domain_inputs(
                device, n_z_scales=1, n_oob=1
            )
            module = FusedGaussianLosses(z_scale_threshold=threshold)
            module._cuda_available = use_cuda
            _, _, loss_z, loss_oob = module(
                scales, densities, z, positions, cuboid_dims
            )
            assert torch.equal(loss_z, torch.zeros_like(loss_z))
            assert torch.equal(loss_oob, torch.zeros_like(loss_oob))
            (loss_z.sum() + loss_oob.sum()).backward()
            assert torch.equal(z.grad, torch.zeros_like(z))
            assert torch.equal(positions.grad, torch.zeros_like(positions))

    def test_legacy_shared_cloud_matches_public_references_bitwise(self):
        """Equal member counts reproduce the pre-generalization contract: the
        outputs equal the public per-loss references bit for bit (the same
        guarantee the pre-existing suite asserts)."""
        device = torch.device("cuda")
        n = 1024
        inputs = _make_domain_inputs(
            device, n_scales=n, n_densities=n, n_z_scales=n, n_oob=n
        )
        scales, densities, z_scales, positions, cuboid_dims = inputs
        visibility = _make_visibility(device, n_scales=n, n_densities=n)
        outputs = FusedGaussianLosses(z_scale_threshold=THRESHOLD)(
            *inputs, visibility=visibility
        )
        assert torch.equal(
            outputs[0], gaussian_scale_reg(scales, visibility=visibility)
        )
        assert torch.equal(
            outputs[1], gaussian_density_reg(densities, visibility=visibility)
        )
        assert torch.equal(outputs[2], gaussian_z_scale_reg(z_scales, THRESHOLD))
        assert torch.equal(outputs[3], out_of_bound_loss(positions, cuboid_dims))

    def test_pre_extension_positional_op_calls_keep_working(self):
        """Schema compatibility: the raw ops accept the pre-extension
        positional arity (no deform arguments, no mode flag) and produce the
        same results as the extended call with the defaults spelled out."""
        device = torch.device("cuda")
        n = 257
        inputs = _make_domain_inputs(
            device, n_scales=n, n_densities=n, n_z_scales=n, n_oob=n
        )
        scales, densities, z_scales, positions, cuboid_dims = inputs

        legacy = [torch.empty_like(t) for t in (scales, densities, z_scales, positions)]
        torch.ops.gsplat.gaussian_losses_fwd(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            None,
            THRESHOLD,
            *legacy,
        )
        extended = [
            torch.empty_like(t) for t in (scales, densities, z_scales, positions)
        ]
        torch.ops.gsplat.gaussian_losses_fwd(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            None,
            THRESHOLD,
            *extended,
            None,
            None,
            None,
            None,
            False,
        )
        for legacy_out, extended_out in zip(legacy, extended):
            assert torch.equal(legacy_out, extended_out)

        upstreams = [torch.ones_like(t) for t in legacy]
        legacy_grads = [
            torch.empty_like(t) for t in (scales, densities, z_scales, positions)
        ]
        torch.ops.gsplat.gaussian_losses_bwd(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            None,
            THRESHOLD,
            *upstreams,
            *legacy_grads,
        )
        extended_grads = [
            torch.empty_like(t) for t in (scales, densities, z_scales, positions)
        ]
        torch.ops.gsplat.gaussian_losses_bwd(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            None,
            THRESHOLD,
            *upstreams,
            *extended_grads,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
        )
        for legacy_grad, extended_grad in zip(legacy_grads, extended_grads):
            assert torch.equal(legacy_grad, extended_grad)


# ---------------------------------------------------------------------------
# Pre-activation mode — CUDA path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA or fused losses not available")
class TestPreactivationCUDA:
    """Log-space member math through the fused CUDA kernel."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("with_visibility", [False, True])
    def test_forward_matches_reference(self, dtype, with_visibility):
        device = torch.device("cuda")
        inputs = _make_domain_inputs(device, preactivation=True, dtype=dtype)
        visibility = _make_visibility(device, dtype=dtype) if with_visibility else None
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD, preactivation=True)
        outputs = module(*inputs, visibility=visibility)
        refs = _reference(*inputs, visibility=visibility, preactivation=True)
        tols = _tols(dtype, preactivation=True)
        for name, out, ref in zip(
            ("scale", "density", "z_scale", "oob"), outputs, refs
        ):
            assert torch.allclose(
                out, ref, **tols
            ), f"{name} max diff: {(out - ref).abs().max()}"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("with_visibility", [False, True])
    def test_backward_matches_reference(self, dtype, with_visibility):
        device = torch.device("cuda")
        visibility = _make_visibility(device, dtype=dtype) if with_visibility else None

        inputs_cuda = _make_domain_inputs(
            device, preactivation=True, dtype=dtype, requires_grad=True
        )
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD, preactivation=True)
        outputs = module(*inputs_cuda, visibility=visibility)
        upstreams = _make_upstreams(outputs)
        grads = _grads(inputs_cuda[:4], outputs, upstreams)

        inputs_ref = _make_domain_inputs(
            device, preactivation=True, dtype=dtype, requires_grad=True
        )
        ref_outputs = _reference(*inputs_ref, visibility=visibility, preactivation=True)
        ref_grads = _grads(inputs_ref[:4], ref_outputs, upstreams)

        tols = _tols(dtype, preactivation=True)
        for name, grad, ref in zip(
            ("scales", "densities", "z_scales", "positions"), grads, ref_grads
        ):
            assert torch.allclose(
                grad, ref, **tols
            ), f"{name} grad max diff: {(grad - ref).abs().max()}"

    def test_folded_segment_weights_match_weighted_reference(self):
        """Segments with weights folded as +log(w) reproduce w * exp(preact)
        per segment through the kernel (values and gradients)."""
        device = torch.device("cuda")
        torch.manual_seed(5)
        weights = (2.0, 1.0, 0.5)
        segments = [torch.randn(64, 3, device=device) for _ in weights]
        folded = torch.cat(
            [fold_log_space_weight(seg, w) for seg, w in zip(segments, weights)]
        ).requires_grad_(True)
        n = folded.shape[0]
        _, densities, z_scales, positions, cuboid_dims = _make_domain_inputs(
            device, n_scales=n, preactivation=True
        )
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD, preactivation=True)
        loss_scale, *_ = module(folded, densities, z_scales, positions, cuboid_dims)
        expected = torch.cat([w * seg.exp() for seg, w in zip(segments, weights)])
        assert torch.allclose(loss_scale, expected, rtol=1e-5, atol=1e-7)
        loss_scale.sum().backward()
        assert torch.allclose(folded.grad, expected, rtol=1e-5, atol=1e-7)

    def test_nonpositive_weight_segment_is_exact_zero_even_on_overflow(self):
        """A folded zero-weight segment whose pre-activations overflow exp()
        still produces exact zeros for value and gradient. Weighting by
        multiplication after the exp() would produce NaN here (0 * inf) — the
        hazard the log-space fold exists to rule out."""
        device = torch.device("cuda")
        torch.manual_seed(5)
        live = torch.randn(32, 3, device=device)
        overflowing = torch.full((32, 3), 600.0, device=device)  # exp → inf (fp32)
        assert torch.isnan(0.0 * overflowing.exp()).all()  # the naive hazard
        folded = torch.cat(
            [fold_log_space_weight(live, 1.5), fold_log_space_weight(overflowing, 0.0)]
        ).requires_grad_(True)
        n = folded.shape[0]
        _, densities, z_scales, positions, cuboid_dims = _make_domain_inputs(
            device, n_scales=n, preactivation=True
        )
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD, preactivation=True)
        loss_scale, *_ = module(folded, densities, z_scales, positions, cuboid_dims)
        assert torch.isfinite(loss_scale).all()
        assert torch.equal(loss_scale[32:], torch.zeros_like(loss_scale[32:]))
        assert torch.allclose(loss_scale[:32], 1.5 * live.exp(), rtol=1e-5, atol=1e-7)
        loss_scale.sum().backward()
        assert torch.equal(folded.grad[32:], torch.zeros_like(folded.grad[32:]))

    def test_zero_visibility_is_exact_zero_even_on_overflow(self):
        """A zero-visibility lane takes the same exact-zero no-read branch as
        a folded non-positive weight: overflowing pre-activations under
        vis == 0 come back as exact zeros for value and gradient (evaluating
        exp() first would give inf * 0 = NaN), identically through the fused
        kernel and the pure-PyTorch fallback."""
        device = torch.device("cuda")
        n = 64
        torch.manual_seed(5)
        live = torch.randn(32, 3, device=device)
        overflowing = torch.full((32, 3), 600.0, device=device)  # exp → inf
        assert torch.isnan(overflowing.exp() * 0.0).all()  # the naive hazard
        _, densities, z_scales, positions, cuboid_dims = _make_domain_inputs(
            device, n_scales=n, n_densities=n, preactivation=True
        )
        visibility = torch.ones(n, device=device)
        visibility[32:] = 0.0
        for use_cuda in (True, False):
            scales = torch.cat([live, overflowing]).requires_grad_(True)
            module = FusedGaussianLosses(
                z_scale_threshold=THRESHOLD, preactivation=True
            )
            module._cuda_available = use_cuda
            loss_scale, *_ = module(
                scales,
                densities,
                z_scales,
                positions,
                cuboid_dims,
                visibility=visibility,
            )
            assert torch.isfinite(loss_scale).all()
            assert torch.equal(loss_scale[32:], torch.zeros_like(loss_scale[32:]))
            assert torch.allclose(loss_scale[:32], live.exp(), rtol=1e-5, atol=1e-7)
            loss_scale.sum().backward()
            assert torch.isfinite(scales.grad).all()
            assert torch.equal(scales.grad[32:], torch.zeros_like(scales.grad[32:]))
            assert torch.allclose(scales.grad[:32], live.exp(), rtol=1e-5, atol=1e-7)

    def test_unused_members_with_overflowing_preactivations_get_zero_grads(self):
        """set_materialize_grads(False) hands an unused output's cotangent to
        the op as exact zeros; the kernel must branch on the zero upstream
        instead of evaluating exp() * 0 (NaN once the pre-activation
        overflows). Only the density and oob outputs join the objective here
        — the overflowing scale and z-scale members must come back with
        finite, exact-zero gradients while the used members keep their
        reference gradients."""
        device = torch.device("cuda")
        scales = torch.full((33, 3), 600.0, device=device, requires_grad=True)
        z_scales = torch.full((17,), 600.0, device=device, requires_grad=True)
        _, densities, _, positions, cuboid_dims = _make_domain_inputs(
            device, n_scales=33, n_z_scales=17, preactivation=True
        )
        densities.requires_grad_(True)
        positions.requires_grad_(True)
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD, preactivation=True)
        _, loss_density, _, loss_oob = module(
            scales, densities, z_scales, positions, cuboid_dims
        )
        (loss_density.sum() + loss_oob.sum()).backward()
        for grad in (scales.grad, z_scales.grad):
            assert grad is not None
            assert torch.equal(grad, torch.zeros_like(grad))
        assert torch.equal(densities.grad, torch.sign(densities.detach()))
        positions_ref = positions.detach().clone().requires_grad_(True)
        out_of_bound_loss(positions_ref, cuboid_dims).sum().backward()
        assert torch.equal(positions.grad, positions_ref.grad)

    def test_masked_objective_zero_upstream_takes_exact_zero_branch(self):
        """A materialized upstream carrying exact-zero elements (here from a
        sliced objective) takes the same per-lane exact-zero branch: the
        kernel must not evaluate exp() * 0 on the masked-out overflowing
        rows."""
        device = torch.device("cuda")
        torch.manual_seed(5)
        live = torch.randn(32, 3, device=device)
        overflowing = torch.full((32, 3), 600.0, device=device)
        scales = torch.cat([live, overflowing]).requires_grad_(True)
        _, densities, z_scales, positions, cuboid_dims = _make_domain_inputs(
            device, n_scales=64, preactivation=True
        )
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD, preactivation=True)
        loss_scale, *_ = module(scales, densities, z_scales, positions, cuboid_dims)
        # Only the live rows join the objective; the slice backward
        # materializes the overflowing rows' upstream as exact zeros.
        loss_scale[:32].sum().backward()
        assert torch.isfinite(scales.grad).all()
        assert torch.equal(scales.grad[32:], torch.zeros_like(scales.grad[32:]))
        assert torch.allclose(scales.grad[:32], live.exp(), rtol=1e-5, atol=1e-7)

    def test_z_scale_relu_gate_at_exact_threshold(self):
        """exp(z) exactly equal to the threshold sits on the relu kink: zero
        loss and an exact-zero gradient through the strict > gate (exp(0) ==
        1 is exact in every precision and math mode), kernel and fallback
        alike."""
        device = torch.device("cuda")
        threshold = 1.0
        for use_cuda in (True, False):
            z = torch.zeros(3, device=device, requires_grad=True)  # exp(0) == 1
            scales, densities, _, positions, cuboid_dims = _make_domain_inputs(
                device, n_z_scales=3, preactivation=True
            )
            module = FusedGaussianLosses(
                z_scale_threshold=threshold, preactivation=True
            )
            module._cuda_available = use_cuda
            _, _, loss_z, _ = module(scales, densities, z, positions, cuboid_dims)
            assert torch.equal(loss_z, torch.zeros_like(loss_z))
            loss_z.sum().backward()
            assert torch.equal(z.grad, torch.zeros_like(z))

    def test_z_scale_threshold_stays_in_post_activation_units(self):
        """relu(exp(z) - threshold): rows below threshold produce zero loss
        AND zero gradient; rows above produce exp(z) - threshold with
        gradient exp(z)."""
        device = torch.device("cuda")
        threshold = 1.0  # exp(z) > 1 ⇔ z > 0
        z = torch.tensor([-2.0, -0.5, 0.5, 2.0], device=device, requires_grad=True)
        n = z.shape[0]
        scales, densities, _, positions, cuboid_dims = _make_domain_inputs(
            device, n_z_scales=n, preactivation=True
        )
        module = FusedGaussianLosses(z_scale_threshold=threshold, preactivation=True)
        _, _, loss_z, _ = module(scales, densities, z, positions, cuboid_dims)
        expected = torch.relu(z.detach().exp() - threshold)
        assert torch.allclose(loss_z, expected, rtol=1e-5, atol=1e-7)
        assert (loss_z[:2] == 0).all()
        loss_z.sum().backward()
        assert torch.equal(z.grad[:2], torch.zeros(2, device=device))
        assert torch.allclose(z.grad[2:], z.detach()[2:].exp(), rtol=1e-5, atol=1e-7)

    def test_density_gradient_uses_sign(self):
        """Signed densities (folded weights): |d| forward, sign(d) backward,
        with the zero subgradient at exactly zero."""
        device = torch.device("cuda")
        densities = torch.tensor([-1.5, -0.25, 0.0, 0.75], device=device)
        densities.requires_grad_(True)
        n = densities.shape[0]
        scales, _, z_scales, positions, cuboid_dims = _make_domain_inputs(
            device, n_densities=n, preactivation=True
        )
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD, preactivation=True)
        _, loss_density, _, _ = module(
            scales, densities, z_scales, positions, cuboid_dims
        )
        assert torch.equal(loss_density, densities.detach().abs())
        loss_density.sum().backward()
        assert torch.equal(
            densities.grad, torch.tensor([-1.0, -1.0, 0.0, 1.0], device=device)
        )

    def test_gradcheck_fp64(self):
        """Full-fidelity fp64 gradcheck through the CUDA path with distinct
        member counts (inputs kept away from the relu/threshold kinks)."""
        device = torch.device("cuda")
        torch.manual_seed(9)
        scales = (torch.randn(5, 3, device=device, dtype=torch.float64)).requires_grad_(
            True
        )
        densities = (
            torch.tensor(
                [-1.2, 0.7, 1.4, -0.6, 0.9, -1.1, 0.8],
                device=device,
                dtype=torch.float64,
            )
        ).requires_grad_(True)
        z_scales = (
            torch.tensor([-1.5, 1.0, -0.8], device=device, dtype=torch.float64)
        ).requires_grad_(
            True
        )  # exp(z) far from THRESHOLD
        positions = (
            torch.tensor(
                [[3.0, -0.1, -2.5], [0.2, 4.0, 0.1]], device=device, dtype=torch.float64
            )
        ).requires_grad_(
            True
        )  # per-axis clearly inside or outside
        cuboid_dims = torch.full((2, 3), 2.0, device=device, dtype=torch.float64)
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD, preactivation=True)

        def fn(s, d, z, p):
            return module(s, d, z, p, cuboid_dims)

        assert torch.autograd.gradcheck(
            fn, (scales, densities, z_scales, positions), nondet_tol=0.0
        )

    def test_disable_cuda_uses_fallback(self):
        """With the module's CUDA flag off, CUDA inputs route through the
        pure-PyTorch pre-activation fallback and agree with the kernel."""
        device = torch.device("cuda")
        inputs = _make_domain_inputs(device, preactivation=True)
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD, preactivation=True)
        cuda_outputs = module(*inputs)
        module._cuda_available = False  # force fallback
        fallback_outputs = module(*inputs)
        tols = _tols(torch.float32, preactivation=True)
        for cuda_out, fallback_out in zip(cuda_outputs, fallback_outputs):
            assert torch.allclose(cuda_out, fallback_out, **tols)


# ---------------------------------------------------------------------------
# Deform-smoothness member composition with the new modes — CUDA path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA or fused losses not available")
class TestDeformCompositionCUDA:
    """The grouped deform member is independent of the gaussian members: it
    must compose with per-member domains and pre-activation mode, and never be
    gated on the gaussian counts."""

    def _deform_inputs(self, device, n=257):
        torch.manual_seed(7)
        deformation = torch.randn(n, 3, device=device, requires_grad=True)
        mask = torch.rand(n, 1, device=device) * 1.5
        mask[::3] = 0.0
        mask.requires_grad_(True)
        return deformation, mask

    @pytest.mark.parametrize("preactivation", [False, True])
    def test_deform_composes_with_domains(self, preactivation):
        """Distinct member counts (one member empty) + the deform member: the
        gaussian outputs are unchanged by the rider and the deform loss
        matches its reference, forward and backward."""
        device = torch.device("cuda")
        inputs = _make_domain_inputs(device, n_z_scales=0, preactivation=preactivation)
        deformation, mask = self._deform_inputs(device)
        module = FusedGaussianLosses(
            z_scale_threshold=THRESHOLD, preactivation=preactivation
        )
        outputs = module(*inputs, deformation=deformation, deform_mask=mask)
        assert len(outputs) == 5
        *gaussian_losses, dl = outputs
        for out, ref in zip(gaussian_losses, module(*inputs)):
            assert torch.equal(out, ref)
        ref_dl = deform_smoothness_loss(deformation.detach(), mask.detach())
        assert torch.allclose(dl, ref_dl, rtol=1e-5, atol=1e-6)

        dl.backward()
        d_ref, m_ref = self._deform_inputs(device)
        deform_smoothness_loss(d_ref, m_ref).backward()
        assert torch.allclose(deformation.grad, d_ref.grad, rtol=1e-5, atol=1e-6)
        assert torch.allclose(mask.grad, m_ref.grad, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("preactivation", [False, True])
    def test_all_gaussian_members_empty_never_skips_deform(self, preactivation):
        """Empty gaussian work in every member must not skip the deform
        member's launch, in either mode."""
        device = torch.device("cuda")
        inputs = _make_domain_inputs(
            device,
            n_scales=0,
            n_densities=0,
            n_z_scales=0,
            n_oob=0,
            preactivation=preactivation,
        )
        deformation, mask = self._deform_inputs(device)
        module = FusedGaussianLosses(
            z_scale_threshold=THRESHOLD, preactivation=preactivation
        )
        *gaussian_losses, dl = module(
            *inputs, deformation=deformation, deform_mask=mask
        )
        assert all(out.numel() == 0 for out in gaussian_losses)
        ref = deform_smoothness_loss(deformation.detach(), mask.detach())
        assert dl.item() != 0.0
        assert torch.allclose(dl, ref, rtol=1e-5, atol=1e-6)
        dl.backward()
        assert deformation.grad is not None and mask.grad is not None

    def test_preactivation_gaussian_grads_and_deform_grads_coexist(self):
        """One backward through all five outputs in pre-activation mode:
        every gaussian member and the deform member receive their reference
        gradients from the same fused op call."""
        device = torch.device("cuda")
        inputs = _make_domain_inputs(device, preactivation=True, requires_grad=True)
        deformation, mask = self._deform_inputs(device)
        module = FusedGaussianLosses(z_scale_threshold=THRESHOLD, preactivation=True)
        *gaussian_losses, dl = module(
            *inputs, deformation=deformation, deform_mask=mask
        )
        (sum(out.sum() for out in gaussian_losses) + dl).backward()

        inputs_ref = _make_domain_inputs(device, preactivation=True, requires_grad=True)
        ref_outputs = _reference(*inputs_ref, preactivation=True)
        d_ref, m_ref = self._deform_inputs(device)
        (
            sum(out.sum() for out in ref_outputs) + deform_smoothness_loss(d_ref, m_ref)
        ).backward()

        tols = _tols(torch.float32, preactivation=True)
        for grad, ref in zip(
            (t.grad for t in inputs[:4]), (t.grad for t in inputs_ref[:4])
        ):
            assert torch.allclose(grad, ref, **tols)
        assert torch.allclose(deformation.grad, d_ref.grad, rtol=1e-5, atol=1e-6)
        assert torch.allclose(mask.grad, m_ref.grad, rtol=1e-5, atol=1e-6)
