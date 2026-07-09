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

"""Tests for the fused SSIM loss module."""

import pytest
import torch

from gsplat import LossFlag, has_losses
from gsplat.losses_fused import FusedSSIMLosses, _cuda_ssim_losses_available

if LossFlag is not None:
    _FLAG_INVALID = int(LossFlag.INVALID)
else:
    _FLAG_INVALID = None

SSIM_CUDA = torch.cuda.is_available() and _cuda_ssim_losses_available()


def _make_ssim_inputs(device, requires_grad=False, shape=(2, 40, 37, 3)):
    torch.manual_seed(42)
    B, H, W, C = shape
    flags = torch.zeros(B, H, W, dtype=torch.int32, device=device)
    flags[:, :4, :] = _FLAG_INVALID
    flags[:, :, W // 3 : W // 3 + 2] = _FLAG_INVALID
    pred = torch.rand(B, H, W, C, device=device, dtype=torch.float32)
    target = torch.rand(B, H, W, C, device=device, dtype=torch.float32)
    if requires_grad:
        pred.requires_grad_(True)
    return flags, pred, target


def test_requires_extension(monkeypatch):
    """The per-pixel flag vocabulary is sourced from the compiled extension."""
    import gsplat.losses_fused as lf

    monkeypatch.setattr(lf, "_FLAG_INVALID", None)
    with pytest.raises(RuntimeError, match="requires the compiled gsplat loss"):
        lf.FusedSSIMLosses()


@pytest.mark.skipif(
    not has_losses(), reason="fused loss extension (flag vocabulary) not built"
)
class TestFusedSSIMLossesFallback:
    def test_forward_shapes(self):
        flags, pred, target = _make_ssim_inputs(torch.device("cpu"))
        loss = FusedSSIMLosses(mask_mode="constant", constant_mask_value=0.5)(
            flags, pred, target, factor=0.01
        )
        assert loss.shape == (2, 40, 37, 1)

    def test_accepts_flags_with_trailing_channel(self):
        flags, pred, target = _make_ssim_inputs(torch.device("cpu"))
        loss1 = FusedSSIMLosses()(flags, pred, target, factor=0.01)
        loss2 = FusedSSIMLosses()(flags.unsqueeze(-1), pred, target, factor=0.01)
        assert torch.allclose(loss1, loss2)

    def test_invalid_masked_out(self):
        flags, pred, target = _make_ssim_inputs(torch.device("cpu"))
        loss = FusedSSIMLosses(mask_mode="constant", constant_mask_value=0.5)(
            flags, pred, target, factor=0.01
        )
        invalid = (flags & _FLAG_INVALID) != 0
        assert (loss[..., 0][invalid] == 0).all()

    def test_non_contiguous_matches_contiguous(self):
        torch.manual_seed(7)
        B, H, W, C = 1, 24, 25, 3
        flags = torch.zeros(B, H, W, dtype=torch.int32)
        flags[:, 2:5, :] = _FLAG_INVALID
        pred = torch.rand(B, C, H, W).permute(0, 2, 3, 1)
        target = torch.rand(B, C, H, W).permute(0, 2, 3, 1)
        assert not pred.is_contiguous()
        module = FusedSSIMLosses(mask_mode="constant", constant_mask_value=0.5)

        actual = module(flags, pred, target, factor=0.01)
        expected = module(
            flags.contiguous(),
            pred.contiguous(),
            target.contiguous(),
            factor=0.01,
        )

        assert torch.allclose(actual, expected)

    def test_disabled_factor_is_nan_safe_exact_zero(self):
        flags, pred, target = _make_ssim_inputs(torch.device("cpu"), requires_grad=True)
        pred.data[0, 0, 0, 0] = float("nan")
        loss = FusedSSIMLosses()(flags, pred, target, factor=0.0)
        assert not torch.isnan(loss).any()
        assert (loss == 0).all()
        loss.sum().backward()
        assert pred.grad is not None
        assert (pred.grad == 0).all()

    def test_empty_input(self):
        flags = torch.empty(0, 8, 8, dtype=torch.int32)
        pred = torch.empty(0, 8, 8, 3)
        target = torch.empty(0, 8, 8, 3)
        loss = FusedSSIMLosses()(flags, pred, target, factor=0.01)
        assert loss.shape == (0, 8, 8, 1)

    def test_target_requires_grad_rejected(self):
        flags, pred, target = _make_ssim_inputs(torch.device("cpu"))
        target.requires_grad_(True)
        with pytest.raises(ValueError, match="target"):
            FusedSSIMLosses()(flags, pred, target, factor=0.01)

    def test_float64_rejected(self):
        flags, pred, target = _make_ssim_inputs(torch.device("cpu"))
        with pytest.raises(ValueError, match="float32"):
            FusedSSIMLosses()(flags, pred.double(), target.double(), factor=0.01)


@pytest.mark.skipif(not SSIM_CUDA, reason="CUDA or fused SSIM losses not available")
class TestFusedSSIMLossesCUDA:
    def test_forward_matches_pytorch(self):
        device = torch.device("cuda")
        flags, pred, target = _make_ssim_inputs(device)
        module_cuda = FusedSSIMLosses(mask_mode="constant", constant_mask_value=0.5)
        loss_cuda = module_cuda(flags, pred, target, factor=0.01)

        module_ref = FusedSSIMLosses(mask_mode="constant", constant_mask_value=0.5)
        module_ref._cuda_available = False
        loss_ref = module_ref(flags, pred, target, factor=0.01)

        assert torch.allclose(
            loss_cuda, loss_ref, rtol=1e-5, atol=1e-5
        ), f"ssim loss max diff: {(loss_cuda - loss_ref).abs().max()}"

    def test_forward_matches_pytorch_target_mask(self):
        device = torch.device("cuda")
        flags, pred, target = _make_ssim_inputs(device)
        module_cuda = FusedSSIMLosses(mask_mode="target")
        loss_cuda = module_cuda(flags, pred, target, factor=0.01)

        module_ref = FusedSSIMLosses(mask_mode="target")
        module_ref._cuda_available = False
        loss_ref = module_ref(flags, pred, target, factor=0.01)

        assert torch.allclose(
            loss_cuda, loss_ref, rtol=1e-5, atol=1e-5
        ), f"ssim loss max diff: {(loss_cuda - loss_ref).abs().max()}"

    def test_backward_matches_pytorch_target_mask(self):
        device = torch.device("cuda")
        factor = 0.01

        flags1, pred1, target1 = _make_ssim_inputs(device, requires_grad=True)
        module_ref = FusedSSIMLosses(mask_mode="target")
        module_ref._cuda_available = False
        loss1 = module_ref(flags1, pred1, target1, factor=factor)
        loss1.sum().backward()

        flags2, pred2, target2 = _make_ssim_inputs(device, requires_grad=True)
        module_cuda = FusedSSIMLosses(mask_mode="target")
        loss2 = module_cuda(flags2, pred2, target2, factor=factor)
        loss2.sum().backward()

        assert torch.allclose(
            loss2, loss1, rtol=1e-5, atol=1e-5
        ), f"ssim loss max diff: {(loss2 - loss1).abs().max()}"
        assert torch.allclose(
            pred2.grad, pred1.grad, rtol=1e-3, atol=1e-4
        ), f"ssim pred grad max diff: {(pred2.grad - pred1.grad).abs().max()}"

    def test_backward_matches_pytorch(self):
        device = torch.device("cuda")
        factor = 0.01

        flags1, pred1, target1 = _make_ssim_inputs(device, requires_grad=True)
        module_ref = FusedSSIMLosses(mask_mode="constant", constant_mask_value=0.5)
        module_ref._cuda_available = False
        loss1 = module_ref(flags1, pred1, target1, factor=factor)
        loss1.sum().backward()

        flags2, pred2, target2 = _make_ssim_inputs(device, requires_grad=True)
        module_cuda = FusedSSIMLosses(mask_mode="constant", constant_mask_value=0.5)
        loss2 = module_cuda(flags2, pred2, target2, factor=factor)
        loss2.sum().backward()

        assert torch.allclose(
            loss2, loss1, rtol=1e-5, atol=1e-5
        ), f"ssim loss max diff: {(loss2 - loss1).abs().max()}"
        assert torch.allclose(
            pred2.grad, pred1.grad, rtol=1e-3, atol=1e-4
        ), f"ssim pred grad max diff: {(pred2.grad - pred1.grad).abs().max()}"

    def test_empty_input(self):
        device = torch.device("cuda")
        flags = torch.empty(0, 8, 8, dtype=torch.int32, device=device)
        pred = torch.empty(0, 8, 8, 3, device=device)
        target = torch.empty(0, 8, 8, 3, device=device)
        loss = FusedSSIMLosses()(flags, pred, target, factor=0.01)
        assert loss.shape == (0, 8, 8, 1)

    def test_disabled_factor(self):
        device = torch.device("cuda")
        flags, pred, target = _make_ssim_inputs(device)
        loss = FusedSSIMLosses()(flags, pred, target, factor=-1.0)
        assert (loss == 0).all()

    def test_multitile_non_multiple_shape(self):
        device = torch.device("cuda")
        flags, pred, target = _make_ssim_inputs(
            device, requires_grad=True, shape=(1, 128, 257, 3)
        )
        loss = FusedSSIMLosses(mask_mode="constant", constant_mask_value=0.5)(
            flags, pred, target, factor=0.01
        )
        loss.sum().backward()
        assert loss.shape == (1, 128, 257, 1)
        assert pred.grad is not None
