# SPDX-License-Identifier: Apache-2.0
"""Affine color matching on nearly uniform and rank-deficient channels."""

import pytest
import torch

from gsplat.color_correct import color_correct_affine


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("spread", [1e-2, 1e-4, 1e-5])
def test_affine_color_fit_recovers_low_variance_reference(dtype, spread):
    ref = 0.5 + spread * torch.linspace(-1, 1, 101, dtype=dtype)[:, None]
    img = 0.6 * ref + 0.1
    corrected = color_correct_affine(img, ref)
    torch.testing.assert_close(corrected, ref, atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("constant_input", [False, True])
def test_constant_reference_channel_matches_reference_mean(constant_input):
    img = torch.linspace(0.2, 0.7, 101, dtype=torch.float64)[:, None]
    if constant_input:
        img = torch.full_like(img, 0.4)
    ref = torch.full_like(img, 0.6)
    torch.testing.assert_close(
        color_correct_affine(img, ref), ref, atol=1e-12, rtol=1e-12
    )


def test_mixed_identifiable_and_constant_channels():
    ref0 = torch.linspace(0.1, 0.9, 101, dtype=torch.float64)
    ref = torch.stack([ref0, torch.full_like(ref0, 0.4), 1 - ref0], -1)
    img = torch.stack([0.7 * ref0 + 0.1, 0.2 + 0.3 * ref0, -0.5 * (1 - ref0) + 0.6], -1)
    torch.testing.assert_close(
        color_correct_affine(img, ref), ref, atol=1e-12, rtol=1e-12
    )
