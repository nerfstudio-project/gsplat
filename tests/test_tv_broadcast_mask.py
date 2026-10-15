# SPDX-License-Identifier: Apache-2.0
"""Broadcasting a TV mask must not change the normalization."""

import pytest
import torch

from gsplat.regularizers import compute_tv_loss_targeted


@pytest.mark.parametrize(
    "mask_shape", [(1, 1, 4, 5), (3, 1, 1, 5), (1, 1, 4, 1), (3, 2, 4, 5)]
)
def test_tv_mask_broadcast_equals_expanded_mask(mask_shape):
    image = torch.randn(
        3,
        2,
        4,
        5,
        generator=torch.Generator().manual_seed(3),
        dtype=torch.float64,
        requires_grad=True,
    )
    mask = torch.ones(mask_shape, dtype=torch.float64)
    mask[..., 0] = 0
    expanded = mask.expand_as(image)
    vertical = (image[:, :, 1:] - image[:, :, :-1]).abs()
    horizontal = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs()
    mv, mh = expanded[:, :, 1:], expanded[:, :, :, 1:]
    expected = (vertical * mv).sum() / (mv.sum() + 1e-8) + (horizontal * mh).sum() / (
        mh.sum() + 1e-8
    )
    actual = compute_tv_loss_targeted(image, mask)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, image, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, image)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_broadcast_tv_mask_is_invariant_to_batch_repetition():
    image = torch.arange(20, dtype=torch.float64).reshape(1, 1, 4, 5)
    mask = torch.ones(1, 1, 4, 5, dtype=torch.float64)
    torch.testing.assert_close(
        compute_tv_loss_targeted(image.repeat(3, 1, 1, 1), mask),
        compute_tv_loss_targeted(image, mask),
    )
