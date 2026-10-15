# SPDX-License-Identifier: Apache-2.0
"""Nearest-neighbour scale must depend on relative positions, not world origin."""

import pytest
import torch

from gsplat.init_utils import knn_scale_init


def reference_log_scale(xyz, k, eps=1e-7):
    distances = torch.linalg.vector_norm(xyz[:, None] - xyz[None, :], dim=-1)
    distances = distances.masked_fill(
        torch.eye(len(xyz), dtype=torch.bool), float("inf")
    )
    nearest = distances.topk(k, largest=False).values
    return nearest.square().mean(-1).sqrt().clamp_min(eps).log()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
@pytest.mark.parametrize("k", [1, 3])
def test_knn_scale_is_translation_invariant(dtype, chunk_size, k):
    xyz = torch.zeros(32, 3, dtype=dtype)
    xyz[:, 0] = torch.arange(32, dtype=dtype)
    translated = xyz + 10000
    expected = reference_log_scale(xyz, k)
    torch.testing.assert_close(
        knn_scale_init(translated, k, chunk_size=chunk_size), expected
    )
    torch.testing.assert_close(knn_scale_init(xyz, k, chunk_size=chunk_size), expected)


@pytest.mark.parametrize("k", [1, 3])
def test_knn_duplicate_points_keep_other_zero_distance_neighbours(k):
    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 2.0, 1.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(knn_scale_init(xyz, k), reference_log_scale(xyz, k))


def test_knn_scale_gradient_matches_relative_distance_reference():
    xyz = torch.randn(
        32, 3, generator=torch.Generator().manual_seed(7), dtype=torch.float64
    )
    xyz = (xyz + 10000).requires_grad_()
    actual = knn_scale_init(xyz, k=3, chunk_size=7)
    expected = reference_log_scale(xyz, 3)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    actual_grad = torch.autograd.grad(actual.sum(), xyz)[0]
    expected_grad = torch.autograd.grad(expected.sum(), xyz)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-9, rtol=1e-9)
