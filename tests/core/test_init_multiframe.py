# SPDX-License-Identifier: Apache-2.0
"""Tests for the G-SHARP v0.2 init utilities in ``gsplat.init_utils`` (branch: vnath_gsharp).

Covers the two ported functions:

- :func:`gsplat.init_utils.multi_frame_depth_unprojection` — masked depth →
  world-space point cloud via per-frame pinhole intrinsics + camera-to-world
  poses.
- :func:`gsplat.init_utils.knn_scale_init` — pure-torch per-point log of
  mean-k-NN distance, used as Gaussian log-scale init.

Seed is set to 42 by the autouse fixture in ``conftest.py``.
"""

import pytest
import torch

from tests._cuda import cuda_is_available

from gsplat.init_utils import knn_scale_init, multi_frame_depth_unprojection


# ---------------------------------------------------------------------------
# multi_frame_depth_unprojection
# ---------------------------------------------------------------------------


def test_multiframe_unprojection_recovers_synthetic_grid():
    """Single frame, identity pose, constant depth: unprojection should
    recover an analytically known world-space grid.

    With ``fx = fy = 4``, ``cx = cy = 2``, ``depth = 1.0`` everywhere, the
    pixel ``(u, v)`` maps to camera-space ``((u - 2)/4, (v - 2)/4, 1.0)``.
    Identity camera-to-world keeps that as world-space.
    """
    h, w = 4, 4
    fx = fy = 4.0
    cx = cy = 2.0

    images = torch.rand(1, h, w, 3)
    depths = torch.full((1, h, w), 1.0)
    masks = torch.ones(1, h, w)
    poses = torch.eye(4).unsqueeze(0)
    intrinsics = torch.tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]])

    xyz, rgb = multi_frame_depth_unprojection(images, depths, masks, poses, intrinsics)

    assert xyz.shape == (h * w, 3)
    assert rgb.shape == (h * w, 3)
    # All z = depth = 1.0.
    assert torch.allclose(xyz[:, 2], torch.ones(h * w), atol=1e-6)
    # x in [(0-cx)/fx, (w-1-cx)/fx] = [-0.5, 0.25] for w=4.
    assert xyz[:, 0].min() >= -0.5 - 1e-6
    assert xyz[:, 0].max() <= 0.25 + 1e-6
    assert xyz[:, 1].min() >= -0.5 - 1e-6
    assert xyz[:, 1].max() <= 0.25 + 1e-6


def test_multiframe_unprojection_rgb_matches_source_pixels():
    """All four source RGBs should appear in the output point set (one per pixel)."""
    images = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            ]
        ]
    )  # (1, 2, 2, 3)
    depths = torch.full((1, 2, 2), 1.0)
    masks = torch.ones(1, 2, 2)
    poses = torch.eye(4).unsqueeze(0)
    intrinsics = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]])

    _, rgb = multi_frame_depth_unprojection(images, depths, masks, poses, intrinsics)

    assert rgb.shape == (4, 3)
    expected = images.reshape(-1, 3)
    for r in expected:
        assert any(
            torch.allclose(out, r, atol=1e-6) for out in rgb
        ), f"Expected RGB {r.tolist()} not found in output."


def test_multiframe_unprojection_frame_count_mismatch_raises():
    images = torch.rand(2, 4, 4, 3)
    depths = torch.rand(3, 4, 4)  # mismatch
    masks = torch.ones(2, 4, 4)
    poses = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    intrinsics = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    with pytest.raises(ValueError, match="leading dim"):
        multi_frame_depth_unprojection(images, depths, masks, poses, intrinsics)


def test_multiframe_unprojection_all_masked_returns_empty():
    images = torch.rand(2, 4, 4, 3)
    depths = torch.full((2, 4, 4), 1.0)
    masks = torch.zeros(2, 4, 4)
    poses = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    intrinsics = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    xyz, rgb = multi_frame_depth_unprojection(images, depths, masks, poses, intrinsics)
    assert xyz.shape == (0, 3)
    assert rgb.shape == (0, 3)


def test_multiframe_unprojection_zero_depth_pixels_excluded():
    """Pixels with depth == 0 should be dropped even when mask is on."""
    h, w = 4, 4
    images = torch.rand(1, h, w, 3)
    depths = torch.full((1, h, w), 1.0)
    depths[0, :2, :] = 0.0  # top half: no depth
    masks = torch.ones(1, h, w)
    poses = torch.eye(4).unsqueeze(0)
    intrinsics = torch.tensor([[[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]])

    xyz, rgb = multi_frame_depth_unprojection(images, depths, masks, poses, intrinsics)
    # Only bottom half (8 pixels) should remain.
    assert xyz.shape == (8, 3)
    assert rgb.shape == (8, 3)


def test_multiframe_unprojection_max_points_subsamples():
    h, w = 4, 4
    images = torch.rand(1, h, w, 3)
    depths = torch.full((1, h, w), 1.0)
    masks = torch.ones(1, h, w)
    poses = torch.eye(4).unsqueeze(0)
    intrinsics = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]])

    xyz, rgb = multi_frame_depth_unprojection(
        images, depths, masks, poses, intrinsics, max_points=5
    )
    assert xyz.shape == (5, 3)
    assert rgb.shape == (5, 3)


# ---------------------------------------------------------------------------
# knn_scale_init
# ---------------------------------------------------------------------------


def test_knn_scale_init_matches_sklearn_reference():
    """Pure-torch result must agree with sklearn's NearestNeighbors path
    used by G-SHARP's ``knn`` helper (skips if sklearn is not installed).
    """
    pytest.importorskip("sklearn.neighbors")
    from sklearn.neighbors import NearestNeighbors

    xyz = torch.randn(50, 3)
    result = knn_scale_init(xyz, k=3)

    model = NearestNeighbors(n_neighbors=4, metric="euclidean").fit(xyz.numpy())
    distances, _ = model.kneighbors(xyz.numpy())
    neighbor_dists = torch.from_numpy(distances[:, 1:].astype("float32"))
    expected = neighbor_dists.pow(2).mean(dim=-1).sqrt().clamp_min(1e-7).log()

    assert torch.allclose(result, expected, atol=1e-5)


def test_knn_scale_init_too_few_points_raises():
    xyz = torch.randn(2, 3)  # need k+1=4 for default k=3
    with pytest.raises(ValueError, match="at least"):
        knn_scale_init(xyz, k=3)


def test_knn_scale_init_duplicate_points_no_inf():
    """All-zero point cloud → all neighbour distances are 0 → eps-clamped log
    must stay finite (no -inf).
    """
    xyz = torch.zeros(10, 3)
    result = knn_scale_init(xyz, k=3)
    assert torch.isfinite(result).all()


def test_knn_scale_init_uniform_grid_constant_scale():
    """Points on a regular 1D grid have a known nearest-neighbour distance
    (the grid spacing). The k=1 result should be exactly ``log(spacing)``
    everywhere except the two end points (which we exclude from the check).
    """
    spacing = 0.5
    xyz = torch.zeros(8, 3)
    xyz[:, 0] = torch.arange(8, dtype=torch.float32) * spacing
    result = knn_scale_init(xyz, k=1)
    # Interior points have nearest-neighbor at exactly *spacing*; endpoints too.
    expected = torch.full_like(result, torch.tensor(spacing).log().item())
    assert torch.allclose(result, expected, atol=1e-6)


@pytest.mark.skipif(not cuda_is_available(), reason="needs CUDA")
def test_knn_scale_init_large_n_peak_memory_bounded():
    """Regression test pinning the chunked KNN memory contract.

    The old ``cdist(xyz, xyz)`` allocated ~N^2 * 4 bytes = ~10 GB at
    N=50_000. The chunked impl is O(chunk_size * N). With the default
    ``chunk_size=1024`` that's ~200 MB at N=50_000 — leave comfortable
    headroom and assert well below 1 GB so a silent regression back to
    the dense path fails loudly. The input is only ~600 KB
    (50k * 3 * 4 bytes), so the measured peak is dominated by the kNN
    scan itself.
    """
    n = 50_000
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_alloc = torch.cuda.memory_allocated()

    xyz = torch.randn(n, 3, device="cuda")
    result = knn_scale_init(xyz, k=3)  # chunk_size default = 1024

    peak_alloc_during = torch.cuda.max_memory_allocated() - baseline_alloc

    assert result.shape == (n,)
    assert torch.isfinite(result).all()

    one_gb = 1 * 1024**3
    assert peak_alloc_during < one_gb, (
        f"knn_scale_init peak CUDA allocation "
        f"{peak_alloc_during / 1024**3:.2f} GB exceeds the 1 GB ceiling — "
        f"looks like a regression back to the O(N^2) dense cdist path."
    )
