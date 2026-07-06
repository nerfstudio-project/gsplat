import numpy as np
import torch

from citygs.partition.contraction import contract, foreground_bounds, uncontract


def test_contract_identity_inside_box():
    x = np.array([[0.5, -0.3, 0.1], [0.0, 0.0, 0.0]])
    y = contract(x, [0, 0, 0], [1, 1, 1])
    np.testing.assert_allclose(y, x, atol=1e-9)


def test_contract_bounded_and_monotonic():
    x = np.array([[100.0, 0.0, 0.0], [1e6, 1e6, -1e6], [3.0, 0.5, 0.0]])
    y = contract(x, [0, 0, 0], [1, 1, 1])
    assert np.abs(y).max() < 2.0
    # Farther points contract to larger coordinates.
    a = contract(np.array([[5.0, 0, 0]]), [0, 0, 0], [1, 1, 1])[0, 0]
    b = contract(np.array([[50.0, 0, 0]]), [0, 0, 0], [1, 1, 1])[0, 0]
    assert 1.0 < a < b < 2.0


def test_contract_uncontract_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.normal(scale=10.0, size=(500, 3))
    center = np.array([1.0, -2.0, 0.5])
    half_extent = np.array([3.0, 2.0, 1.0])
    y = contract(x, center, half_extent)
    x2 = uncontract(y, center, half_extent)
    np.testing.assert_allclose(x2, x, rtol=1e-6, atol=1e-6)


def test_contract_torch_matches_numpy():
    x = np.random.default_rng(2).normal(scale=5.0, size=(100, 3))
    y_np = contract(x, [0, 0, 0], [2, 2, 2])
    y_t = contract(torch.from_numpy(x), [0, 0, 0], [2, 2, 2]).numpy()
    np.testing.assert_allclose(y_np, y_t, atol=1e-9)


def test_foreground_bounds_rejects_outliers():
    rng = np.random.default_rng(3)
    inliers = rng.uniform(-1, 1, size=(200, 3))
    outliers = np.array([[500.0, 0, 0], [0, -800.0, 0]])
    positions = np.concatenate([inliers, outliers])
    center, half_extent = foreground_bounds(positions)
    assert np.abs(center).max() < 0.5
    assert half_extent.max() < 2.0  # outliers did not blow up the box


def test_foreground_bounds_degenerate_axis():
    # Constant-altitude rig: z is exactly constant.
    rng = np.random.default_rng(4)
    positions = rng.uniform(-1, 1, size=(100, 3))
    positions[:, 2] = 5.0
    center, half_extent = foreground_bounds(positions)
    assert half_extent[2] > 0  # never collapses
    assert abs(center[2] - 5.0) < 1e-6
