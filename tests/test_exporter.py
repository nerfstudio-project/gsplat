# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools

import numpy as np
import pytest
import torch
from gsplat.exporter import export_splats, sort_centers


def _morton_code(coordinates):
    # Independent scalar bit-interleaving reference for the 10-bit grid.
    return sum(
        ((coordinate >> bit) & 1) << (3 * bit + axis)
        for axis, coordinate in enumerate(coordinates)
        for bit in range(10)
    )


def _corners(active_axes, dtype):
    points = []
    grid = []
    for corner in itertools.product((0, 1), repeat=len(active_axes)):
        point = [0, 0, 0]
        for axis, value in zip(active_axes, corner):
            point[axis] = value
        points.append(point)
        grid.append([1023 * value for value in point])
    permutation = list(reversed(range(len(points))))
    return (
        torch.tensor(points, dtype=dtype)[permutation],
        [grid[index] for index in permutation],
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("active_axes", [(0,), (1,), (2,), (0, 1, 2)])
def test_sort_centers_inclusive_maximum(active_axes, dtype):
    centers, grid = _corners(active_axes, dtype)
    centers = centers * torch.tensor([2, 3, 5], dtype=dtype) + 7
    labels = torch.arange(len(centers)) + 100
    expected = labels[
        sorted(range(len(grid)), key=lambda index: _morton_code(grid[index]))
    ]

    result = sort_centers(centers, labels)

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sort_centers_identical_points_preserves_indices(dtype):
    centers = torch.full((8, 3), -3, dtype=dtype)
    labels = torch.arange(8) + 100
    before = centers.clone()

    result = sort_centers(centers, labels)

    torch.testing.assert_close(result.sort().values, labels)
    torch.testing.assert_close(centers, before)


def _export(means, format, sh0=None):
    n = len(means)
    quats = means.new_zeros((n, 4))
    quats[:, 0] = 1
    if sh0 is None:
        sh0 = means.new_zeros((n, 1, 3))
    return export_splats(
        means=means,
        scales=means.new_zeros((n, 3)),
        quats=quats,
        opacities=means.new_ones(n),
        sh0=sh0,
        shN=means.new_zeros((n, 3, 3)),
        format=format,
    )


def _read_compressed_positions(data):
    # Decode the public byte format independently, using its 11/10/11-bit
    # position fields and each group of 256 vertices' float32 bounds.
    header, payload = data.split(b"end_header\n", 1)
    lines = header.decode("ascii").splitlines()
    chunk_count = int(
        next(line for line in lines if line.startswith("element chunk ")).split()[-1]
    )
    vertex_count = int(
        next(line for line in lines if line.startswith("element vertex ")).split()[-1]
    )
    chunk_bytes = chunk_count * 18 * 4
    bounds = np.frombuffer(payload[:chunk_bytes], dtype="<f4").reshape(chunk_count, 18)
    vertices = np.frombuffer(
        payload[chunk_bytes : chunk_bytes + vertex_count * 16], dtype="<u4"
    ).reshape(vertex_count, 4)
    packed = vertices[:, 0].astype(np.uint64)
    fractions = np.column_stack(
        [
            (packed >> 21) / 2047,
            ((packed >> 11) & 1023) / 1023,
            (packed & 2047) / 2047,
        ]
    )
    chunk = bounds[np.arange(vertex_count) // 256]
    return chunk[:, :3] + fractions * (chunk[:, 3:6] - chunk[:, :3]), bounds


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("axis,bits", [(0, 11), (1, 10), (2, 11)])
def test_compressed_export_keeps_distant_maximum_out_of_near_chunk(dtype, axis, bits):
    means = torch.zeros((257, 3), dtype=dtype)
    means[:256, axis] = torch.linspace(0, 1, 256, dtype=dtype)
    means[-1, axis] = 100

    decoded, bounds = _read_compressed_positions(_export(means, "ply_compressed"))

    near = decoded[decoded[:, axis] < 2, axis]
    assert len(near) == 256
    original = means[:256, axis].double().numpy()
    distances = np.abs(original[:, None] - near[None, :])
    # The near group spans one unit. Nearest rounding to b bits has absolute
    # error at most 1/(2*(2**b - 1)); a far point must not expand that range.
    error = max(distances.min(axis=0).max(), distances.min(axis=1).max())
    assert error <= 0.5 / ((1 << bits) - 1) + 2e-7
    np.testing.assert_array_equal(decoded[-1], means[-1].numpy())
    assert bounds[0, axis + 3] == 1
    assert bounds[1, axis] == 100


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_splat_export_preserves_feature_alignment_in_morton_order(dtype):
    means, grid = _corners((0, 1, 2), dtype)
    sh0 = torch.arange(24, dtype=dtype).reshape(8, 1, 3) / 10 - 1
    expected_order = sorted(
        range(len(grid)), key=lambda index: _morton_code(grid[index])
    )

    data = _export(means, "splat", sh0)
    records = np.frombuffer(
        data,
        dtype=np.dtype(
            [
                ("position", "<f4", (3,)),
                ("scale", "<f4", (3,)),
                ("color", "u1", (4,)),
                ("rotation", "u1", (4,)),
            ]
        ),
    )

    np.testing.assert_array_equal(records["position"], means.numpy()[expected_order])
    np.testing.assert_array_equal(records["scale"], np.ones((8, 3)))
    expected_rgb = (
        ((sh0[:, 0].numpy() * 0.28209479177387814 + 0.5) * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )
    np.testing.assert_array_equal(records["color"][:, :3], expected_rgb[expected_order])
