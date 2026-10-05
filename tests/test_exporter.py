# SPDX-License-Identifier: Apache-2.0
import struct

import pytest
import torch

PlyData = pytest.importorskip("plyfile").PlyData

from gsplat.exporter import export_splats


@pytest.mark.parametrize("format", ["ply", "ply_compressed", "splat"])
@pytest.mark.parametrize("case", ["empty", "invalid", "valid"])
def test_export_splats_handles_empty_and_filtered_inputs(tmp_path, format, case):
    count = 0 if case == "empty" else 2
    means = torch.arange(count * 3).reshape(count, 3).float()
    if case == "invalid":
        means[:] = float("nan")
    path = tmp_path / f"scene.{format}"
    export_splats(
        means=means,
        scales=torch.zeros(count, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        opacities=torch.zeros(count),
        sh0=torch.zeros(count, 1, 3),
        shN=torch.zeros(count, 0, 3),
        save_to=str(path),
        format=format,
    )
    expected_count = count if case == "valid" else 0
    if format == "splat":
        assert path.stat().st_size == expected_count * 32
        if expected_count:
            assert struct.unpack("<fff", path.read_bytes()[:12]) == (0.0, 1.0, 2.0)
    else:
        ply = PlyData.read(path)
        assert len(ply["vertex"]) == expected_count
        if format == "ply_compressed":
            assert len(ply["chunk"]) == int(expected_count > 0)


def test_compressed_export_handles_opacity_filter_removing_every_splat(tmp_path):
    path = tmp_path / "scene.ply"
    export_splats(
        means=torch.zeros(2, 3),
        scales=torch.zeros(2, 3),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        opacities=torch.full((2,), -20.0),
        sh0=torch.zeros(2, 1, 3),
        shN=torch.zeros(2, 0, 3),
        save_to=str(path),
        format="ply_compressed",
    )
    ply = PlyData.read(path)
    assert len(ply["vertex"]) == len(ply["chunk"]) == 0
