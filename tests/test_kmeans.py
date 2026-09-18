# SPDX-FileCopyrightText: Copyright 2024-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Tests for the builtin K-means backend of PngCompression (CPU only, no TorchPQ).

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import json
import os
import types

import numpy as np
import pytest
import torch

from gsplat.compression.kmeans import weighted_kmeans
from gsplat.compression.png_compression import (
    PngCompression,
    _compress_kmeans,
    _decompress_kmeans,
    _kmeans_weights,
)


def _points(n=512, d=6, seed=0):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def _splats(n=256, seed=3):
    g = torch.Generator().manual_seed(seed)
    return {
        "means": torch.randn(n, 3, generator=g),
        "scales": torch.randn(n, 3, generator=g) - 3,
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.randn(n, generator=g),
        "sh0": torch.randn(n, 1, 3, generator=g),
        "shN": torch.randn(n, 15, 3, generator=g) * 0.2,
    }


class _StubKMeans:
    """Stand-in for torchpq.clustering.KMeans: a fixed clustering, so the encoder is what
    the test compares, not the clustering."""

    def __init__(self, n_clusters, distance="euclidean", verbose=False, **kwargs):
        self.n_clusters = n_clusters
        self.centroids = None

    def fit(self, x):
        d, n = x.shape
        g = torch.Generator().manual_seed(0)
        self.centroids = x[
            :, torch.randperm(n, generator=g)[: self.n_clusters]
        ].contiguous()
        return torch.randint(0, self.n_clusters, (n,), generator=g)


@pytest.fixture
def stub_torchpq(monkeypatch):
    stub = types.ModuleType("torchpq")
    stub.clustering = types.ModuleType("torchpq.clustering")
    stub.clustering.KMeans = _StubKMeans
    monkeypatch.setitem(__import__("sys").modules, "torchpq", stub)
    monkeypatch.setitem(
        __import__("sys").modules, "torchpq.clustering", stub.clustering
    )
    return stub


def _reference_compress_kmeans(
    compress_dir,
    param_name,
    params,
    n_clusters=65536,
    quantization=6,
    eps=1e-6,
    verbose=True,
):
    """The encoder as it was before the backend option, to pin the file format."""
    from torchpq.clustering import KMeans

    kmeans = KMeans(n_clusters=n_clusters, distance="manhattan", verbose=verbose)
    x = params.reshape(params.shape[0], -1).permute(1, 0).contiguous()
    labels = kmeans.fit(x)
    labels = labels.detach().cpu().numpy()
    centroids = kmeans.centroids.permute(1, 0)

    mins = torch.min(centroids) + eps
    maxs = torch.max(centroids)
    centroids_norm = (centroids - mins) / (maxs - mins)
    centroids_norm = centroids_norm.detach().cpu().numpy()
    centroids_quant = (
        (centroids_norm * (2**quantization - 1)).round().astype(np.uint8)
    )
    labels = labels.astype(np.uint16)
    np.savez_compressed(
        os.path.join(compress_dir, f"{param_name}.npz"),
        centroids=centroids_quant,
        labels=labels,
    )
    return {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "quantization": quantization,
    }


def test_weights_of_one_match_unweighted():
    x = _points()
    a = weighted_kmeans(x, 16, weights=None, max_iter=20)
    b = weighted_kmeans(x, 16, weights=torch.ones(len(x)), max_iter=20)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_deterministic_per_seed():
    x = _points()
    a = weighted_kmeans(x, 16, seed=1, max_iter=20)
    b = weighted_kmeans(x, 16, seed=1, max_iter=20)
    c = weighted_kmeans(x, 16, seed=2, max_iter=20)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])
    # and independent of the caller's numpy state, which it also leaves alone
    np.random.seed(123)
    before = np.random.get_state()
    d = weighted_kmeans(x, 16, seed=1, max_iter=20)
    after = np.random.get_state()
    assert torch.equal(a[0], d[0])
    assert (
        before[0] == after[0]
        and np.array_equal(before[1], after[1])
        and before[2:] == after[2:]
    )


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 512])
def test_chunk_size_does_not_change_the_result(chunk_size):
    x = _points()
    w = torch.rand(len(x), generator=torch.Generator().manual_seed(1)) + 0.05
    ref = weighted_kmeans(x, 16, weights=w, max_iter=20, chunk_size=512)
    got = weighted_kmeans(x, 16, weights=w, max_iter=20, chunk_size=chunk_size)
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])


def test_empty_clusters_become_zero():
    # Only 8 distinct points, so at most 8 of the 32 clusters can hold one.
    x = _points(n=8, d=3).repeat(8, 1)
    centroids, labels = weighted_kmeans(x, 32, max_iter=20)
    empty = torch.tensor([k for k in range(32) if (labels == k).sum() == 0])
    assert len(empty) >= 32 - 8
    assert torch.equal(centroids[empty], torch.zeros(len(empty), x.shape[1]))


def test_weighting_moves_centroids():
    x = _points(n=128, d=4)
    w = torch.zeros(len(x))
    w[:16] = 1.0  # only the first points count
    centroids, labels = weighted_kmeans(x, 4, weights=w, max_iter=20)
    unweighted, _ = weighted_kmeans(x, 4, max_iter=20)
    assert not torch.allclose(centroids, unweighted)
    for k in range(4):
        members = (labels == k) & (w > 0)
        if members.any():
            assert torch.allclose(centroids[k], x[members].mean(0), atol=1e-5)


def test_argument_validation():
    x = _points(n=32, d=2)
    with pytest.raises(ValueError):
        weighted_kmeans(x, 33)
    with pytest.raises(ValueError):
        weighted_kmeans(x, 0)
    with pytest.raises(ValueError):
        weighted_kmeans(x.reshape(-1), 4)
    with pytest.raises(ValueError):
        weighted_kmeans(x, 4, weights=torch.ones(31))
    with pytest.raises(ValueError):
        weighted_kmeans(x, 4, weights=-torch.ones(32))
    for chunk_size in (0, -1):
        with pytest.raises(ValueError):
            weighted_kmeans(x, 4, chunk_size=chunk_size)


def test_kmeans_weights_formulas():
    splats = {
        "opacities": torch.tensor([0.0, 2.0, -1.0]),
        "scales": torch.tensor(
            [[-1.0, -2.0, -3.0], [0.0, 1.0, -5.0], [-4.0, -4.0, -4.0]]
        ),
    }
    assert _kmeans_weights(splats, None) is None
    opacity = torch.sigmoid(splats["opacities"].double())
    assert torch.allclose(_kmeans_weights(splats, "opacity"), opacity)
    top2 = torch.tensor([-3.0, 1.0, -8.0], dtype=torch.float64)
    assert torch.allclose(
        _kmeans_weights(splats, "opacity_area"), opacity * torch.exp(top2)
    )
    with pytest.raises(ValueError):
        _kmeans_weights(splats, "nonsense")


def test_builtin_backend_round_trip(tmp_path):
    splats = _splats()
    meta = _compress_kmeans(
        str(tmp_path), "shN", splats["shN"], n_clusters=32, backend="builtin"
    )
    assert set(meta) == {"shape", "dtype", "mins", "maxs", "quantization"}
    out = _decompress_kmeans(str(tmp_path), "shN", meta)
    assert out.shape == splats["shN"].shape and out.dtype == splats["shN"].dtype
    with np.load(os.path.join(tmp_path, "shN.npz")) as npz:
        assert npz["centroids"].dtype == np.uint8 and npz["labels"].dtype == np.uint16
        assert npz["centroids"].shape == (32, 45)
    # every decoded splat is one of the 32 centroids
    assert len(torch.unique(out.reshape(len(out), -1), dim=0)) <= 32


def test_png_compression_chunk_size(tmp_path, monkeypatch):
    """kmeans_chunk_size reaches weighted_kmeans and does not change the output."""
    import gsplat.compression.png_compression as png_compression

    seen = []
    original = png_compression.weighted_kmeans

    def recording_kmeans(x, n_clusters, **kwargs):
        seen.append(kwargs["chunk_size"])
        # 16 clusters for 256 splats, so the clustering is not trivial (PngCompression asks
        # for 65536, which the builtin backend clamps to the number of splats)
        return original(x, min(n_clusters, 16), **kwargs)

    monkeypatch.setattr(png_compression, "weighted_kmeans", recording_kmeans)
    splats = _splats()
    files = {}
    for chunk_size in (4096, 7):
        out = tmp_path / f"chunk_{chunk_size}"
        out.mkdir()
        method = PngCompression(
            use_sort=False,
            verbose=False,
            kmeans_backend="builtin",
            kmeans_weighting="opacity_area",
            kmeans_chunk_size=chunk_size,
        )
        method.compress(str(out), {k: v.clone() for k, v in splats.items()})
        files[chunk_size] = {p.name: p.read_bytes() for p in sorted(out.iterdir())}
    assert seen == [4096, 7]
    assert PngCompression().kmeans_chunk_size == 4096
    with np.load(tmp_path / "chunk_4096" / "shN.npz") as a, np.load(
        tmp_path / "chunk_7" / "shN.npz"
    ) as b:
        assert len(np.unique(a["labels"])) > 1
        assert np.array_equal(a["centroids"], b["centroids"])
        assert np.array_equal(a["labels"], b["labels"])
    assert files[4096] == files[7]


def test_builtin_backend_clamps_n_clusters(tmp_path):
    splats = _splats(n=100)
    meta = _compress_kmeans(str(tmp_path), "shN", splats["shN"], backend="builtin")
    with np.load(os.path.join(tmp_path, "shN.npz")) as npz:
        assert npz["centroids"].shape[0] == 100  # min(65536, N)
    assert _decompress_kmeans(str(tmp_path), "shN", meta).shape == splats["shN"].shape


def test_unknown_backend(tmp_path):
    with pytest.raises(ValueError):
        _compress_kmeans(str(tmp_path), "shN", _splats()["shN"], backend="nonsense")


def test_weights_rejected_by_the_torchpq_backend(tmp_path):
    splats = _splats()
    with pytest.raises(ValueError):
        _compress_kmeans(
            str(tmp_path), "shN", splats["shN"], weights=torch.ones(len(splats["shN"]))
        )


def test_defaults_match_the_previous_encoder(tmp_path, stub_torchpq):
    """The default backend must write exactly the same file as before the backend option."""
    splats = _splats()
    ref_dir, new_dir = tmp_path / "ref", tmp_path / "new"
    ref_dir.mkdir()
    new_dir.mkdir()
    ref_meta = _reference_compress_kmeans(
        str(ref_dir), "shN", splats["shN"], n_clusters=32, verbose=False
    )
    new_meta = _compress_kmeans(
        str(new_dir), "shN", splats["shN"], n_clusters=32, verbose=False
    )
    assert new_meta == ref_meta
    assert (
        open(ref_dir / "shN.npz", "rb").read() == open(new_dir / "shN.npz", "rb").read()
    )


def test_png_compression_builtin_backend_without_torchpq(tmp_path, monkeypatch):
    """The builtin backend must not import TorchPQ, and must round-trip on the CPU."""
    monkeypatch.setitem(__import__("sys").modules, "torchpq", None)
    monkeypatch.setitem(__import__("sys").modules, "torchpq.clustering", None)
    splats = _splats(n=64)
    for weighting in (None, "opacity", "opacity_area"):
        out = tmp_path / f"w_{weighting}"
        out.mkdir()
        method = PngCompression(
            use_sort=False,
            verbose=False,
            kmeans_backend="builtin",
            kmeans_weighting=weighting,
        )
        method.compress(str(out), {k: v.clone() for k, v in splats.items()})
        decompressed = method.decompress(str(out))
        for name, value in splats.items():
            assert decompressed[name].shape == value.shape
            assert torch.isfinite(decompressed[name]).all()
        meta = json.loads(open(out / "meta.json").read())
        assert meta["shN"]["quantization"] == 6

    missing = tmp_path / "torchpq_missing"
    missing.mkdir()
    with pytest.raises(ImportError):
        PngCompression(use_sort=False, verbose=False).compress(
            str(missing), {k: v.clone() for k, v in splats.items()}
        )


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
