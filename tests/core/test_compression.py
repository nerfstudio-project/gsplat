# SPDX-FileCopyrightText: Copyright 2024 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import sys
from types import SimpleNamespace

import pytest
import torch

device = torch.device("cuda:0")


def _native_cuda_major() -> int:
    """Return the built extension's CUDA major or skip a CPU-only installation."""

    from gsplat.cuda._wrapper import _build_config

    build_config = _build_config()
    if not build_config:
        pytest.skip("CUDA extension was not built")

    cuda_version = build_config.get("cuda_version")
    assert type(cuda_version) is int
    assert cuda_version >= 1000
    return cuda_version // 1000


def test_native_build_config_records_cuda_version():
    """The loaded extension exposes the toolkit version compiled into it."""

    _native_cuda_major()


@pytest.mark.parametrize("cupy_state", ["missing", "wrong-major"])
def test_png_dependency_failures_are_atomic(monkeypatch, tmp_path, cupy_state):
    """Dependency failures leave caller-owned tensors and output untouched."""

    from gsplat.compression import PngCompression

    cuda_major = _native_cuda_major()
    if cupy_state == "missing":
        monkeypatch.setitem(sys.modules, "cupy", None)
        error = "requires a CUDA-matched CuPy"
    else:
        cupy_cuda_major = cuda_major + 1
        monkeypatch.setattr(torch.version, "cuda", f"{cupy_cuda_major}.0")
        monkeypatch.setitem(
            sys.modules,
            "cupy",
            SimpleNamespace(
                cuda=SimpleNamespace(
                    runtime=SimpleNamespace(
                        runtimeGetVersion=lambda: cupy_cuda_major * 1000
                    )
                )
            ),
        )
        error = (
            f"gsplat was built for CUDA {cuda_major}, but the installed CuPy "
            f"targets CUDA {cupy_cuda_major}"
        )

    splats = {
        "means": torch.ones(4, 3),
        "quats": torch.ones(4, 4),
        "shN": torch.ones(4, 1, 3),
    }
    originals = {name: values.clone() for name, values in splats.items()}

    with pytest.raises(ImportError, match=error):
        PngCompression(use_sort=False).compress(str(tmp_path), splats)

    for name, values in splats.items():
        torch.testing.assert_close(values, originals[name])
    assert not any(tmp_path.iterdir())


def _alignment_fixture(side: int = 8) -> dict[str, torch.Tensor]:
    """Return splat fields whose source row is recoverable after sorting."""

    count = side * side
    source_rows = torch.arange(count, dtype=torch.float32)
    return {
        "means": torch.stack(
            (source_rows, source_rows + 100, source_rows + 200), dim=-1
        ),
        "scales": torch.stack((source_rows + 300, source_rows + 400), dim=-1),
        "quats": torch.stack((source_rows + 500, source_rows + 600), dim=-1),
        "opacities": source_rows + 700,
        "sh0": (source_rows + 800).reshape(count, 1, 1),
        "features": torch.stack((source_rows + 900, source_rows + 1000), dim=-1),
    }


def test_flas_sort_preserves_alignment_and_seed():
    """FLAS returns one reproducible permutation shared by every field."""

    from gsplat.compression.sort import sort_splats

    results = []
    for _ in range(2):
        torch.manual_seed(7)
        original = _alignment_fixture()
        sorted_splats = sort_splats(
            {name: values.clone() for name, values in original.items()},
            verbose=False,
        )
        permutation = sorted_splats["means"][:, 0].to(torch.long)
        assert torch.equal(
            torch.sort(permutation).values, torch.arange(len(permutation))
        )
        for name, values in original.items():
            torch.testing.assert_close(sorted_splats[name], values[permutation])
        results.append(sorted_splats)

    for name in results[0]:
        torch.testing.assert_close(results[0][name], results[1][name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_png_compression():
    from gsplat.compression import PngCompression

    torch.manual_seed(42)

    # Prepare Gaussians
    N = 100000
    splats = torch.nn.ParameterDict(
        {
            "means": torch.randn(N, 3),
            "scales": torch.randn(N, 3),
            "quats": torch.randn(N, 4),
            "opacities": torch.randn(N),
            "sh0": torch.randn(N, 1, 3),
            "shN": torch.randn(N, 24, 3),
            "features": torch.randn(N, 128),
        }
    ).to(device)
    compress_dir = "/tmp/gsplat/compression"

    compression_method = PngCompression()
    # run compression and save the compressed files to compress_dir
    compression_method.compress(compress_dir, splats)
    # decompress the compressed files
    splats_c = compression_method.decompress(compress_dir)


if __name__ == "__main__":
    test_png_compression()
