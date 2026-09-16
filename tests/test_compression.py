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

import pytest
import torch

device = torch.device("cuda:0")


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


@pytest.mark.parametrize(
    "shapes",
    [
        {
            "means": (0, 3),
            "scales": (0, 3),
            "quats": (0, 4),
            "opacities": (0,),
            "sh0": (0, 1, 3),
            "features": (0, 8),
        },
        # sh_degree=0: examples/simple_trainer.py creates shN as colors[:, 1:, :],
        # i.e. [N, 0, 3], while all other parameters are non-empty.
        {
            "means": (16, 3),
            "scales": (16, 3),
            "quats": (16, 4),
            "opacities": (16,),
            "sh0": (16, 1, 3),
            "shN": (16, 0, 3),
        },
    ],
    ids=["no_gaussians", "sh_degree_0"],
)
def test_png_compression_empty_splats(tmp_path, shapes):
    pytest.importorskip("imageio")
    from gsplat.compression import PngCompression

    gen = torch.Generator().manual_seed(0)
    splats = {k: torch.randn(shape, generator=gen) for k, shape in shapes.items()}

    compression_method = PngCompression(use_sort=False, verbose=False)
    compression_method.compress(str(tmp_path), splats)
    splats_c = compression_method.decompress(str(tmp_path))

    for k, shape in shapes.items():
        assert isinstance(splats_c[k], torch.Tensor), k
        assert splats_c[k].shape == shape, k
        assert splats_c[k].dtype == torch.float32, k


def test_png_compression_empty_kmeans(tmp_path):
    # Empty input returns before the torchpq import, so torchpq is not needed.
    from gsplat.compression.png_compression import (
        _compress_kmeans,
        _decompress_kmeans,
    )

    meta = _compress_kmeans(
        str(tmp_path), "shN", torch.zeros(0, 15, 3), n_sidelen=0, verbose=False
    )
    params = _decompress_kmeans(str(tmp_path), "shN", meta)

    assert isinstance(params, torch.Tensor)
    assert params.shape == (0, 15, 3)
    assert params.dtype == torch.float32


if __name__ == "__main__":
    test_png_compression()
