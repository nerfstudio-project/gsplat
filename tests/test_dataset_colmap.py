# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the image cache of ``examples/datasets/colmap.py``, as
``examples/simple_trainer.py`` uses it with ``--cache_images``."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

imageio = pytest.importorskip("imageio.v2")
pytest.importorskip("cv2")
pytest.importorskip("pycolmap")
pytest.importorskip("piexif")

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from datasets.colmap import Dataset  # noqa: E402


def _make_parser(root: Path, n_images: int = 12, height: int = 12, width: int = 16):
    """A minimal stand-in for `Parser`: two cameras, one of them distorted."""
    rng = np.random.default_rng(0)
    image_paths, image_names = [], []
    for i in range(n_images):
        path = root / f"{i:03d}.png"
        imageio.imwrite(path, rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        image_paths.append(str(path))
        image_names.append(path.name)
    K = np.array([[20.0, 0, width / 2], [0, 20.0, height / 2], [0, 0, 1]])
    # camera 1 is "distorted": an identity remap followed by a 2-pixel ROI crop
    mapy, mapx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    points = rng.normal(size=(200, 3)) + np.array([0.0, 0.0, 5.0])
    return SimpleNamespace(
        image_names=image_names,
        image_paths=image_paths,
        test_every=4,
        camera_ids=[i % 2 for i in range(n_images)],
        camera_indices=[i % 2 for i in range(n_images)],
        Ks_dict={0: K, 1: K},
        params_dict={0: np.empty(0), 1: np.array([0.1, 0.0, 0.0, 0.0])},
        mapx_dict={1: mapx},
        mapy_dict={1: mapy},
        roi_undist_dict={1: (2, 2, width - 4, height - 4)},
        mask_dict={0: None, 1: None},
        camtoworlds=np.tile(np.eye(4), (n_images, 1, 1)),
        exposure_values=[None] * n_images,
        point_indices={name: np.arange(200) for name in image_names},
        points=points,
    )


def _assert_same_item(cached, uncached):
    assert cached.keys() == uncached.keys()
    assert cached["image"].dtype == torch.uint8
    assert uncached["image"].dtype == torch.float32
    assert torch.equal(cached["image"].float(), uncached["image"])
    for key in cached.keys() - {"image"}:
        if isinstance(cached[key], torch.Tensor):
            assert torch.equal(cached[key], uncached[key]), key
        else:
            assert cached[key] == uncached[key], key


def test_preloaded_images_match_decoded(tmp_path):
    parser = _make_parser(tmp_path)
    uncached = Dataset(parser, split="train")
    cached = Dataset(parser, split="train")
    cached.preload_images(num_workers=2)
    assert len(cached) == len(uncached) == 9
    for i in range(len(cached)):
        _assert_same_item(cached[i], uncached[i])


def test_preloaded_images_with_patches_and_depths(tmp_path):
    parser = _make_parser(tmp_path)
    kwargs = dict(split="train", patch_size=5, load_depths=True)
    uncached = Dataset(parser, **kwargs)
    cached = Dataset(parser, **kwargs)
    cached.preload_images(num_workers=2)
    for i in range(len(cached)):
        np.random.seed(i)
        a = cached[i]
        np.random.seed(i)
        b = uncached[i]
        assert a["image"].shape == (5, 5, 3)
        assert "depths" in a
        _assert_same_item(a, b)


def _image_ids(trainset, steps):
    """Run the training loop's loader and iteration of simple_trainer.py and
    return the image ids it produced."""
    loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    ids = []
    it = iter(loader)
    for _ in range(steps):
        try:
            data = next(it)
        except StopIteration:
            it = iter(loader)
            data = next(it)
        ids.append(int(data["image_id"]))
        torch.rand(1)  # the loop draws from the global RNG between batches
    del it, loader
    return ids


def test_preloading_keeps_the_image_order(tmp_path):
    parser = _make_parser(tmp_path)
    trainset = Dataset(parser, split="train")
    steps = 4 * len(trainset) + 3  # four epochs and the start of a fifth

    torch.manual_seed(42)
    expected = _image_ids(trainset, steps)
    expected_rng = torch.rand(1)

    trainset.preload_images(num_workers=2)
    torch.manual_seed(42)
    assert _image_ids(trainset, steps) == expected
    assert torch.equal(torch.rand(1), expected_rng)
