# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for transparent COLMAP inputs (issue #372)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

imageio = pytest.importorskip("imageio.v2")
pytest.importorskip("cv2")
pytest.importorskip("pycolmap")
pytest.importorskip("piexif")

_EXAMPLES = str(Path(__file__).resolve().parents[1] / "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

from tests.test_dataset_colmap import _make_parser
from datasets.colmap import Dataset, _resize_image_folder, composite_background


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("crop", [None, 5])
def test_rgba_alpha_stays_aligned_through_cache_remap_and_crop(
    tmp_path, cached, crop, split
):
    parser = _make_parser(tmp_path)
    for path in parser.image_paths:
        rgb = imageio.imread(path)
        alpha = np.arange(rgb.shape[0] * rgb.shape[1], dtype=np.uint8).reshape(
            *rgb.shape[:2], 1
        )
        imageio.imwrite(path, np.concatenate([rgb, alpha], axis=-1))
    dataset = Dataset(parser, split=split, load_alpha=True, patch_size=crop)
    if cached:
        dataset.preload_images(num_workers=2)
    np.random.seed(7)
    item = dataset[0]
    np.random.seed(7)
    index = dataset.indices[0]
    source = torch.from_numpy(imageio.imread(parser.image_paths[index]))
    if parser.camera_ids[index] == 1:
        source = source[2:-2, 2:-2]
    if crop is not None:
        h, w = source.shape[:2]
        x = np.random.randint(0, max(w - crop, 1))
        y = np.random.randint(0, max(h - crop, 1))
        source = source[y : y + crop, x : x + crop]
    assert item["image"].shape[-1] == 3
    torch.testing.assert_close(item["image"].float(), source[..., :3].float())
    torch.testing.assert_close(item["alpha"], source[..., 3:4].float() / 255)


def test_rgb_inputs_remain_opaque_and_opt_in(tmp_path):
    parser = _make_parser(tmp_path)
    assert "alpha" not in Dataset(parser)[0]
    assert torch.all(Dataset(parser, load_alpha=True)[0]["alpha"] == 1)


def test_downsampling_retains_alpha(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    imageio.imwrite(source / "rgba.png", np.full((12, 16, 4), 127, dtype=np.uint8))
    dest = tmp_path / "images_2_png"
    _resize_image_folder(str(source), str(dest), 2)
    assert imageio.imread(dest / "rgba.png").shape == (6, 8, 4)


@pytest.mark.parametrize("alpha", [0.0, 0.4, 1.0])
def test_matching_rgba_has_zero_loss_and_preserves_gradients(alpha):
    rgb = torch.tensor([0.2, 0.4, 0.8])
    a = torch.tensor([alpha])
    color = (rgb * a).clone().requires_grad_()
    pred_alpha = a.clone().requires_grad_()
    for background in [None, torch.tensor([0.9, 0.5, 0.1])]:
        prediction, target = composite_background(color, pred_alpha, rgb, a, background)
        torch.testing.assert_close(prediction, target)
    prediction.sum().backward()
    assert torch.isfinite(color.grad).all() and torch.isfinite(pred_alpha.grad).all()


def test_rgb_background_behavior_is_unchanged():
    rgb = torch.rand(2, 4, 3)
    alpha = torch.rand(2, 4, 1)
    bg = torch.rand(3)
    prediction, target = composite_background(rgb, alpha, rgb, background=bg)
    torch.testing.assert_close(prediction, rgb + bg * (1 - alpha))
    assert target is rgb


def test_rgba_preloading_preserves_crop_rng_and_cached_storage(tmp_path):
    parser = _make_parser(tmp_path)
    for path in parser.image_paths:
        rgb = imageio.imread(path)
        imageio.imwrite(path, np.concatenate([rgb, rgb[..., :1]], axis=-1))
    uncached = Dataset(parser, load_alpha=True, patch_size=5)
    cached = Dataset(parser, load_alpha=True, patch_size=5)
    np.random.seed(11)
    torch.manual_seed(11)
    expected = uncached[0]
    expected_np, expected_torch = np.random.rand(), torch.rand(1)
    np.random.seed(11)
    torch.manual_seed(11)
    cached.preload_images(num_workers=2)
    original = cached.images[0].clone()
    actual = cached[0]
    torch.testing.assert_close(actual["image"].float(), expected["image"])
    torch.testing.assert_close(actual["alpha"], expected["alpha"])
    torch.testing.assert_close(actual["K"], expected["K"])
    assert np.random.rand() == expected_np
    assert torch.equal(torch.rand(1), expected_torch)
    assert torch.equal(cached.images[0], original)


def test_rgba_downsampling_preserves_legacy_rgb_values(tmp_path):
    from PIL import Image

    source = tmp_path / "images"
    source.mkdir()
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    image[:, 0, :] = [255, 0, 0, 0]
    image[:, 1, :] = [0, 255, 0, 255]
    imageio.imwrite(source / "rgba.png", image)
    destination = tmp_path / "images_2_png"
    _resize_image_folder(str(source), str(destination), 2)
    actual = imageio.imread(destination / "rgba.png")
    expected_rgb = np.asarray(
        Image.fromarray(image[..., :3]).resize((1, 1), Image.BICUBIC)
    )
    np.testing.assert_array_equal(actual[..., :3], expected_rgb)
