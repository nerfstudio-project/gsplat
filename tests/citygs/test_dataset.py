import os

import numpy as np
import torch

from citygs.scene.dataset import ManifestDataset, _LRUImageCache
from citygs.scene.manifest import SceneManifest


def _write_images(manifest: SceneManifest):
    import imageio.v2 as imageio

    rng = np.random.default_rng(0)
    for cam in manifest.cameras:
        path = os.path.join(manifest.data_dir, cam.image)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.imwrite(
            path, rng.integers(0, 255, size=(cam.height, cam.width, 3), dtype=np.uint8)
        )


def test_lru_cache_bounds():
    cache = _LRUImageCache(2)
    a, b, c = (np.zeros(1), np.ones(1), np.full(1, 2.0))
    cache.put("a", a)
    cache.put("b", b)
    assert cache.get("a") is a  # touch a -> b becomes LRU
    cache.put("c", c)
    assert cache.get("b") is None
    assert cache.get("a") is a and cache.get("c") is c


def test_dataset_basic(toy_manifest):
    _write_images(toy_manifest)
    ds = ManifestDataset(toy_manifest, toy_manifest.train_indices())
    assert len(ds) == len(toy_manifest.train_indices())
    item = ds[0]
    assert item["image"].shape == (48, 64, 3)
    assert item["K"].shape == (3, 3)
    assert item["camtoworld"].shape == (4, 4)
    # image_id is the global manifest index, not the local one.
    assert item["image_id"] == int(toy_manifest.train_indices()[0])


def test_dataset_downsample_scales_K(toy_manifest):
    _write_images(toy_manifest)
    ds1 = ManifestDataset(toy_manifest, [1])
    ds2 = ManifestDataset(toy_manifest, [1], downsample=2)
    i1, i2 = ds1[0], ds2[0]
    assert i2["image"].shape[0] == i1["image"].shape[0] // 2
    ratio = i1["K"][0, 0] / i2["K"][0, 0]
    assert abs(ratio - 2.0) < 0.05


def test_dataset_subset_indices(toy_manifest):
    _write_images(toy_manifest)
    subset = [3, 7, 11]
    ds = ManifestDataset(toy_manifest, subset)
    assert len(ds) == 3
    assert [ds[i]["image_id"] for i in range(3)] == subset


def test_dataset_sparse_depths(toy_manifest, tmp_path):
    _write_images(toy_manifest)
    m = toy_manifest
    # Points on the ground plane under the cameras; each camera sees the
    # points closest to its nadir.
    xyz = np.stack([c.camtoworld_np()[:3, 3] * [1, 1, 0] for c in m.cameras]).astype(
        np.float32
    )
    rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
    indices = np.arange(len(xyz), dtype=np.int32)  # camera i sees point i
    offsets = np.arange(len(xyz) + 1, dtype=np.int64)
    np.savez(
        os.path.join(m._dir, "points.npz"),
        xyz=xyz,
        rgb=rgb,
        cam_point_indices=indices,
        cam_point_offsets=offsets,
    )
    m.points_file = "points.npz"
    path = os.path.join(m._dir, "scene.json")
    m.save(path)
    m = SceneManifest.load(path)

    ds = ManifestDataset(m, [0], load_depths=True)
    item = ds[0]
    # Camera 0 looks straight down from z~1 at its nadir point.
    assert item["points"].shape[0] == 1
    assert abs(item["depths"][0].item() - m.cameras[0].camtoworld_np()[2, 3]) < 1e-4
    # The point projects to the image center.
    cx, cy = m.cameras[0].K_np()[0, 2], m.cameras[0].K_np()[1, 2]
    np.testing.assert_allclose(item["points"][0].numpy(), [cx, cy], atol=1e-3)
