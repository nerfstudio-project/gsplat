"""Manifest-backed torch dataset with a bounded LRU image cache.

Images are decoded lazily and kept in a per-worker LRU cache, so scenes
with thousands of images never need to fit in memory at once. Any subset
of cameras (train split, one block's cameras, ...) is expressed as an
explicit index list — the dataset itself has no notion of blocks.
"""

from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from .manifest import SceneManifest


class _LRUImageCache:
    def __init__(self, max_items: int):
        self.max_items = max_items
        self._data: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def get(self, key: str) -> Optional[np.ndarray]:
        arr = self._data.get(key)
        if arr is not None:
            self._data.move_to_end(key)
        return arr

    def put(self, key: str, arr: np.ndarray) -> None:
        if self.max_items <= 0:
            return
        self._data[key] = arr
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)


class ManifestDataset(torch.utils.data.Dataset):
    """Dataset over an explicit list of manifest camera indices.

    Args:
        manifest: the scene manifest.
        indices: global camera indices to expose.
        load_depths: attach sparse SfM depth samples (projected points) for
            depth supervision. Requires the ingest-produced points.npz.
        downsample: extra on-the-fly downsampling on top of the manifest
            resolution (used by the coarse stage).
        cache_size: LRU capacity in decoded images per dataloader worker.
    """

    def __init__(
        self,
        manifest: SceneManifest,
        indices: Sequence[int],
        load_depths: bool = False,
        downsample: int = 1,
        cache_size: int = 256,
    ):
        self.manifest = manifest
        self.indices = np.asarray(list(indices), dtype=np.int64)
        self.load_depths = load_depths
        self.downsample = max(1, int(downsample))
        self._cache = _LRUImageCache(cache_size)

        self._points = None
        self._depth_index = None
        if load_depths:
            xyz, _, depth_index = manifest.load_points()
            if depth_index is None:
                raise ValueError(
                    "points.npz has no per-camera visibility index; "
                    "re-run ingest to enable depth supervision."
                )
            self._points = xyz
            self._depth_index = depth_index

    def __len__(self) -> int:
        return len(self.indices)

    def _load_image(self, index: int) -> np.ndarray:
        path = self.manifest.image_path(index)
        key = f"{path}@{self.downsample}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        import imageio.v2 as imageio

        image = imageio.imread(path)[..., :3]
        if self.downsample > 1:
            from PIL import Image

            size = (
                round(image.shape[1] / self.downsample),
                round(image.shape[0] / self.downsample),
            )
            image = np.asarray(Image.fromarray(image).resize(size, Image.BICUBIC))
        image = np.require(image, requirements=["C", "W"])
        self._cache.put(key, image)
        return image

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = int(self.indices[item])
        cam = self.manifest.cameras[index]
        image = self._load_image(index)
        K = cam.K_np().copy()
        if self.downsample > 1:
            # Scale to the actual downsampled size (rounding included).
            K[0, :] *= image.shape[1] / cam.width
            K[1, :] *= image.shape[0] / cam.height
        camtoworld = cam.camtoworld_np()

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": index,
        }

        if self.load_depths:
            pt_idx, offsets = self._depth_index
            sel = pt_idx[offsets[index] : offsets[index + 1]]
            points_world = self._points[sel]
            w2c = np.linalg.inv(camtoworld)
            points_cam = (w2c[:3, :3] @ points_world.T + w2c[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            depths = points_cam[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                pts2d = points_proj[:, :2] / points_proj[:, 2:3]
            keep = (
                (depths > 0)
                & (pts2d[:, 0] >= 0)
                & (pts2d[:, 0] < image.shape[1])
                & (pts2d[:, 1] >= 0)
                & (pts2d[:, 1] < image.shape[0])
            )
            data["points"] = torch.from_numpy(pts2d[keep]).float()
            data["depths"] = torch.from_numpy(depths[keep]).float()

        return data
