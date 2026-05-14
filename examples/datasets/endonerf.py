"""EndoNeRF / SCARED dataset loader (example).

Public API:

- :class:`EndoNeRFParser` — parses an EndoNeRF directory layout into
  in-memory arrays (poses, intrinsics, frame paths, times, splits).
- :class:`EndoNeRFDataset` — torch-friendly random-access view returning
  per-frame ``{image, depth, mask, camtoworld, K, time}`` dicts.

Follows the same shape as :mod:`examples.datasets.colmap` and
:mod:`examples.datasets.ncore`. Ported from
``holohub/applications/surgical_scene_recon/training/scene/endo_loader.py``
(EndoNeRF only — SCARED has a different on-disk layout and is currently
a stub that raises :class:`NotImplementedError`).

Expected on-disk layout (``dataset_type="endonerf"``):

.. code-block::

    <data_dir>/
      poses_bounds.npy
      images/   000000.png 000001.png ...
      depth/    000000.png 000001.png ...
      masks/    000000.png 000001.png ...

Conventions matched against G-SHARP:

- ``poses_bounds.npy`` row format: 17 floats per frame — 15 = ``poses[3, 5]``
  flattened (``[R | t | (H, W, focal)]``); last 2 = near/far bounds.
- ``masks/`` is in **tissue=1, tool=0 on-disk** convention; the dataset
  inverts to **tissue=1, tool=0 → tool=1, tissue=0** at load time so the
  resulting mask matches ``gsplat.regularizers.create_invisible_mask``
  (tool=1) and the trainer's mask-aware loss helpers (mask=1 means keep).
  Wait — re-check: G-SHARP does ``mask = 1 - np.array(mask) / 255.0`` in
  the dataset (`endo_loader.py:179`) which inverts to **tool=1**.
  Confirmed: the dataset returns a *tool* mask.
- ``time = idx / n_frames`` per frame.
- ``test_every`` controls train/test split: frame ``i`` is test if
  ``(i - 1) % test_every == 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

import numpy as np
import torch
from PIL import Image

DatasetType = Literal["endonerf", "scared"]


@dataclass
class EndoNeRFParser:
    """Parse an EndoNeRF directory into in-memory arrays.

    Args:
        data_dir: Path to the dataset root (contains ``poses_bounds.npy``,
            ``images/``, ``depth/``, ``masks/``).
        dataset_type: ``"endonerf"`` (default) or ``"scared"``. The latter
            is a stub — it routes to a clear ``NotImplementedError`` so
            callers can detect the unsupported path.
        test_every: Stride used to derive ``test_idxs`` (frame ``i`` is a
            test frame iff ``(i - 1) % test_every == 0``). Default ``8``
            matches G-SHARP.

    Populated attributes (after ``__post_init__``):

    - ``height``, ``width``, ``focal``: image dimensions and focal length
      from ``poses_bounds.npy``.
    - ``K``: ``(3, 3)`` pinhole intrinsics.
    - ``bounds``: ``(N, 2)`` near/far per frame.
    - ``camtoworlds``: ``(N, 4, 4)`` camera-to-world transforms.
    - ``times``: ``(N,)`` per-frame timestamps in ``[0, 1)``.
    - ``image_paths`` / ``depth_paths`` / ``mask_paths``: sorted PNG paths.
    - ``train_idxs`` / ``test_idxs`` / ``video_idxs``: split index lists.

    Raises:
        FileNotFoundError: if ``data_dir`` or ``poses_bounds.npy`` is missing.
        ValueError: on count mismatch between poses and frames, or if the
            first mask is non-binary (sanity check matching the
            ``test_endonerf_parser_non_binary_mask_rejected`` contract).
        NotImplementedError: when ``dataset_type="scared"``.
    """

    data_dir: Path
    dataset_type: DatasetType = "endonerf"
    test_every: int = 8

    # Populated in __post_init__ — declared here so they pickle / serialize cleanly.
    height: int = field(init=False, default=0)
    width: int = field(init=False, default=0)
    focal: float = field(init=False, default=0.0)
    K: np.ndarray = field(
        init=False, default_factory=lambda: np.zeros((3, 3), dtype=np.float32)
    )
    bounds: np.ndarray = field(
        init=False, default_factory=lambda: np.zeros((0, 2), dtype=np.float32)
    )
    camtoworlds: np.ndarray = field(
        init=False, default_factory=lambda: np.zeros((0, 4, 4), dtype=np.float32)
    )
    times: np.ndarray = field(
        init=False, default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )
    image_paths: List[Path] = field(init=False, default_factory=list)
    depth_paths: List[Path] = field(init=False, default_factory=list)
    mask_paths: List[Path] = field(init=False, default_factory=list)
    train_idxs: List[int] = field(init=False, default_factory=list)
    test_idxs: List[int] = field(init=False, default_factory=list)
    video_idxs: List[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")

        if self.dataset_type == "endonerf":
            self._load_endonerf()
        elif self.dataset_type == "scared":
            raise NotImplementedError(
                "EndoNeRFParser: dataset_type='scared' routing is recognised but "
                "the SCARED on-disk layout (JSON calibrations under data/frame_data/) "
                "isn't ported yet — see "
                "holohub/applications/surgical_scene_recon/training/scene/endo_loader.py "
                "class SCARED_Dataset for the reference. PRs welcome."
            )
        else:
            raise ValueError(
                f"EndoNeRFParser: unknown dataset_type {self.dataset_type!r}. "
                f"Expected 'endonerf' or 'scared'."
            )

        n = len(self.image_paths)
        self.train_idxs = [i for i in range(n) if (i - 1) % self.test_every != 0]
        self.test_idxs = [i for i in range(n) if (i - 1) % self.test_every == 0]
        self.video_idxs = list(range(n))

    def _load_endonerf(self) -> None:
        poses_bounds_path = self.data_dir / "poses_bounds.npy"
        if not poses_bounds_path.exists():
            raise FileNotFoundError(
                f"EndoNeRFParser: missing poses_bounds.npy at {poses_bounds_path}."
            )

        poses_arr = np.load(str(poses_bounds_path))
        n = poses_arr.shape[0]
        # poses_arr layout: (N, 17) = (N, 15) flattened (3 x 5 = 15 pose entries) + (N, 2) bounds.
        poses = poses_arr[:, :15].reshape(n, 3, 5)
        self.bounds = poses_arr[:, 15:].astype(np.float32)

        h, w, focal = poses[0, :, -1]
        self.height = int(h)
        self.width = int(w)
        self.focal = float(focal)
        self.K = np.array(
            [
                [focal, 0.0, self.width // 2],
                [0.0, focal, self.height // 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Drop the [H, W, focal] column → (N, 3, 4) → pad to (N, 4, 4) homogeneous.
        c2w_3x4 = poses[..., :4]
        bottom_row = np.broadcast_to(
            np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (n, 1, 4)
        )
        self.camtoworlds = np.concatenate([c2w_3x4, bottom_row], axis=1).astype(
            np.float32
        )

        self.times = np.arange(n, dtype=np.float32) / n

        self.image_paths = sorted((self.data_dir / "images").glob("*.png"))
        self.depth_paths = sorted((self.data_dir / "depth").glob("*.png"))
        self.mask_paths = sorted((self.data_dir / "masks").glob("*.png"))

        for name, paths in (
            ("images", self.image_paths),
            ("depth", self.depth_paths),
            ("masks", self.mask_paths),
        ):
            if len(paths) != n:
                raise ValueError(
                    f"EndoNeRFParser: {name}/ has {len(paths)} files but "
                    f"poses_bounds.npy has {n} frames."
                )

        # Sanity check: the first mask must be binary on disk (values in {0, 255}).
        _validate_mask_binary(self.mask_paths[0])


class EndoNeRFDataset(torch.utils.data.Dataset):
    """Random-access view over the parser, filtered by split.

    Args:
        parser: A fully-initialised :class:`EndoNeRFParser`.
        split: ``"train"``, ``"test"``, or ``"video"`` (default ``"train"``).

    Each item is a dict with keys:

    - ``image``: ``(H, W, 3)`` float32 in ``[0, 1]``.
    - ``depth``: ``(H, W)`` float32 metric depth (0 = no measurement).
    - ``mask``: ``(H, W)`` float32 tool mask (1 = tool, 0 = tissue),
      matching G-SHARP's ``1 - mask/255`` convention at
      ``endo_loader.py:179``.
    - ``camtoworld``: ``(4, 4)`` float32 camera-to-world.
    - ``K``: ``(3, 3)`` float32 intrinsics.
    - ``time``: scalar float32 timestamp in ``[0, 1)``.

    Raises:
        ValueError: if *split* is not one of ``"train"``, ``"test"``, ``"video"``.
    """

    def __init__(self, parser: EndoNeRFParser, split: str = "train") -> None:
        self.parser = parser
        self.split = split
        if split == "train":
            self.indices: Sequence[int] = parser.train_idxs
        elif split == "test":
            self.indices = parser.test_idxs
        elif split == "video":
            self.indices = parser.video_idxs
        else:
            raise ValueError(
                f"EndoNeRFDataset: unknown split {split!r}; "
                f"expected 'train', 'test', or 'video'."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        idx = self.indices[index]

        image = (
            np.array(Image.open(self.parser.image_paths[idx])).astype(np.float32)
            / 255.0
        )
        depth = np.array(Image.open(self.parser.depth_paths[idx])).astype(np.float32)

        mask_raw = np.array(Image.open(self.parser.mask_paths[idx]))
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[..., 0]
        # G-SHARP convention: 1 - mask/255 → tool=1, tissue=0.
        mask = 1.0 - mask_raw.astype(np.float32) / 255.0

        return {
            "image": torch.from_numpy(image),
            "depth": torch.from_numpy(depth),
            "mask": torch.from_numpy(mask),
            "camtoworld": torch.from_numpy(self.parser.camtoworlds[idx]),
            "K": torch.from_numpy(self.parser.K),
            "time": torch.tensor(float(self.parser.times[idx]), dtype=torch.float32),
        }


def _validate_mask_binary(mask_path: Path) -> None:
    """Open *mask_path* and raise if the pixel values aren't binary {0, 255}.

    Cheap sanity check at parse time — keeps the EndoNeRF dataset honest
    about its tool-mask convention without paying per-getitem cost.
    """
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    unique_vals = np.unique(arr)
    if not set(unique_vals.tolist()).issubset({0, 255}):
        raise ValueError(
            f"EndoNeRFParser: mask {mask_path} is non-binary "
            f"(unique values: {sorted(unique_vals.tolist())[:8]}...). "
            f"Masks must be saved as binary {{0, 255}} PNG."
        )
