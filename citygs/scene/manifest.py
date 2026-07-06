"""The scene manifest: the single contract between pipeline stages.

``scene.json`` records the scene geometry (cameras, normalization, bounds),
the block partition, and the artifact paths every stage produces. Stages
never rely on directory-layout conventions — they read the manifest, do
their work, and write their outputs back into it.

Paths inside the manifest are stored relative to the manifest's directory
when possible (so a result tree can be moved as a whole), except image
paths which are relative to ``data_dir``.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MANIFEST_VERSION = 1


@dataclass
class CameraMeta:
    name: str
    image: str  # path relative to data_dir
    width: int
    height: int
    K: List[float]  # 3x3 row-major
    camtoworld: List[float]  # 4x4 row-major, OpenCV convention
    camera_id: int = 0

    def K_np(self) -> np.ndarray:
        return np.asarray(self.K, dtype=np.float64).reshape(3, 3)

    def camtoworld_np(self) -> np.ndarray:
        return np.asarray(self.camtoworld, dtype=np.float64).reshape(4, 4)


@dataclass
class Contraction:
    """Per-axis scene contraction. See :mod:`citygs.partition.contraction`."""

    center: List[float]
    half_extent: List[float]


@dataclass
class BlockMeta:
    block_id: int
    grid_index: Tuple[int, int]
    # Cell bounds in contracted space; the partition of space is exact
    # (cells tile [-2, 2]^3), so merge is a plain concat.
    bmin: List[float]
    bmax: List[float]
    # Global camera indices assigned to this block (train split).
    cameras: List[int] = field(default_factory=list)
    num_gaussians: int = 0  # coarse Gaussians whose center lies in the cell
    cost: float = 0.0  # scheduling cost estimate
    steps: int = 0  # planned finetune steps (0 = trainer default)


@dataclass
class SceneManifest:
    data_dir: str
    factor: int = 1
    test_every: int = 8
    version: int = MANIFEST_VERSION
    # Normalization applied to the raw COLMAP world (4x4 row-major).
    transform: List[float] = field(default_factory=lambda: np.eye(4).flatten().tolist())
    scene_scale: float = 1.0
    cameras: List[CameraMeta] = field(default_factory=list)
    points_file: str = ""  # npz relative to manifest dir
    contraction: Optional[Contraction] = None
    blocks: List[BlockMeta] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)

    # Set on load/save; not serialized.
    _dir: str = ""

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        path = os.path.abspath(path)
        new_dir = os.path.dirname(path)
        if self._dir and new_dir != self._dir:
            # Re-base relative paths so they keep pointing at the same files.
            def rebase(rel: str) -> str:
                if not rel or os.path.isabs(rel):
                    return rel
                return os.path.relpath(os.path.join(self._dir, rel), new_dir)

            self.artifacts = {k: rebase(v) for k, v in self.artifacts.items()}
            self.points_file = rebase(self.points_file)
        self._dir = new_dir
        data = {
            "version": self.version,
            "data_dir": self.data_dir,
            "factor": self.factor,
            "test_every": self.test_every,
            "transform": self.transform,
            "scene_scale": self.scene_scale,
            "points_file": self.points_file,
            "cameras": [vars(c) for c in self.cameras],
            "contraction": vars(self.contraction) if self.contraction else None,
            "blocks": [
                {**vars(b), "grid_index": list(b.grid_index)} for b in self.blocks
            ],
            "artifacts": self.artifacts,
        }
        os.makedirs(self._dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "SceneManifest":
        path = os.path.abspath(path)
        with open(path) as f:
            data = json.load(f)
        if data["version"] != MANIFEST_VERSION:
            raise ValueError(f"Unsupported manifest version {data['version']}")
        manifest = cls(
            data_dir=data["data_dir"],
            factor=data["factor"],
            test_every=data["test_every"],
            transform=data["transform"],
            scene_scale=data["scene_scale"],
            points_file=data.get("points_file", ""),
            cameras=[CameraMeta(**c) for c in data["cameras"]],
            contraction=(
                Contraction(**data["contraction"]) if data.get("contraction") else None
            ),
            blocks=[
                BlockMeta(**{**b, "grid_index": tuple(b["grid_index"])})
                for b in data.get("blocks", [])
            ],
            artifacts=data.get("artifacts", {}),
        )
        manifest._dir = os.path.dirname(path)
        return manifest

    # ----------------------------------------------------------- resolution
    def resolve(self, rel: str) -> str:
        """Resolve a manifest-relative artifact path."""
        if os.path.isabs(rel):
            return rel
        return os.path.join(self._dir, rel)

    def relativize(self, path: str) -> str:
        """Store `path` relative to the manifest dir when it lives under it."""
        path = os.path.abspath(path)
        try:
            return os.path.relpath(path, self._dir) if self._dir else path
        except ValueError:  # different drive on windows
            return path

    def image_path(self, index: int) -> str:
        return os.path.join(self.data_dir, self.cameras[index].image)

    def artifact(self, key: str) -> str:
        if key not in self.artifacts:
            raise KeyError(
                f"Manifest has no '{key}' artifact yet — run the producing stage first. "
                f"Available: {sorted(self.artifacts)}"
            )
        return self.resolve(self.artifacts[key])

    def set_artifact(self, key: str, path: str) -> None:
        self.artifacts[key] = self.relativize(path)

    # -------------------------------------------------------------- queries
    def train_indices(self) -> np.ndarray:
        idx = np.arange(len(self.cameras))
        if self.test_every <= 0:
            return idx
        return idx[idx % self.test_every != 0]

    def val_indices(self) -> np.ndarray:
        idx = np.arange(len(self.cameras))
        if self.test_every <= 0:
            return np.empty(0, dtype=np.int64)
        return idx[idx % self.test_every == 0]

    def camtoworlds(self) -> np.ndarray:
        return np.stack([c.camtoworld_np() for c in self.cameras], axis=0)

    def Ks(self) -> np.ndarray:
        return np.stack([c.K_np() for c in self.cameras], axis=0)

    def camera_positions(self) -> np.ndarray:
        return self.camtoworlds()[:, :3, 3]

    def load_points(self):
        """Load the SfM points npz. Returns (xyz [N,3] f32, rgb [N,3] u8,
        per-camera point indices as (indices, offsets) or None)."""
        if not self.points_file:
            raise ValueError("Manifest has no points file; re-run ingest.")
        data = np.load(self.resolve(self.points_file))
        xyz = data["xyz"]
        rgb = data["rgb"]
        if "cam_point_indices" in data:
            depth_index = (data["cam_point_indices"], data["cam_point_offsets"])
        else:
            depth_index = None
        return xyz, rgb, depth_index

    def block(self, block_id: int) -> BlockMeta:
        for b in self.blocks:
            if b.block_id == block_id:
                return b
        raise KeyError(f"No block {block_id} in manifest ({len(self.blocks)} blocks)")
