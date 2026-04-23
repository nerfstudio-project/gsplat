"""EndoNeRF / SCARED dataset loader (example).

Scaffolding only. Implementation pending on branch ``vnath_gsharp``.
Port targets:
- ``holohub/applications/surgical_scene_recon/training/scene/endo_loader.py``
- ``EndoNeRFParser`` in
  ``holohub/applications/surgical_scene_recon/training/gsplat_train.py``

Follows the same shape as :mod:`examples.datasets.colmap` and
:mod:`examples.datasets.ncore`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DatasetType = Literal["endonerf", "scared"]


@dataclass
class EndoNeRFParser:
    """Parse a directory containing ``poses_bounds.npy``, ``images/``,
    ``depth/``, ``masks/`` into in-memory arrays.
    """

    data_dir: Path
    dataset_type: DatasetType = "endonerf"

    def __post_init__(self) -> None:
        raise NotImplementedError("vnath_gsharp: EndoNeRFParser pending")


class EndoNeRFDataset:
    """Torch-friendly iterable over the parsed frames."""

    def __init__(self, parser: EndoNeRFParser, split: str = "train") -> None:
        raise NotImplementedError("vnath_gsharp: EndoNeRFDataset pending")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError
