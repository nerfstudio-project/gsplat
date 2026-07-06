"""Unified, self-describing checkpoint schema for every stage.

Every checkpoint is::

    {
        "schema": "citygs/v1",
        "step": int,
        "splats": {means, scales, quats, opacities, sh0, shN},  # raw params
        "meta": {stage, block_id?, bmin?, bmax?, world_size?, rank?, ...},
    }

``scales`` are log-scales and ``opacities`` are logits, exactly as they
live in the optimizer.
"""

import os
from typing import Dict, List, Tuple

import torch

SCHEMA = "citygs/v1"
SPLAT_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def save_ckpt(
    path: str, splats: Dict[str, torch.Tensor], step: int, meta: dict
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {
        "schema": SCHEMA,
        "step": step,
        "splats": {k: splats[k].detach().cpu() for k in SPLAT_KEYS},
        "meta": dict(meta),
    }
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)


def load_ckpt(path: str, device: str = "cpu") -> Tuple[Dict[str, torch.Tensor], dict]:
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} checkpoint")
    return data["splats"], {**data["meta"], "step": data["step"]}


def concat_splats(parts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.cat([p[k] for p in parts], dim=0) for k in SPLAT_KEYS}


def crop_splats(
    splats: Dict[str, torch.Tensor], mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    return {k: splats[k][mask].contiguous() for k in SPLAT_KEYS}
