import os

import numpy as np
import pytest
import torch

from citygs.ckpt import concat_splats, crop_splats, load_ckpt, save_ckpt
from citygs.config import MergeConfig
from citygs.merge import run_merge
from citygs.partition.contraction import contract
from citygs.train.finetune_block import block_ckpt_path, compute_block_steps
from citygs.config import BlockConfig

from .conftest import make_splats


def test_ckpt_roundtrip(tmp_path):
    splats = make_splats(100)
    path = str(tmp_path / "a.pt")
    save_ckpt(path, splats, 42, {"stage": "test", "block_id": 7})
    loaded, meta = load_ckpt(path)
    assert meta["step"] == 42 and meta["block_id"] == 7
    torch.testing.assert_close(loaded["means"], splats["means"])


def test_ckpt_rejects_foreign_files(tmp_path):
    path = str(tmp_path / "b.pt")
    torch.save({"something": 1}, path)
    with pytest.raises(ValueError, match="citygs/v1"):
        load_ckpt(path)


def test_crop_and_concat():
    splats = make_splats(50)
    mask = splats["means"][:, 0] > 0
    a = crop_splats(splats, mask)
    b = crop_splats(splats, ~mask)
    both = concat_splats([a, b])
    assert len(both["means"]) == 50


def test_merge_crops_strays_and_concats(toy_manifest, tmp_path):
    manifest = toy_manifest
    blocks_dir = str(tmp_path / "blocks")
    ctr = manifest.contraction

    # Block 0 owns x<0 (contracted); give it 80 inside + 20 strays at x>0.
    inside0 = make_splats(80, low=(-1, -1, 0), high=(-0.1, 1, 1), seed=1)
    stray0 = make_splats(20, low=(0.1, -1, 0), high=(1, 1, 1), seed=2)
    b0 = concat_splats([inside0, stray0])
    save_ckpt(block_ckpt_path(blocks_dir, 0), b0, 10, {"stage": "block", "block_id": 0})
    inside1 = make_splats(60, low=(0.1, -1, 0), high=(1, 1, 1), seed=3)
    save_ckpt(
        block_ckpt_path(blocks_dir, 1), inside1, 10, {"stage": "block", "block_id": 1}
    )

    cfg = MergeConfig(
        manifest=os.path.join(manifest._dir, "scene.json"),
        blocks_dir=blocks_dir,
        result_dir=str(tmp_path / "merged"),
        save_ply=False,
    )
    out = run_merge(cfg)
    merged, meta = load_ckpt(out)
    assert meta["num_blocks"] == 2
    assert len(merged["means"]) == 80 + 60  # strays dropped

    # No merged Gaussian is duplicated: each belongs to exactly one cell.
    contracted = contract(merged["means"].double().numpy(), ctr.center, ctr.half_extent)
    left = (contracted[:, 0] < 0).sum()
    assert left == 80


def test_merge_fails_on_missing_block(toy_manifest, tmp_path):
    cfg = MergeConfig(
        manifest=os.path.join(toy_manifest._dir, "scene.json"),
        blocks_dir=str(tmp_path / "empty"),
        result_dir=str(tmp_path / "merged"),
    )
    with pytest.raises(FileNotFoundError, match="block 0"):
        run_merge(cfg)


def test_adaptive_steps_clamped():
    cfg = BlockConfig()
    cfg.train.max_steps = 10_000
    assert compute_block_steps(cfg, 100, 100.0) == 10_000
    assert compute_block_steps(cfg, 400, 100.0) == 15_000  # sqrt(4)=2 -> clamp 1.5
    assert compute_block_steps(cfg, 1, 100.0) == 5_000  # clamp 0.5
    cfg.adaptive_steps = False
    assert compute_block_steps(cfg, 400, 100.0) == 10_000
