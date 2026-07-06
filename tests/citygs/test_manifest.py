import os

import numpy as np
import pytest

from citygs.config import BlockConfig, PipelineConfig, load_config, save_config
from citygs.scene.manifest import SceneManifest


def test_manifest_roundtrip(toy_manifest, tmp_path):
    m = toy_manifest
    path = os.path.join(str(tmp_path), "roundtrip", "scene.json")
    m.set_artifact("coarse_ckpt", os.path.join(str(tmp_path), "roundtrip", "c.pt"))
    m.save(path)
    m2 = SceneManifest.load(path)
    assert len(m2.cameras) == len(m.cameras)
    assert m2.blocks[1].bmax == m.blocks[1].bmax
    assert m2.blocks[0].grid_index == (0, 0)
    np.testing.assert_allclose(m2.camtoworlds(), m.camtoworlds())
    np.testing.assert_allclose(m2.contraction.center, m.contraction.center, atol=1e-12)
    # Artifact stored relative to the manifest dir, resolved back to abs.
    assert not os.path.isabs(m2.artifacts["coarse_ckpt"])
    assert m2.artifact("coarse_ckpt") == os.path.join(
        str(tmp_path), "roundtrip", "c.pt"
    )


def test_missing_artifact_message(toy_manifest):
    with pytest.raises(KeyError, match="producing stage"):
        toy_manifest.artifact("merged_ckpt")


def test_train_val_split(toy_manifest):
    train = toy_manifest.train_indices()
    val = toy_manifest.val_indices()
    assert len(train) + len(val) == len(toy_manifest.cameras)
    assert set(train).isdisjoint(set(val))
    assert (val % 8 == 0).all()


def test_config_json_roundtrip(tmp_path):
    cfg = PipelineConfig(gpus=[0, 1, 2], data_factor=2)
    cfg.block.train.max_steps = 1234
    cfg.block.grow_grad2d = 5e-4
    path = str(tmp_path / "cfg.json")
    save_config(cfg, path)
    cfg2 = load_config(PipelineConfig, path)
    assert cfg2.gpus == [0, 1, 2]
    assert cfg2.block.train.max_steps == 1234
    assert cfg2.block.grow_grad2d == 5e-4
    assert isinstance(cfg2.block, BlockConfig)
