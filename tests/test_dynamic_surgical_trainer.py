# SPDX-License-Identifier: Apache-2.0
"""Tests for ``examples/dynamic_surgical_trainer.py`` (branch: vnath_gsharp).

Three test groups:

- Config validation — CPU.
- Setup helpers (``build_splats_from_parser``, ``build_deform_modules``) —
  CPU, on a tiny synthetic EndoNeRF fixture.
- End-to-end one-step training — **CUDA-only** because
  :func:`gsplat.rasterization` is CUDA-only. The end-to-end test also
  requires a real-data EndoNeRF directory (the synthetic 32×32 fixture
  produces sub-pixel Gaussian scales after unprojection + kNN); point at
  one via ``ENDONERF_DATA_DIR`` (typically a ``pulling/`` directory).
  Skipped if either CUDA or the env var is missing — that's the same
  contract documented on
  ``tests/test_contrib_dynamic_strategy.py::test_dynamic_strategy_one_step_train_no_nan``.

Seed is set to 42 by the autouse fixture in ``conftest.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Pillow + tqdm are `[examples]` / `[dev]` extras — skip the whole module
# cleanly if a fresh `pip install .` doesn't pull them. Must
# importorskip *before* we touch examples.dynamic_surgical_trainer which
# imports tqdm at module top.
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("tqdm")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.datasets.endonerf import EndoNeRFDataset, EndoNeRFParser  # noqa: E402
from examples.dynamic_surgical_trainer import (  # noqa: E402
    Config,
    build_deform_modules,
    build_splats_from_parser,
    train_step,
)

_ENDONERF_DATA_DIR = os.environ.get("ENDONERF_DATA_DIR")


# ---------------------------------------------------------------------------
# Synthetic fixture for CPU-friendly setup tests
# ---------------------------------------------------------------------------


def _write_trainer_fixture(
    root: Path, n_frames: int = 4, h: int = 32, w: int = 32
) -> Path:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)

    focal = float(w)
    poses = np.zeros((n_frames, 3, 5), dtype=np.float32)
    for i in range(n_frames):
        poses[i, :, :3] = np.eye(3)
        poses[i, :, 3] = [0.0, 0.0, i * 0.1]
        poses[i, :, 4] = [h, w, focal]
    bounds = np.tile([[0.1, 10.0]], (n_frames, 1)).astype(np.float32)
    poses_arr = np.concatenate([poses.reshape(n_frames, 15), bounds], axis=1)
    np.save(root / "poses_bounds.npy", poses_arr)

    rng = np.random.default_rng(42)
    for i in range(n_frames):
        rgb = rng.integers(50, 200, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(root / "images" / f"{i:06d}.png")
        depth = np.tile(np.linspace(2, 8, h, dtype=np.float32).reshape(h, 1), (1, w))
        Image.fromarray(depth.astype(np.uint8)).save(root / "depth" / f"{i:06d}.png")
        # EndoNeRF / G-SHARP convention: on-disk 0 = tissue, 255 = tool.
        # All-zero PNG → dataset returns tissue=1 everywhere (all-tissue).
        mask = np.zeros((h, w), dtype=np.uint8)
        Image.fromarray(mask).save(root / "masks" / f"{i:06d}.png")
    return root


@pytest.fixture
def trainer_dir(tmp_path: Path) -> Path:
    return _write_trainer_fixture(tmp_path)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_invalid_depth_mode():
    with pytest.raises(ValueError, match="depth_mode"):
        Config(data_dir=Path("./_unused"), depth_mode="bogus")


def test_config_rejects_negative_step_counts():
    with pytest.raises(ValueError, match="step counts"):
        Config(data_dir=Path("./_unused"), coarse_steps=-1)
    with pytest.raises(ValueError, match="step counts"):
        Config(data_dir=Path("./_unused"), fine_steps=-1)


def test_config_paths_are_coerced_to_path():
    cfg = Config(data_dir="./somewhere", output_dir="./output/xyz")  # type: ignore[arg-type]
    assert isinstance(cfg.data_dir, Path)
    assert isinstance(cfg.output_dir, Path)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def test_build_splats_from_parser_produces_expected_shapes(trainer_dir: Path):
    cfg = Config(data_dir=trainer_dir, init_max_points=200)
    parser = EndoNeRFParser(data_dir=trainer_dir)
    params, optimizers = build_splats_from_parser(
        parser, cfg, device=torch.device("cpu")
    )

    n = params["means"].shape[0]
    assert n > 0
    assert n <= 200
    assert params["means"].shape == (n, 3)
    assert params["scales"].shape == (n, 3)
    assert params["quats"].shape == (n, 4)
    assert params["opacities"].shape == (n, 1)
    assert params["colors"].shape == (n, 3)
    # build_splats_from_parser returns only the five per-Gaussian trainables;
    # HexPlane / DeformNet trainables live in their own optimizers (see
    # build_deform_modules). See DynamicStrategy class docstring for why.
    assert "hexplane_params" not in params
    assert "deform_mlp_params" not in params
    assert set(params.keys()) == {"means", "scales", "quats", "opacities", "colors"}
    # Optimizer keys must match parameter keys (Strategy.check_sanity contract).
    assert set(optimizers.keys()) == set(params.keys())


def test_build_splats_from_parser_quats_are_unit_norm(trainer_dir: Path):
    cfg = Config(data_dir=trainer_dir, init_max_points=100)
    parser = EndoNeRFParser(data_dir=trainer_dir)
    params, _ = build_splats_from_parser(parser, cfg, device=torch.device("cpu"))
    norms = params["quats"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_build_deform_modules_constructs(trainer_dir: Path):
    cfg = Config(data_dir=trainer_dir)
    hexplane, deform_net, hex_opt, deform_opt = build_deform_modules(
        cfg, device=torch.device("cpu")
    )
    assert hexplane.feat_dim == cfg.hex_output_dim * len(cfg.hex_multires)
    assert deform_net.feature_dim == hexplane.feat_dim
    assert sum(p.numel() for p in hex_opt.param_groups[0]["params"]) > 0
    assert sum(p.numel() for p in deform_opt.param_groups[0]["params"]) > 0


def test_init_means_inside_derived_hexplane_aabb(trainer_dir: Path):
    """Pin the in-AABB invariant.

    After ``build_splats_from_parser`` produces the init point cloud,
    ``train()`` derives ``derived_bounds = max(cfg.hex_bounds,
    init_means.abs().max() * 2.0)``. Every init Gaussian must then be
    inside ``[-derived_bounds, +derived_bounds]^3`` — otherwise
    ``HexPlaneField.forward``'s ``grid_sample(padding_mode="border")``
    clamps the out-of-AABB queries to the same edge feature and the
    deformation field collapses to a spatial constant.
    """
    cfg = Config(data_dir=trainer_dir, init_max_points=200)
    parser = EndoNeRFParser(data_dir=trainer_dir)
    params, _ = build_splats_from_parser(parser, cfg, device=torch.device("cpu"))

    means_abs_max = float(params["means"].detach().abs().max())
    derived_bounds = max(cfg.hex_bounds, means_abs_max * 2.0)

    inside = (params["means"].detach().abs() <= derived_bounds).all()
    assert bool(inside), (
        f"Init means escape derived HexPlane AABB: "
        f"means.abs().max()={means_abs_max:.3f}, "
        f"derived_bounds={derived_bounds:.3f}"
    )
    # And the 2× safety margin must actually exist (not just an equality).
    assert means_abs_max <= derived_bounds * 0.5 + 1e-6


# ---------------------------------------------------------------------------
# End-to-end one-step train (CUDA + real EndoNeRF data)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available() or _ENDONERF_DATA_DIR is None,
    reason=(
        "Requires CUDA and ENDONERF_DATA_DIR env var pointing at a "
        "pulling-style EndoNeRF directory."
    ),
)
def test_trainer_one_step_train_no_nan():
    """End-to-end one-step training pass — no NaN / Inf in losses or params.

    Activates the deferred test
    ``test_dynamic_strategy_one_step_train_no_nan`` from
    ``tests/test_contrib_dynamic_strategy.py``. The synthetic
    ``trainer_dir`` fixture is too small to exercise rasterization
    meaningfully (sub-pixel Gaussian scales after kNN init), so this test
    points at a real dataset via ``ENDONERF_DATA_DIR``.
    """
    from gsplat.contrib.dynamic import DynamicStrategy

    cfg = Config(
        data_dir=Path(_ENDONERF_DATA_DIR),  # type: ignore[arg-type]
        init_max_points=1000,
        coarse_steps=1,
        fine_steps=0,
    )
    device = torch.device("cuda")

    parser = EndoNeRFParser(data_dir=cfg.data_dir)
    dataset = EndoNeRFDataset(parser, split="train")
    params, optimizers = build_splats_from_parser(parser, cfg, device=device)
    hexplane, deform_net, hex_opt, deform_opt = build_deform_modules(cfg, device=device)

    strategy = DynamicStrategy()
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(
        scene_scale=1.0,
        num_gaussians=int(params["means"].shape[0]),
        device=device,
    )

    item = dataset[0]
    losses = train_step(
        cfg=cfg,
        item=item,
        params=params,
        optimizers=optimizers,
        hexplane=hexplane,
        deform_net=deform_net,
        hex_optimizer=hex_opt,
        deform_optimizer=deform_opt,
        strategy=strategy,
        state=state,
        step=0,
    )

    for k, v in losses.items():
        assert np.isfinite(v), f"loss[{k}] = {v} is not finite"
    for k, p in params.items():
        assert torch.isfinite(p).all(), f"param[{k}] has NaN/Inf after one step"
