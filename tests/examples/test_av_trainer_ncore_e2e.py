# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end AV trainer checks on real NCore test data."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import pytest
import torch

from tests._cuda import cuda_is_available

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../examples"))
# Skip cleanly (don't error collection) when the examples extra is absent:
#   - av_trainer imports examples/utils.py -> matplotlib (examples extra only)
#   - the ncore/cuda guards below run at call time, not on this module import
train = pytest.importorskip("av_trainer").train


NCORE_TEST_SCENE_ENV = "GSPLAT_NCORE_TEST_SCENE"
DEFAULT_NCORE_TEST_SCENE = (
    Path(__file__).parent
    / "test_data/ncore_ci_fixture/pai_004c2001-5fc3-43b1-a4d8-bfb0bbb9fdc6.json"
)


def _ncore_test_scene() -> str:
    scene = os.environ.get(NCORE_TEST_SCENE_ENV, str(DEFAULT_NCORE_TEST_SCENE))
    path = Path(scene)
    if not path.exists():
        pytest.fail(f"NCore test scene does not exist: {path}")
    pytest.importorskip("ncore")
    if not cuda_is_available():
        pytest.skip("AV trainer e2e test requires CUDA")
    return str(path)


def _assert_training_outputs(result_dir: Path) -> dict:
    summary_path = result_dir / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())

    assert summary["checkpoints"]
    assert math.isfinite(summary["final_loss"])
    assert math.isfinite(summary["checkpoints"][-1]["mean_psnr"])

    renders = sorted((result_dir / "renders").glob("*.png"))
    assert renders
    image = imageio.imread(renders[0])
    assert image.ndim == 3
    assert image.max() > 0

    return summary


def test_av_trainer_ncore_static_baseline_e2e_test_data(tmp_path: Path):
    """The static Stage.add_scene path can keep selected moving-object returns."""
    scene = _ncore_test_scene()
    result_dir = tmp_path / "static"

    losses, checkpoints = train(
        scene_path=scene,
        max_steps=1,
        lr=0.005,
        log_every=1,
        eval_every=0,
        result_dir=str(result_dir),
        use_mcmc=False,
        cap_max=100_000,
        sh_degree=0,
        save_model=False,
        cameras=["camera_front_wide_120fov"],
        max_lidar=10_000,
        rigid_dynamic_track_class_ids=["automobile"],
        rigid_dynamic_static_baseline=True,
    )

    assert len(losses) == 1
    assert len(checkpoints) == 1
    summary = _assert_training_outputs(result_dir)
    assert summary["scene_path"] == scene
    assert summary["use_mcmc"] is False
    assert summary["scene_ids"] == ["av_scene"]
    assert summary["render_scene_id"] == "av_scene"

    ckpt = torch.load(result_dir / "ckpts" / "ckpt_00001.pt", map_location="cpu")
    assert "scenes" not in ckpt
    assert ckpt["scene_id"] == "av_scene"
    assert ckpt["splats"]["means"].shape[0] == summary["num_gaussians"]


def test_av_trainer_ncore_rigid_dynamic_e2e_test_data(tmp_path: Path):
    """Rigid NCore tracks train through ComponentCollection + RigidTransformOp."""
    scene = _ncore_test_scene()
    # Needed by the sample_inference round-trip at the end; check before
    # spending the GPU training run.
    pytest.importorskip("pycolmap")
    result_dir = tmp_path / "rigid"
    max_lidar = 10_000

    losses, checkpoints = train(
        scene_path=scene,
        max_steps=1,
        lr=0.005,
        log_every=1,
        eval_every=0,
        result_dir=str(result_dir),
        use_mcmc=False,
        cap_max=100_000,
        sh_degree=0,
        save_model=True,
        cameras=["camera_front_wide_120fov"],
        max_lidar=max_lidar,
        rigid_dynamic_track_class_ids=["automobile"],
    )

    assert len(losses) == 1
    assert len(checkpoints) == 1
    summary = _assert_training_outputs(result_dir)
    assert summary["scene_path"] == scene
    assert summary["use_mcmc"] is False
    assert summary["scene_ids"] == ["av_static", "av_dynamic"]
    assert summary["render_scene_id"] == "av_scene"

    ckpt = torch.load(result_dir / "ckpts" / "ckpt_00001.pt", map_location="cpu")
    assert ckpt["render_scene_id"] == "av_scene"
    assert len(ckpt["scenes"]) == 2

    static_state, dynamic_state = ckpt["scenes"]
    assert static_state["component_names"] == ["av_static"]
    assert dynamic_state["component_names"]
    assert dynamic_state["splats"]["means"].shape[0] > 0
    assert dynamic_state["transform_graph"] == {"ops": [{"type": "RigidTransformOp"}]}
    assert set(dynamic_state["ctx_buffer"]) == {"poses", "pose_times"}
    assert dynamic_state["ctx_buffer"]["poses"].shape[-1] == 7
    assert (
        dynamic_state["ctx_buffer"]["pose_times"].shape[0]
        == dynamic_state["ctx_buffer"]["poses"].shape[0]
    )
    assert (
        dynamic_state["component_index"].shape[0]
        == dynamic_state["splats"]["means"].shape[0]
    )

    total_gaussians = (
        static_state["splats"]["means"].shape[0]
        + dynamic_state["splats"]["means"].shape[0]
    )
    assert total_gaussians == max_lidar
    assert summary["num_gaussians"] == total_gaussians

    # model.pt uses the checkpoint schema plus the per-scene state dicts, and
    # sample_inference can rebuild the static + dynamic collection from it.
    model_path = result_dir / "model.pt"
    model = torch.load(model_path)  # no map_location: the export is CPU-native
    assert model["scene_id"] == "av_static"
    assert set(model["splats"]) == set(static_state["splats"])
    assert model["render_scene_id"] == "av_scene"
    assert len(model["scenes"]) == 2
    assert model["splats"]["means"].device.type == "cpu"
    assert model["scenes"][1]["splats"]["means"].device.type == "cpu"
    # The static splats appear top-level and in scenes[0]; they share storage
    # so the file stores them once.
    assert (
        model["splats"]["means"].data_ptr()
        == model["scenes"][0]["splats"]["means"].data_ptr()
    )

    # The training-time dataset selection is recorded for inference.
    assert model["dataset"] == {
        "cameras": ["camera_front_wide_120fov"],
        "duration_sec": None,
        "downscale": 1,
        "rigid_dynamic_track_class_ids": ["automobile"],
        "rigid_dynamic_static_baseline": False,
        "max_dynamic_lidar_points": None,
        "max_dynamic_lidar_points_per_track": 5_000,
        "lidar_step_frame": 1,
        "seed": 42,
    }

    from examples.sample_inference import load_stage

    stage, render_id, _sh_degree, trained_dataset = load_stage(
        str(model_path), torch.device("cuda")
    )
    assert render_id == "av_scene"
    assert stage.scene_ids() == ["av_scene"]
    assert trained_dataset == model["dataset"]


def test_ncore_cuboid_association_splits_returns_from_static_init_test_data():
    """Cuboid-associated returns leave static init and land in rigid tracks."""
    scene = _ncore_test_scene()
    from examples.datasets.ncore import NCoreParser

    common = dict(
        meta_json_path=scene,
        camera_ids=["camera_front_wide_120fov"],
        rigid_dynamic_track_class_ids=["automobile"],
        max_lidar_points=100_000_000,  # no subsampling: compare raw counts
    )
    kept = NCoreParser(keep_dynamic_points_in_static_scene=True, **common)
    split = NCoreParser(keep_dynamic_points_in_static_scene=False, **common)

    # The tracked objects carry local points for the dynamic components.
    assert split.rigid_dynamic_tracks
    assert sum(len(t.points_local) for t in split.rigid_dynamic_tracks) > 0
    # Splitting removes returns inside padded dynamic cuboids from static init.
    assert len(split.points) < len(kept.points)
