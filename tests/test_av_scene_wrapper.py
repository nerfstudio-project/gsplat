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

"""AV trainer integration tests for GaussianScene thin-wrapper usage."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from tests.av_helpers import av_trainer, make_av_splats, make_av_scene

# Skip the whole module when av_trainer's optional dependencies are not
# installed (e.g. upstream GitHub Actions core_tests.yml). The fixture
# av_train_env in conftest applies the same skip, but this guard short-
# circuits collection-time attribute access (e.g. av_trainer.GaussianScene
# in test bodies) when av_trainer is None.
pytestmark = pytest.mark.skipif(
    av_trainer is None,
    reason="av_trainer optional dependencies not installed (e.g. imageio)",
)


def test_av_train_uses_scene_splats_for_optimizer_render_loss_and_eval(
    av_train_env,
) -> None:
    """Verify train() wires GaussianScene.splats through to every subsystem."""
    env = av_train_env
    the_splats = make_av_splats()
    seen: dict[str, object] = {}

    class FakeGaussianScene:
        def __init__(self) -> None:
            self.id = "scene"
            self.splats = the_splats

        @classmethod
        def from_splats(cls, splats, **_kwargs):
            assert splats is the_splats
            return cls()

    def fake_init(loaded_scene, device="cuda", **_kwargs):
        del loaded_scene, device
        return the_splats

    def fake_create_optimizers(passed_splats, lr):
        assert passed_splats is the_splats
        seen["optimizer"] = passed_splats
        return {
            name: torch.optim.Adam([passed_splats[name]], lr=lr)
            for name in passed_splats
        }

    def fake_render(*_args, splats=None, **kwargs):
        assert splats is the_splats
        seen["render"] = splats
        seen["render_count"] = seen.get("render_count", 0) + 1
        height = kwargs.get("H", 8)
        width = kwargs.get("W", 8)
        base = splats["means"].sum() * 0.0
        return (
            base + torch.full((1, height, width, 4), 0.25),
            base + torch.full((1, height, width, 1), 0.5),
            {},
            torch.exp(splats["scales"]),
            torch.sigmoid(splats["opacities"]),
        )

    mp = env.monkeypatch
    mp.setattr(av_trainer, "GaussianScene", FakeGaussianScene)
    mp.setattr(av_trainer, "init_gaussians_from_lidar", fake_init)
    mp.setattr(av_trainer, "create_optimizers", fake_create_optimizers)
    mp.setattr(av_trainer, "render_gaussians", fake_render)

    losses, checkpoints = av_trainer.train(
        scene_path="unused",
        max_steps=1,
        lr=1e-3,
        log_every=1,
        eval_every=1,
        result_dir=env.result_dir,
    )

    assert len(losses) == 1
    assert losses[0] > 0
    assert checkpoints[0]["step"] == 1
    assert checkpoints[0]["checkpoint_path"].endswith("ckpt_00001.pt")
    assert seen["optimizer"] is the_splats
    assert seen["render"] is the_splats
    assert seen["render_count"] >= 1


def test_av_checkpoint_round_trip_preserves_scene_state(tmp_path) -> None:
    gaussian_scene = av_trainer.GaussianScene.from_splats(make_av_splats(), id="av")
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()

    ckpt_path = av_trainer.save_checkpoint(
        gaussian_scene,
        scene_path="assets/test_pandaset.npz",
        ckpt_dir=str(ckpt_dir),
        step=7,
    )

    restored_step, restored_scene, restored_scene_path = av_trainer.load_checkpoint(
        ckpt_path, device="cpu"
    )

    assert restored_step == 7
    assert restored_scene_path == "assets/test_pandaset.npz"
    assert restored_scene.id == "av"
    assert restored_scene.component_names == ["av"]
    for key in gaussian_scene.splats.keys():
        assert torch.equal(restored_scene.splats[key], gaussian_scene.splats[key])


def test_av_evaluate_checkpoint_uses_saved_scene_path(monkeypatch, tmp_path) -> None:
    gaussian_scene = av_trainer.GaussianScene.from_splats(make_av_splats(), id="av")
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    ckpt_path = av_trainer.save_checkpoint(
        gaussian_scene,
        scene_path="assets/from_checkpoint.npz",
        ckpt_dir=str(ckpt_dir),
        step=5,
    )
    eval_scene = make_av_scene(is_test=np.array([True]), height=1, width=1)
    seen: dict[str, object] = {}

    def fake_load_scene(path: str, device: str = "cuda"):
        seen["scene_path"] = path
        del device
        return eval_scene

    def fake_evaluate(passed_splats, loaded_scene, test_frame_ids, W, H, **_kwargs):
        seen["splats"] = passed_splats
        seen["test_frame_ids"] = test_frame_ids
        assert loaded_scene is eval_scene
        assert (W, H) == (1, 1)
        return 12.5

    monkeypatch.setattr(av_trainer, "load_scene", fake_load_scene)
    monkeypatch.setattr(av_trainer, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda: None, raising=False
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0, raising=False)

    result = av_trainer.evaluate_checkpoint(
        ckpt_path=str(ckpt_path),
        scene_path=None,
        result_dir=str(tmp_path / "eval"),
    )

    assert seen["scene_path"] == "assets/from_checkpoint.npz"
    assert seen["test_frame_ids"] == [0]
    assert torch.equal(
        seen["splats"]["colors"].cpu(), gaussian_scene.splats["colors"].cpu()
    )
    assert result["step"] == 5
    assert result["mean_psnr"] == 12.5


def test_av_train_writes_checkpoint_at_eval_step(av_train_env) -> None:
    env = av_train_env

    losses, checkpoints = av_trainer.train(
        scene_path="assets/test_pandaset.npz",
        max_steps=1,
        lr=1e-3,
        log_every=1,
        eval_every=1,
        result_dir=env.result_dir,
    )

    ckpt_path = os.path.join(env.result_dir, "ckpts", "ckpt_00001.pt")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    assert len(losses) == 1
    assert losses[0] > 0
    assert checkpoints[0]["checkpoint_path"] == ckpt_path
    assert payload["step"] == 1
    assert payload["scene_path"] == "assets/test_pandaset.npz"
    assert set(payload.keys()) == {"step", "scene_path", "scene_id", "splats"}
