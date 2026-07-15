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

"""Fixtures shared by tests under ``tests/core/``.

Scoped to this directory (rather than the top-level ``tests/conftest.py``) so
that collecting tests outside ``tests/core/`` never even parses a reference to
``av_helpers``.
"""

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def av_train_env(monkeypatch, tmp_path):
    """Common scaffolding for tests that call av_trainer.train().

    Stubs out load_scene, init_gaussians_from_lidar, render_gaussians,
    and CUDA memory stats so train() runs without a GPU.

    Skips the requesting test when av_trainer's optional dependencies are
    not installed (e.g. upstream GitHub Actions core_tests.yml).
    """
    # Imported lazily: tests/core/ also holds CPU-only tests (test_build_support,
    # test_packaging) that never request this fixture, and av_helpers pulls in
    # gsplat, which JIT-compiles its CUDA extension on import.
    from tests.core.av_helpers import av_trainer, make_av_splats, make_av_scene

    if av_trainer is None:
        pytest.skip("av_trainer optional dependencies not installed (e.g. imageio)")

    scene = make_av_scene()
    result_dir = str(tmp_path / "av_train")

    def fake_load_scene(path: str, device: str = "cuda") -> SimpleNamespace:
        del path, device
        return scene

    def fake_init_gaussians_from_lidar(
        loaded_scene: SimpleNamespace, device: str = "cuda", **_kwargs
    ) -> torch.nn.ParameterDict:
        del loaded_scene, device
        return make_av_splats()

    def fake_render_gaussians(*_args, splats=None, **kwargs):
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

    monkeypatch.setattr(av_trainer, "load_scene", fake_load_scene)
    monkeypatch.setattr(
        av_trainer, "init_gaussians_from_lidar", fake_init_gaussians_from_lidar
    )
    monkeypatch.setattr(av_trainer, "render_gaussians", fake_render_gaussians)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda: None, raising=False
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0, raising=False)

    return SimpleNamespace(
        av_trainer=av_trainer,
        scene=scene,
        result_dir=result_dir,
        monkeypatch=monkeypatch,
    )
