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

"""
Pytest configuration and shared fixtures for gsplat tests.

This file is automatically discovered by pytest and applies to all test files
in this directory and subdirectories.
"""

import gc
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed

from tests.av_helpers import av_trainer, make_av_splats, make_av_scene


# When optional libs/* subpackages are not installed (e.g. on the upstream
# GitHub Actions ``core_tests.yml`` runner that only installs core gsplat),
# drop the corresponding testpaths so ``pytest`` does not try to collect
# tests whose imports would crash. ``pytest.ini`` keeps the full testpaths
# list for the internal NVIDIA GPU validation environment where the libs
# are installed by ``libs/install.sh``.
_LIBS_TESTPATH_TO_PACKAGE = (
    ("libs/geometry/functional", "gsplat_geometry"),
    ("libs/scene/components", "gsplat_scene"),
    ("libs/stage/components", "gsplat_stage"),
)


def pytest_ignore_collect(collection_path, config):
    path_str = str(collection_path)
    for testpath, package in _LIBS_TESTPATH_TO_PACKAGE:
        if testpath in path_str:
            try:
                __import__(package)
            except ImportError:
                return True
    return False


@pytest.fixture(autouse=True)
def setup_test_environment():
    """
    Autouse fixture that runs before every test to ensure:
    1. Deterministic random seed
    2. CUDA cache is cleared
    3. Garbage collection is performed

    This fixture automatically applies to all tests in this directory
    without needing to be explicitly requested.
    """

    seed = 42

    # Set seed based on test name for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    # Run garbage collection
    gc.collect()

    # Yield to run the test
    yield

    # Optional: cleanup after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@pytest.fixture(scope="session")
def dist_init():
    """Initialize a single-process distributed group for testing distributed code paths.

    With world_size=1 the all-gather / all-to-all ops become identity operations,
    but the code path inside ``rasterization(distributed=True)`` is still exercised.
    """
    if not torch.cuda.is_available():
        yield
        return

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
        # Warm up the communicator required by batch_isend_irecv.
        _ = [None]
        torch.distributed.all_gather_object(_, 0)

    yield

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.fixture
def av_train_env(monkeypatch, tmp_path):
    """Common scaffolding for tests that call av_trainer.train().

    Stubs out load_scene, init_gaussians_from_lidar, render_gaussians,
    and CUDA memory stats so train() runs without a GPU.

    Skips the requesting test when av_trainer's optional dependencies are
    not installed (e.g. upstream GitHub Actions core_tests.yml).
    """
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
