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

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCENE_ROOT = Path(__file__).resolve().parent


def test_top_level_import_does_not_load_functional_or_native_backend():
    code = """
import sys

import gsplat.scene

unexpected = {
    "gsplat.scene.functional",
    "gsplat.scene.kernels._backend",
    "gsplat_scene_cuda",
}
loaded = sorted(name for name in unexpected if name in sys.modules)
if loaded:
    raise AssertionError(f"unexpected modules loaded: {loaded}")
if "functional" in gsplat.scene.__dict__:
    raise AssertionError("gsplat.scene.functional was eagerly attached")
assert gsplat.scene.Scene.__name__ == "Scene"
assert gsplat.scene.GaussianScene.__name__ == "GaussianScene"
assert gsplat.scene.GaussianInferenceScene.__name__ == "GaussianInferenceScene"
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_functional_import_does_not_load_native_backend():
    code = """
import sys

import gsplat.scene.functional

unexpected = {
    "gsplat.scene.kernels._backend",
    "gsplat_scene_cuda",
}
loaded = sorted(name for name in unexpected if name in sys.modules)
if loaded:
    raise AssertionError(f"importing gsplat.scene.functional eagerly loaded: {loaded}")
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _repo_root() -> Path:
    # tests/scene/test_package_imports.py -> repo root is two levels up.
    return SCENE_ROOT.parent.parent


def test_root_packaging_ships_scene_cuda_sources():
    """The scene CUDA/JIT sources ship with the main ``gsplat`` package.

    Packaging is driven by root ``pyproject.toml``
    (``[tool.setuptools.package-data]``) and ``MANIFEST.in``. This test asserts
    (a) the CUDA sources physically exist where the package expects them and
    (b) the root packaging metadata references them — without building a wheel.
    """
    repo_root = _repo_root()
    scene_cuda = repo_root / "gsplat" / "scene" / "kernels" / "cuda"

    required_sources = [
        "csrc/ext.cpp",
        "csrc/gaussian_scene_pack.cpp",
        "csrc/gaussian_scene_pack.cuh",
    ]
    for source in required_sources:
        assert (scene_cuda / source).is_file(), f"missing scene CUDA source: {source}"

    # pyproject.toml must declare the scene CUDA sources as package data
    # (mirrored by MANIFEST.in for sdists). Inspect the text rather than
    # invoking the packaging machinery.
    pyproject_text = (repo_root / "pyproject.toml").read_text()
    manifest_text = (repo_root / "MANIFEST.in").read_text()
    assert (
        '"gsplat.scene"' in pyproject_text
    ), "pyproject.toml is missing gsplat.scene package data"
    assert (
        "kernels/cuda/csrc/*" in pyproject_text
    ), "pyproject.toml is missing scene csrc glob"
    assert (
        "gsplat/scene/kernels/cuda" in manifest_text
    ), "MANIFEST.in is missing scene CUDA sources"
