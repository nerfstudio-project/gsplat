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

import fnmatch
import subprocess
import sys
from pathlib import Path

import pytest

SCENE_ROOT = Path(__file__).resolve().parent


def test_top_level_import_does_not_load_functional_or_native_backend():
    code = """
import sys
import tempfile
from pathlib import Path

scene_root = Path(%r)
import_root = Path(tempfile.mkdtemp())
(import_root / "gsplat_scene").symlink_to(scene_root, target_is_directory=True)
sys.path.insert(0, str(import_root))

import gsplat_scene

unexpected = {
    "gsplat_scene.functional",
    "gsplat_scene.kernels._backend",
    "gsplat_scene_cuda",
}
loaded = sorted(name for name in unexpected if name in sys.modules)
if loaded:
    raise AssertionError(f"unexpected modules loaded: {loaded}")
if "functional" in gsplat_scene.__dict__:
    raise AssertionError("gsplat_scene.functional was eagerly attached")
assert gsplat_scene.Scene.__name__ == "Scene"
assert gsplat_scene.GaussianScene.__name__ == "GaussianScene"
assert gsplat_scene.GaussianInferenceScene.__name__ == "GaussianInferenceScene"
""" % str(
        SCENE_ROOT
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_functional_import_does_not_load_native_backend():
    code = """
import sys
import tempfile
from pathlib import Path

scene_root = Path(%r)
import_root = Path(tempfile.mkdtemp())
(import_root / "gsplat_scene").symlink_to(scene_root, target_is_directory=True)
sys.path.insert(0, str(import_root))

import gsplat_scene.functional

unexpected = {
    "gsplat_scene.kernels._backend",
    "gsplat_scene_cuda",
}
loaded = sorted(name for name in unexpected if name in sys.modules)
if loaded:
    raise AssertionError(f"importing gsplat_scene.functional eagerly loaded: {loaded}")
""" % str(
        SCENE_ROOT
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_scene_pyproject_includes_cuda_jit_sources():
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = pytest.importorskip("tomli")

    pyproject = tomllib.loads((SCENE_ROOT / "pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]["gsplat_scene"]

    required_sources = [
        "kernels/cuda/ext.cpp",
        "kernels/cuda/csrc/gaussian_scene_pack.cpp",
        "kernels/cuda/csrc/gaussian_scene_pack.cuh",
    ]
    for source in required_sources:
        assert any(
            fnmatch.fnmatchcase(source, pattern) for pattern in package_data
        ), f"{source} is not matched by package-data"
