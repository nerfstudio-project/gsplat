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
