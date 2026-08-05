# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""CPU-safe tests for accelerator toolkit discovery."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def lazy_backend_module():
    path = REPO_ROOT / "gsplat" / "_lazy_backend.py"
    spec = importlib.util.spec_from_file_location("gsplat_test_lazy_backend", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_torch(monkeypatch, *, hip, rocm_home=None, cuda_home=None):
    torch = ModuleType("torch")
    torch.__path__ = []
    torch.version = SimpleNamespace(hip=hip)

    torch_utils = ModuleType("torch.utils")
    torch_utils.__path__ = []
    cpp_extension = ModuleType("torch.utils.cpp_extension")
    cpp_extension.ROCM_HOME = rocm_home
    cpp_extension._find_cuda_home = lambda: cuda_home

    torch.utils = torch_utils
    torch_utils.cpp_extension = cpp_extension
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension)


def test_rocm_toolkit_available_from_rocm_home(lazy_backend_module, monkeypatch):
    _install_fake_torch(monkeypatch, hip="7.2", rocm_home="/opt/rocm")
    monkeypatch.setattr(
        lazy_backend_module.os.path,
        "isfile",
        lambda path: path == "/opt/rocm/bin/hipcc",
    )

    assert lazy_backend_module.cuda_toolkit_available()


def test_rocm_toolkit_available_from_path(lazy_backend_module, monkeypatch):
    _install_fake_torch(monkeypatch, hip="7.2")
    monkeypatch.setattr(lazy_backend_module.os.path, "isfile", lambda _path: False)
    monkeypatch.setattr(
        subprocess,
        "call",
        lambda command, **_kwargs: 0 if command == ["hipcc", "--version"] else 1,
    )

    assert lazy_backend_module.cuda_toolkit_available()


def test_rocm_toolkit_unavailable(lazy_backend_module, monkeypatch):
    _install_fake_torch(monkeypatch, hip="7.2")
    monkeypatch.setattr(lazy_backend_module.os.path, "isfile", lambda _path: False)

    def missing_hipcc(_command, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "call", missing_hipcc)
    assert not lazy_backend_module.cuda_toolkit_available()


def test_cuda_toolkit_detection_is_unchanged(lazy_backend_module, monkeypatch):
    _install_fake_torch(monkeypatch, hip=None, cuda_home="/usr/local/cuda")
    monkeypatch.setattr(
        lazy_backend_module.os.path,
        "isfile",
        lambda path: path == "/usr/local/cuda/bin/nvcc",
    )

    assert lazy_backend_module.cuda_toolkit_available()
