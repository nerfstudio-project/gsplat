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

"""Lightweight, CPU-safe packaging smoke tests.

These tests inspect the source tree / setup.py packaging metadata using the
filesystem and ``setuptools.find_packages`` only. They deliberately avoid
importing the heavy top-level ``gsplat`` package (which would JIT-build the
native extension) and never require CUDA.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Shipped package source dirs, which must not contain test_*.py.
SHIPPED_PACKAGE_DIRS = [
    REPO_ROOT / "gsplat" / "geometry",
    REPO_ROOT / "gsplat" / "sensors",
    REPO_ROOT / "gsplat" / "scene",
    REPO_ROOT / "gsplat" / "stage",
    REPO_ROOT / "gsplat" / "experimental",
]


def _compute_setup_packages() -> list[str]:
    """Replicate the package list computed by setup.py.

    Mirrors the ``find_packages(exclude=["tests", "tests.*"])`` logic in
    setup.py so we can assert on it without importing setup.py (which pulls in
    torch/setuptools extension machinery at import time).
    """
    from setuptools import find_packages

    packages = find_packages(
        where=str(REPO_ROOT),
        exclude=["tests", "tests.*"],
    )
    return packages


def test_no_test_files_in_shipped_packages():
    """No ``test_*.py`` should exist inside shipped package dirs.

    Ensures test files won't ship in wheels/sdists.
    """
    offenders = []
    for pkg_dir in SHIPPED_PACKAGE_DIRS:
        if not pkg_dir.exists():
            continue
        for path in pkg_dir.rglob("test_*.py"):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"stray test files inside shipped packages: {offenders}"


def test_expected_packages_discoverable():
    """The mapped gsplat.* packages are all in the package list."""
    packages = _compute_setup_packages()
    expected = [
        "gsplat",
        "gsplat.experimental",
        "gsplat.geometry",
        "gsplat.scene",
        "gsplat.stage",
        "gsplat.sensors",
    ]
    missing = [name for name in expected if name not in packages]
    assert not missing, f"missing from package list: {missing} (got {sorted(packages)})"


def test_experimental_published_under_gsplat_namespace():
    """Experimental is published only as ``gsplat.experimental``.

    Experimental lives at ``gsplat/experimental/`` (a normal submodule of the
    gsplat package), so plain ``find_packages`` discovers it under the gsplat
    namespace. The invariant: it must be published as ``gsplat.experimental``
    and never as a bare top-level ``experimental`` package.
    """
    # There must be no repo-root experimental/ dir.
    assert not (REPO_ROOT / "experimental").exists(), (
        "experimental/ must not exist at the repo root; it ships as "
        "gsplat/experimental/"
    )

    # The published package list must expose experimental only under the gsplat
    # namespace, never as a bare top-level package — that is the invariant a
    # wrong layout/exclude would violate.
    packages = _compute_setup_packages()
    assert "gsplat.experimental" in packages
    assert not any(
        p == "experimental" or p.startswith("experimental.") for p in packages
    ), f"bare top-level 'experimental' published: {sorted(packages)}"

    # setup.py must not declare a package_dir remapping for experimental.
    setup_text = (REPO_ROOT / "setup.py").read_text()
    assert "package_dir" not in setup_text, (
        "setup.py must not declare a package_dir remapping; experimental is a "
        "normal gsplat submodule"
    )


@pytest.mark.skipif(
    os.environ.get("RUN_PACKAGING_BUILD_TESTS") != "1",
    reason="heavy sdist build test; set RUN_PACKAGING_BUILD_TESTS=1 to enable",
)
def test_sdist_excludes_tests_includes_cuda(tmp_path):
    """Build an sdist and verify test files are excluded and CUDA csrc included."""
    env = dict(os.environ, BUILD_NO_CUDA="1")
    subprocess.run(
        [sys.executable, "setup.py", "sdist", "--dist-dir", str(tmp_path)],
        cwd=str(REPO_ROOT),
        check=True,
        env=env,
    )
    tarballs = list(tmp_path.glob("*.tar.gz"))
    assert tarballs, "no sdist produced"
    with tarfile.open(tarballs[0]) as tf:
        names = tf.getnames()

    # No test files from the sub-packages.
    stray = [
        n
        for n in names
        if "/test_" in n
        and (
            "/gsplat/geometry/" in n
            or "/gsplat/sensors/" in n
            or "/gsplat/scene/" in n
            or "/gsplat/stage/" in n
            or "/gsplat/experimental/" in n
        )
    ]
    assert not stray, f"sdist shipped relocated test files: {stray}"

    # CUDA csrc sources from the sub-packages ARE included.
    required_substrings = [
        "gsplat/scene/kernels/cuda/csrc/gaussian_scene_pack.cuh",
        "gsplat/geometry/kernels/cuda/csrc/pose.cu",
        "gsplat/sensors/kernels/cuda/csrc/camera_kernel.cu",
        "gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/Projection.cu",
    ]
    for sub in required_substrings:
        assert any(n.endswith(sub) for n in names), f"sdist missing CUDA source: {sub}"
