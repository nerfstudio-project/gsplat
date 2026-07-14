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

These tests inspect source-tree packaging metadata or installed wheel metadata
without importing the heavy top-level ``gsplat`` package (which would JIT-build
the native extension). They never require CUDA.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from importlib import metadata
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10.
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

EXPECTED_PACKAGES = [
    "gsplat",
    "gsplat.experimental",
    "gsplat.geometry",
    "gsplat.scene",
    "gsplat.stage",
    "gsplat.sensors",
]

SHIPPED_PACKAGES = EXPECTED_PACKAGES[1:]
SHIPPED_PACKAGE_PARTS = [tuple(name.split(".")) for name in SHIPPED_PACKAGES]
HAS_SOURCE_TREE = (REPO_ROOT / "gsplat" / "__init__.py").exists()
SEGMENTED_SORT_SOURCES = [
    "gsplat/cuda/csrc/SegmentedSort.cu",
    "gsplat/cuda/csrc/SegmentedSort.h",
]


def test_dev_extra_omits_the_dynamic_cupy_requirement():
    """The dev extra is CUDA-agnostic; CuPy is resolved dynamically."""
    if not HAS_SOURCE_TREE:
        pytest.skip("pyproject.toml is only available in a source checkout")

    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]

    assert not any(dep.startswith("cupy") for dep in extras["dev"])


def test_png_extra_is_public():
    """Source metadata exposes the documented PNG feature name."""

    if not HAS_SOURCE_TREE:
        pytest.skip("pyproject.toml is only available in a source checkout")

    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]

    assert "png" in extras


def _installed_files() -> list[metadata.PackagePath]:
    """Return installed package files without importing gsplat."""
    try:
        files = metadata.files("gsplat")
    except metadata.PackageNotFoundError:
        return []
    return list(files or [])


def _published_packages() -> list[str]:
    """Return packages from the source tree or an installed wheel."""
    if not HAS_SOURCE_TREE:
        return sorted(
            {
                ".".join(path.parts[:-1])
                for path in _installed_files()
                if path.parts[:1] == ("gsplat",) and path.name == "__init__.py"
            }
        )

    # Mirror pyproject package discovery without importing gsplat, which would
    # pull in extension machinery.
    from setuptools import find_packages

    return sorted(
        find_packages(
            where=str(REPO_ROOT),
            exclude=["tests", "tests.*"],
        )
    )


def test_no_test_files_in_shipped_packages():
    """No ``test_*.py`` should exist inside shipped package dirs.

    Ensures test files won't ship in wheels/sdists.
    """
    offenders = []
    if HAS_SOURCE_TREE:
        for parts in SHIPPED_PACKAGE_PARTS:
            pkg_dir = REPO_ROOT.joinpath(*parts)
            if not pkg_dir.exists():
                continue
            for path in pkg_dir.rglob("test_*.py"):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    else:
        for path in _installed_files():
            if (
                path.parts[:2] in SHIPPED_PACKAGE_PARTS
                and path.name.startswith("test_")
                and path.suffix == ".py"
            ):
                offenders.append(str(path))
    assert not offenders, f"stray test files inside shipped packages: {offenders}"


def test_expected_packages_discoverable():
    """The mapped gsplat.* packages are all in the package list."""
    packages = _published_packages()
    missing = [name for name in EXPECTED_PACKAGES if name not in packages]
    assert not missing, f"missing from package list: {missing} (got {sorted(packages)})"


def test_build_support_is_not_published():
    """Source-only build helpers must not expand gsplat's runtime API."""
    packages = _published_packages()
    assert "gsplat.build_support" not in packages


def test_segmented_sort_sources_owned_by_core_cuda():
    """The shared segmented sort utility ships from core, not experimental."""
    if HAS_SOURCE_TREE:
        shipped_files = {
            str(path.relative_to(REPO_ROOT))
            for path in (REPO_ROOT / "gsplat").rglob("SegmentedSort.*")
        }
    else:
        shipped_files = {
            str(path)
            for path in _installed_files()
            if path.name.startswith("SegmentedSort.")
        }

    assert shipped_files == set(SEGMENTED_SORT_SOURCES)


def test_experimental_published_under_gsplat_namespace():
    """Experimental is published only as ``gsplat.experimental``.

    Experimental lives at ``gsplat/experimental/`` (a normal submodule of the
    gsplat package), so plain ``find_packages`` discovers it under the gsplat
    namespace. The invariant: it must be published as ``gsplat.experimental``
    and never as a bare top-level ``experimental`` package.
    """
    # There must be no repo-root experimental/ dir.
    if HAS_SOURCE_TREE:
        assert not (REPO_ROOT / "experimental").exists(), (
            "experimental/ must not exist at the repo root; it ships as "
            "gsplat/experimental/"
        )

    # The published package list must expose experimental only under the gsplat
    # namespace, never as a bare top-level package — that is the invariant a
    # wrong layout/exclude would violate.
    packages = _published_packages()
    assert "gsplat.experimental" in packages
    assert not any(
        p == "experimental" or p.startswith("experimental.") for p in packages
    ), f"bare top-level 'experimental' published: {sorted(packages)}"


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
        *SEGMENTED_SORT_SOURCES,
        "gsplat/scene/kernels/cuda/csrc/gaussian_scene_pack.cuh",
        "gsplat/geometry/kernels/cuda/csrc/pose.cu",
        "gsplat/sensors/kernels/cuda/csrc/camera_kernel.cu",
        "gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/Projection.cu",
    ]
    for sub in required_substrings:
        assert any(n.endswith(sub) for n in names), f"sdist missing CUDA source: {sub}"
