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
without importing the heavy top-level ``gsplat`` package, which would import
native-extension wrappers. They never require CUDA.
"""

from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10.
    import tomli as tomllib

pytestmark = [pytest.mark.wheel_smoke]

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


def test_feature_extras_are_public():
    """Source metadata exposes the documented PNG and test feature names."""

    if not HAS_SOURCE_TREE:
        pytest.skip("pyproject.toml is only available in a source checkout")

    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]

    assert {"png", "test"}.issubset(extras)


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

    Tests belong only in the private ``gsplat/_testdata`` tree, never beside
    production modules in the public packages.
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
    """Core owns segmented sort sources, which compiled wheels exclude."""
    if HAS_SOURCE_TREE:
        shipped_files = {
            str(path.relative_to(REPO_ROOT))
            for path in (REPO_ROOT / "gsplat").rglob("SegmentedSort.*")
        }
        assert shipped_files == set(SEGMENTED_SORT_SOURCES)
    else:
        shipped_files = {
            str(path)
            for path in _installed_files()
            if path.name.startswith("SegmentedSort.")
        }
        assert not shipped_files, f"wheel shipped CUDA sources: {sorted(shipped_files)}"


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
    HAS_SOURCE_TREE,
    reason="installed-wheel manifest assertion",
)
def test_installed_wheel_contains_test_payload():
    """The test-enabled wheel contains every required payload category."""

    installed_files = _installed_files()
    installed_names = {path.as_posix() for path in installed_files}

    # These names are runtime interfaces: pytest discovers its configuration
    # and test-tree conftest by name, while gsplat-test invokes the C++
    # executable by name. Other payload files are implementation details and
    # are checked structurally below so routine renames do not change this test.
    cpp_executable_name = "gsplat/_testdata/bin/gsplat_cpp_tests"
    runtime_names = {
        cpp_executable_name,
        "gsplat/_testdata/tests/conftest.py",
        "gsplat/_testdata/pytest.ini",
    }
    missing_runtime_names = sorted(runtime_names - installed_names)
    assert (
        not missing_runtime_names
    ), f"wheel is missing test runtime entries: {missing_runtime_names}"

    test_root = ("gsplat", "_testdata", "tests")
    expected_suites = {"core", "examples"} | {
        package.rsplit(".", 1)[-1] for package in SHIPPED_PACKAGES
    }
    installed_suites = {
        path.parts[len(test_root)]
        for path in installed_files
        if path.parts[: len(test_root)] == test_root
        and len(path.parts) > len(test_root)
        and path.name.startswith("test_")
        and path.suffix == ".py"
    }
    missing_suites = sorted(expected_suites - installed_suites)
    assert (
        not missing_suites
    ), f"wheel has no Python tests for these suites: {missing_suites}"

    def payload_exists(prefix, suffix=None, *, direct_child=False, excluded_names=()):
        """Return whether an installed payload matches a structural category."""

        return any(
            path.parts[: len(prefix)] == prefix
            and len(path.parts) > len(prefix)
            and (not direct_child or len(path.parts) == len(prefix) + 1)
            and (suffix is None or path.suffix == suffix)
            and path.name not in excluded_names
            for path in installed_files
        )

    testdata_root = ("gsplat", "_testdata")
    payload_categories = {
        "asset": payload_exists(testdata_root + ("assets",), direct_child=True),
        "example module": payload_exists(
            testdata_root + ("examples",),
            ".py",
            direct_child=True,
            excluded_names={"__init__.py"},
        ),
        "dataset support module": payload_exists(
            testdata_root + ("examples", "datasets"),
            ".py",
            direct_child=True,
            excluded_names={"__init__.py"},
        ),
        "JSON fixture": payload_exists(test_root, ".json"),
    }
    missing_categories = [
        label for label, is_present in payload_categories.items() if not is_present
    ]
    assert (
        not missing_categories
    ), f"wheel is missing test payload categories: {missing_categories}"

    assert not any(
        name.startswith("gsplat/_testdata/tests/source/") for name in installed_names
    ), "source-distribution-only tests leaked into the wheel"

    shared_library_suffixes = {".dll", ".dylib", ".so"}
    assert any(
        path.parts[:2] == ("gsplat", "lib")
        and shared_library_suffixes.intersection(path.suffixes)
        for path in installed_files
    ), "wheel is missing the internal gsplat core shared library"

    executable_path = next(
        path for path in installed_files if path.as_posix() == cpp_executable_name
    )
    executable = metadata.distribution("gsplat").locate_file(executable_path)
    assert executable.is_file()
    if os.name != "nt":
        assert os.access(executable, os.X_OK), f"not executable: {executable}"
