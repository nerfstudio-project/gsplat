# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-distribution assertions that intentionally do not ship in wheels."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tarfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_build_backend():
    """Load the in-tree build backend without changing the test import path."""

    # Build-tree tests resolve imports from the staged package. This source-only
    # test still needs the backend's canonical Git manifest implementation, so
    # load that one file directly instead of changing the test import path.
    backend_path = (
        REPO_ROOT / "gsplat" / "build_support" / "gsplat_git_sdist_backend.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_gsplat_source_build_backend", backend_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load build backend from {backend_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inside_git_worktree():
    """Return whether source-distribution checks can query a Git manifest."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return False
    if result.returncode != 0:
        return False
    return Path(result.stdout.strip()).resolve() == REPO_ROOT


def _project_torch_requirement():
    """Read the canonical Torch requirement independently from the backend."""

    with (REPO_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    matches = [
        dependency
        for dependency in pyproject["project"]["dependencies"]
        if canonicalize_name(Requirement(dependency).name) == "torch"
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize(
    "hook_name",
    ["get_requires_for_build_editable", "get_requires_for_build_wheel"],
)
def test_cmake_build_requirements_include_project_torch(monkeypatch, hook_name):
    """Wheel-producing hooks preserve delegates and add the declared Torch."""

    backend = _load_build_backend()
    received_settings = []

    def delegate(config_settings=None):
        received_settings.append(config_settings)
        return ["delegate-requirement>=1"]

    monkeypatch.setattr(backend._scikit_build_core, hook_name, delegate)
    settings = {"build-option": "value"}

    requirements = getattr(backend, hook_name)(settings)

    assert received_settings == [settings]
    assert requirements == ["delegate-requirement>=1", _project_torch_requirement()]


@pytest.mark.skipif(
    not _inside_git_worktree(),
    reason="Git-manifest assertions require a source checkout",
)
@pytest.mark.skipif(
    os.environ.get("RUN_PACKAGING_BUILD_TESTS") != "1",
    reason="heavy sdist build test; set RUN_PACKAGING_BUILD_TESTS=1 to enable",
)
def test_sdist_matches_recursive_git_manifest(tmp_path):
    """Build an sdist and verify that it exactly matches the Git manifest."""

    # Keep the backend import inside this opt-in test. Regular CTest jobs only
    # collect and skip it, and therefore do not need the packaging toolchain
    # installed in their separate test environment.
    expected_paths = set(_load_build_backend()._git_sdist_paths())
    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(tmp_path)],
        cwd=str(REPO_ROOT),
        check=True,
        env=os.environ,
    )
    tarballs = list(tmp_path.glob("*.tar.gz"))
    assert tarballs, "no sdist produced"
    with tarfile.open(tarballs[0]) as archive:
        names = archive.getnames()

    # Every sdist member is rooted beneath one generated top-level directory.
    # Strip that generated prefix before comparing repository-relative paths.
    roots = {name.split("/", 1)[0] for name in names}
    assert len(roots) == 1
    assert next(iter(roots)).startswith("gsplat-")
    relative_names = {name.split("/", 1)[1] for name in names if "/" in name}

    # scikit-build-core adds the generated core metadata; every other member
    # must correspond exactly to Git's recursive manifest.
    assert relative_names == expected_paths | {"PKG-INFO"}

    forbidden = [
        name
        for name in names
        if "__pycache__" in name or name.endswith(".pyc") or name.endswith("/.git")
    ]
    assert not forbidden, f"sdist contains generated or VCS-only files: {forbidden}"
