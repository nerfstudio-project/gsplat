# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-distribution assertions that intentionally do not ship in wheels."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tarfile
from email import policy
from email.parser import BytesParser
from importlib import metadata
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

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


def _load_wheel_metadata_provider():
    """Load the wheel metadata provider without importing gsplat."""

    provider_path = REPO_ROOT / "gsplat" / "build_support" / "wheel_build_metadata.py"
    spec = importlib.util.spec_from_file_location(
        "_gsplat_wheel_build_metadata", provider_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load wheel metadata provider from {provider_path}")

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


def _direct_torch_requirements(core_metadata):
    """Return unmarked Torch requirements from parsed core metadata."""

    requirements = [
        Requirement(requirement)
        for requirement in core_metadata.get_all("Requires-Dist", [])
    ]
    return [
        str(requirement)
        for requirement in requirements
        if requirement.marker is None and canonicalize_name(requirement.name) == "torch"
    ]


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


@pytest.mark.parametrize(
    "build_state",
    ["wheel", "editable", "metadata_wheel", "metadata_editable"],
)
@pytest.mark.parametrize(
    ("cuda_version", "cupy_requirement"),
    [("12.8", "cupy-cuda12x"), ("13.0", "cupy-cuda13x")],
)
def test_wheel_metadata_selects_build_torch_and_cupy(
    monkeypatch, build_state, cuda_version, cupy_requirement
):
    """Every wheel state records build-compatible Torch and CuPy requirements."""

    provider_module = _load_wheel_metadata_provider()
    queried_packages = []

    def installed_version(package):
        queried_packages.append(package)
        return "2.10.0+cu130"

    monkeypatch.setattr(provider_module.metadata, "version", installed_version)
    monkeypatch.setattr(
        provider_module, "_build_torch_cuda_version", lambda: cuda_version
    )

    provider = provider_module.WheelBuildMetadataProvider()
    provider.build_state(build_state)

    assert provider.dynamic_metadata({}, {}) == {
        "dependencies": [
            "torch==2.10.0",
            f'{cupy_requirement}; extra == "png"',
        ]
    }
    assert queried_packages == ["torch"]


def test_sdist_metadata_keeps_the_static_torch_compatibility_floor(monkeypatch):
    """An sdist does not require Torch merely to resolve project metadata."""

    provider_module = _load_wheel_metadata_provider()

    def unexpected_version_query(_package):
        pytest.fail("sdist metadata must not query the build environment's Torch")

    def unexpected_cuda_query():
        pytest.fail("sdist metadata must not import the build environment's Torch")

    monkeypatch.setattr(provider_module.metadata, "version", unexpected_version_query)
    monkeypatch.setattr(
        provider_module, "_build_torch_cuda_version", unexpected_cuda_query
    )

    provider = provider_module.WheelBuildMetadataProvider()
    provider.build_state("sdist")

    assert provider.dynamic_metadata({}, {}) == {"dependencies": []}
    assert provider.dynamic_wheel({}) == {"dependencies": True}


def test_wheel_metadata_provider_rejects_invalid_protocol_inputs():
    """Missing state, unknown state, and unsupported settings fail clearly."""

    provider_module = _load_wheel_metadata_provider()
    provider = provider_module.WheelBuildMetadataProvider()

    with pytest.raises(RuntimeError, match="did not provide the build state"):
        provider.dynamic_metadata({}, {})

    provider.build_state("unknown")
    with pytest.raises(RuntimeError, match="unsupported build state: unknown"):
        provider.dynamic_metadata({}, {})

    with pytest.raises(RuntimeError, match="accepts no settings"):
        provider.dynamic_metadata({"unexpected": True}, {})
    with pytest.raises(RuntimeError, match="accepts no settings"):
        provider.dynamic_wheel({"unexpected": True})


def test_prepared_wheel_metadata_selects_build_torch_and_cupy(monkeypatch, tmp_path):
    """The real backend merges build-compatible Torch and CuPy metadata."""

    backend = _load_build_backend()
    # CTest runs source tests from the build tree. PEP 517 hooks, however,
    # discover pyproject.toml from the current working directory.
    monkeypatch.chdir(REPO_ROOT)
    dist_info = backend.prepare_metadata_for_build_wheel(str(tmp_path))
    with (tmp_path / dist_info / "METADATA").open("rb") as metadata_file:
        wheel_metadata = BytesParser(policy=policy.default).parse(metadata_file)

    public_torch_version = Version(metadata.version("torch")).public
    assert _direct_torch_requirements(wheel_metadata) == [
        _project_torch_requirement(),
        f"torch=={public_torch_version}",
    ]

    # Derive the expected name independently from the provider under test.
    # CMake enforces this Torch CUDA major against the selected compiler.
    import torch

    assert torch.version.cuda is not None
    torch_cuda_major, separator, _minor = torch.version.cuda.partition(".")
    assert separator and torch_cuda_major.isdigit()
    cupy_name = f"cupy-cuda{torch_cuda_major}x"
    cupy_requirements = [
        Requirement(requirement)
        for requirement in wheel_metadata.get_all("Requires-Dist", [])
        if canonicalize_name(Requirement(requirement).name)
        == canonicalize_name(cupy_name)
    ]
    assert len(cupy_requirements) == 1
    assert cupy_requirements[0].marker is not None
    assert cupy_requirements[0].marker.evaluate({"extra": "png"})
    assert not cupy_requirements[0].marker.evaluate({"extra": ""})
    assert not wheel_metadata.get_all("Dynamic", [])


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

        # Every sdist member is rooted beneath one generated top-level
        # directory. Resolve PKG-INFO while the archive is still open.
        roots = {name.split("/", 1)[0] for name in names}
        assert len(roots) == 1
        archive_root = next(iter(roots))
        assert archive_root.startswith("gsplat-")

        pkg_info_file = archive.extractfile(f"{archive_root}/PKG-INFO")
        assert pkg_info_file is not None
        sdist_metadata = BytesParser(policy=policy.default).parse(pkg_info_file)

    # Strip the generated prefix before comparing repository-relative paths.
    relative_names = {name.split("/", 1)[1] for name in names if "/" in name}

    assert _direct_torch_requirements(sdist_metadata) == [_project_torch_requirement()]
    assert "Requires-Dist" in sdist_metadata.get_all("Dynamic", [])

    # scikit-build-core adds the generated core metadata; every other member
    # must correspond exactly to Git's recursive manifest.
    assert relative_names == expected_paths | {"PKG-INFO"}

    forbidden = [
        name
        for name in names
        if "__pycache__" in name or name.endswith(".pyc") or name.endswith("/.git")
    ]
    assert not forbidden, f"sdist contains generated or VCS-only files: {forbidden}"
