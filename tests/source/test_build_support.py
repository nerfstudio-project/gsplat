# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for helpers used before gsplat can be imported."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
METADATA_HELPER_PATH = REPO_ROOT / "gsplat/build_support/pyproject_metadata.py"
CUPY_REQUIREMENT_HELPER_PATH = REPO_ROOT / "gsplat/build_support/cupy_requirement.py"
WHEEL_METADATA_PROVIDER_PATH = (
    REPO_ROOT / "gsplat/build_support/wheel_build_metadata.py"
)


def _load_metadata_helper():
    """Load the metadata helper without importing the top-level gsplat package."""
    spec = importlib.util.spec_from_file_location(
        "gsplat_pyproject_metadata", METADATA_HELPER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_wheel_metadata_provider():
    """Load the wheel metadata provider without importing gsplat."""

    spec = importlib.util.spec_from_file_location(
        "_gsplat_wheel_build_metadata", WHEEL_METADATA_PROVIDER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cupy_requirement_helper():
    """Load the CuPy-selection helper without importing gsplat."""

    spec = importlib.util.spec_from_file_location(
        "_gsplat_cupy_requirement", CUPY_REQUIREMENT_HELPER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="metadata_helper")
def fixture_metadata_helper():
    """Return the directly loaded pyproject metadata helper module."""
    return _load_metadata_helper()


@pytest.fixture(name="pyproject_path")
def fixture_pyproject_path(tmp_path):
    """Create metadata that exercises base, composite, and invalid groups."""
    path = tmp_path / "pyproject.toml"
    path.write_text(
        """
[project]
name = "demo-project"
dependencies = ["base-package>=1"]

[project.optional-dependencies]
lint = ['Black[jupyter] == 22.3.0; python_version >= "3.8"']
test = ["pytest>=8"]
dev = ["demo_project[test,lint]", "coverage"]
third-party-extra = ["other-project[feature]"]
constrained-self = ["demo-project[lint]>=1"]
cycle-a = ["demo-project[cycle-b]"]
cycle-b = ["demo-project[cycle-a]"]
png = ["imageio>=1", 'vc-flas>=1; python_version < "3.12"']
conditional-self = ['demo-project[png]; python_version < "3.13"']
""".lstrip()
    )
    return path


def test_extract_section_handles_base_and_composite_groups(
    metadata_helper, pyproject_path
):
    """Base aliases and recursive self extras expand to concrete requirements."""
    expected_base = ["base-package>=1"]
    assert (
        metadata_helper.extract_section(pyproject_path, "dependencies") == expected_base
    )
    assert metadata_helper.extract_section(pyproject_path, "install") == expected_base
    assert metadata_helper.extract_section(pyproject_path, "dev") == [
        'Black[jupyter] == 22.3.0; python_version >= "3.8"',
        "pytest>=8",
        "coverage",
    ]


def test_expand_optional_group_preserves_non_composite_requirements(
    metadata_helper, pyproject_path
):
    """Third-party extras and constrained self requirements remain literal."""
    project = metadata_helper.load_project(pyproject_path)
    assert metadata_helper.expand_optional_group(project, "third-party-extra") == [
        "other-project[feature]"
    ]
    assert metadata_helper.expand_optional_group(project, "constrained-self") == [
        "demo-project[lint]>=1"
    ]


def test_expand_optional_group_propagates_marker_onto_composite_expansion(
    metadata_helper, pyproject_path
):
    """A marker on a self-reference carries onto every expanded requirement."""
    project = metadata_helper.load_project(pyproject_path)
    expanded = metadata_helper.expand_optional_group(project, "conditional-self")

    assert expanded == [
        'imageio>=1; python_version < "3.13"',
        'vc-flas>=1; python_version < "3.12" and python_version < "3.13"',
    ]


def test_extract_pin_parses_names_extras_whitespace_and_markers(
    metadata_helper, pyproject_path
):
    """Pin lookup follows normalized package names instead of string matching."""
    assert metadata_helper.extract_pin(pyproject_path, "lint", "black") == "22.3.0"
    assert metadata_helper.extract_pin(pyproject_path, "lint", "BLACK") == "22.3.0"


def test_extract_pin_rejects_unpinned_and_missing_requirements(
    metadata_helper, pyproject_path
):
    """Pin lookup reports when a requirement has no exact pin or is absent."""
    with pytest.raises(SystemExit, match="does not pin pytest with =="):
        metadata_helper.extract_pin(pyproject_path, "test", "pytest")
    with pytest.raises(SystemExit, match="no requirement for ruff"):
        metadata_helper.extract_pin(pyproject_path, "lint", "ruff")


def test_expand_optional_group_reports_unknown_and_cyclic_groups(
    metadata_helper, pyproject_path
):
    """Invalid composition fails clearly instead of recursing indefinitely."""
    project = metadata_helper.load_project(pyproject_path)
    with pytest.raises(SystemExit, match="unknown optional dependency group 'missing'"):
        metadata_helper.expand_optional_group(project, "missing")
    with pytest.raises(
        SystemExit,
        match="cyclic optional dependency groups: cycle-a -> cycle-b -> cycle-a",
    ):
        metadata_helper.expand_optional_group(project, "cycle-a")


def test_pyproject_metadata_command_line_interface(pyproject_path):
    """The script interface prints expanded groups and exact pins."""
    expanded = subprocess.run(
        [sys.executable, str(METADATA_HELPER_PATH), str(pyproject_path), "dev"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert expanded.stdout.splitlines() == [
        'Black[jupyter] == 22.3.0; python_version >= "3.8"',
        "pytest>=8",
        "coverage",
    ]

    pin = subprocess.run(
        [
            sys.executable,
            str(METADATA_HELPER_PATH),
            str(pyproject_path),
            "lint",
            "--pin",
            "black",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert pin.stdout == "22.3.0\n"


def test_wheel_metadata_selects_build_cuda_cupy(monkeypatch):
    """Wheels select CuPy from build Torch while sdists remain portable."""

    provider_module = _load_wheel_metadata_provider()
    monkeypatch.setattr(provider_module, "_build_torch_cuda_version", lambda: "12.8")
    provider = provider_module.WheelBuildMetadataProvider()

    provider.build_state("wheel")
    assert provider.dynamic_metadata({}, {}) == {
        "dependencies": ['cupy-cuda12x; extra == "png"']
    }

    provider.build_state("sdist")
    assert provider.dynamic_metadata({}, {}) == {"dependencies": []}
    assert provider.dynamic_wheel({}) == {"dependencies": True}


def test_cupy_requirement_env_override_wins(monkeypatch):
    """CUPY_PACKAGE short-circuits detection entirely."""

    cupy_requirement = _load_cupy_requirement_helper()
    monkeypatch.setenv("CUPY_PACKAGE", "cupy-cuda99x")

    assert cupy_requirement.detect_cupy_requirement() == "cupy-cuda99x"


def test_cupy_requirement_detected_from_nvcc_version_output(monkeypatch):
    """A parseable ``nvcc --version`` selects the matching CuPy wheel."""

    cupy_requirement = _load_cupy_requirement_helper()
    monkeypatch.delenv("CUPY_PACKAGE", raising=False)
    monkeypatch.setattr(
        cupy_requirement.subprocess,
        "check_output",
        lambda *args, **kwargs: "Cuda compilation tools, release 12.4, V12.4.99",
    )

    assert cupy_requirement.detect_cupy_requirement() == "cupy-cuda12x"


def test_cupy_requirement_falls_back_to_cuda_h_when_nvcc_output_unparseable(
    monkeypatch, tmp_path
):
    """An unparseable ``nvcc`` output falls back to ``cuda.h``'s CUDA_VERSION."""

    cupy_requirement = _load_cupy_requirement_helper()
    monkeypatch.delenv("CUPY_PACKAGE", raising=False)
    monkeypatch.setattr(
        cupy_requirement.subprocess,
        "check_output",
        lambda *args, **kwargs: "not a recognizable nvcc banner",
    )

    cuda_home = tmp_path / "cuda"
    (cuda_home / "include").mkdir(parents=True)
    (cuda_home / "include" / "cuda.h").write_text("#define CUDA_VERSION 12040\n")
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    with pytest.warns(UserWarning, match="unparseable"):
        assert cupy_requirement.detect_cupy_requirement() == "cupy-cuda12x"


def test_cupy_requirement_defaults_to_bare_cupy_when_nothing_detected(
    monkeypatch, tmp_path
):
    """No override, no readable ``nvcc``, no readable ``cuda.h`` -> bare ``cupy``."""

    cupy_requirement = _load_cupy_requirement_helper()
    monkeypatch.delenv("CUPY_PACKAGE", raising=False)

    def _nvcc_not_found(*_args, **_kwargs):
        raise FileNotFoundError("nvcc")

    def _cuda_h_not_found(*_args, **_kwargs):
        raise FileNotFoundError("cuda.h")

    monkeypatch.setattr(cupy_requirement.subprocess, "check_output", _nvcc_not_found)
    monkeypatch.setattr(cupy_requirement, "open", _cuda_h_not_found, raising=False)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "does-not-exist"))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    assert cupy_requirement.detect_cupy_requirement() == "cupy"
