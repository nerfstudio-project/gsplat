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


def _load_metadata_helper():
    """Load the metadata helper without importing the top-level gsplat package."""
    spec = importlib.util.spec_from_file_location(
        "gsplat_pyproject_metadata", METADATA_HELPER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
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
