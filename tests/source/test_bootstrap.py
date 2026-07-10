# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the development bootstrap."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = REPO_ROOT / "bootstrap.sh"


def _fake_cuda(tmp_path: Path, version: str = "12.8") -> Path:
    """Create a minimal CUDA root whose nvcc reports *version*."""

    cuda_root = tmp_path / f"cuda-{version}"
    nvcc = cuda_root / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text(
        "#!/usr/bin/env bash\n"
        f"echo 'Cuda compilation tools, release {version}, V{version}.0'\n"
    )
    nvcc.chmod(0o755)
    return cuda_root


def _dry_run(
    tmp_path: Path,
    *arguments: str,
    environment: dict[str, str | None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run bootstrap without changing the Python environment or Git hooks."""

    process_environment = os.environ.copy()
    process_environment["PYTHONPYCACHEPREFIX"] = str(tmp_path / "pycache")
    if environment:
        for name, value in environment.items():
            if value is None:
                process_environment.pop(name, None)
            else:
                process_environment[name] = value

    return subprocess.run(
        [
            str(BOOTSTRAP),
            "--dry-run",
            *arguments,
        ],
        cwd=REPO_ROOT,
        env=process_environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _printed_commands(output: str) -> list[list[str]]:
    """Parse shell-escaped commands printed by bootstrap's dry-run mode."""

    return [
        shlex.split(line[2:]) for line in output.splitlines() if line.startswith("+ ")
    ]


def test_bootstrap_expands_declared_development_dependencies(tmp_path: Path) -> None:
    """The bootstrap installs every declared dependency class."""

    completed = _dry_run(
        tmp_path,
        "--python",
        sys.executable,
        "--cuda",
        "12.8",
        "--",
        "--disable-pip-version-check",
        environment={"CUDACXX": "/nonexistent/nvcc"},
    )
    commands = _printed_commands(completed.stdout)
    pip_commands = [
        command for command in commands if command[1:4] == ["-m", "pip", "install"]
    ]

    assert len(pip_commands) == 2
    assert "CUDA: 12.8 (selected by --cuda)" in completed.stdout
    torch_command, dependency_command = pip_commands
    assert "torch>=2.7" in torch_command
    assert "https://download.pytorch.org/whl/cu128" in torch_command
    assert "--force-reinstall" not in torch_command

    expected_requirements = {
        "scikit-build-core>=1.0",
        "cmake>=3.26",
        "ninja>=1.5",
        "numpy",
        "pytest>=8.3.5",
        "scipy",
        "black==22.3.0",
        "cupy-cuda12x",
        "torchpq>=0.3.0.6",
        "vc-flas>=0.1.7",
    }
    assert expected_requirements.issubset(dependency_command)
    assert "gsplat[test]" not in dependency_command
    assert str(REPO_ROOT) not in dependency_command
    assert all("--disable-pip-version-check" in command for command in pip_commands)


def test_bootstrap_defaults_to_active_python_and_cuda(tmp_path: Path) -> None:
    """Default discovery honors Python on PATH and CUDA_HOME."""

    cuda_root = _fake_cuda(tmp_path)
    bin_dir = tmp_path / "active-environment" / "bin"
    bin_dir.mkdir(parents=True)
    # A wrapper script rather than a symlink: symlinking into a venv whose
    # own binaries are absolute out-of-venv symlinks breaks CPython's prefix
    # discovery (and with it `python -m pip`); exec keeps the interpreter's
    # real path as argv0, so its environment activates normally.
    python_stub = bin_dir / "python"
    python_stub.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python_stub.chmod(0o755)

    completed = _dry_run(
        tmp_path,
        environment={
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CUDA_HOME": str(cuda_root),
            "CUDACXX": None,
            "CUDA_PATH": None,
        },
    )

    assert f"CUDA: 12.8 ({cuda_root}/bin/nvcc)" in completed.stdout
    pip_commands = [
        command
        for command in _printed_commands(completed.stdout)
        if command[1:4] == ["-m", "pip", "install"]
    ]
    assert pip_commands
    assert all(command[0] == str(bin_dir / "python") for command in pip_commands)


def test_bootstrap_warns_when_cuda_option_disagrees_with_detected_nvcc(
    tmp_path: Path,
) -> None:
    """--cuda selects binary dependencies; a differing toolkit draws a warning."""

    cuda_root = _fake_cuda(tmp_path, version="13.0")
    completed = _dry_run(
        tmp_path,
        "--python",
        sys.executable,
        "--cuda",
        "12.8",
        environment={
            "CUDA_HOME": str(cuda_root),
            "CUDACXX": None,
            "CUDA_PATH": None,
        },
    )
    assert "CUDA: 12.8 (selected by --cuda)" in completed.stdout
    assert "WARNING" in completed.stderr
    assert "13.0" in completed.stderr


def test_bootstrap_does_not_warn_when_cuda_option_matches_detected_nvcc(
    tmp_path: Path,
) -> None:
    """A toolkit matching --cuda stays quiet."""

    cuda_root = _fake_cuda(tmp_path, version="12.8")
    completed = _dry_run(
        tmp_path,
        "--python",
        sys.executable,
        "--cuda",
        "12.8",
        environment={
            "CUDA_HOME": str(cuda_root),
            "CUDACXX": None,
            "CUDA_PATH": None,
        },
    )
    assert "CUDA: 12.8 (selected by --cuda)" in completed.stdout
    assert "WARNING" not in completed.stderr
