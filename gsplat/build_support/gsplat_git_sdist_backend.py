# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PEP 517 wrapper that gathers gsplat sdist sources from Git."""

from __future__ import annotations

import glob
import os
import subprocess
from pathlib import Path
from collections.abc import Sequence

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from scikit_build_core import build as _scikit_build_core


# This module sits two directories below the source root.
_SOURCE_ROOT = Path(__file__).resolve().parents[2]


# Wheel construction, metadata generation, and sdist requirements remain
# exactly scikit-build-core's implementation.
build_editable = _scikit_build_core.build_editable
build_wheel = _scikit_build_core.build_wheel
get_requires_for_build_sdist = _scikit_build_core.get_requires_for_build_sdist
prepare_metadata_for_build_editable = (
    _scikit_build_core.prepare_metadata_for_build_editable
)
prepare_metadata_for_build_wheel = _scikit_build_core.prepare_metadata_for_build_wheel


ConfigSettings = dict[str, str | list[str]] | None


def _project_dependency(name: str) -> str:
    """Return one dependency from ``project.dependencies`` by package name."""

    with (_SOURCE_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    expected_name = canonicalize_name(name)
    matches = [
        dependency
        for dependency in pyproject.get("project", {}).get("dependencies", [])
        if canonicalize_name(Requirement(dependency).name) == expected_name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {name!r} entry in project.dependencies, "
            f"found {len(matches)}"
        )
    return matches[0]


def get_requires_for_build_editable(
    config_settings: ConfigSettings = None,
) -> list[str]:
    """Return requirements for an editable CMake build, including Torch."""

    requirements = _scikit_build_core.get_requires_for_build_editable(config_settings)
    return [*requirements, _project_dependency("torch")]


def get_requires_for_build_wheel(
    config_settings: ConfigSettings = None,
) -> list[str]:
    """Return requirements for a wheel CMake build, including Torch."""

    requirements = _scikit_build_core.get_requires_for_build_wheel(config_settings)
    return [*requirements, _project_dependency("torch")]


def _run_git(arguments: Sequence[str]) -> bytes:
    """Run Git in the source checkout and return its byte-oriented output."""

    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(_SOURCE_ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise RuntimeError("building the gsplat sdist requires Git") from error
    except subprocess.CalledProcessError as error:
        detail = os.fsdecode(error.stderr).strip()
        raise RuntimeError(
            "failed to query the Git manifest for the gsplat sdist"
            + (f": {detail}" if detail else "")
        ) from error
    return completed.stdout


def _check_source_checkout() -> None:
    """Require this backend's directory to be the queried checkout root."""

    top_level = Path(
        os.fsdecode(_run_git(("rev-parse", "--show-toplevel"))).strip()
    ).resolve()
    if top_level != _SOURCE_ROOT:
        raise RuntimeError(
            "building the gsplat sdist requires its source directory to be "
            f"a Git worktree root (found {top_level}, expected {_SOURCE_ROOT})"
        )


def _decode_nul_paths(output: bytes) -> list[str]:
    """Decode Git's NUL-delimited paths using the filesystem encoding."""

    return [os.fsdecode(path) for path in output.split(b"\0") if path]


def _check_submodules_initialized() -> None:
    """Fail clearly when a clone has an unusable required submodule."""

    status = os.fsdecode(_run_git(("submodule", "status", "--recursive")))
    missing = [line[1:].strip() for line in status.splitlines() if line.startswith("-")]
    conflicted = [
        line[1:].strip() for line in status.splitlines() if line.startswith("U")
    ]
    if missing:
        formatted = "\n  ".join(missing)
        raise RuntimeError(
            "building the gsplat sdist requires initialized submodules; run "
            f"'git submodule update --init --recursive' for:\n  {formatted}"
        )
    if conflicted:
        formatted = "\n  ".join(conflicted)
        raise RuntimeError(
            "cannot build the gsplat sdist with conflicted submodules:\n  "
            f"{formatted}"
        )


def _submodule_paths() -> list[str]:
    """Return initialized submodule paths relative to the repository root."""

    # Git supplies displaypath to the foreach shell and recurses into nested
    # submodules. NUL delimiters preserve every path accepted by the filesystem.
    output = _run_git(
        (
            "submodule",
            "foreach",
            "--recursive",
            "--quiet",
            'printf "%s\\0" "$displaypath"',
        )
    )
    return _decode_nul_paths(output)


def _git_sdist_paths() -> list[str]:
    """Return tracked and non-ignored development files for the sdist.

    ``--recurse-submodules`` expands Git's otherwise opaque gitlink entries.
    Additional queries admit newly created source files in the superproject and
    every initialized submodule before their first commit. Git's ignore rules
    keep build products and worktree-local files out. A submodule checked out at
    a different commit is accepted deliberately: the sdist captures the current
    worktree snapshot, while a clean release checkout remains reproducible.
    """

    _check_source_checkout()
    _check_submodules_initialized()
    tracked = _decode_nul_paths(
        _run_git(("ls-files", "--cached", "--recurse-submodules", "-z", "--"))
    )
    untracked = _decode_nul_paths(
        _run_git(("ls-files", "--others", "--exclude-standard", "-z", "--"))
    )
    for submodule in _submodule_paths():
        submodule_untracked = _decode_nul_paths(
            _run_git(
                (
                    "-C",
                    submodule,
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "-z",
                    "--",
                )
            )
        )
        untracked.extend(f"{submodule}/{path}" for path in submodule_untracked)

    files = {
        path
        for path in (*tracked, *untracked)
        if (_SOURCE_ROOT / path).is_file() or (_SOURCE_ROOT / path).is_symlink()
    }
    return sorted(files)


def _literal_pathspec(path: str) -> str:
    """Turn a repository-relative path into an anchored literal pathspec."""

    return "/" + glob.escape(path)


def build_sdist(
    sdist_directory: str,
    config_settings: ConfigSettings = None,
) -> str:
    """Build an sdist containing exactly Git-known repository files.

    The source must be a Git clone with initialized submodules. Non-ignored
    untracked files are included to support validating a change before it is
    committed; reproducible release builds naturally contain tracked files
    only.
    """

    settings = dict(config_settings or {})
    settings["skbuild.sdist.inclusion-mode"] = "manual"
    # In scikit-build-core's manual mode, unmatched files are still admitted.
    # Make Git's manifest a true allowlist by excluding every path that did not
    # match one of the more specific include rules below. Explicit includes
    # take precedence over exclusions in scikit-build-core's file processor.
    settings["skbuild.sdist.exclude"] = ["*"]
    settings["skbuild.sdist.include"] = [
        _literal_pathspec(path) for path in _git_sdist_paths()
    ]
    return _scikit_build_core.build_sdist(sdist_directory, settings)


__all__ = [
    "build_editable",
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "prepare_metadata_for_build_wheel",
]
