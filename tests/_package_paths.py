# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve Python sources from either a build tree or an installed wheel."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path


def _gsplat_package_roots() -> list[Path]:
    """Return candidate source-tree and installed-package roots."""

    # In a checkout, tests/ and gsplat/ are siblings. In an installed test
    # payload this candidate does not exist, so fall through to the package
    # spec without importing gsplat and triggering its native wrappers.
    candidates = [Path(__file__).resolve().parent.parent / "gsplat"]
    spec = find_spec("gsplat")
    if spec is not None and spec.submodule_search_locations is not None:
        candidates.extend(Path(path) for path in spec.submodule_search_locations)

    roots = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_dir() and resolved not in seen:
            seen.add(resolved)
            roots.append(resolved)
    return roots


def gsplat_package_file(relative_path: str) -> Path:
    """Return an existing file beneath a source or installed package root."""

    searched = []
    for package_root in _gsplat_package_roots():
        candidate = package_root / relative_path
        searched.append(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{relative_path!r} was not found beneath gsplat package roots: {searched}"
    )
