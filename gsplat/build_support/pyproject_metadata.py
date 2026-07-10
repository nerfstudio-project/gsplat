# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single owner of the project's dependency-metadata semantics.

Docker dependency validation and formatter tooling need the project's
dependency lists and pinned tool versions. Loading this module by file path
keeps those consumers on one implementation without importing gsplat.

Also usable as a command:

    python3 pyproject_metadata.py <pyproject.toml> <section>

where ``section`` is ``dependencies`` (or ``install``) for the base
dependency list, or an optional-dependency group name. Prints one
requirement per line.
"""

from __future__ import annotations

import pathlib
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10.
    import tomli as tomllib  # type: ignore[import-not-found]

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ImportError:  # pragma: no cover - minimal environments ship pip only.
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name


def load_project(pyproject_path):
    """Return the ``[project]`` table of the given pyproject.toml."""

    with pathlib.Path(pyproject_path).open("rb") as f:
        return tomllib.load(f).get("project", {})


def expand_optional_group(project, group, _parents=()):
    """Return an optional-dependency group with composite groups expanded.

    A requirement on this project with extras, such as
    ``gsplat[dev-common]``, composes another optional-dependency group. Expand
    those references so metadata consumers such as Docker dependency checks
    see the concrete requirements that pip would install.
    """

    optional = project.get("optional-dependencies", {})
    try:
        requirements = optional[group]
    except KeyError:
        known = ", ".join(sorted(optional))
        raise SystemExit(
            f"unknown optional dependency group '{group}' (known: {known})"
        ) from None

    if group in _parents:
        cycle = " -> ".join((*_parents, group))
        raise SystemExit(f"cyclic optional dependency groups: {cycle}")

    project_name = canonicalize_name(project.get("name", ""))
    expanded = []
    for line in requirements:
        requirement = Requirement(line)
        is_composite_group = (
            canonicalize_name(requirement.name) == project_name
            and requirement.extras
            and not requirement.specifier
            and requirement.url is None
            and requirement.marker is None
        )
        if not is_composite_group:
            expanded.append(line)
            continue

        for extra in sorted(requirement.extras):
            expanded.extend(expand_optional_group(project, extra, (*_parents, group)))
    return expanded


def extract_section(pyproject_path, section):
    """Return the requirement list for a pyproject section.

    ``dependencies`` / ``install`` name the base list; any other value names
    an optional-dependency group.
    """

    project = load_project(pyproject_path)
    if section in {"dependencies", "install"}:
        return list(project.get("dependencies", []))
    return expand_optional_group(project, section)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <pyproject.toml> <section>")
    print("\n".join(extract_section(sys.argv[1], sys.argv[2])))
