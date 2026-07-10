# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that the active environment satisfies declared dependencies.

Usage:

    python3 environment_check.py <pyproject.toml> <section>...

where each ``section`` is ``dependencies`` for the base dependency list or an
optional-dependency group name, optionally restricted to named packages as
``<section>:<name>[,<name>...]``. Prints one ``<name> <version>`` line per
satisfied requirement on stdout and one line per unmet requirement on stderr,
exiting non-zero when any requirement is unmet. A requirement repeated by a
later section (for example through a same-project extra such as
``gsplat[test]`` inside ``dev``) is checked once, under the first
section that declares it.
Only reports — never installs.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import sys

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ImportError:  # pragma: no cover - minimal environments ship pip only.
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pyproject_metadata import extract_section  # noqa: E402


def check_requirements(pyproject_path, section, seen, only=None):
    """Yield a ``(satisfied, message)`` pair per applicable ``section`` requirement.

    ``seen`` carries the requirement lines already checked by earlier sections;
    ``only`` optionally restricts the check to the named packages.
    """

    for line in extract_section(pyproject_path, section):
        if line in seen:
            continue
        seen.add(line)
        requirement = Requirement(line)
        if only is not None and canonicalize_name(requirement.name) not in only:
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            yield False, f"[{section}] '{line}' is not installed"
            continue
        # Direct-URL requirements pin a source, not a version; presence is all
        # the environment can be checked for.
        if requirement.url is None and not requirement.specifier.contains(
            installed, prereleases=True
        ):
            yield False, f"[{section}] '{line}' is unmet by the installed version {installed}"
        else:
            constraint = (
                f" ({requirement.specifier})" if str(requirement.specifier) else ""
            )
            yield True, f"{requirement.name} {installed}{constraint}"


def main(argv):
    if len(argv) < 3:
        raise SystemExit(f"usage: {argv[0]} <pyproject.toml> <section>...")
    unmet = 0
    seen = set()
    for spec in argv[2:]:
        section, _, names = spec.partition(":")
        only = {canonicalize_name(n) for n in names.split(",")} if names else None
        for satisfied, message in check_requirements(argv[1], section, seen, only):
            print(message, file=sys.stdout if satisfied else sys.stderr)
            unmet += not satisfied
    return 1 if unmet else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
