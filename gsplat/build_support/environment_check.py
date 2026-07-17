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
``gsplat[test]`` inside ``dev``) is checked once, under the first section that
declares it. Additional build-dependent requirements may be passed as
``--require <requirement>``.
Only reports — never installs.
"""

from __future__ import annotations

import argparse
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


def check_requirement(requirement_text, section, seen, only=None):
    """Yield the result for one applicable requirement.

    ``seen`` carries the requirement lines already checked by earlier sections;
    ``only`` optionally restricts the check to the named packages.
    """

    if requirement_text in seen:
        return
    seen.add(requirement_text)
    requirement = Requirement(requirement_text)
    if only is not None and canonicalize_name(requirement.name) not in only:
        return
    if requirement.marker is not None and not requirement.marker.evaluate():
        return
    try:
        installed = importlib.metadata.version(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        yield False, f"[{section}] '{requirement_text}' is not installed"
        return
    # Direct-URL requirements pin a source, not a version; presence is all the
    # environment can be checked for.
    if requirement.url is None and not requirement.specifier.contains(
        installed, prereleases=True
    ):
        yield (
            False,
            f"[{section}] '{requirement_text}' is unmet by the installed version "
            f"{installed}",
        )
    else:
        constraint = f" ({requirement.specifier})" if str(requirement.specifier) else ""
        yield True, f"{requirement.name} {installed}{constraint}"


def check_requirements(pyproject_path, section, seen, only=None):
    """Yield results for the requirements declared by ``section``."""

    for requirement_text in extract_section(pyproject_path, section):
        yield from check_requirement(requirement_text, section, seen, only)


def main(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        description="verify declared and build-dependent Python requirements",
    )
    parser.add_argument("pyproject_path")
    parser.add_argument("sections", nargs="*")
    parser.add_argument(
        "--require",
        dest="build_requirements",
        action="append",
        default=[],
        help="additional build-dependent requirement to verify",
    )
    arguments = parser.parse_args(argv[1:])
    if not arguments.sections and not arguments.build_requirements:
        parser.error("at least one section or --require is required")

    unmet = 0
    seen = set()
    for spec in arguments.sections:
        section, _, names = spec.partition(":")
        only = {canonicalize_name(n) for n in names.split(",")} if names else None
        for satisfied, message in check_requirements(
            arguments.pyproject_path, section, seen, only
        ):
            print(message, file=sys.stdout if satisfied else sys.stderr)
            unmet += not satisfied
    for requirement_text in arguments.build_requirements:
        for satisfied, message in check_requirement(
            requirement_text, "build metadata", seen
        ):
            print(message, file=sys.stdout if satisfied else sys.stderr)
            unmet += not satisfied
    return 1 if unmet else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
