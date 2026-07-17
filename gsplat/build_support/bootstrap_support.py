# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python helpers for bootstrap.sh.

bootstrap.sh orchestrates the shell-level work (arguments, venv, pip, git);
the environment inspection and dependency parsing that need Python live here
so the shell stays small. Third-party imports are deferred into the commands
that use them, so ``check-python`` runs on an interpreter that has nothing
installed yet. Subcommands:

    python3 bootstrap_support.py check-python <pyproject.toml>
        Exit non-zero, with a message, unless the running interpreter meets
        the requires-python floor declared in pyproject.toml.

    python3 bootstrap_support.py requirements <pyproject.toml> <cuda-version>
        Emit the bootstrap dependency set as NUL-delimited requirements. The
        first item is the Torch requirement; the rest are the build
        requirements, project dependencies, recursively expanded ``dev``
        extra, and the CuPy distribution matching the CUDA version.

    python3 bootstrap_support.py inspect-torch [--requirement <spec>]
        Report the installed Torch. Prints ``missing`` when Torch is absent,
        otherwise ``<status>\t<distribution-version>\t<torch-version>\t<cuda>``
        where ``status`` is ``compatible``/``incompatible`` against the given
        spec, or ``unknown`` when no spec is passed, and ``cuda`` is the CUDA
        toolkit version Torch targets or ``cpu``.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


def check_python(pyproject_path):
    """Exit unless this interpreter meets pyproject's requires-python floor.

    Uses only the standard library so it runs on the interpreter being
    validated, before any dependency is installed.
    """

    pyproject = pathlib.Path(pyproject_path).read_text(encoding="utf-8")
    declaration = re.search(
        r'^\s*requires-python\s*=\s*["\']([^"\']+)["\']', pyproject, re.MULTILINE
    )
    minimum = None
    if declaration:
        floor = re.search(r">=\s*(\d+)\.(\d+)", declaration.group(1))
        if floor:
            minimum = (int(floor.group(1)), int(floor.group(2)))
    if minimum is None:
        return
    if sys.version_info[:2] < minimum:
        raise SystemExit(
            f"gsplat requires Python {minimum[0]}.{minimum[1]} or newer; "
            f"this interpreter is {sys.version_info.major}.{sys.version_info.minor}"
        )


def bootstrap_requirements(pyproject_path, cuda_version):
    """Return the bootstrap requirements, Torch first.

    torchpq imports CuPy without declaring it, so the CuPy distribution for
    ``cuda_version`` is appended here; bootstrap validates Torch against that
    CUDA version, and CMake later validates the compiler against Torch.
    """

    # Deferred so check-python stays dependency-free; tomli/pip._vendor cover
    # Python 3.10 and minimal environments.
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            from pip._vendor import tomli as tomllib
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.utils import canonicalize_name

    # Load the sibling metadata helpers by directory, matching
    # environment_check.py.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from pyproject_metadata import expand_optional_group
    from wheel_build_metadata import cupy_requirement_for_cuda

    data = tomllib.loads(pathlib.Path(pyproject_path).read_text(encoding="utf-8"))
    project = data.get("project", {})

    requirements = [
        *data.get("build-system", {}).get("requires", []),
        *project.get("dependencies", []),
        *expand_optional_group(project, "dev"),
        cupy_requirement_for_cuda(cuda_version),
    ]

    # Deduplicate while preserving order; sections legitimately repeat a
    # requirement. Torch is returned first because the caller installs it from
    # the CUDA-tagged index, separately from the default-index remainder.
    deduplicated = list(dict.fromkeys(requirements))
    torch = [
        text
        for text in deduplicated
        if canonicalize_name(Requirement(text).name) == "torch"
    ]
    if len(torch) != 1:
        raise SystemExit(
            f"expected exactly one direct Torch requirement, found {len(torch)}"
        )
    rest = [text for text in deduplicated if text != torch[0]]
    return [torch[0], *rest]


def inspect_torch(requirement_text):
    """Return the installed-Torch report line, or ``"missing"``."""

    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        return "missing"

    from importlib.metadata import version

    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement

    distribution_version = version("torch")
    if requirement_text is None:
        status = "unknown"
    elif Requirement(requirement_text).specifier.contains(
        distribution_version, prereleases=True
    ):
        status = "compatible"
    else:
        status = "incompatible"
    cuda = torch.version.cuda or "cpu"
    return "\t".join((status, distribution_version, torch.__version__, cuda))


def main(argv):
    """Parse the subcommand and print its result for bootstrap.sh."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check-python", help="verify the interpreter meets requires-python"
    )
    check.add_argument("pyproject")

    requirements = subparsers.add_parser(
        "requirements", help="emit the NUL-delimited bootstrap requirements"
    )
    requirements.add_argument("pyproject")
    requirements.add_argument("cuda_version")

    inspect = subparsers.add_parser(
        "inspect-torch", help="report the installed Torch distribution"
    )
    inspect.add_argument("--requirement", default=None)

    arguments = parser.parse_args(argv)

    if arguments.command == "check-python":
        check_python(arguments.pyproject)
    elif arguments.command == "requirements":
        for text in bootstrap_requirements(arguments.pyproject, arguments.cuda_version):
            sys.stdout.buffer.write(text.encode() + b"\0")
    else:
        print(inspect_torch(arguments.requirement))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
