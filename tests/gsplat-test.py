# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the self-contained test payload installed with a gsplat wheel."""

from __future__ import annotations

import importlib.util
import os
import site
import subprocess
import sys
import sysconfig
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from pathlib import Path


_USAGE = """usage: gsplat-test [cpp [GTEST_ARGS...] | python [PYTEST_ARGS...]]

Run the tests packaged in this gsplat wheel. With no subcommand, the complete
C++ and Python suites run in sequence. Both suites run even when the first one
fails. Arguments after ``cpp`` or ``python`` are forwarded directly to
GoogleTest or pytest, respectively.
"""


def _unique_existing_directories(paths: Iterable[Path]) -> list[Path]:
    """Return existing directories in order, with filesystem aliases removed."""

    result: list[Path] = []
    seen = set()
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        if not resolved.is_dir():
            continue

        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def _python_install_roots() -> list[Path]:
    """Return Python installation roots that can contain dependent libraries."""

    roots = [Path(candidate) for candidate in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))

    # A virtual environment's executable normally lives in <venv>/bin.
    roots.append(Path(sys.executable).resolve().parent.parent)
    return _unique_existing_directories(roots)


def _runtime_library_directories() -> list[Path]:
    """Find Python, Torch, and CUDA runtime library directories.

    Wheel RUNPATH entries cover standard virtual-environment layouts. These
    directories also support nonstandard Python installations and CUDA
    packages whose transitive libraries live in separate ``nvidia`` wheels.
    """

    candidates: list[Path] = []

    # libpython is needed by the standalone C++ test executable.
    for variable in ("LIBDIR", "LIBPL"):
        value = sysconfig.get_config_var(variable)
        if value:
            candidates.append(Path(value))

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None and torch_spec.submodule_search_locations:
        for package_dir in torch_spec.submodule_search_locations:
            candidates.append(Path(package_dir) / "lib")

    for root in _python_install_roots():
        # ``root`` may be either a site-packages directory or an environment
        # prefix. The site-packages candidates cover the common wheel layout.
        candidates.extend(root.glob("nvidia/*/lib"))
        candidates.extend(root.glob("nvidia/*/lib64"))

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.extend((Path(cuda_home) / "lib64", Path(cuda_home) / "lib"))

    return _unique_existing_directories(candidates)


def _prepend_path_list(
    environment: MutableMapping[str, str], name: str, paths: Iterable[Path]
) -> None:
    """Prepend unique filesystem paths to a path-list environment variable."""

    entries = [str(path) for path in _unique_existing_directories(paths)]
    existing = environment.get(name)
    if existing:
        entries.extend(part for part in existing.split(os.pathsep) if part)

    # Preserve existing entries while removing aliases introduced above.
    unique_entries: list[str] = []
    seen = set()
    for entry in entries:
        key = os.path.normcase(os.path.realpath(entry))
        if key not in seen:
            seen.add(key)
            unique_entries.append(entry)
    environment[name] = os.pathsep.join(unique_entries)


def _subprocess_environment() -> Mapping[str, str]:
    """Construct the environment shared by installed C++ and Python tests."""

    environment = os.environ.copy()

    if sys.platform == "win32":
        library_path_name = "PATH"
    elif sys.platform == "darwin":
        library_path_name = "DYLD_LIBRARY_PATH"
    else:
        library_path_name = "LD_LIBRARY_PATH"
    _prepend_path_list(environment, library_path_name, _runtime_library_directories())
    return environment


def _run(
    command: Sequence[str], test_root: Path, environment: Mapping[str, str]
) -> int:
    """Run one installed suite from the private test root."""

    print(f"+ {' '.join(command)}", flush=True)
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(test_root),
            env=dict(environment),
            check=False,
        )
    except OSError as error:
        # Treat loader/permission failures like an ordinary suite failure. In
        # complete-suite mode this allows Python tests to run even when the
        # installed C++ executable cannot be launched.
        print(
            f"gsplat-test: failed to launch {command[0]!r}: {error}",
            file=sys.stderr,
        )
        return 2
    return completed.returncode


def _run_cpp(
    test_root: Path, environment: Mapping[str, str], arguments: Sequence[str]
) -> int:
    """Run the packaged GoogleTest executable, reporting a missing payload."""

    executable = test_root / "bin" / "gsplat_cpp_tests"
    if not executable.is_file():
        print(
            f"gsplat-test: missing C++ test executable: {executable}", file=sys.stderr
        )
        return 2
    return _run((str(executable), *arguments), test_root, environment)


def _run_python(
    test_root: Path, environment: Mapping[str, str], arguments: Sequence[str]
) -> int:
    """Run packaged pytest files, reporting a missing payload."""

    python_tests = test_root / "tests"
    if not python_tests.is_dir():
        print(
            f"gsplat-test: missing Python test directory: {python_tests}",
            file=sys.stderr,
        )
        return 2
    return _run(
        (sys.executable, "-m", "pytest", *arguments),
        test_root,
        environment,
    )


def _parse_command_line(argv: Sequence[str]) -> tuple[str, list[str]]:
    """Parse the small command surface while leaving tool arguments untouched."""

    arguments = list(argv)
    if not arguments:
        return "complete", []
    if arguments[0] in ("-h", "--help"):
        print(_USAGE)
        raise SystemExit(0)

    suite = arguments.pop(0)
    if suite not in ("cpp", "python"):
        print(_USAGE, file=sys.stderr)
        raise ValueError(f"unknown test suite: {suite!r}")
    return suite, arguments


def main(argv: Sequence[str] | None = None) -> int:
    """Run tests from the installed wheel and return a process exit status.

    Args:
        argv: Arguments after the executable name. ``None`` uses
            :data:`sys.argv`.

    Returns:
        Zero when every selected suite passes, otherwise a nonzero status.
    """

    try:
        suite, forwarded_arguments = _parse_command_line(
            sys.argv[1:] if argv is None else argv
        )
    except ValueError as error:
        print(f"gsplat-test: {error}", file=sys.stderr)
        return 2

    test_root = Path(__file__).resolve().parent / "_testdata"
    if not test_root.is_dir():
        print(
            "gsplat-test: this wheel does not contain the test payload "
            "(it was likely built with GSPLAT_BUILD_TESTS=OFF).",
            file=sys.stderr,
        )
        return 2

    environment = _subprocess_environment()
    if suite == "cpp":
        return _run_cpp(test_root, environment, forwarded_arguments)
    if suite == "python":
        return _run_python(test_root, environment, forwarded_arguments)

    print("==> gsplat C++ tests", flush=True)
    cpp_status = _run_cpp(test_root, environment, ())
    print("==> gsplat Python tests", flush=True)
    python_status = _run_python(test_root, environment, ())
    return 0 if cpp_status == 0 and python_status == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
