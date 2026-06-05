# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import glob
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass, field

import pytest
import torch
import torch.utils.cpp_extension as jit

CPP_TEST_TARGET = "gsplat_cpp_tests"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GTEST_FILTER_ARG_LIMIT = 16000

# Native C++ tests are surfaced as ordinary pytest items:
#
# 1. Collection first checks whether tests/cpp/*.cpp exists. An empty native
#    suite becomes a skipped pytest item, so the tree does not need a placeholder
#    C++ test just to keep collection healthy.
# 2. When C++ tests exist, collection calls build_cpp_tests(), which resolves
#    the gsplat implementation object files via get_gsplat_link_inputs():
#      - If gsplat.csrc is already loaded (setup.py / editable install), reuse
#        the .o files under <repo>/build/temp.*-cpython-<tag>/gsplat/cuda/ —
#        same artifacts that produced the loaded csrc.so, so ABI matches.
#      - Otherwise (pure JIT environment), trigger the gsplat JIT build and
#        use the resulting .o files.
#    ext.o is excluded in both paths so the gtest binary does not re-register
#    TORCH_LIBRARY(gsplat).
# 3. build_cpp_tests() then asks PyTorch JIT for a standalone executable target
#    that compiles GoogleTest plus tests/cpp/*.cpp and links those object
#    files. This keeps gsplat's C++/CUDA sources compiled once while still
#    letting PyTorch own compiler flags, include paths, and the generated
#    Ninja files.
# 4. Collection runs the gtest binary with --gtest_list_tests and a JSON report,
#    then parametrizes one pytest item per gtest. This makes pytest --collect-only,
#    node ids, and -k selection work on individual native tests.
# 5. Running pytest invokes the gtest binary once for the pytest-selected native
#    tests, captures a JSON report, and replays each gtest result through its
#    corresponding pytest item. Full native stdout/stderr is printed once after
#    the module's pytest items when verbose mode is enabled and any C++ test
#    fails.


@dataclass(frozen=True)
class GTestCaseResult:
    """GoogleTest execution result for one pytest-visible native test item."""

    name: str
    result: str
    status: str
    failures: tuple[str, ...]
    stdout: str = field(repr=False)
    stderr: str = field(repr=False)
    returncode: int

    @classmethod
    def failed_by_harness(
        cls, name: str, message: str, completed: subprocess.CompletedProcess[str]
    ) -> "GTestCaseResult":
        """Create a synthetic failure when the gtest run itself is incomplete."""
        return cls(
            name=name,
            result="HARNESS_FAILURE",
            status="RUN",
            failures=(message,),
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )

    def skipped(self) -> bool:
        """Return whether GoogleTest reported this test as skipped."""
        return self.result == "SKIPPED" or self.status == "NOTRUN"

    def passed(self) -> bool:
        """Return whether GoogleTest reported this test as passing."""
        return self.result == "COMPLETED" and not self.failures

    def failure_message(self) -> str:
        """Return a pytest assertion message for this gtest result."""
        sections = [
            f"gtest {self.name!r} failed",
            f"result: {self.result}",
            f"status: {self.status}",
            f"returncode: {self.returncode}",
        ]
        if self.failures:
            sections.append("failures:\n" + "\n\n".join(self.failures))
        return "\n\n".join(sections)


class GTestBatchResults(dict[str, GTestCaseResult]):
    """Per-test gtest results plus process-level output for the native batch."""

    def __init__(self) -> None:
        super().__init__()
        self._stdout_parts: list[str] = []
        self._stderr_parts: list[str] = []

    def append_process_output(
        self, completed: subprocess.CompletedProcess[str]
    ) -> None:
        """Remember stdout/stderr from one gtest executable invocation."""
        if completed.stdout:
            self._stdout_parts.append(completed.stdout)
        if completed.stderr:
            self._stderr_parts.append(completed.stderr)

    def has_failure(self) -> bool:
        """Return whether any replayed gtest failed."""
        return any(
            not result.passed() and not result.skipped() for result in self.values()
        )

    def stdout(self) -> str:
        """Return combined stdout from all native gtest invocations."""
        return "\n".join(part.rstrip() for part in self._stdout_parts if part).rstrip()

    def stderr(self) -> str:
        """Return combined stderr from all native gtest invocations."""
        return "\n".join(part.rstrip() for part in self._stderr_parts if part).rstrip()


def _cpp_tests_can_run() -> tuple[bool, str | None]:
    """Return whether this environment can build and run native C++ tests."""
    # Keep this probe independent from gsplat's extension loader. Collection can
    # call it before deciding whether native tests are available, and importing
    # the extension here would defeat the goal of keeping pytest setup cheap.
    def cuda_toolkit_available() -> bool:
        """Return whether PyTorch can find CUDA without importing gsplat."""
        if torch.version.cuda is None:
            return False
        cuda_home = jit.CUDA_HOME
        if cuda_home:
            nvcc = os.path.join(cuda_home, "bin", "nvcc")
            if os.path.isfile(nvcc):
                return True
        return shutil.which("nvcc") is not None

    if os.getenv("BUILD_NO_CUDA", "0") == "1":
        return False, "gsplat CUDA extension is disabled by BUILD_NO_CUDA=1"
    if not cuda_toolkit_available():
        return False, "CUDA toolkit is not available"

    # When gsplat.csrc is loaded but setup.py artifacts are missing or mismatched,
    # translate that into a clean pytest skip with the same diagnostic message.
    from gsplat.cuda.build import (
        get_build_parameters,
        get_gsplat_link_inputs_skip_reason,
    )

    reason = get_gsplat_link_inputs_skip_reason(get_build_parameters())
    if reason:
        return False, reason
    return True, None


def _runtime_env() -> dict[str, str]:
    """Build the runtime environment needed to execute the gtest binary."""
    # The standalone executable is not imported by Python, so it does not get
    # Python's normal extension-loading behavior. It needs the same torch/CUDA
    # shared libraries on the dynamic-loader path when the process starts.
    def prepend_env_path(env: dict[str, str], name: str, paths: list[str]) -> None:
        """Prepend existing directories to a PATH-like environment variable."""
        existing = env.get(name, "")
        present = [path for path in paths if path and os.path.isdir(path)]
        if not present:
            return
        env[name] = os.pathsep.join(present + ([existing] if existing else []))

    env = os.environ.copy()
    library_paths = jit.library_paths("cuda")
    if os.name == "nt":
        # Windows resolves DLLs through PATH, not LD_LIBRARY_PATH. Add both
        # torch's DLL directory and CUDA's bin directory when they exist.
        cuda_bin = os.path.join(jit.CUDA_HOME, "bin") if jit.CUDA_HOME else None
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        prepend_env_path(env, "PATH", [torch_lib, cuda_bin])
    else:
        # On Linux CI, torch.utils.cpp_extension already knows the CUDA and
        # torch library directories that should be visible at runtime.
        prepend_env_path(env, "LD_LIBRARY_PATH", library_paths)
    return env


def _write_jit_object_stamp(
    build_dir: str, objects: list[str], force_rebuild: bool = False
) -> str:
    """Create the generated source that makes reused objects visible to JIT."""
    # PyTorch's JIT versioner hashes source files and flags. The gsplat objects
    # are passed as linker inputs, so they need a source-level proxy that changes
    # whenever those object inputs change.
    def object_digest(objects: list[str]) -> str:
        """Hash object paths and metadata so relinks track JIT object changes."""
        digest = hashlib.sha256()
        for obj in sorted(objects):
            # Hashing full object contents would be more expensive and is not
            # needed for Ninja's dependency model. Path, size, and nanosecond
            # mtime are enough to notice the object that JIT just rebuilt.
            st = os.stat(obj)
            digest.update(os.path.normcase(os.path.abspath(obj)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(st.st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(st.st_mtime_ns).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def stamp_has_digest(path: str, digest: str) -> bool:
        """Check whether the generated stamp source already records this digest."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f'return "{digest}";' in f.read()
        except OSError:
            return False

    def write_if_changed(path: str, content: str) -> None:
        """Write only changed content so PyTorch JIT avoids false rebuilds."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                if f.read() == content:
                    return
        except OSError:
            pass
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    os.makedirs(build_dir, exist_ok=True)
    digest = object_digest(objects)
    stamp_path = os.path.join(build_dir, "gsplat_cpp_tests_jit_stamp.cpp")
    if not force_rebuild and stamp_has_digest(stamp_path, digest):
        # Avoid touching the generated source when the digest is unchanged; a
        # gratuitous write would make PyTorch/Ninja relink on every collection.
        return stamp_path

    # PyTorch's JIT versioner hashes source contents but does not know that
    # these object files are link inputs. A tiny generated source bridges that
    # gap: when any reused gsplat object changes, this file changes and PyTorch
    # regenerates/relinks the gtest executable.
    rebuild_nonce = ""
    if force_rebuild:
        # If an earlier process died while holding PyTorch's JIT lock, the
        # versioner can skip a missing executable. A nonce makes a retry visible
        # as a real source change in that same Python process.
        rebuild_nonce = f"// Rebuild nonce: {time.time_ns()}\n"
    content = f"""// Generated by tests/test_cpp.py. Do not edit.
{rebuild_nonce}\
extern "C" const char *gsplat_cpp_tests_jit_object_digest() {{
    return "{digest}";
}}
"""
    write_if_changed(stamp_path, content)
    return stamp_path


def _gtest_sources() -> list[str]:
    """Return vendored GoogleTest + GoogleMock sources for the test binary."""
    # Build GoogleTest from source inside the same JIT target. That avoids a
    # separate CMake/configure step and lets PyTorch/Ninja use one compiler mode.
    # gmock-all.cc is compiled in too so tests can use GoogleMock matchers (e.g.
    # ThrowsMessage); gtest_main provides main() and is sufficient for matchers.
    gtest_root = os.path.join(REPO_ROOT, "third_party", "googletest", "googletest")
    gmock_root = os.path.join(REPO_ROOT, "third_party", "googletest", "googlemock")
    sources = [
        os.path.join(gtest_root, "src", "gtest-all.cc"),
        os.path.join(gmock_root, "src", "gmock-all.cc"),
        os.path.join(gtest_root, "src", "gtest_main.cc"),
    ]
    if not all(os.path.exists(source) for source in sources):
        raise RuntimeError(
            "googletest submodule is not initialized. Run: "
            "git submodule update --init --recursive third_party/googletest"
        )
    return sources


def _test_sources() -> list[str]:
    """Return project C++ test sources discovered under tests/cpp."""
    # Keep discovery intentionally narrow for now: every file under tests/cpp is
    # part of the single native test binary surfaced through pytest. An empty
    # list is valid: pytest_generate_tests handles it before any JIT build.
    return sorted(glob.glob(os.path.join(REPO_ROOT, "tests", "cpp", "*.cpp")))


def _include_paths(build_params) -> list[str]:
    """Return include directories for gtest, gsplat headers, and JIT parameters."""
    # Include both the repository root and csrc directories so tests can include
    # public gsplat CUDA headers without mirroring extension-relative paths.
    gtest_root = os.path.join(REPO_ROOT, "third_party", "googletest", "googletest")
    gmock_root = os.path.join(REPO_ROOT, "third_party", "googletest", "googlemock")
    return [
        REPO_ROOT,
        os.path.join(REPO_ROOT, "gsplat", "cuda"),
        os.path.join(REPO_ROOT, "gsplat", "cuda", "csrc"),
        os.path.join(gtest_root, "include"),
        gtest_root,
        os.path.join(gmock_root, "include"),
        gmock_root,
    ] + build_params.extra_include_paths


def _cflags(build_params) -> list[str]:
    """Return compile flags compatible with both gsplat and GoogleTest sources."""
    # Reuse gsplat's extension flags, but do not impose gsplat's -Werror policy
    # on vendored GoogleTest or test-only translation units.
    return [flag for flag in build_params.extra_cflags if flag != "-Werror"]


def _cuda_cflags(build_params) -> list[str]:
    """Return CUDA compile flags compatible with both gsplat and test sources."""
    # Mirror _cflags for nvcc. gsplat's CUDA flags carry the C++ standard, the
    # gsplat -D defines, and diagnostic suppressions; without them a test .cu
    # falls back to PyTorch's default nvcc standard (one behind the rest of the
    # build) and misses gsplat's macros. Drop only the -Werror policy so it never
    # applies to test-only CUDA TUs -- nvcc spells it `-Xcompiler=-Werror` and
    # `--Werror all-warnings`.
    werror = {"-Xcompiler=-Werror", "--Werror", "all-warnings"}
    return [flag for flag in build_params.extra_cuda_cflags if flag not in werror]


def _python_link_flags() -> list[str]:
    """Return Python/torch_python link flags needed by reused pybind objects."""
    # gsplat's reusable objects intentionally come from the Python extension
    # build. Some of those translation units instantiate pybind/Python symbols,
    # so the standalone gtest executable has to link the same Python-facing
    # libraries that a normal extension can leave unresolved until import time.
    if os.name == "nt":
        python_lib_path = os.path.join(sys.base_exec_prefix, "libs")
        python_lib = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
        return ["torch_python.lib", f"/LIBPATH:{python_lib_path}", python_lib]

    flags = ["-ltorch_python"]
    for key in ("LIBPL", "LIBDIR"):
        libdir = sysconfig.get_config_var(key)
        if libdir:
            flag = f"-L{libdir}"
            if flag not in flags:
                flags.append(flag)

    python_library = sysconfig.get_config_var("LDLIBRARY")
    if python_library and python_library.startswith("lib"):
        # Convert names like libpython3.10.so or libpython3.10.a into a linker
        # flag form that works with the -L directories added above.
        library_name = python_library[3:]
        for suffix in (".so", ".a", ".dylib"):
            suffix_index = library_name.find(suffix)
            if suffix_index != -1:
                library_name = library_name[:suffix_index]
                break
        flags.append(f"-l{library_name}")
    else:
        flags.append(f"-lpython{sys.version_info.major}.{sys.version_info.minor}")

    for key in ("LIBS", "SYSLIBS"):
        value = sysconfig.get_config_var(key)
        if value:
            flags.extend(shlex.split(value))
    return flags


def _ldflags(build_params, core_objects: list[str]) -> list[str]:
    """Return linker flags for the gtest executable and reused gsplat objects.

    ``core_objects`` is the list of implementation ``.o`` files — either the
    setup.py-built objects (when ``gsplat.csrc`` is loaded) or the JIT-built
    objects (pure JIT environment). ``ext.o`` is excluded from both lists, so
    the gtest binary links the implementation it exercises without dragging in
    the ``TORCH_LIBRARY(gsplat)`` registration that ``ext.cpp`` owns.
    """
    ldflags = list(core_objects) + list(build_params.extra_ldflags)
    ldflags.extend(_python_link_flags())
    if sys.platform != "win32" and "-fopenmp" in build_params.extra_cflags:
        # Some compilers require OpenMP on both compile and link commands.
        ldflags.append("-fopenmp")
    return ldflags


def build_cpp_tests() -> str:
    """Build gsplat once, then build and return the standalone gtest executable."""

    def load_cpp_test_executable(
        build_dir: str, build_params, core_objects: list[str], stamp_source: str
    ) -> str:
        """Ask PyTorch JIT to emit Ninja and build the standalone gtest binary."""
        envvars_to_remove = []
        try:
            # A compact Ninja status line makes local builds and CI logs easier
            # to read without overriding an explicit user-provided setting.
            if not os.getenv("NINJA_STATUS"):
                envvars_to_remove.append("NINJA_STATUS")
                os.environ["NINJA_STATUS"] = "[%f/%t %r %es] "

            # jit.load() writes the Ninja files and immediately invokes Ninja.
            # is_standalone=True returns the executable path instead of trying
            # to import a Python module.
            return jit.load(
                name=CPP_TEST_TARGET,
                sources=_gtest_sources() + _test_sources() + [stamp_source],
                extra_cflags=_cflags(build_params),
                extra_cuda_cflags=_cuda_cflags(build_params),
                extra_include_paths=_include_paths(build_params),
                extra_ldflags=_ldflags(build_params, core_objects),
                build_directory=build_dir,
                verbose=os.getenv("VERBOSE", "0") == "1",
                with_cuda=True,
                is_python_module=False,
                is_standalone=True,
            )
        finally:
            for envvar in envvars_to_remove:
                os.environ.pop(envvar)

    # Keep gsplat imports lazy. Importing this test module should define pytest
    # hooks only; it should not build or import the CUDA extension during pytest
    # startup paths such as `pytest --help`.
    from gsplat.cuda.build import (
        get_build_parameters,
        get_gsplat_link_inputs,
        get_gsplat_link_inputs_skip_reason,
    )

    build_params = get_build_parameters()
    reason = get_gsplat_link_inputs_skip_reason(build_params)
    if reason:
        raise RuntimeError(reason)
    core_objects = get_gsplat_link_inputs()

    # The native test binary gets its own JIT build directory. It links the
    # already-built gsplat objects instead of recompiling gsplat sources.
    build_dir = jit._get_build_directory(CPP_TEST_TARGET, verbose=False)
    stamp_source = _write_jit_object_stamp(build_dir, core_objects)
    executable = load_cpp_test_executable(
        build_dir, build_params, core_objects, stamp_source
    )
    if os.path.exists(executable):
        return executable

    # PyTorch can occasionally return a versioned executable path that is absent
    # after an interrupted prior build. Force a stamp change once so the same
    # process asks JIT/Ninja for a fresh target.
    stamp_source = _write_jit_object_stamp(build_dir, core_objects, force_rebuild=True)
    executable = load_cpp_test_executable(
        build_dir, build_params, core_objects, stamp_source
    )
    if os.path.exists(executable):
        return executable

    raise FileNotFoundError(f"PyTorch JIT did not create {executable}")


def _run_gtest(
    args: list[str], executable: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the gtest executable with the provided GoogleTest arguments."""
    # Capture output so pytest can print the complete gtest failure after the
    # child process exits, without interleaving partial lines during runs.
    if executable is None:
        executable = build_cpp_tests()
    return subprocess.run(
        [executable, *args],
        check=False,
        capture_output=True,
        env=_runtime_env(),
        text=True,
    )


def _gtest_json_path(prefix: str, chunk_index: int | None = None) -> str:
    """Return a process-local path for GoogleTest JSON output."""
    suffix = f"_{chunk_index}" if chunk_index is not None else ""
    return os.path.join(
        jit._get_build_directory(CPP_TEST_TARGET, verbose=False),
        f"{prefix}_{os.getpid()}{suffix}.json",
    )


def _full_gtest_name(suite_name: str, test_name: str) -> str:
    """Return the full gtest name used by --gtest_filter."""
    return f"{suite_name}.{test_name}"


@functools.lru_cache(maxsize=None)
def _list_gtest_tests(gtest_filter: str | None) -> list[str]:
    """List gtests matching an optional GoogleTest filter for pytest collection."""

    def parse_gtest_list(output: str) -> list[str]:
        """Parse legacy stdout from --gtest_list_tests into full test names."""
        # The text format is suite headers followed by indented test names. Keep
        # this fallback for older GoogleTest revisions that do not emit JSON for
        # --gtest_list_tests.
        tests = []
        suite = None
        for line in output.splitlines():
            if not line.strip() or line.startswith("Running main()"):
                continue
            if not line.startswith(" "):
                suite = line.strip()
                continue
            if suite is None:
                continue
            test_name = line.strip().split("#", 1)[0].strip().split()[0]
            tests.append(f"{suite}{test_name}")
        return tests

    def parse_gtest_json_list(path: str) -> list[str]:
        """Parse GoogleTest's JSON test-list report into full test names."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tests = []
        for suite in data.get("testsuites", []):
            suite_name = suite.get("name")
            if not suite_name:
                continue
            for test_case in suite.get("testsuite", []):
                test_name = test_case.get("name")
                if test_name:
                    tests.append(_full_gtest_name(suite_name, test_name))
        return tests

    # GoogleTest's stdout listing is meant to be human-readable. Recent gtest
    # also writes a JSON representation for --gtest_list_tests, and that report
    # already respects --gtest_filter, so pytest collection prefers it.
    # Include the process id in the filename so xdist workers and repeated local
    # collection attempts do not race over the same temporary report.
    json_path = _gtest_json_path("gtest_list")
    try:
        os.remove(json_path)
    except FileNotFoundError:
        pass

    # Collection must list tests from an executable that has just gone through
    # PyTorch JIT's up-to-date check. Otherwise pytest can collect names from a
    # stale gtest binary and later execute a rebuilt binary with different test
    # names.
    executable = build_cpp_tests()

    args = ["--gtest_list_tests", f"--gtest_output=json:{json_path}"]
    if gtest_filter:
        args.append(f"--gtest_filter={gtest_filter}")
    result = _run_gtest(args, executable=executable)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to list gtest tests\n"
            f"command: {executable} {' '.join(args)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if os.path.exists(json_path):
        try:
            return parse_gtest_json_list(json_path)
        finally:
            os.remove(json_path)
    return parse_gtest_list(result.stdout)


def _format_gtest_failure(failure: object) -> str:
    """Return a readable failure string from one GoogleTest JSON failure entry."""
    if not isinstance(failure, dict):
        return str(failure)

    lines = []
    location = ":".join(
        str(failure[key]) for key in ("file", "line") if failure.get(key) is not None
    )
    if location:
        lines.append(location)
    failure_type = failure.get("type")
    if failure_type:
        lines.append(str(failure_type))
    message = failure.get("failure") or failure.get("message")
    if message:
        lines.append(str(message))
    if not lines:
        lines.append(json.dumps(failure, sort_keys=True))
    return "\n".join(lines)


def _parse_gtest_run_json(
    path: str, completed: subprocess.CompletedProcess[str]
) -> dict[str, GTestCaseResult]:
    """Parse GoogleTest's execution JSON into results keyed by full gtest name."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    for suite in data.get("testsuites", []):
        suite_name = suite.get("name")
        if not suite_name:
            continue
        for test_case in suite.get("testsuite", []):
            test_name = test_case.get("name")
            if not test_name:
                continue
            full_name = _full_gtest_name(suite_name, test_name)
            failures = tuple(
                _format_gtest_failure(failure)
                for failure in test_case.get("failures", [])
            )
            results[full_name] = GTestCaseResult(
                name=full_name,
                result=str(test_case.get("result", "")),
                status=str(test_case.get("status", "")),
                failures=failures,
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
            )
    return results


def _gtest_filter_chunks(gtest_names: list[str]) -> list[list[str]]:
    """Split exact gtest names into the fewest conservative filter chunks."""
    # One colon-separated --gtest_filter keeps execution to one process in the
    # common case. Keep the argument below a conservative limit so Windows and
    # long parametrized gtest names still work; only very large selections need
    # a second process.
    chunks = []
    current = []
    current_len = len("--gtest_filter=")
    for name in gtest_names:
        added_len = len(name) + (1 if current else 0)
        if current and current_len + added_len > GTEST_FILTER_ARG_LIMIT:
            chunks.append(current)
            current = []
            current_len = len("--gtest_filter=")
            added_len = len(name)
        current.append(name)
        current_len += added_len
    if current:
        chunks.append(current)
    return chunks


def _run_gtest_batch(gtest_names: list[str]) -> GTestBatchResults:
    """Run selected gtests with as few executable invocations as possible."""
    executable = build_cpp_tests()
    all_results = GTestBatchResults()
    for chunk_index, chunk in enumerate(_gtest_filter_chunks(gtest_names)):
        json_path = _gtest_json_path("gtest_run", chunk_index)
        try:
            os.remove(json_path)
        except FileNotFoundError:
            pass

        args = [
            f"--gtest_filter={':'.join(chunk)}",
            f"--gtest_output=json:{json_path}",
        ]
        completed = _run_gtest(args, executable=executable)
        all_results.append_process_output(completed)
        if os.path.exists(json_path):
            try:
                chunk_results = _parse_gtest_run_json(json_path, completed)
            finally:
                os.remove(json_path)
        else:
            message = (
                "GoogleTest did not produce JSON output for the selected native "
                f"tests. Command: {executable} {' '.join(args)}"
            )
            for name in chunk:
                all_results[name] = GTestCaseResult.failed_by_harness(
                    name, message, completed
                )
            continue

        # A normal assertion failure is already attached to a specific gtest in
        # the JSON report. If the process failed without any such per-test
        # failure, treat it as a harness-level problem so pytest cannot pass a
        # run that ended with a crashing binary, loader error, or global failure.
        reported_failure = any(
            not result.passed() and not result.skipped()
            for name, result in chunk_results.items()
            if name in chunk
        )
        if completed.returncode != 0 and not reported_failure:
            message = (
                "GoogleTest exited with a nonzero status without reporting a "
                "per-test failure in JSON output."
            )
            for name in chunk:
                all_results[name] = GTestCaseResult.failed_by_harness(
                    name, message, completed
                )
            continue

        all_results.update(chunk_results)
        for name in chunk:
            if name not in all_results:
                message = (
                    "GoogleTest JSON output did not contain a result for the "
                    f"selected test {name!r}."
                )
                all_results[name] = GTestCaseResult.failed_by_harness(
                    name, message, completed
                )
    return all_results


def _selected_gtest_names(request: pytest.FixtureRequest) -> list[str]:
    """Return native gtests selected by pytest after -k/node-id filtering."""
    # At fixture setup time pytest has already applied command-line selection
    # such as -k and explicit node ids. Walking session.items lets the native
    # batch contain exactly those gtest cases, while collection still exposes
    # each case as an ordinary pytest item.
    this_file = os.path.abspath(__file__)
    selected = []
    seen = set()
    for item in request.session.items:
        item_path = getattr(item, "path", None)
        if item_path is None:
            item_path = getattr(item, "fspath", "")
        if os.path.abspath(str(item_path)) != this_file:
            continue
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        gtest_name = callspec.params.get("gtest_name")
        if gtest_name and gtest_name not in seen:
            selected.append(gtest_name)
            seen.add(gtest_name)
    return selected


def _write_gtest_native_output_summary(
    request: pytest.FixtureRequest, results: GTestBatchResults
) -> None:
    """Print full native gtest output once after verbose failed C++ test runs."""
    if request.config.getoption("verbose", 0) <= 0 or not results.has_failure():
        return

    stdout = results.stdout()
    stderr = results.stderr()
    if not stdout and not stderr:
        return

    terminalreporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if terminalreporter is None:
        return

    terminalreporter.write_sep("=", "GoogleTest native output")
    if stdout:
        terminalreporter.write_sep("-", "stdout")
        terminalreporter.write(stdout + "\n")
    if stderr:
        terminalreporter.write_sep("-", "stderr")
        terminalreporter.write(stderr + "\n")


@pytest.fixture(scope="module")
def gtest_results(request: pytest.FixtureRequest) -> dict[str, GTestCaseResult]:
    """Run selected gtests once and expose per-test results to pytest items."""
    gtest_names = _selected_gtest_names(request)
    if not gtest_names:
        yield {}
        return

    results = _run_gtest_batch(gtest_names)
    yield results
    _write_gtest_native_output_summary(request, results)


def pytest_generate_tests(metafunc):
    """Parametrize one pytest item per discovered GoogleTest test case."""
    if "gtest_name" not in metafunc.fixturenames:
        return

    # This is collection-time work by design: pytest needs the concrete gtest
    # names before it can support --collect-only, -k, and explicit node ids.
    test_sources = _test_sources()
    if not test_sources:
        # No native test body exists yet. Report this as a skipped pytest item
        # and avoid building gsplat or the standalone gtest executable.
        metafunc.parametrize(
            "gtest_name",
            [
                pytest.param(
                    "",
                    marks=pytest.mark.skip(
                        reason="no C++ test sources found under tests/cpp"
                    ),
                    id="no-sources",
                )
            ],
        )
        return

    can_run, reason = _cpp_tests_can_run()
    if not can_run:
        # Emit one skipped pytest item instead of failing collection on machines
        # that cannot build CUDA tests.
        metafunc.parametrize(
            "gtest_name",
            [pytest.param("", marks=pytest.mark.skip(reason=reason), id="unavailable")],
        )
        return

    # --gtest_filter narrows the native tests before pytest parametrization, so
    # the collected pytest items match what the gtest binary would execute.
    gtest_filter = metafunc.config.getoption("gtest_filter", default=None)
    test_names = _list_gtest_tests(gtest_filter)
    if not test_names:
        reason = f"no gtest tests matched {gtest_filter!r}"
        metafunc.parametrize(
            "gtest_name",
            [pytest.param("", marks=pytest.mark.skip(reason=reason), id="no-tests")],
        )
        return

    metafunc.parametrize("gtest_name", test_names, ids=test_names)


def test_cpp_gtest(gtest_name: str, gtest_results: dict[str, GTestCaseResult]):
    """Replay one exact GoogleTest case as a pytest test item."""
    # pytest creates one item per native test, but the module fixture executes
    # the native binary once for all pytest-selected names. This final function
    # only maps the already-captured gtest result back onto pytest's pass, fail,
    # or skip outcome for the current item.
    result = gtest_results.get(gtest_name)
    if result is None:
        pytest.fail(f"GoogleTest did not run selected test {gtest_name!r}")

    if result.skipped():
        reason = (
            result.failure_message() if result.failures else f"{gtest_name} skipped"
        )
        pytest.skip(reason)

    if not result.passed():
        pytest.fail(result.failure_message())
