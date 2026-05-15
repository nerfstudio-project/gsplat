# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import shlex
import shutil
import time
import glob
import sys
import torch
import platform
import json
from types import SimpleNamespace
from torch.utils.cpp_extension import CUDA_HOME

try:
    import torch.utils.cpp_extension as jit
except ImportError as e:
    if "pkg_resources" in str(e):
        raise ImportError(
            "torch.utils.cpp_extension failed to import because 'pkg_resources' "
            "is no longer available in setuptools >= 82. "
            "Fix: pip install 'setuptools<82'\n"
            "This is a known issue with PyTorch < 2.9. "
            "Alternatively, upgrade to PyTorch >= 2.9."
        ) from e
    raise

# `jit._get_build_directory` is a private/internal PyTorch helper (underscore
# prefix; not in `torch.utils.cpp_extension.__all__`). Pin the dependency with
# an attribute check so a torch upgrade that drops or renames it fails loudly
# here instead of silently misdirecting the gtest harness to an empty build
# directory.
if not hasattr(jit, "_get_build_directory"):
    raise RuntimeError(
        "torch.utils.cpp_extension.jit._get_build_directory is missing — "
        "PyTorch upgrade broke a private API gsplat relied on. Update "
        "get_gsplat_build_directory in gsplat/cuda/build.py."
    )

from contextlib import nullcontext, contextmanager

try:
    from rich.console import Console

    _console = Console()
except ImportError:
    _console = None

PATH = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.getenv("DEBUG", "0") == "1"
FAST_MATH = os.getenv("FAST_MATH", "1") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "1" if DEBUG else "0") == "1"
NVCC_FLAGS = os.getenv("NVCC_FLAGS", "")
MAX_JOBS = os.getenv("MAX_JOBS")
NINJA_STATUS = os.getenv("NINJA_STATUS")
VERBOSE = os.getenv("VERBOSE", "0") == "1"

BUILD_3DGUT = os.getenv("BUILD_3DGUT")
BUILD_3DGS = os.getenv("BUILD_3DGS")
BUILD_2DGS = os.getenv("BUILD_2DGS")
BUILD_ADAM = os.getenv("BUILD_ADAM")
BUILD_RELOC = os.getenv("BUILD_RELOC")
BUILD_LOSSES = os.getenv("BUILD_LOSSES")
BUILD_CAMERA_WRAPPERS = os.getenv("BUILD_CAMERA_WRAPPERS", "1" if DEBUG else "0") == "1"

NUM_CHANNELS = os.getenv("NUM_CHANNELS")


def get_build_parameters():
    name = "gsplat_cuda"
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Include paths -----------------------------------
    extra_include_paths = [
        os.path.join(PATH, "include/"),
        os.path.join(current_dir, "csrc", "third_party", "glm"),
    ]
    # Fix for CUDA 12+ in conda environment
    if CUDA_HOME and os.path.isdir(os.path.join(CUDA_HOME, "targets")):
        for arch in os.listdir(os.path.join(CUDA_HOME, "targets")):
            if os.path.isdir(p := os.path.join(CUDA_HOME, "targets", arch, "include")):
                extra_include_paths.append(p)
                # In CCCL 3.0 (bundled with CUDA 13.0 and later) and later, the CCCL headers have
                # been moved to the cccl subdirectory.
                if os.path.isdir(
                    p := os.path.join(CUDA_HOME, "targets", arch, "include", "cccl")
                ):
                    extra_include_paths.append(p)

    # Source files ------------------------------------
    sources = (
        list(glob.glob(os.path.join(PATH, "csrc/*.cu")))
        + list(glob.glob(os.path.join(PATH, "csrc/*.cpp")))
        + [os.path.join(PATH, "ext.cpp")]
    )

    # Compiler flags ----------------------------------
    extra_cflags = []
    extra_cuda_cflags = []
    extra_ldflags = []

    if sys.platform == "win32":
        extra_cflags += ["/std:c++20", "/Zc:preprocessor", "-DWIN32_LEAN_AND_MEAN"]
        extra_cuda_cflags += [
            "-std=c++20",
            "-allow-unsupported-compiler",
            "-Xcompiler",
            "/Zc:preprocessor",
            "-DWIN32_LEAN_AND_MEAN",
        ]
    else:
        extra_cflags = ["-std=c++20"]

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_cflags += ["-arch", "arm64"]
        extra_ldflags += ["-arch", "arm64"]

    extra_cuda_cflags += ["--forward-unknown-opts"]

    # Debug/Release mode
    # MSVC (cl) does not support -O3/-O0; use -O2/-Od (torch converts - to /)
    if DEBUG:
        extra_cflags += ["-g", "-O0"]
        # -lineinfo is emitted below via WITH_SYMBOLS (auto-enabled when DEBUG=1),
        # so no need to add it here; WITH_SYMBOLS is the single source of truth.
        if sys.platform != "win32":  # MSVC equivalent (/W4 /WX) is untested
            extra_cflags += ["-Wall"]
            extra_cuda_cflags += [
                # nvcc intercepts bare -Werror as its own --Werror flag, so
                # pass it via -Xcompiler instead of --forward-unknown-opts.
                "-Xcompiler=-Werror",
                "--Werror",
                "all-warnings",
            ]
        else:
            extra_cflags += ["/Zi", "/Od"]
            extra_cuda_cflags += ["-Od"]
    else:
        if sys.platform != "win32":
            extra_cflags += ["-O3", "-DNDEBUG"]
        else:
            extra_cflags += ["/O2", "-DNDEBUG"]
            extra_cuda_cflags += ["-O2", "-DNDEBUG"]

    extra_cuda_cflags += ["-use_fast_math"] if FAST_MATH else []

    # Silencing of warnings
    # GLM/Torch has spammy and very annoyingly verbose warnings that this suppresses
    extra_cuda_cflags += ["-diag-suppress", "20012,186"]
    if not os.name == "nt":
        extra_cflags += ["-Wno-attributes"]
        # #pragma unroll is standard CUDA idiom but unknown to gcc
        extra_cflags += ["-Wno-unknown-pragmas"]

    if BUILD_2DGS is not None:
        extra_cflags += [f"-DGSPLAT_BUILD_2DGS={BUILD_2DGS}"]
        if sys.platform == "win32":
            extra_cuda_cflags += [f"-DGSPLAT_BUILD_2DGS={BUILD_2DGS}"]
    if BUILD_3DGS is not None:
        extra_cflags += [f"-DGSPLAT_BUILD_3DGS={BUILD_3DGS}"]
        if sys.platform == "win32":
            extra_cuda_cflags += [f"-DGSPLAT_BUILD_3DGS={BUILD_3DGS}"]
    if BUILD_3DGUT is not None:
        extra_cflags += [f"-DGSPLAT_BUILD_3DGUT={BUILD_3DGUT}"]
        if sys.platform == "win32":
            extra_cuda_cflags += [f"-DGSPLAT_BUILD_3DGUT={BUILD_3DGUT}"]
    if BUILD_ADAM is not None:
        extra_cflags += [f"-DGSPLAT_BUILD_ADAM={BUILD_ADAM}"]
        if sys.platform == "win32":
            extra_cuda_cflags += [f"-DGSPLAT_BUILD_ADAM={BUILD_ADAM}"]
    if BUILD_RELOC is not None:
        extra_cflags += [f"-DGSPLAT_BUILD_RELOC={BUILD_RELOC}"]
        if sys.platform == "win32":
            extra_cuda_cflags += [f"-DGSPLAT_BUILD_RELOC={BUILD_RELOC}"]
    if BUILD_LOSSES is not None:
        extra_cflags += [f"-DGSPLAT_BUILD_LOSSES={BUILD_LOSSES}"]
        if sys.platform == "win32":
            extra_cuda_cflags += [f"-DGSPLAT_BUILD_LOSSES={BUILD_LOSSES}"]
    if BUILD_CAMERA_WRAPPERS:
        extra_cflags += ["-DGSPLAT_BUILD_CAMERA_WRAPPERS=1"]
        if sys.platform == "win32":
            extra_cuda_cflags += ["-DGSPLAT_BUILD_CAMERA_WRAPPERS=1"]
    else:
        # Remove 'csrc/CameraWrappers.cu' from the sources list if it exists
        sources = [s for s in sources if not s.endswith("csrc/CameraWrappers.cu")]

    extra_ldflags += [] if WITH_SYMBOLS or sys.platform == "win32" else ["-s"]

    if WITH_SYMBOLS:
        extra_cuda_cflags += ["-lineinfo"]

    if torch.version.hip:
        # USE_ROCM was added to later versions of PyTorch.
        # Define here to support older PyTorch versions as well:
        extra_cflags += ["-DUSE_ROCM", "-U__HIP_NO_HALF_CONVERSIONS__"]
    else:
        extra_cuda_cflags += ["--expt-relaxed-constexpr"]

    parinfo = torch.__config__.parallel_info()
    if (
        "backend: OpenMP" in parinfo
        and "OpenMP not found" not in parinfo
        and sys.platform != "darwin"
    ):
        extra_cflags += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_cflags += ["/openmp"]
            extra_cuda_cflags += ["-Xcompiler", "/openmp"]
        else:
            extra_cflags += ["-fopenmp"]
        if sys.platform == "win32":
            extra_cuda_cflags += ["-DAT_PARALLEL_OPENMP"]
    else:
        print("Compiling without OpenMP...")

    if sys.platform != "win32":
        extra_cuda_cflags += extra_cflags

    # Add -Werror after the copy so it reaches gcc but not nvcc (see DEBUG block above).
    if DEBUG and sys.platform != "win32":
        extra_cflags += ["-Werror"]

    if NUM_CHANNELS is not None:
        # nvcc has a bug where you need to escape the commas in macro values defined with -D.
        extra_cuda_cflags += [
            '-DGSPLAT_NUM_CHANNELS="' + NUM_CHANNELS.replace(",", "\\,") + '"'
        ]
        # gcc would not grok the backslash, so here we just pass NUM_CHANNELS as is.
        extra_cflags += [f"-DGSPLAT_NUM_CHANNELS={NUM_CHANNELS}"]

    extra_cuda_cflags += [] if NVCC_FLAGS == "" else NVCC_FLAGS.split(" ")

    return SimpleNamespace(
        name=name,
        extra_include_paths=extra_include_paths,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
    )


def get_gsplat_build_directory(build_params=None):
    if build_params is None:
        build_params = get_build_parameters()
    return jit._get_build_directory(build_params.name, verbose=False)


def _objects_in_build(build_dir, object_dir=None, recursive=False):
    # Object files under `object_dir` (defaults to `build_dir`) that the
    # current build.ninja in `build_dir` still references. PyTorch's JIT
    # and setup.py builds both use ninja and neither cleans stale `.o`
    # files when a source is renamed or removed, so a raw glob would pick
    # up dead intermediates. A `.o` whose basename no longer appears as a
    # word-delimited token in `build.ninja` is no longer part of the
    # current build graph. The word-boundary anchor prevents a future
    # rename from silently matching a basename that is a substring of
    # another (e.g. `Foo.o` matching inside `BarFoo.o`).
    if object_dir is None:
        object_dir = build_dir
    ninja_path = os.path.join(build_dir, "build.ninja")
    if not os.path.exists(ninja_path):
        raise RuntimeError(f"Expected JIT build file does not exist: {ninja_path}")

    with open(ninja_path, "r", encoding="utf-8") as f:
        ninja_text = f.read()

    if recursive:
        patterns = [
            os.path.join(object_dir, "**", "*.o"),
            os.path.join(object_dir, "**", "*.obj"),
        ]
        objects = sorted(p for pat in patterns for p in glob.glob(pat, recursive=True))
    else:
        objects = sorted(
            glob.glob(os.path.join(object_dir, "*.o"))
            + glob.glob(os.path.join(object_dir, "*.obj"))
        )
    return [
        obj
        for obj in objects
        if re.search(r"\b" + re.escape(os.path.basename(obj)) + r"\b", ninja_text)
    ]


def _exclude_dispatcher_objects(objects):
    """Drop ext.cpp's compiled output from a list of gsplat objects.

    ``ext.cpp`` owns the Python module entrypoint and ``TORCH_LIBRARY(gsplat)``
    registration. Native test binaries link against implementation objects
    directly, so pulling in ``ext.*`` would re-register the namespace.
    """
    return [obj for obj in objects if not os.path.basename(obj).startswith("ext.")]


def get_gsplat_core_objects(build_params=None, build_dir=None):
    if build_params is None:
        build_params = get_build_parameters()
    if build_dir is None:
        build_dir = get_gsplat_build_directory(build_params)

    core_objects = _exclude_dispatcher_objects(_objects_in_build(build_dir))

    missing = [obj for obj in core_objects if not os.path.exists(obj)]
    if missing:
        raise RuntimeError(
            "JIT build did not produce expected gsplat objects:\n" + "\n".join(missing)
        )
    if not core_objects:
        raise RuntimeError(
            "No reusable gsplat core objects were found in the JIT build"
        )
    return core_objects


def build_and_load_gsplat():
    build_params = get_build_parameters()

    build_dir = get_gsplat_build_directory(build_params)

    # If JIT is interrupted it might leave a lock in the build directory.
    # We dont want it to exist in any case.
    try:
        os.remove(os.path.join(build_dir, "lock"))
    except OSError:
        pass

    # Check if the build parameters have changed since last build (if any
    saved_build_params_fname = os.path.join(build_dir, "build_params.json")
    saved_build_params = None
    build_params_changed = False
    try:
        if os.path.exists(saved_build_params_fname):
            with open(saved_build_params_fname, "r") as f:
                saved_build_params = SimpleNamespace(**json.load(f))
            build_params_changed = saved_build_params != build_params
    except Exception as e:
        if _console is not None:
            _console.print(
                f"[bold yellow]gsplat: rebuilding due to error loading saved build parameters: {e}"
            )
        else:
            print(
                f"gsplat: rebuilding due to error loading saved build parameters: {e}"
            )

    # If parameters have changed,
    if build_params_changed:
        # Build gsplat from scratch
        shutil.rmtree(build_dir)
        # Print out what triggered the rebuild (for debugging...)
        if saved_build_params is not None:
            if _console is not None:
                _console.print(
                    f"[bold yellow]gsplat: rebuilding due to build parameter change"
                )
            else:
                print("gsplat: rebuilding due to build parameter change")
            saved_dict = saved_build_params.__dict__
            current_dict = build_params.__dict__
            for k in sorted(set(saved_dict) | set(current_dict)):
                saved_val = saved_dict.get(k, "<missing>")
                current_val = current_dict.get(k, "<missing>")
                if saved_val != current_val:
                    if _console is not None:
                        _console.print(f"[white] old {k}: {saved_val}")
                        _console.print(f"[white] new {k}: {current_val}")
                    else:
                        print(f"  old {k}: {saved_val}")
                        print(f"  new {k}: {current_val}")

    # Make sure the build directory exists.
    if build_dir:
        os.makedirs(build_dir, exist_ok=True)

    # Save our current build parameters
    with open(saved_build_params_fname, "w") as f:
        json.dump(build_params.__dict__, f)

    @contextmanager
    def status_context():
        tic = time.time()
        msg = f"gsplat: Setting up CUDA with MAX_JOBS={MAX_JOBS if MAX_JOBS else 'max'} (This may take a few minutes the first time)"
        if _console is not None:
            ctx = _console.status(f"[bold yellow]{msg}", spinner="bouncingBall")
        else:
            print(msg)
            ctx = nullcontext()
        with ctx:
            yield

        toc = time.time()
        if _console is not None:
            _console.print(
                f"[green]gsplat: CUDA extension has been set up successfully in {toc - tic:.2f} seconds.[/green]"
            )
        else:
            print(
                f"gsplat: CUDA extension has been set up successfully in {toc - tic:.2f} seconds."
            )

    # If the build exists, we assume the extension has been built
    # and we can load it.
    module_exists = os.path.exists(
        os.path.join(build_dir, f"{build_params.name}.so")
    ) or os.path.exists(os.path.join(build_dir, f"{build_params.name}.lib"))

    with (
        status_context() if not module_exists or build_params_changed else nullcontext()
    ):
        # If the JIT build happens concurrently in multiple processes,
        # race conditions can occur when removing the lock file at:
        # https://github.com/pytorch/pytorch/blob/e3513fb2af7951ddf725d8c5b6f6d962a053c9da/torch/utils/cpp_extension.py#L1736
        # But it's ok so we catch this exception and ignore it.
        envvars_to_remove = []
        try:
            if not NINJA_STATUS:
                envvars_to_remove.append("NINJA_STATUS")
                os.environ["NINJA_STATUS"] = "[%f/%t %r %es] "

            gsplat_module = jit.load(
                name=build_params.name,
                sources=build_params.sources,
                extra_cflags=build_params.extra_cflags,
                extra_cuda_cflags=build_params.extra_cuda_cflags,
                extra_include_paths=build_params.extra_include_paths,
                extra_ldflags=build_params.extra_ldflags,
                build_directory=build_dir,
                verbose=VERBOSE,
            )
            return gsplat_module
        except OSError:
            # The module should already be compiled if we get OSError
            return jit._import_module_from_library(build_params.name, build_dir, True)
        finally:
            for envvar in envvars_to_remove:
                os.environ.pop(envvar)


def get_loaded_csrc_path():
    """Return the path of the loaded ``gsplat.csrc`` extension, if any."""
    module = sys.modules.get("gsplat.csrc")
    path = getattr(module, "__file__", None) if module else None
    return path if path and os.path.exists(path) else None


def _setup_py_core_objects_for_loaded_csrc(return_build_dir=False):
    """Return the setup.py-built ``.o`` files matching the loaded ``csrc.so``.

    For an editable install (``pip install -e .``), ``csrc.so`` lives at
    ``<repo>/gsplat/csrc.so`` and the matching object files are at
    ``<repo>/build/temp.<platform>-cpython-<tag>/gsplat/cuda/{csrc/*.o,ext.o}``.
    ``ext.o`` is excluded for the same reason ``get_gsplat_core_objects`` skips
    it: ``ext.cpp`` owns ``TORCH_LIBRARY(gsplat)`` registration and the gtest
    binary calls implementation entry points directly.

    Returns ``None`` if the layout doesn't match — e.g., ``csrc.so`` was
    installed from a wheel or to site-packages, so there is no local ``build``
    directory.
    """
    csrc_path = get_loaded_csrc_path()
    if not csrc_path:
        return None
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(csrc_path)))
    build_root = os.path.join(repo_root, "build")
    if not os.path.isdir(build_root):
        return None
    temp_dirs = glob.glob(
        os.path.join(build_root, f"temp.*-{sys.implementation.cache_tag}")
    )
    if not temp_dirs:
        return None
    # Multiple platform builds shouldn't normally exist side-by-side. Pick the
    # most recently touched one as a tie-breaker.
    temp_dir = max(temp_dirs, key=os.path.getmtime)
    cuda_dir = os.path.join(temp_dir, "gsplat", "cuda")
    if not os.path.isdir(cuda_dir):
        return None
    # Without a build.ninja we can't tell live objects from stale leftovers,
    # so treat that the same as a layout mismatch rather than erroring out.
    if not os.path.exists(os.path.join(temp_dir, "build.ninja")):
        return None
    core_objects = _exclude_dispatcher_objects(
        _objects_in_build(temp_dir, object_dir=cuda_dir, recursive=True)
    )
    if return_build_dir:
        return core_objects or None, temp_dir
    return core_objects or None


def _read_setup_py_ninja_flag_tokens(prefix):
    """Return tokens of a ``<prefix>= ...`` line from setup.py's build.ninja."""
    result = _setup_py_core_objects_for_loaded_csrc(return_build_dir=True)
    if result is None:
        return None
    _, temp_dir = result
    ninja_path = os.path.join(temp_dir, "build.ninja")
    with open(ninja_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(prefix):
                return shlex.split(line.split("=", 1)[1])
    return None


def _setup_py_extra_cflags_for_loaded_csrc():
    tokens = _read_setup_py_ninja_flag_tokens("post_cflags = ")
    if tokens is None:
        return None
    ignored_flags = {"-DTORCH_API_INCLUDE_EXTENSION_H"}
    return [
        flag
        for flag in tokens
        if flag not in ignored_flags and not flag.startswith("-DTORCH_EXTENSION_NAME=")
    ]


def _setup_py_extra_cuda_cflags_for_loaded_csrc():
    """Reconstruct ``extra_cuda_cflags`` from setup.py's build.ninja.

    Strips the prefix/suffix that torch's CUDAExtension path injects on
    Linux/macOS (see ``unix_cuda_flags`` and ``COMMON_NVCC_FLAGS`` in
    ``torch.utils.cpp_extension``) so the result is comparable with the
    ``extra_cuda_cflags`` produced by ``get_build_parameters()``.
    """
    tokens = _read_setup_py_ninja_flag_tokens("cuda_post_cflags = ")
    if tokens is None:
        return None
    ignored_flags = {
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
    }
    ignored_prefixes = ("-DTORCH_EXTENSION_NAME=", "-gencode=")
    # `--compiler-options '-fPIC'` from unix_cuda_flags and `-ccbin $CC` are
    # two-token sequences; build.py never emits either token.
    paired_flags = {"--compiler-options", "-ccbin"}

    out = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in paired_flags and i + 1 < len(tokens):
            i += 2
            continue
        if token in ignored_flags or any(token.startswith(p) for p in ignored_prefixes):
            i += 1
            continue
        out.append(token)
        i += 1
    return out


def _setup_py_build_parameters_mismatch_reason(build_params):
    recorded_cflags = _setup_py_extra_cflags_for_loaded_csrc()
    recorded_cuda_cflags = _setup_py_extra_cuda_cflags_for_loaded_csrc()
    if NUM_CHANNELS is not None:
        recorded_cflags += [f"-DGSPLAT_NUM_CHANNELS={NUM_CHANNELS}"]
        recorded_cuda_cflags += [
            '-DGSPLAT_NUM_CHANNELS="' + NUM_CHANNELS.replace(",", "\\,") + '"'
        ]
    # `_setup_py_extra_cuda_cflags_for_loaded_csrc()` drops every torch-injected
    # CUDA flag, including `--expt-relaxed-constexpr`, which build.py also adds.
    # Strip the build_params copy so the two sides remain comparable.
    expected_cuda_cflags = [
        f for f in build_params.extra_cuda_cflags if f != "--expt-relaxed-constexpr"
    ]
    if (recorded_cflags is None or recorded_cflags == build_params.extra_cflags) and (
        recorded_cuda_cflags is None or recorded_cuda_cflags == expected_cuda_cflags
    ):
        return None

    return (
        "gsplat.csrc is loaded from setup.py-built objects compiled with "
        "different compiler flags than the native C++ test harness would use. "
        "Rebuild from source with `pip install -e .` or rerun pytest with "
        "matching GSPLAT build environment variables.\n"
        f"setup.py extra_cflags: {recorded_cflags}\n"
        f"pytest extra_cflags: {build_params.extra_cflags}\n"
        f"setup.py extra_cuda_cflags: {recorded_cuda_cflags}\n"
        f"pytest extra_cuda_cflags: {expected_cuda_cflags}"
    )


_LINK_INPUTS_MISSING_MSG = (
    "gsplat.csrc is loaded but no setup.py-built .o files were found under "
    "<repo>/build/temp.*-cpython-<tag>/gsplat/cuda/. Native C++ tests need "
    "ABI-matching object files (some test symbols are not exported by "
    "csrc.so). Rebuild from source with `pip install -e .` to produce the "
    "local build artifacts."
)


def get_gsplat_link_inputs_skip_reason(build_params=None):
    """Return a skip reason when native C++ tests cannot run safely against
    the loaded ``gsplat.csrc``, else None.

    Detects:
      * ``csrc.so`` is loaded but no setup.py-built ``.o`` files are
        available (e.g., wheel install, or modern editable mode that
        cleans up ``build/temp.*/``).
      * ``csrc.so`` was compiled with C++ flags that differ from the
        pytest harness's ``build_params.extra_cflags`` — linking the
        reused objects would produce an ABI-mixed gtest binary.

    Other failure modes of ``get_gsplat_link_inputs()`` (JIT build errors,
    missing CUDA toolkit) are not detected here — they surface at the call
    site.
    """
    if not get_loaded_csrc_path():
        return None
    if _setup_py_core_objects_for_loaded_csrc() is None:
        return _LINK_INPUTS_MISSING_MSG
    if build_params is not None:
        return _setup_py_build_parameters_mismatch_reason(build_params)
    return None


def get_gsplat_link_inputs():
    """Return ABI-matching link inputs for the loaded gsplat extension.

    Native test binaries that exercise gsplat C++ entry points need to link
    against the same compiled objects that produced the loaded ``csrc.so``.
    There are two supported sources:

    * setup.py-built ``gsplat.csrc`` (editable install). Reuse the ``.o``
      files at ``<repo>/build/temp.*-cpython-<tag>/gsplat/cuda/`` so the
      gtest binary picks up implementation symbols — including
      hidden-visibility ones and other non-exported extern symbols that csrc.so
      does not surface in its dynamic symbol table — with ABI identical to the
      loaded extension. Symbols with internal linkage (static at file scope,
      anonymous-namespace members) remain unreachable from another TU even via
      direct .o linking.

    * Pure JIT environment. Trigger the JIT build and return the
      implementation ``.o`` files (``ext.o`` excluded).

    Raises ``FileNotFoundError`` if ``gsplat.csrc`` is loaded but its build
    artifacts are not available locally (e.g., wheel install). Callers that
    prefer to skip native tests in that case should consult
    ``get_gsplat_link_inputs_skip_reason()`` first.
    """
    if get_loaded_csrc_path():
        objects = _setup_py_core_objects_for_loaded_csrc()
        if objects is not None:
            return objects
        # Setup.py-built csrc.so is loaded but its .o files aren't on disk:
        # Fall through to JIT — `ext.o` is still excluded, so no namespace
        # re-registration.
    build_and_load_gsplat()
    build_params = get_build_parameters()
    return get_gsplat_core_objects(
        build_params, get_gsplat_build_directory(build_params)
    )


__all__ = [
    "build_and_load_gsplat",
    "get_build_parameters",
    "get_gsplat_build_directory",
    "get_gsplat_core_objects",
    "get_gsplat_link_inputs",
    "get_gsplat_link_inputs_skip_reason",
    "get_loaded_csrc_path",
]
