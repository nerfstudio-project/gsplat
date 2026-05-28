# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tri-target build entry point for gsplat.

Selects between three backends at install/JIT time via the
``GSPLAT_BACKEND`` environment variable:

- ``cuda``         (default) — upstream NVIDIA CUDA path. Sources are the
                                ``gsplat/cuda/csrc/*.cu`` files and the
                                upstream submodule glm under
                                ``gsplat/cuda/csrc/third_party/glm``.
- ``rocm_radeon``  — AMD RDNA (gfx1100/gfx1200). Sources are the SHARED
                      ``.hip`` files from ``gsplat/cuda/csrc/`` plus the
                      Radeon-specific overrides from ``arch/radeon/...``.
- ``rocm_instinct`` — AMD CDNA (gfx942/gfx950). Same as Radeon but pulling
                      ``arch/instinct/...`` overrides.

The two ROCm backends additionally pull translator-shared ROCm overrides
(``arch/rocm/...``) for files that are byte-identical between Radeon and
Instinct but differ from upstream (e.g. ``CameraWrappers.h``).

This file is intentionally small. All backend-specific compile-flag and
include-path logic lives below in three clearly separated branches; the
ROCm branches preserve the hand-tuned per-arch flag sets from the original
Radeon and Instinct ``build.py`` files (notably FAST_MATH semantics).
"""

import glob
import json
import os
import platform
import shutil
import sys
import time
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import torch

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

try:
    from rich.console import Console
    _console = Console()
except ImportError:
    _console = None

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
_VALID_BACKENDS = ("cuda", "rocm_radeon", "rocm_instinct")
GSPLAT_BACKEND = os.environ.get("GSPLAT_BACKEND", "cuda").lower()
if GSPLAT_BACKEND not in _VALID_BACKENDS:
    raise RuntimeError(
        f"GSPLAT_BACKEND={GSPLAT_BACKEND!r} is invalid; "
        f"expected one of {_VALID_BACKENDS}."
    )

# PYTORCH_ROCM_ARCH resolution for ROCm backends. Compiling for many archs
# multiplies build time by N (each --offload-arch is a separate AMDGPU codegen
# pass). Resolution order, highest precedence first:
#   1. user-set PYTORCH_ROCM_ARCH wins (escape hatch for portable builds).
#   2. GSPLAT_ROCM_ARCH=<list> wins if set.
#   3. GSPLAT_ROCM_ARCH=auto|detect|unset -> query torch for the live GPU and
#      compile for that single arch (fastest first-import path).
#   4. detection fails (no GPU visible) -> fall back to the 2-arch backend
#      default so the build does not silently produce a binary for nothing.
def _detect_current_rocm_arch() -> str | None:
    """Return the gfx name of GPU 0, or None if no GPU is visible."""
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return None
        # gcnArchName looks like "gfx942:sramecc+:xnack-"; strip feature flags.
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        return None


_ROCM_BACKEND_DEFAULTS = {
    "rocm_radeon": "gfx1100;gfx1200",
    "rocm_instinct": "gfx942;gfx950",
}

if GSPLAT_BACKEND in _ROCM_BACKEND_DEFAULTS:
    # Treat empty/whitespace-only PYTORCH_ROCM_ARCH as unset. The rocm/pytorch
    # base image ships with an empty value, which would otherwise look like
    # "user explicitly chose empty" and skip auto-detect → torch falls back to
    # its full fat-list default and the build takes 5-6x longer.
    if os.environ.get("PYTORCH_ROCM_ARCH", "").strip():
        _arch_source = "PYTORCH_ROCM_ARCH (user)"
    else:
        _user_choice = os.environ.get("GSPLAT_ROCM_ARCH", "auto").strip()
        if _user_choice and _user_choice.lower() not in ("auto", "detect"):
            os.environ["PYTORCH_ROCM_ARCH"] = _user_choice
            _arch_source = "GSPLAT_ROCM_ARCH (user)"
        else:
            _detected = _detect_current_rocm_arch()
            if _detected:
                os.environ["PYTORCH_ROCM_ARCH"] = _detected
                _arch_source = f"auto-detect ({_detected})"
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = _ROCM_BACKEND_DEFAULTS[GSPLAT_BACKEND]
                _arch_source = "backend default (no GPU visible to detect)"
    print(
        f"[gsplat.build] PYTORCH_ROCM_ARCH = {os.environ['PYTORCH_ROCM_ARCH']} "
        f"[source: {_arch_source}]"
    )

print(f"[gsplat.build] GSPLAT_BACKEND = {GSPLAT_BACKEND}")

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PATH = os.path.dirname(os.path.abspath(__file__))           # gsplat/cuda/
GSPLAT_DIR = os.path.dirname(PATH)                          # gsplat/
REPO_ROOT = os.path.dirname(GSPLAT_DIR)                     # repo root
ARCH_DIR = os.path.join(REPO_ROOT, "arch")
ARCH_RADEON = os.path.join(ARCH_DIR, "radeon")
ARCH_INSTINCT = os.path.join(ARCH_DIR, "instinct")
ARCH_ROCM = os.path.join(ARCH_DIR, "rocm")

# ---------------------------------------------------------------------------
# Common env-var knobs (identical across backends)
# ---------------------------------------------------------------------------
DEBUG = os.getenv("DEBUG", "0") == "1"
FAST_MATH = os.getenv("FAST_MATH", "1") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
NVCC_FLAGS = os.getenv("NVCC_FLAGS", "")
MAX_JOBS = os.getenv("MAX_JOBS")
NINJA_STATUS = os.getenv("NINJA_STATUS")
VERBOSE = os.getenv("VERBOSE", "0") == "1"

BUILD_3DGUT = os.getenv("BUILD_3DGUT")
BUILD_3DGS = os.getenv("BUILD_3DGS")
BUILD_2DGS = os.getenv("BUILD_2DGS")
BUILD_ADAM = os.getenv("BUILD_ADAM")
BUILD_RELOC = os.getenv("BUILD_RELOC")
BUILD_CAMERA_WRAPPERS = (
    os.getenv("BUILD_CAMERA_WRAPPERS", "1" if DEBUG else "0") == "1"
)

NUM_CHANNELS = os.getenv("NUM_CHANNELS")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_rocm() -> bool:
    return GSPLAT_BACKEND in ("rocm_radeon", "rocm_instinct")


def _rocm_sources() -> list[str]:
    """Build the ROCm source list as a UNION of:
    - SHARED ``.hip`` files at ``gsplat/cuda/csrc/`` (from arch/rocm or
      already inlined here in the merged repo) — actually note in the
      merged tri-target layout the ``.hip`` files live under
      ``arch/<variant>/gsplat/cuda/csrc/``, NOT under ``gsplat/cuda/csrc/``
      (which holds only upstream ``.cu`` files).
    - The .cpp dispatchers at ``arch/rocm/gsplat/cuda/csrc/`` (rad==ins).
    - The arch-specific ``.hip`` overrides from ``arch/radeon`` or
      ``arch/instinct`` (every translated ``.hip`` is divergent in the
      gsplat port, so all of them come from the variant tree).
    - The single ``ext.cpp`` at ``gsplat/cuda/ext.cpp`` (upstream).
    """
    variant_dir = ARCH_RADEON if GSPLAT_BACKEND == "rocm_radeon" else ARCH_INSTINCT
    csrc_variant = os.path.join(variant_dir, "gsplat", "cuda", "csrc")
    csrc_rocm = os.path.join(ARCH_ROCM, "gsplat", "cuda", "csrc")

    sources: list[str] = []
    # All .hip files from the variant tree (arch/radeon or arch/instinct)
    sources += sorted(glob.glob(os.path.join(csrc_variant, "*.hip")))
    # All shared .cpp dispatchers from arch/rocm/
    sources += sorted(glob.glob(os.path.join(csrc_rocm, "*.cpp")))
    # ext.cpp (upstream, identical across all backends)
    sources += [os.path.join(PATH, "ext.cpp")]
    return sources


def _cuda_sources() -> list[str]:
    """Upstream CUDA source list — unchanged from upstream build.py."""
    return (
        list(glob.glob(os.path.join(PATH, "csrc/*.cu")))
        + list(glob.glob(os.path.join(PATH, "csrc/*.cpp")))
        + [os.path.join(PATH, "ext.cpp")]
    )


def _isolate_glm_for_rocm(src_glm: str) -> str:
    """Mirror vendored glm to a path OUTSIDE the source tree.

    PyTorch's ``torch.utils.cpp_extension.load`` invokes ``hipify_python``
    on every directory reachable from ``extra_include_paths`` whose path
    sits inside the project. Hipify will then create ``*_hip.hpp`` sibling
    headers next to glm's vendored ``*.hpp`` files (e.g. ``qualifier_hip.hpp``
    next to ``qualifier.hpp``). Both headers redefine ``enum glm::qualifier``
    and the same templates, and ``#pragma once`` cannot dedupe because they
    are physically distinct files. The TU then sees multiple definitions and
    the build fails with hundreds of glm redefinition errors in ``ext.cpp``.

    The fix is to expose glm to the compiler from a path that hipify cannot
    discover. We mirror only the upstream glm tree (skipping any pre-existing
    ``*_hip*`` artifacts from a previous polluted build) into a stable
    location outside the merged tree, then return that path so hipify never
    walks it. Originals already in the source tree are left untouched.
    """
    dst_root = os.path.join(
        "/tmp" if os.path.isdir("/tmp") else os.path.expanduser("~"),
        "gsplat_glm_isolated",
        f"{GSPLAT_BACKEND}",
    )
    dst_glm = os.path.join(dst_root, "glm")
    if os.path.isdir(dst_glm):
        shutil.rmtree(dst_glm)
    os.makedirs(dst_root, exist_ok=True)

    def _ignore(_dir: str, names: list[str]) -> list[str]:
        return [n for n in names if "_hip" in n]

    src_inner = os.path.join(src_glm, "glm")
    if os.path.isdir(src_inner):
        shutil.copytree(src_inner, dst_glm, ignore=_ignore)
    else:
        shutil.copytree(src_glm, dst_glm, ignore=_ignore)
    return dst_root


def _rocm_include_paths() -> list[str]:
    """Include path list for ROCm builds.

    Search order (most specific first):
    1. ``arch/<variant>/gsplat/cuda/csrc`` — variant-divergent headers
       co-located with the .hip files (Config.h, MacroUtils.h).
    2. ``arch/<variant>/gsplat/cuda/include/hip_shim/cooperative_groups`` —
       variant's cooperative_groups shim (DIVERGENT_NEW).
    3. ``arch/<variant>/gsplat/cuda/include`` — variant-divergent headers
       (Common.h, etc.).
    4. ``arch/rocm/gsplat/cuda/csrc`` — shared-ROCm dispatcher headers.
    5. ``arch/rocm/gsplat/cuda/include`` — shared-ROCm overlay headers.
    6. ``gsplat/cuda/csrc`` — upstream csrc (fallback).
    7. ``gsplat/cuda/include`` — upstream include dir (fallback).
    8. ``third_party/cgshim`` — the shared ``cuda::std`` polyfills.
    9. **Isolated glm** at ``/tmp/gsplat_glm_isolated/<backend>`` — a clean
       hipify-proof copy of the vendored glm tree (see
       ``_isolate_glm_for_rocm``). The ``/opt/glm`` system install is used
       directly when present, since system paths sit outside the project
       tree and hipify will not touch them.
    """
    variant_dir = ARCH_RADEON if GSPLAT_BACKEND == "rocm_radeon" else ARCH_INSTINCT
    csrc_variant = os.path.join(variant_dir, "gsplat", "cuda", "csrc")
    inc_variant = os.path.join(variant_dir, "gsplat", "cuda", "include")
    csrc_rocm_shared = os.path.join(ARCH_ROCM, "gsplat", "cuda", "csrc")
    inc_rocm_shared = os.path.join(ARCH_ROCM, "gsplat", "cuda", "include")
    csrc_upstream = os.path.join(PATH, "csrc")
    inc_upstream = os.path.join(PATH, "include")
    cgshim = os.path.join(REPO_ROOT, "third_party", "cgshim")

    # NOTE: glm is intentionally NOT in the returned ``-I`` list. PyTorch's
    # hipify scans every directory that lands in ``extra_include_paths`` and
    # writes ``*_hip.hpp`` siblings next to the originals; this breaks
    # ``#pragma once`` (the originals and the ``_hip`` copies redefine the
    # same templates) and the ext.cpp TU explodes with hundreds of glm
    # redefinitions. Instead glm is added as ``-isystem`` further down in
    # ``_rocm_flags()`` via ``extra_cflags`` so hipify never sees it.
    return [
        csrc_variant,
        os.path.join(inc_variant, "hip_shim", "cooperative_groups"),
        os.path.join(inc_variant, "hip_shim"),
        inc_variant,
        csrc_rocm_shared,
        inc_rocm_shared,
        csrc_upstream,
        inc_upstream,
        cgshim,
    ]


def _rocm_glm_path() -> str:
    """Return a glm root suitable for ``-isystem``.

    Prefers ``/opt/glm`` if available; otherwise mirrors the upstream glm
    submodule at ``gsplat/cuda/csrc/third_party/glm`` to
    ``/tmp/gsplat_glm_isolated/<backend>`` (skipping any pre-existing
    ``_hip`` artifacts), well outside any directory that PyTorch's hipify
    will visit. This is the same glm source the CUDA backend uses, so both
    backends compile against a single pinned version.
    """
    if os.path.isdir("/opt/glm"):
        return "/opt/glm"
    return _isolate_glm_for_rocm(
        os.path.join(PATH, "csrc", "third_party", "glm")
    )


def _cuda_include_paths() -> list[str]:
    return [
        os.path.join(PATH, "include/"),
        os.path.join(PATH, "csrc", "third_party", "glm"),
    ]


# ---------------------------------------------------------------------------
# Backend-specific compile flags
# ---------------------------------------------------------------------------
def _cuda_flags():
    """Upstream CUDA flag set (mirrors the original upstream build.py)."""
    extra_cflags: list[str] = []
    extra_cuda_cflags: list[str] = []
    extra_ldflags: list[str] = []

    if sys.platform == "win32":
        extra_cflags += ["/std:c++20", "/Zc:preprocessor", "-DWIN32_LEAN_AND_MEAN"]
        extra_cuda_cflags += [
            "-std=c++20", "-allow-unsupported-compiler",
            "-Xcompiler", "/Zc:preprocessor", "-DWIN32_LEAN_AND_MEAN",
        ]
    else:
        extra_cflags = ["-std=c++20"]

    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_cflags += ["-arch", "arm64"]
        extra_ldflags += ["-arch", "arm64"]

    extra_cuda_cflags += ["--forward-unknown-opts"]

    if sys.platform == "win32":
        if DEBUG:
            extra_cflags += ["/Zi", "/Od"]
            extra_cuda_cflags += ["-Od"]
        else:
            extra_cflags += ["/O2", "-DNDEBUG"]
            extra_cuda_cflags += ["-O2", "-DNDEBUG"]
    else:
        if DEBUG:
            extra_cflags += ["-g", "-O0", "-Wall"]
            extra_cuda_cflags += [
                "-lineinfo", "-Xcompiler=-Werror", "--Werror", "all-warnings",
            ]
        else:
            extra_cflags += ["-O3", "-DNDEBUG"]

    if FAST_MATH:
        extra_cuda_cflags += ["-use_fast_math"]

    extra_cuda_cflags += ["-diag-suppress", "20012,186"]
    if not os.name == "nt":
        extra_cflags += ["-Wno-attributes", "-Wno-unknown-pragmas"]

    return extra_cflags, extra_cuda_cflags, extra_ldflags


def _rocm_flags():
    """Shared ROCm flag set, with a per-variant FAST_MATH branch.

    Radeon uses ``-ffast-math`` (full unsafe-math). Instinct uses the
    conservative set that mirrors NVIDIA ``--use_fast_math`` semantics
    without reordering atomic-summed gradient terms (lesson from the
    Instinct port — see commit history).
    """
    extra_cflags: list[str] = []
    extra_cuda_cflags: list[str] = []
    extra_ldflags: list[str] = []

    if sys.platform == "win32":
        extra_cflags += ["/std:c++20", "/Zc:preprocessor", "-DWIN32_LEAN_AND_MEAN"]
        extra_cuda_cflags += [
            "-std=c++20", "-allow-unsupported-compiler",
            "-Xcompiler", "/Zc:preprocessor", "-DWIN32_LEAN_AND_MEAN",
        ]
    else:
        extra_cflags = ["-std=c++20"]

    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_cflags += ["-arch", "arm64"]
        extra_ldflags += ["-arch", "arm64"]

    # --forward-unknown-opts is nvcc-only; hipcc/amdclang++ rejects it.

    if sys.platform == "win32":
        if DEBUG:
            extra_cflags += ["/Zi", "/Od"]
            extra_cuda_cflags += ["-Od"]
        else:
            extra_cflags += ["/O2", "-DNDEBUG"]
            extra_cuda_cflags += ["-O2", "-DNDEBUG"]
    else:
        if DEBUG:
            extra_cflags += ["-g", "-O0", "-Wall"]
            # ROCm 7.x amdclang++ ICEs at -O0 — bump to -O1 minimum.
            extra_cuda_cflags += ["-g", "-O1", "-Wall"]
        else:
            extra_cflags += ["-O3", "-DNDEBUG"]
            extra_cuda_cflags += ["-O3", "-DNDEBUG"]

    if FAST_MATH:
        if GSPLAT_BACKEND == "rocm_radeon":
            extra_cuda_cflags += ["-ffast-math"]
        else:  # rocm_instinct — conservative set, see docstring above
            extra_cuda_cflags += [
                "-fno-math-errno",
                "-ffp-contract=fast",
                "-fdenormal-fp-math=preserve-sign",
                "-fgpu-flush-denormals-to-zero",
            ]

    # No nvcc -diag-suppress; mute via gcc-style flags.
    extra_cuda_cflags += [
        "-Wno-unused-result",
        "-Wno-unused-variable",
        "-Wno-deprecated-declarations",
    ]
    if not os.name == "nt":
        extra_cflags += ["-Wno-attributes", "-Wno-unknown-pragmas"]

    # glm via -isystem (NOT -I) so PyTorch's hipify cannot scan it; see
    # ``_rocm_glm_path`` and the comment in ``_rocm_include_paths``.
    glm_root = _rocm_glm_path()
    extra_cflags += ["-isystem", glm_root]
    extra_cuda_cflags += ["-isystem", glm_root]

    # Force-include the at::cuda::* compat shim on every TU so PyTorch ROCm
    # builds that lack the masquerade aliases (notably 2.8 release) get the
    # missing `at::cuda::getCurrentCUDAStream` and `at::cuda::OptionalCUDAGuard`
    # synthesized via the `c10::hip` masquerade. No-op on PyTorch versions
    # that already define them. See arch/rocm/.../torch_rocm_compat.h.
    compat_shim = os.path.join(
        ARCH_ROCM, "gsplat", "cuda", "include", "torch_rocm_compat.h"
    )
    if os.path.isfile(compat_shim):
        extra_cflags += ["-include", compat_shim]
        extra_cuda_cflags += ["-include", compat_shim]

    return extra_cflags, extra_cuda_cflags, extra_ldflags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_build_parameters():
    name = "gsplat_cuda"

    if _is_rocm():
        sources = _rocm_sources()
        extra_include_paths = _rocm_include_paths()
        extra_cflags, extra_cuda_cflags, extra_ldflags = _rocm_flags()
        # Strip CameraWrappers.hip if not enabled (upstream parity).
        if not BUILD_CAMERA_WRAPPERS:
            sources = [s for s in sources if not s.endswith("CameraWrappers.hip")]
    else:
        sources = _cuda_sources()
        extra_include_paths = _cuda_include_paths()
        extra_cflags, extra_cuda_cflags, extra_ldflags = _cuda_flags()
        if not BUILD_CAMERA_WRAPPERS:
            sources = [s for s in sources if not s.endswith("csrc/CameraWrappers.cu")]

    # Shared per-feature -D macros.
    for var, name_macro in [
        (BUILD_2DGS, "GSPLAT_BUILD_2DGS"),
        (BUILD_3DGS, "GSPLAT_BUILD_3DGS"),
        (BUILD_3DGUT, "GSPLAT_BUILD_3DGUT"),
        (BUILD_ADAM, "GSPLAT_BUILD_ADAM"),
        (BUILD_RELOC, "GSPLAT_BUILD_RELOC"),
    ]:
        if var is not None:
            extra_cflags += [f"-D{name_macro}={var}"]
            if sys.platform == "win32":
                extra_cuda_cflags += [f"-D{name_macro}={var}"]
    if BUILD_CAMERA_WRAPPERS:
        extra_cflags += ["-DGSPLAT_BUILD_CAMERA_WRAPPERS=1"]
        if sys.platform == "win32":
            extra_cuda_cflags += ["-DGSPLAT_BUILD_CAMERA_WRAPPERS=1"]

    extra_ldflags += [] if WITH_SYMBOLS or sys.platform == "win32" else ["-s"]

    if torch.version.hip:
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
            extra_cuda_cflags += ["-Xcompiler", "/openmp", "-DAT_PARALLEL_OPENMP"]
        else:
            extra_cflags += ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    if sys.platform != "win32":
        extra_cuda_cflags += extra_cflags

    if DEBUG and sys.platform != "win32":
        extra_cflags += ["-Werror"]

    if NUM_CHANNELS is not None:
        extra_cuda_cflags += [
            '-DGSPLAT_NUM_CHANNELS="' + NUM_CHANNELS.replace(",", "\\,") + '"'
        ]
        extra_cflags += [f"-DGSPLAT_NUM_CHANNELS={NUM_CHANNELS}"]

    extra_cuda_cflags += [] if NVCC_FLAGS == "" else NVCC_FLAGS.split(" ")

    print(
        f"[gsplat.build] backend={GSPLAT_BACKEND}  sources={len(sources)}  "
        f"includes={len(extra_include_paths)}"
    )

    return SimpleNamespace(
        name=name,
        extra_include_paths=extra_include_paths,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
    )


def build_and_load_gsplat():
    build_params = get_build_parameters()

    build_dir = jit._get_build_directory(build_params.name, verbose=False)
    try:
        os.remove(os.path.join(build_dir, "lock"))
    except OSError:
        pass

    saved_build_params_fname = os.path.join(build_dir, "build_params.json")
    saved_build_params = None
    build_params_changed = False
    try:
        if os.path.exists(saved_build_params_fname):
            with open(saved_build_params_fname, "r") as f:
                saved_build_params = SimpleNamespace(**json.load(f))
            build_params_changed = saved_build_params != build_params
    except Exception as e:
        msg = f"gsplat: rebuilding due to error loading saved build parameters: {e}"
        if _console is not None:
            _console.print(f"[bold yellow]{msg}")
        else:
            print(msg)

    if build_params_changed:
        shutil.rmtree(build_dir)
        if saved_build_params is not None:
            msg = "gsplat: rebuilding due to build parameter change"
            if _console is not None:
                _console.print(f"[bold yellow]{msg}")
            else:
                print(msg)
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

    if build_dir:
        os.makedirs(build_dir, exist_ok=True)

    with open(saved_build_params_fname, "w") as f:
        json.dump(build_params.__dict__, f)

    @contextmanager
    def status_context():
        tic = time.time()
        msg = (
            f"gsplat: Setting up {GSPLAT_BACKEND} extension with "
            f"MAX_JOBS={MAX_JOBS if MAX_JOBS else 'max'} "
            "(This may take a few minutes the first time)"
        )
        if _console is not None:
            ctx = _console.status(f"[bold yellow]{msg}", spinner="bouncingBall")
        else:
            print(msg)
            ctx = nullcontext()
        with ctx:
            yield
        toc = time.time()
        done = (
            f"gsplat: {GSPLAT_BACKEND} extension has been set up "
            f"successfully in {toc - tic:.2f} seconds."
        )
        if _console is not None:
            _console.print(f"[green]{done}[/green]")
        else:
            print(done)

    module_exists = os.path.exists(
        os.path.join(build_dir, f"{build_params.name}.so")
    ) or os.path.exists(os.path.join(build_dir, f"{build_params.name}.lib"))

    with (
        status_context() if not module_exists or build_params_changed else nullcontext()
    ):
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
            return jit._import_module_from_library(build_params.name, build_dir, True)
        finally:
            for envvar in envvars_to_remove:
                os.environ.pop(envvar)


__all__ = ["get_build_parameters", "build_and_load_gsplat", "GSPLAT_BACKEND"]
