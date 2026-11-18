# SPDX-FileCopyrightText: Copyright 2023-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import glob
import os
import os.path as osp
import pathlib
import platform
import re
import site
import subprocess
import sys


# Read ``__version__`` from ``gsplat/version.py`` when the rest of the source
# tree is alongside us (the canonical context for ``pip install`` / wheel
# builds). When setup.py is imported standalone (e.g. by
# ``docker/check_deps.sh`` from a directory with only setup.py) we leave
# ``__version__`` as ``None`` — the dep-extraction path doesn't need it, and
# ``_setup()`` will surface a clear setuptools error if invoked without it.
__version__ = None
_SETUP_DIR = os.path.dirname(os.path.abspath(__file__))
_version_file = os.path.join(_SETUP_DIR, "gsplat", "version.py")
if os.path.exists(_version_file):
    with open(_version_file, "r") as f:
        exec(f.read())

URL = "https://github.com/nerfstudio-project/gsplat"

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
BUILD_EXPERIMENTAL = os.getenv("BUILD_EXPERIMENTAL", "1") == "1"


def _torch_is_rocm_build() -> bool:
    """Whether the installed PyTorch is a ROCm/HIP build.

    Uses the installed distribution's version metadata (ROCm wheels are tagged
    ``+rocmX.Y``) instead of importing torch: ``setup.py`` must be importable
    without torch (e.g. ``docker/check_deps.sh``), and the ROCm environment has
    to be bootstrapped before torch's cpp_extension module is imported.
    """
    try:
        from importlib.metadata import version

        return "rocm" in version("torch").lower()
    except Exception:
        return False


# Build for ROCm/HIP instead of CUDA. Auto-detected from the installed PyTorch;
# set ``BUILD_ROCM=1`` / ``BUILD_ROCM=0`` to force it on or off (for example for
# a source-built ROCm PyTorch whose version string carries no ``+rocm`` tag).
_BUILD_ROCM_ENV = os.getenv("BUILD_ROCM")
IS_ROCM = (
    _BUILD_ROCM_ENV == "1" if _BUILD_ROCM_ENV is not None else _torch_is_rocm_build()
)

# ROCm-only build knobs.
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"
ENABLE_TEST_COVERAGE = os.getenv("ENABLE_TEST_COVERAGE", "0") == "1"

def _find_rocm_sdk_in_site_packages():
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        if not sp:
            continue
        sdk_devel = osp.join(sp, "_rocm_sdk_devel")
        if osp.isdir(osp.join(sdk_devel, "include", "hip")):
            return sdk_devel
        sdk_core = osp.join(sp, "_rocm_sdk_core")
        if osp.isdir(osp.join(sdk_core, "include", "hip")):
            return sdk_core
    return None


def _bootstrap_rocm_env():
    """Point ROCm tooling at pip SDK (devel preferred) before torch import."""
    rocm_home = os.environ.get("ROCM_HOME")
    if not rocm_home or not osp.isdir(osp.join(rocm_home, "include", "hip")):
        rocm_home = _find_rocm_sdk_in_site_packages()
        if not rocm_home and osp.isdir(osp.join("/opt/rocm", "include", "hip")):
            rocm_home = "/opt/rocm"
        if not rocm_home:
            rocm_home = "/opt/rocm"

    rocm_home = osp.abspath(rocm_home)
    rocm_bin = osp.join(rocm_home, "bin")
    rocm_llvm_bin = osp.join(rocm_home, "lib", "llvm", "bin")
    os.environ["ROCM_HOME"] = rocm_home
    os.environ["ROCM_PATH"] = rocm_home
    os.environ["PATH"] = os.pathsep.join(
        [rocm_bin, rocm_llvm_bin, os.environ.get("PATH", "")]
    )



# Point ROCm tooling at the SDK before torch's cpp_extension is imported, and cap
# parallel compile jobs (the HIP raster kernels are memory hungry) unless the
# user chose a value.
need_to_unset_max_jobs = False
if IS_ROCM:
    _bootstrap_rocm_env()
    if not os.getenv("MAX_JOBS"):
        need_to_unset_max_jobs = True
        os.environ["MAX_JOBS"] = "10"
        print(f"Setting MAX_JOBS to {os.environ['MAX_JOBS']}")


def _read_config_variable(name: str) -> str:
    """Read a scalar variable from the repository's simple ``config.yaml``."""

    config_path = os.path.join(_SETUP_DIR, "config.yaml")
    if not os.path.exists(config_path):
        raise RuntimeError(f"{config_path} is required to read {name}")

    prefix = f"{name}:"
    with open(config_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip().strip("\"'")

    raise RuntimeError(f"{name} is not set in {config_path}")


def _detect_cupy_requirement() -> str:
    """Pick a CuPy distribution that matches the local CUDA toolkit.

    The bare ``cupy`` package on PyPI is a source distribution that compiles
    against the local CUDA toolkit and routinely fails when stub libs or
    optional headers (e.g. cusparseLt) aren't present. Prefer a prebuilt
    wheel keyed on the detected CUDA major version. Set ``CUPY_PACKAGE`` to
    override (e.g. ``CUPY_PACKAGE=cupy-cuda12x`` or ``CUPY_PACKAGE=cupy``).
    """
    override = os.getenv("CUPY_PACKAGE")
    if override:
        return override

    # cupy-cudaXx wheels target NVIDIA CUDA and are irrelevant when building
    # against ROCm/HIP; skip the nvcc/CUDA-toolkit probing entirely.
    if IS_ROCM:
        return "cupy"

    import re
    import subprocess
    import warnings

    cuda_roots = []
    for env in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env)
        if root:
            cuda_roots.append(root)
    cuda_roots.append("/usr/local/cuda")

    # 1. Try ``nvcc --version`` from each candidate root, then PATH.
    nvcc_candidates = [os.path.join(r, "bin", "nvcc") for r in cuda_roots]
    nvcc_candidates.append("nvcc")

    for nvcc in nvcc_candidates:
        try:
            out = subprocess.check_output(
                [nvcc, "--version"], stderr=subprocess.STDOUT, text=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            continue
        m = re.search(r"release (\d+)\.", out)
        if m:
            return f"cupy-cuda{m.group(1)}x"
        # nvcc ran but its --version output didn't match. That's distinct from
        # the "nvcc not found" case (silently skipped above) — it usually
        # signals a broken shim (ccache wrapper, non-NVIDIA stub) rather than
        # a legitimate format change, since the CUDA toolkit's --version has
        # been stable for years. Surface the anomaly so a developer who later
        # ends up with a source-built cupy isn't left guessing why.
        warnings.warn(
            f"nvcc at {nvcc} returned an unparseable --version output; "
            "skipping this candidate.",
            stacklevel=2,
        )

    # 2. Fall back to cuda.h's CUDA_VERSION macro for environments where
    # nvcc is missing (runtime-only CUDA install) or wrapped in a way
    # that breaks ``--version`` (e.g. a misbehaving ccache shim).
    # CUDA_VERSION encodes major as the integer division by 1000
    # (e.g. 13020 → 13). cuda.h ships with every CUDA toolkit.
    for root in cuda_roots:
        try:
            with open(os.path.join(root, "include", "cuda.h")) as f:
                content = f.read()
        except (FileNotFoundError, OSError):
            continue
        m = re.search(r"^#define\s+CUDA_VERSION\s+(\d+)", content, re.MULTILINE)
        if m:
            return f"cupy-cuda{int(m.group(1)) // 1000}x"

    return "cupy"


INSTALL_REQUIRES = [
    "ninja",
    "numpy",
    "jaxtyping",
    "nvtx",
    "rich>=12",
    # N.B. Starting with PyTorch 2.11, the default install wheel uses CUDA 13.
    # However, PyTorch >= 2.11 still provides a prebuilt CUDA 12.6 wheel
    # for compatibility with older CUDA drivers and versions which can be installed
    # by passing --index-url https://download.pytorch.org/whl/cu126
    # torch.library.register_autograd needs PyTorch >=2.4;
    # Blackwell (sm_120) support needs PyTorch >=2.7
    "torch>=2.7",
    "typing_extensions; python_version<'3.8'",
]


def get_extras_require() -> dict:
    """Return the ``extras_require`` mapping consumed by ``setup()``.

    Exposed as a function (rather than a literal) so external tooling — e.g.
    ``docker/check_deps.sh`` — can ``import setup`` and read the resolved
    dependency list without re-parsing the source.
    """
    return {
        # lidar dependencies. Install them by `pip install gsplat[lidar]`
        "lidar": [
            "scipy",
        ],
        # examples / tutorial dependencies. The dynamic-surgical trainer and
        # the EndoNeRF parser/dataset import these at module top, but they
        # are not needed to use the core gsplat library — install with
        # `pip install gsplat[examples]`.
        "examples": [
            "Pillow",
            "tqdm",
            "tyro",
            "imageio>=2.37.2",
        ],
        # dev dependencies. Install them by `pip install gsplat[dev]`
        "dev": [
            "black[jupyter]==22.3.0",
            f"clang-format=={_read_config_variable('CLANG_FORMAT_VERSION')}",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest",
            "pytest-check",
            "pytest-env",
            "pytest-xdist==2.5.0",
            # Tests for examples/datasets/endonerf.py and the dynamic-surgical
            # trainer import Pillow + tqdm at module top — without these the
            # test collection ImportErrors on a fresh `[dev]` install.
            "Pillow",
            "tqdm",
            "typeguard>=2.13.3",
            "pyyaml>=6.0.1",
            "build",
            "twine",
            _detect_cupy_requirement(),
            "nerfacc>=0.5.3",
            "PLAS @ git+https://github.com/fraunhoferhhi/PLAS.git",
            "imageio>=2.37.2",
            "torchpq>=0.3.0.6",
        ],
    }


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_rocm_sdk_root():
    """Resolve ROCm SDK root: ROCM_HOME, pip SDK in venv, or /opt/rocm."""
    rocm_home = os.environ.get("ROCM_HOME")
    if rocm_home and osp.isdir(osp.join(rocm_home, "include", "hip")):
        return osp.abspath(rocm_home)
    found = _find_rocm_sdk_in_site_packages()
    if found:
        return found
    if osp.isdir(osp.join("/opt/rocm", "include", "hip")):
        return "/opt/rocm"
    return "/opt/rocm"


def sdk_has_real_libcudacxx(rocm_home):
    """True if the ROCm SDK ships a real libcu++/CCCL ``cuda/std`` tree.

    Newer ROCm SDKs (>= 7.x) vendor full libcu++. In that case the local
    ``rocm_compat`` stubs must NOT be placed ahead of the SDK includes, or they
    shadow the real ``<cuda/std/*>`` headers and break thrust (which pulls in
    ``cuda::std::is_arithmetic`` et al. via ``thrust/complex.h``).
    """
    sdk_include = osp.join(rocm_home, "include")
    return osp.isfile(
        osp.join(sdk_include, "cuda", "std", "__internal", "namespaces.h")
    )


def get_rocm_include_dirs(rocm_home):
    """Collect ROCm-related include directories (HIP, thrust, rocprim, hipcub)."""
    repo_root = osp.dirname(osp.abspath(__file__))
    sdk_include = osp.join(rocm_home, "include")
    include_dirs = [sdk_include]
    vendored_glm = osp.join(
        repo_root, "gsplat", "cuda", "csrc", "third_party", "glm", "glm", "glm.hpp"
    )
    if not osp.isfile(vendored_glm):
        include_dirs.append(osp.expanduser("~/.local/include"))

    extra_includes = os.environ.get("ROCM_EXTRA_INCLUDES", "")
    if extra_includes:
        include_dirs.extend(
            p.strip() for p in extra_includes.split(os.pathsep) if p.strip()
        )

    sdk_has_thrust = osp.isfile(osp.join(sdk_include, "thrust", "complex.h"))
    sdk_has_rocprim = osp.isfile(
        osp.join(sdk_include, "rocprim", "type_traits.hpp")
    )
    sdk_has_hipcub = osp.isfile(osp.join(sdk_include, "hipcub", "hipcub.hpp"))
    if sdk_has_thrust and sdk_has_rocprim and sdk_has_hipcub:
        # Prefer the pip SDK tree as a single consistent header set.
        seen = set()
        unique_dirs = []
        for d in include_dirs:
            if d not in seen and osp.isdir(d):
                seen.add(d)
                unique_dirs.append(d)
        return unique_dirs

    thrust_candidates = [
        os.environ.get("ROCM_THRUST_INCLUDE"),
        osp.join(repo_root, ".rocm_deps", "rocthrust"),
        osp.expanduser("~/.local/rocthrust"),
        sdk_include,
        "/opt/rocm/include",
    ]
    rocprim_candidates = [
        osp.join(repo_root, ".rocm_deps", "rocprim", "rocprim", "include"),
        sdk_include,
        "/opt/rocm/include",
    ]
    hipcub_candidates = [
        osp.join(repo_root, ".rocm_deps", "hipcub", "hipcub", "include"),
        sdk_include,
        "/opt/rocm/include",
    ]

    for path in thrust_candidates:
        if path and osp.isfile(osp.join(path, "thrust", "complex.h")):
            include_dirs.append(path)
            break

    for path in rocprim_candidates:
        if path and osp.isfile(osp.join(path, "rocprim", "type_traits.hpp")):
            include_dirs.append(path)
            break

    for path in hipcub_candidates:
        if path and osp.isfile(osp.join(path, "hipcub", "hipcub.hpp")):
            include_dirs.append(path)
            break

    # Deduplicate while preserving order.
    seen = set()
    unique_dirs = []
    for d in include_dirs:
        if d not in seen and osp.isdir(d):
            seen.add(d)
            unique_dirs.append(d)
    return unique_dirs


def get_rocm_archs():
    """
    Return target GPU architectures for HIP offload.

    Uses ROCM_ARCHS env (comma-separated), else rocminfo if available,
    else defaults to gfx942 and gfx950 for cross-compile without a GPU.
    """
    env_archs = os.environ.get("ROCM_ARCHS", "").strip()
    if env_archs:
        archs = [a.strip() for a in env_archs.split(",") if a.strip()]
        print(f"Using ROCM_ARCHS from environment: {archs}")
        return archs

    if ENABLE_TEST_COVERAGE:
        return ["gfx942"]

    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"Name:\s+(gfx[0-9a-z]+)", result.stdout)
        if not match:
            match = re.search(r"\b(gfx[0-9a-z]+)\b", result.stdout)
        if match:
            gfx_code = match.group(1)
            print(f"Detected ROCm GPU architecture from rocminfo: {gfx_code}")
            return [gfx_code]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    default_archs = ["gfx942", "gfx950"]
    print(
        "rocminfo unavailable or no GPU detected; "
        f"using default architectures: {default_archs}"
    )
    return default_archs


def _merge_include_dirs(*dir_lists):
    seen = set()
    merged = []
    for dir_list in dir_lists:
        for d in dir_list:
            if d not in seen and osp.isdir(d):
                seen.add(d)
                merged.append(d)
    return merged


def _sync_hip_glm_tree(repo_root):
    """Copy full GLM submodule into hipify tree (hipify omits .inl files)."""
    import shutil

    src = osp.join(repo_root, "gsplat", "cuda", "csrc", "third_party", "glm")
    dst = osp.join(repo_root, "gsplat", "hip", "csrc", "third_party", "glm")
    if not osp.isdir(src):
        return
    dst_parent = osp.dirname(dst)
    os.makedirs(dst_parent, exist_ok=True)
    if osp.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def _fix_hipified_glm_includes(repo_root):
    """Rewrite hipify's broken relative GLM includes to standard <glm/...> paths."""
    import re

    hip_root = osp.join(repo_root, "gsplat", "hip")
    if not osp.isdir(hip_root):
        return
    rel_glm = re.compile(
        r'#include\s+<(?:\.\./)+hip/csrc/third_party/glm/glm/([^>]+)>'
    )

    for root, _, files in os.walk(hip_root):
        for name in files:
            if not name.endswith((".h", ".cuh", ".cpp", ".hip", ".hpp")):
                continue
            path = osp.join(root, name)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            updated = rel_glm.sub(r"#include <glm/\1>", text)
            if updated != text:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(updated)


def _fix_hip_include_paths(repo_root):
    """Rewrite hipify's ../../hip/include/... includes to use -I gsplat/cuda/include."""
    import re

    pat = re.compile(r'#include\s+"((?:\.\./)+)hip/include/([^"]+)"')
    hip_csrc_pat = re.compile(r'#include\s+"((?:\.\./)+)hip/csrc/([^"]+)"')
    for tree in (
        osp.join(repo_root, "gsplat", "hip"),
        osp.join(repo_root, "gsplat", "cuda", "include"),
    ):
        if not osp.isdir(tree):
            continue
        for root, _, files in os.walk(tree):
            for name in files:
                if not name.endswith((".hip", ".h", ".cpp", ".cuh", ".hpp")):
                    continue
                path = osp.join(root, name)
                with open(path, encoding="utf-8") as f:
                    text = f.read()
                updated = pat.sub(r'#include "\2"', text)
                updated = hip_csrc_pat.sub(r'#include "\2"', updated)
                if updated != text:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(updated)


def _remove_hip_csrc_duplicate_cuhs(repo_root):
    """Drop hipify copies of cuda/csrc headers that shadow the canonical tree."""
    import shutil

    cuda_csrc = osp.join(repo_root, "gsplat", "cuda", "csrc")
    hip_csrc = osp.join(repo_root, "gsplat", "hip", "csrc")
    if not osp.isdir(cuda_csrc) or not osp.isdir(hip_csrc):
        return
    for name in os.listdir(cuda_csrc):
        if not name.endswith(".cuh"):
            continue
        hip_copy = osp.join(hip_csrc, name)
        if osp.isfile(hip_copy):
            os.remove(hip_copy)


def _post_hipify_fixup(repo_root):
    _sync_hip_glm_tree(repo_root)
    _fix_hipified_glm_includes(repo_root)
    _fix_hip_include_paths(repo_root)
    _remove_hip_csrc_duplicate_cuhs(repo_root)
    hip_include = osp.join(repo_root, "gsplat", "hip", "include")
    if osp.isdir(hip_include):
        import shutil

        shutil.rmtree(hip_include)


def _filter_rocm_host_flags(cflags):
    """Drop CUDA-only flags from build.py before passing them to hipcc host compiles."""
    skip = {
        "--forward-unknown-opts",
        "--expt-relaxed-constexpr",
        "-use_fast_math",
        "-lineinfo",
    }
    skip_prefixes = ("-diag-suppress", "-Xcompiler=")
    out = []
    for flag in cflags:
        if flag in skip or any(flag.startswith(p) for p in skip_prefixes):
            continue
        if flag.startswith("-DGSPLAT_NUM_CHANNELS="):
            out.append(flag.replace("\\,", ","))
            continue
        out.append(flag)
    return out


def _make_rocm_extension(sources, extra_include_paths, params):
    """Build gsplat.csrc with upstream sources and ROCm/HIP compile flags."""
    import torch
    from torch.utils.cpp_extension import CUDAExtension

    print("ROCM detected, compiling with HIP support...")
    rocm_home = get_rocm_sdk_root()
    rocm_bin = osp.join(rocm_home, "bin")
    rocm_llvm_bin = osp.join(rocm_home, "lib", "llvm", "bin")
    os.environ["ROCM_HOME"] = rocm_home
    os.environ["ROCM_PATH"] = rocm_home
    os.environ["PATH"] = os.pathsep.join(
        [rocm_bin, rocm_llvm_bin, os.environ.get("PATH", "")]
    )
    rocm_lib = osp.join(rocm_home, "lib")
    gpu_archs = get_rocm_archs()
    print(f"ROCm SDK root: {rocm_home}")
    print(f"GPU architectures: {gpu_archs}")

    undef_macros = []
    define_macros = []

    cxx_flags = [
        "-D__HIP_PLATFORM_AMD__",
        "-D__HIP__",
        "-DCUDA_VERSION=12000",
        "-Wno-sign-compare",
        "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE",
        "-DUSE_ROCM",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
        "-DAT_PARALLEL_OPENMP",
        "-fopenmp",
    ]
    if WITH_SYMBOLS:
        cxx_flags += ["-g", "-O0"]
    else:
        cxx_flags += ["-O3", "-Wno-attributes", "-Wno-switch", "-Wno-comment"]
    extra_compile_args = {"cxx": cxx_flags}

    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    hipcc_flags = [
        "-D__HIP_PLATFORM_AMD__",
        "-D__HIP_PLATFORM_HCC__=1",
        "-D__HIP__",
        "-DCUDA_VERSION=12000",
        "-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE",
        "-DUSE_ROCM",
        f"--rocm-path={rocm_home}",
    ]
    for arch in gpu_archs:
        hipcc_flags.append(f"--offload-arch={arch}")
    if WITH_SYMBOLS:
        hipcc_flags += ["-g", "-ggdb", "-O0"]
    else:
        hipcc_flags += ["-O3"]
        # hipcc warns when #pragma unroll cannot be honored (-Wpass-failed=transform-warning).
        hipcc_flags += ["-Wno-pass-failed"]
    if LINE_INFO:
        hipcc_flags += ["-gline-tables-only"]
    # ROCm's clang defaults to the newest installed gcc libstdc++. gcc-14's
    # <format> pulls in bits/unicode.h, which has a libstdc++ bug (`++this;`
    # instead of `++*this;`) that clang rejects but g++ tolerates. Pin clang to
    # a gcc toolchain whose headers compile — prefer 13, fall back to 12. Both
    # the device (hipcc) and host (.cpp via CXX=hipcc) compiles are clang-based,
    # so the flag must go on both flag sets.
    for _gcc_ver in ("13", "12"):
        _gcc_dir = f"/usr/lib/gcc/x86_64-linux-gnu/{_gcc_ver}"
        if osp.isdir(_gcc_dir):
            hipcc_flags.append(f"--gcc-install-dir={_gcc_dir}")
            cxx_flags.append(f"--gcc-install-dir={_gcc_dir}")
            break
    define_macros += [
        ("USE_ROCM", "1"),
        ("C10_CUDA_NO_CMAKE_CONFIGURE_FILE", "1"),
    ]
    undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
    if ENABLE_TEST_COVERAGE:
        extra_compile_args["cxx"] += [
            "-fprofile-instr-generate",
            "-fcoverage-mapping",
            "-Qunused-arguments",
            "--gcc-toolchain=/usr",
        ]
        hipcc_flags += ["-fprofile-instr-generate", "-fcoverage-mapping"]
        extra_link_args += ["-fprofile-instr-generate"]

    for flag in _filter_rocm_host_flags(params.extra_cflags):
        if flag not in cxx_flags:
            cxx_flags.append(flag)
        if flag.startswith("-D") and flag not in hipcc_flags:
            hipcc_flags.append(flag)

    extra_compile_args["nvcc"] = hipcc_flags

    current_dir = pathlib.Path(__file__).parent.resolve()
    project_include_dirs = [
        osp.join(current_dir, "gsplat", "cuda", "include"),
        osp.join(current_dir, "gsplat", "cuda", "csrc"),
    ]
    if not sdk_has_real_libcudacxx(rocm_home):
        # Older ROCm SDKs lack libcu++/CCCL; inject the local cuda/std stubs
        # ahead of the SDK. On SDKs that ship real libcu++ this would shadow the
        # real <cuda/std/*> headers and break thrust, so it is skipped.
        project_include_dirs.insert(
            0, osp.join(current_dir, "gsplat", "cuda", "include", "rocm_compat")
        )
    include_dirs = _merge_include_dirs(
        project_include_dirs,
        extra_include_paths,
        get_rocm_include_dirs(rocm_home),
    )
    print(f"ROCm include dirs: {include_dirs}")

    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:
        include_dirs.append(osp.join(conda_prefix, "include"))

    library_dirs = [rocm_lib]
    if conda_prefix:
        library_dirs.append(osp.join(conda_prefix, "lib"))

    extension = CUDAExtension(
        "gsplat.csrc",
        sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    _post_hipify_fixup(current_dir)
    return extension


def _patch_compiler_abi_check():
    """Tolerate ROCm clang version strings (e.g. ``17.0git``) in torch's ABI check."""
    import torch.utils.cpp_extension as ce

    def fixed_get_compiler_abi_compatibility_and_version(compiler):
        try:
            return ce.original_get_compiler_abi_compatibility_and_version(compiler)
        except ValueError:
            return ("gcc", (17, 0))

    if not hasattr(ce, "original_get_compiler_abi_compatibility_and_version"):
        ce.original_get_compiler_abi_compatibility_and_version = (
            ce.get_compiler_abi_compatibility_and_version
        )
        ce.get_compiler_abi_compatibility_and_version = (
            fixed_get_compiler_abi_compatibility_and_version
        )


def get_extensions():
    from torch.utils.cpp_extension import CUDAExtension

    # Use the same build parameters as the JIT build. However, directly
    # importing the gsplat.cuda.build module would trigger a circular
    # dependency where gsplat is imported before it is built. To avoid
    # this, we sidestep the traditional Python import mechanism and construct
    # the module directly from build.py.
    import importlib.util

    def _load_build_module(module_name, build_py_path):
        spec = importlib.util.spec_from_file_location(module_name, build_py_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    setup_dir = os.path.dirname(os.path.abspath(__file__))

    # --- gsplat main extension ---
    gsplat_build = _load_build_module(
        "gsplat_cuda_build", os.path.join("gsplat", "cuda", "build.py")
    )
    params = gsplat_build.get_build_parameters()
    sources = [os.path.relpath(s, setup_dir) for s in params.sources]

    if IS_ROCM:
        _patch_compiler_abi_check()
        return [_make_rocm_extension(sources, params.extra_include_paths, params)]

    gsplat_ext = CUDAExtension(
        "gsplat.csrc",
        sources=sources,
        include_dirs=params.extra_include_paths,
        extra_compile_args={
            "cxx": params.extra_cflags,
            "nvcc": params.extra_cuda_cflags,
        },
        extra_link_args=params.extra_ldflags,
    )

    if not BUILD_EXPERIMENTAL:
        return [gsplat_ext]

    # --- experimental Inference render extension ---
    inference_build = _load_build_module(
        "experimental_gaussian_render_inference_scene_build",
        os.path.join("gsplat", "experimental", "render", "kernels", "cuda", "build.py"),
    )
    inference_params = inference_build.get_build_parameters()
    inference_sources = [
        os.path.relpath(s, setup_dir) for s in inference_params.sources
    ]
    inference_ext = CUDAExtension(
        # The native extension's fully-qualified module name matches its
        # location under ``gsplat/experimental/render/kernels/``.
        "gsplat.experimental.render.kernels.csrc",
        sources=inference_sources,
        include_dirs=inference_params.extra_include_paths,
        extra_compile_args={
            "cxx": inference_params.extra_cflags,
            "nvcc": inference_params.extra_cuda_cflags,
        },
        extra_link_args=inference_params.extra_ldflags,
    )

    return [gsplat_ext, inference_ext]


def _setup():
    # Imported lazily so external tooling can ``import setup`` without
    # needing setuptools installed (e.g. docker/check_deps.sh runs against
    # a fresh venv before pip has populated it).
    from setuptools import find_packages, setup

    # --- Package discovery -------------------------------------------------
    packages = find_packages(exclude=["tests", "tests.*"])

    setup(
        name="gsplat",
        version=__version__,
        description=" Python package for differentiable rasterization of gaussians",
        keywords="gaussian, splatting, cuda",
        url=URL,
        download_url=f"{URL}/archive/gsplat-{__version__}.tar.gz",
        python_requires=">=3.7",
        install_requires=INSTALL_REQUIRES,
        extras_require=get_extras_require(),
        ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
        cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
        packages=packages,
        # Ship the CUDA / JIT sources for the sub-packages so wheels and
        # sdists can JIT-build (or be inspected). Paths are relative to each
        # package's source dir.
        # Globs cover every file under each package's cuda/ tree (all .cpp at the
        # cuda/ root, and the full csrc/ subtree) rather than enumerating
        # extensions, so a future source/header isn't silently dropped from
        # wheels (and matches MANIFEST.in's recursive-include breadth).
        package_data={
            "gsplat.geometry": [
                "kernels/cuda/*.cpp",
                "kernels/cuda/csrc/*",
            ],
            "gsplat.sensors": [
                "kernels/cuda/*.cpp",
                "kernels/cuda/csrc/*",
            ],
            "gsplat.scene": [
                "kernels/cuda/*.cpp",
                "kernels/cuda/csrc/*",
            ],
            "gsplat.experimental": [
                "render/kernels/cuda/*.cpp",
                "render/kernels/cuda/csrc/*",
                "render/kernels/cuda/csrc/gaussian_inference/*",
            ],
        },
        # We keep include_package_data=True so MANIFEST.in stays authoritative
        # for what lands in the sdist, while the explicit package_data above
        # guarantees the per-package CUDA sources are also copied into wheels.
        # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
        include_package_data=True,
    )


if __name__ == "__main__":
    _setup()

if need_to_unset_max_jobs:
    print("Unsetting MAX_JOBS")
    os.environ.pop("MAX_JOBS")
