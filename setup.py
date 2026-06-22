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
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest",
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
