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

"""Build-only setuptools glue for the torch CUDA extensions.

All project metadata lives in ``pyproject.toml``. This file exists because
the torch ``cpp_extension`` build (``ext_modules`` + ``cmdclass``) cannot be
expressed there.
"""

import os

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
BUILD_EXPERIMENTAL = os.getenv("BUILD_EXPERIMENTAL", "1") == "1"


def _load_build_module(module_name, build_py_path):
    """Load build-only helpers without importing the unbuilt gsplat package."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, build_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_gsplat_build_parameters():
    """Return the main extension parameters used by metadata and compilation."""

    gsplat_build = _load_build_module(
        "gsplat_cuda_build", os.path.join("gsplat", "cuda", "build.py")
    )
    return gsplat_build.get_build_parameters()


def _egg_info_class():
    """Create an egg_info command that records build-compatible CuPy metadata."""

    from setuptools.command.egg_info import egg_info

    class GSplatEggInfo(egg_info):
        def run(self):
            """Extend the static PNG extra with the build-matched CuPy pin.

            egg_info is the command that actually writes the metadata
            (PKG-INFO / *.egg-info, which sdist and wheel builds both read
            from), so patching self.distribution here is a documented
            command-extension point rather than a Distribution-internals
            override.
            """

            if not BUILD_NO_CUDA:
                # Metadata preparation runs before compilation. Reuse the
                # exact build-parameter path so the selected CuPy major is
                # the compiler major already checked against Torch, rather
                # than the driver's maximum supported CUDA version.
                cuda_major = _get_gsplat_build_parameters().cuda_major
                cupy_requirement = f"cupy-cuda{cuda_major}x"
                png_requirements = list(self.distribution.extras_require["png"])
                if cupy_requirement not in png_requirements:
                    png_requirements.append(cupy_requirement)

                self.distribution.extras_require = {
                    **self.distribution.extras_require,
                    "png": png_requirements,
                }
                self.distribution.metadata.extras_require = (
                    self.distribution.extras_require
                )

            super().run()

    return GSplatEggInfo


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_extensions():
    from torch.utils.cpp_extension import CUDAExtension

    setup_dir = os.path.dirname(os.path.abspath(__file__))

    # --- gsplat main extension ---
    params = _get_gsplat_build_parameters()
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
    from setuptools import setup

    cmdclass: dict = {"egg_info": _egg_info_class()}
    if not BUILD_NO_CUDA:
        cmdclass["build_ext"] = get_ext()

    setup(
        ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
        cmdclass=cmdclass,
    )


if __name__ == "__main__":
    _setup()
