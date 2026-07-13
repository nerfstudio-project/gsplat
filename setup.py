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
    from setuptools import setup

    setup(
        ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
        cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
    )


if __name__ == "__main__":
    _setup()
