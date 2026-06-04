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

from setuptools import find_packages, setup

__version__ = None
exec(open("gsplat/version.py", "r").read())

URL = "https://github.com/nerfstudio-project/gsplat"

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"


def _patch_hipify_ignore_glm():
    """Keep the bundled third_party/glm out of torch's hipify on ROCm.

    torch's hipify (via CUDAExtension) walks every .hpp under the build dir and
    the extension include dirs into its file set, then content-rewrites any GLM
    header a source pulls in -- which drops GLM's .inl files (hipify only copies
    .hpp/.h) and mangles GLM's __CUDACC__/__HIP__ compiler detection, breaking
    the build. GLM 1.0.2 already detects __HIP__ and compiles verbatim under the
    -x hip pass, so the fix is simply to leave it untouched: add the glm dir to
    hipify's ``ignores`` and drop it from ``header_include_dirs``. The source
    keeps including <glm/...> via -I, resolved against the pristine bundled tree.
    """
    import torch

    if not torch.version.hip:
        return
    from torch.utils.hipify import hipify_python

    glm_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "gsplat", "cuda", "csrc", "third_party", "glm",
    )
    glm_patterns = [os.path.join(glm_dir, "*"), glm_dir + "*"]
    orig_hipify = hipify_python.hipify

    def hipify_no_glm(*args, **kwargs):
        kwargs["ignores"] = list(kwargs.get("ignores", ())) + glm_patterns
        kwargs["header_include_dirs"] = [
            d for d in kwargs.get("header_include_dirs", [])
            if os.path.abspath(d) != os.path.abspath(glm_dir)
        ]
        return orig_hipify(*args, **kwargs)

    hipify_python.hipify = hipify_no_glm


_patch_hipify_ignore_glm()


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

    import torch

    # The experimental inference renderer (experimental/render) is a separate,
    # NVIDIA-tuned extension (inline PTX asm, half2 intrinsics like __hmax2 /
    # __hlt2_mask, cub/block internals, 32-bit warp masks) that is not part of the
    # core differentiable rasterizer and is not imported by the gsplat API. It is
    # not yet ported to HIP, so skip it on ROCm and build only gsplat.csrc.
    if torch.version.hip:
        return [gsplat_ext]

    # --- experimental Inference render extension ---
    inference_build = _load_build_module(
        "experimental_gaussian_render_inference_scene_build",
        os.path.join("experimental", "render", "kernels", "cuda", "build.py"),
    )
    inference_params = inference_build.get_build_parameters()
    inference_sources = [
        os.path.relpath(s, setup_dir) for s in inference_params.sources
    ]
    inference_ext = CUDAExtension(
        "experimental.render.kernels.csrc",
        sources=inference_sources,
        include_dirs=inference_params.extra_include_paths,
        extra_compile_args={
            "cxx": inference_params.extra_cflags,
            "nvcc": inference_params.extra_cuda_cflags,
        },
        extra_link_args=inference_params.extra_ldflags,
    )

    return [gsplat_ext, inference_ext]


setup(
    name="gsplat",
    version=__version__,
    description=" Python package for differentiable rasterization of gaussians",
    keywords="gaussian, splatting, cuda",
    url=URL,
    download_url=f"{URL}/archive/gsplat-{__version__}.tar.gz",
    python_requires=">=3.7",
    install_requires=[
        "ninja",
        "numpy",
        "jaxtyping",
        "rich>=12",
        "torch",
        "typing_extensions; python_version<'3.8'",
        # gsplat-scene / gsplat-stage live under libs/scene and libs/stage and
        # are installed via ``libs/install.sh scene && libs/install.sh stage``;
        # they are not published, so listing them here would break ``pip install``.
    ],
    extras_require={
        # lidar dependencies. Install them by `pip install gsplat[lidar]`
        "lidar": [
            "scipy",
        ],
        # dev dependencies. Install them by `pip install gsplat[dev]`
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.3",
            "pytest-env==0.8.1",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml>=6.0.1",
            "build",
            "twine",
            "cupy",
            "nerfacc>=0.5.3",
            "PLAS @ git+https://github.com/fraunhoferhhi/PLAS.git",
            "imageio>=2.37.2",
            "torchpq>=0.3.0.6",
        ],
    },
    ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
    cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
    packages=find_packages(),
    # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
    include_package_data=True,
)
