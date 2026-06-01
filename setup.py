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

    spec = importlib.util.spec_from_file_location(
        "gsplat_cuda_build", os.path.join("gsplat", "cuda", "build.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    params = module.get_build_parameters()

    setup_dir = os.path.dirname(os.path.abspath(__file__))
    sources = [os.path.relpath(s, setup_dir) for s in params.sources]

    extension = CUDAExtension(
        "gsplat.csrc",
        sources=sources,
        include_dirs=params.extra_include_paths,
        extra_compile_args={
            "cxx": params.extra_cflags,
            "nvcc": params.extra_cuda_cflags,
        },
        extra_link_args=params.extra_ldflags,
    )
    return [extension]


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
        "nvtx",
        "rich>=12",
        # Upper bound: torch 2.12 dropped support for older CUDA drivers,
        # so a fresh install on a host with driver < the new minimum will
        # pull a wheel that fails at import time. Pin to the last known-good
        # major series.
        "torch>=2.0,<2.12",
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
