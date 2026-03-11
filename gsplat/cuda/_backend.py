# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
Trigger compiling (for debugging):

VERBOSE=1 DEBUG=1 TORCH_CUDA_ARCH_LIST="8.9" python -c "from gsplat.cuda._backend import _C"
"""

import os
from subprocess import DEVNULL, call
import torch.utils.cpp_extension as jit
from .build import build_and_load_gsplat
from rich.console import Console


def cuda_toolkit_available():
    """
    Check more robustly if the CUDA toolkit is available.
    1. Attempt to locate `CUDA_HOME` using PyTorch’s internal method.
    2. Check if nvcc is present in that location.
    """
    cuda_home = jit._find_cuda_home()  # This tries various heuristics
    if not cuda_home:
        return False

    # If we have a cuda_home, check if nvcc exists there:
    nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc_path):
        # Maybe still on PATH, try calling "nvcc" directly:
        try:
            call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
            return True
        except FileNotFoundError:
            return False
    return True


_C = None

try:
    # Try to import the compiled module (via setup.py or pre-built .so)
    from gsplat import csrc as _C
except ImportError:
    # if that fails, try with JIT compilation
    if cuda_toolkit_available():
        _C = build_and_load_gsplat()
    else:
        Console().print(
            "[yellow]gsplat: No CUDA toolkit found. gsplat will be disabled.[/yellow]"
        )

__all__ = ["_C"]
