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
from subprocess import DEVNULL, call
import torch
import torch.utils.cpp_extension as jit
from .cuda.build import build_and_load_experimental_gaussian_render_inference_scene

try:
    from rich.console import Console

    _console = Console()
except ImportError:
    _console = None


def cuda_toolkit_available():
    """
    Check more robustly if the CUDA toolkit is available.
    1. Attempt to locate ``CUDA_HOME`` using PyTorch's internal method.
    2. Check if nvcc is present in that location.
    """
    cuda_home = jit._find_cuda_home()
    if not cuda_home:
        return False

    nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc_path):
        try:
            call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
            return True
        except FileNotFoundError:
            return False
    return True


_C = None


def _warn(message):
    """Print an inference backend warning through rich when it is available."""
    if _console is not None:
        _console.print(f"[yellow]{message}[/yellow]")
    else:
        print(message)


def _inference_op_registered():
    """Return whether loading the extension registered the Inference torch op."""
    return hasattr(torch.ops.experimental, "gaussian_render_inference_only")


try:
    # Try to import the compiled module first, matching gsplat.cuda._backend.
    # The module intentionally lives outside cuda.csrc so in-tree source
    # checkouts fall through to JIT instead of importing the source directory as
    # a namespace package.
    from experimental.render.kernels import csrc as _C
except ImportError:
    # Fall back to JIT compilation.
    if cuda_toolkit_available():
        try:
            _C = build_and_load_experimental_gaussian_render_inference_scene()
        except Exception as _build_err:
            _warn(
                "experimental: Inference JIT build failed "
                f"({_build_err}). Inference render will be disabled."
            )
    else:
        _warn("experimental: No CUDA toolkit found. Inference render will be disabled.")

if _C is not None and not _inference_op_registered():
    _warn(
        "experimental: Inference CUDA extension loaded but did not register "
        "torch.ops.experimental.gaussian_render_inference_only. "
        "Inference render will be disabled."
    )
    _C = None

__all__ = ["_C"]
