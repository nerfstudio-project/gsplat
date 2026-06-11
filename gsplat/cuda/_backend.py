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

"""
Trigger compiling (for debugging):

VERBOSE=1 DEBUG=1 TORCH_CUDA_ARCH_LIST="8.9" python -c "from gsplat.cuda._backend import _C"
"""

from gsplat._lazy_backend import cuda_toolkit_available
from .build import build_and_load_gsplat
from rich.console import Console

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
