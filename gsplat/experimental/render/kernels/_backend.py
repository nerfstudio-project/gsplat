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

import torch

from gsplat._lazy_backend import cuda_toolkit_available

try:
    from rich.console import Console

    _console = Console()
except ImportError:
    _console = None


# Cache for the loaded native extension. Sentinel ``object()`` distinguishes
# "not yet attempted" from "attempted and unavailable" (``None``). Intentionally
# NOT named ``_C``: a module-level ``_C`` binding would shadow the PEP 562
# ``__getattr__`` hook below and defeat lazy loading.
_UNSET = object()
_BACKEND = _UNSET


def _warn(message):
    """Print an inference backend warning through rich when it is available."""
    if _console is not None:
        _console.print(f"[yellow]{message}[/yellow]")
    else:
        print(message)


def _inference_op_registered():
    """Return whether loading the extension registered the Inference torch op."""
    return hasattr(torch.ops.experimental, "gaussian_render_inference_only")


def _get_backend():
    """Load and cache the native Inference CUDA extension on first use.

    Returns the loaded extension module, or ``None`` if it is unavailable
    (no CUDA toolkit, build failure, or the op did not register).
    """
    global _BACKEND
    if _BACKEND is not _UNSET:
        return _BACKEND

    _C = None
    try:
        # Try to import the compiled module first, matching gsplat.cuda._backend.
        # The module intentionally lives outside cuda.csrc so in-tree source
        # checkouts fall through to JIT instead of importing the source
        # directory as a namespace package. The relative import resolves the
        # ``csrc`` sibling regardless of the mapped package name.
        from . import csrc as _C
    except ImportError:
        # Fall back to JIT compilation. Import the builder lazily so importing
        # this module does not pull in .cuda.build / torch.utils.cpp_extension.
        if cuda_toolkit_available():
            try:
                from .cuda.build import (
                    build_and_load_experimental_gaussian_render_inference_scene,
                )

                _C = build_and_load_experimental_gaussian_render_inference_scene()
            except Exception as _build_err:
                _warn(
                    "experimental: Inference JIT build failed "
                    f"({_build_err}). Inference render will be disabled."
                )
        else:
            _warn(
                "experimental: No CUDA toolkit found. Inference render will be disabled."
            )

    if _C is not None and not _inference_op_registered():
        _warn(
            "experimental: Inference CUDA extension loaded but did not register "
            "torch.ops.experimental.gaussian_render_inference_only. "
            "Inference render will be disabled."
        )
        _C = None

    _BACKEND = _C
    return _BACKEND


def __getattr__(name):
    """Lazily resolve ``_C`` on first attribute access (PEP 562)."""
    if name == "_C":
        return _get_backend()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["_C"]
