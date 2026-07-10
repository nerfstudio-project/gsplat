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

import importlib

import torch

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

    Returns the loaded extension module, or ``None`` if it is unavailable or
    does not register the expected operator.
    """
    global _BACKEND
    if _BACKEND is not _UNSET:
        return _BACKEND

    _C = None
    try:
        _C = importlib.import_module(
            "experimental_gaussian_render_inference_scene_cuda"
        )
    except ImportError as import_error:
        _warn(
            "experimental: CMake-built inference extension is unavailable "
            f"({import_error}). Inference render will be disabled."
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
