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

"""Shared lazy loader for gsplat's CMake-built native extensions.

Importing a Python-only gsplat API must not eagerly load CUDA libraries. Each
native backend therefore resolves its extension on first access through a PEP
562 module ``__getattr__`` hook. Successful imports and failures are cached so
subsequent dispatches avoid repeated module resolution.
"""

from __future__ import annotations

import importlib
import sys

_UNSET = object()


def make_lazy_backend(
    *,
    module_name: str,
    public_name: str,
    extension_module: str,
):
    """Create a lazy accessor for a CMake-built native extension.

    Args:
        module_name: ``__name__`` of the calling ``_backend`` module (for error
            messages).
        public_name: Sentinel attribute that triggers loading on first access
            (e.g. ``"_GEOMETRY_CUDA"``).
        extension_module: Importable name of the native extension (e.g.
            ``"gsplat_geometry_cuda"``).

    Returns:
        ``(get_backend, module_getattr)``. Bind these as ``_get_backend`` and
        ``__getattr__`` in the calling module.
    """
    cache: dict = {"result": _UNSET}

    def get_backend():
        """Load and cache the native extension on first use."""
        result = cache["result"]
        if result is not _UNSET:
            if isinstance(result, BaseException):
                raise result
            return result

        try:
            backend = importlib.import_module(extension_module)
        except ImportError as import_error:
            error = ImportError(
                f"Failed to import CMake-built extension {extension_module!r}. "
                "Build or install gsplat before using native operations."
            )
            cache["result"] = error
            raise error from import_error

        cache["result"] = backend
        # Binding only after resolution preserves laziness while making later
        # accesses hit the module dictionary directly.
        setattr(sys.modules[module_name], public_name, backend)
        return backend

    def module_getattr(name: str):
        """Lazily resolve ``public_name`` on first attribute access (PEP 562)."""
        if name == public_name:
            return get_backend()
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return get_backend, module_getattr
