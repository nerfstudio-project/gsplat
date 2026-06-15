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

"""Shared lazy native-extension loader for gsplat subpackages.

Several gsplat subpackages (``gsplat.geometry``, ``gsplat.scene``,
``gsplat.sensors``) load a native CUDA extension with the same policy: import a
prebuilt extension if present, otherwise JIT-build it, otherwise raise. Loading
must be *lazy* — importing the package (and its ``functional`` API) on a
CPU-only machine must not build or import the extension — so the extension is
resolved only on first access to a sentinel attribute via PEP 562 module
``__getattr__``.

:func:`make_lazy_backend` factors that boilerplate out. A kernels ``_backend.py``
wires it in as::

    from gsplat._lazy_backend import make_lazy_backend

    def _build():  # deferred so importing _backend doesn't import .cuda.build
        from .cuda.build import build_and_load_geometry_cuda
        return build_and_load_geometry_cuda()

    _get_backend, __getattr__ = make_lazy_backend(
        module_name=__name__,
        public_name="_GEOMETRY_CUDA",
        prebuilt_module="gsplat_geometry_cuda",
        jit_loader=_build,
    )
    __all__ = ["_GEOMETRY_CUDA"]

The returned ``__getattr__`` resolves ``public_name`` on first access (and works
for ``from ..._backend import <public_name>``); any other attribute raises
``AttributeError``. Note ``public_name`` must NOT also be assigned as a module
global, or it would shadow ``__getattr__`` and defeat lazy loading.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Callable

_UNSET = object()


def cuda_toolkit_available() -> bool:
    """Return True if a usable CUDA toolkit (nvcc) is discoverable.

    Shared by the native-extension loaders so the probe lives in one place.
    ``torch.utils.cpp_extension`` is imported lazily so importing this module
    (and therefore the lazy ``_backend`` modules) stays cheap.
    """
    from subprocess import DEVNULL, call

    import torch.utils.cpp_extension as jit

    cuda_home = jit._find_cuda_home()  # tries various heuristics
    if not cuda_home:
        return False
    nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc_path):
        # Maybe still on PATH; try invoking nvcc directly.
        try:
            call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
            return True
        except FileNotFoundError:
            return False
    return True


def make_lazy_backend(
    *,
    module_name: str,
    public_name: str,
    prebuilt_module: str,
    jit_loader: Callable[[], object],
    force_jit_env: str | None = None,
):
    """Build the lazy loader + module ``__getattr__`` for a native extension.

    Args:
        module_name: ``__name__`` of the calling ``_backend`` module (for error
            messages).
        public_name: Sentinel attribute that triggers loading on first access
            (e.g. ``"_GEOMETRY_CUDA"``).
        prebuilt_module: Importable name of the prebuilt extension (e.g.
            ``"gsplat_geometry_cuda"``).
        jit_loader: Zero-arg callable that JIT-builds and returns the extension
            when the prebuilt import is unavailable. Keep its ``.cuda.build``
            import inside the callable so importing ``_backend`` stays cheap.
        force_jit_env: Optional env var name; when set to ``"1"`` the prebuilt
            import is skipped and ``jit_loader`` is used directly (handy when
            iterating on the native sources locally).

    Returns:
        ``(get_backend, module_getattr)`` — bind these as ``_get_backend`` and
        ``__getattr__`` in the calling module.
    """
    # Holds the loaded extension after success, or the ImportError after a
    # failed attempt, so a persistent build failure fails fast on retry instead
    # of re-running the (multi-second) JIT build on every access. ``_UNSET``
    # means "not yet attempted".
    cache: dict = {"result": _UNSET}

    def get_backend():
        """Load and cache the native extension on first use."""
        result = cache["result"]
        if result is not _UNSET:
            if isinstance(result, BaseException):
                raise result
            return result

        forced_jit = bool(force_jit_env and os.getenv(force_jit_env, "0") == "1")
        backend = None
        prebuilt_error = None
        if not forced_jit:
            try:
                backend = importlib.import_module(prebuilt_module)
            except ImportError as error:
                prebuilt_error = error

        if backend is None:
            try:
                backend = jit_loader()
            except Exception as jit_error:
                prebuilt_note = (
                    f"prebuilt import skipped ({force_jit_env}=1)"
                    if forced_jit
                    else f"prebuilt import error: {prebuilt_error!r}"
                )
                error = ImportError(
                    f"Failed to load {prebuilt_module} via JIT build/load"
                    f"{'' if forced_jit else ' (and prebuilt import)'}.\n"
                    f"{prebuilt_note}\n"
                    f"JIT build/load error: {jit_error!r}"
                )
                cache["result"] = error  # fail fast on subsequent access
                raise error from jit_error

        cache["result"] = backend
        # Bind as a real module attribute so subsequent accesses hit the module
        # __dict__ directly and skip this __getattr__ + get_backend indirection
        # on every op dispatch. Binding only AFTER first resolution preserves
        # laziness (a binding at import would shadow __getattr__).
        setattr(sys.modules[module_name], public_name, backend)
        return backend

    def module_getattr(name: str):
        """Lazily resolve ``public_name`` on first attribute access (PEP 562)."""
        if name == public_name:
            return get_backend()
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return get_backend, module_getattr
