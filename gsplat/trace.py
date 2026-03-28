# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Lightweight NVTX helpers for gsplat.
#
# The public API is intentionally small:
# - `trace_push(name)` / `trace_pop()` for large parent scopes
# - `trace_range(name)` for short inline scopes
# - `trace_function(name)` to annotate an existing function boundary
#
# Range names are emitted exactly as provided by callers.
#
# Push/pop example:
#
#   trace_push("proj")
#   ...
#   trace_pop()  # proj
#
# Range example:
#
#   with trace_range("output"):
#       ...
#
# Decorator example:
#
#   @trace_function("isect-camera")
#   def isect_tiles(...):
#       ...
#
# Nested example:
#
#   trace_push("proj")
#   try:
#       with trace_range("output"):
#           ...
#   finally:
#       trace_pop()  # proj
#
# This produces sibling ranges named `proj` and `output`.
#
# The public helpers are reduced to no-ops when NVTX tracing is not enabled for
# the `gsplat` domain.

from contextlib import nullcontext
from contextlib import ContextDecorator
from typing import Any, Callable, ContextManager, TypeVar

try:
    import nvtx
except ModuleNotFoundError:
    nvtx = None

_F = TypeVar("_F", bound=Callable)
_ENABLED = nvtx is not None and nvtx.enabled()
_DOMAIN = None if not _ENABLED else nvtx.get_domain("gsplat")

if not _ENABLED:

    def trace_range(name: str, **kwargs: Any) -> ContextManager[None]:
        return nullcontext()

    def trace_push(name: str, **kwargs: Any) -> None:
        return None

    def trace_pop() -> None:
        return None

    def trace_function(name: str, **kwargs: Any) -> Callable[[_F], _F]:
        def decorator(fn: _F) -> _F:
            return fn

        return decorator

else:

    class _Trace(ContextDecorator):
        def __init__(self, name: str, **kwargs: Any):
            self._name = name
            self._kwargs = kwargs

        def __enter__(self):
            _DOMAIN.push_range(message=self._name, **self._kwargs)
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            _DOMAIN.pop_range()
            return False

    def trace_range(name: str, **kwargs: Any) -> ContextManager[None]:
        return _Trace(name, **kwargs)

    def trace_function(name: str, **kwargs: Any) -> Callable[[_F], _F]:
        return _Trace(name, **kwargs)

    def trace_push(name: str, **kwargs: Any) -> None:
        _DOMAIN.push_range(message=name, **kwargs)

    trace_pop = _DOMAIN.pop_range
