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
# The public helpers route through the `gsplat` NVTX domain when its per-domain
# API is complete; otherwise they fall back to module-level `nvtx.push_range` /
# `nvtx.pop_range` (no custom domain) or, if NVTX is unavailable or disabled,
# to no-ops.

import inspect
from contextlib import ContextDecorator
from typing import Any, Callable, ContextManager, TypeVar

try:
    import nvtx
except ModuleNotFoundError:
    nvtx = None

_F = TypeVar("_F", bound=Callable)
_ENABLED = nvtx is not None and nvtx.enabled()


def _get_valid_gsplat_nvtx_domain():
    """Return the per-domain object if its API is complete; else None.

    `nvtx.get_domain` was first re-exported at module level in nvtx 0.2.12
    (NVTX v3.2.0, May 2025); on older builds the call raises AttributeError.
    On May-2025+ builds the per-domain `push_range` / `pop_range` /
    `get_event_attributes` methods can still be missing on specific releases
    (notably 0.2.13), which is why we additionally probe the surface before
    committing to the Domain path.
    """
    if not _ENABLED:
        return None
    if not hasattr(nvtx, "get_domain"):
        return None  # nvtx < 0.2.12 — no module-level Domain API
    domain = nvtx.get_domain("gsplat")
    if not all(
        callable(getattr(domain, _m, None))
        for _m in ("push_range", "pop_range", "get_event_attributes")
    ):
        return None  # Domain object present but missing the per-domain API
    return domain


_DOMAIN = _get_valid_gsplat_nvtx_domain()

# Kwargs supported by the loaded `nvtx.push_range`. Computed once at import
# time so the module-level fallback can forward only kwargs the loaded nvtx
# actually accepts. nvtx 0.2.10+ has the full surface
# (message, color, domain, category, payload); 0.2.8 lacks `payload`.
_PUSH_RANGE_PARAMS = (
    frozenset(inspect.signature(nvtx.push_range).parameters)
    if _ENABLED and hasattr(nvtx, "push_range")
    else frozenset()
)


def _filter_push_range_kwargs(kwargs):
    return {k: v for k, v in kwargs.items() if k in _PUSH_RANGE_PARAMS}


if not _ENABLED:

    class _Trace(ContextDecorator):
        def __init__(self, name: str, **kwargs: Any):
            pass

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    def trace_push(name: str, **kwargs: Any) -> None:
        return None

    def trace_pop() -> None:
        return None

elif _DOMAIN is not None:

    class _Trace(ContextDecorator):
        def __init__(self, name: str, **kwargs: Any):
            self._name = name
            self._kwargs = kwargs

        def __enter__(self):
            # Older nvtx builds (i.e. 0.2.13) can expose a broken
            # Domain.push_range(kwargs) path even though the documented API
            # accepts `message=` and other keyword attributes. Build the
            # EventAttributes explicitly and use the low-level fast path.
            attrs = _DOMAIN.get_event_attributes(message=self._name, **self._kwargs)
            _DOMAIN.push_range(attrs)
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            _DOMAIN.pop_range()
            return False

    def trace_push(name: str, **kwargs: Any) -> None:
        # Avoid the kwargs-based Domain.push_range() path for compatibility
        # with older nvtx builds (i.e. 0.2.13) whose compiled bindings only
        # accept a positional EventAttributes object.
        attrs = _DOMAIN.get_event_attributes(message=name, **kwargs)
        _DOMAIN.push_range(attrs)

    trace_pop = _DOMAIN.pop_range

else:

    # Module-level fallback: nvtx is loaded and enabled but the gsplat domain
    # is missing the per-domain API. Use module-level push_range / pop_range
    # (annotations still appear in nsys, just not under a custom domain).
    # Kwargs are filtered to what the loaded nvtx.push_range actually accepts
    # so unsupported kwargs degrade gracefully instead of raising TypeError.
    class _Trace(ContextDecorator):
        def __init__(self, name: str, **kwargs: Any):
            self._name = name
            self._kwargs = _filter_push_range_kwargs(kwargs)

        def __enter__(self):
            nvtx.push_range(self._name, **self._kwargs)
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            nvtx.pop_range()
            return False

    def trace_push(name: str, **kwargs: Any) -> None:
        nvtx.push_range(name, **_filter_push_range_kwargs(kwargs))

    trace_pop = nvtx.pop_range


# `_Trace` differs per branch but the context-manager and decorator surface
# is uniform — define `trace_range` / `trace_function` once at module-top.
def trace_range(name: str, **kwargs: Any) -> ContextManager[None]:
    return _Trace(name, **kwargs)


def trace_function(name: str, **kwargs: Any) -> Callable[[_F], _F]:
    return _Trace(name, **kwargs)
