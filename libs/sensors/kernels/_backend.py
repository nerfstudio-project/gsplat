# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load native sensor registrations, preferring prebuilt then JIT build.

Environment variables:

- ``GSPLAT_SENSORS_FORCE_JIT``: When set to ``"1"``, skip the prebuilt
  ``gsplat_sensors_cuda`` wheel import and fall straight through to the JIT
  build path in :mod:`.cuda.build`. Useful when iterating on the native
  sources locally and you want every interpreter restart to pick up your
  edits without uninstalling/reinstalling the wheel. Any other value (or
  unset) keeps the default fast path of importing the prebuilt extension.
"""

from __future__ import annotations

import os

_C = None

prebuilt_error = None
if os.getenv("GSPLAT_SENSORS_FORCE_JIT", "0") != "1":
    try:
        import gsplat_sensors_cuda as _C  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        prebuilt_error = error

if _C is None:
    from .cuda.build import build_and_load_sensors_cuda

    try:
        _C = build_and_load_sensors_cuda()
    except Exception as jit_error:
        raise ImportError(
            "Failed to load gsplat_sensors_cuda via both prebuilt import and "
            "JIT build/load.\n"
            f"Prebuilt import error: {prebuilt_error!r}\n"
            f"JIT build/load error: {jit_error!r}"
        ) from jit_error


__all__ = ["_C"]
