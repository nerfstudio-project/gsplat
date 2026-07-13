# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python types for the spinning-LiDAR kernel layer.

Exposes the registered ``RowOffsetStructuredSpinningLidarProjection``
TorchScript custom class, the :class:`SpinningDirection` enum (verified against
the C++ source of truth at import time), and the shared ``script_class_name``
dispatch helper.
"""

from __future__ import annotations

from enum import IntEnum

import torch

from .. import _backend  # loads torch.classes.gsplat_sensors on backend access


class SpinningDirection(IntEnum):
    """Spinning-LiDAR azimuth sweep direction.

    Mirrors ``gsplat_sensors::SpinningDirection`` declared in
    ``gsplat/sensors/kernels/cuda/csrc/lidar_params.h``. The C++ definition is the
    source of truth; this class is verified against it at import time by
    :func:`_verify_spinning_direction_matches_cpp`.
    """

    CLOCKWISE = 0
    COUNTERCLOCKWISE = 1


def _verify_spinning_direction_matches_cpp() -> None:
    cpp_enum = getattr(_backend._SENSORS_CUDA, "SpinningDirection", None)
    if cpp_enum is None:
        raise RuntimeError(
            "Loaded gsplat_sensors_cuda extension does not expose SpinningDirection. "
            "Rebuild the native extension (see "
            "gsplat/sensors/kernels/cuda/csrc/ext.cpp)."
        )
    for member in SpinningDirection:
        cpp_member = getattr(cpp_enum, member.name, None)
        if cpp_member is None:
            raise RuntimeError(
                f"C++ SpinningDirection is missing member {member.name!r}; "
                "Python SpinningDirection in gsplat/sensors/kernels/lidars/types.py "
                "and C++ in gsplat/sensors/kernels/cuda/csrc/lidar_params.h are "
                "out of sync."
            )
        if int(cpp_member) != member.value:
            raise RuntimeError(
                f"SpinningDirection.{member.name} mismatch: "
                f"Python={member.value}, C++={int(cpp_member)}. "
                "Update one side to match the other (C++ source of truth: "
                "gsplat/sensors/kernels/cuda/csrc/lidar_params.h)."
            )


_verify_spinning_direction_matches_cpp()


# Bound to the TorchScript custom class registered by the gsplat_sensors C++
# extension via ``torch::class_<T>``. Behaves like a Python dataclass but is a
# ``torch.classes.*`` descriptor created at import time.
RowOffsetStructuredSpinningLidarProjection = (
    torch.classes.gsplat_sensors.RowOffsetStructuredSpinningLidarProjection
)
"""Structured spinning-LiDAR projection parameters.

Per-row elevation / per-column azimuth angle tables plus optional per-row
azimuth offsets and scalar FOV / spinning-direction fields. Parameters are
stored as separate per-component tensors on the C++
``RowOffsetStructuredSpinningLidarProjection`` struct.

Attributes:
    row_elevations_rad: (n_rows,) per-row elevation in radians.
    column_azimuths_rad: (n_columns,) per-column azimuth in radians.
    row_azimuth_offsets_rad: (n_rows,) per-row azimuth offset, or empty (0,)
        when ``has_row_offsets`` is False.
    fov_vert_start_rad / fov_vert_span_rad: vertical FOV start / span.
    fov_horiz_start_rad / fov_horiz_span_rad: horizontal FOV start / span.
    spinning_direction: :class:`SpinningDirection` integer value.
    has_row_offsets: whether per-row azimuth offsets are present.

Note:
    Use :func:`script_class_name` to retrieve the registered class name when
    dispatching across projection types.
"""

REGISTERED_LIDAR_PROJECTIONS = (RowOffsetStructuredSpinningLidarProjection,)
"""Tuple of all supported LiDAR projection classes."""

# Registered TorchScript class names, parallel to REGISTERED_LIDAR_PROJECTIONS.
# Kept as an explicit string tuple because the torch.classes.* class objects are
# torch.ScriptClass instances and do not expose a usable __name__ attribute;
# script_class_name() only works on instances, not on the registered class
# objects themselves.
REGISTERED_LIDAR_PROJECTION_NAMES = ("RowOffsetStructuredSpinningLidarProjection",)


def script_class_name(obj: object) -> str:
    """Return the registered TorchScript class name for a LiDAR parameter object.

    LiDAR projection classes are registered with the PyTorch custom-class
    registry via ``torch::class_<T>`` in the C++ extension. Their
    ``type(obj).__name__`` is always ``CustomClassHolder``, so a plain
    ``isinstance`` / ``type()`` check cannot distinguish them. This helper calls
    the ``._type()`` method injected by TorchScript to recover the original
    registered name.

    Falls back to ``type(obj).__name__`` for plain Python objects.

    Args:
        obj: A TorchScript ``CustomClassHolder`` or a plain Python object.

    Returns:
        The registered class name string.
    """
    type_fn = getattr(obj, "_type", None)
    if type_fn is None:
        return type(obj).__name__
    return type_fn().name()


__all__ = [
    "REGISTERED_LIDAR_PROJECTIONS",
    "REGISTERED_LIDAR_PROJECTION_NAMES",
    "RowOffsetStructuredSpinningLidarProjection",
    "SpinningDirection",
    "script_class_name",
]
