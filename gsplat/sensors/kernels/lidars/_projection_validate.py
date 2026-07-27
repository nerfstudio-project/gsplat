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

"""Opt-in value-level validation for spinning-LiDAR projection components.

The C++ constructor validates cheap shape, dtype, and scalar constraints without
crossing the host/device boundary. The checks here are stricter value-level
invariants that require reading tensor entries, so callers opt in explicitly via
``validate_lidar_projection``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from .types import (
    REGISTERED_LIDAR_PROJECTION_NAMES,
    RowOffsetStructuredSpinningLidarProjection,
    script_class_name,
)


def _check_finite(t: Tensor, name: str) -> None:
    mask = ~torch.isfinite(t)
    if mask.any():
        idx = int(mask.flatten().nonzero()[0].item())
        value = t.detach().flatten()[idx].item()
        raise ValueError(f"{name}[{idx}] must be finite; got {value!r}")


def _validate_spinning_lidar_projection(
    projection: RowOffsetStructuredSpinningLidarProjection,
) -> None:
    """Raise ``ValueError`` if any angle-table entry is non-finite.

    Per-row elevations and per-column azimuths (plus per-row azimuth offsets when
    present) must be finite so the sensor-frame angle math and the inverse
    nearest-row/column lookup are well-defined. Shape, dtype, and scalar-range
    checks are enforced by the C++ constructor; this adds the per-entry
    finiteness check that needs a host sync.
    """
    _check_finite(projection.row_elevations_rad, "row_elevations_rad")
    _check_finite(projection.column_azimuths_rad, "column_azimuths_rad")
    if projection.has_row_offsets:
        _check_finite(projection.row_azimuth_offsets_rad, "row_azimuth_offsets_rad")


_ProjectionValidator = Callable[[RowOffsetStructuredSpinningLidarProjection], None]

_PROJECTION_VALIDATORS: dict[str, _ProjectionValidator | None] = {
    name: _validate_spinning_lidar_projection
    for name in REGISTERED_LIDAR_PROJECTION_NAMES
}


def validate_lidar_projection(
    projection: RowOffsetStructuredSpinningLidarProjection,
) -> None:
    """Run opt-in value-level validation for a supported LiDAR projection."""
    class_name = script_class_name(projection)
    try:
        validator = _PROJECTION_VALIDATORS[class_name]
    except KeyError as exc:
        raise TypeError(f"Unknown LiDAR projection class: {class_name}") from exc
    if validator is not None:
        validator(projection)


__all__ = ["validate_lidar_projection"]
