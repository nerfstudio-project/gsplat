# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utility functions for camera and LiDAR models."""

from __future__ import annotations

from torch import Tensor

from ...kernels.common.utils import (
    poses_to_matrix,
    valid_flags_to_indices,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def compute_scaled_resolution(
    original_resolution: tuple[int, int],
    scale: float | tuple[float, float],
    new_resolution: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Compute new resolution after scaling, with optional explicit override.

    Args:
        original_resolution: (width, height) in pixels.
        scale: Isotropic (float) or anisotropic (sx, sy) scaling factor.
        new_resolution: If provided, returned as-is (overrides scale computation).

    Returns:
        New (width, height) tuple.
    """
    if new_resolution is not None:
        return (int(new_resolution[0]), int(new_resolution[1]))
    if isinstance(scale, tuple):
        sx, sy = scale
    else:
        sx = sy = scale
    return (
        int(round(original_resolution[0] * sx)),
        int(round(original_resolution[1] * sy)),
    )


def filter_by_validity(
    data: Tensor | None, valid_flags: Tensor | None, return_all: bool
) -> Tensor | None:
    """Filter tensor rows by a boolean validity mask.

    Args:
        data: (N, ...) tensor to filter, or None.
        valid_flags: (N,) boolean mask, or None.
        return_all: If True, skip filtering and return ``data`` unchanged.

    Returns:
        Filtered tensor, or ``data`` unchanged when filtering is skipped.
    """
    if data is None or valid_flags is None or return_all:
        return data
    return data[valid_flags]


__all__ = [
    "compute_scaled_resolution",
    "filter_by_validity",
    "poses_to_matrix",
    "valid_flags_to_indices",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]
