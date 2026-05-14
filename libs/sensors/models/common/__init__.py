# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities for sensor models.

This module provides the Frame base class, the FrameId type alias, and utility
functions shared by camera models.
"""

from .frame import Frame, FrameId
from .utils import (
    compute_scaled_resolution,
    filter_by_validity,
    poses_to_matrix,
    valid_flags_to_indices,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

__all__ = [
    "Frame",
    "FrameId",
    "compute_scaled_resolution",
    "filter_by_validity",
    "poses_to_matrix",
    "valid_flags_to_indices",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]
