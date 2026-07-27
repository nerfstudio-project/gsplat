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
