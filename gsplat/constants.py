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
"""Extension-free numeric constants mirrored from the CUDA headers.

Python mirrors of C++ ``constexpr`` values that must stay importable without
compiling any CUDA extension (pure-PyTorch fallbacks run CPU-only, so they
cannot read the values off a compiled backend).
"""

# Mirrors gsplat_geometry::kSlerpSmallAngleDotThreshold
# (gsplat/geometry/kernels/cuda/csrc/quaternion.cuh): the quaternion dot above
# which SLERP uses a normalized linear blend instead of the sin/acos path.
# The compiled geometry extension exports its own copy of the C++ value and
# verifies it against this mirror at import time
# (gsplat/geometry/kernels/quaternion_ops.py).
SLERP_SMALL_ANGLE_DOT_THRESHOLD = 0.9995
