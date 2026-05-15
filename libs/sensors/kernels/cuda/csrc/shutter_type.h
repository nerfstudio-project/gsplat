/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace gsplat_sensors {

// Source of truth for shutter-type constants used by the OpenCV pinhole
// kernels. Bound to Python in `ext.cpp` and re-exported as a Python IntEnum
// from `libs/sensors/kernels/cameras/types.py`. The Python module verifies at
// import time that its integer values match this enum.
//
// Distinct from upstream gsplat's `gsplat::ShutterType` in
// `gsplat/cuda/include/Cameras.h` (values 0..4); placed in the
// `gsplat_sensors` namespace so the two cannot collide.
enum class ShutterType : int64_t {
    ROLLING_TOP_TO_BOTTOM = 1, // rows exposed first-to-last (portrait scan top down)
    ROLLING_LEFT_TO_RIGHT = 2, // columns exposed first-to-last (landscape scan left to right)
    ROLLING_BOTTOM_TO_TOP = 3, // rows exposed last-to-first (portrait scan bottom up)
    ROLLING_RIGHT_TO_LEFT = 4, // columns exposed last-to-first (landscape scan right to left)
    GLOBAL = 5,                // all pixels exposed simultaneously; no per-row time offset
};

} // namespace gsplat_sensors
