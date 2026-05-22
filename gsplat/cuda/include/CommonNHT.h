/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NHT-specific compile-time constants. Included only by NHT translation units
 * so that the rest of the codebase has no dependency on the harmonic encoding
 * or tetrahedral feature interpolation parameters.
 */

#pragma once

// NHT: harmonic encoding (sin, cos to separate channels)
#define NUM_ENCODING_FREQUENCIES 1
#define FREQUENCY_SCALE_EXPONENTIAL 0  // 0=linear (k+1), 1=exponential (2^k)
#define ENCF (NUM_ENCODING_FREQUENCIES * 2)
#define FREQ_IDX(k, f) ((k) * ENCF + (f))

// NHT: tetrahedral feature interpolation (4 vertices per 3DGS primitive)
#define VERTEX_PER_PRIM 4
