/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
