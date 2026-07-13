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
 */

#pragma once

// If user wants to explicitly build at least one, they have to specify all modules they want to build
// If they don't specify any, or they don't want to build some, all the others that aren't specified
// will be built.
// Ex:
//   GSPLAT_BUILD_3DGUT=1 GSPLAT_BUILD_2DGS=1 -> build 3dgut and 2dgs only, do not build anything else.
//   GSPLAT_BUILD_3DGUT=0 GSPLAT_BUILD_2DGS=0 -> build everything except 3dgut and 2dgs
//   <no GSPLAT_BUILD_* defined> -> build everything

#if defined(GSPLAT_BUILD_2DGS) && GSPLAT_BUILD_2DGS      \
    || defined(GSPLAT_BUILD_3DGS) && GSPLAT_BUILD_3DGS   \
    || defined(GSPLAT_BUILD_3DGUT) && GSPLAT_BUILD_3DGUT \
    || defined(GSPLAT_BUILD_ADAM) && GSPLAT_BUILD_ADAM   \
    || defined(GSPLAT_BUILD_RELOC) && GSPLAT_BUILD_RELOC \
    || defined(GSPLAT_BUILD_LOSSES) && GSPLAT_BUILD_LOSSES

#    define GSPLAT_DEFAULT_ENABLE_BUILD 0
#else
#    define GSPLAT_DEFAULT_ENABLE_BUILD 1
#endif

#ifndef GSPLAT_BUILD_2DGS
#    define GSPLAT_BUILD_2DGS GSPLAT_DEFAULT_ENABLE_BUILD
#endif

#ifndef GSPLAT_BUILD_3DGS
#    define GSPLAT_BUILD_3DGS GSPLAT_DEFAULT_ENABLE_BUILD
#endif

#ifndef GSPLAT_BUILD_3DGUT
#    define GSPLAT_BUILD_3DGUT GSPLAT_DEFAULT_ENABLE_BUILD
#endif

#ifndef GSPLAT_BUILD_ADAM
#    define GSPLAT_BUILD_ADAM GSPLAT_DEFAULT_ENABLE_BUILD
#endif

#ifndef GSPLAT_BUILD_RELOC
#    define GSPLAT_BUILD_RELOC GSPLAT_DEFAULT_ENABLE_BUILD
#endif

#ifndef GSPLAT_BUILD_LOSSES
#    define GSPLAT_BUILD_LOSSES GSPLAT_DEFAULT_ENABLE_BUILD
#endif

// Camera wrappers are a Python-exposed testing facility, not a core module:
// build.py omits CameraWrappers.cu from the source list when this flag is off,
// so it defaults to 0 rather than GSPLAT_DEFAULT_ENABLE_BUILD.
#ifndef GSPLAT_BUILD_CAMERA_WRAPPERS
#    define GSPLAT_BUILD_CAMERA_WRAPPERS 0
#endif

// Bilateral-grid affine-matrix dimensions (see FusedBgGridLosses). The grid
// stores one 3x4 affine matrix per (b, d, h, w) cell, flattened into 12
// channels. This layout is structural — the kernels and the Python wrapper
// both hard-assume a 3x4 (12-channel) affine — so it is fixed here and is
// deliberately NOT -D-overrideable (a different count would silently
// disagree with the wrapper's cell-count math).
#define LOSSES_GRID_NUM_ROWS     3
#define LOSSES_GRID_NUM_COLS     4
#define LOSSES_GRID_NUM_CHANNELS (LOSSES_GRID_NUM_ROWS * LOSSES_GRID_NUM_COLS)

#ifndef GSPLAT_NUM_CHANNELS
#    define GSPLAT_NUM_CHANNELS 1, 2, 3, 4, 5, 6, 8, 9, 16, 17, 21, 23, 24, 32
#endif

#ifndef GSPLAT_MACRO_TILE_RENDER_TILE_SIZES
#    define GSPLAT_MACRO_TILE_RENDER_TILE_SIZES 8, 16
#endif
