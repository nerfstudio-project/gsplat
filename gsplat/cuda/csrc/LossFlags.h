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

// Per-ray flag bitmask values. Downstream consumers may override these at
// compile time via -D flags; standalone gsplat builds use these defaults.

#ifndef GSPLAT_LOSS_FLAG_RGB_LABEL
#    define GSPLAT_LOSS_FLAG_RGB_LABEL 1
#endif

#ifndef GSPLAT_LOSS_FLAG_SKY_SEMANTIC
#    define GSPLAT_LOSS_FLAG_SKY_SEMANTIC 4
#endif

#ifndef GSPLAT_LOSS_FLAG_DROPPED
#    define GSPLAT_LOSS_FLAG_DROPPED 64
#endif

#ifndef GSPLAT_LOSS_FLAG_INVALID
#    define GSPLAT_LOSS_FLAG_INVALID 256
#endif

#ifndef GSPLAT_LOSS_FLAG_DIFIXED
#    define GSPLAT_LOSS_FLAG_DIFIXED 512
#endif

#ifndef GSPLAT_LOSS_FLAG_SYNTHETIC
#    define GSPLAT_LOSS_FLAG_SYNTHETIC 1024
#endif
