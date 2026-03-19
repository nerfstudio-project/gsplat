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

/*
 * GSPLAT_FOR_EACH(macro, a, b, c, ...):
 *   Expands to: macro(a) macro(b) macro(c) ...
 *   - Works with zero or more arguments.
 *   - When called with no arguments, it expands to nothing.
 *   - Example:
 *       #define F(x) FEATURE_ITEM(x)
 *       GSPLAT_FOR_EACH(F, 1, 2, 3)
 *     expands to:
 *       FEATURE_ITEM(1) FEATURE_ITEM(2) FEATURE_ITEM(3)
 */

#define GSPLAT_PARENS ()

#define GSPLAT_EXPAND(...) GSPLAT_EXPAND4(GSPLAT_EXPAND4(GSPLAT_EXPAND4(GSPLAT_EXPAND4(__VA_ARGS__))))
#define GSPLAT_EXPAND4(...) GSPLAT_EXPAND3(GSPLAT_EXPAND3(GSPLAT_EXPAND3(GSPLAT_EXPAND3(__VA_ARGS__))))
#define GSPLAT_EXPAND3(...) GSPLAT_EXPAND2(GSPLAT_EXPAND2(GSPLAT_EXPAND2(GSPLAT_EXPAND2(__VA_ARGS__))))
#define GSPLAT_EXPAND2(...) GSPLAT_EXPAND1(GSPLAT_EXPAND1(GSPLAT_EXPAND1(GSPLAT_EXPAND1(__VA_ARGS__))))
#define GSPLAT_EXPAND1(...) __VA_ARGS__

#define GSPLAT_FOR_EACH(macro, ...)                                    \
  __VA_OPT__(GSPLAT_EXPAND(GSPLAT_FOR_EACH_HELPER(macro, __VA_ARGS__)))
#define GSPLAT_FOR_EACH_HELPER(macro, a1, ...)                         \
  macro(a1)                                                     \
  __VA_OPT__(GSPLAT_FOR_EACH_AGAIN GSPLAT_PARENS (macro, __VA_ARGS__))
#define GSPLAT_FOR_EACH_AGAIN() GSPLAT_FOR_EACH_HELPER
