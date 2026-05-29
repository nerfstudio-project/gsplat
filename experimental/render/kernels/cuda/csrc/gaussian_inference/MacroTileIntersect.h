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

#include <ATen/core/Tensor.h>
#include <cstdint>

namespace higs {

// Sort gaussian IDs by depth key within each macro-tile segment.
// Only sorted values (gaussian IDs) are guaranteed in the output;
// key buffers are used as working space and their final contents
// are undefined.
void launch_mt_segmented_sort(int64_t n_macro_isects,
                              int32_t n_macro_tiles,
                              const at::Tensor in_mt_gauss_offsets,
                              at::Tensor inout_mt_depth_keys,
                              at::Tensor inout_mt_gauss_ids,
                              at::Tensor out_mt_depth_keys_sorted,
                              at::Tensor out_mt_gauss_ids_sorted);

} // namespace higs
