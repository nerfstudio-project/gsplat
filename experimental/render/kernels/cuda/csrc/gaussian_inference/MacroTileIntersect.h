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

// Sort gaussian IDs by depth key within each macro-tile segment. The inout_*
// and tmp_* buffers are used as ping-pong working space; their final contents
// are otherwise undefined. RETURNS the tensor (a zero-copy handle aliasing
// either inout_mt_gauss_ids or tmp_mt_gauss_ids) that holds the sorted gaussian
// IDs. The returned tensor is always defined; for n_macro_isects <= 0 it is
// inout_mt_gauss_ids unchanged. No data is copied.
at::Tensor launch_mt_segmented_sort(int64_t n_macro_isects,
                                    int32_t n_macro_tiles,
                                    const at::Tensor in_mt_gauss_offsets,
                                    at::Tensor inout_mt_depth_keys,
                                    at::Tensor inout_mt_gauss_ids,
                                    at::Tensor tmp_mt_depth_keys,
                                    at::Tensor tmp_mt_gauss_ids);

} // namespace higs
