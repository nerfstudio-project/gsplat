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

// Shared macro-tile segmented sort helper used by the fused intersection pipeline.

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

#include "SegmentedSort.h"
#include "MacroTileIntersect.h"

namespace higs
{
at::Tensor launch_mt_segmented_sort(
    int64_t n_macro_isects,
    int32_t n_macro_tiles,
    const at::Tensor in_mt_gauss_offsets,
    at::Tensor inout_mt_depth_keys,
    at::Tensor inout_mt_gauss_ids,
    at::Tensor tmp_mt_depth_keys,
    at::Tensor tmp_mt_gauss_ids
)
{
    if(n_macro_isects <= 0)
    {
        return inout_mt_gauss_ids;
    }

    int32_t *d_keys_in       = inout_mt_depth_keys.data_ptr<int32_t>();
    int32_t *d_keys_out      = tmp_mt_depth_keys.data_ptr<int32_t>();
    int32_t *d_vals_in       = inout_mt_gauss_ids.data_ptr<int32_t>();
    int32_t *d_vals_out      = tmp_mt_gauss_ids.data_ptr<int32_t>();
    const int32_t *d_offsets = in_mt_gauss_offsets.data_ptr<int32_t>();
    cudaStream_t stream      = at::cuda::getCurrentCUDAStream();

    auto &alloc = *::c10::cuda::CUDACachingAllocator::get();

    SegmentedSortState state;
    size_t scratch_bytes = SegmentedSortSetup(state, n_macro_tiles, (int32_t)n_macro_isects, nullptr);
    auto scratch         = alloc.allocate(scratch_bytes);
    SegmentedSortSetup(state, n_macro_tiles, (int32_t)n_macro_isects, scratch.get());

    int32_t *d_keys[2]   = {d_keys_in, d_keys_out};
    int32_t *d_values[2] = {d_vals_in, d_vals_out};

    int result_buf = SegmentedSortAsync(state, d_offsets, d_keys, d_values, stream);

    // result_buf is the index of the buffer holding the sorted data: 0 = inout_*
    // (input), 1 = tmp_* (scratch). Return that buffer's gaussian-ids tensor as a
    // zero-copy handle so the caller reads the right one without any copy/alias.
    return result_buf == 0 ? inout_mt_gauss_ids : tmp_mt_gauss_ids;
}
} // namespace higs
