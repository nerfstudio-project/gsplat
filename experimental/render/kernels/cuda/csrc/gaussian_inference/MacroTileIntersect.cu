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

namespace higs {

void launch_mt_segmented_sort(int64_t n_macro_isects,
                              int32_t n_macro_tiles,
                              const at::Tensor in_mt_gauss_offsets,
                              at::Tensor inout_mt_depth_keys,
                              at::Tensor inout_mt_gauss_ids,
                              at::Tensor out_mt_depth_keys_sorted,
                              at::Tensor out_mt_gauss_ids_sorted)
{
    if (n_macro_isects <= 0)
    {
        return;
    }

    int32_t *d_keys_in       = inout_mt_depth_keys.data_ptr<int32_t>();
    int32_t *d_keys_out      = out_mt_depth_keys_sorted.data_ptr<int32_t>();
    int32_t *d_vals_in       = inout_mt_gauss_ids.data_ptr<int32_t>();
    int32_t *d_vals_out      = out_mt_gauss_ids_sorted.data_ptr<int32_t>();
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

    if (result_buf == 0)
    {
        // copy_ instead of set_ to avoid aliasing the input and output
        // tensors — high-water-mark reuse across frames would otherwise
        // cause the sort's ping-pong buffers to overlap on the next frame.
        // Narrow to n_macro_isects since the buffers may have different
        // high-water-mark capacities.
        out_mt_depth_keys_sorted.narrow(0, 0, n_macro_isects).copy_(
            inout_mt_depth_keys.narrow(0, 0, n_macro_isects));
        out_mt_gauss_ids_sorted.narrow(0, 0, n_macro_isects).copy_(
            inout_mt_gauss_ids.narrow(0, 0, n_macro_isects));
    }
}

} // namespace higs
