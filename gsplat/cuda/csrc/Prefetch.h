/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/irange.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace gsplat
{
struct TilePrefetchRange
{
    int64_t offset;
    int64_t count;
    uint32_t num_tiles_h;
    uint32_t num_tiles_w;
    uint32_t image_height;
    uint32_t image_width;
    uint32_t tile_size;
};

inline void append_per_camera_prefetch_ranges(
    std::vector<void *> &prefetch_ptrs,
    std::vector<size_t> &prefetch_sizes,
    at::TensorList tensors,
    int64_t camera_offset,
    int64_t camera_count
)
{
    for(const at::Tensor &tensor: tensors)
    {
        TORCH_INTERNAL_ASSERT(tensor.is_contiguous());
        TORCH_INTERNAL_ASSERT(camera_offset + camera_count <= tensor.size(0));
        const size_t row_size = tensor.stride(0) * tensor.element_size();
        const size_t size     = camera_count * row_size;
        if(size > 0)
        {
            prefetch_ptrs.emplace_back(static_cast<uint8_t *>(tensor.data_ptr()) + camera_offset * row_size);
            prefetch_sizes.emplace_back(size);
        }
    }
}

namespace detail
{
    inline void append_per_tile_tensor_range(
        std::vector<void *> &prefetch_ptrs,
        std::vector<size_t> &prefetch_sizes,
        const at::Tensor &tensor,
        const TilePrefetchRange &range
    )
    {
        const size_t tile_size_bytes = tensor.stride(2) * tensor.element_size();
        const size_t size            = range.count * tile_size_bytes;
        if(size > 0)
        {
            prefetch_ptrs.emplace_back(static_cast<uint8_t *>(tensor.data_ptr()) + range.offset * tile_size_bytes);
            prefetch_sizes.emplace_back(size);
        }
    }

    inline void append_per_tile_image_ranges(
        std::vector<void *> &prefetch_ptrs,
        std::vector<size_t> &prefetch_sizes,
        const at::Tensor &tensor,
        const TilePrefetchRange &range,
        int64_t tiles_per_camera
    )
    {
        const size_t pixel_stride_bytes  = tensor.stride(2) * tensor.element_size();
        const size_t row_stride_bytes    = tensor.stride(1) * tensor.element_size();
        const size_t camera_stride_bytes = tensor.stride(0) * tensor.element_size();
        const int64_t tile_range_end     = range.offset + range.count;
        int64_t current_tile             = range.offset;

        while(current_tile < tile_range_end)
        {
            const int64_t camera_id         = current_tile / tiles_per_camera;
            const int64_t camera_tile_begin = camera_id * tiles_per_camera;
            const int64_t next_camera_tile  = std::min(tile_range_end, camera_tile_begin + tiles_per_camera);
            const int64_t first_tile        = current_tile - camera_tile_begin;
            const int64_t last_tile         = next_camera_tile - camera_tile_begin - 1;

            const uint32_t first_row = first_tile / range.num_tiles_w;
            const uint32_t first_col = first_tile % range.num_tiles_w;
            const uint32_t last_row  = last_tile / range.num_tiles_w;
            const uint32_t last_col  = last_tile % range.num_tiles_w;

            const uint32_t row_start = first_row * range.tile_size;
            const uint32_t row_end   = std::min(range.image_height, (last_row + 1) * range.tile_size);
            const uint32_t col_start = first_col * range.tile_size;
            const uint32_t col_end   = std::min(range.image_width, (last_col + 1) * range.tile_size);

            if(row_start < row_end)
            {
                auto *camera_ptr = static_cast<uint8_t *>(tensor.data_ptr()) + camera_id * camera_stride_bytes;
                auto *begin      = camera_ptr + row_start * row_stride_bytes + col_start * pixel_stride_bytes;
                auto *end        = camera_ptr + (row_end - 1) * row_stride_bytes + col_end * pixel_stride_bytes;
                if(begin < end)
                {
                    prefetch_ptrs.emplace_back(begin);
                    prefetch_sizes.emplace_back(static_cast<size_t>(end - begin));
                }
            }
            current_tile = next_camera_tile;
        }
    }
} // namespace detail

inline void append_per_tile_prefetch_ranges(
    std::vector<void *> &prefetch_ptrs,
    std::vector<size_t> &prefetch_sizes,
    at::TensorList tensors,
    const TilePrefetchRange &range
)
{
    if(range.count == 0)
    {
        return;
    }
    const int64_t tiles_per_camera = static_cast<int64_t>(range.num_tiles_h) * range.num_tiles_w;
    for(const at::Tensor &tensor: tensors)
    {
        TORCH_INTERNAL_ASSERT(tensor.is_contiguous());
        TORCH_INTERNAL_ASSERT(tensor.dim() >= 3);
        TORCH_INTERNAL_ASSERT(range.offset + range.count <= tensor.size(0) * tiles_per_camera);

        if(tensor.size(1) == range.num_tiles_h && tensor.size(2) == range.num_tiles_w)
        {
            detail::append_per_tile_tensor_range(prefetch_ptrs, prefetch_sizes, tensor, range);
        }
        else
        {
            TORCH_INTERNAL_ASSERT(tensor.size(1) == range.image_height && tensor.size(2) == range.image_width);
            detail::append_per_tile_image_ranges(prefetch_ptrs, prefetch_sizes, tensor, range, tiles_per_camera);
        }
    }
}

inline void mem_prefetch_batch_async(
    std::vector<void *> &prefetch_ptrs, std::vector<size_t> &prefetch_sizes, int device_id, cudaStream_t stream
)
{
    if(prefetch_ptrs.empty())
    {
        return;
    }
#if CUDART_VERSION < 13000
    for(const auto range_id: c10::irange(prefetch_ptrs.size()))
    {
        C10_CUDA_CHECK(cudaMemPrefetchAsync(prefetch_ptrs[range_id], prefetch_sizes[range_id], device_id, stream));
    }
#else
    const cudaMemLocation location                  = {cudaMemLocationTypeDevice, device_id};
    std::vector<cudaMemLocation> prefetch_locations = {location};
    std::vector<size_t> prefetch_location_indices   = {0};
    C10_CUDA_CHECK(cudaMemPrefetchBatchAsync(
        prefetch_ptrs.data(),
        prefetch_sizes.data(),
        prefetch_ptrs.size(),
        prefetch_locations.data(),
        prefetch_location_indices.data(),
        prefetch_locations.size(),
        0,
        stream
    ));
#endif
}

inline void per_camera_memset_async(
    at::TensorList tensors, int64_t camera_offset, int64_t camera_count, cudaStream_t stream
)
{
    for(const at::Tensor &tensor: tensors)
    {
        TORCH_INTERNAL_ASSERT(tensor.is_contiguous());
        TORCH_INTERNAL_ASSERT(camera_offset + camera_count <= tensor.size(0));
        const size_t row_size = tensor.stride(0) * tensor.element_size();
        C10_CUDA_CHECK(cudaMemsetAsync(
            static_cast<uint8_t *>(tensor.data_ptr()) + camera_offset * row_size, 0, camera_count * row_size, stream
        ));
    }
}
} // namespace gsplat
