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
#include <c10/cuda/CUDAStream.h>

#include "Common.h"
#include "RasterizeContributingCommon.cuh"
#include "RasterizeContributingCommonSparse.cuh"

// Host-side launch boilerplate shared by the contributing-gaussian ops
// (num-contributing, contributing-ids, top-contributing). Every op runs the
// same traversal kernel (`rasterize_contributing_common_kernel` or its sparse
// counterpart): the op-specific outputs live entirely in an Accumulator, so the
// kernels are launched with an identical argument list apart from that
// Accumulator. These helpers own the grid/shared-memory setup, the dynamic
// shared-memory opt-in, and the tile_size -> (TILE_SIZE, CTA_SIZE) dispatch;
// each op supplies only a `make_accum` factory.
//
// `make_accum` is a C++20 templated lambda
// `[&]<uint32_t PIXELS_PER_THREAD>() { return SomeAccumulator<...>{...}; }`;
// it is invoked once the dispatch has fixed PIXELS_PER_THREAD. Each Accumulator
// reports any dynamic shared memory it needs beyond the batch staging buffers
// via `extra_shmem_bytes(num_pixels_per_cta)`.

namespace gsplat
{
template<class MakeAccum>
void launch_contributing_dense(
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor opacities,
    const uint32_t I,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    MakeAccum make_accum
)
{
    const uint32_t grid_h   = tile_offsets.size(-2);
    const uint32_t grid_w   = tile_offsets.size(-1);
    const uint32_t n_isects = flatten_ids.size(0);
    const dim3 grid         = {I, grid_h, grid_w};

    auto launch_variant = [&]<uint32_t TILE_SIZE, uint32_t CTA_SIZE>()
    {
        const dim3 threads                   = dim3{CTA_SIZE, 1, 1};
        constexpr uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE;
        auto accum                           = make_accum.template operator()<PIXELS_PER_THREAD>();
        using Accumulator                    = decltype(accum);
        const int64_t shmem_size             = rasterize_contributing_common_shmem_size<CTA_SIZE>()
                                             + accum.extra_shmem_bytes(CTA_SIZE * PIXELS_PER_THREAD);

        if(cudaFuncSetAttribute(
               rasterize_contributing_common_kernel<TILE_SIZE, CTA_SIZE, Accumulator>,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               shmem_size
           )
           != cudaSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shmem_size,
                " bytes), try lowering tile_size or num_depth_samples."
            );
        }

        rasterize_contributing_common_kernel<TILE_SIZE, CTA_SIZE, Accumulator>
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                n_isects,
                reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
                reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                opacities.const_data_ptr<float>(),
                image_width,
                image_height,
                tile_offsets.const_data_ptr<int32_t>(),
                flatten_ids.const_data_ptr<int32_t>(),
                accum
            );
    };

    if(tile_size == 16)
    {
        launch_variant.template operator()<16, 64>();
    }
    else if(tile_size == 4)
    {
        launch_variant.template operator()<4, 16>();
    }
    else
    {
        AT_ERROR("Unsupported tile_size ", tile_size, "; supported values are {4, 16}.");
    }
}

template<class MakeAccum>
void launch_contributing_sparse(
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor opacities,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::Tensor active_tiles,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    const at::Tensor tile_pixel_mask,
    const at::Tensor tile_pixel_cumsum,
    const at::Tensor pixel_map,
    MakeAccum make_accum
)
{
    const uint32_t AT = active_tiles.size(0);
    if(AT == 0)
    {
        return;
    }
    const uint32_t words = tile_pixel_mask.size(1);
    const dim3 grid      = {AT, 1, 1};

    auto launch_variant = [&]<uint32_t TILE_SIZE, uint32_t CTA_SIZE>()
    {
        const dim3 threads                   = dim3{CTA_SIZE, 1, 1};
        constexpr uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE;
        auto accum                           = make_accum.template operator()<PIXELS_PER_THREAD>();
        using Accumulator                    = decltype(accum);
        const int64_t shmem_size             = rasterize_contributing_common_shmem_size<CTA_SIZE>()
                                             + accum.extra_shmem_bytes(CTA_SIZE * PIXELS_PER_THREAD);

        if(cudaFuncSetAttribute(
               rasterize_contributing_common_sparse_kernel<TILE_SIZE, CTA_SIZE, Accumulator>,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               shmem_size
           )
           != cudaSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shmem_size,
                " bytes), try lowering tile_size or num_depth_samples."
            );
        }

        rasterize_contributing_common_sparse_kernel<TILE_SIZE, CTA_SIZE, Accumulator>
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
                reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                opacities.const_data_ptr<float>(),
                image_width,
                image_height,
                tile_width,
                tile_height,
                active_tiles.const_data_ptr<int32_t>(),
                tile_offsets.const_data_ptr<int32_t>(),
                flatten_ids.const_data_ptr<int32_t>(),
                tile_pixel_mask.const_data_ptr<uint64_t>(),
                tile_pixel_cumsum.const_data_ptr<int64_t>(),
                pixel_map.const_data_ptr<int64_t>(),
                words,
                accum
            );
    };

    if(tile_size == 16)
    {
        launch_variant.template operator()<16, 256>();
    }
    else if(tile_size == 4)
    {
        launch_variant.template operator()<4, 16>();
    }
    else
    {
        AT_ERROR("Unsupported tile_size ", tile_size, "; supported values are {4, 16}.");
    }
}
} // namespace gsplat
