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

#include "Config.h"

#if GSPLAT_BUILD_3DGS

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>

#include "Rasterization.h"
#include "RasterizeContributingCommon.cuh"
#include "RasterizeContributingCommonSparse.cuh"

namespace gsplat {

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

template <uint32_t PIXELS_PER_THREAD> struct NumContributingAccumulator {
    int32_t *num_contributing; // [I, image_height, image_width]
    float *alphas;             // [I, image_height, image_width]
    float T[PIXELS_PER_THREAD];
    int32_t counts[PIXELS_PER_THREAD];

    __device__ __forceinline__ void init_image(
        const uint32_t image_id,
        const uint32_t image_width,
        const uint32_t image_height
    ) {
        const uint32_t image_stride = image_height * image_width;
        num_contributing += image_id * image_stride;
        alphas += image_id * image_stride;
    }

    __device__ __forceinline__ void init_shared(void *) {}

    __device__ __forceinline__ void
    init(const uint32_t p, const uint32_t, const int32_t) {
        T[p] = 1.0f;
        counts[p] = 0;
    }

    __device__ __forceinline__ int32_t local_id(const int32_t) const {
        return 0;
    }

    __device__ __forceinline__ float transmittance(const uint32_t p) const {
        return T[p];
    }

    __device__ __forceinline__ void on_hit(
        const uint32_t p,
        const uint32_t,
        const int32_t,
        const int32_t,
        const float,
        const float,
        const float next_T
    ) {
        counts[p] += 1;
        T[p] = next_T;
    }

    __device__ __forceinline__ void
    finalize(const uint32_t p, const uint32_t, const int32_t pix_id) {
        num_contributing[pix_id] = counts[p];
        alphas[pix_id] = 1.0f - T[p];
    }
};

void launch_rasterize_num_contributing_gaussians_kernel(
    // Gaussian parameters
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor opacities,
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    // outputs
    at::Tensor num_contributing,
    at::Tensor alphas
) {
    const uint32_t I = num_contributing.numel() / (image_height * image_width);
    const uint32_t grid_h = tile_offsets.size(-2);
    const uint32_t grid_w = tile_offsets.size(-1);
    const uint32_t n_isects = flatten_ids.size(0);
    const dim3 grid = {I, grid_h, grid_w};

    auto launch_variant = [&]<uint32_t TILE_SIZE, uint32_t CTA_SIZE>() {
        const dim3 threads = dim3{CTA_SIZE, 1, 1};
        constexpr uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE;
        using Accumulator = NumContributingAccumulator<PIXELS_PER_THREAD>;
        const int64_t shmem_size =
            rasterize_contributing_common_shmem_size<CTA_SIZE>();

        if (cudaFuncSetAttribute(
                rasterize_contributing_common_kernel<
                    TILE_SIZE,
                    CTA_SIZE,
                    Accumulator>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shmem_size
            ) != cudaSuccess) {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shmem_size,
                " bytes), try lowering tile_size."
            );
        }

        Accumulator accum{
            num_contributing.data_ptr<int32_t>(),
            alphas.data_ptr<float>(),
        };

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

    if (tile_size == 16) {
        launch_variant.template operator()<16, 64>();
    } else if (tile_size == 4) {
        launch_variant.template operator()<4, 16>();
    } else {
        AT_ERROR(
            "Unsupported tile_size ",
            tile_size,
            "; supported values are {4, 16}."
        );
    }
}

// Sparse counterpart: reuses NumContributingAccumulator with the sparse
// traversal harness. Outputs are packed in original-pixel order ([P]). One
// thread per pixel (CTA_SIZE == tile_size^2), matching the sparse rasterizer.
void launch_rasterize_num_contributing_gaussians_sparse_kernel(
    // Gaussian parameters
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor opacities,
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // sparse layout
    const at::Tensor active_tiles,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    const at::Tensor tile_pixel_mask,
    const at::Tensor tile_pixel_cumsum,
    const at::Tensor pixel_map,
    // outputs ([P])
    at::Tensor num_contributing,
    at::Tensor alphas
) {
    const uint32_t AT = active_tiles.size(0);
    if (AT == 0) {
        return;
    }
    const uint32_t words = tile_pixel_mask.size(1);
    const dim3 grid = {AT, 1, 1};

    auto launch_variant = [&]<uint32_t TILE_SIZE, uint32_t CTA_SIZE>() {
        const dim3 threads = dim3{CTA_SIZE, 1, 1};
        constexpr uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE;
        using Accumulator = NumContributingAccumulator<PIXELS_PER_THREAD>;
        const int64_t shmem_size =
            rasterize_contributing_common_shmem_size<CTA_SIZE>();

        if (cudaFuncSetAttribute(
                rasterize_contributing_common_sparse_kernel<
                    TILE_SIZE,
                    CTA_SIZE,
                    Accumulator>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shmem_size
            ) != cudaSuccess) {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shmem_size,
                " bytes), try lowering tile_size."
            );
        }

        Accumulator accum{
            num_contributing.data_ptr<int32_t>(),
            alphas.data_ptr<float>(),
        };

        rasterize_contributing_common_sparse_kernel<
            TILE_SIZE,
            CTA_SIZE,
            Accumulator>
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
                reinterpret_cast<const uint64_t *>(
                    tile_pixel_mask.const_data_ptr<int64_t>()
                ),
                tile_pixel_cumsum.const_data_ptr<int64_t>(),
                pixel_map.const_data_ptr<int64_t>(),
                words,
                accum
            );
    };

    if (tile_size == 16) {
        launch_variant.template operator()<16, 256>();
    } else if (tile_size == 4) {
        launch_variant.template operator()<4, 16>();
    } else {
        AT_ERROR(
            "Unsupported tile_size ",
            tile_size,
            "; supported values are {4, 16}."
        );
    }
}

} // namespace gsplat

#endif
