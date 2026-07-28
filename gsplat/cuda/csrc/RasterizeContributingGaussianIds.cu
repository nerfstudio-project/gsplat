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

#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAStream.h>

#    include "Rasterization.h"
#    include "RasterizeContributingLaunch.cuh"

namespace gsplat
{
////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

template<uint32_t PIXELS_PER_THREAD>
struct ContributingIdsAccumulator
{
    uint32_t N;
    bool packed;
    uint32_t max_num_contributing;
    int32_t *contributing_ids;   // [I, image_height, image_width, K]
    float *contributing_weights; // [I, image_height, image_width, K]
    float T[PIXELS_PER_THREAD];
    uint32_t num_written[PIXELS_PER_THREAD];

    __device__ __forceinline__ void init_image(
        const uint32_t image_id, const uint32_t image_width, const uint32_t image_height
    )
    {
        const uint32_t image_stride  = image_height * image_width * max_num_contributing;
        contributing_ids            += image_id * image_stride;
        contributing_weights        += image_id * image_stride;
    }

    __device__ __forceinline__ void init_shared(void *) { }

    __device__ __forceinline__ void init(const uint32_t p, const uint32_t, const int32_t)
    {
        T[p]           = 1.0f;
        num_written[p] = 0;
    }

    __device__ __forceinline__ int32_t local_id(const int32_t g) const
    {
        return packed ? g : (g % N);
    }

    __device__ __forceinline__ float transmittance(const uint32_t p) const
    {
        return T[p];
    }

    __device__ __forceinline__ void on_hit(
        const uint32_t p,
        const uint32_t,
        const int32_t pix_id,
        const int32_t local_id,
        const float alpha,
        const float T_before,
        const float next_T
    )
    {
        const uint32_t write_idx = num_written[p]++;
        if(write_idx < max_num_contributing)
        {
            const uint32_t out_base        = pix_id * max_num_contributing + write_idx;
            contributing_ids[out_base]     = local_id;
            contributing_weights[out_base] = alpha * T_before;
        }
        T[p] = next_T;
    }

    __device__ __forceinline__ void finalize(const uint32_t, const uint32_t, const int32_t) { }

    int64_t extra_shmem_bytes(const uint32_t) const
    {
        return 0;
    }
};

void launch_rasterize_contributing_gaussian_ids_kernel(
    // Gaussian parameters
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor opacities,
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t max_num_contributing,
    // intersections
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    // outputs
    at::Tensor contributing_ids,
    at::Tensor contributing_weights
)
{
    const bool packed = means2d.dim() == 2;
    const uint32_t N  = packed ? 0 : means2d.size(-2); // number of gaussians
    const uint32_t I  = contributing_ids.numel() / (image_height * image_width * max_num_contributing);
    launch_contributing_dense(
        means2d,
        conics,
        opacities,
        I,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids,
        [&]<uint32_t PIXELS_PER_THREAD>()
        {
            return ContributingIdsAccumulator<PIXELS_PER_THREAD>{
                N,
                packed,
                max_num_contributing,
                contributing_ids.data_ptr<int32_t>(),
                contributing_weights.data_ptr<float>(),
            };
        }
    );
}

// Sparse counterpart: reuses ContributingIdsAccumulator with the sparse
// traversal harness. Outputs are packed in original-pixel order ([P, K]). One
// thread per pixel (CTA_SIZE == tile_size^2).
void launch_rasterize_contributing_gaussian_ids_sparse_kernel(
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
    const uint32_t max_num_contributing,
    // sparse layout
    const at::Tensor active_tiles,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    const at::Tensor tile_pixel_mask,
    const at::Tensor tile_pixel_cumsum,
    const at::Tensor pixel_map,
    // outputs ([P, K])
    at::Tensor contributing_ids,
    at::Tensor contributing_weights
)
{
    const bool packed = means2d.dim() == 2;
    const uint32_t N  = packed ? 0 : means2d.size(-2);
    launch_contributing_sparse(
        means2d,
        conics,
        opacities,
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        active_tiles,
        tile_offsets,
        flatten_ids,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
        [&]<uint32_t PIXELS_PER_THREAD>()
        {
            return ContributingIdsAccumulator<PIXELS_PER_THREAD>{
                N,
                packed,
                max_num_contributing,
                contributing_ids.data_ptr<int32_t>(),
                contributing_weights.data_ptr<float>(),
            };
        }
    );
}
} // namespace gsplat

#endif
