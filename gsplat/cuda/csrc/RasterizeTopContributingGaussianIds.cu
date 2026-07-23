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
namespace
{
    __device__ void insertion_sort_samples_by_depth(
        uint32_t *__restrict__ depth_indices,
        float *__restrict__ radiance_weights,
        int32_t *__restrict__ gaussian_ids,
        const uint32_t num_depth_samples
    )
    {
        for(uint32_t i = 1; i < num_depth_samples; ++i)
        {
            const uint32_t key_depth = depth_indices[i];
            const float key_weight   = radiance_weights[i];
            const int32_t key_id     = gaussian_ids[i];

            int32_t j = static_cast<int32_t>(i) - 1;
            while(j >= 0 && depth_indices[j] > key_depth)
            {
                depth_indices[j + 1]    = depth_indices[j];
                radiance_weights[j + 1] = radiance_weights[j];
                gaussian_ids[j + 1]     = gaussian_ids[j];
                --j;
            }

            depth_indices[j + 1]    = key_depth;
            radiance_weights[j + 1] = key_weight;
            gaussian_ids[j + 1]     = key_id;
        }
    }
} // namespace

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

template<uint32_t PIXELS_PER_THREAD>
struct TopContributingIdsAccumulator
{
    uint32_t N;
    bool packed;
    uint32_t num_depth_samples;
    int32_t *top_ids;   // [I, image_height, image_width, K]
    float *top_weights; // [I, image_height, image_width, K]
    int32_t *sample_ids;
    float *sample_weights;
    uint32_t *sample_depths;
    float T[PIXELS_PER_THREAD];
    uint32_t depth_counter[PIXELS_PER_THREAD];

    __device__ __forceinline__ void init_image(
        const uint32_t image_id, const uint32_t image_width, const uint32_t image_height
    )
    {
        const uint32_t image_stride  = image_height * image_width * num_depth_samples;
        top_ids                     += image_id * image_stride;
        top_weights                 += image_id * image_stride;
    }

    __device__ __forceinline__ void init_shared(void *shared)
    {
        const uint32_t num_pixels_per_cta = blockDim.x * PIXELS_PER_THREAD;
        sample_ids                        = reinterpret_cast<int32_t *>(shared);
        sample_weights = reinterpret_cast<float *>(&sample_ids[num_pixels_per_cta * num_depth_samples]);
        sample_depths  = reinterpret_cast<uint32_t *>(&sample_weights[num_pixels_per_cta * num_depth_samples]);
    }

    __device__ __forceinline__ void init(const uint32_t p, const uint32_t tid, const int32_t)
    {
        T[p]             = 1.0f;
        depth_counter[p] = 0;

        const uint32_t sample_base = (tid * PIXELS_PER_THREAD + p) * num_depth_samples;
        for(uint32_t k = 0; k < num_depth_samples; ++k)
        {
            sample_ids[sample_base + k]     = -1;
            sample_weights[sample_base + k] = 0.0f;
            sample_depths[sample_base + k]  = 0xFFFFFFFFu;
        }
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
        const uint32_t tid,
        const int32_t,
        const int32_t local_id,
        const float alpha,
        const float T_before,
        const float next_T
    )
    {
        const float radiance_weight = alpha * T_before;
        const uint32_t sample_base  = (tid * PIXELS_PER_THREAD + p) * num_depth_samples;

        uint32_t min_idx = 0;
        for(uint32_t k = 1; k < num_depth_samples; ++k)
        {
            if(sample_weights[sample_base + k] < sample_weights[sample_base + min_idx])
            {
                min_idx = k;
            }
        }

        if(radiance_weight > sample_weights[sample_base + min_idx])
        {
            sample_weights[sample_base + min_idx] = radiance_weight;
            sample_depths[sample_base + min_idx]  = depth_counter[p];
            sample_ids[sample_base + min_idx]     = local_id;
        }

        depth_counter[p] += 1;
        T[p]              = next_T;
    }

    __device__ __forceinline__ void finalize(const uint32_t p, const uint32_t tid, const int32_t pix_id)
    {
        const uint32_t sample_base = (tid * PIXELS_PER_THREAD + p) * num_depth_samples;
        insertion_sort_samples_by_depth(
            &sample_depths[sample_base], &sample_weights[sample_base], &sample_ids[sample_base], num_depth_samples
        );

        const uint32_t out_base = pix_id * num_depth_samples;
        for(uint32_t k = 0; k < num_depth_samples; ++k)
        {
            top_ids[out_base + k]     = sample_ids[sample_base + k];
            top_weights[out_base + k] = sample_weights[sample_base + k];
        }
    }

    // Per-pixel top-K sample scratch (ids + weights + depths), laid out by
    // init_shared above; the common batch staging buffers are sized separately.
    int64_t extra_shmem_bytes(const uint32_t num_pixels_per_cta) const
    {
        return static_cast<int64_t>(num_pixels_per_cta)
             * num_depth_samples
             * (sizeof(int32_t) + sizeof(float) + sizeof(uint32_t));
    }
};

void launch_rasterize_top_contributing_gaussian_ids_kernel(
    // Gaussian parameters
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor opacities,
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t num_depth_samples,
    // intersections
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    // outputs
    at::Tensor top_ids,
    at::Tensor top_weights
)
{
    const bool packed = means2d.dim() == 2;
    const uint32_t N  = packed ? 0 : means2d.size(-2); // number of gaussians
    const uint32_t I  = top_ids.numel() / (image_height * image_width * num_depth_samples);
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
            return TopContributingIdsAccumulator<PIXELS_PER_THREAD>{
                N,
                packed,
                num_depth_samples,
                top_ids.data_ptr<int32_t>(),
                top_weights.data_ptr<float>(),
            };
        }
    );
}

// Sparse counterpart: reuses TopContributingIdsAccumulator with the sparse
// traversal harness. Outputs are packed in original-pixel order
// ([P, num_depth_samples]). One thread per pixel (CTA_SIZE == tile_size^2).
void launch_rasterize_top_contributing_gaussian_ids_sparse_kernel(
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
    const uint32_t num_depth_samples,
    // sparse layout
    const at::Tensor active_tiles,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    const at::Tensor tile_pixel_mask,
    const at::Tensor tile_pixel_cumsum,
    const at::Tensor pixel_map,
    // outputs ([P, num_depth_samples])
    at::Tensor top_ids,
    at::Tensor top_weights
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
            return TopContributingIdsAccumulator<PIXELS_PER_THREAD>{
                N,
                packed,
                num_depth_samples,
                top_ids.data_ptr<int32_t>(),
                top_weights.data_ptr<float>(),
            };
        }
    );
}
} // namespace gsplat

#endif
