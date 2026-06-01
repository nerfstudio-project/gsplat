/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cassert>
#include <cooperative_groups.h>
#include <cuda/std/functional>
#include <tuple>
#include <vector>

// for CUB_WRAPPER
#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>

#include "MathUtils.h"
#include "Common.h"
#include "Intersect.h"
#include "Utils.cuh"

namespace gsplat
{
namespace cg = cooperative_groups;

#define CUB_WRAPPER_ASYNC(stream, func, ...)                                         \
    do                                                                               \
    {                                                                                \
        size_t temp_storage_bytes = 0;                                               \
        void *temp_storage        = nullptr;                                         \
        C10_CUDA_CHECK(func(temp_storage, temp_storage_bytes, __VA_ARGS__, stream)); \
        C10_CUDA_CHECK(cudaMallocAsync(&temp_storage, temp_storage_bytes, stream));  \
        C10_CUDA_CHECK(func(temp_storage, temp_storage_bytes, __VA_ARGS__, stream)); \
        C10_CUDA_CHECK(cudaFreeAsync(temp_storage, stream));                         \
    } while(false)

struct TileColumnRange
{
    int32_t start;
    int32_t end;
    bool done;
};

inline __device__ TileColumnRange row_key_column_range(
    const int64_t image_id,
    const int32_t row,
    const uint32_t tile_width,
    const uint32_t n_tiles,
    const int2 rect_min,
    const int2 rect_max,
    const int64_t key_start,
    const int64_t key_end
)
{
    const int64_t row_key_offset = image_id * n_tiles + static_cast<int64_t>(row) * tile_width;
    const bool done              = row_key_offset + rect_min.x >= key_end;
    const int64_t col_start
        = min(static_cast<int64_t>(rect_max.x), max(static_cast<int64_t>(rect_min.x), key_start - row_key_offset));
    const int64_t col_end
        = max(static_cast<int64_t>(rect_min.x), min(static_cast<int64_t>(rect_max.x), key_end - row_key_offset));
    return {static_cast<int32_t>(col_start), static_cast<int32_t>(col_end), done};
}

// ============================================================
// SNUGBOX + AccuTile helper functions
// (ported from test_viewer/src/cuda/Intersect.cu)
// ============================================================

inline __device__ float2
    accutile_ellipse_intersection(float A, float B, float C, float disc, float t, float2 p, bool isY, float coord)
{
    float p_u   = isY ? p.y : p.x;
    float p_v   = isY ? p.x : p.y;
    float coeff = isY ? A : C;

    float h         = coord - p_u;
    float sqrt_term = sqrtf(disc * h * h + t * coeff);

    return {(-B * h - sqrt_term) / coeff + p_v, (-B * h + sqrt_term) / coeff + p_v};
}

inline __device__ uint32_t accutile_process_tiles(
    float A,
    float B,
    float C,
    float disc,
    float t,
    float2 p,
    float2 bbox_min,
    float2 bbox_max,
    float2 bbox_argmin,
    float2 bbox_argmax,
    int2 rect_min,
    int2 rect_max,
    uint32_t tile_size,
    uint32_t tile_width,
    uint32_t n_tiles,
    bool isY,
    int64_t image_id,
    int64_t key_start,
    int64_t key_end,
    int64_t iid_enc,
    uint32_t tile_n_bits,
    int64_t depth_id_enc,
    uint32_t flatten_idx,
    int64_t *isect_ids,
    int32_t *flatten_ids,
    int64_t *cur_idx
)
{
    float BLOCK = (float)tile_size;

    if(isY)
    {
        rect_min    = {rect_min.y, rect_min.x};
        rect_max    = {rect_max.y, rect_max.x};
        bbox_min    = {bbox_min.y, bbox_min.x};
        bbox_max    = {bbox_max.y, bbox_max.x};
        bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
        bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
    }

    uint32_t tiles_count = 0;
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    intersect_max_line = {bbox_max.y, bbox_min.y};

    min_line = rect_min.x * BLOCK;
    if(bbox_min.x <= min_line)
    {
        intersect_min_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, min_line);
    }
    else
    {
        intersect_min_line = intersect_max_line;
    }

#pragma unroll 1
    for(int u = rect_min.x; u < rect_max.x; ++u)
    {
        max_line = min_line + BLOCK;
        if(max_line <= bbox_max.x)
        {
            intersect_max_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, max_line);
        }

        if(min_line <= bbox_argmin.y && bbox_argmin.y < max_line)
        {
            ellipse_min = bbox_min.y;
        }
        else
        {
            ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        if(min_line <= bbox_argmax.y && bbox_argmax.y < max_line)
        {
            ellipse_max = bbox_max.y;
        }
        else
        {
            ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }

        int min_tile_v = max(rect_min.y, min(rect_max.y, (int)(ellipse_min / BLOCK)));
        int max_tile_v = min(rect_max.y, max(rect_min.y, (int)(ellipse_max / BLOCK + 1)));

#pragma unroll 1
        for(int v = min_tile_v; v < max_tile_v; v++)
        {
            int64_t tile_id  = isY ? (int64_t)(u * tile_width + v) : (int64_t)(v * tile_width + u);
            int64_t tile_key = image_id * n_tiles + tile_id;
            if(tile_key < key_start || tile_key >= key_end)
            {
                continue;
            }
            ++tiles_count;

            if(isect_ids != nullptr)
            {
                isect_ids[*cur_idx]   = iid_enc | (tile_id << 32) | depth_id_enc;
                flatten_ids[*cur_idx] = static_cast<int32_t>(flatten_idx);
                ++(*cur_idx);
            }
        }

        intersect_min_line = intersect_max_line;
        min_line           = max_line;
    }
    return tiles_count;
}

// ============================================================
// Main intersection kernel
// ============================================================

template<typename scalar_t>
__global__ void intersect_tile_kernel(
    const int64_t offset,
    const int64_t count,
    // if the data is [...,  N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over I * N, only used if packed is False
    const uint32_t I,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ image_ids,    // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const scalar_t *__restrict__ means2d,            // [..., N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [..., N, 2] or [nnz, 2]
    const scalar_t *__restrict__ depths,             // [..., N] or [nnz]
    const float *__restrict__ conics,                // [..., N, 3] or [nnz, 3]  (Sigma^{-1} upper tri)
    const float *__restrict__ opacities,             // [..., N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [..., N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    const uint32_t image_n_bits,
    const int64_t key_start,
    const int64_t key_end,
    const bool *__restrict__ tile_mask,    // [I, tile_height, tile_width] optional
    int32_t *__restrict__ tiles_per_gauss, // [..., N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
)
{
    // parallelize over I * N.
    uint32_t idx    = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if(idx >= count)
    {
        return;
    }
    idx += offset;

    const float radius_x = radii[idx * 2];
    const float radius_y = radii[idx * 2 + 1];
    if(radius_x <= 0 || radius_y <= 0)
    {
        if(first_pass)
        {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    float2 mean2d = {(float)means2d[2 * idx], (float)means2d[2 * idx + 1]};

    int64_t iid          = packed ? image_ids[idx] : idx / N;
    int64_t iid_enc      = 0;
    int64_t depth_id_enc = 0;
    if(!first_pass)
    {
        iid_enc = iid << (32 + tile_n_bits);

        // Narrow to float so the 32-bit key is a monotonic depth ordering for
        // any scalar_t (a bare 32-bit reinterpret of a double would read only
        // half its bits). Monotonic for the non-negative depths that reach
        // here after near-plane culling; the sign bit would invert ordering.
        float depth_f = static_cast<float>(depths[idx]);
        // The float-bit key is monotonic only for non-negative depths: a set
        // sign bit would invert the unsigned ordering. isect_tiles is a
        // standalone op, so pin the invariant the comment relies on.
        assert(depth_f >= 0.f);
        // Bit-level reinterpret, zero-extended into the low 32 bits of the key.
        depth_id_enc = __float_as_uint(depth_f);
    }

    if(conics != nullptr && opacities != nullptr)
    {
        // AccuTile: conservative ellipse intersection using the full 2x2 inverse covariance.
        // conic = (a, b, c) = upper triangle of Sigma^{-1}
        // Quadratic form: a*dx^2 + 2*b*dx*dy + c*dy^2
        const float A = conics[idx * 3];
        const float B = conics[idx * 3 + 1];
        const float C = conics[idx * 3 + 2];

        // disc = B^2 - A*C = -(det Sigma^{-1})
        float disc = B * B - A * C;

        // Opacity-aware isocontour level: alpha = opacity * exp(-0.5 * q) >= ALPHA_THRESHOLD
        // => q <= 2 * ln(opacity / ALPHA_THRESHOLD). Cap at GAUSSIAN_EXTEND^2 (same as the gsplat radius budget).
        const float opacity = opacities[idx];
        float t             = fminf(GAUSSIAN_EXTEND * GAUSSIAN_EXTEND, 2.0f * __logf(opacity / ALPHA_THRESHOLD));

        // SNUGBOX: tight axis-aligned bounding box of the ellipse
        float neg_t_over_disc = -t / disc;
        float x_extent        = sqrtf(neg_t_over_disc * C);
        float y_extent        = sqrtf(neg_t_over_disc * A);

        float2 bbox_min = {mean2d.x - x_extent, mean2d.y - y_extent};
        float2 bbox_max = {mean2d.x + x_extent, mean2d.y + y_extent};

        float Bx_over_C    = B * x_extent / C;
        float By_over_A    = B * y_extent / A;
        float2 bbox_argmin = {mean2d.y + Bx_over_C, mean2d.x + By_over_A};
        float2 bbox_argmax = {mean2d.y - Bx_over_C, mean2d.x - By_over_A};

        float tile_size_f = (float)tile_size;
        int2 rect_min
            = {max(0, min((int)tile_width, (int)(bbox_min.x / tile_size_f))),
               max(0, min((int)tile_height, (int)(bbox_min.y / tile_size_f)))};
        int2 rect_max
            = {max(0, min((int)tile_width, (int)(bbox_max.x / tile_size_f + 1.f))),
               max(0, min((int)tile_height, (int)(bbox_max.y / tile_size_f + 1.f)))};

        int y_span = rect_max.y - rect_min.y;
        int x_span = rect_max.x - rect_min.x;
        if(y_span * x_span == 0)
        {
            if(first_pass)
            {
                tiles_per_gauss[idx] = 0;
            }
            return;
        }

        bool isY        = y_span < x_span;
        int64_t cur_idx = first_pass ? 0 : ((idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1]);

        uint32_t count = accutile_process_tiles(
            A,
            B,
            C,
            disc,
            t,
            mean2d,
            bbox_min,
            bbox_max,
            bbox_argmin,
            bbox_argmax,
            rect_min,
            rect_max,
            tile_size,
            tile_width,
            tile_width * tile_height,
            isY,
            iid,
            key_start,
            key_end,
            iid_enc,
            tile_n_bits,
            depth_id_enc,
            idx,
            first_pass ? nullptr : isect_ids,
            first_pass ? nullptr : flatten_ids,
            &cur_idx
        );

        if(first_pass)
        {
            tiles_per_gauss[idx] = static_cast<int32_t>(count);
        }
    }
    else
    {
        // AABB fallback: used when conics/opacities are not available (e.g. 2DGS).
        float tile_radius_x = radius_x / static_cast<float>(tile_size);
        float tile_radius_y = radius_y / static_cast<float>(tile_size);
        float tile_x        = mean2d.x / static_cast<float>(tile_size);
        float tile_y        = mean2d.y / static_cast<float>(tile_size);

        // tile_min is inclusive, tile_max is exclusive
        int2 tile_min, tile_max;
        tile_min.x = min(max(0, (int32_t)floor(tile_x - tile_radius_x)), tile_width);
        tile_min.y = min(max(0, (int32_t)floor(tile_y - tile_radius_y)), tile_height);
        tile_max.x = min(max(0, (int32_t)ceil(tile_x + tile_radius_x)), tile_width);
        tile_max.y = min(max(0, (int32_t)ceil(tile_y + tile_radius_y)), tile_height);

        // Sparse path: restrict to caller-active tiles. The same mask is applied
        // in both passes so cum_tiles_per_gauss stays consistent with the emit.
        // Dense callers pass tile_mask == nullptr and keep the original
        // closed-form count and unfiltered emit, byte-for-byte unchanged.
        const bool *mask_img = nullptr;
        if(tile_mask != nullptr)
        {
            const int64_t iid_for_mask = packed ? image_ids[idx] : static_cast<int64_t>(idx / N);
            mask_img                   = tile_mask + iid_for_mask * static_cast<int64_t>(tile_width) * tile_height;
        }

        if(first_pass)
        {
            int32_t count = 0;
            for(int32_t i = tile_min.y; i < tile_max.y; ++i)
            {
                TileColumnRange range = row_key_column_range(
                    iid, i, tile_width, tile_width * tile_height, tile_min, tile_max, key_start, key_end
                );
                if(range.done)
                {
                    break;
                }
                if(range.start < range.end)
                {
                    if(mask_img == nullptr)
                    {
                        count += range.end - range.start;
                    }
                    else
                    {
                        for(int32_t j = range.start; j < range.end; ++j)
                        {
                            if(mask_img[i * tile_width + j])
                            {
                                ++count;
                            }
                        }
                    }
                }
            }
            tiles_per_gauss[idx] = count;
            return;
        }

        int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
        for(int32_t i = tile_min.y; i < tile_max.y; ++i)
        {
            TileColumnRange range = row_key_column_range(
                iid, i, tile_width, tile_width * tile_height, tile_min, tile_max, key_start, key_end
            );
            if(range.done)
            {
                break;
            }
            if(range.start >= range.end)
            {
                continue;
            }
            for(int32_t j = range.start; j < range.end; ++j)
            {
                if(mask_img != nullptr && !mask_img[i * tile_width + j])
                {
                    continue;
                }
                int64_t tile_id      = i * tile_width + j;
                // e.g. tile_n_bits = 22:
                // image id (10 bits) | tile id (22 bits) | depth (32 bits)
                isect_ids[cur_idx]   = iid_enc | (tile_id << 32) | depth_id_enc;
                // the flatten index in [I * N] or [nnz]
                flatten_ids[cur_idx] = static_cast<int32_t>(idx);
                ++cur_idx;
            }
        }
    }
}

void launch_intersect_tile_kernel(
    // inputs
    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> conics,       // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> opacities,    // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [..., N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [..., N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids,     // [n_isects]
    // sparse-only: restrict enumeration to active tiles ([I, tile_height,
    // tile_width] bool). nullopt for the dense path.
    const at::optional<at::Tensor> tile_mask
)
{
    bool packed = means2d.dim() == 2;

    uint32_t N = 0, nnz = 0;
    int64_t n_elements;
    if(packed)
    {
        nnz        = means2d.size(0); // total number of gaussians
        n_elements = nnz;
    }
    else
    {
        N          = means2d.size(-2); // number of gaussians per image
        n_elements = I * N;
    }

    const uint32_t n_tiles      = tile_width * tile_height;
    // the number of bits needed to encode the image id and tile id; must match
    // the packing in intersect_tile so the (image, tile) id unpacks correctly.
    const uint32_t image_n_bits = bits_for_count(I);
    const uint32_t tile_n_bits  = bits_for_count(n_tiles);
    const int64_t key_start     = 0;
    const int64_t key_end       = static_cast<int64_t>(I) * n_tiles;
    // the first 32 bits are used for the image id and tile id altogether, so
    // check if we have enough bits for them.
    TORCH_CHECK(
        image_n_bits + tile_n_bits <= 32,
        "intersect_tile: (image, tile) id packing needs ",
        image_n_bits + tile_n_bits,
        " bits but only 32 are available (I=",
        I,
        ", n_tiles=",
        n_tiles,
        ")."
    );

    constexpr unsigned int threads = 256;
    unsigned int blocks            = ::cuda::ceil_div(n_elements, threads);

    if(n_elements == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(
        means2d.scalar_type(),
        "intersect_tile_kernel",
        [&]()
        {
            intersect_tile_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                0,
                n_elements,
                packed,
                I,
                N,
                nnz,
                image_ids.has_value() ? image_ids.value().const_data_ptr<int64_t>() : nullptr,
                gaussian_ids.has_value() ? gaussian_ids.value().const_data_ptr<int64_t>() : nullptr,
                means2d.const_data_ptr<scalar_t>(),
                radii.const_data_ptr<int32_t>(),
                depths.const_data_ptr<scalar_t>(),
                conics.has_value() ? conics.value().const_data_ptr<float>() : nullptr,
                opacities.has_value() ? opacities.value().const_data_ptr<float>() : nullptr,
                cum_tiles_per_gauss.has_value() ? cum_tiles_per_gauss.value().const_data_ptr<int64_t>() : nullptr,
                tile_size,
                tile_width,
                tile_height,
                tile_n_bits,
                image_n_bits,
                key_start,
                key_end,
                tile_mask.has_value() ? tile_mask.value().const_data_ptr<bool>() : nullptr,
                tiles_per_gauss.has_value() ? tiles_per_gauss.value().data_ptr<int32_t>() : nullptr,
                isect_ids.has_value() ? isect_ids.value().data_ptr<int64_t>() : nullptr,
                flatten_ids.has_value() ? flatten_ids.value().data_ptr<int32_t>() : nullptr
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
}

void launch_intersect_tile_kernels(
    // inputs
    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> conics,       // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> opacities,    // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [..., N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [..., N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids      // [n_isects]
)
{
    bool packed = means2d.dim() == 2;

    uint32_t N = 0, nnz = 0;
    int64_t n_elements;
    if(packed)
    {
        nnz        = means2d.size(0);
        n_elements = nnz;
    }
    else
    {
        N          = means2d.size(-2);
        n_elements = I * N;
    }

    const uint32_t n_tiles      = tile_width * tile_height;
    const uint32_t image_n_bits = bits_for_count(I);
    const uint32_t tile_n_bits  = bits_for_count(n_tiles);
    const int64_t key_start     = 0;
    const int64_t key_end       = static_cast<int64_t>(I) * n_tiles;
    assert(image_n_bits + tile_n_bits <= 32);

    constexpr unsigned int threads = 256;
    if(n_elements == 0)
    {
        return;
    }

    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        int64_t offset, count;
        std::tie(offset, count) = chunk(n_elements, device_id);
        unsigned int blocks     = ::cuda::ceil_div(count, threads);
        if(blocks > 0)
        {
            AT_DISPATCH_FLOATING_TYPES(
                means2d.scalar_type(),
                "intersect_tile_kernel",
                [&]()
                {
                    intersect_tile_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                        offset,
                        count,
                        packed,
                        I,
                        N,
                        nnz,
                        image_ids.has_value() ? image_ids.value().const_data_ptr<int64_t>() : nullptr,
                        gaussian_ids.has_value() ? gaussian_ids.value().const_data_ptr<int64_t>() : nullptr,
                        means2d.const_data_ptr<scalar_t>(),
                        radii.const_data_ptr<int32_t>(),
                        depths.const_data_ptr<scalar_t>(),
                        conics.has_value() ? conics.value().const_data_ptr<float>() : nullptr,
                        opacities.has_value() ? opacities.value().const_data_ptr<float>() : nullptr,
                        cum_tiles_per_gauss.has_value() ? cum_tiles_per_gauss.value().const_data_ptr<int64_t>()
                                                        : nullptr,
                        tile_size,
                        tile_width,
                        tile_height,
                        tile_n_bits,
                        image_n_bits,
                        key_start,
                        key_end,
                        nullptr,
                        tiles_per_gauss.has_value() ? tiles_per_gauss.value().data_ptr<int32_t>() : nullptr,
                        isect_ids.has_value() ? isect_ids.value().data_ptr<int64_t>() : nullptr,
                        flatten_ids.has_value() ? flatten_ids.value().data_ptr<int32_t>() : nullptr
                    );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }
            );
        }
    }
    merge_streams();
}

__global__ void add_counts_kernel(
    const int64_t count, const int32_t *__restrict__ local_counts, int32_t *__restrict__ tiles_per_gauss
)
{
    int64_t idx = cg::this_grid().thread_rank();
    if(idx >= count)
    {
        return;
    }
    int32_t local_count = local_counts[idx];
    if(local_count > 0)
    {
        atomicAdd_system(tiles_per_gauss + idx, local_count);
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile_kernels_privateuseone(
    // inputs
    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> conics,       // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> opacities,    // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort
)
{
    bool packed = means2d.dim() == 2;

    uint32_t N = 0, nnz = 0;
    int64_t n_elements;
    if(packed)
    {
        nnz        = means2d.size(0);
        n_elements = nnz;
    }
    else
    {
        N          = means2d.size(-2);
        n_elements = static_cast<int64_t>(I) * N;
    }

    auto opt                   = depths.options();
    at::Tensor tiles_per_gauss = at::zeros_like(depths, opt.dtype(at::kInt));
    at::Tensor isect_ids;
    at::Tensor flatten_ids;
    if(n_elements == 0)
    {
        isect_ids   = at::empty({0}, opt.dtype(at::kLong));
        flatten_ids = at::empty({0}, opt.dtype(at::kInt));
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }

    const uint32_t n_tiles      = tile_width * tile_height;
    const uint32_t image_n_bits = bits_for_count(I);
    const uint32_t tile_n_bits  = bits_for_count(n_tiles);
    assert(image_n_bits + tile_n_bits <= 32);

    constexpr unsigned int threads = 256;
    const int device_count         = static_cast<int>(c10::cuda::device_count());
    TORCH_CHECK(device_count > 0, "PrivateUse1 intersect_tile requires at least one CUDA device");
    const int64_t total_tile_keys = static_cast<int64_t>(I) * n_tiles;
    const unsigned int blocks     = ::cuda::ceil_div(n_elements, threads);

    std::vector<int32_t *> device_counts(device_count, nullptr);
    std::vector<int64_t *> device_cumsum(device_count, nullptr);
    std::vector<int64_t> device_intersection_offsets(device_count, 0);
    std::vector<int64_t> device_intersection_counts(device_count, 0);

    int64_t *device_totals = nullptr;
    C10_CUDA_CHECK(cudaMallocHost(&device_totals, device_count * sizeof(int64_t)));

    for(const auto device_id: c10::irange(device_count))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        C10_CUDA_CHECK(cudaMallocAsync(&device_counts[device_id], n_elements * sizeof(int32_t), stream));
        C10_CUDA_CHECK(cudaMallocAsync(&device_cumsum[device_id], n_elements * sizeof(int64_t), stream));

        size_t key_offset, key_count;
        std::tie(key_offset, key_count) = chunk(total_tile_keys, device_id);
        const int64_t key_start         = static_cast<int64_t>(key_offset);
        const int64_t key_end           = static_cast<int64_t>(key_offset + key_count);

        AT_DISPATCH_FLOATING_TYPES(
            means2d.scalar_type(),
            "intersect_tile_kernel_privateuseone_count",
            [&]()
            {
                intersect_tile_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    0,
                    n_elements,
                    packed,
                    I,
                    N,
                    nnz,
                    image_ids.has_value() ? image_ids.value().const_data_ptr<int64_t>() : nullptr,
                    gaussian_ids.has_value() ? gaussian_ids.value().const_data_ptr<int64_t>() : nullptr,
                    means2d.const_data_ptr<scalar_t>(),
                    radii.const_data_ptr<int32_t>(),
                    depths.const_data_ptr<scalar_t>(),
                    conics.has_value() ? conics.value().const_data_ptr<float>() : nullptr,
                    opacities.has_value() ? opacities.value().const_data_ptr<float>() : nullptr,
                    nullptr,
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    image_n_bits,
                    key_start,
                    key_end,
                    nullptr,
                    device_counts[device_id],
                    nullptr,
                    nullptr
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        );

        CUB_WRAPPER_ASYNC(
            stream, cub::DeviceScan::InclusiveSum, device_counts[device_id], device_cumsum[device_id], n_elements
        );

        add_counts_kernel<<<blocks, threads, 0, stream>>>(
            n_elements, device_counts[device_id], tiles_per_gauss.data_ptr<int32_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        C10_CUDA_CHECK(cudaMemcpyAsync(
            &device_totals[device_id],
            device_cumsum[device_id] + n_elements - 1,
            sizeof(int64_t),
            cudaMemcpyDeviceToHost,
            stream
        ));
    }

    int64_t n_isects = 0;
    for(const auto device_id: c10::irange(device_count))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        device_intersection_offsets[device_id]  = n_isects;
        device_intersection_counts[device_id]   = device_totals[device_id];
        n_isects                               += device_totals[device_id];
    }
    C10_CUDA_CHECK(cudaFreeHost(device_totals));

    isect_ids   = at::empty({n_isects}, opt.dtype(at::kLong));
    flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
    if(n_isects == 0)
    {
        for(const auto device_id: c10::irange(device_count))
        {
            C10_CUDA_CHECK(cudaSetDevice(device_id));
            auto stream = c10::cuda::getCurrentCUDAStream(device_id);
            C10_CUDA_CHECK(cudaFreeAsync(device_counts[device_id], stream));
            C10_CUDA_CHECK(cudaFreeAsync(device_cumsum[device_id], stream));
        }
        merge_streams();
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }

    for(const auto device_id: c10::irange(device_count))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        size_t key_offset, key_count;
        std::tie(key_offset, key_count) = chunk(total_tile_keys, device_id);
        const int64_t key_start         = static_cast<int64_t>(key_offset);
        const int64_t key_end           = static_cast<int64_t>(key_offset + key_count);
        const int64_t isect_offset      = device_intersection_offsets[device_id];
        const int64_t isect_count       = device_intersection_counts[device_id];
        if(isect_count == 0)
        {
            C10_CUDA_CHECK(cudaFreeAsync(device_counts[device_id], stream));
            C10_CUDA_CHECK(cudaFreeAsync(device_cumsum[device_id], stream));
            continue;
        }

        AT_DISPATCH_FLOATING_TYPES(
            means2d.scalar_type(),
            "intersect_tile_kernel_privateuseone_emit",
            [&]()
            {
                intersect_tile_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                    0,
                    n_elements,
                    packed,
                    I,
                    N,
                    nnz,
                    image_ids.has_value() ? image_ids.value().const_data_ptr<int64_t>() : nullptr,
                    gaussian_ids.has_value() ? gaussian_ids.value().const_data_ptr<int64_t>() : nullptr,
                    means2d.const_data_ptr<scalar_t>(),
                    radii.const_data_ptr<int32_t>(),
                    depths.const_data_ptr<scalar_t>(),
                    conics.has_value() ? conics.value().const_data_ptr<float>() : nullptr,
                    opacities.has_value() ? opacities.value().const_data_ptr<float>() : nullptr,
                    device_cumsum[device_id],
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    image_n_bits,
                    key_start,
                    key_end,
                    nullptr,
                    nullptr,
                    isect_ids.data_ptr<int64_t>() + isect_offset,
                    flatten_ids.data_ptr<int32_t>() + isect_offset
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        );

        C10_CUDA_CHECK(cudaFreeAsync(device_counts[device_id], stream));
        C10_CUDA_CHECK(cudaFreeAsync(device_cumsum[device_id], stream));
    }

    if(sort)
    {
        for(const auto device_id: c10::irange(device_count))
        {
            C10_CUDA_CHECK(cudaSetDevice(device_id));
            auto stream = c10::cuda::getCurrentCUDAStream(device_id);

            const int64_t isect_offset = device_intersection_offsets[device_id];
            const int64_t isect_count  = device_intersection_counts[device_id];
            if(isect_count > 0)
            {
                CUB_WRAPPER_ASYNC(
                    stream,
                    cub::DeviceMergeSort::SortPairs,
                    isect_ids.data_ptr<int64_t>() + isect_offset,
                    flatten_ids.data_ptr<int32_t>() + isect_offset,
                    isect_count,
                    ::cuda::std::less<int64_t>{}
                );
            }
        }
    }
    merge_streams();
    return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
}

__global__ void intersect_offset_kernel(
    const uint32_t offset,
    const uint32_t count,
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t I,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ offsets // [I, n_tiles]
)
{
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if(idx >= count)
    {
        return;
    }
    idx += offset;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t iid_curr      = isect_id_curr >> (tile_n_bits);
    int64_t tid_curr      = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr       = iid_curr * n_tiles + tid_curr;

    if(idx == 0)
    {
        // write out the offsets until the first valid tile (inclusive)
        for(uint32_t i = 0; i < id_curr + 1; ++i)
        {
            offsets[i] = static_cast<int32_t>(idx);
        }
    }
    if(idx == n_isects - 1)
    {
        // write out the rest of the offsets
        for(uint32_t i = id_curr + 1; i < I * n_tiles; ++i)
        {
            offsets[i] = static_cast<int32_t>(n_isects);
        }
    }

    if(idx > 0)
    {
        // visit the current and previous isect_id and check if the (bid, cid,
        // tile_id) tuple changes.
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if(isect_id_prev == isect_id_curr)
        {
            return;
        }

        // write out the offsets between the previous and current tiles
        int64_t iid_prev = isect_id_prev >> (tile_n_bits);
        int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        int64_t id_prev  = iid_prev * n_tiles + tid_prev;
        for(uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
        {
            offsets[i] = static_cast<int32_t>(idx);
        }
    }
}

void launch_intersect_offset_kernel(
    // inputs
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t I,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs
    at::Tensor offsets // [I, tile_height, tile_width]
)
{
    int64_t n_elements             = isect_ids.size(0); // total number of intersections
    constexpr unsigned int threads = 256;
    unsigned int blocks            = ::cuda::ceil_div(n_elements, threads);

    if(n_elements == 0)
    {
        offsets.fill_(0);
        return;
    }

    const uint32_t n_tiles     = tile_width * tile_height;
    // Must match the packing in launch_intersect_tile_kernel so the (image,
    // tile) id unpacks with the same field width it was packed with.
    const uint32_t tile_n_bits = bits_for_count(n_tiles);
    auto stream                = at::cuda::getCurrentCUDAStream();
    intersect_offset_kernel<<<blocks, threads, 0, stream>>>(
        0,
        n_elements,
        n_elements,
        isect_ids.const_data_ptr<int64_t>(),
        I,
        n_tiles,
        tile_n_bits,
        offsets.data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_intersect_offset_kernels(
    // inputs
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t I,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs
    at::Tensor offsets // [I, tile_height, tile_width]
)
{
    int64_t n_elements             = isect_ids.size(0);
    constexpr unsigned int threads = 256;

    if(n_elements == 0)
    {
        offsets.fill_(0);
        return;
    }

    const uint32_t n_tiles     = tile_width * tile_height;
    const uint32_t tile_n_bits = bits_for_count(n_tiles);

    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        int64_t offset, count;
        std::tie(offset, count) = chunk(n_elements, device_id);
        unsigned int blocks     = ::cuda::ceil_div(count, threads);
        if(blocks > 0)
        {
            intersect_offset_kernel<<<blocks, threads, 0, stream>>>(
                offset,
                count,
                n_elements,
                isect_ids.const_data_ptr<int64_t>(),
                I,
                n_tiles,
                tile_n_bits,
                offsets.data_ptr<int32_t>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    merge_streams();
}

// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
// DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
void radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
)
{
    if(n_isects <= 0)
    {
        return;
    }

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<int64_t> d_keys(isect_ids.data_ptr<int64_t>(), isect_ids_sorted.data_ptr<int64_t>());
    cub::DoubleBuffer<int32_t> d_values(flatten_ids.data_ptr<int32_t>(), flatten_ids_sorted.data_ptr<int32_t>());
    CUB_WRAPPER(
        cub::DeviceRadixSort::SortPairs,
        d_keys,
        d_values,
        n_isects,
        0,
        32 + tile_n_bits + image_n_bits,
        at::cuda::getCurrentCUDAStream()
    );
    switch(d_keys.selector)
    {
    case 0: // sorted items are stored in isect_ids
        isect_ids_sorted.set_(isect_ids);
        break;
    case 1: // sorted items are stored in isect_ids_sorted
        break;
    }
    switch(d_values.selector)
    {
    case 0: // sorted items are stored in flatten_ids
        flatten_ids_sorted.set_(flatten_ids);
        break;
    case 1: // sorted items are stored in flatten_ids_sorted
        break;
    }
}

// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceSegmentedRadixSort.html
// DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
void segmented_radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t n_segments,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    const at::Tensor offsets,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
)
{
    if(n_isects <= 0)
    {
        return;
    }

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<int64_t> d_keys(isect_ids.data_ptr<int64_t>(), isect_ids_sorted.data_ptr<int64_t>());
    cub::DoubleBuffer<int32_t> d_values(flatten_ids.data_ptr<int32_t>(), flatten_ids_sorted.data_ptr<int32_t>());
    // image dimensions are contiguous in the isect_ids,
    // so we can use DeviceSegmentedRadixSort to only sort the lower
    // (tile_n_bits + 32) bits
    CUB_WRAPPER(
        cub::DeviceSegmentedRadixSort::SortPairs,
        d_keys,
        d_values,
        n_isects,
        n_segments, // number of segments
        offsets.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>() + 1,
        0,
        32 + tile_n_bits,
        at::cuda::getCurrentCUDAStream()
    );
    switch(d_keys.selector)
    {
    case 0: // sorted items are stored in isect_ids
        isect_ids_sorted.set_(isect_ids);
        break;
    case 1: // sorted items are stored in isect_ids_sorted
        break;
    }
    switch(d_values.selector)
    {
    case 0: // sorted items are stored in flatten_ids
        flatten_ids_sorted.set_(flatten_ids);
        break;
    case 1: // sorted items are stored in flatten_ids_sorted
        break;
    }
}
} // namespace gsplat
