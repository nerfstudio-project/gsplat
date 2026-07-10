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
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>

#include "Common.h"
#include "GSplatBuildConfig.h"
#include "KernelUtils.cuh"
#include "Dispatch.h"
#include "IntersectAccuTile.cuh"
#include "Intersect.h"
#include "IntersectMTConfig.h"
#include "IntersectMacroTile.h"
#include "LaunchBounds.h"
#include "SegmentedSort.h"

namespace gsplat
{
enum class TileBinMode : int32_t
{
    COUNT,
    FILL
};

static constexpr int32_t MT_BINNING_CTA_THREADS       = 768;
// __launch_bounds__ occupancy hint: tune with register pressure/spill metrics.
static constexpr int32_t MT_BINNING_MIN_BLOCKS_PER_SM = 2;
static constexpr int32_t MT_BINNING_GAUSSIANS_PER_CTA = 4096;
static constexpr int32_t RT_MIN_BLOCKS                = arch_limited_blocks_per_sm(24); // min blocks for RT kernels
static constexpr int32_t RT_GAUSS_BATCHES_PER_CTA     = 1;

using SupportedRenderTileSizesPx = dispatch::IntParam<GSPLAT_MACRO_TILE_RENDER_TILE_SIZES>;

static constexpr float INV_ALPHA_THRESHOLD = 1.0f / ALPHA_THRESHOLD;

static constexpr bool is_power_of_2(int32_t value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

struct GaussianEllipseBounds
{
    float A, B, C, disc, t;
    float2 center;
    // Speedy-Splat SnugBox tips
    float2 left_tip, right_tip, bottom_tip, top_tip;
};

inline __device__ float2 bbox_min(const GaussianEllipseBounds &ellipse)
{
    return {ellipse.left_tip.x, ellipse.bottom_tip.y};
}

inline __device__ float2 bbox_max(const GaussianEllipseBounds &ellipse)
{
    return {ellipse.right_tip.x, ellipse.top_tip.y};
}

// Not sure if std::clamp is safe to use here so we use our own.
inline __device__ int32_t clamp_int32(int32_t value, int32_t lower, int32_t upper)
{
    return max(lower, min(upper, value));
}

inline __device__ float clamp_float(float value, float lower, float upper)
{
    return fmaxf(lower, fminf(upper, value));
}

// ============================================================
// Device helpers
// ============================================================

template<typename T, typename U, typename V>
inline __device__ void AssignAs(U &u, const V &v)
{
    reinterpret_cast<T &>(u) = reinterpret_cast<const T &>(v);
}

// Compute the snug box contact points for the opacity-thresholded support ellipse..
inline __device__ bool try_compute_gaussian_ellipse_bounds(
    const float *__restrict__ means2d,   // [N, 2] float32 (x, y)
    const float *__restrict__ conics,    // [N, 3] float32 (a, b, c) upper triangle of conic matrix
    const float *__restrict__ opacities, // [N]    float32
    int32_t gauss_id,
    GaussianEllipseBounds &ellipse // [out] snug box contact points and derived parameters
)
{
    ellipse.A           = conics[gauss_id * 3 + 0];
    ellipse.B           = conics[gauss_id * 3 + 1];
    ellipse.C           = conics[gauss_id * 3 + 2];
    const float opacity = opacities[gauss_id];

    ellipse.disc = ellipse.B * ellipse.B - ellipse.A * ellipse.C;
    // A bounded inverse-covariance ellipse requires B^2 - A*C < 0.
    // The negated comparison also rejects NaN before the division below.
    if(!(ellipse.disc < 0.f))
    {
        return false;
    }
    ellipse.t = min(GAUSSIAN_EXTEND * GAUSSIAN_EXTEND, 2.0f * __logf(opacity * INV_ALPHA_THRESHOLD));

    const float neg_t_over_disc = -ellipse.t / ellipse.disc;
    const float x_extent        = sqrtf(neg_t_over_disc * ellipse.C);
    const float y_extent        = sqrtf(neg_t_over_disc * ellipse.A);

    AssignAs<float2>(ellipse.center, means2d[gauss_id * 2]);
    const float Bx_over_C = ellipse.B * x_extent / ellipse.C;
    const float By_over_A = ellipse.B * y_extent / ellipse.A;
    ellipse.left_tip      = {ellipse.center.x - x_extent, ellipse.center.y + Bx_over_C};
    ellipse.right_tip     = {ellipse.center.x + x_extent, ellipse.center.y - Bx_over_C};
    ellipse.bottom_tip    = {ellipse.center.x + By_over_A, ellipse.center.y - y_extent};
    ellipse.top_tip       = {ellipse.center.x - By_over_A, ellipse.center.y + y_extent};

    const float2 box_min = bbox_min(ellipse);
    const float2 box_max = bbox_max(ellipse);
    const float dx       = box_max.x - box_min.x;
    const float dy       = box_max.y - box_min.y;
    if(dx <= 0.f || dy <= 0.f)
    {
        return false;
    }
    return true;
}

template<TileBinMode MODE, IntersectionType IT, int32_t TILE_SIZE_PX>
__global__ void __launch_bounds__(MT_BINNING_CTA_THREADS, MT_BINNING_MIN_BLOCKS_PER_SM) mt_binning_kernel(
    const int32_t I,
    const int32_t N,
    const float *__restrict__ means2d,   // [I, N, 2] float32
    const int32_t *__restrict__ radii,   // [I, N, 2] int32
    const float *__restrict__ depths,    // [I, N] float32
    const float *__restrict__ conics,    // [I, N, 3] float32
    const float *__restrict__ opacities, // [I, N] float32
    const int32_t macro_tile_cols,
    const int32_t macro_tile_rows,
    // TileBinMode::COUNT
    int32_t *__restrict__ mt_gauss_counts, // [I, n_macro_tiles] int32, output
    // TileBinMode::FILL
    const int32_t *__restrict__ mt_gauss_offsets, // [I * n_macro_tiles + 1] int32
    int32_t *__restrict__ mt_gauss_write_cursor,  // [I, n_macro_tiles] int32, input-output
    int32_t *__restrict__ mt_depth_keys,          // [n_isects] int32, output
    int32_t *__restrict__ mt_gauss_ids            // [n_isects] int32, output
)
{
    static_assert(
        SupportedRenderTileSizesPx::contains(TILE_SIZE_PX),
        "Macro-tile binning kernel instantiated with unsupported render tile size"
    );
    const int32_t n_macro_tiles       = macro_tile_rows * macro_tile_cols;
    const int32_t n_chunks_per_image  = (N + MT_BINNING_GAUSSIANS_PER_CTA - 1) / MT_BINNING_GAUSSIANS_PER_CTA;
    const int32_t image_idx           = blockIdx.x / n_chunks_per_image;
    const int32_t image_chunk_idx     = blockIdx.x - image_idx * n_chunks_per_image;
    const int32_t macro_tile_base     = image_idx * n_macro_tiles;
    const int32_t gaussian_image_base = image_idx * N;
    if(image_idx >= I)
    {
        return;
    }

    extern __shared__ int32_t smem_mt_gauss_counts[]; // [n_macro_tiles] int32
    // FILL mode: per-macro-tile offset reserved for this CTA chunk.
    // The offset is computed via atomicAdds to the global write cursor.
    int32_t *const smem_mt_chunk_offsets = smem_mt_gauss_counts + n_macro_tiles; // [n_macro_tiles] int32

#pragma unroll 1
    for(int32_t i = threadIdx.x; i < n_macro_tiles; i += blockDim.x)
    {
        smem_mt_gauss_counts[i] = 0;
        if constexpr(MODE == TileBinMode::FILL)
        {
            smem_mt_chunk_offsets[i] = 0;
        }
    }
    __syncthreads();

    const int32_t gauss_chunk_start = image_chunk_idx * MT_BINNING_GAUSSIANS_PER_CTA;
    const int32_t gauss_chunk_end   = min(gauss_chunk_start + MT_BINNING_GAUSSIANS_PER_CTA, N);

    constexpr int32_t MACRO_TILE_WIDTH_PX  = MACRO_TILE_WIDTH * TILE_SIZE_PX;
    constexpr int32_t MACRO_TILE_HEIGHT_PX = MACRO_TILE_HEIGHT * TILE_SIZE_PX;
    constexpr float INV_MT_WIDTH_PX        = 1.0f / (float)MACRO_TILE_WIDTH_PX;
    constexpr float INV_MT_HEIGHT_PX       = 1.0f / (float)MACRO_TILE_HEIGHT_PX;

    auto for_each_macro_tile_hit = [&](int32_t gauss_id, auto emit_mt_id)
    {
        // Projection writes non-positive radii for culled Gaussians; skip those
        // sentinels because they cannot intersect any macro-tile.
        const int32_t global_gauss_id = gaussian_image_base + gauss_id;
        if(radii[global_gauss_id * 2] <= 0 || radii[global_gauss_id * 2 + 1] <= 0)
        {
            return;
        }

        if constexpr(IT == IntersectionType::ELLIPSE)
        {
            GaussianEllipseBounds ellipse;
            if(!try_compute_gaussian_ellipse_bounds(
                   means2d,
                   conics,
                   opacities,
                   global_gauss_id,
                   /*out=*/ellipse
               ))
            {
                // Skip Gaussians with degenerate ellipse bounds.
                return;
            }

            const float2 box_min = bbox_min(ellipse);
            const float2 box_max = bbox_max(ellipse);
            const int2 macro_tile_rect_begin
                = {clamp_int32((int)(box_min.x * INV_MT_WIDTH_PX), 0, macro_tile_cols),
                   clamp_int32((int)(box_min.y * INV_MT_HEIGHT_PX), 0, macro_tile_rows)};
            const int2 macro_tile_rect_end
                = {clamp_int32((int)(box_max.x * INV_MT_WIDTH_PX + 1.f), 0, macro_tile_cols),
                   clamp_int32((int)(box_max.y * INV_MT_HEIGHT_PX + 1.f), 0, macro_tile_rows)};

            const int y_span = macro_tile_rect_end.y - macro_tile_rect_begin.y;
            const int x_span = macro_tile_rect_end.x - macro_tile_rect_begin.x;
            if(y_span * x_span == 0)
            {
                return;
            }

            const bool isY = y_span < x_span;

            const float2 bbox_argmin = {ellipse.left_tip.y, ellipse.bottom_tip.x};
            const float2 bbox_argmax = {ellipse.right_tip.y, ellipse.top_tip.x};
            accutile_walk_tile_strips(
                ellipse.A,
                ellipse.B,
                ellipse.C,
                ellipse.disc,
                ellipse.t,
                ellipse.center,
                box_min,
                box_max,
                bbox_argmin,
                bbox_argmax,
                macro_tile_rect_begin,
                macro_tile_rect_end,
                MACRO_TILE_WIDTH_PX,
                MACRO_TILE_HEIGHT_PX,
                macro_tile_cols,
                isY,
                std::move(emit_mt_id)
            );
        }
        else
        {
            // TODO: implement IntersectionType::AABB
            static_assert(
                IT == IntersectionType::ELLIPSE, "IntersectionType::AABB is not implemented for macro-tile binning"
            );
        }
    };

    // Count how many Gaussians hit for each macro-tile in this CTA chunk.
#pragma unroll 1
    for(int32_t gauss_id = gauss_chunk_start + threadIdx.x; gauss_id < gauss_chunk_end; gauss_id += blockDim.x)
    {
        for_each_macro_tile_hit(gauss_id, [&](int32_t mt) { atomicAdd(&smem_mt_gauss_counts[mt], 1); });
    }
    __syncthreads();

    if constexpr(MODE == TileBinMode::COUNT)
    {
        // Accumulate the chunk counts into the global histogram.
#pragma unroll 1
        for(int32_t mt = threadIdx.x; mt < n_macro_tiles; mt += blockDim.x)
        {
            if(smem_mt_gauss_counts[mt] > 0)
            {
                atomicAdd(&mt_gauss_counts[macro_tile_base + mt], smem_mt_gauss_counts[mt]);
            }
        }
    }
    else
    {
        static_assert(MODE == TileBinMode::FILL, "Unsupported TileBinMode");
        // Claim write positions used by this CTA chunk for each macro-tile with hits.
#pragma unroll 1
        for(int32_t mt = threadIdx.x; mt < n_macro_tiles; mt += blockDim.x)
        {
            if(smem_mt_gauss_counts[mt] > 0)
            {
                smem_mt_chunk_offsets[mt]
                    = atomicAdd(&mt_gauss_write_cursor[macro_tile_base + mt], smem_mt_gauss_counts[mt]);
            }
            // Reset counts so we can reuse as local write cursors.
            smem_mt_gauss_counts[mt] = 0;
        }
        __syncthreads();

        // Alias the shared histogram for its new role as local write cursors.
        int32_t *const smem_mt_write_cursor = smem_mt_gauss_counts; // [n_macro_tiles] int32

        // Write (depth, gauss_id) pairs to the global output.
#pragma unroll 1
        for(int32_t gauss_id = gauss_chunk_start + threadIdx.x; gauss_id < gauss_chunk_end; gauss_id += blockDim.x)
        {
            // Segmented radix sort consumes int32 keys. Depths are non-negative, so their
            // float32 bit patterns preserve numeric order when reinterpreted as int32.
            const int32_t global_gauss_id = gaussian_image_base + gauss_id;
            const int32_t depth_i32       = *reinterpret_cast<const int32_t *>(&depths[global_gauss_id]);

            for_each_macro_tile_hit(
                gauss_id,
                [&](int32_t mt)
                {
                    const int32_t chunk_slot = atomicAdd(&smem_mt_write_cursor[mt], 1);
                    const int32_t mt_global  = macro_tile_base + mt;
                    const int32_t write_pos  = mt_gauss_offsets[mt_global] + smem_mt_chunk_offsets[mt] + chunk_slot;
                    mt_depth_keys[write_pos] = depth_i32;
                    mt_gauss_ids[write_pos]  = global_gauss_id;
                }
            );
        }
    }
}

template<TileBinMode MODE>
static void launch_mt_binning_impl(
    // Host-only dispatch parameters.
    const IntersectionType intersection_type,
    const int32_t tile_size_px,
    // mt_binning_kernel arguments.
    const int32_t I,
    const int32_t N,
    const float *means2d,
    const int32_t *radii,
    const float *depths,
    const float *conics,
    const float *opacities,
    const int32_t macro_tile_cols,
    const int32_t macro_tile_rows,
    // TileBinMode::COUNT
    int32_t *mt_gauss_counts,
    // TileBinMode::FILL
    const int32_t *mt_gauss_offsets,
    int32_t *mt_gauss_write_cursor,
    int32_t *mt_depth_keys,
    int32_t *mt_gauss_ids
)
{
    if(I == 0 || N == 0)
    {
        return;
    }

    TORCH_CHECK(
        intersection_type == IntersectionType::ELLIPSE, "macro-tile AABB intersection (3DGUT) is not supported yet"
    );

    const int32_t n_macro_tiles      = macro_tile_rows * macro_tile_cols;
    const int32_t n_chunks_per_image = (N + MT_BINNING_GAUSSIANS_PER_CTA - 1) / MT_BINNING_GAUSSIANS_PER_CTA;
    const int32_t n_blocks           = I * n_chunks_per_image;

    // COUNT: 1 smem array; FILL: 2 (+chunk_base_idx).
    const std::size_t smem_bytes = (MODE == TileBinMode::FILL ? 2 : 1) * n_macro_tiles * sizeof(int32_t);

    TORCH_CHECK_VALUE(SupportedRenderTileSizesPx ::contains(tile_size_px), "Unsupported tile_size: ", tile_size_px);

    const auto launch_kernel = [&]<typename TileSizeConst>() -> void
    {
        constexpr auto kernel     = mt_binning_kernel<MODE, IntersectionType::ELLIPSE, TileSizeConst::value>;
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const cudaError_t status
            = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        if(status != cudaSuccess)
        {
            AT_ERROR(
                "Failed to set shared memory size for mt_binning_kernel "
                "(requested ",
                smem_bytes,
                " bytes, ",
                n_macro_tiles,
                " macro-tiles)"
            );
        }
        kernel<<<n_blocks, MT_BINNING_CTA_THREADS, smem_bytes, stream>>>(
            I,
            N,
            means2d,
            radii,
            depths,
            conics,
            opacities,
            macro_tile_cols,
            macro_tile_rows,
            mt_gauss_counts,
            mt_gauss_offsets,
            mt_gauss_write_cursor,
            mt_depth_keys,
            mt_gauss_ids
        );
    };

    const bool dispatched = dispatch::dispatch(SupportedRenderTileSizesPx{tile_size_px}, std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}

void launch_mt_binning_count(
    const IntersectionType intersection_type,
    const at::Tensor &means2d,
    const at::Tensor &radii,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const int32_t tile_size_px,
    const int32_t macro_tile_cols,
    const int32_t macro_tile_rows,
    at::Tensor &mt_gauss_counts
)
{
    const int32_t I = static_cast<int32_t>(means2d.size(0));
    const int32_t N = static_cast<int32_t>(means2d.size(1));
    launch_mt_binning_impl<TileBinMode::COUNT>(
        intersection_type,
        tile_size_px,
        I,
        N,
        means2d.data_ptr<float>(),
        radii.data_ptr<int32_t>(),
        nullptr, // depths
        conics.data_ptr<float>(),
        opacities.data_ptr<float>(),
        macro_tile_cols,
        macro_tile_rows,
        // TileBinMode::COUNT
        mt_gauss_counts.data_ptr<int32_t>(),
        // TileBinMode::FILL
        nullptr,
        nullptr,
        nullptr,
        nullptr
    );
}

void launch_mt_binning_fill(
    const IntersectionType intersection_type,
    const at::Tensor &means2d,
    const at::Tensor &radii,
    const at::Tensor &depths,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const int32_t tile_size_px,
    const int32_t macro_tile_cols,
    const int32_t macro_tile_rows,
    const at::Tensor &mt_gauss_offsets,
    at::Tensor &mt_gauss_write_cursor,
    at::Tensor &mt_depth_keys,
    at::Tensor &mt_gauss_ids
)
{
    const int32_t I = static_cast<int32_t>(means2d.size(0));
    const int32_t N = static_cast<int32_t>(means2d.size(1));
    launch_mt_binning_impl<TileBinMode::FILL>(
        intersection_type,
        tile_size_px,
        I,
        N,
        means2d.data_ptr<float>(),
        radii.data_ptr<int32_t>(),
        depths.data_ptr<float>(),
        conics.data_ptr<float>(),
        opacities.data_ptr<float>(),
        macro_tile_cols,
        macro_tile_rows,
        // TileBinMode::COUNT
        nullptr,
        // TileBinMode::FILL
        mt_gauss_offsets.data_ptr<int32_t>(),
        mt_gauss_write_cursor.data_ptr<int32_t>(),
        mt_depth_keys.data_ptr<int32_t>(),
        mt_gauss_ids.data_ptr<int32_t>()
    );
}

at::Tensor launch_mt_segmented_sort(
    const int64_t n_macro_isects,
    const int32_t n_macro_tiles,
    const at::Tensor &mt_gauss_offsets,
    at::Tensor &mt_depth_keys,
    at::Tensor &mt_gauss_ids,
    at::Tensor &mt_tmp_depth_keys,
    at::Tensor &mt_tmp_gauss_ids
)
{
    if(n_macro_isects <= 0 || n_macro_tiles == 0)
    {
        return mt_gauss_ids;
    }

    TORCH_CHECK(
        n_macro_isects <= std::numeric_limits<int32_t>::max(),
        "macro-tile segmented sort expects at most INT_MAX intersections, got ",
        n_macro_isects
    );

    int32_t *d_keys_in       = mt_depth_keys.data_ptr<int32_t>();
    int32_t *d_keys_out      = mt_tmp_depth_keys.data_ptr<int32_t>();
    int32_t *d_vals_in       = mt_gauss_ids.data_ptr<int32_t>();
    int32_t *d_vals_out      = mt_tmp_gauss_ids.data_ptr<int32_t>();
    const int32_t *d_offsets = mt_gauss_offsets.data_ptr<int32_t>();
    cudaStream_t stream      = at::cuda::getCurrentCUDAStream();

    auto &alloc = *::c10::cuda::CUDACachingAllocator::get();

    SegmentedSortState state;
    const std::size_t scratch_bytes = SegmentedSortSetup(state, n_macro_tiles, (int32_t)n_macro_isects, nullptr);
    auto scratch                    = alloc.allocate(scratch_bytes);
    SegmentedSortSetup(state, n_macro_tiles, (int32_t)n_macro_isects, scratch.get());

    int32_t *d_keys[2]   = {d_keys_in, d_keys_out};
    int32_t *d_values[2] = {d_vals_in, d_vals_out};

    const int result_buf = SegmentedSortAsync(state, d_offsets, d_keys, d_values, stream);

    return result_buf == 0 ? mt_gauss_ids : mt_tmp_gauss_ids;
}

__global__ void mt_gauss_batch_offsets_kernel(
    const int32_t total_macro_tiles,
    const int32_t *__restrict__ mt_gauss_offsets, // [I * n_macro_tiles + 1] int32
    int32_t *__restrict__ mt_gauss_batch_offsets  // [I * n_macro_tiles + 1] int32, output
)
{
    // Serial prefix sum: only one thread writes the offsets.
    if(blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    mt_gauss_batch_offsets[0] = 0;
    for(int32_t mt = 0; mt < total_macro_tiles; ++mt)
    {
        const int32_t n_gauss          = mt_gauss_offsets[mt + 1] - mt_gauss_offsets[mt];
        const int32_t n_batches        = (n_gauss + RT_GAUSS_BATCH_SIZE - 1) / RT_GAUSS_BATCH_SIZE;
        mt_gauss_batch_offsets[mt + 1] = mt_gauss_batch_offsets[mt] + n_batches;
    }
}

void launch_mt_gauss_batch_offsets(
    const at::Tensor &mt_gauss_offsets, at::Tensor &mt_gauss_batch_offsets, const int32_t total_macro_tiles
)
{
    mt_gauss_batch_offsets_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        total_macro_tiles, mt_gauss_offsets.data_ptr<int32_t>(), mt_gauss_batch_offsets.data_ptr<int32_t>()
    );
}

inline __device__ int32_t
    find_macro_tile(const int32_t *mt_gauss_batch_offsets, int32_t n_macro_tiles, int32_t batch_idx)
{
    const int32_t target = batch_idx + 1;
    int32_t begin        = 0;
    int32_t end          = n_macro_tiles + 1;
    while(begin < end)
    {
        const int32_t mid = begin + (end - begin) / 2;
        if(mt_gauss_batch_offsets[mid] < target)
        {
            begin = mid + 1;
        }
        else
        {
            end = mid;
        }
    }
    return begin - 1;
}

template<IntersectionType IT, int32_t TILE_SIZE>
__global__ void
    __launch_bounds__(RT_GAUSS_BATCHES_PER_CTA * MACRO_TILE_WIDTH * MACRO_TILE_HEIGHT, RT_MIN_BLOCKS) rt_gauss_batch_intersection_masks_kernel(
        const int32_t total_gauss_batches,
        const int32_t total_macro_tiles,
        const int32_t n_macro_tiles_per_image,
        const int32_t macro_tile_cols,
        const float *__restrict__ means2d,                  // [I, N, 2] float32
        const float *__restrict__ conics,                   // [I, N, 3] float32
        const float *__restrict__ opacities,                // [I, N] float32
        const int32_t *__restrict__ mt_gauss_offsets,       // [I * n_macro_tiles + 1] int32
        const int32_t *__restrict__ mt_gauss_ids_sorted,    // [n_isects] int32
        const int32_t *__restrict__ mt_gauss_batch_offsets, // [I * n_macro_tiles + 1] int32
        uint32_t *__restrict__ rt_bitmasks                  // [gauss_batch][mask_word][local_rt] uint32, output
    )
{
    static_assert(
        SupportedRenderTileSizesPx::contains(TILE_SIZE),
        "Render-tile mask kernel instantiated with unsupported render tile size"
    );
    constexpr int32_t NUM_RT_PER_MT = MACRO_TILE_WIDTH * MACRO_TILE_HEIGHT;
    static_assert(is_power_of_2(MACRO_TILE_WIDTH), "MACRO_TILE_WIDTH must be power of 2");
    static_assert(is_power_of_2(MACRO_TILE_HEIGHT), "MACRO_TILE_HEIGHT must be power of 2");
    static_assert(NUM_RT_PER_MT >= 32, "macro-tile must be at least warp size");
    static_assert(NUM_RT_PER_MT % 32 == 0, "macro-tile size must be a multiple of warp size");

    const int32_t tid       = threadIdx.x;
    const int32_t local_tid = tid % NUM_RT_PER_MT;
    const int32_t sub_cta   = tid / NUM_RT_PER_MT;
    const int32_t batch_idx = blockIdx.x * RT_GAUSS_BATCHES_PER_CTA + sub_cta;

    if(batch_idx >= total_gauss_batches)
    {
        return;
    }

    constexpr int32_t SMEM_FLOATS_PER_GAUSSIAN = 3
                                               + // lower Cholesky factor entries: l00, l10, l11
                                                 1
                                               + // ellipse q(x, y) <= t
                                                 1
                                               +    // precomputed -l10 / C
                                                 2; // Gaussian center x/y in macro-tile coords in render-tile units
    constexpr int32_t SMEM_FLOATS_PER_SUB_CTA  = SMEM_FLOATS_PER_GAUSSIAN * RT_GAUSSIANS_PER_MASK;

    extern __shared__ float smem_base[];
    float *const sub_cta_smem       = smem_base + sub_cta * SMEM_FLOATS_PER_SUB_CTA;
    float *const smem_l00           = sub_cta_smem;
    float *const smem_l10           = smem_l00 + RT_GAUSSIANS_PER_MASK;
    float *const smem_l11           = smem_l10 + RT_GAUSSIANS_PER_MASK;
    float *const smem_t             = smem_l11 + RT_GAUSSIANS_PER_MASK;
    float *const smem_neg_l10_rcp_C = smem_t + RT_GAUSSIANS_PER_MASK;
    float *const smem_gauss_x_in_mt = smem_neg_l10_rcp_C + RT_GAUSSIANS_PER_MASK;
    float *const smem_gauss_y_in_mt = smem_gauss_x_in_mt + RT_GAUSSIANS_PER_MASK;

    const int32_t mt_global = find_macro_tile(mt_gauss_batch_offsets, total_macro_tiles, batch_idx);
    const int32_t mt_id     = mt_global % n_macro_tiles_per_image;

    const int32_t mt_col = mt_id % macro_tile_cols;
    const int32_t mt_row = mt_id / macro_tile_cols;

    constexpr float INV_TILE_SIZE = 1.0f / TILE_SIZE;
    // Macro-tile center in render-tile units
    const float mt_cx = static_cast<float>(mt_col * MACRO_TILE_WIDTH) + static_cast<float>(MACRO_TILE_WIDTH) * 0.5f;
    const float mt_cy = static_cast<float>(mt_row * MACRO_TILE_HEIGHT) + static_cast<float>(MACRO_TILE_HEIGHT) * 0.5f;
    // Local render-tile center in render-tile units relative to macro-tile center.
    const float local_rt_cx
        = static_cast<float>(local_tid % MACRO_TILE_WIDTH) + 0.5f - static_cast<float>(MACRO_TILE_WIDTH) * 0.5f;
    const float local_rt_cy
        = static_cast<float>(local_tid / MACRO_TILE_WIDTH) + 0.5f - static_cast<float>(MACRO_TILE_HEIGHT) * 0.5f;

    const int32_t mt_start          = mt_gauss_offsets[mt_global];
    const int32_t mt_end            = mt_gauss_offsets[mt_global + 1];
    const int32_t mt_batch_idx      = batch_idx - mt_gauss_batch_offsets[mt_global];
    const int32_t batch_gauss_start = mt_start + mt_batch_idx * RT_GAUSS_BATCH_SIZE;
    const int32_t batch_gauss_end   = min(batch_gauss_start + RT_GAUSS_BATCH_SIZE, mt_end);
    const int32_t n_gauss           = batch_gauss_end - batch_gauss_start;
    const int32_t num_mask_words    = (n_gauss + RT_GAUSSIANS_PER_MASK - 1) / RT_GAUSSIANS_PER_MASK;

    auto load_gaussian = [&](int32_t smem_gauss_idx, int32_t mt_gauss_list_idx)
    {
        const int32_t gauss_id = mt_gauss_ids_sorted[mt_gauss_list_idx];

        const float a       = conics[gauss_id * 3 + 0];
        const float b       = conics[gauss_id * 3 + 1];
        const float c       = conics[gauss_id * 3 + 2];
        const float opacity = opacities[gauss_id];

        constexpr float CHOL_EPS = 1e-20f;
        // Add eps * I to make conic matrix slightly positive-definite.
        const float a_reg        = a + CHOL_EPS;
        const float c_reg        = c + CHOL_EPS;
        // Cholesky factorization of the conic matrix.
        const float l00_px       = sqrtf(fmaxf(a_reg, CHOL_EPS));
        const float l10_px       = b / l00_px;
        const float l11_px       = sqrtf(fmaxf(c_reg - l10_px * l10_px, CHOL_EPS));

        // Scale Cholesky factors to operate on deltas measured in render-tile units.
        const float l00 = l00_px * TILE_SIZE;
        const float l10 = l10_px * TILE_SIZE;
        const float l11 = l11_px * TILE_SIZE;
        const float C   = l10 * l10 + l11 * l11;
        const float t   = fminf(GAUSSIAN_EXTEND * GAUSSIAN_EXTEND, 2.0f * __logf(opacity * INV_ALPHA_THRESHOLD));

        smem_l00[smem_gauss_idx]           = l00;
        smem_l10[smem_gauss_idx]           = l10;
        smem_l11[smem_gauss_idx]           = l11;
        smem_t[smem_gauss_idx]             = t;
        smem_neg_l10_rcp_C[smem_gauss_idx] = -l10 / C;

        // Convert Gaussian center to measured in render-tile units and relative to macro-tile center.
        float2 pos;
        AssignAs<float2>(pos, means2d[gauss_id * 2]);
        smem_gauss_x_in_mt[smem_gauss_idx] = pos.x * INV_TILE_SIZE - mt_cx;
        smem_gauss_y_in_mt[smem_gauss_idx] = pos.y * INV_TILE_SIZE - mt_cy;
    };

    // Cholesky-based ellipse-tile overlap test in normalized tile-local
    // coords. nx, ny: Gaussian center relative to tile center, tile is
    // [-0.5, +0.5]^2. In Cholesky space (u, v) = (l00*dx + l10*dy, l11*dy),
    // which is L^T * [dx, dy]^T for lower Cholesky factor L. Thus
    // q = u^2 + v^2 = [dx, dy] * L * L^T * [dx, dy]^T, so the overlap
    // test finds min q over the tile boundary and checks q <= t. q is convex
    // (sum of squares of affine functions), so the global minimum over the
    // boundary is on the nearest horizontal AND nearest vertical edge.
    auto ellipse_overlaps_tile
        = [](float l00, float l10, float l11, float t, float neg_l10_rcp_C, float nx, float ny) -> bool
    {
        // If Gaussian center is inside the tile
        // x in [-0.5, 0.5] and y in [-0.5, 0.5]
        // then the Gaussian overlaps the tile.
        if(fabsf(nx) < 0.5f && fabsf(ny) < 0.5f)
        {
            return true;
        }

        // Check if the Gaussian overlaps the two nearest horizontal and vertical edges.

        const float dx0 = -0.5f - nx;
        const float dy0 = -0.5f - ny;
        const float dy1 = dy0 + 1.f;

        const float l00_dx0 = l00 * dx0;
        const float l00_dx1 = l00_dx0 + l00;

        // Nearest horizontal edge: minimize q with fixed dy and dx in
        // [dx0, dx0+1]. Since u = l00*dx + l10*dy and v = l11*dy,
        // v is constant on this edge. The best unconstrained u is 0, so
        // clamp 0 to the u values reachable at the edge endpoints.
        const float dy_h   = (ny >= 0.f) ? dy1 : dy0;
        const float l10_dy = l10 * dy_h;
        const float v_h    = l11 * dy_h;
        const float u_h    = clamp_float(0.f, l00_dx0 + l10_dy, l00_dx1 + l10_dy);
        const float q_h    = u_h * u_h + v_h * v_h;

        // Nearest vertical edge: minimize q at fixed dx, dy in [dy0, dy1].
        // Unconstrained optimum at dy = l00_dx * neg_l10_rcp_C, clamped.
        const float l00_dx = (nx >= 0.f) ? l00_dx1 : l00_dx0;
        const float dy_v   = clamp_float(l00_dx * neg_l10_rcp_C, dy0, dy1);
        const float u_v    = l10 * dy_v + l00_dx;
        const float v_v    = l11 * dy_v;
        const float q_v    = u_v * u_v + v_v * v_v;

        return fminf(q_h, q_v) <= t;
    };

#pragma unroll 1
    for(int32_t mask_word = 0; mask_word < num_mask_words; ++mask_word)
    {
        const int32_t mask_gauss_start = batch_gauss_start + mask_word * RT_GAUSSIANS_PER_MASK;
        const int32_t mask_gauss_count = min(RT_GAUSSIANS_PER_MASK, batch_gauss_end - mask_gauss_start);

        // Cooperatively load Gaussian data into smem.
        if constexpr(RT_GAUSSIANS_PER_MASK <= NUM_RT_PER_MT)
        {
            if(local_tid < mask_gauss_count)
            {
                load_gaussian(local_tid, mask_gauss_start + local_tid);
            }
        }
        else
        {
            for(int32_t i = local_tid; i < mask_gauss_count; i += NUM_RT_PER_MT)
            {
                load_gaussian(i, mask_gauss_start + i);
            }
        }

        cta_sync<NUM_RT_PER_MT>();

        // Build bitmask.
        uint32_t local_mask = 0;
        if constexpr(IT == IntersectionType::ELLIPSE)
        {
#pragma unroll 1
            for(int32_t j = 0; j < mask_gauss_count; ++j)
            {
                // Gaussian center relative to local render-tile center.
                const float nx = smem_gauss_x_in_mt[j] - local_rt_cx;
                const float ny = smem_gauss_y_in_mt[j] - local_rt_cy;

                if(ellipse_overlaps_tile(
                       smem_l00[j], smem_l10[j], smem_l11[j], smem_t[j], smem_neg_l10_rcp_C[j], nx, ny
                   ))
                {
                    local_mask |= (1u << j);
                }
            }
        }
        else
        {
            static_assert(
                IT == IntersectionType::ELLIPSE, "IntersectionType::AABB is not implemented for macro-tile binning yet"
            );
            // TODO: Implement IntersectionType::AABB
        }

        const int32_t batch_mask_word                            = batch_idx * RT_MASKS_PER_GAUSS_BATCH + mask_word;
        rt_bitmasks[batch_mask_word * NUM_RT_PER_MT + local_tid] = local_mask;
    }

    // Zero trailing unused mask words in the batch.
#pragma unroll 1
    for(int32_t mask_word = num_mask_words; mask_word < RT_MASKS_PER_GAUSS_BATCH; ++mask_word)
    {
        const int32_t batch_mask_word                            = batch_idx * RT_MASKS_PER_GAUSS_BATCH + mask_word;
        rt_bitmasks[batch_mask_word * NUM_RT_PER_MT + local_tid] = 0;
    }
}

void launch_rt_gauss_batch_intersection_masks(
    const IntersectionType intersection_type,
    const int32_t total_gauss_batches,
    const int32_t total_macro_tiles,
    const int32_t n_macro_tiles_per_image,
    const int32_t macro_tile_cols,
    const int32_t rt_tile_size_px,
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &mt_gauss_offsets,
    const at::Tensor &mt_gauss_ids_sorted,
    const at::Tensor &mt_gauss_batch_offsets,
    at::Tensor &rt_bitmasks
)
{
    if(total_gauss_batches == 0)
    {
        return;
    }

    TORCH_CHECK(
        intersection_type == IntersectionType::ELLIPSE,
        "render-tile AABB intersection masks (3DGUT) are not implemented yet."
    );

    const std::size_t smem_bytes = RT_GAUSS_BATCHES_PER_CTA * 7 * RT_GAUSSIANS_PER_MASK * sizeof(float);
    const dim3 threads(RT_GAUSS_BATCHES_PER_CTA * MACRO_TILE_WIDTH * MACRO_TILE_HEIGHT);
    const dim3 grid((total_gauss_batches + RT_GAUSS_BATCHES_PER_CTA - 1) / RT_GAUSS_BATCHES_PER_CTA);

    TORCH_CHECK_VALUE(
        SupportedRenderTileSizesPx ::contains(rt_tile_size_px), "Unsupported tile_size: ", rt_tile_size_px
    );

    const auto launch_kernel = [&]<typename TileSizeConst>() -> void
    {
        constexpr int TILE_SIZE   = TileSizeConst::value;
        constexpr auto kernel     = rt_gauss_batch_intersection_masks_kernel<IntersectionType::ELLIPSE, TILE_SIZE>;
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        kernel<<<grid, threads, smem_bytes, stream>>>(
            total_gauss_batches,
            total_macro_tiles,
            n_macro_tiles_per_image,
            macro_tile_cols,
            means2d.data_ptr<float>(),
            conics.data_ptr<float>(),
            opacities.data_ptr<float>(),
            mt_gauss_offsets.data_ptr<int32_t>(),
            mt_gauss_ids_sorted.data_ptr<int32_t>(),
            mt_gauss_batch_offsets.data_ptr<int32_t>(),
            reinterpret_cast<uint32_t *>(rt_bitmasks.data_ptr<int32_t>())
        );
    };

    const bool dispatched = dispatch::dispatch(SupportedRenderTileSizesPx{rt_tile_size_px}, std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}

__global__ void rt_count_gauss_and_prefix_offsets_kernel(
    const int32_t total_macro_tiles,
    const int32_t n_macro_tiles_per_image,
    const int32_t macro_tile_cols,
    const int32_t rt_tile_cols,
    const int32_t rt_tile_rows,
    const int32_t *__restrict__ mt_gauss_batch_offsets, // [I * n_macro_tiles + 1] int32
    const uint32_t *__restrict__ rt_bitmasks,           // [gauss_batch][mask_word][local_rt] uint32
    int32_t *__restrict__ rt_batch_gauss_offsets,       // [gauss_batch][local_rt] int32, output
    int32_t *__restrict__ rt_gauss_counts               // [I, rt_tile_rows, rt_tile_cols] int32, output
)
{
    constexpr int32_t NUM_RT_PER_MT = MACRO_TILE_WIDTH * MACRO_TILE_HEIGHT;

    const int32_t mt_global = blockIdx.x;
    const int32_t local_rt  = threadIdx.x;

    if(mt_global >= total_macro_tiles || local_rt >= NUM_RT_PER_MT)
    {
        return;
    }

    const int32_t batch_idx_start = mt_gauss_batch_offsets[mt_global];
    const int32_t batch_idx_end   = mt_gauss_batch_offsets[mt_global + 1];

    int32_t gauss_count_accum = 0;
    for(int32_t batch_idx = batch_idx_start; batch_idx < batch_idx_end; ++batch_idx)
    {
        rt_batch_gauss_offsets[batch_idx * NUM_RT_PER_MT + local_rt] = gauss_count_accum;

        // Number of Gaussians in the batch that intersect the local render tile.
        int32_t batch_gauss_count = 0;
#pragma unroll
        for(int32_t m = 0; m < RT_MASKS_PER_GAUSS_BATCH; ++m)
        {
            batch_gauss_count
                += __popc(rt_bitmasks[(batch_idx * RT_MASKS_PER_GAUSS_BATCH + m) * NUM_RT_PER_MT + local_rt]);
        }
        gauss_count_accum += batch_gauss_count;
    }

    const int32_t image_idx     = mt_global / n_macro_tiles_per_image;
    const int32_t mt_id         = mt_global - image_idx * n_macro_tiles_per_image;
    const int32_t global_rt_col = (mt_id % macro_tile_cols) * MACRO_TILE_WIDTH + (local_rt % MACRO_TILE_WIDTH);
    const int32_t global_rt_row = (mt_id / macro_tile_cols) * MACRO_TILE_HEIGHT + (local_rt / MACRO_TILE_WIDTH);
    const bool valid            = (global_rt_col < rt_tile_cols) && (global_rt_row < rt_tile_rows);
    if(valid)
    {
        rt_gauss_counts[(image_idx * rt_tile_rows + global_rt_row) * rt_tile_cols + global_rt_col] = gauss_count_accum;
    }
}

void launch_rt_count_gauss_and_prefix_offsets(
    const int32_t total_macro_tiles,
    const int32_t n_macro_tiles_per_image,
    const int32_t macro_tile_cols,
    const int32_t rt_tile_size_px,
    const int32_t rt_tile_cols,
    const int32_t rt_tile_rows,
    const at::Tensor &mt_gauss_batch_offsets,
    const at::Tensor &rt_bitmasks,
    at::Tensor &rt_batch_gauss_offsets,
    at::Tensor &rt_gauss_counts
)
{
    if(total_macro_tiles == 0)
    {
        return;
    }

    constexpr int32_t CTA_SIZE = MACRO_TILE_WIDTH * MACRO_TILE_HEIGHT;
    const dim3 threads(CTA_SIZE);
    const dim3 grid(total_macro_tiles);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rt_count_gauss_and_prefix_offsets_kernel<<<grid, threads, 0, stream>>>(
        total_macro_tiles,
        n_macro_tiles_per_image,
        macro_tile_cols,
        rt_tile_cols,
        rt_tile_rows,
        mt_gauss_batch_offsets.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t *>(rt_bitmasks.data_ptr<int32_t>()),
        rt_batch_gauss_offsets.data_ptr<int32_t>(),
        rt_gauss_counts.data_ptr<int32_t>()
    );
}

template<int32_t TILE_SIZE>
__global__ void
    __launch_bounds__(RT_GAUSS_BATCHES_PER_CTA * MACRO_TILE_WIDTH * MACRO_TILE_HEIGHT, RT_MIN_BLOCKS) rt_binning_fill_kernel(
        const int32_t total_gauss_batches,
        const int32_t total_macro_tiles,
        const int32_t n_macro_tiles_per_image,
        const int32_t macro_tile_cols,
        const int32_t rt_tile_cols,
        const int32_t rt_tile_rows,
        const int32_t *__restrict__ mt_gauss_offsets,       // [I * n_macro_tiles + 1] int32
        const int32_t *__restrict__ mt_gauss_ids_sorted,    // [n_isects] int32
        const int32_t *__restrict__ mt_gauss_batch_offsets, // [I * n_macro_tiles + 1] int32
        const int32_t *__restrict__ rt_gauss_offsets,       // [I * n_rt + 1] int32
        const int32_t *__restrict__ rt_batch_gauss_offsets, // [gauss_batch][local_rt] int32
        const uint32_t *__restrict__ rt_bitmasks,           // [gauss_batch][mask_word][local_rt] uint32
        int32_t *__restrict__ rt_gauss_ids                  // [n_rt_isects] int32, output
    )
{
    static_assert(
        SupportedRenderTileSizesPx::contains(TILE_SIZE),
        "Render-tile fill kernel instantiated with unsupported render tile size"
    );
    constexpr int32_t NUM_RT_PER_MT = MACRO_TILE_WIDTH * MACRO_TILE_HEIGHT;

    const int32_t tid       = threadIdx.x;
    const int32_t local_tid = tid % NUM_RT_PER_MT;
    const int32_t sub_cta   = tid / NUM_RT_PER_MT;
    const int32_t batch_idx = blockIdx.x * RT_GAUSS_BATCHES_PER_CTA + sub_cta;

    if(batch_idx >= total_gauss_batches)
    {
        return;
    }

    extern __shared__ int32_t smem[];
    int32_t *const smem_gid = smem + sub_cta * RT_GAUSSIANS_PER_MASK;

    const int32_t mt_global = find_macro_tile(mt_gauss_batch_offsets, total_macro_tiles, batch_idx);
    const int32_t image_idx = mt_global / n_macro_tiles_per_image;
    const int32_t mt_id     = mt_global - image_idx * n_macro_tiles_per_image;

    const int32_t mt_col = mt_id % macro_tile_cols;
    const int32_t mt_row = mt_id / macro_tile_cols;
    const int32_t rt_col = mt_col * MACRO_TILE_WIDTH + (local_tid % MACRO_TILE_WIDTH);
    const int32_t rt_row = mt_row * MACRO_TILE_HEIGHT + (local_tid / MACRO_TILE_WIDTH);
    const bool valid     = (rt_col < rt_tile_cols) && (rt_row < rt_tile_rows);

    const int32_t rt_global = (image_idx * rt_tile_rows + rt_row) * rt_tile_cols + rt_col;
    const int32_t write_pos
        = valid ? rt_gauss_offsets[rt_global] + rt_batch_gauss_offsets[batch_idx * NUM_RT_PER_MT + local_tid]
                : 0; // Placeholder for invalid border RTs; guarded writes never use it.

    const int32_t mt_gauss_start    = mt_gauss_offsets[mt_global];
    const int32_t mt_gauss_end      = mt_gauss_offsets[mt_global + 1];
    const int32_t mt_batch_idx      = batch_idx - mt_gauss_batch_offsets[mt_global];
    const int32_t batch_gauss_start = mt_gauss_start + mt_batch_idx * RT_GAUSS_BATCH_SIZE;
    const int32_t batch_gauss_end   = min(batch_gauss_start + RT_GAUSS_BATCH_SIZE, mt_gauss_end);
    const int32_t n_gauss           = batch_gauss_end - batch_gauss_start;
    const int32_t num_mask_words    = (n_gauss + RT_GAUSSIANS_PER_MASK - 1) / RT_GAUSSIANS_PER_MASK;

    int32_t write_offset = 0;
#pragma unroll 1
    for(int32_t mask_word = 0; mask_word < num_mask_words; ++mask_word)
    {
        const int32_t mask_gauss_start = batch_gauss_start + mask_word * RT_GAUSSIANS_PER_MASK;
        const int32_t mask_gauss_count = min(RT_GAUSSIANS_PER_MASK, batch_gauss_end - mask_gauss_start);

        // Cooperatively load Gaussian IDs into smem.
        if constexpr(RT_GAUSSIANS_PER_MASK <= NUM_RT_PER_MT)
        {
            if(local_tid < mask_gauss_count)
            {
                smem_gid[local_tid] = mt_gauss_ids_sorted[mask_gauss_start + local_tid];
            }
        }
        else
        {
            for(int32_t i = local_tid; i < mask_gauss_count; i += NUM_RT_PER_MT)
            {
                smem_gid[i] = mt_gauss_ids_sorted[mask_gauss_start + i];
            }
        }

        cta_sync<NUM_RT_PER_MT>();

        const int32_t batch_mask_word = batch_idx * RT_MASKS_PER_GAUSS_BATCH + mask_word;
        uint32_t mask                 = rt_bitmasks[batch_mask_word * NUM_RT_PER_MT + local_tid];
        // Iterate set bits of the bitmask and scatter matching IDs.
        while(mask)
        {
            const int32_t j  = __ffs(mask) - 1;
            mask            &= mask - 1;
            if(valid)
            {
                rt_gauss_ids[write_pos + write_offset] = smem_gid[j];
            }
            write_offset++;
        }
    }
}

void launch_rt_binning_fill(
    const int32_t total_gauss_batches,
    const int32_t total_macro_tiles,
    const int32_t n_macro_tiles_per_image,
    const int32_t macro_tile_cols,
    const int32_t rt_tile_size_px,
    const int32_t rt_tile_cols,
    const int32_t rt_tile_rows,
    const at::Tensor &mt_gauss_offsets,
    const at::Tensor &mt_gauss_ids_sorted,
    const at::Tensor &mt_gauss_batch_offsets,
    const at::Tensor &rt_gauss_offsets,
    const at::Tensor &rt_batch_gauss_offsets,
    const at::Tensor &rt_bitmasks,
    at::Tensor &rt_gauss_ids
)
{
    if(total_gauss_batches == 0)
    {
        return;
    }

    const std::size_t smem_bytes = RT_GAUSS_BATCHES_PER_CTA * RT_GAUSSIANS_PER_MASK * sizeof(int32_t);
    const dim3 threads(RT_GAUSS_BATCHES_PER_CTA * MACRO_TILE_WIDTH * MACRO_TILE_HEIGHT);
    const dim3 grid((total_gauss_batches + RT_GAUSS_BATCHES_PER_CTA - 1) / RT_GAUSS_BATCHES_PER_CTA);

    TORCH_CHECK_VALUE(
        SupportedRenderTileSizesPx::contains(rt_tile_size_px), "Unsupported tile_size: ", rt_tile_size_px
    );

    const auto launch_kernel = [&]<typename TileSizeConst>() -> void
    {
        constexpr int TILE_SIZE   = TileSizeConst::value;
        constexpr auto kernel     = rt_binning_fill_kernel<TILE_SIZE>;
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        kernel<<<grid, threads, smem_bytes, stream>>>(
            total_gauss_batches,
            total_macro_tiles,
            n_macro_tiles_per_image,
            macro_tile_cols,
            rt_tile_cols,
            rt_tile_rows,
            mt_gauss_offsets.data_ptr<int32_t>(),
            mt_gauss_ids_sorted.data_ptr<int32_t>(),
            mt_gauss_batch_offsets.data_ptr<int32_t>(),
            rt_gauss_offsets.data_ptr<int32_t>(),
            rt_batch_gauss_offsets.data_ptr<int32_t>(),
            reinterpret_cast<const uint32_t *>(rt_bitmasks.data_ptr<int32_t>()),
            rt_gauss_ids.data_ptr<int32_t>()
        );
    };

    const bool dispatched = dispatch::dispatch(SupportedRenderTileSizesPx{rt_tile_size_px}, std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}
} // namespace gsplat
