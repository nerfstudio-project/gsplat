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

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

// for CUB_WRAPPER
#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>

#include "Common.h"
#include "Intersect.h"
#include "Utils.cuh"
#include "Ops.h"
#include "Cameras.cuh"
#include "Lidars.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

__device__
int cdf_region_sum(const int32_t *raycdf, int raycdf_stride, int range_min_el, int range_max_el, int range_min_az, int range_max_az)
{
    return raycdf[range_max_az*raycdf_stride + range_max_el]
           - raycdf[range_min_az*raycdf_stride + range_max_el]
           - raycdf[range_max_az*raycdf_stride + range_min_el]
           + raycdf[range_min_az*raycdf_stride + range_min_el];
};

__device__
bool has_any_rays_in_tile(
    const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar,
    int raycdf_size_el,
    int raycdf_size_az,
    int range_min_el,
    int range_max_el,
    int range_min_az,
    int range_max_az
)
{
    if (range_min_az >= range_max_az)
        return false;

    assert(0 <= range_min_az && range_min_az <= raycdf_size_az);
    assert(0 <= range_max_az && range_max_az <= raycdf_size_az);

    // Optimization in case of full cover, assumes there are always rays.
    if (range_min_az <= 0 && range_max_az >= raycdf_size_az)
        return true;

    const int raycdf_stride = raycdf_size_el + 1;
    const int *raycdf = lidar.cdf_dense_ray_mask;
    const int num_rays = cdf_region_sum(raycdf, raycdf_stride, range_min_el, range_max_el, range_min_az, range_max_az);
    return num_rays > 0;
}

constexpr struct { __device__ float operator()(float x) const { return floorf(x); } } RoundFloor;
constexpr struct { __device__ float operator()(float x) const { return ceilf(x); }  } RoundCeil;

namespace {
    template <auto RoundFn>
    __device__ int sample_dense_az(float pix_az, float fov_span_pix_az, int cdf_resolution)
    {
        // NOTE: need to use proper fdiv instead of the approximation enabled with -use_fast_math
        // So that if pix_az == fov_span_pix_az, the index is exactly cdf_resolution.
        // It also makes the results match the reference implementation.
        int idx = static_cast<int>(RoundFn(__fdiv_rn(pix_az, fov_span_pix_az) * cdf_resolution));
        assert(0 <= idx);
        assert(idx <= cdf_resolution);
        return idx;
    }

    template <auto RoundFn>
    __device__ int sample_dense_el(float pix_el, float fov_span_pix_el, int cdf_resolution)
    {
        int idx = static_cast<int>(RoundFn(__fdiv_rn(pix_el, fov_span_pix_el) * cdf_resolution));
        assert(0 <= idx);
        assert(idx <= cdf_resolution);
        return idx;
    }

    template <auto RoundFn>
    __device__ int sample_tile_az(float pix_az, float fov_span_pix_az, int n_bins)
    {
        int idx = static_cast<int>(RoundFn(__fdiv_rn(pix_az, fov_span_pix_az) * n_bins));
        assert(0 <= idx);
        assert(idx <= n_bins);
        return idx;
    }

    __device__ inline int sample_tile_el(int dense_el, const int *cdf_elevation, int elevation_cdf_resolution)
    {
        assert(0 <= dense_el);
        assert(dense_el <= elevation_cdf_resolution);
        return cdf_elevation[dense_el];
    }
}

template <typename scalar_t>
__global__ void intersect_tile_lidar_kernel(
    // if the data is [...,  N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over I * N, only used if packed is False
    const uint32_t I,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    RowOffsetStructuredSpinningLidarModel sensor,
    const int64_t *__restrict__ image_ids,    // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const scalar_t *__restrict__ means2d,            // [..., N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [..., N, 2] or [nnz, 2]
    const scalar_t *__restrict__ depths,             // [..., N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [..., N] or [nnz]
    const uint32_t tile_n_bits,
    const uint32_t image_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [..., N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // parallelize over I * N.
    uint32_t idxgauss = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idxgauss >= (packed ? nnz : I * N)) {
        return;
    }

    const float radius_x = radii[idxgauss * 2];
    const float radius_y = radii[idxgauss * 2 + 1];
    if (radius_x <= 0 || radius_y <= 0)
    {
        if (first_pass)
        {
            tiles_per_gauss[idxgauss] = 0;
        }
        return;
    }

    const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar = sensor.parameters.lidar;

    const int raycdf_size_az = lidar.cdf_resolution_azimuth;
    const int raycdf_size_el = lidar.cdf_resolution_elevation;

    vec2 mean2d = glm::make_vec2(means2d + 2*idxgauss);
    // TODO: need to transpose it to x==azimuth, y==elevation when mean2d.x/y are transposed upstream
    const float elevation_pix = mean2d.x;
    const float azimuth_pix = mean2d.y;

    const float fov_span_pix_el = lidar.fov_vert_rad.span * lidar.ANGLE_TO_PIXEL_SCALING_FACTOR;
    const float fov_span_pix_az = lidar.fov_horiz_rad.span * lidar.ANGLE_TO_PIXEL_SCALING_FACTOR;
    const float full_circle_pix = 2*PI*lidar.ANGLE_TO_PIXEL_SCALING_FACTOR;

    // Compute relative angles once on the mean.
    const auto [mean_rel_el, mean_rel_az]
        = sensor.relative_sensor_angles(elevation_pix, azimuth_pix, lidar.ANGLE_TO_PIXEL_SCALING_FACTOR);

    // Now calculate the gaussian range
    const float beg_az = mean_rel_az - radius_y;
    const float end_az_raw = mean_rel_az + radius_y;
    const float beg_el = min(max(mean_rel_el - radius_x, 0.f), fov_span_pix_el);
    const float end_el = min(max(mean_rel_el + radius_x, 0.f), fov_span_pix_el);

    // Check full_cover before capping to 2*pi, since the cap can push end
    // below fov_span for wide FOVs even though the gaussian covers the full FOV.
    const bool full_cover = (beg_az <= 0.f) && (end_az_raw >= fov_span_pix_az);

    // Don't let gaussian size to be more than 2*pi
    const float end_az = min(end_az_raw, beg_az + full_circle_pix);

    const bool underflows = (beg_az < 0.f) && !full_cover;
    const bool overflows  = (end_az > full_circle_pix) && !full_cover;

    // Compute region A's pixel range.
    // full_cover:  A = [0, fov_span)
    // underflows:  A = [0, end)
    // overflows:   A = [beg, fc)
    // inside:      A = [beg, end)
    float begA_pix_az, endA_pix_az;
    if (full_cover)
    {
        begA_pix_az = 0.f;
        endA_pix_az = fov_span_pix_az;
    }
    else if (underflows)
    {
        begA_pix_az = 0.f;
        endA_pix_az = end_az;
    }
    else if (overflows)
    {
        begA_pix_az = beg_az;
        endA_pix_az = full_circle_pix;
    }
    else
    {
        begA_pix_az = beg_az;
        endA_pix_az = end_az;
    }

    // Clamp pixel regions to [0, fov_span] so that the sampling functions
    // normalize to [0, 1] and produce in-range indices.
    begA_pix_az = min(max(begA_pix_az, 0.f), fov_span_pix_az);
    endA_pix_az = min(max(endA_pix_az, 0.f), fov_span_pix_az);

    const int n_bins_az = (int)lidar.n_bins_azimuth;

    // Sample elevation (shared by A and B).
    const int min_dense_el = sample_dense_el<RoundFloor>(beg_el, fov_span_pix_el, raycdf_size_el);
    const int max_dense_el = sample_dense_el<RoundCeil>(end_el, fov_span_pix_el, raycdf_size_el);
    if (min_dense_el >= max_dense_el)
    {
        if (first_pass)
        {
            tiles_per_gauss[idxgauss] = 0;
        }
        return;
    }
    // [min_dense_el,max_dense_el) is a half-open range.
    // Make sure that [tile_min_el,tile_max_el) also is.
    // - If min_dense_el==max_dense_el -> tile_min_el==tile_max_el (no tiles)
    // - If min_dense_el<max_dense_el -> tile_min_el<tile_max_el. (at least one tile)
    const int tile_min_el = sample_tile_el(min_dense_el, lidar.cdf_elevation, raycdf_size_el);
    assert(max_dense_el >= 1);
    const int tile_max_el = min(sample_tile_el(max_dense_el - 1, lidar.cdf_elevation, raycdf_size_el) + 1,
                                (int)lidar.n_bins_elevation);

    // Sample A azimuth.
    const int begA_dense = sample_dense_az<RoundFloor>(begA_pix_az, fov_span_pix_az, raycdf_size_az);
    const int endA_dense = sample_dense_az<RoundCeil>(endA_pix_az, fov_span_pix_az, raycdf_size_az);
    const bool has_raysA = has_any_rays_in_tile(lidar, raycdf_size_el, raycdf_size_az,
                                                min_dense_el, max_dense_el, begA_dense, endA_dense);

    int begA_tile = 0, endA_tile = 0;
    if(has_raysA)
    {
        begA_tile = sample_tile_az<RoundFloor>(begA_pix_az, fov_span_pix_az, n_bins_az);
        endA_tile = sample_tile_az<RoundCeil>(endA_pix_az, fov_span_pix_az, n_bins_az);
    }

    // Sample B azimuth -- only exists for underflows or overflows.
    // underflows: B = [beg+fc, fc),  overflows: B = [0, end-fc)
    bool has_raysB = false;
    int begB_tile = 0, endB_tile = 0;
    if (underflows || overflows)
    {
        const float begB_pix_az = min(max(underflows ? (beg_az + full_circle_pix) : 0.f, 0.f), fov_span_pix_az);
        const float endB_pix_az = min(max(underflows ? full_circle_pix : (end_az - full_circle_pix), 0.f), fov_span_pix_az);

        const int begB_dense = sample_dense_az<RoundFloor>(begB_pix_az, fov_span_pix_az, raycdf_size_az);
        const int endB_dense = sample_dense_az<RoundCeil>(endB_pix_az, fov_span_pix_az, raycdf_size_az);
        has_raysB = has_any_rays_in_tile(lidar, raycdf_size_el, raycdf_size_az,
                                         min_dense_el, max_dense_el, begB_dense, endB_dense);
        if(has_raysB)
        {
            begB_tile = sample_tile_az<RoundFloor>(begB_pix_az, fov_span_pix_az, n_bins_az);
            endB_tile = sample_tile_az<RoundCeil>(endB_pix_az, fov_span_pix_az, n_bins_az);
        }
    }

    if (!has_raysA && !has_raysB)
    {
        if (first_pass)
        {
            tiles_per_gauss[idxgauss] = 0;
        }
        return;
    }

    const bool periodic_az = fov_span_pix_az >= full_circle_pix;

    int az_ranges[2][2] = {{0, 0}, {0, 0}};
    if (has_raysA)
    {
        az_ranges[0][0] = begA_tile;
        az_ranges[0][1] = endA_tile;
    }
    if (has_raysB)
    {
        az_ranges[1][0] = begB_tile;
        az_ranges[1][1] = endB_tile;
    }

    if (periodic_az)
    {
        // For periodic azimuth, tiles wrap around (tile n_bins-1 is adjacent
        // to tile 0). Merge B into A by extending A across the 0/n_bins seam:
        //   underflows: A=[0, endA), B=[begB, n_bins) -> merged=[begB-n_bins, endA)
        //   overflows:  A=[begA, n_bins), B=[0, endB) -> merged=[begA, endB+n_bins)
        // The kernel then uses az % n_bins to map back to valid tile indices.
        if (has_raysB && underflows)
            az_ranges[0][0] = az_ranges[1][0] - n_bins_az;
        if (has_raysB && overflows)
            az_ranges[0][1] = az_ranges[1][1] + n_bins_az;
        // Cap to at most n_bins wide to prevent double-counting tiles at the
        // seam when ceil(endA) and floor(begB) produce overlapping tile indices.
        az_ranges[0][0] = max(az_ranges[0][0], az_ranges[0][1] - n_bins_az);
        az_ranges[0][1] = min(az_ranges[0][1], az_ranges[0][0] + n_bins_az);
        az_ranges[1][0] = 0;
        az_ranges[1][1] = 0;
    }
    else
    {
        // For non-periodic azimuth, A and B are generally disjoint (e.g.,
        // behind-sensor gaussians with tips at both edges of the FOV).
        // However, when extent is very close to pi, ceil(endA) and floor(begB)
        // can produce overlapping tile indices. Since one range always anchors
        // at 0 and the other at n_bins, any overlap means the gaussian covers
        // the entire FOV. Merge into [0, n_bins) in that case.
        if (has_raysA && has_raysB
            && az_ranges[1][0] < az_ranges[0][1]
            && az_ranges[0][0] < az_ranges[1][1])
        {
            az_ranges[0][0] = 0;
            az_ranges[0][1] = n_bins_az;
            az_ranges[1][0] = 0;
            az_ranges[1][1] = 0;
        }
    }

    assert(tile_min_el <= tile_max_el);

    if (first_pass)
    {
        const int az_span = max(az_ranges[0][1] - az_ranges[0][0], 0)
                          + max(az_ranges[1][1] - az_ranges[1][0], 0);
        tiles_per_gauss[idxgauss] = (tile_max_el - tile_min_el) * az_span;
        return;
    }

    int64_t image_id;
    if (packed)
    {
        image_id = image_ids[idxgauss];
    }
    else
    {
        image_id = idxgauss / N;
    }
    const int64_t image_id_enc = image_id << (32 + tile_n_bits);

    int64_t depth_id_enc;
    if constexpr(sizeof(depths[idxgauss]) == sizeof(uint32_t))
    {
        depth_id_enc = reinterpret_cast<const uint32_t &>(depths[idxgauss]);
    }
    else
    {
        static_assert(sizeof(depths[idxgauss]) == sizeof(uint64_t));
        depth_id_enc = reinterpret_cast<const uint64_t &>(depths[idxgauss]);
        assert((depth_id_enc & ~0xFF'FF'FF'FFULL) == 0);
        depth_id_enc &= 0xFF'FF'FF'FFULL;
    }

    int64_t idxflatten = (idxgauss == 0) ? 0 : cum_tiles_per_gauss[idxgauss - 1];
    #pragma unroll
    for (int r = 0; r < 2; ++r)
    {
        for (int32_t az = az_ranges[r][0]; az < az_ranges[r][1]; ++az)
        {
            const int32_t actual_az = periodic_az ? ((az + n_bins_az) % n_bins_az) : az;
            for (int32_t el = tile_min_el; el < tile_max_el; ++el)
            {
                const int64_t tile_id_enc = actual_az*lidar.n_bins_elevation + el;
                isect_ids[idxflatten] = image_id_enc | (tile_id_enc << 32) | depth_id_enc;
                flatten_ids[idxflatten] = idxgauss;
                idxflatten += 1;
            }
        }
    }
    assert(idxflatten == cum_tiles_per_gauss[idxgauss]);
}

void launch_intersect_tile_lidar_kernel(
    // inputs
    const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &lidar,
    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [..., N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [..., N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids      // [n_isects]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N, nnz;
    int64_t n_elements;
    if (packed) {
        nnz = means2d.size(0); // total number of gaussians
        n_elements = nnz;
    } else {
        nnz = 0; // Won't be used, but at least initialize it.
        N = means2d.size(-2); // number of gaussians per image
        n_elements = I * N;
    }

    uint32_t n_tiles = lidar->n_bins_azimuth*lidar->n_bins_elevation;
    // the number of bits needed to encode the image id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t image_n_bits = std::bit_width(I);
    uint32_t image_n_bits = I == 0 ? 0 : ((uint32_t)floor(log2(I)) + 1);
    uint32_t tile_n_bits = n_tiles == 0 ? 0 : ((uint32_t)floor(log2(n_tiles)) + 1);
    // the first 32 bits are used for the image id and tile id altogether, so
    // check if we have enough bits for them.
    assert(image_n_bits + tile_n_bits <= 32);

    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means2d.scalar_type(),
        "intersect_tile_lidar_kernel",
        [&]() {
            intersect_tile_lidar_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    packed,
                    I,
                    N,
                    nnz,
                    RowOffsetStructuredSpinningLidarModel{*lidar},
                    image_ids.has_value()
                        ? image_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    gaussian_ids.has_value()
                        ? gaussian_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    means2d.data_ptr<scalar_t>(),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    cum_tiles_per_gauss.has_value()
                        ? cum_tiles_per_gauss.value().data_ptr<int64_t>()
                        : nullptr,
                    tile_n_bits,
                    image_n_bits,
                    tiles_per_gauss.has_value()
                        ? tiles_per_gauss.value().data_ptr<int32_t>()
                        : nullptr,
                    isect_ids.has_value()
                        ? isect_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    flatten_ids.has_value()
                        ? flatten_ids.value().data_ptr<int32_t>()
                        : nullptr
                );
        }
    );
}

} // namespace gsplat
