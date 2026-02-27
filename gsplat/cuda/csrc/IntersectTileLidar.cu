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
// TODO: Create an bbox abstraction to avoid passing those 4 values around.
bool has_any_rays_in_tile(const int32_t *raycdf, int raycdf_size_el, int raycdf_size_az, int range_min_el, int range_max_el, int range_min_az, int range_max_az)
{
    assert(0 <= range_min_el && range_min_el < range_max_el);
    assert(range_max_el <= raycdf_size_el);
    assert(range_min_az <= range_max_az);

    auto cdf_region_sum = [&](int range_min_el, int range_max_el, int range_min_az, int range_max_az) -> int
    {
        int stride = raycdf_size_el+1;

        return raycdf[range_max_az*stride + range_max_el]
               - raycdf[range_min_az*stride + range_max_el]
               - raycdf[range_max_az*stride + range_min_el]
               + raycdf[range_min_az*stride + range_min_el];
    };

    int region_sum = 0;

    // Gaussian's azimuth covers the whole fov horiz?
    if(range_min_az <= 0 && range_max_az >= raycdf_size_az)
    {
        // TODO: double check if we indeed don't need to check the elevation.
        return true;
    }
    // Entirely to the left of the FOV start
    else if(range_max_az <= 0)
    {
        return false;
    }
    // Entirely to the right of the FOV end
    else if(range_min_az >= raycdf_size_az)
    {
        return false;
    }
    // Fully inside FOV
    else if(range_min_az >= 0 && range_max_az <= raycdf_size_az)
    {
        region_sum =  cdf_region_sum(range_min_el, range_max_el, range_min_az, range_max_az);
    }
    // Wraps at left edge
    // max_az is exclusive, so if it's == 0, the range falls outside FOV.
    else if(range_min_az < 0 && range_max_az > 0)
    {
        region_sum =  cdf_region_sum(range_min_el, range_max_el, 0, range_max_az);
        region_sum += cdf_region_sum(range_min_el, range_max_el, range_min_az + raycdf_size_az, raycdf_size_az);
    }
    // Wraps at right edge
    else if(range_min_az < raycdf_size_az && range_max_az > raycdf_size_az)
    {
        region_sum =  cdf_region_sum(range_min_el, range_max_el, range_min_az, raycdf_size_az);
        region_sum += cdf_region_sum(range_min_el, range_max_el, 0, range_max_az-raycdf_size_az);
    }

    return region_sum > 0;
}

struct LidarSampleTileIdReturn
{
    int idx_el, idx_az;
    int idxdense_el, idxdense_az;
};

__device__
LidarSampleTileIdReturn lidar_sample_tileid(const RowOffsetStructuredSpinningLidarModelParametersExtDevice &lidar, float fov_span_pix_el, float fov_span_pix_az, float rel_pix_el, float rel_pix_az)
{
    float norm_az = rel_pix_az / fov_span_pix_az;
    float norm_el = rel_pix_el / fov_span_pix_el;

    LidarSampleTileIdReturn retval;
    retval.idxdense_az = static_cast<int>(floorf(norm_az*lidar.cdf_resolution_azimuth));
    retval.idxdense_el = static_cast<int>(floorf(norm_el*lidar.cdf_resolution_elevation));
    // Clamp to valid range
    retval.idxdense_el = min(max(retval.idxdense_el, 0), lidar.cdf_resolution_elevation-1);

    retval.idx_az = static_cast<int>(floorf(norm_az*lidar.n_bins_azimuth));
    retval.idx_el = lidar.cdf_elevation[retval.idxdense_el];
    return retval;
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

    const auto [relative_elevation_pix, relative_azimuth_pix]
        = sensor.relative_sensor_angles(elevation_pix, azimuth_pix, lidar.ANGLE_TO_PIXEL_SCALING_FACTOR);

    // Calculate the gaussian extent in tiles and dense tiles (for cdf)

    // TODO: need to transpose it to x==azimuth, y==elevation
    const float min_pix_el = relative_elevation_pix - radius_x;
    const float min_pix_az = relative_azimuth_pix - radius_y;
    const float max_pix_el = relative_elevation_pix + radius_x;
    const float max_pix_az = relative_azimuth_pix + radius_y;

    const LidarSampleTileIdReturn min_sample_tileid = lidar_sample_tileid(lidar, fov_span_pix_el, fov_span_pix_az, min_pix_el, min_pix_az);
    const LidarSampleTileIdReturn max_sample_tileid = lidar_sample_tileid(lidar, fov_span_pix_el, fov_span_pix_az, max_pix_el, max_pix_az);

    // We know that in the relative angle space, elevation and azimuth are
    // monotonically increasing, so we can use this to check if a wrap around
    // the periodic boundary happens. If this happens, we need to unwrap the
    // azimuth tile ID in order to get the correct particle projection result.

    const int min_dense_el = min_sample_tileid.idxdense_el;
    const int min_dense_az = min_sample_tileid.idxdense_az;

    const int max_dense_el = min(max_sample_tileid.idxdense_el+1, min_dense_el+raycdf_size_el);
    const int max_dense_az = min(max_sample_tileid.idxdense_az+1, min_dense_az+raycdf_size_az);

    assert(min_dense_el <= max_dense_el);
    assert(min_dense_az <= max_dense_az);

    if(!has_any_rays_in_tile(lidar.cdf_dense_ray_mask, raycdf_size_el, raycdf_size_az,
                             min_dense_el, max_dense_el, min_dense_az, max_dense_az))
    {
        if (first_pass)
        {
            tiles_per_gauss[idxgauss] = 0;
        }
        return;
    }

    const int tile_min_el = min_sample_tileid.idx_el;
    const int tile_min_az = min_sample_tileid.idx_az;
    const int tile_max_el = min(max_sample_tileid.idx_el+1, tile_min_el+lidar.n_bins_elevation);
    const int tile_max_az = min(max_sample_tileid.idx_az+1, tile_min_az+lidar.n_bins_azimuth);

    assert(tile_min_el <= tile_max_el);
    assert(tile_min_az <= tile_max_az);

    if (first_pass)
    {
        // first pass only writes out tiles_per_gauss
        // TODO: Avoid calling this kernel twice.
        //       Since the order we write out the intersections doesn't matter at this point (they'll be sorted later),
        //       we could use atomic ops to reserve space in the output arrays and write to them freely.
        tiles_per_gauss[idxgauss] = (tile_max_el - tile_min_el) * (tile_max_az - tile_min_az);
        return;
    }

    int64_t image_id;
    if (packed)
    {
        // parallelize over nnz
        image_id = image_ids[idxgauss];
    }
    else
    {
        // parallelize over I * N
        image_id = idxgauss / N;
    }
    const int64_t image_id_enc = image_id << (32 + tile_n_bits);

    // tolerance for negative depth
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
    for (int32_t az = tile_min_az; az < tile_max_az; ++az)
    {
        int mod_az = az%lidar.n_bins_azimuth;
        if(mod_az < 0)
        {
            mod_az += (int)lidar.n_bins_azimuth;
        }
        for (int32_t el = tile_min_el; el < tile_max_el; ++el)
        {
            const int64_t tile_id_enc = mod_az*lidar.n_bins_elevation + el;
            // e.g. tile_n_bits = 22:
            // image id (10 bits) | tile id (22 bits) | depth (32 bits)
            isect_ids[idxflatten] = image_id_enc | (tile_id_enc << 32) | depth_id_enc;
            // the flatten index in [I * N] or [nnz]
            flatten_ids[idxflatten] = idxgauss;
            idxflatten += 1;
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
