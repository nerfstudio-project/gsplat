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

#include <cstdint>

namespace gsplat
{
inline __device__ int32_t accutile_clamp_int32(int32_t value, int32_t lower, int32_t upper)
{
    return max(lower, min(upper, value));
}

inline __device__ float2
    accutile_ellipse_intersection(float A, float B, float C, float disc, float t, float2 center, bool isY, float coord)
{
    // Ax^2 + 2Bxy + Cy^2 = t, where x and y are relative to the ellipse center.
    // Fix one coordinate and solve the quadratic for the two intersections.
    const float p_u   = isY ? center.y : center.x;
    const float p_v   = isY ? center.x : center.y;
    const float coeff = isY ? A : C;

    const float d         = coord - p_u;
    const float sqrt_term = sqrtf(disc * d * d + t * coeff);

    return {(-B * d - sqrt_term) / coeff + p_v, (-B * d + sqrt_term) / coeff + p_v};
}

// AccuTile-style walk over the SnugBox tile extent: iterate strips along one
// axis and emit tile cells intersected by the ellipse.
template<typename EmitFn>
inline __device__ void accutile_walk_tile_strips(
    float A,
    float B,
    float C,
    float disc,
    float t,
    float2 center,
    float2 bbox_min,
    float2 bbox_max,
    float2 bbox_argmin,
    float2 bbox_argmax,
    int2 tile_rect_begin,
    int2 tile_rect_end,
    int32_t tile_size_x_px,
    int32_t tile_size_y_px,
    int32_t grid_cols,
    bool isY,
    EmitFn emit_tile_id
)
{
    float tile_size_u_px, tile_size_v_px;
    if(isY)
    {
        tile_rect_begin = {tile_rect_begin.y, tile_rect_begin.x};
        tile_rect_end   = {tile_rect_end.y, tile_rect_end.x};
        bbox_min        = {bbox_min.y, bbox_min.x};
        bbox_max        = {bbox_max.y, bbox_max.x};
        bbox_argmin     = {bbox_argmin.y, bbox_argmin.x};
        bbox_argmax     = {bbox_argmax.y, bbox_argmax.x};
        tile_size_u_px  = static_cast<float>(tile_size_y_px);
        tile_size_v_px  = static_cast<float>(tile_size_x_px);
    }
    else
    {
        tile_size_u_px = static_cast<float>(tile_size_x_px);
        tile_size_v_px = static_cast<float>(tile_size_y_px);
    }

    float2 intersection_strip_begin, intersection_strip_end;
    float ellipse_min, ellipse_max;
    float strip_begin, strip_end;

    intersection_strip_end = {bbox_max.y, bbox_min.y};

    strip_begin = tile_rect_begin.x * tile_size_u_px;
    if(bbox_min.x <= strip_begin)
    {
        intersection_strip_begin = accutile_ellipse_intersection(A, B, C, disc, t, center, isY, strip_begin);
    }
    else
    {
        intersection_strip_begin = intersection_strip_end;
    }

    // From here on, coordinates are in the generic (u, v) frame. The int2/float2
    // fields are still named .x/.y, but .x maps to u and .y maps to v.
#pragma unroll 1
    for(int32_t u = tile_rect_begin.x; u < tile_rect_end.x; ++u)
    {
        strip_end = strip_begin + tile_size_u_px;
        if(strip_end <= bbox_max.x)
        {
            intersection_strip_end = accutile_ellipse_intersection(A, B, C, disc, t, center, isY, strip_end);
        }

        if(strip_begin <= bbox_argmin.y && bbox_argmin.y < strip_end)
        {
            // If the min-v tip lies inside this strip, the strip minimum is the
            // SnugBox bound, not either boundary intersection.
            ellipse_min = bbox_min.y;
        }
        else
        {
            ellipse_min = min(intersection_strip_begin.x, intersection_strip_end.x);
        }

        if(strip_begin <= bbox_argmax.y && bbox_argmax.y < strip_end)
        {
            ellipse_max = bbox_max.y;
        }
        else
        {
            ellipse_max = max(intersection_strip_begin.y, intersection_strip_end.y);
        }

        const int32_t min_tile_v = accutile_clamp_int32(
            static_cast<int32_t>(ellipse_min / tile_size_v_px), tile_rect_begin.y, tile_rect_end.y
        );
        const int32_t max_tile_v = accutile_clamp_int32(
            static_cast<int32_t>(ellipse_max / tile_size_v_px + 1), tile_rect_begin.y, tile_rect_end.y
        );

#pragma unroll 1
        for(int32_t v = min_tile_v; v < max_tile_v; v++)
        {
            const int32_t tile_id = isY ? (u * grid_cols + v) : (v * grid_cols + u);
            emit_tile_id(tile_id);
        }

        intersection_strip_begin = intersection_strip_end;
        strip_begin              = strip_end;
    }
}
} // namespace gsplat
