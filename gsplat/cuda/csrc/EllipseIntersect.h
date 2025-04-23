#pragma once
#include <cstdint>
#include "Common.h"

namespace gsplat {

// Compute the equation 15: https://arxiv.org/pdf/2412.00578
__device__ inline float2 compute_y_term(
    const float b, // conic.y
    const float c, // conic.z
    const float x_d, // pixel coordinate - mean_x
    const float det, // the determinant of the conic same as the b^2 - ac in the equations in the paper
    const float t // the threshold where opacity * Gaussian = 1 / 255
) {
    const float sqrt_term = sqrtf(det * x_d * x_d + t * c);
    const float b_x_d = b * x_d;
    const float inv_c = 1.0f / c;
    return {
        (-b_x_d - sqrt_term) * inv_c,
        (-b_x_d + sqrt_term) * inv_c
    };
}

// Compute the critical y values from the x term
__device__ inline float2 critical_y_from_x_term(
    const float b, // conic.y
    const float c, // conic.z
    const float x_d, // pixel coordinate - mean_x
    const float det, // the determinant of the conic same as the b^2 - ac in the equations in the paper
    const float t // the threshold where opacity * Gaussian = 1 / 255
)
{
    const float sqrt_term = sqrtf(det * x_d * x_d + t * c);
    const float b_x_d = b * x_d;
    const float inv_c = 1.0f / c;
    return {
        (b_x_d - sqrt_term) * inv_c,
        // (b_x_d + sqrt_term) * inv_c // not used 
        // (-b_x_d - sqrt_term) * inv_c, // not used
        (-b_x_d + sqrt_term) * inv_c
    };
}

__device__ inline uint32_t find_tiles(
    vec2 mean2d,
    vec3 conic,
    const float det,
    const float t,

    float2 bbox_min,
    float2 bbox_max,
    float2 bbox_argmin,
    float2 bbox_argmax,

    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,

    const uint32_t idx,
    uint32_t offset,
    const int64_t depth_id_enc,
    const int64_t cid_enc,
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    const float tile_size_inv = __fdividef(1.0f, tile_size);

    uint2 tile_min = {
        max(0, min((uint32_t)tile_width, (uint32_t)(bbox_min.x * tile_size_inv))),
        max(0, min((uint32_t)tile_height, (uint32_t)(bbox_min.y * tile_size_inv)))
    };
    uint2 tile_max = {
        max(0, min((uint32_t)tile_width, (uint32_t)(bbox_max.x * tile_size_inv + 1))),
        max(0, min((uint32_t)tile_height, (uint32_t)(bbox_max.y * tile_size_inv + 1)))
    };

    const uint32_t y_span = tile_max.y - tile_min.y;
    const uint32_t x_span = tile_max.x - tile_min.x;

    if (y_span * x_span == 0) {
        return 0;
    }

    const bool is_y_major = y_span < x_span;

    if (is_y_major) {
        tile_min = {tile_min.y, tile_min.x};
        tile_max = {tile_max.y, tile_max.x};
        bbox_min = {bbox_min.y, bbox_min.x};
        bbox_max = {bbox_max.y, bbox_max.x};
        bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
        bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
        mean2d = {mean2d.y, mean2d.x};
        conic = {conic.z, conic.y, conic.x}; // Swap x and z components
    }
    // AccuTile
    uint32_t tiles_count = 0;
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line = tile_min.x * tile_size;

    intersect_max_line = {bbox_max.y, bbox_min.y}; // Init

    if (bbox_min.x <= min_line) {
        const float2 y_term = compute_y_term(conic.y, conic.z, tile_min.x * tile_size - mean2d.x, det, t);
        intersect_min_line = {y_term.x + mean2d.y, y_term.y + mean2d.y};
    } else {
        intersect_min_line = intersect_max_line;
    }

    for (uint32_t u = tile_min.x; u < tile_max.x; ++u) {
        const float max_line = min_line + tile_size;

        if (max_line <= bbox_max.x) {
            const float2 y_term = compute_y_term(conic.y, conic.z, max_line - mean2d.x, det, t);
            intersect_max_line = {y_term.x + mean2d.y, y_term.y + mean2d.y};
        }

        ellipse_min = (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) ? bbox_min.y : min(intersect_min_line.x, intersect_max_line.x);
        ellipse_max = (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) ? bbox_max.y : max(intersect_min_line.y, intersect_max_line.y);

        uint32_t min_tile_v = max(tile_min.y, min(tile_max.y, (uint32_t)(ellipse_min * tile_size_inv)));
        uint32_t max_tile_v = min(tile_max.y, max(tile_min.y, (uint32_t)(ellipse_max * tile_size_inv + 1)));

        tiles_count += max_tile_v - min_tile_v;

        if (isect_ids != nullptr && tiles_count > 0) {
            for (uint32_t v = min_tile_v; v < max_tile_v; v++) {
                uint64_t key = is_y_major ?  (u * tile_width + v) : (v * tile_width + u);
                isect_ids[offset] = cid_enc | (key << 32) | depth_id_enc;
                flatten_ids[offset] = idx;
                offset++;
            }
        }

        intersect_min_line = intersect_max_line;
        min_line = max_line;
    }

    return tiles_count;
}

__device__ inline uint32_t find_tiles_touched(
    const vec2 mean2d,
    const vec3 conic,
    const float opacity,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,

    uint32_t idx,
    uint32_t offset,
    int64_t depth_id_enc,
    int64_t cid_enc,

    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
    )
{
    float det = conic.y * conic.y - conic.x * conic.z; // determinant of the conics


    // If ill-formed ellipse, return 0
    if (conic.x <= 0 || conic.z <= 0 || det >= 0) {
        return 0;
    }

    float t = 2.0f * __logf(opacity * 255.0f); // threshold: opacity * Gaussian = 1 / 255

    float x_term = sqrtf(-(conic.y * conic.y * t) / (det * conic.x)); // x_term for y critical The equation 16: https://arxiv.org/pdf/2412.00578
    x_term = (conic.y < 0) ? x_term : -x_term;
    float y_term = sqrtf(-(conic.y * conic.y * t) / (det * conic.z)); // y_term for x critical
    y_term = (conic.y < 0) ? y_term : -y_term;

    float2 bbox_argmin = { 
      mean2d.y - y_term,  // for which y x is the critical point
      mean2d.x - x_term  // for which x y is the critical point
    };
    float2 bbox_argmax = {
      mean2d.y + y_term,  // for which y x is the critical point
      mean2d.x + x_term  // for which x y is the critical point
    };


    float2 critical_y_from_x = critical_y_from_x_term(conic.y, conic.z, x_term, det, t);
    float2 critical_x_from_y = critical_y_from_x_term(conic.y, conic.x, y_term, det, t);

    float2 bbox_min = {
        critical_x_from_y.x + mean2d.x, // x_min
        critical_y_from_x.x + mean2d.y // y_min
    };
    float2 bbox_max = {
        critical_x_from_y.y + mean2d.x, // x_max
        critical_y_from_x.y + mean2d.y // y_max
    };
    return find_tiles(
        mean2d,
        conic,
        det,
        t,
        bbox_min, bbox_max,
        bbox_argmin, bbox_argmax,
        tile_size,
        tile_width,
        tile_height,
        idx, offset,
        depth_id_enc,
        cid_enc,
        isect_ids,
        flatten_ids
    );
}


} // namespace gsplat