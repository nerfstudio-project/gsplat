/*
 * SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#include <algorithm>

#include "ExternalDistortion.h"
#include "Common.h"

// ---------------------------------------------------------------------------------------------

namespace gsplat::extdist
{

struct BivariateWindshieldModelDeviceParams;

// NOTE: Some of the device functions herein are marked GSPLAT_NOINLINE to prevent compilation
// combinatorial explosion. e.g. eval_bivariate_poly and distort_camera_ray in ProjectionUT3DGSFused.cu:
// - 7 sigma points (unrolled loop - inside world_gaussian_to_image_gaussian_unscented_transform_shutter_pose)
// - 5 camera models
// - N CUDA architectures/compute capability targets (e.g. 80, 86, 89, 90, 100, 120)
// - Inlining into all contexts creates O(n³) register allocation complexity

inline __host__ __device__ int32_t compute_order(int32_t num_coeffs) {
    // For bivariate polynomial: num_coeffs = (order + 1) * (order + 2) / 2
    // Solve: order^2 + 3*order + 2 - 2*num_coeffs = 0
    // Using quadratic formula: order = (-3 + sqrt(1 + 8*num_coeffs)) / 2
    int32_t sqrt_discriminant = static_cast<int32_t>(std::sqrt(1 + 8 * num_coeffs));
    return (-3 + sqrt_discriminant) / 2;
}

// Evaluate 2D bivariate polynomial at (x, y)
inline __device__ GSPLAT_NOINLINE float eval_bivariate_poly(
    const float* poly_coeffs,
    int32_t order,
    float x,
    float y
) {
    auto const horner_range = [](const float* poly, float x, int32_t idx_start, int32_t idx_end) {
        float result = 0.f;
        for (int32_t idx = idx_end - 1; idx >= idx_start; idx--) {
            result = result * x + poly[idx];
        }
        return result;
    };

    float outer_coeffs[BivariateWindshieldModelParameters::MAX_ORDER + 1]; // 6 outter coeffs with MAX_ORDER=5
    int start_idx = 0;
    for (int32_t inner_order = order; inner_order >= 0; inner_order--) {
        outer_coeffs[order - inner_order] = horner_range(poly_coeffs, x, start_idx, start_idx + inner_order + 1);
        start_idx += inner_order + 1;
    }
    return horner_range(outer_coeffs, y, 0, order + 1);
}

struct BivariateWindshieldModelDeviceParams {

    inline __host__ BivariateWindshieldModelDeviceParams(const BivariateWindshieldModelParameters& params)
    : reference_poly(params.reference_poly)
    , horizontal_poly_order(extdist::compute_order(params.horizontal_poly.size(-1)))
    , vertical_poly_order(extdist::compute_order(params.vertical_poly.size(-1)))
    , horizontal_poly_inverse_order(extdist::compute_order(params.horizontal_poly_inverse.size(-1)))
    , vertical_poly_inverse_order(extdist::compute_order(params.vertical_poly_inverse.size(-1)))
    {
        horizontal_poly_ptr = static_cast<const float*>(params.horizontal_poly.data_ptr());
        vertical_poly_ptr = static_cast<const float*>(params.vertical_poly.data_ptr());
        horizontal_poly_inverse_ptr = static_cast<const float*>(params.horizontal_poly_inverse.data_ptr());
        vertical_poly_inverse_ptr = static_cast<const float*>(params.vertical_poly_inverse.data_ptr());
    }

    ReferencePolynomialType reference_poly = ReferencePolynomialType::FORWARD;

    const float *horizontal_poly_ptr = nullptr;
    const float *vertical_poly_ptr = nullptr;
    const float *horizontal_poly_inverse_ptr = nullptr;
    const float *vertical_poly_inverse_ptr = nullptr;

    int32_t horizontal_poly_order = 0;
    int32_t vertical_poly_order = 0;
    int32_t horizontal_poly_inverse_order = 0;
    int32_t vertical_poly_inverse_order = 0;
};

// ---------------------------------------------------------------------------------------------

// External Distortion Models

struct BivariateWindshieldModel {
    gsplat::extdist::BivariateWindshieldModelDeviceParams params;

    __device__ BivariateWindshieldModel(const gsplat::extdist::BivariateWindshieldModelDeviceParams& device_params)
        : params(device_params)
    {
    }

    static __device__ GSPLAT_NOINLINE glm::fvec3 distort_camera_ray(
        const glm::fvec3& ray,
        const float* horizontal_poly,
        const float* vertical_poly,
        int32_t horizontal_order,
        int32_t vertical_order)
    {
        const float ray_length = glm::length(ray);
        if (ray_length < 1e-6f) return ray;

        const float phi = std::asin(ray.x / ray_length);
        const float theta = std::asin(ray.y / ray_length);

        const float x = std::sin(eval_bivariate_poly(horizontal_poly, horizontal_order, phi, theta));
        const float y = std::sin(eval_bivariate_poly(vertical_poly, vertical_order, phi, theta));
        const float z = std::sqrt(1.f - std::clamp(x * x + y * y, 0.f, 1.f)) * (ray.z < 0.f ? -1.f : 1.f);

        return glm::fvec3(x, y, z);
    }

    __device__ glm::fvec3 distort_camera_ray(glm::fvec3 ray) const {
        return distort_camera_ray(
            ray,
            params.horizontal_poly_ptr,
            params.vertical_poly_ptr,
            params.horizontal_poly_order,
            params.vertical_poly_order
        );
    }

    __device__ glm::fvec3 undistort_camera_ray(glm::fvec3 ray) const {
        return distort_camera_ray(
            ray,
            params.horizontal_poly_inverse_ptr,
            params.vertical_poly_inverse_ptr,
            params.horizontal_poly_inverse_order,
            params.vertical_poly_inverse_order
        );
    }
};

} // namespace gsplat::extdist
