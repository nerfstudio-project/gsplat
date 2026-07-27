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

#include <algorithm>
#include <array>
#include <variant>

#include "ExternalDistortion.h"
#include "Common.h"
#include "TypeList.h"

// ---------------------------------------------------------------------------------------------

namespace gsplat::extdist
{
constexpr int32_t compute_order(int32_t num_coeffs)
{
    // MAX_ORDER is so small we can just avoid math and use
    // (order+1)*(order+2)/2 = num_coeffs to inverse check equivalence.
    if(num_coeffs == 1)
    {
        return 0;
    }
    if(num_coeffs == 3)
    {
        return 1;
    }
    if(num_coeffs == 6)
    {
        return 2;
    }
    if(num_coeffs == 10)
    {
        return 3;
    }
    if(num_coeffs == 15)
    {
        return 4;
    }
    if(num_coeffs == 21)
    {
        return BivariateWindshieldModelParameters::MAX_ORDER;
    }
    return -1;
}

// Device-side polynomial evaluation — always iterates over MAX_ORDER.
// Coefficients are expected in MAX_ORDER triangular layout (21 elements, zero-padded).
inline __device__ float eval_bivariate_poly(const float *poly_coeffs, float x, float y)
{
    constexpr int32_t MAX_ORDER = BivariateWindshieldModelParameters::MAX_ORDER;

    float outer_coeffs[MAX_ORDER + 1];
    int32_t start_idx = 0;

#pragma unroll
    for(int32_t inner_order = MAX_ORDER; inner_order >= 0; inner_order--)
    {
        float result = 0.f;
#pragma unroll
        for(int32_t idx = start_idx + inner_order; idx >= start_idx; idx--)
        {
            result = result * x + poly_coeffs[idx];
        }
        outer_coeffs[MAX_ORDER - inner_order]  = result;
        start_idx                             += inner_order + 1;
    }

    float result = 0.f;
#pragma unroll
    for(int32_t idx = MAX_ORDER; idx >= 0; idx--)
    {
        result = result * y + outer_coeffs[idx];
    }
    return result;
}

// Host-side coefficient padding matching the layout for Horner bivariate polynomial evaluation.
//   Group 0 (y^0 term): (order+1) x-coefficients
//   Group 1 (y^1 term): (order)   x-coefficients
//   ...
//   Group k (y^k term): (order-k+1) x-coefficients
//
// For MAX_ORDER layout, each group k has (MAX_ORDER-k+1) elements.
// Original coefficients are placed at the start of each group, zero-padded.
// Groups beyond the original order are all zeros.
inline std::array<float, BivariateWindshieldModelParameters::MAX_COEFFS> pad_coefficients_to_max_order(
    const float *src, int32_t src_order
)
{
    constexpr int32_t MAX_ORDER  = BivariateWindshieldModelParameters::MAX_ORDER;
    constexpr int32_t MAX_COEFFS = BivariateWindshieldModelParameters::MAX_COEFFS;

    std::array<float, MAX_COEFFS> dst{};

    int32_t src_offset = 0;
    int32_t dst_offset = 0;
    for(int32_t k = 0; k <= MAX_ORDER; k++)
    {
        int32_t dst_group_size = MAX_ORDER - k + 1;
        int32_t src_group_size = (k <= src_order) ? (src_order - k + 1) : 0;
        std::copy_n(src + src_offset, src_group_size, dst.data() + dst_offset);
        src_offset += src_group_size;
        dst_offset += dst_group_size;
    }
    return dst;
}

// Get a padded MAX_COEFFS coefficient array from a tensor.
inline std::array<float, BivariateWindshieldModelParameters::MAX_COEFFS> pad_tensor_coefficients(
    const at::Tensor &tensor
)
{
    auto tensor_contig = tensor.contiguous().cpu();
    const float *src   = tensor_contig.data_ptr<float>();
    int32_t src_order  = compute_order(static_cast<int32_t>(tensor_contig.numel()));
    TORCH_CHECK(
        src_order >= 0,
        "Invalid number of bivariate polynomial coefficients: ",
        tensor_contig.numel(),
        ". Expected triangular number: 1, 3, 6, 10, 15, or 21."
    );
    return pad_coefficients_to_max_order(src, src_order);
}

// Device-side parameters for the bivariate windshield distortion model.
// Stores polynomial coefficients as fixed-size arrays (zero-padded to MAX_ORDER layout).
struct BivariateWindshieldModelDeviceParams
{
    inline __host__ __device__ BivariateWindshieldModelDeviceParams() = default;

    inline __host__ BivariateWindshieldModelDeviceParams(const BivariateWindshieldModelParameters &params)
        : horizontal_poly(pad_tensor_coefficients(params.horizontal_poly))
        , vertical_poly(pad_tensor_coefficients(params.vertical_poly))
        , horizontal_poly_inverse(pad_tensor_coefficients(params.horizontal_poly_inverse))
        , vertical_poly_inverse(pad_tensor_coefficients(params.vertical_poly_inverse))
    {
    }

    std::array<float, BivariateWindshieldModelParameters::MAX_COEFFS> horizontal_poly         = {};
    std::array<float, BivariateWindshieldModelParameters::MAX_COEFFS> vertical_poly           = {};
    std::array<float, BivariateWindshieldModelParameters::MAX_COEFFS> horizontal_poly_inverse = {};
    std::array<float, BivariateWindshieldModelParameters::MAX_COEFFS> vertical_poly_inverse   = {};
};

// ---------------------------------------------------------------------------------------------

// External Distortion Models

struct BivariateWindshieldModel
{
    using KernelParameters = gsplat::extdist::BivariateWindshieldModelDeviceParams;

    gsplat::extdist::BivariateWindshieldModelDeviceParams params;

    __device__ BivariateWindshieldModel(const KernelParameters &device_params, int camera_index)
        : params(device_params)
    {
    }

    static __device__ glm::fvec3 distort_camera_ray(
        const glm::fvec3 &ray, const float *horizontal_poly, const float *vertical_poly
    )
    {
        const float ray_length = glm::length(ray);
        if(ray_length < 1e-6f)
        {
            return ray;
        }

        // Clamp to [-1, 1] before asin: mathematically |ray.x / ray_length| <= 1,
        // but --use_fast_math can introduce rounding that pushes the ratio past
        // 1.0 by an ULP, which makes asin return NaN.
        const float phi   = std::asin(std::clamp(ray.x / ray_length, -1.f, 1.f));
        const float theta = std::asin(std::clamp(ray.y / ray_length, -1.f, 1.f));

        const float x = std::sin(eval_bivariate_poly(horizontal_poly, phi, theta));
        const float y = std::sin(eval_bivariate_poly(vertical_poly, phi, theta));
        const float z = std::sqrt(1.f - std::min(x * x + y * y, 1.f)) * (ray.z < 0.f ? -1.f : 1.f);

        return glm::fvec3(x, y, z);
    }

    // Distort a camera ray using forward polynomials.
    __device__ glm::fvec3 distort_camera_ray(glm::fvec3 ray) const
    {
        return distort_camera_ray(ray, params.horizontal_poly.data(), params.vertical_poly.data());
    }

    __device__ glm::fvec3 undistort_camera_ray(glm::fvec3 ray) const
    {
        return distort_camera_ray(ray, params.horizontal_poly_inverse.data(), params.vertical_poly_inverse.data());
    }
};

struct EmptyExternalDistortionModel
{
    struct KernelParameters
    {
    };

    __device__ EmptyExternalDistortionModel(const KernelParameters &kernel_parameters, int camera_index) { }

    __device__ glm::fvec3 distort_camera_ray(glm::fvec3 ray) const
    {
        return ray;
    }

    __device__ glm::fvec3 undistort_camera_ray(glm::fvec3 ray) const
    {
        return ray;
    }
};

// Type list of all external distortion models
using ExternalDistortionModelTypes = TypeList<EmptyExternalDistortionModel, BivariateWindshieldModel>;

using ExternalDistortionModelKernelParamsVariant = TypeListToKernelParamsVariant<ExternalDistortionModelTypes>;

// Map a KernelParameters type back to its distortion model type.
template<typename KP>
using DistortionModelFromKernelParams = FindByKernelParams<KP, ExternalDistortionModelTypes>;
} // namespace gsplat::extdist
