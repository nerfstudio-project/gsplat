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

/**
 * @file ExternalDistortionWrappers.h
 * @brief Host-callable wrappers for external distortion CUDA functions.
 *
 * These wrappers expose the device-only functions in ExternalDistortion.cuh
 * to Python for testing. Guarded by GSPLAT_BUILD_CAMERA_WRAPPERS.
 */

#pragma once

#if GSPLAT_BUILD_CAMERA_WRAPPERS

#    include <torch/extension.h>
#    include "ExternalDistortion.h"
#    include "TorchUtils.h"

namespace gsplat::extdist
{
/**
 * @brief Distort or undistort camera rays using the bivariate windshield model.
 *
 * Launches a CUDA kernel that calls BivariateWindshieldModel::distort_camera_ray()
 * on each input ray.
 *
 * @param rays Input rays [N, 3] (float32, CUDA)
 * @param params Bivariate windshield model parameters (polynomial tensors must be CUDA)
 * @param inverse If false, apply forward distortion. If true, apply inverse (undistort).
 * @return Distorted rays [N, 3] (float32, CUDA)
 */
torch::Tensor distort_camera_rays(
    const torch::Tensor &rays, const BivariateWindshieldModelParameters &params, bool inverse
);

/**
 * @brief Evaluate a 2D bivariate polynomial at (x, y) points.
 *
 * Launches a CUDA kernel that calls eval_bivariate_poly() on each input pair.
 *
 * @param x Input x values [N] (float32, CUDA)
 * @param y Input y values [N] (float32, CUDA)
 * @param poly_coeffs Polynomial coefficients [num_coeffs] (float32, CUDA)
 * @param order Polynomial order
 * @return Evaluated values [N] (float32, CUDA)
 */
torch::Tensor eval_bivariate_poly_wrapper(
    const torch::Tensor &x, const torch::Tensor &y, const torch::Tensor &poly_coeffs, int64_t order
);
} // namespace gsplat::extdist

namespace gsplat
{
// Marshal the windshield distortion params across the dispatcher so the op takes
// the struct directly: it crosses as its four polynomial tensors plus the
// reference-polynomial enum, matching the registered schema.
template<>
struct TorchArgDef<extdist::BivariateWindshieldModelParameters>
{
    static auto to(const extdist::BivariateWindshieldModelParameters &p)
    {
        return to_torch_args(
            p.horizontal_poly, p.vertical_poly, p.horizontal_poly_inverse, p.vertical_poly_inverse, p.reference_poly
        );
    }

    template<class... Ts>
    static extdist::BivariateWindshieldModelParameters from(std::tuple<Ts...> d)
    {
        extdist::BivariateWindshieldModelParameters params;
        params.horizontal_poly         = std::get<0>(d);
        params.vertical_poly           = std::get<1>(d);
        params.horizontal_poly_inverse = std::get<2>(d);
        params.vertical_poly_inverse   = std::get<3>(d);
        params.reference_poly          = static_cast<extdist::ReferencePolynomialType>(std::get<4>(d));
        return params;
    }
};
} // namespace gsplat

#endif // GSPLAT_BUILD_CAMERA_WRAPPERS
