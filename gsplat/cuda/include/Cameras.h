/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>
#include <variant>

#include <ATen/core/ivalue.h>

// ---------------------------------------------------------------------------------------------

// Camera-specific types (camera model parameters and returns)

enum class ShutterType {
    ROLLING_TOP_TO_BOTTOM,
    ROLLING_LEFT_TO_RIGHT,
    ROLLING_BOTTOM_TO_TOP,
    ROLLING_RIGHT_TO_LEFT,
    GLOBAL
};

// ---------------------------------------------------------------------------------------------

// Gaussian-specific types
struct UnscentedTransformParameters : public torch::CustomClassHolder {
    // See Gustafsson and Hendeby 2012 for sigma point parameterization - this
    // default parameter choice is based on
    //
    // - "The unscented Kalman filter for nonlinear estimation" - Wan and van
    // der Merwe 2000
    UnscentedTransformParameters(
        float alpha = 0.1f,
        float beta = 2.f,
        float kappa = 0.f,
        float in_image_margin_factor = 0.1f,
        bool require_all_sigma_points_valid = false
    )
        : alpha(alpha), beta(beta), kappa(kappa),
          in_image_margin_factor(in_image_margin_factor),
          require_all_sigma_points_valid(require_all_sigma_points_valid) {}

    float alpha;
    float beta;
    float kappa;

    // Parameters controlling validity of the unscented transform results
    float in_image_margin_factor; // 10% out of bounds margin is acceptable for
                                  // "valid" projection state
    bool require_all_sigma_points_valid; // true: all sigma points must be valid
                                         // to mark a projection as "valid"
                                         // false: a single valid sigma point is
                                         // sufficient to mark a projection as
                                         // "valid"
};

// FTheta Camera Support
struct FThetaCameraDistortionParameters : public torch::CustomClassHolder {
    static constexpr size_t PolynomialDegree = 6;
    enum class PolynomialType {
        PIXELDIST_TO_ANGLE,
        ANGLE_TO_PIXELDIST,
    };

    FThetaCameraDistortionParameters(
        PolynomialType reference_poly,
        std::array<float, PolynomialDegree> pixeldist_to_angle_poly,
        std::array<float, PolynomialDegree> angle_to_pixeldist_poly,
        float max_angle,
        std::array<float, 3> linear_cde
    )
        : reference_poly(reference_poly),
          pixeldist_to_angle_poly(pixeldist_to_angle_poly),
          angle_to_pixeldist_poly(angle_to_pixeldist_poly),
          max_angle(max_angle), linear_cde(linear_cde) {}

    PolynomialType reference_poly;
    std::array<float, PolynomialDegree> pixeldist_to_angle_poly; // backward polynomial
    std::array<float, PolynomialDegree> angle_to_pixeldist_poly; // forward polynomial
    float max_angle;
    std::array<float, 3> linear_cde;
};
