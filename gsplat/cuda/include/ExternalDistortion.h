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

#include <ATen/core/ivalue.h>
#include <ATen/core/Tensor.h>

// ---------------------------------------------------------------------------------------------

namespace gsplat::extdist {

// External Distortion Types
enum class ModelType {
    BIVARIATE_WINDSHIELD = 0,
};

enum class ReferencePolynomialType {
    FORWARD = 1,
    BACKWARD = 2,
};

// Windshield distortion model support
struct BivariateWindshieldModelParameters : public torch::CustomClassHolder {
    BivariateWindshieldModelParameters() {};

    static constexpr uint8_t MAX_ORDER = 5;
    static constexpr uint8_t MAX_COEFFS = 21;

    at::Tensor horizontal_poly;
    at::Tensor vertical_poly;
    at::Tensor horizontal_poly_inverse;
    at::Tensor vertical_poly_inverse;

    ReferencePolynomialType reference_poly = ReferencePolynomialType::FORWARD;
};

} // namespace gsplat::extdist
