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

// Kernel-side parameter packs for the bivariate windshield distortion model.
//
// The 42 polynomial coefficients live in a single contiguous float32 tensor;
// these structs carry raw pointers and the per-axis polynomial degrees so
// every kernel thread can read coefficients directly from global memory.

#pragma once

#include <cstdint>

// Empty sentinel — no distortion, no data needed.
struct NoExternalDistortion_KernelParameters {};

// Kernel-visible view of BivariateWindshieldDistortion.
// All pointer members are __restrict__ — the coefficient buffer must not alias
// any other tensor in the same kernel invocation.
struct BivariateWindshieldDistortion_KernelParameters {
    // Pointer into the (42,) flat coefficient tensor.
    // Packed layout: [h_poly(6), v_poly(15), h_poly_inv(6), v_poly_inv(15)].
    // Which 21-element slice is "distort" vs "undistort" is determined at
    // runtime by bivariate_coeff_base() using reference_polynomial.
    const float* __restrict__ distortion_coeffs;
    // 0 = ReferencePolynomial.FORWARD  — h_poly/v_poly are the forward map,
    //                                     h_poly_inv/v_poly_inv are the inverse.
    // 1 = ReferencePolynomial.BACKWARD — roles are swapped.
    uint32_t reference_polynomial;
    // Effective degree of the horizontal polynomial; h_poly_degree in [0, 2].
    // Coefficients beyond triangular(h_poly_degree) are guaranteed zero.
    uint32_t h_poly_degree;
    // Effective degree of the vertical polynomial; v_poly_degree in [0, 4].
    // Coefficients beyond triangular(v_poly_degree) are guaranteed zero.
    uint32_t v_poly_degree;
};
