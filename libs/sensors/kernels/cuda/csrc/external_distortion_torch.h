/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// TorchScript custom-class wrappers for external distortion types.
//
// Each struct subclasses torch::CustomClassHolder so it can be held in a
// TorchScript Tensor-dict or passed across the Python/C++ boundary as an
// opaque object.  The to_kernel_params() method produces the lightweight
// BivariateWindshieldDistortion_KernelParameters struct that the CUDA kernels
// consume (raw pointer + scalar metadata, no reference-count overhead).

#pragma once

#include "external_distortion_params.h"

#include <ATen/core/Tensor.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/custom_class.h>

namespace gsplat_sensors {

// ===========================================================================
// Constants
// ===========================================================================

// Total number of floats in the packed distortion_coeffs tensor:
// h_poly(6) + v_poly(15) + h_poly_inv(6) + v_poly_inv(15) = 42.
inline constexpr int64_t kBivariateWindshieldCoeffCount = 42;

// ===========================================================================
// No-distortion sentinel
// ===========================================================================

// Identity external distortion — to_kernel_params() returns an empty struct.
struct NoExternalDistortion : public torch::CustomClassHolder {
    NoExternalDistortion_KernelParameters to_kernel_params() const;
};

// ===========================================================================
// Bivariate windshield distortion
// ===========================================================================

// Host-side holder for BivariateWindshieldDistortion parameters.
//
// distortion_coeffs — (42,) float32 CUDA tensor, packed as
//                     [h_poly(6), v_poly(15), h_poly_inv(6), v_poly_inv(15)].
// reference_polynomial — 0 (FORWARD) or 1 (BACKWARD); selects which half of
//                        distortion_coeffs is the distort polynomial.
// h_poly_degree — effective degree of the horizontal polynomial in [0, 2].
// v_poly_degree — effective degree of the vertical polynomial in [0, 4].
struct BivariateWindshieldDistortion : public torch::CustomClassHolder {
    at::Tensor distortion_coeffs;
    int64_t reference_polynomial;
    int64_t h_poly_degree;
    int64_t v_poly_degree;

    BivariateWindshieldDistortion(
        at::Tensor distortion_coeffs,
        int64_t reference_polynomial,
        int64_t h_poly_degree,
        int64_t v_poly_degree);

    // Build the kernel-side parameter pack (raw pointer + scalar metadata).
    // Caller must ensure this object outlives any kernel launch using the result.
    BivariateWindshieldDistortion_KernelParameters to_kernel_params() const;
};

// ===========================================================================
// Validation helper
// ===========================================================================

// Assert all fields of `distortion` satisfy the invariants expected by the
// CUDA kernel:
//   - distortion_coeffs is defined and has exactly kBivariateWindshieldCoeffCount elements.
//   - reference_polynomial is 0 or 1.
//   - h_poly_degree in [0, 2], v_poly_degree in [0, 4].
// Raises via TORCH_CHECK on violation.
void check_bivariate_windshield_distortion(
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& distortion);

} // namespace gsplat_sensors
