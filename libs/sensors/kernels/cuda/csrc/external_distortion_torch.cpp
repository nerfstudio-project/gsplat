/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Implementation of the TorchScript custom-class wrappers declared in
// external_distortion_torch.h.  This is a host-only (gcc) translation unit;
// it must not include any CUDA device headers.
//
// Registered TorchScript class:
//   gsplat_sensors::BivariateWindshieldDistortion — to_kernel_params() builds
//   the kernel-side parameter struct consumed by the bivariate forward and
//   backward CUDA kernels.

#include "external_distortion_torch.h"

#include <utility>

namespace gsplat_sensors {

// ===========================================================================
// NoExternalDistortion
// ===========================================================================

NoExternalDistortion_KernelParameters NoExternalDistortion::to_kernel_params() const {
    return {};
}

// ===========================================================================
// BivariateWindshieldDistortion
// ===========================================================================

BivariateWindshieldDistortion::BivariateWindshieldDistortion(
    at::Tensor distortion_coeffs_,
    int64_t reference_polynomial_,
    int64_t h_poly_degree_,
    int64_t v_poly_degree_)
    : distortion_coeffs(std::move(distortion_coeffs_)),
      reference_polynomial(reference_polynomial_),
      h_poly_degree(h_poly_degree_),
      v_poly_degree(v_poly_degree_) {}

// Cast int64_t fields to uint32_t for the kernel struct; valid after
// check_bivariate_windshield_distortion() has confirmed range constraints.
BivariateWindshieldDistortion_KernelParameters BivariateWindshieldDistortion::to_kernel_params() const {
    return {
        distortion_coeffs.const_data_ptr<float>(),
        static_cast<uint32_t>(reference_polynomial),
        static_cast<uint32_t>(h_poly_degree),
        static_cast<uint32_t>(v_poly_degree),
    };
}

// ===========================================================================
// Validation
// ===========================================================================

void check_bivariate_windshield_distortion(
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& distortion) {
    TORCH_CHECK(distortion != nullptr, "external_distortion must be BivariateWindshieldDistortion");
    TORCH_CHECK(distortion->distortion_coeffs.defined(), "distortion_coeffs must be defined");
    TORCH_CHECK(
        distortion->distortion_coeffs.numel() == kBivariateWindshieldCoeffCount,
        "distortion_coeffs must have ",
        kBivariateWindshieldCoeffCount,
        " elements");
    TORCH_CHECK(
        distortion->reference_polynomial == 0 || distortion->reference_polynomial == 1,
        "reference_polynomial must be 0 (FORWARD) or 1 (BACKWARD)");
    TORCH_CHECK(
        distortion->h_poly_degree >= 0 && distortion->h_poly_degree <= 2,
        "h_poly_degree must be in [0, 2]");
    TORCH_CHECK(
        distortion->v_poly_degree >= 0 && distortion->v_poly_degree <= 4,
        "v_poly_degree must be in [0, 4]");
}

} // namespace gsplat_sensors
