#pragma once

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
struct BivariateWindshieldModelParameters {

    static constexpr uint8_t MAX_ORDER = 5;
    static constexpr uint8_t MAX_COEFFS = 21;

    at::Tensor horizontal_poly;
    at::Tensor vertical_poly;
    at::Tensor horizontal_poly_inverse;
    at::Tensor vertical_poly_inverse;

    ReferencePolynomialType reference_poly = ReferencePolynomialType::FORWARD;
};

} // namespace gsplat::extdist
