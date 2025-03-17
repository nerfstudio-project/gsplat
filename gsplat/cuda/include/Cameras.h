// Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <variant>

#if defined(__CUDACC__)
#include <cuda.h>
#endif
#include <glm/fwd.hpp>

// ---------------------------------------------------------------------------------------------

// Camera-specific types (camera model parameters and returns)

enum class ShutterType {
    ROLLING_TOP_TO_BOTTOM,
    ROLLING_LEFT_TO_RIGHT,
    ROLLING_BOTTOM_TO_TOP,
    ROLLING_RIGHT_TO_LEFT,
    GLOBAL
};

struct CameraModelParameters {
    std::array<uint64_t, 2> resolution;
    ShutterType shutter_type;
};

struct OpenCVPinholeCameraModelParameters : CameraModelParameters {
    std::array<float, 2> principal_point;
    std::array<float, 2> focal_length;
    std::array<float, 6> radial_coeffs;
    std::array<float, 2> tangential_coeffs;
    std::array<float, 4> thin_prism_coeffs;

    auto is_perfect_pinhole() const -> bool {
        auto const is_all_zero = [](auto const &arr) {
            return std::all_of(arr.begin(), arr.end(), [](auto const &value) {
                return std::abs(value) < std::numeric_limits<float>::epsilon();
            });
        };

        return is_all_zero(radial_coeffs) && is_all_zero(tangential_coeffs) &&
               is_all_zero(thin_prism_coeffs);
    }
};

struct OpenCVFisheyeCameraModelParameters : CameraModelParameters {
    std::array<float, 2> principal_point;
    std::array<float, 2> focal_length;
    std::array<float, 4> radial_coeffs;
    float max_angle;
};

struct FThetaCameraModelParameters : CameraModelParameters {
    enum class PolynomialType {
        PIXELDIST_TO_ANGLE,
        ANGLE_TO_PIXELDIST,
    };
    std::array<float, 2> principal_point;
    PolynomialType reference_poly;
    static constexpr size_t PolynomialDegree = 6;
    std::array<float, PolynomialDegree>
        pixeldist_to_angle_poly; // backward polynomial
    std::array<float, PolynomialDegree>
        angle_to_pixeldist_poly; // forward polynomial
    float max_angle;
};

struct RollingShutterParameters {
    std::array<float, 7 * 2>
        T_world_sensors; // represents two tquat [t,q] poses
    std::array<int64_t, 2> timestamps_us;
};

using CameraModelParametersVariant = std::variant<
    OpenCVPinholeCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    FThetaCameraModelParameters>;

struct CameraNRE {
    CameraModelParametersVariant camera_model_parameters;
    RollingShutterParameters rolling_shutter_parameters;

    std::tuple<int, int> set_resolution() const {
        auto const &resolution = std::visit(
            [](auto const &params) { return params.resolution; },
            camera_model_parameters
        );
        return {resolution[0], resolution[1]};
    }
};

struct CameraGSplat {
    std::array<uint64_t, 2> resolution;
    // glm::vec3 cam_pos;
    std::array<float, 3> _position;
    // glm::mat4x3 viewmatrix_mat;
    std::array<float, 16> _viewmatrix;
    // glm::mat4x4 projmatrix_mat;
    std::array<float, 16> _projmatrix;
    // glm::mat4x4 inv_viewprojmatrix_mat;
    std::array<float, 16> _inv_viewprojmatrix;
    float tan_fovx;
    float tan_fovy;

    std::tuple<int, int> set_resolution() const {
        return {resolution[0], resolution[1]};
    }

    glm::vec3 position() const;
    glm::mat4x3 viewmatrix() const;
    glm::mat4x4 projmatrix() const;
    glm::mat4x4 inv_viewprojmatrix() const;
};

using CameraInputVariant = std::variant<CameraNRE, CameraGSplat>;

// ---------------------------------------------------------------------------------------------

// Gaussian-specific types
struct UnscentedTransformParameters {
    // See Gustafsson and Hendeby 2012 for sigma point parameterization - this
    // default parameter choice is based on
    //
    // - "The unscented Kalman filter for nonlinear estimation" - Wan and van
    // der Merwe 2000
    float alpha = 0.1;
    float beta = 2.f;
    float kappa = 0.f;

    // Parameters controlling validity of the unscented transform results
    float in_image_margin_factor =
        0.1f; // 10% out of bounds margin is acceptable for "valid" projection
              // state
    bool require_all_sigma_points_valid =
        false; // true: all sigma points must be valid to mark a projection as
               // "valid" false: a single valid sigma point is sufficient to
               // mark a projection as "valid"
};

struct MonteCarloTransformParameters {
    size_t n_samples = 500;

    // Parameters controlling validity of the Monte Carlo transform results
    float in_image_margin_factor =
        0.1f; // 10% out of bounds margin is acceptable for "valid" projection
              // state
    bool require_all_sample_points_valid =
        false; // true: all sample points must be valid to mark a projection as
               // "valid" false: a single valid sample point is sufficient to
               // mark a projection as "valid"
};

// ---------------------------------------------------------------------------------------------

// Helper variadic template to be able to use lambda expressions in std::visit
template <class... Ts> struct OverloadVisitor : Ts... {
    using Ts::operator()...;
};
template <class... Ts> OverloadVisitor(Ts...) -> OverloadVisitor<Ts...>;
