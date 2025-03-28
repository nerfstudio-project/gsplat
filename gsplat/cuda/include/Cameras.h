// Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <variant>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

// ---------------------------------------------------------------------------------------------

// Camera-specific types (camera model parameters and returns)

enum class ShutterType {
    ROLLING_TOP_TO_BOTTOM,
    ROLLING_LEFT_TO_RIGHT,
    ROLLING_BOTTOM_TO_TOP,
    ROLLING_RIGHT_TO_LEFT,
    GLOBAL
};

struct RollingShutterParameters {
    // represents two tquat [t,q] poses
    std::array<float, 7 * 2> T_world_sensors; 
    glm::fvec3 t_start;
    glm::fquat q_start;
    glm::fvec3 t_end;
    glm::fquat q_end;
};

struct CameraModelParameters {
    std::array<uint32_t, 2> resolution;
    ShutterType shutter_type;
    std::array<float, 2> principal_point;
    std::array<float, 2> focal_length;
};

struct PerfectPinholeCameraModelParameters : CameraModelParameters {
};

struct OpenCVPinholeCameraModelParameters : CameraModelParameters {
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
    std::array<float, 4> radial_coeffs;
    float max_angle;
};


using CameraModelParametersVariant = std::variant<
    PerfectPinholeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    OpenCVFisheyeCameraModelParameters>;

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

// ---------------------------------------------------------------------------------------------

// Helper variadic template to be able to use lambda expressions in std::visit
template <class... Ts> struct OverloadVisitor : Ts... {
    using Ts::operator()...;
};
template <class... Ts> OverloadVisitor(Ts...) -> OverloadVisitor<Ts...>;
