// Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

#pragma once

#include <array>
#include <cmath>
#include <limits>

#include <thrust/random.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Silence warnings / errors of the form
//
// __device__ / __host__ annotation is ignored on a function("XXX") that is explicitly defaulted on its first declaration
//
// in GLM
#define GLM_ENABLE_EXPERIMENTAL
#pragma nv_diag_suppress = esa_on_defaulted_function_ignored
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/matrix_operation.hpp> // needs define GLM_ENABLE_EXPERIMENTAL
#pragma nv_diag_default = esa_on_defaulted_function_ignored

#include "cameras.h"

#include "auxiliary.h"

// ---------------------------------------------------------------------------------------------

// quaternion : apply to point

// TODO[janickm]: use glm's quaternion math directly
inline __device__ __host__ glm::fvec3 apply_quaternion(glm::fquat const& q, glm::fvec3 const& p) {
    const glm::fmat3 R = glm::mat3_cast(q);
    return R * p;
}

inline __device__ __host__ glm::fvec3 apply_quaternion(glm::fvec4 const& q, glm::fvec3 const& p) {
    // Quaternion rotation matrix coefficients
    auto const xx = q.x * q.x;
    auto const yy = q.y * q.y;
    auto const zz = q.z * q.z;
    auto const xy = q.x * q.y;
    auto const xz = q.x * q.z;
    auto const yz = q.y * q.z;
    auto const wx = q.w * q.x;
    auto const wy = q.w * q.y;
    auto const wz = q.w * q.z;

    // Apply quaternion rotation to point
    auto const &p_x = p.x, p_y = p.y, p_z = p.z;
    auto const x = p_x * (1 - 2 * yy - 2 * zz) + p_y * (2 * xy - 2 * wz) + p_z * (2 * xz + 2 * wy);
    auto const y = p_x * (2 * xy + 2 * wz) + p_y * (1 - 2 * xx - 2 * zz) + p_z * (2 * yz - 2 * wx);
    auto const z = p_x * (2 * xz - 2 * wy) + p_y * (2 * yz + 2 * wx) + p_z * (1 - 2 * xx - 2 * yy);

    return {x, y, z};
}

// quaternion : slerp interpolation

inline __device__ __host__ glm::fvec4 unitquat_slerp(glm::fvec4 const& q_start, glm::fvec4 q_end, float t) {
    // omega is the 'angle' between both quaternions
    auto cos_omega = glm::dot(q_start, q_end);

    // flip quaternions with negative angle to perform shortest arc interpolation
    if (cos_omega < 0.0f) {
        cos_omega *= -1.f;
        q_end *= -1.f;
    }

    // true if q_start and q_end are close
    auto const nearby_quaternions = cos_omega > (1.0f - 1e-3);

    // General approach (use linear interpolation for nearby quaternions)
    auto const omega = std::acos(cos_omega);
    auto const alpha = nearby_quaternions ? (1.f - t) : std::sin((1.f - t) * omega);
    auto const beta  = nearby_quaternions ? t : std::sin(t * omega);

    // Interpolate
    auto const ret = glm::normalize(alpha * q_start + beta * q_end);

    return ret;
}

// ---------------------------------------------------------------------------------------------

// Math helpers (polynomial evaluation / stable norms)

inline __device__ float numerically_stable_norm2(float x, float y) {
    // Computes 2-norm of a [x,y] vector in a numerically stable way
    auto const abs_x = std::fabs(x);
    auto const abs_y = std::fabs(y);
    auto const min   = std::fmin(abs_x, abs_y);
    auto const max   = std::fmax(abs_x, abs_y);

    if (max <= 0.f)
        return 0.f;

    auto const min_max_ratio = min / max;
    return max * std::sqrt(1.f + min_max_ratio * min_max_ratio);
}

template <size_t N_COEFFS>
inline __device__ float eval_poly_horner(std::array<float, N_COEFFS> const& poly, float x) {
    // Evaluates a polynomial y=f(x) with
    //
    // f(x) = c_0*x^0 + c_1*x^1 + c_2*x^2 + c_3*x^3 + c_4*x^4 ...
    //
    // given by poly_coefficients c_i at points x using numerically stable Horner scheme.
    //
    // The degree of the polynomial is N_COEFFS - 1

    auto y = float{0};
    for (auto cit = poly.rbegin(); cit != poly.rend(); ++cit)
        y = x * y + (*cit);
    return y;
}

template <size_t N_COEFFS>
inline __device__ float eval_poly_odd_horner(std::array<float, N_COEFFS> const& poly_odd, float x) {
    // Evaluates an odd-only polynomial y=f(x) with
    //
    // f(x) = c_0*x^1 + c_1*x^3 + c_2*x^5 + c_3*x^7 + c_4*x^9 ...
    //
    // given by poly_coefficients c_i at points x using numerically stable Horner scheme.
    //
    // The degree of the polynomial is 2*N_COEFFS - 1

    return x * eval_poly_horner(poly_odd, x * x); // evaluate x^2-based "regular" polynomial after facting out one x term
}

template <size_t N_COEFFS>
inline __device__ float eval_poly_even_horner(std::array<float, N_COEFFS> const& poly_even, float x) {
    // Evaluates an even-only polynomial y=f(x) with
    //
    // f(x) = c_0 + c_1*x^2 + c_2*x^4 + c_3*x^6 + c_4*x^8 ...
    //
    // given by poly_coefficients c_i at points x using numerically stable Horner scheme.
    //
    // The degree of the polynomial is 2*(N_COEFFS - 1)

    return eval_poly_horner(poly_even, x * x); // evaluate x^2-substituted "regular" polynomial
}

// Enum to represent the type of polynomial
enum class PolynomialType {
    FULL, // Represents a full polynomial with all terms
    EVEN, // Represents an even-only polynomial
    ODD   // Represents an odd-only polynomial
};

template <PolynomialType POLYNOMIAL_TYPE, size_t N_COEFFS>
struct PolynomialProxy {
    std::array<float, N_COEFFS> const& coeffs;

    // Evaluate the polynomial using Horner's method based on the polynomial type
    inline __device__ float eval_horner(float x) const {
        if constexpr (POLYNOMIAL_TYPE == PolynomialType::FULL) {
            // Evaluate a full polynomial
            return eval_poly_horner(coeffs, x);
        } else if constexpr (POLYNOMIAL_TYPE == PolynomialType::EVEN) {
            // Evaluate an even-only polynomial
            return eval_poly_even_horner(coeffs, x);
        } else if constexpr (POLYNOMIAL_TYPE == PolynomialType::ODD) {
            // Evaluate an odd-only polynomial
            return eval_poly_odd_horner(coeffs, x);
        }
    }
};

template <size_t N_NEWTON_ITERATIONS, class PolyProxy, class DPolyProxy, class TInvPolyApproxProxy>
inline __device__ float eval_poly_inverse_horner_newton(PolyProxy const& poly,
                                                        DPolyProxy const& dpoly,
                                                        TInvPolyApproxProxy const& inv_poly_approx,
                                                        float y) {
    // Evaluates the inverse x = f^{-1}(y) of a reference polynomial y=f(x) (given by poly_coefficients) at points y
    // using numerically stable Horner scheme and Newton iterations starting from an approximate solution \\hat{x} = \\hat{f}^{-1}(y)
    // (given by inv_poly_approx) and the polynomials derivative df/dx (given by poly_derivative_coefficients)

    static_assert(N_NEWTON_ITERATIONS >= 0, "Require at least a single Newton iteration");

    // approximation / starting points - also returned for zero iterations
    auto x = inv_poly_approx.eval_horner(y);

#pragma unroll
    for (auto j = 0; j < N_NEWTON_ITERATIONS; ++j) {
        auto const dfdx     = dpoly.eval_horner(x);
        auto const residual = poly.eval_horner(x) - y;
        x -= residual / dfdx;
    }

    return x;
}

// ---------------------------------------------------------------------------------------------

// Camera models

inline __device__ auto interpolate_shutter_pose_t(float relative_frame_time, RollingShutterParameters const& rolling_shutter_parameters)
    -> glm::fvec3 {
    auto const& frame_T_world_sensors = rolling_shutter_parameters.T_world_sensors;
    auto const t_start                = glm::fvec3(frame_T_world_sensors[0],
                                                   frame_T_world_sensors[1],
                                                   frame_T_world_sensors[2]);
    auto const q_start                = glm::fvec4(frame_T_world_sensors[3],
                                                   frame_T_world_sensors[4],
                                                   frame_T_world_sensors[5],
                                                   frame_T_world_sensors[6]);
    auto const t_end                  = glm::fvec3(frame_T_world_sensors[7],
                                                   frame_T_world_sensors[8],
                                                   frame_T_world_sensors[9]);
    auto const q_end                  = glm::fvec4(frame_T_world_sensors[10],
                                                   frame_T_world_sensors[11],
                                                   frame_T_world_sensors[12],
                                                   frame_T_world_sensors[13]);

    // Interpolate a pose linearly for a relative frame time
    return (1.f - relative_frame_time) * t_start + relative_frame_time * t_end;
}

/**
 * @brief Checks if a given image point is within the image bounds considering a margin.
 *
 * This function determines whether a specified point in image coordinates lies within the bounds
 * of the image, taking into account a margin factor. The margin is calculated as a fraction of the
 * image resolution.
 *
 * @param image_point The point in image coordinates to check.
 * @param resolution The resolution of the image as an array where resolution[0] is the width and resolution[1] is the height.
 * @param margin_factor The factor by which the margin is calculated. The margin is computed as margin_factor * resolution.
 * @return true if the image point is within the image bounds considering the margin, false otherwise.
 */
__forceinline__ __device__ __host__ bool image_point_in_image_bounds_margin(glm::vec2 const& image_point,
                                                                            std::array<uint64_t, 2> const& resolution,
                                                                            float margin_factor) {
	const float MARGIN_X = resolution[0] * margin_factor;
    const float MARGIN_Y = resolution[1] * margin_factor;
    bool valid = true;
    valid &= (-MARGIN_X) <= image_point.x && image_point.x < (resolution[0]+MARGIN_X);
    valid &= (-MARGIN_Y) <= image_point.y && image_point.y < (resolution[1]+MARGIN_Y);
    return valid;
}

struct WorldRay {
    glm::fvec3 ray_org;
    glm::fvec3 ray_dir;
};

struct ShutterPose {
    glm::fvec3 t;
    glm::fquat q;

    inline __device__ __host__ auto camera_world_position() const -> glm::fvec3 {
        return apply_quaternion(glm::inverse(q), -t);
    }

    inline __device__ __host__ auto camera_ray_to_world_ray(glm::fvec3 const& camera_ray) const -> WorldRay {
        auto const R_inv = glm::mat3_cast(glm::inverse(q));

        return {-R_inv * t, R_inv * camera_ray};
    }
};

inline __device__ __host__ auto interpolate_shutter_pose(float relative_frame_time,
                                                         RollingShutterParameters const& rolling_shutter_parameters)
    -> ShutterPose {
    auto const& frame_T_world_sensors = rolling_shutter_parameters.T_world_sensors;
    auto const t_start                = glm::fvec3(frame_T_world_sensors[0],
                                                   frame_T_world_sensors[1],
                                                   frame_T_world_sensors[2]);
    auto const q_start                = glm::fvec4(frame_T_world_sensors[3],
                                                   frame_T_world_sensors[4],
                                                   frame_T_world_sensors[5],
                                                   frame_T_world_sensors[6]);
    auto const t_end                  = glm::fvec3(frame_T_world_sensors[7],
                                                   frame_T_world_sensors[8],
                                                   frame_T_world_sensors[9]);
    auto const q_end                  = glm::fvec4(frame_T_world_sensors[10],
                                                   frame_T_world_sensors[11],
                                                   frame_T_world_sensors[12],
                                                   frame_T_world_sensors[13]);

    // Interpolate a pose linearly for a relative frame time
    auto const t_rs = (1.f - relative_frame_time) * t_start + relative_frame_time * t_end;
    auto const q_rs = unitquat_slerp(q_start, q_end, relative_frame_time); // xyzw representation
    return ShutterPose{ t_rs, glm::fquat{ q_rs[3], q_rs[0], q_rs[1], q_rs[2] } };
}

inline __device__ __host__ auto interpolate_shutter_pose_se3(float relative_frame_time, RollingShutterParameters const& rolling_shutter_parameters) {
    auto pose = interpolate_shutter_pose(relative_frame_time, rolling_shutter_parameters);
    auto ret = glm::mat4x3{glm::mat3_cast(pose.q)};
    ret[3] = pose.t;
    return ret;
}

template <class DerivedCameraModel>
struct BaseCameraModel {
    // CRTP base class for all camera model types

    // Function to compute the relative frame time for a given image point based on the shutter type
    inline __device__ auto
    shutter_relative_frame_time(glm::fvec2 const& image_point) const -> float {
        auto derived = static_cast<DerivedCameraModel const*>(this);

        auto t                 = 0.f;
        auto const& resolution = derived->parameters.resolution;
        switch (derived->parameters.shutter_type) {
        case ShutterType::ROLLING_TOP_TO_BOTTOM:
            t = std::floor(image_point.y) / (resolution[1] - 1);
            break;

        case ShutterType::ROLLING_LEFT_TO_RIGHT:
            t = std::floor(image_point.x) / (resolution[0] - 1);
            break;

        case ShutterType::ROLLING_BOTTOM_TO_TOP:
            t = (resolution[1] - std::ceil(image_point.y)) / (resolution[1] - 1);
            break;

        case ShutterType::ROLLING_RIGHT_TO_LEFT:
            t = (resolution[0] - std::ceil(image_point.x)) / (resolution[0] - 1);
            break;
        }

        return t;
    };

    inline __device__ auto
    image_point_to_world_ray_shutter_pose(glm::fvec2 const& image_point,
                                          RollingShutterParameters const& rolling_shutter_parameters) const -> WorldRay {
        // Unproject ray and transform to world using shutter pose

        auto derived = static_cast<DerivedCameraModel const*>(this);

        auto const camera_ray = derived->image_point_to_camera_ray(image_point);

        return interpolate_shutter_pose(shutter_relative_frame_time(image_point),
                                        rolling_shutter_parameters)
            .camera_ray_to_world_ray(camera_ray);
    };

    struct CameraRayToImagePointReturn {
        glm::fvec2 imagePoint;
        bool valid_flag;
    };

    struct WorldPointToImagePointReturn {
        glm::fvec2 imagePoint;
        bool valid_flag;
        int64_t timestamp_us;
        using TQuatArray = std::array<float, 7>;
        TQuatArray T_world_sensor;
    };

    template <size_t N_ROLLING_SHUTTER_ITERATIONS>
    inline __device__ auto
    world_point_to_image_point_shutter_pose(
        glm::fvec3 const& world_point,
        RollingShutterParameters const& rolling_shutter_parameters,
        float margin_factor
    ) const -> WorldPointToImagePointReturn 
    {
        // Perform rolling-shutter-based world point to image point projection / optimization

        auto derived = static_cast<DerivedCameraModel const*>(this);

        auto const& frame_T_world_sensors = rolling_shutter_parameters.T_world_sensors;
        auto const t_start                = glm::fvec3(frame_T_world_sensors[0],
                                                       frame_T_world_sensors[1],
                                                       frame_T_world_sensors[2]);
        auto const q_start                = glm::fvec4(frame_T_world_sensors[3],
                                                       frame_T_world_sensors[4],
                                                       frame_T_world_sensors[5],
                                                       frame_T_world_sensors[6]);
        auto const t_end                  = glm::fvec3(frame_T_world_sensors[7],
                                                       frame_T_world_sensors[8],
                                                       frame_T_world_sensors[9]);
        auto const q_end                  = glm::fvec4(frame_T_world_sensors[10],
                                                       frame_T_world_sensors[11],
                                                       frame_T_world_sensors[12],
                                                       frame_T_world_sensors[13]);

        // Always perform transformation using start pose
        auto const [image_point_start, valid_start] = derived->camera_ray_to_image_point(apply_quaternion(q_start, world_point) + t_start, margin_factor);

        if (derived->parameters.shutter_type == ShutterType::GLOBAL) {
            // Exit early if we have a global shutter sensor
            return {{image_point_start.x, image_point_start.y},
                    valid_start,
                    rolling_shutter_parameters.timestamps_us[0],
                    {
                        t_start.x,
                        t_start.y,
                        t_start.z,
                        q_start.x,
                        q_start.y,
                        q_start.z,
                        q_start.w,
                    }};
        }

        // Do initial transformations using both start and end poses to determine all candidate
        // points and take union of valid projections as iteration starting points
        auto const [image_point_end, valid_end] = derived->camera_ray_to_image_point(apply_quaternion(q_end, world_point) + t_end, margin_factor);

        // This selection prefers points at the start-of-frame pose over end-of-frame points
        // - the optimization will determine the final timestamp for each point
        auto init_image_point = glm::fvec2{};
        if (valid_start) {
            init_image_point = image_point_start;
        } else if (valid_end) {
            init_image_point = image_point_end;
        } else {
            // No valid projection at start or finish -> mark point as invalid. Still
            // return projection result at end of frame to be consistent with ncore, as
            // this will be condensed at the python interface level
            return {{image_point_end.x, image_point_end.y},
                    false,
                    rolling_shutter_parameters.timestamps_us[1],
                    {
                        t_end.x,
                        t_end.y,
                        t_end.z,
                        q_end.x,
                        q_end.y,
                        q_end.z,
                        q_end.w,
                    }};
        }

        // Compute the new timestamp and project again
        auto image_points_rs_prev = init_image_point;
        auto relative_frame_time  = float{};
        auto t_rs                 = glm::fvec3{};
        auto q_rs                 = glm::fvec4{};
#pragma unroll
        for (auto j = 0; j < N_ROLLING_SHUTTER_ITERATIONS; ++j) {
            relative_frame_time = shutter_relative_frame_time(image_points_rs_prev);

            t_rs = (1.f - relative_frame_time) * t_start + relative_frame_time * t_end;
            q_rs = unitquat_slerp(q_start, q_end, relative_frame_time);

            auto const [image_point_rs, valid_rs] = derived->camera_ray_to_image_point(apply_quaternion(q_rs, world_point) + t_rs, margin_factor);

            image_points_rs_prev = image_point_rs;
        }

        return {{image_points_rs_prev.x, image_points_rs_prev.y},
                true,
                rolling_shutter_parameters.timestamps_us[0] + int64_t(relative_frame_time * (rolling_shutter_parameters.timestamps_us[1] - rolling_shutter_parameters.timestamps_us[0])),
                {
                    t_rs.x,
                    t_rs.y,
                    t_rs.z,
                    q_rs.x,
                    q_rs.y,
                    q_rs.z,
                    q_rs.w,
                }};
    }
};

struct PerfectPinholeCameraModel : BaseCameraModel<PerfectPinholeCameraModel> {
    // OpenCV-like pinhole camera model without any distortion (NCore conventions)

    using Base = BaseCameraModel<PerfectPinholeCameraModel>;

    struct Parameters : CameraModelParameters {
        std::array<float, 2> principal_point;
        std::array<float, 2> focal_length;
    };

    __host__ __device__ PerfectPinholeCameraModel(Parameters const& parameters)
        : parameters(parameters) {
    }

    Parameters parameters;

    inline __device__ auto camera_ray_to_image_point(glm::fvec3 const& cam_ray, float margin_factor) const -> typename Base::CameraRayToImagePointReturn {
        auto image_point = glm::fvec2{0.f, 0.f};

        // Treat all the points behind the camera plane to invalid / projecting to origin (NCore convention)
        if (cam_ray.z <= 0.f)
            return {image_point, false};

        // Project using ideal pinhole model
        image_point = (glm::fvec2(cam_ray.x, cam_ray.y) / cam_ray.z) * glm::fvec2(parameters.focal_length[0],
                                                                                  parameters.focal_length[1]) +
                      glm::fvec2(parameters.principal_point[0],
                                 parameters.principal_point[1]);

        // Check if the image points fall within the image, set points that have too large distortion or fall outside the image sensor to invalid
        auto valid = true;
        valid &= image_point_in_image_bounds_margin(image_point, parameters.resolution, margin_factor);

        return {image_point, valid};
    }

    inline __device__ glm::fvec3 image_point_to_camera_ray(glm::fvec2 image_point) const {
        // Transform the image point to uv coordinate
        auto const uv = (image_point - glm::fvec2{parameters.principal_point[0], parameters.principal_point[1]}) / glm::fvec2{parameters.focal_length[0], parameters.focal_length[1]};

        // Unproject the image point to camera ray
        auto const camera_ray = glm::fvec3{uv.x, uv.y, 1.f};

        // Make sure ray is normalized
        return camera_ray / length(camera_ray);
    }
};

template <size_t N_MAX_UNDISTORTION_ITERATIONS = 5 /* half the number of maximum iterations as in NCore reference model (currently using 10)*/>
struct OpenCVPinholeCameraModel : BaseCameraModel<OpenCVPinholeCameraModel<N_MAX_UNDISTORTION_ITERATIONS>> {
    // OpenCV-compatible pinhole camera model (NCore conventions)

    using Base = BaseCameraModel<OpenCVPinholeCameraModel<N_MAX_UNDISTORTION_ITERATIONS>>;

    __host__ __device__ OpenCVPinholeCameraModel(OpenCVPinholeCameraModelParameters const& parameters, float stop_undistortion_square_error_px2 = 1e-12)
        : parameters(parameters), undistortion_stop_square_error_px2(stop_undistortion_square_error_px2) {}

    OpenCVPinholeCameraModelParameters parameters;
    float undistortion_stop_square_error_px2;

    struct DistortionReturn {
        float icD;
        glm::fvec2 delta;
        float r2;
    };

    inline __device__ auto compute_distortion(glm::fvec2 const& uv) const -> DistortionReturn {
        // Computes the radial, tangential, and thin-prism distortion given the camera ray
        auto const uv_squared = glm::fvec2(uv[0] * uv[0], uv[1] * uv[1]);
        auto const r2         = uv_squared.x + uv_squared.y;
        auto const a1         = 2.f * uv[0] * uv[1];
        auto const a2         = r2 + 2.f * uv_squared.x;
        auto const a3         = r2 + 2.f * uv_squared.y;

        auto const icD_numerator   = 1.f + r2 * (parameters.radial_coeffs[0] + r2 * (parameters.radial_coeffs[1] + r2 * parameters.radial_coeffs[2]));
        auto const icD_denominator = 1.f + r2 * (parameters.radial_coeffs[3] + r2 * (parameters.radial_coeffs[4] + r2 * parameters.radial_coeffs[5]));
        auto const icD             = icD_numerator / icD_denominator;

        auto const delta_x = parameters.tangential_coeffs[0] * a1 + parameters.tangential_coeffs[1] * a2 + r2 * (parameters.thin_prism_coeffs[0] + r2 * parameters.thin_prism_coeffs[1]);
        auto const delta_y = parameters.tangential_coeffs[0] * a3 + parameters.tangential_coeffs[1] * a1 + r2 * (parameters.thin_prism_coeffs[2] + r2 * parameters.thin_prism_coeffs[3]);

        return {icD, glm::fvec2{delta_x, delta_y}, r2};
    }

    inline __device__ auto camera_ray_to_image_point(glm::fvec3 const& cam_ray, float margin_factor) const -> typename Base::CameraRayToImagePointReturn {
        auto image_point = glm::fvec2{0.f, 0.f};

        // Treat all the points behind the camera plane to invalid / projecting to origin (NCore convention)
        if (cam_ray.z <= 0.f)
            return {image_point, false};

        // Evalutate distortion
        auto const uv_normalized    = glm::fvec2(cam_ray.x, cam_ray.y) / cam_ray.z;
        auto const [icD, delta, r2] = compute_distortion(uv_normalized);

        auto constexpr k_min_radial_dist = 0.8f, k_max_radial_dist = 1.2f;
        auto const valid_radial = (icD > k_min_radial_dist) && (icD < k_max_radial_dist);

        // Project using ideal pinhole model (apply radial / tangential / thin-prism distortions)
        // in case radial distortion is within limits
        auto const uvND = icD * uv_normalized + delta;

        if (valid_radial) {
            image_point = uvND * glm::fvec2(parameters.focal_length[0],
                                            parameters.focal_length[1]) +
                          glm::fvec2(parameters.principal_point[0],
                                     parameters.principal_point[1]);
        } else {
            // If the radial distortion is out-of-limits, the computed coordinates will be unreasonable
            // (might even flip signs) - check on which side of the image we overshoot, and set the coordinates
            // out of the image bounds accordingly. The coordinates will be clipped to
            // viable range and direction but the exact values cannot be trusted / are still invalid
            auto const roi_clipping_radius = std::hypotf(parameters.resolution[0], parameters.resolution[1]);
            image_point                    = (roi_clipping_radius / std::sqrt(r2)) * uv_normalized + glm::fvec2(parameters.principal_point[0],
                                                                                                                parameters.principal_point[1]);
        }

        // Check if the image points fall within the image, set points that have too large distortion or fall outside the image sensor to invalid
        auto valid = valid_radial;
        valid &= image_point_in_image_bounds_margin(image_point, parameters.resolution, margin_factor);

        return {image_point, valid};
    }

    inline __device__ glm::fvec2 compute_undistortion_iterative(glm::fvec2 const& image_point) const {
        // Iteratively undistorts the image point using the inverse distortion model

        // Initial guess for the undistorted point
        auto const uv_0 = (image_point - glm::fvec2{parameters.principal_point[0], parameters.principal_point[1]}) / glm::fvec2{parameters.focal_length[0], parameters.focal_length[1]};

        auto uv = uv_0;
        for (auto j = 0; j < N_MAX_UNDISTORTION_ITERATIONS; ++j) {
            // Compute the distortion for the current estimate
            auto const [icD, delta, r2] = compute_distortion(uv);

            // Update the estimate using the inverse distortion model
            auto const uv_next = (uv_0 - delta) / icD;

            // Check for convergence
            if (auto const residual_vec = uv - uv_next; glm::dot(residual_vec, residual_vec) < undistortion_stop_square_error_px2)
                break;

            uv = uv_next;
        }

        return uv;
    }

    inline __device__ glm::fvec3 image_point_to_camera_ray(glm::fvec2 image_point) const {
        // Undistort the image point to uv coordinate
        auto const uv = compute_undistortion_iterative(image_point);

        // Unproject the undistorted image point to camera ray
        auto const camera_ray = glm::fvec3{uv.x, uv.y, 1.f};

        // Make sure ray is normalized
        return camera_ray / length(camera_ray);
    }
};

template <size_t N_NEWTON_ITERATIONS = 3 /* fixed number of Netwon iteration for polynomial inversion - same as in NCore */>
struct OpenCVFisheyeCameraModel : BaseCameraModel<OpenCVFisheyeCameraModel<N_NEWTON_ITERATIONS>> {
    // OpenCV-compatible fisheye camera model (NCore conventions)

    using Base = BaseCameraModel<OpenCVFisheyeCameraModel<N_NEWTON_ITERATIONS>>;

    OpenCVFisheyeCameraModel(OpenCVFisheyeCameraModelParameters const& parameters,
                             float min_2d_norm = 1e-6f)
        : parameters(parameters), min_2d_norm(min_2d_norm) {
        // initialize ninth-degree odd-only forward polynomial (mapping angles to normalized distances) theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
        auto const& [k1, k2, k3, k4] = parameters.radial_coeffs;
        forward_poly_odd             = {1.f, k1, k2, k3, k4};

        // eighth-degree differential of forward polynomial 1 + 3*k1*theta^2 + 5*k2*theta^4 + 7*k3*theta^8 + 9*k4*theta^8
        dforward_poly_even = {1, 3 * k1, 5 * k2, 7 * k3, 9 * k4};

        // approximate backward poly (mapping normalized distances to angles) *very crudely* by linear interpolation / equidistant angle model (also assuming image-centered principal point)
        auto const max_normalized_dist = std::max(parameters.resolution[0] / 2.f / parameters.focal_length[0], parameters.resolution[1] / 2.f / parameters.focal_length[1]);
        approx_backward_poly           = {0.f, parameters.max_angle / max_normalized_dist};
    }

    OpenCVFisheyeCameraModelParameters parameters;
    float min_2d_norm;
    std::array<float, 5> forward_poly_odd;
    std::array<float, 5> dforward_poly_even;
    std::array<float, 2> approx_backward_poly;

    inline __device__ auto camera_ray_to_image_point(glm::fvec3 const& cam_ray, float margin_factor) const -> typename Base::CameraRayToImagePointReturn {
        // Make sure norm is non-vanishing (norm vanishes for points along the principal-axis)
        auto cam_ray_xy_norm = numerically_stable_norm2(cam_ray.x, cam_ray.y);
        if (cam_ray_xy_norm <= 0.f)
            cam_ray_xy_norm = std::numeric_limits<float>::epsilon();

        auto const theta_full = std::atan2(cam_ray_xy_norm, cam_ray.z);

        // Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
        // In particular for omnidirectional cameras, this prevents points outside the FOV to be
        // wrongly projected to in-image-domain points because of badly constrained polynomials outside
        // the effective FOV (which is different to the image boundaries).
        //
        // These FOV-clamped projections will be marked as *invalid*
        auto const theta = theta_full < parameters.max_angle ? theta_full : parameters.max_angle;

        // Evaluate forward polynomial (correspond to the radial distances to the principal point in the normalized image domain (up to focal length scales))
        auto const delta = eval_poly_odd_horner(forward_poly_odd, theta) / cam_ray_xy_norm;

        auto const image_point = glm::fvec2{parameters.focal_length[0] * delta * cam_ray.x + parameters.principal_point[0],
                                            parameters.focal_length[1] * delta * cam_ray.y + parameters.principal_point[1]};

        auto valid = true;
        valid &= image_point_in_image_bounds_margin(image_point, parameters.resolution, margin_factor);
        valid &= theta < parameters.max_angle; // explicitly check for strictly smaller angles to classify FOV-clamped points as invalid

        return {image_point, valid};
    }

    inline __device__ glm::fvec3 image_point_to_camera_ray(glm::fvec2 image_point) const {
        // Normalize the image point coordinates
        auto const uv = (image_point - glm::fvec2{parameters.principal_point[0], parameters.principal_point[1]}) / glm::fvec2{parameters.focal_length[0], parameters.focal_length[1]};

        // Compute the radial distance from the principal point
        auto const delta = length(uv);

        // Evaluate the inverse polynomial to find the angle theta
        auto const theta = eval_poly_inverse_horner_newton<N_NEWTON_ITERATIONS>(PolynomialProxy<PolynomialType::ODD, 5>{forward_poly_odd},
                                                                                PolynomialProxy<PolynomialType::EVEN, 5>{dforward_poly_even},
                                                                                PolynomialProxy<PolynomialType::FULL, 2>{approx_backward_poly},
                                                                                delta);

        // Compute the camera ray and set the ones at the image center to [0,0,1]
        if (delta >= min_2d_norm) {
            // Scale the uv coordinates by the sine of the angle theta
            auto const scale_factor = std::sin(theta) / delta;
            return glm::fvec3{scale_factor * uv.x, scale_factor * uv.y, std::cos(theta)};
        } else {
            // For points at the image center, return a ray pointing straight ahead
            return glm::fvec3{0.f, 0.f, 1.f};
        }
    }
};

template <size_t N_NEWTON_ITERATIONS = 3 /* fixed number of Netwon iteration for polynomial inversion - same as in NCore */>
struct BackwardsFThetaCameraModel : BaseCameraModel<BackwardsFThetaCameraModel<N_NEWTON_ITERATIONS>> {
    // NV-compatible FTheta camera model (NCore conventions)

    using Base = BaseCameraModel<BackwardsFThetaCameraModel<N_NEWTON_ITERATIONS>>;

    BackwardsFThetaCameraModel(FThetaCameraModelParameters const& parameters,
                               float min_2d_norm = 1e-6f)
        : parameters(parameters), min_2d_norm(min_2d_norm), dpixeldist_to_angle_poly{} {
        if (parameters.reference_poly != FThetaCameraModelParameters::PolynomialType::PIXELDIST_TO_ANGLE)
            throw std::runtime_error("Only supporting backwards reference polynomials");

        // FThetaCameraModelParameters are defined such that the image coordinate origin corresponds to
        // the center of the first pixel. To conform to the NCore CameraModel specification (having the image
        // coordinate origin aligned with the top-left corner of the first pixel) we therefore need to
        // offset the principal point by half a pixel. Please see NCore documentation for more information.
        this->parameters.principal_point[0] += .5f;
        this->parameters.principal_point[1] += .5f;

// compute first derivative of the backwards polynomial
#pragma unroll
        for (auto j = 0; j < std::size(dpixeldist_to_angle_poly); ++j)
            dpixeldist_to_angle_poly[j] = (j + 1) * parameters.pixeldist_to_angle_poly.at(j + 1);
    }

    FThetaCameraModelParameters parameters;
    float min_2d_norm;
    std::array<float, FThetaCameraModelParameters::PolynomialDegree - 1> dpixeldist_to_angle_poly; // coefficient of first derivative of the backwards polynomial

    inline __device__ auto camera_ray_to_image_point(glm::fvec3 const& cam_ray, float margin_factor) const -> typename Base::CameraRayToImagePointReturn {
        // Make sure norm is non-vanishing (norm vanishes for points along the principal-axis)
        auto cam_ray_xy_norm = numerically_stable_norm2(cam_ray.x, cam_ray.y);
        if (cam_ray_xy_norm <= 0.f)
            cam_ray_xy_norm = std::numeric_limits<float>::epsilon();

        auto const alpha_full = std::atan2(cam_ray_xy_norm, cam_ray.z);

        // Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
        // In particular for omnidirectional cameras, this prevents points outside the FOV to be
        // wrongly projected to in-image-domain points because of badly constrained polynomials outside
        // the effective FOV (which is different to the image boundaries).
        //
        // These FOV-clamped projections will be marked as *invalid*
        auto const alpha = alpha_full < parameters.max_angle ? alpha_full : parameters.max_angle;

        auto const delta = eval_poly_inverse_horner_newton<N_NEWTON_ITERATIONS>(PolynomialProxy<PolynomialType::FULL, 6>{parameters.pixeldist_to_angle_poly},
                                                                                PolynomialProxy<PolynomialType::FULL, 5>{dpixeldist_to_angle_poly},
                                                                                PolynomialProxy<PolynomialType::FULL, 6>{parameters.angle_to_pixeldist_poly},
                                                                                alpha);

        auto const theta       = delta / cam_ray_xy_norm;
        auto const image_point = glm::fvec2{theta * cam_ray.x + parameters.principal_point[0], theta * cam_ray.y + parameters.principal_point[1]};

        auto valid = true;
        valid &= image_point_in_image_bounds_margin(image_point, parameters.resolution, margin_factor);
        valid &= alpha < parameters.max_angle;

        return {image_point, valid};
    }

    inline __device__ glm::fvec3 image_point_to_camera_ray(glm::fvec2 image_point) const {
        auto const image_point_dist = image_point - glm::fvec2{parameters.principal_point[0], parameters.principal_point[1]};

        auto const rdist = length(image_point_dist);

        // Evaluate backward polynomial
        auto const alpha = eval_poly_horner(parameters.pixeldist_to_angle_poly, rdist);

        // Compute the camera ray and set the ones at the image center to [0,0,1]
        if (rdist >= min_2d_norm) {
            auto const scale_factor = std::sin(alpha) / rdist;
            return glm::fvec3{scale_factor * image_point_dist.x, scale_factor * image_point_dist.y, std::cos(alpha)};
        } else {
            return glm::fvec3{0.f, 0.f, 1.f};
        }
    }
};

// ---------------------------------------------------------------------------------------------

// Gaussian projections

// The approximation of a transformed distribution is performed by an UnscentedTransform,
// which is based on the idea to approximate the transformed distribution's momements by
// mapping a set of sigma points, which are sampled according to the input distribution.
//
// In general, unscented transform are *derivative-free* and provide an attractive compromisse
// in terms of accuracy (better approximation quality than simple linearized distribution updates)
// and are faster to evaluate compared to more accurate estimations (like Monte Carlo simulations),
// as they provide a "guided" selection of the sample points to transform and require
// much less transformation function evaluations.

// See
//
// - "Some Relations Between Extended and Unscented Kalman Filters" - Gustafsson and Hendeby 2012
// - "On Unscented Kalman Filtering for State Estimation of Continuous-Time Nonlinear Systems" - Särkkä 2007
//
// for references

struct SigmaPoints {
    std::array<glm::fvec3, 2 * 3 + 1> points;
    std::array<float, 2 * 3 + 1> weights_mean;
    std::array<float, 2 * 3 + 1> weights_covariance;
};

inline __device__ auto world_gaussian_sigma_points(
    UnscentedTransformParameters const& unscented_transform_parameters,
    glm::fvec3 const& gaussian_world_mean,
    glm::fvec3 const& gaussian_world_scale,
    glm::fquat const& gaussian_world_rot) -> SigmaPoints {
    size_t constexpr static D = 3;
    auto const& alpha         = unscented_transform_parameters.alpha;
    auto const& beta          = unscented_transform_parameters.beta;
    auto const& kappa         = unscented_transform_parameters.kappa;
    auto const lambda         = alpha * alpha * (D + kappa) - D;

    // Compute rotation matrix R from quaternion (scaling matrix S is diag(s_i))
    glm::fmat3 R = glm::mat3_cast(gaussian_world_rot);

    // The _factored_ Gaussian covariance parametrization C = (S * R)^T * (S * R)
    // provides a closed form of it's SVD C = U * Σ * U^T with U = R^T and Σ = S^T*S = diag((s_i)^2).

    // Use this closed form SVD to compute sigma points and weights from a (D + lambda)-scaled covariance C
    // (cf. "On Unscented Kalman Filtering for State Estimation of Continuous-Time Nonlinear Systems" - Särkkä 2007).
    // In particular, this means that the singular values are given by σ_i = s_i and the singular vectors u_i are the
    // columns of R
    auto ret = SigmaPoints{};

    ret.points[0] = gaussian_world_mean;

    DEBUG_PRINTF_CUDA("lambda %f\n", lambda);

#pragma unroll
    for (auto i = 0u; i < D; ++i) {
        auto const delta = std::sqrt(D + lambda) * gaussian_world_scale[i] * R[i];
        // "m + sqrt((n+lambda)*C)_i"
        ret.points[i + 1] = gaussian_world_mean + delta;
        // "m - sqrt((n+lambda)*C)_i"
        ret.points[i + 1 + D] = gaussian_world_mean - delta;
    }

    // Compute weights
    ret.weights_mean[0]       = lambda / (D + lambda);
    ret.weights_covariance[0] = lambda / (D + lambda) + (1 - alpha * alpha + beta);

    DEBUG_PRINTF_CUDA("ret.weights_mean[0] %f\n", ret.weights_mean[0]);
    DEBUG_PRINTF_CUDA("ret.weights_covariance[0] %f\n", ret.weights_covariance[0]);

#pragma unroll
    for (auto i = 0u; i < 2 * D; ++i) {
        ret.weights_mean[i + 1]       = 1 / (2 * (D + lambda));
        ret.weights_covariance[i + 1] = 1 / (2 * (D + lambda));

        DEBUG_PRINTF_CUDA("ret.weights_mean[i + 1] %f\n", ret.weights_mean[i + 1]);
        DEBUG_PRINTF_CUDA("ret.weights_covariance[i + 1] %f\n", ret.weights_covariance[i + 1]);
    }

    return ret;
}

struct ImageGaussianReturn {
    glm::fvec2 mean;
    glm::fvec3 covariance; // upper triangular part of covariance
    bool valid;
};

template <size_t N_ROLLING_SHUTTER_ITERATIONS, class CameraModel>
inline __device__ auto world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
    CameraModel const& camera_model,
    RollingShutterParameters const& rolling_shutter_parameters,
    UnscentedTransformParameters const& unscented_transform_parameters,
    glm::fvec3 const& gaussian_world_mean,
    glm::fvec3 const& gaussian_world_scale,
    glm::fquat const& gaussian_world_rot) -> ImageGaussianReturn {

    DEBUG_PRINTF_CUDA("gaussian_world_mean %f %f %f\n", gaussian_world_mean.x, gaussian_world_mean.y, gaussian_world_mean.z);
    DEBUG_PRINTF_CUDA("gaussian_world_scale %f %f %f\n", gaussian_world_scale.x, gaussian_world_scale.y, gaussian_world_scale.z);
    DEBUG_PRINTF_CUDA("gaussian_world_rot %f %f %f %f\n", gaussian_world_rot.x, gaussian_world_rot.y, gaussian_world_rot.z, gaussian_world_rot.w);

    // Compute sigma points for input distribution
    auto const sigma_points = world_gaussian_sigma_points(unscented_transform_parameters, gaussian_world_mean, gaussian_world_scale, gaussian_world_rot);

    DEBUG_PRINTF_CUDA("sigma_points 0 %f %f %f\n", sigma_points.points[0].x, sigma_points.points[0].y, sigma_points.points[0].z);
    DEBUG_PRINTF_CUDA("sigma_points 1 %f %f %f\n", sigma_points.points[1].x, sigma_points.points[1].y, sigma_points.points[1].z);
    DEBUG_PRINTF_CUDA("sigma_points 2 %f %f %f\n", sigma_points.points[2].x, sigma_points.points[2].y, sigma_points.points[2].z);
    DEBUG_PRINTF_CUDA("sigma_points 3 %f %f %f\n", sigma_points.points[3].x, sigma_points.points[3].y, sigma_points.points[3].z);
    DEBUG_PRINTF_CUDA("sigma_points 4 %f %f %f\n", sigma_points.points[4].x, sigma_points.points[4].y, sigma_points.points[4].z);
    DEBUG_PRINTF_CUDA("sigma_points 5 %f %f %f\n", sigma_points.points[5].x, sigma_points.points[5].y, sigma_points.points[5].z);
    DEBUG_PRINTF_CUDA("sigma_points 6 %f %f %f\n", sigma_points.points[6].x, sigma_points.points[6].y, sigma_points.points[6].z);

    DEBUG_PRINTF_CUDA("sigma_points.weights_mean 0 %f\n", sigma_points.weights_mean[0]);
    DEBUG_PRINTF_CUDA("sigma_points.weights_mean 1 %f\n", sigma_points.weights_mean[1]);
    DEBUG_PRINTF_CUDA("sigma_points.weights_mean 2 %f\n", sigma_points.weights_mean[2]);
    DEBUG_PRINTF_CUDA("sigma_points.weights_mean 3 %f\n", sigma_points.weights_mean[3]);
    DEBUG_PRINTF_CUDA("sigma_points.weights_mean 4 %f\n", sigma_points.weights_mean[4]);
    DEBUG_PRINTF_CUDA("sigma_points.weights_mean 5 %f\n", sigma_points.weights_mean[5]);
    DEBUG_PRINTF_CUDA("sigma_points.weights_mean 6 %f\n", sigma_points.weights_mean[6]);

    // Transform sigma points / compute approximation of output distribution via sample mean / covariance
    bool valid        = unscented_transform_parameters.require_all_sigma_points_valid;
    auto image_points = std::array<glm::fvec2, 2 * 3 + 1>{};
    auto image_mean   = glm::fvec2{0};
#pragma unroll
    for (auto i = 0u; i < std::size(image_points); ++i) {
        auto const [image_point, point_valid, timestamp_us, T_world_sensor] =
            // annotate with 'template' to avoid warnings: #174-D: expression has no effect
            camera_model.template world_point_to_image_point_shutter_pose<N_ROLLING_SHUTTER_ITERATIONS>(
                sigma_points.points[i], rolling_shutter_parameters, unscented_transform_parameters.in_image_margin_factor);

        if (unscented_transform_parameters.require_all_sigma_points_valid) {
            valid &= point_valid; // all have to be valid
        } else {
            valid |= point_valid; // any valid is sufficient
        }

        image_points[i] = {image_point.x, image_point.y};

        image_mean += sigma_points.weights_mean[i] * image_points[i];

        DEBUG_PRINTF_CUDA("image_points[%i] %f %f  -- image_mean %f %f\n", i,
                          image_points[i].x, image_points[i].y, image_mean.x, image_mean.y);
        // DEBUG_PRINTF_CUDA("sigma_points.points[%i] %f %f\n", i, sigma_points.points[i].x, sigma_points.points[i].y);
        // DEBUG_PRINTF_CUDA("sigma_points.weights_mean[%i] %f\n", i, sigma_points.weights_mean[i]);
    }

    auto image_covariance = glm::fmat2{0};
    for (auto i = 0u; i < std::size(image_points); ++i) {
        auto const image_mean_vec = image_points[i] - image_mean;

        DEBUG_PRINTF_CUDA("image_mean_vec[%i] %f %f\n", i, image_mean_vec.x, image_mean_vec.y);

        image_covariance += sigma_points.weights_covariance[i] *
                            glm::outerProduct(image_mean_vec, image_mean_vec);

        DEBUG_PRINTF_CUDA("image_covariance[%i] %f %f\n", i, image_covariance[0][0], image_covariance[1][0]);
        DEBUG_PRINTF_CUDA("  %f %f\n", image_covariance[0][1], image_covariance[1][1]);
    }

    return {image_mean, {image_covariance[0][0], image_covariance[1][0], image_covariance[1][1]}, valid};
}

template <size_t N_ROLLING_SHUTTER_ITERATIONS, class CameraModel>
inline __device__ auto world_gaussian_to_image_gaussian_monte_carlo_transform_shutter_pose(
    CameraModel const& camera_model,
    RollingShutterParameters const& rolling_shutter_parameters,
    MonteCarloTransformParameters const& monte_carlo_transform_parameters,
    glm::fvec3 const& gaussian_world_mean,
    glm::fvec3 const& gaussian_world_scale,
    glm::fquat const& gaussian_world_rot) -> ImageGaussianReturn {

    /// Computes 2D screen-space projection for a Gaussian via Monte Carlo sampling of the projection

    // seed a random number generator
    thrust::default_random_engine rng(cg::this_grid().thread_rank());

    // standard normal distribution (mean 0, variance 1)
    thrust::normal_distribution<float> d{0.f, 1.f};

    // Compute offset transformation L such that x = mean + L * u is normal distributed according to mean and covariance C = L * L^T
    // (see https://juanitorduz.github.io/multivariate_normal/)
    // As C = R * S * S^T * R^T, we have that L = R * S (note, this is not lower-triangular / Cholesky, but more a QR decomposition with a diagonal term,
    // which is sufficient for the transformation below)

    auto const R = glm::mat3_cast(gaussian_world_rot); // quat is local -> world
    auto const S = glm::mat3{gaussian_world_scale.x, 0.f, 0.f,
                             0.f, gaussian_world_scale.y, 0.f,
                             0.f, 0.f, gaussian_world_scale.z};
    auto const L = R * S;

    // Welford's online algorithm for 2D mean and covariance
    glm::fvec2 mean;  // running sum of online mean
    glm::mat2 SDiffs; // running sum of online outer product of mean differences

    bool valid = monte_carlo_transform_parameters.require_all_sample_points_valid;

    for (auto i = 1; i <= monte_carlo_transform_parameters.n_samples; ++i) {
        // sample point in world space according to original Gaussian distribution
        auto const world_sample_point = gaussian_world_mean + L * glm::fvec3{d(rng), d(rng), d(rng)};

        // project the sample point to 2D screen space
        auto const [sample_point_2d, sample_point_valid, sample_point_timestamp_us, sample_point_T_world_sensor] =
            camera_model.template world_point_to_image_point_shutter_pose<N_ROLLING_SHUTTER_ITERATIONS>(
                world_sample_point,
                rolling_shutter_parameters,
                monte_carlo_transform_parameters.in_image_margin_factor);

        if (monte_carlo_transform_parameters.require_all_sample_points_valid) {
            valid &= sample_point_valid; // all have to be valid
        } else {
            valid |= sample_point_valid; // any valid is sufficient
        }

        // update using Welford's online algorithm for 3d mean and covariance (~outer product of differences to running means)

        if (i == 1) {
            // init estimates (single sample / no outer product difference)
            mean   = sample_point_2d;
            SDiffs = {0.f, 0.f, 0.f, 0.f};
        } else {
            auto const mean_vec_old = sample_point_2d - mean;
            mean += mean_vec_old / float(i);
            SDiffs += glm::outerProduct(mean_vec_old, sample_point_2d - mean); // todo [JME]: speed up / don't compute lower diagonal
        }
    }

    auto const image_mean       = mean; // sample mean
    auto const image_covariance = glm::fvec3{SDiffs[0][0],
                                             SDiffs[1][0],
                                             SDiffs[1][1]} /
                                  float(monte_carlo_transform_parameters.n_samples - 1); // unbiased sample covariance

    return {image_mean, image_covariance, valid};
}

inline glm::vec3 CameraGSplat::position() const { 
    return glm::vec3{ _position[0], _position[1], _position[2] }; 
}

inline glm::mat4x3 CameraGSplat::viewmatrix() const { 
    glm::mat4x3 mat;
	for (int i = 0; i < 4; i++) {
		float4 tmp = *((float4*)(_viewmatrix.data() + i * 4));
		mat[i][0] = tmp.x;
		mat[i][1] = tmp.y;
		mat[i][2] = tmp.z;
	}
	return mat;
}

inline glm::mat4x4 CameraGSplat::projmatrix() const {
    glm::mat4x4 mat;
	for (int i = 0; i < 4; i++) {
		float4 tmp = *((float4*)(_projmatrix.data() + i * 4));
		mat[i][0] = tmp.x;
		mat[i][1] = tmp.y;
		mat[i][2] = tmp.z;
		mat[i][3] = tmp.w;
	}
	return mat;
}

inline glm::mat4x4 CameraGSplat::inv_viewprojmatrix() const {
    glm::mat4x4 mat;
	for (int i = 0; i < 4; i++) {
		float4 tmp = *((float4*)(_inv_viewprojmatrix.data() + i * 4));
		mat[i][0] = tmp.x;
		mat[i][1] = tmp.y;
		mat[i][2] = tmp.z;
		mat[i][3] = tmp.w;
	}
	return mat;
}

// std::variants are too complicated for CUDA, 
template<typename CAMERA_MODEL>
struct DeviceCameraInput {
	const CAMERA_MODEL camera_model;
	const RollingShutterParameters rolling_shutter_parameters;

    DeviceCameraInput(const CAMERA_MODEL& camera_model, const RollingShutterParameters& rolling_shutter_parameters)
        : camera_model(camera_model)
        , rolling_shutter_parameters(rolling_shutter_parameters) 
    {}

    __forceinline__ __device__ 
    const std::array<uint64_t, 2>& resolution() const {
        return camera_model.parameters.resolution;
    }

    __forceinline__ __device__
    WorldRay generate_ray(const float2& pixf) const {
        return camera_model.image_point_to_world_ray_shutter_pose(glm::vec2(pixf.x, pixf.y), rolling_shutter_parameters);    
    }
};

template<>
struct DeviceCameraInput<CameraGSplat> {
    std::array<uint64_t, 2> _resolution;
	glm::mat4 projmatrix_inv;
	glm::vec3 cam_pos;

    DeviceCameraInput(const CameraGSplat& camera)
        : _resolution{camera.resolution}
        , projmatrix_inv{camera.inv_viewprojmatrix()}
        , cam_pos{camera.position()}
    {}

    __forceinline__ __device__ 
    const std::array<uint64_t, 2>& resolution() const {
        return _resolution;
    }

    __forceinline__ __device__ 
    WorldRay generate_ray(const float2& pixf) const {
        const glm::vec3 dir = computeViewRay(projmatrix_inv, cam_pos, pixf, _resolution[0], _resolution[1]);
        return { cam_pos, dir };
    }
};
