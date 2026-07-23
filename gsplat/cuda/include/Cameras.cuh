/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

// Some of the code are referenced from 3DGRUT codebase.
// https://github.com/nv-tlabs/3dgrut
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <cuda_runtime.h>

// Silence warnings / errors of the form
//
// __device__ / __host__ annotation is ignored on a function("XXX") that is
// explicitly defaulted on its first declaration
//
// in GLM
#define GLM_ENABLE_EXPERIMENTAL
#pragma nv_diag_suppress = esa_on_defaulted_function_ignored
#include <glm/gtx/matrix_operation.hpp> // needs define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>       // glm rotate
#pragma nv_diag_default = esa_on_defaulted_function_ignored

#include "Cameras.h"
#include "ExternalDistortion.cuh"
#include "Utils.cuh"

template<typename T, std::size_t N>
__device__ std::array<T, N> make_array(const T *ptr)
{
    std::array<T, N> arr;
#pragma unroll
    for(std::size_t i = 0; i < N; ++i)
    {
        arr[i] = ptr[i];
    }
    return arr;
}

// Per-sensor codegen knobs for the UT projection kernel. The default keeps
// camera models close to the baseline path; specializations opt into measured
// faster codegen/solver policies.
template<typename SensorModel>
struct ProjectionUTCodegenTraits
{
    static constexpr int kMinBlocks                = 1;
    static constexpr unsigned kUTUnroll            = 2u * 3u + 1u;
    static constexpr bool kNeedsCulling            = true;
    static constexpr bool kUseGaussianScopeSlerper = false;

    template<size_t N_ROLLING_SHUTTER_ITERATIONS>
    static constexpr auto rolling_shutter_unroll() -> unsigned
    {
        return N_ROLLING_SHUTTER_ITERATIONS == 0u ? 1u : N_ROLLING_SHUTTER_ITERATIONS;
    }
};

struct RollingShutterParameters
{
    glm::fvec3 t_start;
    glm::fquat q_start;
    glm::fvec3 t_end;
    glm::fquat q_end;

    __device__ RollingShutterParameters() = default;

    __device__ RollingShutterParameters(const float *se3_start, const float *se3_end)
    {
        // input is row-major, but glm is column-major
        q_start = glm::quat_cast(
            glm::mat3(
                se3_start[0],
                se3_start[4],
                se3_start[8],
                se3_start[1],
                se3_start[5],
                se3_start[9],
                se3_start[2],
                se3_start[6],
                se3_start[10]
            )
        );
        t_start = glm::fvec3(se3_start[3], se3_start[7], se3_start[11]);

        if(se3_end == nullptr)
        {
            q_end = q_start;
            t_end = t_start;
        }
        else
        {
            q_end = glm::quat_cast(
                glm::mat3(
                    se3_end[0],
                    se3_end[4],
                    se3_end[8],
                    se3_end[1],
                    se3_end[5],
                    se3_end[9],
                    se3_end[2],
                    se3_end[6],
                    se3_end[10]
                )
            );
            t_end = glm::fvec3(se3_end[3], se3_end[7], se3_end[11]);
        }
    }
};

// ---------------------------------------------------------------------------------------------

// Math helpers (polynomial evaluation / stable norms)

inline __device__ float numerically_stable_norm2(float x, float y)
{
    // Computes 2-norm of a [x,y] vector in a numerically stable way
    const auto abs_x = std::fabs(x);
    const auto abs_y = std::fabs(y);
    const auto min   = std::fmin(abs_x, abs_y);
    const auto max   = std::fmax(abs_x, abs_y);

    if(max <= 0.f)
    {
        return 0.f;
    }

    const auto min_max_ratio = min / max;
    return max * std::sqrt(1.f + min_max_ratio * min_max_ratio);
}

template<size_t N_COEFFS>
inline __host__ __device__ float eval_poly_horner(const std::array<float, N_COEFFS> &poly, float x)
{
    // Evaluates a polynomial y=f(x) with
    //
    // f(x) = c_0*x^0 + c_1*x^1 + c_2*x^2 + c_3*x^3 + c_4*x^4 ...
    //
    // given by poly_coefficients c_i at points x using numerically stable
    // Horner scheme.
    //
    // The degree of the polynomial is N_COEFFS - 1

    auto y = float{0};
    for(auto cit = poly.rbegin(); cit != poly.rend(); ++cit)
    {
        y = x * y + (*cit);
    }
    return y;
}

template<size_t N_COEFFS>
inline __host__ __device__ float eval_poly_odd_horner(const std::array<float, N_COEFFS> &poly_odd, float x)
{
    // Evaluates an odd-only polynomial y=f(x) with
    //
    // f(x) = c_0*x^1 + c_1*x^3 + c_2*x^5 + c_3*x^7 + c_4*x^9 ...
    //
    // given by poly_coefficients c_i at points x using numerically stable
    // Horner scheme.
    //
    // The degree of the polynomial is 2*N_COEFFS - 1

    return x * eval_poly_horner(poly_odd, x * x); // evaluate x^2-based "regular" polynomial after facting out
                                                  // one x term
}

template<size_t N_COEFFS>
inline __host__ __device__ float eval_poly_even_horner(const std::array<float, N_COEFFS> &poly_even, float x)
{
    // Evaluates an even-only polynomial y=f(x) with
    //
    // f(x) = c_0 + c_1*x^2 + c_2*x^4 + c_3*x^6 + c_4*x^8 ...
    //
    // given by poly_coefficients c_i at points x using numerically stable
    // Horner scheme.
    //
    // The degree of the polynomial is 2*(N_COEFFS - 1)

    return eval_poly_horner(poly_even, x * x); // evaluate x^2-substituted "regular" polynomial
}

// Enum to represent the type of polynomial
enum class PolynomialType
{
    FULL, // Represents a full polynomial with all terms
    EVEN, // Represents an even-only polynomial
    ODD   // Represents an odd-only polynomial
};

template<PolynomialType POLYNOMIAL_TYPE, size_t N_COEFFS>
struct PolynomialProxy
{
    const std::array<float, N_COEFFS> &coeffs;

    // Evaluate the polynomial using Horner's method based on the polynomial
    // type
    inline __host__ __device__ float eval_horner(float x) const
    {
        if constexpr(POLYNOMIAL_TYPE == PolynomialType::FULL)
        {
            // Evaluate a full polynomial
            return eval_poly_horner(coeffs, x);
        }
        else if constexpr(POLYNOMIAL_TYPE == PolynomialType::EVEN)
        {
            // Evaluate an even-only polynomial
            return eval_poly_even_horner(coeffs, x);
        }
        else if constexpr(POLYNOMIAL_TYPE == PolynomialType::ODD)
        {
            // Evaluate an odd-only polynomial
            return eval_poly_odd_horner(coeffs, x);
        }
    }
};

template<size_t N_NEWTON_ITERATIONS, class PolyProxy, class DPolyProxy, class TInvPolyApproxProxy>
inline __host__ __device__ float eval_poly_inverse_horner_newton(
    const PolyProxy &poly, const DPolyProxy &dpoly, const TInvPolyApproxProxy &inv_poly_approx, float y, bool &converged
)
{
    // Evaluates the inverse x = f^{-1}(y) of a reference polynomial y=f(x)
    // (given by poly_coefficients) at points y using numerically stable Horner
    // scheme and Newton iterations starting from an approximate solution
    // \\hat{x} = \\hat{f}^{-1}(y) (given by inv_poly_approx) and the
    // polynomials derivative df/dx (given by poly_derivative_coefficients).
    //
    // `converged` is advisory: the polynomial-evaluation rounding error sets
    // a per-coefficient floor on |dx| that for typical FTheta / OpenCV-
    // fisheye fits sits at ~10⁻⁵ in x-units (e.g. ~3.5e-5 when `delta` is a
    // pixel distance of a few hundred), so the `|dx| < 1e-6` threshold is
    // generally not reachable in FP32 even though the final x is accurate.
    // Callers that gate on `converged` should consider that Newton's result
    // is still usable for projection purposes when |dx| oscillates at the
    // FP32 noise floor.

    static_assert(N_NEWTON_ITERATIONS >= 0, "Require at least a single Newton iteration");

    // approximation / starting points - also returned for zero iterations
    auto x = inv_poly_approx.eval_horner(y);

#pragma unroll
    for(auto j = 0; j < N_NEWTON_ITERATIONS; ++j)
    {
        const auto dfdx      = dpoly.eval_horner(x);
        const auto residual  = poly.eval_horner(x) - y;
        const auto dx        = residual / dfdx;
        x                   -= dx;
        if(std::fabs(dx) < 1e-6f)
        {
            // Convergence check: if the change is small enough, we can stop
            converged = true;
            break;
        }
    }

    return x;
}

// ---------------------------------------------------------------------------------------------

// Camera models

/**
 * @brief Checks if a given image point is within the image bounds considering a
 * margin.
 *
 * This function determines whether a specified point in image coordinates lies
 * within the bounds of the image, taking into account a margin factor. The
 * margin is calculated as a fraction of the image resolution.
 *
 * @param image_point The point in image coordinates to check.
 * @param resolution The resolution of the image as an array where resolution[0]
 * is the width and resolution[1] is the height.
 * @param margin_factor The factor by which the margin is calculated. The margin
 * is computed as margin_factor * resolution.
 * @return true if the image point is within the image bounds considering the
 * margin, false otherwise.
 */
__forceinline__ __device__ bool image_point_in_image_bounds_margin(
    const glm::vec2 &image_point, const std::array<uint32_t, 2> &resolution, float margin_factor
)
{
    const float MARGIN_X  = resolution[0] * margin_factor;
    const float MARGIN_Y  = resolution[1] * margin_factor;
    bool valid            = true;
    valid                &= (-MARGIN_X) <= image_point.x && image_point.x < (resolution[0] + MARGIN_X);
    valid                &= (-MARGIN_Y) <= image_point.y && image_point.y < (resolution[1] + MARGIN_Y);
    return valid;
}

struct WorldRay
{
    glm::fvec3 ray_org;
    glm::fvec3 ray_dir;
    bool valid_flag;
};

struct CameraRay
{
    glm::fvec3 ray_dir;
    bool valid_flag;
};

struct ShutterPose
{
    glm::fvec3 t;
    glm::fquat q;

    inline __device__ auto camera_world_position() const -> glm::fvec3
    {
        return glm::rotate(glm::inverse(q), -t);
    }

    inline __device__ auto camera_ray_to_world_ray(const glm::fvec3 &camera_ray) const -> WorldRay
    {
        const auto R_inv = glm::mat3_cast(glm::inverse(q));

        return {-R_inv * t, R_inv * camera_ray, true};
    }
};

// When precise quaternion normalization is needed, CUDA <= 12.8 does not keep
// enough precision under -use_fast_math and needs the explicit operation below.
inline __device__ auto precise_normalize(const glm::fquat &q) -> glm::fquat
{
#if defined(__CUDACC_VER_MAJOR__) \
    && ((__CUDACC_VER_MAJOR__ < 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ <= 8))
    const auto inv_norm = __frsqrt_rn(glm::dot(q, q));
    return q * inv_norm;
#else
    return glm::normalize(q);
#endif
}

// Precomputes the loop-invariant part of a quaternion slerp between two fixed
// endpoints so the rolling-shutter solver can re-interpolate at successive
// relative frame times without recomputing the inter-quaternion angle on every
// iteration. operator()(a) reproduces glm::slerp(q_start, q_end, a) bit-for-bit:
// the only quantities cached are those that are identical across calls (the
// sign-corrected endpoint, the angle and its sine), so dividing by the cached
// sin(angle) is equivalent to glm recomputing it each call.
template<bool kEnabled = true>
struct QuaternionSlerper
{
    glm::fquat x    = glm::fquat{0.f, 0.f, 0.f, 0.f}; // start endpoint
    glm::fquat z    = glm::fquat{0.f, 0.f, 0.f, 0.f}; // end endpoint, sign-flipped onto the short arc
    float angle     = 0.f;                            // angle between x and z
    float sin_angle = 0.f;
    bool linear     = false; // near-parallel endpoints -> component-wise mix

    QuaternionSlerper() = default;

    inline __device__ QuaternionSlerper(const glm::fquat &q_start, const glm::fquat &q_end)
        : x(q_start)
        , z(q_end)
    {
        float cos_theta = glm::dot(q_start, q_end);
        if(cos_theta < 0.f)
        {
            z         = -q_end;
            cos_theta = -cos_theta;
        }
        if(cos_theta > 1.f - glm::epsilon<float>())
        {
            linear = true;
        }
        else
        {
            angle     = glm::acos(cos_theta);
            sin_angle = glm::sin(angle);
        }
    }

    inline __device__ auto operator()(float a) const -> glm::fquat
    {
        if(linear)
        {
            // Component-wise lerp, matching glm::slerp's near-parallel branch.
            // Open-coded rather than glm::lerp(x, z, a) because glm::lerp
            // asserts 0 <= a <= 1, which fails in debug builds when the
            // rolling-shutter frame time lands slightly outside [0, 1].
            return glm::fquat(
                glm::mix(x.w, z.w, a), glm::mix(x.x, z.x, a), glm::mix(x.y, z.y, a), glm::mix(x.z, z.z, a)
            );
        }
        return (glm::sin((1.f - a) * angle) * x + glm::sin(a * angle) * z) / sin_angle;
    }
};

template<>
struct QuaternionSlerper<false>
{
    QuaternionSlerper() = default;

    inline __device__ QuaternionSlerper(const glm::fquat &, const glm::fquat &) { }
};

inline __device__ auto interpolate_shutter_pose(
    float relative_frame_time, const RollingShutterParameters &rolling_shutter_parameters
) -> ShutterPose
{
    const auto t_start = rolling_shutter_parameters.t_start;
    const auto q_start = rolling_shutter_parameters.q_start;
    const auto t_end   = rolling_shutter_parameters.t_end;
    const auto q_end   = rolling_shutter_parameters.q_end;
    // Interpolate a pose linearly for a relative frame time
    const auto t_rs    = (1.f - relative_frame_time) * t_start + relative_frame_time * t_end;
    const auto q_rs    = glm::normalize(glm::slerp(q_start, q_end, relative_frame_time));
    return ShutterPose{t_rs, q_rs};
}

template<class DerivedCameraModel, typename ExternalDistortionModel>
struct BaseCameraModel
{
    // CRTP base class for all camera model types
    struct KernelParameters
    {
        std::array<uint32_t, 2> resolution;
        ShutterType shutter_type;
        ExternalDistortionModel::KernelParameters external_distortion;
    };

    struct Parameters
    {
        std::array<uint32_t, 2> resolution;
        ShutterType shutter_type;
        ExternalDistortionModel external_distortion;

        inline __device__ Parameters(const KernelParameters &kernel_parameters, int camera_index)
            : resolution(kernel_parameters.resolution)
            , shutter_type(kernel_parameters.shutter_type)
            , external_distortion(kernel_parameters.external_distortion, camera_index)
        {
        }
    };

    struct ImagePointReturn
    {
        glm::fvec2 imagePoint;
        bool valid_flag;
    };

    // Apply external distortion before camera projection (forward)
    inline __device__ auto camera_ray_to_image_point(const glm::fvec3 &cam_ray, float margin_factor) const
        -> ImagePointReturn
    {
        auto derived       = static_cast<const DerivedCameraModel *>(this);
        auto distorted_ray = derived->parameters.external_distortion.distort_camera_ray(cam_ray);

        return derived->camera_ray_to_image_point_impl(distorted_ray, margin_factor);
    }

    // Undo external distortion after camera unprojection (inverse)
    inline __device__ CameraRay image_point_to_camera_ray(glm::fvec2 image_point) const
    {
        auto derived = static_cast<const DerivedCameraModel *>(this);
        auto cam_ray = derived->image_point_to_camera_ray_impl(image_point);

        if(cam_ray.valid_flag)
        {
            cam_ray.ray_dir = derived->parameters.external_distortion.undistort_camera_ray(cam_ray.ray_dir);
        }

        return cam_ray;
    }

    // Convert pixel indices (column j, row i) to image-point coordinates.
    // For pixel-based cameras, returns pixel centers (j+0.5, i+0.5).
    // Lidar overrides this to return scaled-angle coordinates.
    inline __device__ glm::fvec2 element_to_image_point(int j, int i) const
    {
        return {(float)j + 0.5f, (float)i + 0.5f};
    }

    // Generate a world ray for pixel (j, i) using rolling shutter parameters.
    // Chains element_to_image_point → image_point_to_world_ray_shutter_pose.
    inline __device__ WorldRay
        element_to_world_ray_shutter_pose(int j, int i, const RollingShutterParameters &rs_params) const
    {
        auto derived      = static_cast<const DerivedCameraModel *>(this);
        const auto img_pt = derived->element_to_image_point(j, i);
        return derived->image_point_to_world_ray_shutter_pose(img_pt, rs_params);
    }

    // Function to compute the relative frame time for a given image point based
    // on the shutter type
    inline __device__ auto shutter_relative_frame_time(const glm::fvec2 &image_point) const -> float
    {
        auto derived = static_cast<const DerivedCameraModel *>(this);

        auto t                 = 0.f;
        const auto &resolution = derived->parameters.resolution;
        switch(derived->parameters.shutter_type)
        {
        case ShutterType::ROLLING_TOP_TO_BOTTOM: t = std::floor(image_point.y) / (resolution[1] - 1); break;

        case ShutterType::ROLLING_LEFT_TO_RIGHT: t = std::floor(image_point.x) / (resolution[0] - 1); break;

        case ShutterType::ROLLING_BOTTOM_TO_TOP:
            t = (resolution[1] - std::ceil(image_point.y)) / (resolution[1] - 1);
            break;

        case ShutterType::ROLLING_RIGHT_TO_LEFT:
            t = (resolution[0] - std::ceil(image_point.x)) / (resolution[0] - 1);
            break;
        }

        return t;
    }

    inline __device__ auto image_point_to_world_ray_shutter_pose(
        const glm::fvec2 &image_point, const RollingShutterParameters &rolling_shutter_parameters
    ) const -> WorldRay
    {
        // Unproject ray and transform to world using shutter pose

        auto derived = static_cast<const DerivedCameraModel *>(this);

        const auto camera_ray = derived->image_point_to_camera_ray(image_point);
        // If the camera ray is invalid, return an invalid world ray
        if(!camera_ray.valid_flag)
        {
            return {glm::fvec3{}, glm::fvec3{}, false};
        }

        return interpolate_shutter_pose(derived->shutter_relative_frame_time(image_point), rolling_shutter_parameters)
            .camera_ray_to_world_ray(camera_ray.ray_dir);
    }

    template<size_t N_ROLLING_SHUTTER_ITERATIONS = 10, bool UsePrecomputedSlerper = false>
    inline __device__ auto world_point_to_image_point_shutter_pose(
        const glm::fvec3 &world_point,
        const RollingShutterParameters &rolling_shutter_parameters,
        float margin_factor,
        QuaternionSlerper<UsePrecomputedSlerper> precomputed_slerper = {}
    ) const -> ImagePointReturn
    {
        // Perform rolling-shutter-based world point to image point projection /
        // optimization

        auto derived = static_cast<const DerivedCameraModel *>(this);

        const auto t_start = rolling_shutter_parameters.t_start;
        const auto q_start = rolling_shutter_parameters.q_start;
        const auto t_end   = rolling_shutter_parameters.t_end;
        const auto q_end   = rolling_shutter_parameters.q_end;

        // Always perform transformation using start pose
        const auto [image_point_start, valid_start]
            = derived->camera_ray_to_image_point(glm::rotate(q_start, world_point) + t_start, margin_factor);

        if(derived->parameters.shutter_type == ShutterType::GLOBAL)
        {
            // Exit early if we have a global shutter sensor
            return {
                {image_point_start.x, image_point_start.y},
                valid_start
            };
        }

        // Do initial transformations using both start and end poses to
        // determine all candidate points and take union of valid projections as
        // iteration starting points
        const auto [image_point_end, valid_end]
            = derived->camera_ray_to_image_point(glm::rotate(q_end, world_point) + t_end, margin_factor);

        // This selection prefers points at the start-of-frame pose over
        // end-of-frame points
        auto init_image_point = glm::fvec2{};
        if(valid_start)
        {
            init_image_point = image_point_start;
        }
        else if(valid_end)
        {
            init_image_point = image_point_end;
        }
        else
        {
            // No valid projection at start or finish -> mark point as invalid.
            // Still return projection result at end of frame
            return {
                {image_point_end.x, image_point_end.y},
                false
            };
        }

        // Compute the new timestamp and project again
        auto image_points_rs_prev = init_image_point;
        auto t_rs                 = glm::fvec3{};
        auto q_rs                 = glm::fquat{};
        bool valid                = true;

        constexpr unsigned kRollingShutterUnroll = ProjectionUTCodegenTraits<DerivedCameraModel>::
            template rolling_shutter_unroll<N_ROLLING_SHUTTER_ITERATIONS>();
        auto prev_relative_frame_time = -std::numeric_limits<float>::max();
#pragma unroll kRollingShutterUnroll
        for(auto j = 0; j < N_ROLLING_SHUTTER_ITERATIONS; ++j)
        {
            const auto relative_frame_time = derived->shutter_relative_frame_time(image_points_rs_prev);

            if(relative_frame_time == prev_relative_frame_time)
            {
                // Same frame time means the same pose and reprojection for
                // all remaining iterations, so this is bit-identical to
                // continuing.
                break;
            }
            prev_relative_frame_time = relative_frame_time;

            t_rs = (1.f - relative_frame_time) * t_start + relative_frame_time * t_end;
            // LiDAR reuses a Gaussian-level slerper across sigma points. Other
            // sensors pass an empty slerper and construct the local one here.
            decltype(auto) slerper
                = gsplat::select_provided_or_construct<UsePrecomputedSlerper, QuaternionSlerper<true>>(
                    precomputed_slerper, q_start, q_end
                );

            // The rolling-shutter optimization loop is sensitive to small
            // pose updates, so use the precise normalization here.
            q_rs = precise_normalize(slerper(relative_frame_time));

            const auto [image_point_rs, valid_rs]
                = derived->camera_ray_to_image_point(glm::rotate(q_rs, world_point) + t_rs, margin_factor);

            image_points_rs_prev = image_point_rs;
            valid                = valid_rs;
        }

        return {
            {image_points_rs_prev.x, image_points_rs_prev.y},
            valid
        };
    }
};

template<typename ExternalDistortionModel>
struct PerfectPinholeCameraModel
    : BaseCameraModel<PerfectPinholeCameraModel<ExternalDistortionModel>, ExternalDistortionModel>
{
    // OpenCV-like pinhole camera model without any distortion

    using Base = BaseCameraModel<PerfectPinholeCameraModel, ExternalDistortionModel>;

    struct KernelParameters : Base::KernelParameters
    {
        const float *__restrict__ Ks;
    };

    struct Parameters : Base::Parameters
    {
        std::array<float, 2> principal_point;
        std::array<float, 2> focal_length;

        inline __device__ Parameters(const KernelParameters &kernel_parameters, int camera_index)
            : Base::Parameters(kernel_parameters, camera_index)
            , principal_point({kernel_parameters.Ks[camera_index * 9 + 2], kernel_parameters.Ks[camera_index * 9 + 5]})
            , focal_length({kernel_parameters.Ks[camera_index * 9 + 0], kernel_parameters.Ks[camera_index * 9 + 4]})
        {
        }
    };

    inline __device__ PerfectPinholeCameraModel(const KernelParameters &kernel_parameters, int camera_index)
        : parameters(kernel_parameters, camera_index)
    {
    }

    Parameters parameters;

    inline __device__ auto camera_ray_to_image_point_impl(const glm::fvec3 &cam_ray, float margin_factor) const ->
        typename Base::ImagePointReturn
    {
        auto image_point = glm::fvec2{0.f, 0.f};

        // Treat all the points behind the camera plane to invalid / projecting
        // to origin
        if(cam_ray.z <= 0.f)
        {
            return {image_point, false};
        }

        // Project using ideal pinhole model
        image_point = (glm::fvec2(cam_ray.x, cam_ray.y) / cam_ray.z)
                        * glm::fvec2(parameters.focal_length[0], parameters.focal_length[1])
                    + glm::fvec2(parameters.principal_point[0], parameters.principal_point[1]);

        // Check if the image points fall within the image, set points that have
        // too large distortion or fall outside the image sensor to invalid
        auto valid  = true;
        valid      &= image_point_in_image_bounds_margin(image_point, parameters.resolution, margin_factor);

        return {image_point, valid};
    }

    inline __device__ CameraRay image_point_to_camera_ray_impl(glm::fvec2 image_point) const
    {
        // Transform the image point to uv coordinate
        const auto uv = (image_point - glm::fvec2{parameters.principal_point[0], parameters.principal_point[1]})
                      / glm::fvec2{parameters.focal_length[0], parameters.focal_length[1]};

        // Unproject the image point to camera ray
        const auto camera_ray = glm::fvec3{uv.x, uv.y, 1.f};

        // Make sure ray is normalized
        return {camera_ray / length(camera_ray), true};
    }
};

template<typename ExternalDistortionModel>
struct OrthographicCameraModel
    : BaseCameraModel<OrthographicCameraModel<ExternalDistortionModel>, ExternalDistortionModel>
{
    using Base = BaseCameraModel<OrthographicCameraModel<ExternalDistortionModel>, ExternalDistortionModel>;

    struct KernelParameters : Base::KernelParameters
    {
        const float *__restrict__ Ks;
    };

    struct Parameters : Base::Parameters
    {
        std::array<float, 2> principal_point;
        std::array<float, 2> focal_length;

        inline __device__ Parameters(const KernelParameters &kernel_parameters, int camera_index)
            : Base::Parameters(kernel_parameters, camera_index)
            , principal_point({kernel_parameters.Ks[camera_index * 9 + 2], kernel_parameters.Ks[camera_index * 9 + 5]})
            , focal_length({kernel_parameters.Ks[camera_index * 9 + 0], kernel_parameters.Ks[camera_index * 9 + 4]})
        {
        }
    };

    inline __device__ OrthographicCameraModel(const KernelParameters &kernel_parameters, int camera_index)
        : parameters(kernel_parameters, camera_index)
    {
    }

    Parameters parameters;

    // External distortion is defined on camera rays. Interpret orthographic
    // image-plane x/y as homogeneous coordinates on the z=1 plane. The
    // distortion model normalizes this ray internally.
    inline __device__ auto orthographic_point_to_distortion_ray(const glm::fvec2 &xy) const -> glm::fvec3
    {
        return {xy.x, xy.y, 1.f};
    }

    // Map a forward-hemisphere distortion ray back to the orthographic plane.
    // Together with orthographic_point_to_distortion_ray this is a bijection
    // for every finite x/y, so identity external distortion remains a no-op.
    inline __device__ auto distortion_ray_to_orthographic_point(const glm::fvec3 &ray) const ->
        typename Base::ImagePointReturn
    {
        if(!(ray.z > 0.f))
        {
            return {
                glm::fvec2{0.f, 0.f},
                false
            };
        }

        const auto point = glm::fvec2{ray.x, ray.y} / ray.z;
        if(!std::isfinite(point.x) || !std::isfinite(point.y))
        {
            return {
                glm::fvec2{0.f, 0.f},
                false
            };
        }

        return {point, true};
    }

    inline __device__ auto camera_ray_to_image_point(const glm::fvec3 &cam_point, float margin_factor) const ->
        typename Base::ImagePointReturn
    {
        // Override the full forward hook, not just camera_ray_to_image_point_impl:
        // the base hook treats its input as a perspective camera ray and applies
        // external distortion before projection. Orthographic projection receives
        // a camera-space point instead, so x/y must stay image-plane coordinates
        // unless explicitly routed through the hemisphere proxy below.
        // The z <= 0 rejection is for 3DGS Gaussian culling, not standard ortho math.
        if constexpr(std::is_same_v<ExternalDistortionModel, gsplat::extdist::EmptyExternalDistortionModel>)
        {
            return camera_ray_to_image_point_impl(cam_point, margin_factor);
        }
        else
        {
            if(cam_point.z <= 0.f)
            {
                return {
                    glm::fvec2{0.f, 0.f},
                    false
                };
            }

            const auto ortho_ray     = orthographic_point_to_distortion_ray({cam_point.x, cam_point.y});
            const auto distorted_ray = parameters.external_distortion.distort_camera_ray(ortho_ray);
            const auto [distorted_point, distortion_valid] = distortion_ray_to_orthographic_point(distorted_ray);
            if(!distortion_valid)
            {
                return {
                    glm::fvec2{0.f, 0.f},
                    false
                };
            }

            return camera_ray_to_image_point_impl({distorted_point.x, distorted_point.y, cam_point.z}, margin_factor);
        }
    }

    inline __device__ auto camera_ray_to_image_point_impl(const glm::fvec3 &cam_point, float margin_factor) const ->
        typename Base::ImagePointReturn
    {
        auto image_point = glm::fvec2{0.f, 0.f};

        if(cam_point.z <= 0.f)
        {
            return {image_point, false};
        }

        image_point
            = glm::fvec2(cam_point.x, cam_point.y) * glm::fvec2(parameters.focal_length[0], parameters.focal_length[1])
            + glm::fvec2(parameters.principal_point[0], parameters.principal_point[1]);

        auto valid  = true;
        valid      &= image_point_in_image_bounds_margin(image_point, parameters.resolution, margin_factor);

        return {image_point, valid};
    }

    // Curiously Recurring Template Pattern interface stub. In practice ortho
    // uses image_point_to_world_ray_shutter_pose below, because a direction-only
    // camera ray would lose the per-pixel x/y origin that defines ortho rays.
    inline __device__ CameraRay image_point_to_camera_ray_impl(glm::fvec2) const
    {
        return {
            glm::fvec3{0.f, 0.f, 1.f},
            true
        };
    }

    inline __device__ auto image_point_to_world_ray_shutter_pose(
        const glm::fvec2 &image_point, const RollingShutterParameters &rolling_shutter_parameters
    ) const -> WorldRay
    {
        const auto uv = (image_point - glm::fvec2{parameters.principal_point[0], parameters.principal_point[1]})
                      / glm::fvec2{parameters.focal_length[0], parameters.focal_length[1]};

        const auto distorted_ray = orthographic_point_to_distortion_ray(uv);
        const auto camera_ray    = parameters.external_distortion.undistort_camera_ray(distorted_ray);
        const auto [camera_point, distortion_valid] = distortion_ray_to_orthographic_point(camera_ray);
        if(!distortion_valid)
        {
            return {glm::fvec3{}, glm::fvec3{}, false};
        }

        const auto shutter_pose
            = interpolate_shutter_pose(this->shutter_relative_frame_time(image_point), rolling_shutter_parameters);
        const auto R_inv      = glm::mat3_cast(glm::inverse(shutter_pose.q));
        // z=0 is intentional: ortho rays start on the image plane at x/y,
        // unlike perspective rays that all originate at the camera center.
        const auto origin_cam = glm::fvec3{camera_point.x, camera_point.y, 0.f};
        const auto dir_cam    = glm::fvec3{0.f, 0.f, 1.f};

        return {R_inv * (origin_cam - shutter_pose.t), R_inv * dir_cam, true};
    }
};

template<typename ExternalDistortionModel, size_t N_MAX_UNDISTORTION_ITERATIONS = 5>
struct OpenCVPinholeCameraModel
    : BaseCameraModel<
          OpenCVPinholeCameraModel<ExternalDistortionModel, N_MAX_UNDISTORTION_ITERATIONS>,
          ExternalDistortionModel
      >
{
    // OpenCV-compatible pinhole camera model

    using Base = BaseCameraModel<
        OpenCVPinholeCameraModel<ExternalDistortionModel, N_MAX_UNDISTORTION_ITERATIONS>,
        ExternalDistortionModel
    >;

    struct KernelParameters : Base::KernelParameters
    {
        const float *__restrict__ Ks;
        const float *__restrict__ radial_coeffs;
        const float *__restrict__ tangential_coeffs;
        const float *__restrict__ thin_prism_coeffs;
    };

    struct Parameters : Base::Parameters
    {
        std::array<float, 2> principal_point;
        std::array<float, 2> focal_length;
        std::array<float, 6> radial_coeffs     = {0.f};
        std::array<float, 2> tangential_coeffs = {0.f};
        std::array<float, 4> thin_prism_coeffs = {0.f};

        inline __device__ Parameters(const KernelParameters &kernel_parameters, int camera_index)
            : Base::Parameters(kernel_parameters, camera_index)
            , principal_point({kernel_parameters.Ks[camera_index * 9 + 2], kernel_parameters.Ks[camera_index * 9 + 5]})
            , focal_length({kernel_parameters.Ks[camera_index * 9 + 0], kernel_parameters.Ks[camera_index * 9 + 4]})
            , radial_coeffs(
                  kernel_parameters.radial_coeffs
                      ? make_array<float, 6>(kernel_parameters.radial_coeffs + camera_index * 6)
                      : std::array<float, 6>{0.f, 0.f, 0.f, 0.f, 0.f, 0.f}
              )
            , tangential_coeffs(
                  kernel_parameters.tangential_coeffs
                      ? make_array<float, 2>(kernel_parameters.tangential_coeffs + camera_index * 2)
                      : std::array<float, 2>{0.f, 0.f}
              )
            , thin_prism_coeffs(
                  kernel_parameters.thin_prism_coeffs
                      ? make_array<float, 4>(kernel_parameters.thin_prism_coeffs + camera_index * 4)
                      : std::array<float, 4>{0.f, 0.f, 0.f, 0.f}
              )
        {
        }
    };

    inline __device__ OpenCVPinholeCameraModel(const KernelParameters &kernel_parameters, int camera_index)
        : parameters(kernel_parameters, camera_index)
    {
    }

    Parameters parameters;

    struct DistortionReturn
    {
        float icD;
        glm::fvec2 delta;
        float r2;
    };

    inline __device__ auto compute_distortion(const glm::fvec2 &uv) const -> DistortionReturn
    {
        // Computes the radial, tangential, and thin-prism distortion given the
        // camera ray
        const auto uv_squared = glm::fvec2(uv[0] * uv[0], uv[1] * uv[1]);
        const auto r2         = uv_squared.x + uv_squared.y;
        const auto a1         = 2.f * uv[0] * uv[1];
        const auto a2         = r2 + 2.f * uv_squared.x;
        const auto a3         = r2 + 2.f * uv_squared.y;

        const auto icD_numerator   = 1.f
                                   + r2
                                         * (parameters.radial_coeffs[0]
                                            + r2 * (parameters.radial_coeffs[1] + r2 * parameters.radial_coeffs[2]));
        const auto icD_denominator = 1.f
                                   + r2
                                         * (parameters.radial_coeffs[3]
                                            + r2 * (parameters.radial_coeffs[4] + r2 * parameters.radial_coeffs[5]));
        // Guard against the rational distortion model having a pole (denominator
        // = 0) at this r².  Setting icD = 0 makes the downstream validity check
        // (icD > 0.8) reject this point, which is the correct behavior — the
        // camera model cannot reliably project at this radius.
        const auto icD             = (std::fabs(icD_denominator) > 1e-8f) ? icD_numerator / icD_denominator : 0.f;

        const auto delta_x = parameters.tangential_coeffs[0] * a1
                           + parameters.tangential_coeffs[1] * a2
                           + r2 * (parameters.thin_prism_coeffs[0] + r2 * parameters.thin_prism_coeffs[1]);
        const auto delta_y = parameters.tangential_coeffs[0] * a3
                           + parameters.tangential_coeffs[1] * a1
                           + r2 * (parameters.thin_prism_coeffs[2] + r2 * parameters.thin_prism_coeffs[3]);

        return {
            icD, glm::fvec2{delta_x, delta_y},
             r2
        };
    }

    inline __device__ auto camera_ray_to_image_point_impl(const glm::fvec3 &cam_ray, float margin_factor) const ->
        typename Base::ImagePointReturn
    {
        auto image_point = glm::fvec2{0.f, 0.f};

        // Treat all the points behind the camera plane to invalid / projecting
        // to origin
        if(cam_ray.z <= 0.f)
        {
            return {image_point, false};
        }

        // Evalutate distortion
        const auto uv_normalized    = glm::fvec2(cam_ray.x, cam_ray.y) / cam_ray.z;
        const auto [icD, delta, r2] = compute_distortion(uv_normalized);

        // auto constexpr k_min_radial_dist = 0.8f, k_max_radial_dist = 1.2f;
        // auto const valid_radial;
        //     (icD > k_min_radial_dist) && (icD < k_max_radial_dist);

        // Note(ruilong): Negative icD means the distortion makes point flipped
        // across the image center. This cannot be produced by real lenses. We
        // use a threshold larger than zero here because we want to skip those
        // rays that are **close** to be flipped. This is important for
        // unscented transform on Gaussians as part of the Gaussian might across
        // the flip boundary.
        const auto valid_radial = icD > 0.8f;

        // Project using ideal pinhole model (apply radial / tangential /
        // thin-prism distortions) in case radial distortion is within limits
        const auto uvND = icD * uv_normalized + delta;
        // if (valid_radial) {
        image_point     = uvND * glm::fvec2(parameters.focal_length[0], parameters.focal_length[1])
                        + glm::fvec2(parameters.principal_point[0], parameters.principal_point[1]);
        // } else {
        //     // If the radial distortion is out-of-limits, the computed
        //     // coordinates will be unreasonable (might even flip signs) -
        //     check
        //     // on which side of the image we overshoot, and set the
        //     coordinates
        //     // out of the image bounds accordingly. The coordinates will be
        //     // clipped to viable range and direction but the exact values
        //     cannot
        //     // be trusted / are still invalid
        //     auto const roi_clipping_radius =
        //         std::hypotf(parameters.resolution[0],
        //         parameters.resolution[1]);
        //     image_point =
        //         (roi_clipping_radius / std::sqrt(r2)) * uv_normalized +
        //         glm::fvec2(
        //             parameters.principal_point[0],
        //             parameters.principal_point[1]
        //         );
        // }

        // Check if the image points fall within the image, set points that have
        // too large distortion or fall outside the image sensor to invalid
        auto valid  = valid_radial;
        valid      &= image_point_in_image_bounds_margin(image_point, parameters.resolution, margin_factor);

        return {image_point, valid};
    }

    inline __device__ glm::fvec2 compute_undistortion_iterative(const glm::fvec2 &image_point) const
    {
        // Iteratively undistorts the image point using the inverse distortion
        // model

        // Initial guess for the undistorted point
        const auto uv_0 = (image_point - glm::fvec2{parameters.principal_point[0], parameters.principal_point[1]})
                        / glm::fvec2{parameters.focal_length[0], parameters.focal_length[1]};

        auto uv = uv_0;
        for(auto j = 0; j < N_MAX_UNDISTORTION_ITERATIONS; ++j)
        {
            // Compute the distortion for the current estimate
            const auto [icD, delta, r2] = compute_distortion(uv);

            // Update the estimate using the inverse distortion model
            const auto uv_next = (uv_0 - delta) / icD;

            // Check for convergence
            constexpr float STOP_UNDISTORTION_SQUARE_ERROR_PX2 = 1e-12f;
            if(const auto residual_vec = uv - uv_next;
               glm::dot(residual_vec, residual_vec) < STOP_UNDISTORTION_SQUARE_ERROR_PX2)
            {
                break;
            }

            uv = uv_next;
        }

        return uv;
    }

    struct JacobianReturn
    {
        float fx, fy, fx_x, fx_y, fy_x, fy_y, valid_flag;
    };

    inline __device__ auto compute_residual_and_jacobian(float x, float y, float xd, float yd) const -> JacobianReturn
    {
        const auto &[k1, k2, k3, k4, k5, k6] = parameters.radial_coeffs;
        const auto &[p1, p2]                 = parameters.tangential_coeffs;
        const auto &[s1, s2, s3, s4]         = parameters.thin_prism_coeffs;

        // let r(x, y) = x^2 + y^2;
        //     alpha(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x,
        //     y)^3; beta(x, y) = 1 + k4 * r(x, y) + k5 * r(x, y) ^2 + k6 * r(x,
        //     y)^3; d(x, y) = alpha(x, y) / beta(x, y);
        const float r     = x * x + y * y;
        const float r2    = r * r;
        const float alpha = 1.0f + r * (k1 + r * (k2 + r * k3));
        const float beta  = 1.0f + r * (k4 + r * (k5 + r * k6));
        const float d     = alpha / beta; // icD

        // Negative icD means the distortion makes point flipped across the
        // image center. This cannot be produced by real lenses.
        if(d <= 0.f)
        {
            return {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, false};
        }

        // The perfect projection is:
        // xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) + s1 * r
        // + s2 * r2; yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 *
        // y^2) + s3 * r + s4 * r2;

        // Let's define
        // fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) +
        // s1 * r + s2 * r2 - xd; fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 *
        // (r(x, y) + 2 * y^2) + s3 * r + s4 * r2 - yd;

        // We are looking for a solution that satisfies
        // fx(x, y) = fy(x, y) = 0;
        float fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) + s1 * r + s2 * r2 - xd;
        float fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) + s3 * r + s4 * r2 - yd;

        // Compute derivative of alpha, beta over r. `f` suffixes pin the
        // multiply chain to fp32 (unsuffixed literals would promote each
        // multiply to fp64).
        const float alpha_r = k1 + r * (2.0f * k2 + r * (3.0f * k3));
        const float beta_r  = k4 + r * (2.0f * k5 + r * (3.0f * k6));

        // Compute derivative of d over [x, y]
        const float d_r = (alpha_r * beta - alpha * beta_r) / (beta * beta);
        const float d_x = 2.0f * x * d_r;
        const float d_y = 2.0f * y * d_r;

        // Compute derivative of fx over x and y.
        float fx_x  = d + d_x * x + 2.0f * p1 * y + 6.0f * p2 * x;
        fx_x       += 2.0f * x * (s1 + 2.0f * s2 * r);
        float fx_y  = d_y * x + 2.0f * p1 * x + 2.0f * p2 * y;
        fx_y       += 2.0f * y * (s1 + 2.0f * s2 * r);

        // Compute derivative of fy over x and y.
        float fy_x  = d_x * y + 2.0f * p2 * y + 2.0f * p1 * x;
        fy_x       += 2.0f * x * (s3 + 2.0f * s4 * r);
        float fy_y  = d + d_y * y + 2.0f * p2 * x + 6.0f * p1 * y;
        fy_y       += 2.0f * y * (s3 + 2.0f * s4 * r);

        return {fx, fy, fx_x, fx_y, fy_x, fy_y, true};
    }

    inline __device__ glm::fvec2 compute_undistortion_newton(const glm::fvec2 &image_point, bool &converged) const
    {
        // Iteratively undistorts the image point using the newton method

        // Initial guess for the undistorted point
        const auto uv_0 = (image_point - glm::fvec2{parameters.principal_point[0], parameters.principal_point[1]})
                        / glm::fvec2{parameters.focal_length[0], parameters.focal_length[1]};

        auto xd = uv_0.x, yd = uv_0.y;
        auto x = xd, y = yd;
        constexpr auto eps = 1e-6f;
        converged          = false;
        int iter           = 0;
        while(iter < N_MAX_UNDISTORTION_ITERATIONS)
        {
            iter++;
            const auto [fx, fy, fx_x, fx_y, fy_x, fy_y, valid] = compute_residual_and_jacobian(x, y, xd, yd);
            if(!valid)
            {
                break;
            }

            // Compute the Jacobian.
            const auto det = fx_y * fy_x - fx_x * fy_y;
            if(fabsf(det) < eps)
            {
                break;
            }

            // Update
            const auto dx  = (fx * fy_y - fy * fx_y) / det;
            const auto dy  = (fy * fx_x - fx * fy_x) / det;
            x             += dx;
            y             += dy;

            // Check for convergence.
            if(fabs(dx) < eps && fabs(dy) < eps)
            {
                converged = true;
                break;
            }
        }
        return {x, y};
    }

    inline __device__ CameraRay image_point_to_camera_ray_impl(glm::fvec2 image_point) const
    {
        // Undistort the image point to uv coordinate. Newton method is more
        // accurate than iterative method, but slower.
        // auto uv = compute_undistortion_iterative(image_point);
        bool valid;
        auto uv = compute_undistortion_newton(image_point, valid);

        // Unproject the undistorted image point to camera ray
        const auto camera_ray = glm::fvec3{uv.x, uv.y, 1.f};

        // Make sure ray is normalized
        return {camera_ray / length(camera_ray), valid};
    }
};

#define PI 3.14159265358979323846f

// solve 1 + ax + bx^2 + cx^3 = 0
inline __host__ __device__ float compute_opencv_fisheye_max_angle(float a, float b, float c)
{
    constexpr float INF = std::numeric_limits<float>::max();

    if(c == 0.0f)
    {
        if(b == 0.0f)
        {
            if(a >= 0.0f)
            {
                return INF;
            }
            else
            {
                return -1.0f / a;
            }
        }
        float delta = a * a - 4.0f * b;
        if(delta >= 0.0f)
        {
            delta = std::sqrt(delta) - a;
            if(delta > 0.0f)
            {
                return 2.0f / delta;
            }
        }
    }
    else
    {
        float boc  = b / c;
        float boc2 = boc * boc;

        float t1    = (9.0f * a * boc - 2.0f * b * boc2 - 27.0f) / c;
        float t2    = 3.0f * a / c - boc2;
        float delta = t1 * t1 + 4.0f * t2 * t2 * t2;

        if(delta >= 0.0f)
        {
            float d2        = std::sqrt(delta);
            float cube_root = std::cbrt((d2 + t1) / 2.0f);
            if(cube_root != 0.0f)
            {
                float soln = (cube_root - (t2 / cube_root) - boc) / 3.0f;
                if(soln > 0.0f)
                {
                    return soln;
                }
            }
        }
        else
        {
            // Complex root case (delta < 0): 3 real roots
            float theta                  = std::atan2(std::sqrt(-delta), t1) / 3.0f;
            constexpr float two_third_pi = 2.0f * PI / 3.0f;

            float t3   = 2.0f * std::sqrt(-t2);
            float soln = INF;
            for(int i: {-1, 0, 1})
            {
                float angle = theta + i * two_third_pi;
                float s     = (t3 * std::cos(angle) - boc) / 3.0f;
                if(s > 0.0f)
                {
                    soln = std::min(soln, s);
                }
            }
            return soln;
        }
    }

    return INF;
}

template<typename ExternalDistortionModel, size_t N_NEWTON_ITERATIONS = 20>
struct OpenCVFisheyeCameraModel
    : BaseCameraModel<OpenCVFisheyeCameraModel<ExternalDistortionModel, N_NEWTON_ITERATIONS>, ExternalDistortionModel>
{
    // OpenCV-compatible fisheye camera model

    using Base = BaseCameraModel<
        OpenCVFisheyeCameraModel<ExternalDistortionModel, N_NEWTON_ITERATIONS>,
        ExternalDistortionModel
    >;

    struct KernelParameters : Base::KernelParameters
    {
        const float *__restrict__ Ks;
        const float *__restrict__ radial_coeffs;
    };

    struct Parameters : Base::Parameters
    {
        std::array<float, 2> principal_point;
        std::array<float, 2> focal_length;
        std::array<float, 4> radial_coeffs = {0.f};

        inline __device__ Parameters(const KernelParameters &kernel_parameters, int camera_index)
            : Base::Parameters(kernel_parameters, camera_index)
            , principal_point({kernel_parameters.Ks[camera_index * 9 + 2], kernel_parameters.Ks[camera_index * 9 + 5]})
            , focal_length({kernel_parameters.Ks[camera_index * 9 + 0], kernel_parameters.Ks[camera_index * 9 + 4]})
            , radial_coeffs(
                  kernel_parameters.radial_coeffs
                      ? make_array<float, 4>(kernel_parameters.radial_coeffs + camera_index * 4)
                      : std::array<float, 4>{0.f, 0.f, 0.f, 0.f}
              )
        {
        }
    };

    inline __device__ OpenCVFisheyeCameraModel(const KernelParameters &kernel_parameters, int camera_index)
        : parameters(kernel_parameters, camera_index)
    {
        // initialize ninth-degree odd-only forward polynomial (mapping angles
        // to normalized distances) theta + k1*theta^3 + k2*theta^5 + k3*theta^7
        // + k4*theta^9
        const auto &[k1, k2, k3, k4] = parameters.radial_coeffs;
        forward_poly_odd             = {1.f, k1, k2, k3, k4};

        // eighth-degree differential of forward polynomial 1 + 3*k1*theta^2 +
        // 5*k2*theta^4 + 7*k3*theta^6 + 9*k4*theta^8
        dforward_poly_even = {1, 3 * k1, 5 * k2, 7 * k3, 9 * k4};

        const auto max_diag_x
            = max(parameters.resolution[0] - parameters.principal_point[0], parameters.principal_point[0]);
        const auto max_diag_y
            = max(parameters.resolution[1] - parameters.principal_point[1], parameters.principal_point[1]);
        const auto max_radius_pixels = std::sqrt(max_diag_x * max_diag_x + max_diag_y * max_diag_y);

        if(k4 == 0)
        {
            max_angle = std::sqrt(compute_opencv_fisheye_max_angle(3.f * k1, 5.f * k2, 7.f * k3));
        }
        else
        {
            std::array<float, 4> ddforward_poly_odd = {6 * k1, 20 * k2, 42 * k3, 72 * k4};
            std::array<float, 1> approx             = {1.57f};

            bool converged = false;
            max_angle      = eval_poly_inverse_horner_newton<N_NEWTON_ITERATIONS>(
                PolynomialProxy<PolynomialType::EVEN, 5>{dforward_poly_even},
                PolynomialProxy<PolynomialType::ODD, 4>{ddforward_poly_odd},
                PolynomialProxy<PolynomialType::EVEN, 1>{approx},
                0.f,
                converged
            );
            if(!converged || max_angle <= 0.f)
            {
                max_angle = std::numeric_limits<float>::max();
            }
        }

        max_angle = min(
            max_angle,
            max(max_radius_pixels / parameters.focal_length[0], max_radius_pixels / parameters.focal_length[1])
        );

        // approximate backward poly (mapping normalized distances to angles)
        // *very crudely* by linear interpolation / equidistant angle model
        // (also assuming image-centered principal point)
        const auto max_normalized_dist = std::max(
            parameters.resolution[0] / 2.f / parameters.focal_length[0],
            parameters.resolution[1] / 2.f / parameters.focal_length[1]
        );
        approx_backward_poly = {0.f, max_angle / max_normalized_dist};
    }

    Parameters parameters;
    std::array<float, 5> forward_poly_odd;
    std::array<float, 5> dforward_poly_even;
    std::array<float, 2> approx_backward_poly;
    float max_angle;

    inline __device__ auto camera_ray_to_image_point_impl(const glm::fvec3 &cam_ray, float margin_factor) const ->
        typename Base::ImagePointReturn
    {
        if(cam_ray.z <= 0.f)
        {
            return {
                {0.f, 0.f},
                false
            };
        }

        // Make sure norm is non-vanishing (norm vanishes for points along the
        // principal-axis)
        auto cam_ray_xy_norm = numerically_stable_norm2(cam_ray.x, cam_ray.y);
        if(cam_ray_xy_norm <= 0.f)
        {
            cam_ray_xy_norm = std::numeric_limits<float>::epsilon();
        }

        const auto theta_full = std::atan2(cam_ray_xy_norm, cam_ray.z);

        // Limit angles to max_angle to prevent projected points to leave valid
        // cone around max_angle. In particular for omnidirectional cameras,
        // this prevents points outside the FOV to be wrongly projected to
        // in-image-domain points because of badly constrained polynomials
        // outside the effective FOV (which is different to the image
        // boundaries).
        //
        // These FOV-clamped projections will be marked as *invalid*
        const auto theta = theta_full < max_angle ? theta_full : max_angle;

        // Evaluate forward polynomial (correspond to the radial distances to
        // the principal point in the normalized image domain (up to focal
        // length scales))
        const auto delta = eval_poly_odd_horner(forward_poly_odd, theta) / cam_ray_xy_norm;

        // Negative delta means the distortion makes point flipped across the
        // image center. This cannot be produced by real lenses.
        if(delta <= 0.f)
        {
            return {
                {0.f, 0.f},
                false
            };
        }

        const auto image_point = glm::fvec2{
            parameters.focal_length[0] * delta * cam_ray.x + parameters.principal_point[0],
            parameters.focal_length[1] * delta * cam_ray.y + parameters.principal_point[1]
        };

        // auto point = glm::fvec2{
        //     cam_ray.x / cam_ray.z, // * parameters.focal_length[0] +
        //     parameters.principal_point[0], cam_ray.y / cam_ray.z // *
        //     parameters.focal_length[1] + parameters.principal_point[1]
        // };
        // printf("point: %f, %f; theta_full: %f, theta: %f, delta: %f,
        // max_angle: %f, image_point: %f, %f, cam_ray: %f, %f, %f\n",
        //     point.x, point.y, theta_full, theta, delta,
        //     max_angle, image_point.x, image_point.y,
        //     cam_ray.x, cam_ray.y, cam_ray.z
        // );
        // printf("parameters: %f, %f, %f, %f\n",
        //     parameters.focal_length[0], parameters.focal_length[1],
        //     parameters.principal_point[0], parameters.principal_point[1]
        // );

        auto valid  = true;
        valid      &= image_point_in_image_bounds_margin(image_point, parameters.resolution, margin_factor);
        valid      &= theta_full < max_angle; // compare against the pre-clamp angle —
                                              // `theta` was clamped to `max_angle` above so the
                                              // post-clamp comparison would be a tautology.

        return {image_point, valid};
    }

    inline __device__ CameraRay image_point_to_camera_ray_impl(glm::fvec2 image_point) const
    {
        // Normalize the image point coordinates
        const auto uv = (image_point - glm::fvec2{parameters.principal_point[0], parameters.principal_point[1]})
                      / glm::fvec2{parameters.focal_length[0], parameters.focal_length[1]};

        // Compute the radial distance from the principal point
        const auto delta = length(uv);

        // Evaluate the inverse polynomial to find the angle theta
        bool converged   = false;
        const auto theta = eval_poly_inverse_horner_newton<N_NEWTON_ITERATIONS>(
            PolynomialProxy<PolynomialType::ODD, 5>{forward_poly_odd},
            PolynomialProxy<PolynomialType::EVEN, 5>{dforward_poly_even},
            PolynomialProxy<PolynomialType::FULL, 2>{approx_backward_poly},
            delta,
            converged
        );

        // Flipped points is not physically meaningful.
        if(theta < 0.f || theta >= max_angle || !converged)
        {
            return {
                glm::fvec3{0.f, 0.f, 1.f},
                false
            };
        }

        // Compute the camera ray and set the ones at the image center to
        // [0,0,1]
        constexpr float MIN_2D_NORM = 1e-6f;
        if(delta >= MIN_2D_NORM)
        {
            // Scale the uv coordinates by the sine of the angle theta
            const auto scale_factor = std::sin(theta) / delta;
            return {
                glm::fvec3{scale_factor * uv.x, scale_factor * uv.y, std::cos(theta)},
                true
            };
        }
        else
        {
            // For points at the image center, return a ray pointing straight
            // ahead
            return {
                glm::fvec3{0.f, 0.f, 1.f},
                true
            };
        }
    }
};

struct FThetaCameraDistortionDeviceParams
{
    inline __device__ FThetaCameraDistortionDeviceParams() { }

    inline __host__ FThetaCameraDistortionDeviceParams(const FThetaCameraDistortionParameters &params)
        : reference_poly(params.reference_poly)
        , pixeldist_to_angle_poly(params.pixeldist_to_angle_poly)
        , angle_to_pixeldist_poly(params.angle_to_pixeldist_poly)
        , max_angle(params.max_angle)
        , linear_cde(params.linear_cde)
    {
    }

    FThetaCameraDistortionParameters::PolynomialType reference_poly;
    std::array<float, FThetaCameraDistortionParameters::PolynomialDegree>
        pixeldist_to_angle_poly; // backward polynomial
    std::array<float, FThetaCameraDistortionParameters::PolynomialDegree> angle_to_pixeldist_poly; // forward polynomial
    float max_angle;
    std::array<float, 3> linear_cde;
};

template<typename ExternalDistortionModel, size_t N_NEWTON_ITERATIONS = 3>
struct FThetaCameraModel
    : BaseCameraModel<FThetaCameraModel<ExternalDistortionModel, N_NEWTON_ITERATIONS>, ExternalDistortionModel>
{
    // FTheta camera model
public:
    using Base
        = BaseCameraModel<FThetaCameraModel<ExternalDistortionModel, N_NEWTON_ITERATIONS>, ExternalDistortionModel>;

    struct KernelParameters : Base::KernelParameters
    {
        const float *__restrict__ Ks;
        FThetaCameraDistortionDeviceParams dist;
    };

    struct Parameters : Base::Parameters
    {
        FThetaCameraDistortionDeviceParams dist;
        std::array<float, 2> principal_point;

        inline __device__ Parameters(const KernelParameters &kernel_parameters, int camera_index)
            : Base::Parameters(kernel_parameters, camera_index)
            , dist(kernel_parameters.dist)
            , principal_point({kernel_parameters.Ks[camera_index * 9 + 2], kernel_parameters.Ks[camera_index * 9 + 5]})
        {
        }
    };

    inline __device__ FThetaCameraModel(const KernelParameters &kernel_parameters, int camera_index)
        : parameters(kernel_parameters, camera_index)
        , dreference_poly{}
    {

        const auto dist = parameters.dist;

        if(dist.reference_poly == FThetaCameraDistortionParameters::PolynomialType::PIXELDIST_TO_ANGLE)
        {
            // compute first derivative of the backwards polynomial
            dreference_poly
                = {1.f * dist.pixeldist_to_angle_poly.at(1),
                   2.f * dist.pixeldist_to_angle_poly.at(2),
                   3.f * dist.pixeldist_to_angle_poly.at(3),
                   4.f * dist.pixeldist_to_angle_poly.at(4),
                   5.f * dist.pixeldist_to_angle_poly.at(5)};
        }
        else
        {
            // compute first derivative of the forward polynomial
            dreference_poly
                = {1.f * dist.angle_to_pixeldist_poly.at(1),
                   2.f * dist.angle_to_pixeldist_poly.at(2),
                   3.f * dist.angle_to_pixeldist_poly.at(3),
                   4.f * dist.angle_to_pixeldist_poly.at(4),
                   5.f * dist.angle_to_pixeldist_poly.at(5)};
        }

        // FThetaCameraModelParameters are defined such that the image coordinate origin corresponds to
        // the center of the first pixel. We therefore need to offset the principal point by half a pixel.
        this->parameters.principal_point[0] += .5f;
        this->parameters.principal_point[1] += .5f;
    }

    Parameters parameters;
    std::array<float, 5> dreference_poly; // coefficient of first derivative of the reference polynomial

    inline __device__ auto camera_ray_to_image_point_impl(const glm::fvec3 &cam_ray, float margin_factor) const ->
        typename Base::ImagePointReturn
    {
        if(cam_ray.z <= 0.f)
        {
            return {
                {0.f, 0.f},
                false
            };
        }

        // Make sure norm is non-vanishing (norm vanishes for points along the principal-axis)
        auto cam_ray_xy_norm = numerically_stable_norm2(cam_ray.x, cam_ray.y);
        if(cam_ray_xy_norm <= 0.f)
        {
            cam_ray_xy_norm = std::numeric_limits<float>::epsilon();
        }

        const auto theta_full = atan2f(cam_ray_xy_norm, cam_ray.z);

        // Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
        // In particular for omnidirectional cameras, this prevents points outside the FOV to be
        // wrongly projected to in-image-domain points because of badly constrained polynomials outside
        // the effective FOV (which is different to the image boundaries).

        // These FOV-clamped projections will be marked as *invalid*
        const auto theta = theta_full < parameters.dist.max_angle ? theta_full : parameters.dist.max_angle;

        // Evaluate forward polynomial, giving delta = f(theta) factors.
        // We ignore the Newton `converged` flag here: the polynomial-eval
        // FP32 noise floor on |dx| sits above the `|dx|<1e-6` threshold for
        // typical FTheta fits (~3.5e-5 when delta is a pixel distance of a
        // few hundred), so the flag stays False even though Newton's `delta`
        // is accurate to FP32. Treating it as a failure would spuriously
        // cull every Gaussian whose centre projects through this branch.
        float delta;
        bool _converged = false;
        if(parameters.dist.reference_poly == FThetaCameraDistortionParameters::PolynomialType::PIXELDIST_TO_ANGLE)
        {
            // bw poly is reference, evaluate its inverse via Newton-based inversion
            delta = eval_poly_inverse_horner_newton<N_NEWTON_ITERATIONS>(
                PolynomialProxy<PolynomialType::FULL, 6>{parameters.dist.pixeldist_to_angle_poly},
                PolynomialProxy<PolynomialType::FULL, 5>{dreference_poly},
                PolynomialProxy<PolynomialType::FULL, 6>{parameters.dist.angle_to_pixeldist_poly},
                theta,
                _converged
            );
        }
        else
        {
            // fw is reference, evaluate it directly
            delta = eval_poly_horner(parameters.dist.angle_to_pixeldist_poly, theta);
        }

        // Apply linear term A=[c,d;e,1] to f(theta)-weighted normalized 2d vectors, relative to principal point
        const auto &[c, d, e] = parameters.dist.linear_cde;
        auto image_point      = delta * (glm::fvec2{cam_ray.x, cam_ray.y} / cam_ray_xy_norm);
        image_point           = glm::fvec2{c * image_point.x + d * image_point.y, e * image_point.x + image_point.y}
                              + glm::fvec2{parameters.principal_point[0], parameters.principal_point[1]};

        auto valid  = true;
        valid      &= image_point_in_image_bounds_margin(image_point, parameters.resolution, margin_factor);
        // Compare against the pre-clamp angle — `theta` was clamped to
        // `max_angle` above so the post-clamp comparison would be a tautology.
        valid      &= theta_full < parameters.dist.max_angle;

        return {image_point, valid};
    }

    inline __device__ CameraRay image_point_to_camera_ray_impl(glm::fvec2 image_point) const
    {
        // Get f(theta)-weighted normalized 2d vectors around principal point,
        // undoing linear term A = [c,d;e;1] via A^-1 = [1,-d;-e,c] / (c-e*d)
        const auto &[c, d, e]  = parameters.dist.linear_cde;
        image_point           -= glm::fvec2{parameters.principal_point[0], parameters.principal_point[1]};
        // The CDE linear matrix [[c,d],[e,1]] must be invertible for
        // unprojection.  A singular matrix (det = c - e*d ≈ 0) means the
        // calibration is degenerate — return an invalid ray.
        const auto cde_det     = c - e * d;
        if(std::fabs(cde_det) < 1e-8f)
        {
            return {
                glm::fvec3{0.f, 0.f, 1.f},
                false
            };
        }
        const auto uv = glm::fvec2{image_point.x - d * image_point.y, -e * image_point.x + c * image_point.y} / cde_det;

        // Compute the radial distance from the principal point
        const auto delta = length(uv);

        // Evaluate backward polynomial to get theta = f^-1(delta) factor
        bool converged;
        float theta;
        if(parameters.dist.reference_poly == FThetaCameraDistortionParameters::PolynomialType::PIXELDIST_TO_ANGLE)
        {
            // bw is reference, evaluate it directly
            converged = true;
            theta     = eval_poly_horner(parameters.dist.pixeldist_to_angle_poly, delta);
        }
        else
        {
            // fw is reference, evaluate its inverse via Newton-based inversion
            converged = false;
            theta     = eval_poly_inverse_horner_newton<N_NEWTON_ITERATIONS>(
                PolynomialProxy<PolynomialType::FULL, 6>{parameters.dist.angle_to_pixeldist_poly},
                PolynomialProxy<PolynomialType::FULL, 5>{dreference_poly},
                PolynomialProxy<PolynomialType::FULL, 6>{parameters.dist.pixeldist_to_angle_poly},
                delta,
                converged
            );
        }

        if(!converged)
        {
            return {
                glm::fvec3{0.f, 0.f, 1.f},
                false
            };
        }

        // Compute the camera ray and set the ones at the image center to
        // [0,0,1]
        constexpr float MIN_2D_NORM = 1e-6f;
        if(delta >= MIN_2D_NORM)
        {
            // Scale the uv coordinates by the sine of the angle theta
            const auto scale_factor = std::sin(theta) / delta;
            return {
                glm::fvec3{scale_factor * uv.x, scale_factor * uv.y, std::cos(theta)},
                true
            };
        }
        else
        {
            // For points at the image center, return a ray pointing straight
            // ahead
            return {
                glm::fvec3{0.f, 0.f, 1.f},
                true
            };
        }
    }
};




template <typename ExternalDistortionModel>
struct EUCMCameraModel : BaseCameraModel<EUCMCameraModel<ExternalDistortionModel>, ExternalDistortionModel> {
    // EUCM camera model

    using Base = BaseCameraModel<EUCMCameraModel, ExternalDistortionModel>;

    struct KernelParameters : Base::KernelParameters {
        const float *__restrict__ Ks;
        const float *__restrict__ tangential_coeffs;
    };

    struct Parameters : Base::Parameters {
        std::array<float, 2> principal_point;
        std::array<float, 2> focal_length;
        std::array<float, 2> tangential_coeffs = {0.f};

        inline __device__ Parameters(const KernelParameters& kernel_parameters, int camera_index)
            : Base::Parameters(kernel_parameters, camera_index)
            , principal_point({kernel_parameters.Ks[camera_index * 9 + 2], kernel_parameters.Ks[camera_index * 9 + 5]})
            , focal_length({kernel_parameters.Ks[camera_index * 9 + 0], kernel_parameters.Ks[camera_index * 9 + 4]})
            , tangential_coeffs(
                kernel_parameters.tangential_coeffs
                ? make_array<float, 2>(kernel_parameters.tangential_coeffs + camera_index * 2)
                : std::array<float, 2>{0.f, 0.f}
            ) {}
    };

    inline __device__ EUCMCameraModel(const KernelParameters& kernel_parameters, int camera_index)
        : parameters(kernel_parameters, camera_index)
    {
    }

    Parameters parameters;

    inline __device__ auto camera_ray_to_image_point_impl(
        glm::fvec3 const &cam_ray, float margin_factor
    ) const -> typename Base::ImagePointReturn {
        auto image_point = glm::fvec2{0.f, 0.f};
        const float eps = 1e-8f;

        // Treat all the points behind the camera plane to invalid / projecting
        // to origin
        if (cam_ray.z <= 0.f)
            return {image_point, false};

        float alpha = parameters.tangential_coeffs[0];
        float beta = parameters.tangential_coeffs[1];

        float rho2 = beta * (cam_ray.x * cam_ray.x + cam_ray.y * cam_ray.y) +
                     cam_ray.z * cam_ray.z;
        float rho = std::sqrt(rho2);
        float den = alpha * rho + (1.f - alpha) * cam_ray.z;
        if (std::fabs(den) < eps)
            return {image_point, false};

        image_point =
            (glm::fvec2(cam_ray.x, cam_ray.y) / den) *
                glm::fvec2(
                    parameters.focal_length[0], parameters.focal_length[1]
                ) +
            glm::fvec2(
                parameters.principal_point[0], parameters.principal_point[1]
            );

        // Check if the image points fall within the image, set points that have
        // too large distortion or fall outside the image sensor to invalid
        auto valid = true;
        valid &= image_point_in_image_bounds_margin(
            image_point, parameters.resolution, margin_factor
        );

        return {image_point, valid};
    }

    inline __device__ CameraRay image_point_to_camera_ray_impl(glm::fvec2 image_point
    ) const {
        const float eps = 1e-8f;
        // Transform the image point to uv coordinate
        auto const uv =
            (image_point -
             glm::fvec2{
                 parameters.principal_point[0], parameters.principal_point[1]
             }) /
            glm::fvec2{parameters.focal_length[0], parameters.focal_length[1]};

        float r2 = uv.x * uv.x + uv.y * uv.y;

        float alpha = parameters.tangential_coeffs[0];
        float beta = parameters.tangential_coeffs[1];
        float gamma = 1.0f - alpha;
        float radicand = 1.0f - (alpha - gamma) * beta * r2;
        if (radicand < 0.f)
            return {{0.f, 0.f, 0.f}, false};

        float helper_den = alpha * std::sqrt(radicand) + gamma;
        if (std::fabs(helper_den) < eps)
            return {{0.f, 0.f, 0.f}, false};

        float helper = (1.0f - alpha * alpha * beta * r2) / helper_den;
        if (std::fabs(helper) < eps)
            return {{0.f, 0.f, 0.f}, false};

        // Unproject the image point to camera ray
        auto const camera_ray = glm::fvec3{uv.x/helper, uv.y/helper, 1.f};

        // Make sure ray is normalized
        return {camera_ray / length(camera_ray), true};
    }
};

// ---------------------------------------------------------------------------------------------

// Gaussian projections

// The approximation of a transformed distribution is performed by an
// UnscentedTransform, which is based on the idea to approximate the transformed
// distribution's momements by mapping a set of sigma points, which are sampled
// according to the input distribution.
//
// In general, unscented transform are *derivative-free* and provide an
// attractive compromisse in terms of accuracy (better approximation quality
// than simple linearized distribution updates) and are faster to evaluate
// compared to more accurate estimations (like Monte Carlo simulations), as they
// provide a "guided" selection of the sample points to transform and require
// much less transformation function evaluations.

// See
//
// - "Some Relations Between Extended and Unscented Kalman Filters" - Gustafsson
// and Hendeby 2012
// - "On Unscented Kalman Filtering for State Estimation of Continuous-Time
// Nonlinear Systems" - Särkkä 2007
//
// for references

struct SigmaPoints
{
    std::array<glm::fvec3, 2 * UnscentedTransformParameters::D + 1> points;
};

inline __device__ auto ut_lambda(const UnscentedTransformParameters &unscented_transform_parameters) -> float
{
    const auto &alpha2 = unscented_transform_parameters.alpha2;
    const auto &kappa  = unscented_transform_parameters.kappa;
    return alpha2 * (UnscentedTransformParameters::D + kappa) - UnscentedTransformParameters::D;
}

inline __device__ auto ut_weights(const UnscentedTransformParameters &unscented_transform_parameters, float lambda)
    -> std::tuple<float, float, float>
{
    const auto &alpha2       = unscented_transform_parameters.alpha2;
    const auto &beta         = unscented_transform_parameters.beta;
    const auto lambda_plus_D = UnscentedTransformParameters::D + lambda;

    const auto mean0       = lambda / lambda_plus_D;
    const auto covariance0 = mean0 + (1 - alpha2 + beta);
    const auto rest        = 1 / (2 * lambda_plus_D);

    return {mean0, covariance0, rest};
}

inline __device__ auto world_gaussian_sigma_points(
    const UnscentedTransformParameters &unscented_transform_parameters,
    const glm::fvec3 &gaussian_world_mean,
    const glm::fvec3 &gaussian_world_scale,
    const glm::fquat &gaussian_world_rot
) -> std::tuple<float, SigmaPoints>
{
    static constexpr auto D = UnscentedTransformParameters::D;
    const auto lambda       = ut_lambda(unscented_transform_parameters);

    // Compute rotation matrix R from quaternion (scaling matrix S is diag(s_i))
    glm::fmat3 R = glm::mat3_cast(gaussian_world_rot);

    // The _factored_ Gaussian covariance parametrization C = (S * R)^T * (S *
    // R) provides a closed form of it's SVD C = U * Σ * U^T with U = R^T and Σ
    // = S^T*S = diag((s_i)^2).

    // Use this closed form SVD to compute sigma points.
    auto ret = SigmaPoints{};

    ret.points[0] = gaussian_world_mean;

#pragma unroll
    for(auto i = 0u; i < D; ++i)
    {
        const auto delta      = std::sqrt(D + lambda) * gaussian_world_scale[i] * R[i];
        // "m + sqrt((n+lambda)*C)_i"
        ret.points[i + 1]     = gaussian_world_mean + delta;
        // "m - sqrt((n+lambda)*C)_i"
        ret.points[i + 1 + D] = gaussian_world_mean - delta;
    }

    return {lambda, ret};
}

struct ImageGaussianReturn
{
    glm::fvec2 mean;
    glm::fmat2 covariance;
    bool valid;
};

template<class CameraModel>
inline __device__ auto world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
    const CameraModel &camera_model,
    const RollingShutterParameters &rolling_shutter_parameters,
    const UnscentedTransformParameters &unscented_transform_parameters,
    const glm::fvec3 &gaussian_world_mean,
    const glm::fvec3 &gaussian_world_scale,
    const glm::fquat &gaussian_world_rot
) -> ImageGaussianReturn
{
    // Compute sigma points for input distribution
    const auto [lambda, sigma_points] = world_gaussian_sigma_points(
        unscented_transform_parameters, gaussian_world_mean, gaussian_world_scale, gaussian_world_rot
    );

    // Transform sigma points / compute approximation of output distribution via
    // sample mean / covariance
    bool valid            = unscented_transform_parameters.require_all_sigma_points_valid;
    auto image_points     = std::array<glm::fvec2, 2 * UnscentedTransformParameters::D + 1>{};
    auto image_mean       = glm::fvec2{0};
    auto image_covariance = glm::fmat2{0};

    auto [mean, covariance, rest] = ut_weights(unscented_transform_parameters, lambda);

    // De-unroll the two 7-point loops only for measured wins. The fixed 7-trip
    // camera loops are cheap enough that the LiDAR register/codegen policy is
    // not a universal improvement.
    constexpr unsigned kUTUnroll = ProjectionUTCodegenTraits<CameraModel>::kUTUnroll;

    // The rolling-shutter slerp endpoints are identical for every sigma point of
    // this Gaussian, so compute the inter-quaternion angle once here and share it
    // across all 7 sigma-point projections when rolling shutter is active.
    using GaussianSlerper = std::conditional_t<
        ProjectionUTCodegenTraits<CameraModel>::kUseGaussianScopeSlerper,
        QuaternionSlerper<true>,
        QuaternionSlerper<false>
    >;
    GaussianSlerper rs_slerper{};
    if constexpr(ProjectionUTCodegenTraits<CameraModel>::kUseGaussianScopeSlerper)
    {
        if(camera_model.parameters.shutter_type != ShutterType::GLOBAL)
        {
            rs_slerper = QuaternionSlerper<>(rolling_shutter_parameters.q_start, rolling_shutter_parameters.q_end);
        }
    }
#pragma unroll kUTUnroll
    for(auto i = 0u; i < std::size(image_points); ++i)
    {
        typename CameraModel::ImagePointReturn image_point_return{};
        image_point_return = camera_model.template world_point_to_image_point_shutter_pose<>(
            sigma_points.points[i],
            rolling_shutter_parameters,
            unscented_transform_parameters.in_image_margin_factor,
            rs_slerper
        );

        const auto [image_point, point_valid] = image_point_return;

        if(unscented_transform_parameters.require_all_sigma_points_valid)
        {
            valid &= point_valid; // all have to be valid
            if(!point_valid)
            {
                // Early exit if invalid
                return {image_mean, image_covariance, false};
            }
        }
        else
        {
            valid |= point_valid; // any valid is sufficient
        }
        image_points[i]  = {image_point.x, image_point.y};
        image_mean      += mean * image_points[i];
        mean             = rest;
    }

    if(!valid)
    {
        // Early exit if invalid
        return {image_mean, image_covariance, false};
    }

#pragma unroll kUTUnroll
    for(auto i = 0u; i < std::size(image_points); ++i)
    {
        const auto image_mean_vec  = image_points[i] - image_mean;
        image_covariance          += covariance * glm::outerProduct(image_mean_vec, image_mean_vec);
        covariance                 = rest;
    }

    return {image_mean, image_covariance, valid};
}

// ---------------------------------------------------------------------------
// Camera model type list: cartesian product of cameras × distortion models
// ---------------------------------------------------------------------------

// Wrappers that adapt each camera model template for CartesianProduct.
template<template<typename> class CameraTemplate>
struct CameraModelWrapper
{
    template<typename D>
    using Apply = CameraTemplate<D>;
};

using CameraModelWrappers = TypeList<
    CameraModelWrapper<PerfectPinholeCameraModel>,
    CameraModelWrapper<OrthographicCameraModel>,
    CameraModelWrapper<OpenCVPinholeCameraModel>,
    CameraModelWrapper<OpenCVFisheyeCameraModel>,
    CameraModelWrapper<FThetaCameraModel>,
    CameraModelWrapper<EUCMCameraModel>
>;

// All camera model types: every camera instantiated with every distortion model.
using CameraModelTypes = CartesianProduct<CameraModelWrappers, gsplat::extdist::ExternalDistortionModelTypes>;

// Camera model kernel parameters variant.
using CameraModelKernelParamsVariant = gsplat::TypeListToKernelParamsVariant<CameraModelTypes>;

// Build camera model KernelParameters for a given camera template and distortion.
//
// Visits the external distortion variant to recover the concrete distortion
// KernelParameters type, then constructs SensorModel<DistortionModel>::KernelParameters
// with the base parameters (resolution, shutter, distortion) plus any
// camera-specific arguments (Ks pointer, distortion coefficient pointers, etc.).
template<template<typename> class SensorModel, typename... Args>
inline auto get_camera_model_kernel_params(
    std::array<uint32_t, 2> resolution,
    ShutterType shutter_type,
    const gsplat::extdist::ExternalDistortionModelKernelParamsVariant &external_distortion_kernel_params,
    Args... args
) -> CameraModelKernelParamsVariant
{
    return std::visit(
        [&](const auto &distortion_kernel_params) -> CameraModelKernelParamsVariant
        {
            using DistortionKernelParams = std::decay_t<decltype(distortion_kernel_params)>;
            using DistortionModel        = gsplat::extdist::DistortionModelFromKernelParams<DistortionKernelParams>;
            return typename SensorModel<DistortionModel>::KernelParameters{
                {
                 resolution, shutter_type,
                 distortion_kernel_params, },
                std::forward<Args>(args)...
            };
        },
        external_distortion_kernel_params
    );
}
