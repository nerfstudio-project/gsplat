// Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

#pragma once

#include <array>
#include <cmath>
#include <limits>

// Silence warnings / errors of the form
//
// __device__ / __host__ annotation is ignored on a function("XXX") that is
// explicitly defaulted on its first declaration
//
// in GLM
#define GLM_ENABLE_EXPERIMENTAL
#pragma nv_diag_suppress = esa_on_defaulted_function_ignored
#include <glm/gtx/matrix_operation.hpp> // needs define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp> // glm rotate
#pragma nv_diag_default = esa_on_defaulted_function_ignored

#include "Cameras.h"

template <typename T, std::size_t N>
__host__ __device__ std::array<T, N> make_array(const T* ptr) {
    std::array<T, N> arr;
    std::copy(ptr, ptr + N, arr.begin());
    return arr;
}

struct RollingShutterParameters {
    glm::fvec3 t_start;
    glm::fquat q_start;
    glm::fvec3 t_end;
    glm::fquat q_end;

    __host__ __device__ RollingShutterParameters(
        const float *se3_start, const float *se3_end
    ) {
        // input is row-major, but glm is column-major
        q_start = glm::quat_cast(glm::mat3(
            se3_start[0], se3_start[4], se3_start[8],
            se3_start[1], se3_start[5], se3_start[9],
            se3_start[2], se3_start[6], se3_start[10]
        ));
        t_start = glm::fvec3(se3_start[3], se3_start[7], se3_start[11]);

        if (se3_end == nullptr) {
            q_end = q_start;
            t_end = t_start;
        } else {
            q_end = glm::quat_cast(glm::mat3(
                se3_end[0], se3_end[4], se3_end[8],
                se3_end[1], se3_end[5], se3_end[9],
                se3_end[2], se3_end[6], se3_end[10]
            ));
            t_end = glm::fvec3(se3_end[3], se3_end[7], se3_end[11]);
        }
    }
};

// ---------------------------------------------------------------------------------------------

// Math helpers (polynomial evaluation / stable norms)

inline __device__ float numerically_stable_norm2(float x, float y) {
    // Computes 2-norm of a [x,y] vector in a numerically stable way
    auto const abs_x = std::fabs(x);
    auto const abs_y = std::fabs(y);
    auto const min = std::fmin(abs_x, abs_y);
    auto const max = std::fmax(abs_x, abs_y);

    if (max <= 0.f)
        return 0.f;

    auto const min_max_ratio = min / max;
    return max * std::sqrt(1.f + min_max_ratio * min_max_ratio);
}

template <size_t N_COEFFS>
inline __device__ float
eval_poly_horner(std::array<float, N_COEFFS> const &poly, float x) {
    // Evaluates a polynomial y=f(x) with
    //
    // f(x) = c_0*x^0 + c_1*x^1 + c_2*x^2 + c_3*x^3 + c_4*x^4 ...
    //
    // given by poly_coefficients c_i at points x using numerically stable
    // Horner scheme.
    //
    // The degree of the polynomial is N_COEFFS - 1

    auto y = float{0};
    for (auto cit = poly.rbegin(); cit != poly.rend(); ++cit)
        y = x * y + (*cit);
    return y;
}

template <size_t N_COEFFS>
inline __device__ float
eval_poly_odd_horner(std::array<float, N_COEFFS> const &poly_odd, float x) {
    // Evaluates an odd-only polynomial y=f(x) with
    //
    // f(x) = c_0*x^1 + c_1*x^3 + c_2*x^5 + c_3*x^7 + c_4*x^9 ...
    //
    // given by poly_coefficients c_i at points x using numerically stable
    // Horner scheme.
    //
    // The degree of the polynomial is 2*N_COEFFS - 1

    return x * eval_poly_horner(
                   poly_odd, x * x
               ); // evaluate x^2-based "regular" polynomial after facting out
                  // one x term
}

template <size_t N_COEFFS>
inline __device__ float
eval_poly_even_horner(std::array<float, N_COEFFS> const &poly_even, float x) {
    // Evaluates an even-only polynomial y=f(x) with
    //
    // f(x) = c_0 + c_1*x^2 + c_2*x^4 + c_3*x^6 + c_4*x^8 ...
    //
    // given by poly_coefficients c_i at points x using numerically stable
    // Horner scheme.
    //
    // The degree of the polynomial is 2*(N_COEFFS - 1)

    return eval_poly_horner(
        poly_even, x * x
    ); // evaluate x^2-substituted "regular" polynomial
}

// Enum to represent the type of polynomial
enum class PolynomialType {
    FULL, // Represents a full polynomial with all terms
    EVEN, // Represents an even-only polynomial
    ODD   // Represents an odd-only polynomial
};

template <PolynomialType POLYNOMIAL_TYPE, size_t N_COEFFS>
struct PolynomialProxy {
    std::array<float, N_COEFFS> const &coeffs;

    // Evaluate the polynomial using Horner's method based on the polynomial
    // type
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

template <
    size_t N_NEWTON_ITERATIONS,
    class PolyProxy,
    class DPolyProxy,
    class TInvPolyApproxProxy>
inline __device__ float eval_poly_inverse_horner_newton(
    PolyProxy const &poly,
    DPolyProxy const &dpoly,
    TInvPolyApproxProxy const &inv_poly_approx,
    float y
) {
    // Evaluates the inverse x = f^{-1}(y) of a reference polynomial y=f(x)
    // (given by poly_coefficients) at points y using numerically stable Horner
    // scheme and Newton iterations starting from an approximate solution
    // \\hat{x} = \\hat{f}^{-1}(y) (given by inv_poly_approx) and the
    // polynomials derivative df/dx (given by poly_derivative_coefficients)

    static_assert(
        N_NEWTON_ITERATIONS >= 0, "Require at least a single Newton iteration"
    );

    // approximation / starting points - also returned for zero iterations
    auto x = inv_poly_approx.eval_horner(y);

#pragma unroll
    for (auto j = 0; j < N_NEWTON_ITERATIONS; ++j) {
        auto const dfdx = dpoly.eval_horner(x);
        auto const residual = poly.eval_horner(x) - y;
        x -= residual / dfdx;
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
__forceinline__ __device__ __host__ bool image_point_in_image_bounds_margin(
    glm::vec2 const &image_point,
    std::array<uint32_t, 2> const &resolution,
    float margin_factor
) {
    const float MARGIN_X = resolution[0] * margin_factor;
    const float MARGIN_Y = resolution[1] * margin_factor;
    bool valid = true;
    valid &= (-MARGIN_X) <= image_point.x &&
             image_point.x < (resolution[0] + MARGIN_X);
    valid &= (-MARGIN_Y) <= image_point.y &&
             image_point.y < (resolution[1] + MARGIN_Y);
    return valid;
}

struct WorldRay {
    glm::fvec3 ray_org;
    glm::fvec3 ray_dir;
};

struct ShutterPose {
    glm::fvec3 t;
    glm::fquat q;

    inline __device__ __host__ auto
    camera_world_position() const -> glm::fvec3 {
        return glm::rotate(glm::inverse(q), -t);
    }

    inline __device__ __host__ auto
    camera_ray_to_world_ray(glm::fvec3 const &camera_ray) const -> WorldRay {
        auto const R_inv = glm::mat3_cast(glm::inverse(q));

        return {-R_inv * t, R_inv * camera_ray};
    }
};

inline __device__ __host__ auto interpolate_shutter_pose(
    float relative_frame_time,
    RollingShutterParameters const &rolling_shutter_parameters
) -> ShutterPose {
    auto const t_start = rolling_shutter_parameters.t_start;
    auto const q_start = rolling_shutter_parameters.q_start;
    auto const t_end = rolling_shutter_parameters.t_end;
    auto const q_end = rolling_shutter_parameters.q_end;
    // Interpolate a pose linearly for a relative frame time
    auto const t_rs =
        (1.f - relative_frame_time) * t_start + relative_frame_time * t_end;
    auto const q_rs = glm::slerp(q_start, q_end, relative_frame_time);
    return ShutterPose{t_rs, q_rs};
}

template <class DerivedCameraModel> struct BaseCameraModel {
    // CRTP base class for all camera model types

    struct Parameters {
        std::array<uint32_t, 2> resolution;
        ShutterType shutter_type;
    };

    // Function to compute the relative frame time for a given image point based
    // on the shutter type
    inline __device__ auto
    shutter_relative_frame_time(glm::fvec2 const &image_point) const -> float {
        auto derived = static_cast<DerivedCameraModel const *>(this);

        auto t = 0.f;
        auto const &resolution = derived->parameters.resolution;
        switch (derived->parameters.shutter_type) {
        case ShutterType::ROLLING_TOP_TO_BOTTOM:
            t = std::floor(image_point.y) / (resolution[1] - 1);
            break;

        case ShutterType::ROLLING_LEFT_TO_RIGHT:
            t = std::floor(image_point.x) / (resolution[0] - 1);
            break;

        case ShutterType::ROLLING_BOTTOM_TO_TOP:
            t = (resolution[1] - std::ceil(image_point.y)) /
                (resolution[1] - 1);
            break;

        case ShutterType::ROLLING_RIGHT_TO_LEFT:
            t = (resolution[0] - std::ceil(image_point.x)) /
                (resolution[0] - 1);
            break;
        }

        return t;
    };

    inline __device__ auto image_point_to_world_ray_shutter_pose(
        glm::fvec2 const &image_point,
        RollingShutterParameters const &rolling_shutter_parameters
    ) const -> WorldRay {
        // Unproject ray and transform to world using shutter pose

        auto derived = static_cast<DerivedCameraModel const *>(this);

        auto const camera_ray = derived->image_point_to_camera_ray(image_point);

        return interpolate_shutter_pose(
                   shutter_relative_frame_time(image_point),
                   rolling_shutter_parameters
        )
            .camera_ray_to_world_ray(camera_ray);
    };

    struct ImagePointReturn {
        glm::fvec2 imagePoint;
        bool valid_flag;
    };

    template <size_t N_ROLLING_SHUTTER_ITERATIONS>
    inline __device__ auto world_point_to_image_point_shutter_pose(
        glm::fvec3 const &world_point,
        RollingShutterParameters const &rolling_shutter_parameters,
        float margin_factor
    ) const -> ImagePointReturn {
        // Perform rolling-shutter-based world point to image point projection /
        // optimization

        auto derived = static_cast<DerivedCameraModel const *>(this);

        auto const t_start = rolling_shutter_parameters.t_start;
        auto const q_start = rolling_shutter_parameters.q_start;
        auto const t_end = rolling_shutter_parameters.t_end;
        auto const q_end = rolling_shutter_parameters.q_end;

        // Always perform transformation using start pose
        auto const [image_point_start, valid_start] =
            derived->camera_ray_to_image_point(
                glm::rotate(q_start, world_point) + t_start, margin_factor
            );

        if (derived->parameters.shutter_type == ShutterType::GLOBAL) {
            // Exit early if we have a global shutter sensor
            return {
                {image_point_start.x, image_point_start.y},
                valid_start
            };
        }

        // Do initial transformations using both start and end poses to
        // determine all candidate points and take union of valid projections as
        // iteration starting points
        auto const [image_point_end, valid_end] =
            derived->camera_ray_to_image_point(
                glm::rotate(q_end, world_point) + t_end, margin_factor
            );

        // This selection prefers points at the start-of-frame pose over
        // end-of-frame points
        auto init_image_point = glm::fvec2{};
        if (valid_start) {
            init_image_point = image_point_start;
        } else if (valid_end) {
            init_image_point = image_point_end;
        } else {
            // No valid projection at start or finish -> mark point as invalid.
            // Still return projection result at end of frame
            return {
                {image_point_end.x, image_point_end.y},
                false
            };
        }

        // Compute the new timestamp and project again
        auto image_points_rs_prev = init_image_point;
        auto relative_frame_time = float{};
        auto t_rs = glm::fvec3{};
        auto q_rs = glm::fquat{};
#pragma unroll
        for (auto j = 0; j < N_ROLLING_SHUTTER_ITERATIONS; ++j) {
            relative_frame_time =
                shutter_relative_frame_time(image_points_rs_prev);

            t_rs = (1.f - relative_frame_time) * t_start +
                   relative_frame_time * t_end;
            q_rs = glm::slerp(q_start, q_end, relative_frame_time);

            auto const [image_point_rs, valid_rs] =
                derived->camera_ray_to_image_point(
                    glm::rotate(q_rs, world_point) + t_rs, margin_factor
                );

            image_points_rs_prev = image_point_rs;
        }

        return {
            {image_points_rs_prev.x, image_points_rs_prev.y},
            true
        };
    }
};

struct PerfectPinholeCameraModel : BaseCameraModel<PerfectPinholeCameraModel> {
    // OpenCV-like pinhole camera model without any distortion

    using Base = BaseCameraModel<PerfectPinholeCameraModel>;

    struct Parameters: Base::Parameters {
        std::array<float, 2> principal_point;
        std::array<float, 2> focal_length;
    };

    __host__ __device__ PerfectPinholeCameraModel(
        Parameters const &parameters)
        : parameters(parameters) {}

    Parameters parameters;

    inline __device__ auto camera_ray_to_image_point(
        glm::fvec3 const &cam_ray, float margin_factor
    ) const -> typename Base::ImagePointReturn {
        auto image_point = glm::fvec2{0.f, 0.f};

        // Treat all the points behind the camera plane to invalid / projecting
        // to origin
        if (cam_ray.z <= 0.f)
            return {image_point, false};

        // Project using ideal pinhole model
        image_point =
            (glm::fvec2(cam_ray.x, cam_ray.y) / cam_ray.z) *
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

    inline __device__ glm::fvec3
    image_point_to_camera_ray(glm::fvec2 image_point) const {
        // Transform the image point to uv coordinate
        auto const uv =
            (image_point -
             glm::fvec2{
                 parameters.principal_point[0], parameters.principal_point[1]
             }) /
            glm::fvec2{parameters.focal_length[0], parameters.focal_length[1]};

        // Unproject the image point to camera ray
        auto const camera_ray = glm::fvec3{uv.x, uv.y, 1.f};

        // Make sure ray is normalized
        return camera_ray / length(camera_ray);
    }
};

template <size_t N_MAX_UNDISTORTION_ITERATIONS = 5>
struct OpenCVPinholeCameraModel : BaseCameraModel<OpenCVPinholeCameraModel<N_MAX_UNDISTORTION_ITERATIONS>> {
    // OpenCV-compatible pinhole camera model

    using Base = BaseCameraModel<
        OpenCVPinholeCameraModel<N_MAX_UNDISTORTION_ITERATIONS>>;

    struct Parameters : Base::Parameters {
        std::array<float, 2> principal_point;
        std::array<float, 2> focal_length;
        std::array<float, 6> radial_coeffs;
        std::array<float, 2> tangential_coeffs;
        std::array<float, 4> thin_prism_coeffs;
    };

    __host__ __device__ OpenCVPinholeCameraModel(
        Parameters const &parameters,
        float stop_undistortion_square_error_px2 = 1e-12
    )
        : parameters(parameters),
          undistortion_stop_square_error_px2(stop_undistortion_square_error_px2
          ) {}

    Parameters parameters;
    float undistortion_stop_square_error_px2;

    struct DistortionReturn {
        float icD;
        glm::fvec2 delta;
        float r2;
    };

    inline __device__ auto compute_distortion(glm::fvec2 const &uv
    ) const -> DistortionReturn {
        // Computes the radial, tangential, and thin-prism distortion given the
        // camera ray
        auto const uv_squared = glm::fvec2(uv[0] * uv[0], uv[1] * uv[1]);
        auto const r2 = uv_squared.x + uv_squared.y;
        auto const a1 = 2.f * uv[0] * uv[1];
        auto const a2 = r2 + 2.f * uv_squared.x;
        auto const a3 = r2 + 2.f * uv_squared.y;

        auto const icD_numerator =
            1.f + r2 * (parameters.radial_coeffs[0] +
                        r2 * (parameters.radial_coeffs[1] +
                              r2 * parameters.radial_coeffs[2]));
        auto const icD_denominator =
            1.f + r2 * (parameters.radial_coeffs[3] +
                        r2 * (parameters.radial_coeffs[4] +
                              r2 * parameters.radial_coeffs[5]));
        auto const icD = icD_numerator / icD_denominator;

        auto const delta_x = parameters.tangential_coeffs[0] * a1 +
                             parameters.tangential_coeffs[1] * a2 +
                             r2 * (parameters.thin_prism_coeffs[0] +
                                   r2 * parameters.thin_prism_coeffs[1]);
        auto const delta_y = parameters.tangential_coeffs[0] * a3 +
                             parameters.tangential_coeffs[1] * a1 +
                             r2 * (parameters.thin_prism_coeffs[2] +
                                   r2 * parameters.thin_prism_coeffs[3]);

        return {icD, glm::fvec2{delta_x, delta_y}, r2};
    }

    inline __device__ auto camera_ray_to_image_point(
        glm::fvec3 const &cam_ray, float margin_factor
    ) const -> typename Base::ImagePointReturn {
        auto image_point = glm::fvec2{0.f, 0.f};

        // Treat all the points behind the camera plane to invalid / projecting
        // to origin 
        if (cam_ray.z <= 0.f)
            return {image_point, false};

        // Evalutate distortion
        auto const uv_normalized = glm::fvec2(cam_ray.x, cam_ray.y) / cam_ray.z;
        auto const [icD, delta, r2] = compute_distortion(uv_normalized);

        auto constexpr k_min_radial_dist = 0.8f, k_max_radial_dist = 1.2f;
        auto const valid_radial =
            (icD > k_min_radial_dist) && (icD < k_max_radial_dist);

        // Project using ideal pinhole model (apply radial / tangential /
        // thin-prism distortions) in case radial distortion is within limits
        auto const uvND = icD * uv_normalized + delta;

        if (valid_radial) {
            image_point =
                uvND *
                    glm::fvec2(
                        parameters.focal_length[0], parameters.focal_length[1]
                    ) +
                glm::fvec2(
                    parameters.principal_point[0], parameters.principal_point[1]
                );
        } else {
            // If the radial distortion is out-of-limits, the computed
            // coordinates will be unreasonable (might even flip signs) - check
            // on which side of the image we overshoot, and set the coordinates
            // out of the image bounds accordingly. The coordinates will be
            // clipped to viable range and direction but the exact values cannot
            // be trusted / are still invalid
            auto const roi_clipping_radius =
                std::hypotf(parameters.resolution[0], parameters.resolution[1]);
            image_point =
                (roi_clipping_radius / std::sqrt(r2)) * uv_normalized +
                glm::fvec2(
                    parameters.principal_point[0], parameters.principal_point[1]
                );
        }

        // Check if the image points fall within the image, set points that have
        // too large distortion or fall outside the image sensor to invalid
        auto valid = valid_radial;
        valid &= image_point_in_image_bounds_margin(
            image_point, parameters.resolution, margin_factor
        );

        return {image_point, valid};
    }

    inline __device__ glm::fvec2
    compute_undistortion_iterative(glm::fvec2 const &image_point) const {
        // Iteratively undistorts the image point using the inverse distortion
        // model

        // Initial guess for the undistorted point
        auto const uv_0 =
            (image_point -
             glm::fvec2{
                 parameters.principal_point[0], parameters.principal_point[1]
             }) /
            glm::fvec2{parameters.focal_length[0], parameters.focal_length[1]};

        auto uv = uv_0;
        for (auto j = 0; j < N_MAX_UNDISTORTION_ITERATIONS; ++j) {
            // Compute the distortion for the current estimate
            auto const [icD, delta, r2] = compute_distortion(uv);

            // Update the estimate using the inverse distortion model
            auto const uv_next = (uv_0 - delta) / icD;

            // Check for convergence
            if (auto const residual_vec = uv - uv_next;
                glm::dot(residual_vec, residual_vec) <
                undistortion_stop_square_error_px2)
                break;

            uv = uv_next;
        }

        return uv;
    }

    inline __device__ glm::fvec3
    image_point_to_camera_ray(glm::fvec2 image_point) const {
        // Undistort the image point to uv coordinate
        auto const uv = compute_undistortion_iterative(image_point);

        // Unproject the undistorted image point to camera ray
        auto const camera_ray = glm::fvec3{uv.x, uv.y, 1.f};

        // Make sure ray is normalized
        return camera_ray / length(camera_ray);
    }
};

template <size_t N_NEWTON_ITERATIONS = 3>
struct OpenCVFisheyeCameraModel : BaseCameraModel<OpenCVFisheyeCameraModel<N_NEWTON_ITERATIONS>> {
    // OpenCV-compatible fisheye camera model

    using Base = BaseCameraModel<OpenCVFisheyeCameraModel<N_NEWTON_ITERATIONS>>;

    struct Parameters : Base::Parameters {
        std::array<float, 2> principal_point;
        std::array<float, 2> focal_length;
        std::array<float, 4> radial_coeffs;
        float max_angle;
    };

    __host__ __device__ OpenCVFisheyeCameraModel(
        Parameters const &parameters,
        float min_2d_norm = 1e-6f
    )
        : parameters(parameters), min_2d_norm(min_2d_norm) {
        // initialize ninth-degree odd-only forward polynomial (mapping angles
        // to normalized distances) theta + k1*theta^3 + k2*theta^5 + k3*theta^7
        // + k4*theta^9
        auto const k1 = parameters.radial_coeffs[0];
        auto const k2 = parameters.radial_coeffs[1];
        auto const k3 = parameters.radial_coeffs[2];
        auto const k4 = parameters.radial_coeffs[3];
        forward_poly_odd = {1.f, k1, k2, k3, k4};

        // eighth-degree differential of forward polynomial 1 + 3*k1*theta^2 +
        // 5*k2*theta^4 + 7*k3*theta^8 + 9*k4*theta^8
        dforward_poly_even = {1, 3 * k1, 5 * k2, 7 * k3, 9 * k4};

        // approximate backward poly (mapping normalized distances to angles)
        // *very crudely* by linear interpolation / equidistant angle model
        // (also assuming image-centered principal point)
        auto const max_normalized_dist = std::max(
            parameters.resolution[0] / 2.f / parameters.focal_length[0],
            parameters.resolution[1] / 2.f / parameters.focal_length[1]
        );
        approx_backward_poly = {
            0.f, parameters.max_angle / max_normalized_dist
        };
    }

    Parameters parameters;
    float min_2d_norm;
    std::array<float, 5> forward_poly_odd;
    std::array<float, 5> dforward_poly_even;
    std::array<float, 2> approx_backward_poly;

    inline __device__ auto camera_ray_to_image_point(
        glm::fvec3 const &cam_ray, float margin_factor
    ) const -> typename Base::ImagePointReturn {
        // Make sure norm is non-vanishing (norm vanishes for points along the
        // principal-axis)
        auto cam_ray_xy_norm = numerically_stable_norm2(cam_ray.x, cam_ray.y);
        if (cam_ray_xy_norm <= 0.f)
            cam_ray_xy_norm = std::numeric_limits<float>::epsilon();

        auto const theta_full = std::atan2(cam_ray_xy_norm, cam_ray.z);

        // Limit angles to max_angle to prevent projected points to leave valid
        // cone around max_angle. In particular for omnidirectional cameras,
        // this prevents points outside the FOV to be wrongly projected to
        // in-image-domain points because of badly constrained polynomials
        // outside the effective FOV (which is different to the image
        // boundaries).
        //
        // These FOV-clamped projections will be marked as *invalid*
        auto const theta = theta_full < parameters.max_angle
                               ? theta_full
                               : parameters.max_angle;

        // Evaluate forward polynomial (correspond to the radial distances to
        // the principal point in the normalized image domain (up to focal
        // length scales))
        auto const delta =
            eval_poly_odd_horner(forward_poly_odd, theta) / cam_ray_xy_norm;

        auto const image_point = glm::fvec2{
            parameters.focal_length[0] * delta * cam_ray.x +
                parameters.principal_point[0],
            parameters.focal_length[1] * delta * cam_ray.y +
                parameters.principal_point[1]
        };

        auto valid = true;
        valid &= image_point_in_image_bounds_margin(
            image_point, parameters.resolution, margin_factor
        );
        valid &= theta <
                 parameters
                     .max_angle; // explicitly check for strictly smaller angles
                                 // to classify FOV-clamped points as invalid

        return {image_point, valid};
    }

    inline __device__ glm::fvec3
    image_point_to_camera_ray(glm::fvec2 image_point) const {
        // Normalize the image point coordinates
        auto const uv =
            (image_point -
             glm::fvec2{
                 parameters.principal_point[0], parameters.principal_point[1]
             }) /
            glm::fvec2{parameters.focal_length[0], parameters.focal_length[1]};

        // Compute the radial distance from the principal point
        auto const delta = length(uv);

        // Evaluate the inverse polynomial to find the angle theta
        auto const theta = eval_poly_inverse_horner_newton<N_NEWTON_ITERATIONS>(
            PolynomialProxy<PolynomialType::ODD, 5>{forward_poly_odd},
            PolynomialProxy<PolynomialType::EVEN, 5>{dforward_poly_even},
            PolynomialProxy<PolynomialType::FULL, 2>{approx_backward_poly},
            delta
        );

        // Compute the camera ray and set the ones at the image center to
        // [0,0,1]
        if (delta >= min_2d_norm) {
            // Scale the uv coordinates by the sine of the angle theta
            auto const scale_factor = std::sin(theta) / delta;
            return glm::fvec3{
                scale_factor * uv.x, scale_factor * uv.y, std::cos(theta)
            };
        } else {
            // For points at the image center, return a ray pointing straight
            // ahead
            return glm::fvec3{0.f, 0.f, 1.f};
        }
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

struct SigmaPoints {
    std::array<glm::fvec3, 2 * 3 + 1> points;
    std::array<float, 2 * 3 + 1> weights_mean;
    std::array<float, 2 * 3 + 1> weights_covariance;
};

inline __device__ auto world_gaussian_sigma_points(
    UnscentedTransformParameters const &unscented_transform_parameters,
    glm::fvec3 const &gaussian_world_mean,
    glm::fvec3 const &gaussian_world_scale,
    glm::fquat const &gaussian_world_rot
) -> SigmaPoints {
    size_t constexpr static D = 3;
    auto const &alpha = unscented_transform_parameters.alpha;
    auto const &beta = unscented_transform_parameters.beta;
    auto const &kappa = unscented_transform_parameters.kappa;
    auto const lambda = alpha * alpha * (D + kappa) - D;

    // Compute rotation matrix R from quaternion (scaling matrix S is diag(s_i))
    glm::fmat3 R = glm::mat3_cast(gaussian_world_rot);

    // The _factored_ Gaussian covariance parametrization C = (S * R)^T * (S *
    // R) provides a closed form of it's SVD C = U * Σ * U^T with U = R^T and Σ
    // = S^T*S = diag((s_i)^2).

    // Use this closed form SVD to compute sigma points and weights from a (D +
    // lambda)-scaled covariance C (cf. "On Unscented Kalman Filtering for State
    // Estimation of Continuous-Time Nonlinear Systems" - Särkkä 2007). In
    // particular, this means that the singular values are given by σ_i = s_i
    // and the singular vectors u_i are the columns of R
    auto ret = SigmaPoints{};

    ret.points[0] = gaussian_world_mean;

#pragma unroll
    for (auto i = 0u; i < D; ++i) {
        auto const delta =
            std::sqrt(D + lambda) * gaussian_world_scale[i] * R[i];
        // "m + sqrt((n+lambda)*C)_i"
        ret.points[i + 1] = gaussian_world_mean + delta;
        // "m - sqrt((n+lambda)*C)_i"
        ret.points[i + 1 + D] = gaussian_world_mean - delta;
    }

    // Compute weights
    ret.weights_mean[0] = lambda / (D + lambda);
    ret.weights_covariance[0] =
        lambda / (D + lambda) + (1 - alpha * alpha + beta);

#pragma unroll
    for (auto i = 0u; i < 2 * D; ++i) {
        ret.weights_mean[i + 1] = 1 / (2 * (D + lambda));
        ret.weights_covariance[i + 1] = 1 / (2 * (D + lambda));
    }

    return ret;
}

struct ImageGaussianReturn {
    glm::fvec2 mean;
    glm::fmat2 covariance;
    bool valid;
};

template <size_t N_ROLLING_SHUTTER_ITERATIONS, class CameraModel>
inline __device__ auto
world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
    CameraModel const &camera_model,
    RollingShutterParameters const &rolling_shutter_parameters,
    UnscentedTransformParameters const &unscented_transform_parameters,
    glm::fvec3 const &gaussian_world_mean,
    glm::fvec3 const &gaussian_world_scale,
    glm::fquat const &gaussian_world_rot
) -> ImageGaussianReturn {
    // Compute sigma points for input distribution
    auto const sigma_points = world_gaussian_sigma_points(
        unscented_transform_parameters,
        gaussian_world_mean,
        gaussian_world_scale,
        gaussian_world_rot
    );

    // Transform sigma points / compute approximation of output distribution via
    // sample mean / covariance
    bool valid = unscented_transform_parameters.require_all_sigma_points_valid;
    auto image_points = std::array<glm::fvec2, 2 * 3 + 1>{};
    auto image_mean = glm::fvec2{0};
    auto image_covariance = glm::fmat2{0};
#pragma unroll
    for (auto i = 0u; i < std::size(image_points); ++i) {
        auto const [image_point, point_valid] =
            // annotate with 'template' to avoid warnings: #174-D: expression
            // has no effect
            camera_model.template world_point_to_image_point_shutter_pose<
                N_ROLLING_SHUTTER_ITERATIONS>(
                sigma_points.points[i],
                rolling_shutter_parameters,
                unscented_transform_parameters.in_image_margin_factor
            );

        if (unscented_transform_parameters.require_all_sigma_points_valid) {
            valid &= point_valid; // all have to be valid
            if (!point_valid) {
                // Early exit if invalid
                return {image_mean, image_covariance, false};
            }
        } else {
            valid |= point_valid; // any valid is sufficient
        }

        image_points[i] = {image_point.x, image_point.y};

        image_mean += sigma_points.weights_mean[i] * image_points[i];
    }

    if (!valid) {
        // Early exit if invalid
        return {image_mean, image_covariance, false};
    }

#pragma unroll
    for (auto i = 0u; i < std::size(image_points); ++i) {
        auto const image_mean_vec = image_points[i] - image_mean;
        image_covariance += sigma_points.weights_covariance[i] *
                            glm::outerProduct(image_mean_vec, image_mean_vec);
    }

    return {image_mean, image_covariance, valid};
}

