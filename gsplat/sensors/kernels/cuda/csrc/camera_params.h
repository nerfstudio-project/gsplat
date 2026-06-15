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

// camera_params.h — kernel parameter-pack structs and CUDA launch prototypes.
//
// Defines the plain-old-data structs that are passed by value into every CUDA
// kernel in camera_kernel.cu / camera_kernel_backward.cu, and forward-declares
// the host-side _launch helpers that camera_torch.cpp calls.  All pointers
// carry __restrict__ so the compiler may assume no aliasing between them.

#pragma once

#include "external_distortion_params.h"

#include <cstdint>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

// Hard upper bound on the rolling-shutter iterative solve loop used by
// project_world_points_shutter_pose (forward and backward).  check_max_iterations
// in camera_torch.cpp rejects user-supplied max_iterations values that exceed
// it; raising it requires recompilation.
constexpr int64_t kMaxRollingShutterIterations = 12;

// Hard upper bound on the FTheta Newton-iteration count exposed via the
// FThetaProjection.newton_iterations field. check_ftheta_projection in
// camera_torch.cpp rejects user-supplied values that exceed it; well above
// the 2-3 iterations FTheta needs in practice for convergence.
constexpr int64_t kFThetaMaxNewtonIterations = 32;

// Hard upper bound on the OpenCV-fisheye Newton-iteration count exposed via the
// OpenCVFisheyeProjection.newton_iterations field. check_opencv_fisheye_projection
// in camera_torch.cpp rejects user-supplied values that exceed it.
constexpr int64_t kFisheyeMaxNewtonIterations = 32;

// Parameter pack consumed by every OpenCV pinhole projection kernel.
// Passed by value so the struct lands in registers / constant memory.
//
// Consumed by: camera_rays_to_image_points, image_points_to_camera_rays,
//              project_world_points_{mean,shutter}_pose,
//              image_points_to_world_rays_{static,shutter}_pose
//              (and their _backward variants), for both no_external and
//              bivariate_windshield distortion flavours.

// Cap on the number of polynomial coefficients shipped to FTheta CUDA kernels.
// Host code uses this for shape/stride validation (e.g. fw_poly.numel() >= 6);
// device code references the same constant for register-allocated POD members.
constexpr int kFThetaMaxPolynomialTerms = 6;

struct OpenCVPinholeProjection_KernelParameters {
    const float* __restrict__ focal_length;      // (2,) float32 — [fx, fy]
    const float* __restrict__ principal_point;   // (2,) float32 — [cx, cy]
    const float* __restrict__ radial_coeffs;     // (6,) float32 — [k1..k6]
    const float* __restrict__ tangential_coeffs; // (2,) float32 — [p1, p2]
    const float* __restrict__ thin_prism_coeffs; // (4,) float32 — [s1..s4]
    int64_t width;  // sensor width in pixels; required for rolling-shutter kernels
    int64_t height; // sensor height in pixels; required for rolling-shutter kernels
};

// FThetaProjection_KernelParameters mirrors the OpenCVPinhole layout but for
// the polynomial-radial F-Theta camera. Component pointers are obtained from
// per-component nn.Parameter tensors via const_data_ptr<float>(). The Python
// class exposes only the five differentiable components (pp, fw_poly, bw_poly,
// A, Ainv) and the six scalar config fields below; polynomial derivatives are
// recomputed device-side via ftheta_poly_eval_derivative.
//
// Pointer fields are `const float* __restrict__`: callers building this POD
// must guarantee the five component tensors do not alias one another (the
// per-component nn.Parameter pattern satisfies this trivially).
struct FThetaProjection_KernelParameters {
    const float* __restrict__ principal_point;     // (2,)
    const float* __restrict__ fw_poly;             // (kFThetaMaxPolynomialTerms,)
    const float* __restrict__ bw_poly;             // (kFThetaMaxPolynomialTerms,)
    const float* __restrict__ A;                   // (4,) row-major 2x2
    const float* __restrict__ Ainv;                // (4,) row-major 2x2
    int64_t width;
    int64_t height;
    int64_t reference_polynomial;                  // 0=FORWARD, 1=BACKWARD
    int64_t fw_poly_degree;                        // <= kFThetaMaxPolynomialTerms - 1
    int64_t bw_poly_degree;                        // <= kFThetaMaxPolynomialTerms - 1
    int64_t newton_iterations;
    float max_angle;
    float min_2d_norm;
};

// Cap on the number of OpenCV-fisheye forward-polynomial coefficients shipped to
// the fisheye CUDA kernels. Host code uses this for shape validation
// (forward_poly.numel() == kFisheyeForwardPolyTerms); device code references the
// same constant for the register-allocated POD coefficient array.
constexpr int kFisheyeForwardPolyTerms = 4;

// OpenCVFisheyeProjection_KernelParameters mirrors the FThetaProjection layout
// but for the OpenCV equidistant fisheye model. Component pointers are obtained
// from per-component nn.Parameter tensors via const_data_ptr<float>(). The
// fisheye surface is a reduced FTheta: there is no A/Ainv image-domain affine,
// no separate backward polynomial, and no reference-polynomial selector. The
// four component tensors are principal_point, focal_length, forward_poly (the
// odd-power radial series k1..k4) and approx_backward_factor. Only the first
// three receive gradients; approx_backward_factor is a Newton initial-guess
// factor and is treated as solver config.
//
// Pointer fields are `const float* __restrict__`: callers building this POD
// must guarantee the four component tensors do not alias one another (the
// per-component nn.Parameter pattern satisfies this trivially).
struct OpenCVFisheyeProjection_KernelParameters {
    const float* __restrict__ principal_point;        // (2,) — [cx, cy]
    const float* __restrict__ focal_length;           // (2,) — [fx, fy]
    const float* __restrict__ forward_poly;           // (kFisheyeForwardPolyTerms,) — [k1..k4]
    const float* __restrict__ approx_backward_factor; // (1,)
    int64_t width;
    int64_t height;
    int64_t newton_iterations;
    float max_angle;
    float min_2d_norm;
};

// ===========================================================================
// no_external distortion — forward launch prototypes
// ===========================================================================

// Kernel 1 — generate_image_points
// Fills image_points (H, W, 2) with pixel-centre coordinates (x, y).
void generate_image_points_launch(
    int64_t width,
    int64_t height,
    float* image_points,
    cudaStream_t stream);

// Kernel 2 — camera_rays_to_image_points (forward, no external distortion)
// Projects (N, 3) camera-space rays to (N, 2) image points via OpenCV pinhole
// model.  scratch is (N, 6) float32 and may be nullptr when no grad is needed.
void camera_rays_to_image_points_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* camera_rays,
    float* image_points,
    bool* valid_flags,
    float* scratch,
    cudaStream_t stream);

// Kernel 3 — image_points_to_camera_rays (forward, no external distortion)
// Unprojects (N, 2) image points to (N, 3) normalised camera rays.
// scratch is (N, 5) float32.
void image_points_to_camera_rays_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* image_points,
    float* camera_rays,
    float* scratch,
    cudaStream_t stream);

// Kernel 4 — project_world_points_mean_pose (forward, no external distortion)
// Projects (N, 3) world points using the interpolated camera pose at the
// midpoint between start_timestamp_us and end_timestamp_us.
// scratch is (N, 9) float32; pose_translations (N, 3), pose_rotations (N, 4)
// carry the per-point interpolated pose for reuse in backward.
void project_world_points_mean_pose_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t mean_timestamp_us,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

// Kernel 5 — project_world_points_shutter_pose (forward, no external distortion)
// Projects (N, 3) world points with per-point rolling-shutter pose solve.
// Iterates up to max_iterations to refine the exposure timestamp per point.
// scratch is (N, 10) float32.
void project_world_points_shutter_pose_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

// Kernel 6 — image_points_to_world_rays_static_pose (forward, no external distortion)
// Unprojects (N, 2) image points to (N, 6) world rays using a single static
// pose drawn from translations[0] / rotations[0].  scratch is (N, 5) float32.
void image_points_to_world_rays_static_pose_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* image_points,
    const float* translations,
    const float* rotations,
    int64_t timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

// Kernel 7 — image_points_to_world_rays_shutter_pose (forward, no external distortion)
// Unprojects (N, 2) image points to (N, 6) world rays with per-pixel shutter
// timestamp derived from the pixel's row or column position (per shutter_type).
// scratch is (N, 9) float32.
void image_points_to_world_rays_shutter_pose_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* image_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

// ===========================================================================
// bivariate_windshield distortion — forward launch prototypes
// ===========================================================================

// Kernel 8 — camera_rays_to_image_points (forward, bivariate windshield)
// Same as Kernel 2 but applies the BivariateWindshieldDistortion on top of
// the pinhole projection.  scratch is (N, 10) float32.
void camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* camera_rays,
    float* image_points,
    bool* valid_flags,
    float* scratch,
    cudaStream_t stream);

// Kernel 9 — image_points_to_camera_rays (forward, bivariate windshield)
// Same as Kernel 3 but applies the inverse bivariate windshield distortion to
// the unprojected camera ray after the pinhole unprojection.  scratch is
// (N, 9) float32.
void image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    float* camera_rays,
    float* scratch,
    cudaStream_t stream);

// Kernel 10 — project_world_points_mean_pose (forward, bivariate windshield)
// Same as Kernel 4 plus bivariate distortion.  scratch is (N, 9) float32.
void project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t mean_timestamp_us,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

// Kernel 11 — project_world_points_shutter_pose (forward, bivariate windshield)
// Same as Kernel 5 plus bivariate distortion.  scratch is (N, 10) float32.
void project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

// Kernel 12 — image_points_to_world_rays_static_pose (forward, bivariate windshield)
// Same as Kernel 6 plus bivariate distortion.  scratch is (N, 9) float32.
void image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* translations,
    const float* rotations,
    int64_t timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

// Kernel 13 — image_points_to_world_rays_shutter_pose (forward, bivariate windshield)
// Same as Kernel 7 plus bivariate distortion.  scratch is (N, 12) float32.
void image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

// ===========================================================================
// no_external distortion — backward launch prototypes
// ===========================================================================

// Kernel 2b — camera_rays_to_image_points backward (no external distortion)
// Grad outputs: grad_camera_rays (N, 3); grad_focal_length (2,);
// grad_principal_point (2,); grad_radial_coeffs (6,);
// grad_tangential_coeffs (2,); grad_thin_prism_coeffs (4,).
// Nullable grad pointers indicate that gradient is not needed.
void camera_rays_to_image_points_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* camera_rays,
    const float* grad_image_points,
    float* grad_camera_rays,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    const float* scratch,
    cudaStream_t stream);

// ===========================================================================
// bivariate_windshield distortion — backward launch prototypes
// ===========================================================================

// Kernel 8b — camera_rays_to_image_points backward (bivariate windshield)
// Adds grad_distortion_coeffs (42,) on top of the no_external backward outputs.
void camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* camera_rays,
    const float* grad_image_points,
    float* grad_camera_rays,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// Kernel 9b — image_points_to_camera_rays backward (bivariate windshield)
void image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* grad_camera_rays,
    float* grad_image_points,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// Kernel 10b — project_world_points_mean_pose backward (bivariate windshield)
void project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// Kernel 11b — project_world_points_shutter_pose backward (bivariate windshield)
void project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    float initial_relative_time,
    const bool* valid_flags,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// Kernel 12b — image_points_to_world_rays_static_pose backward (bivariate windshield)
void image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* translation,
    const float* rotation,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_translation,
    float* grad_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// Kernel 13b — image_points_to_world_rays_shutter_pose backward (bivariate windshield)
void image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* start_rotation,
    const float* end_rotation,
    int64_t shutter_type,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// Kernel 3b — image_points_to_camera_rays backward (no external distortion)
void image_points_to_camera_rays_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* image_points,
    const float* grad_camera_rays,
    float* grad_image_points,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    const float* scratch,
    cudaStream_t stream);

// Kernel 4b — project_world_points_mean_pose backward (no external distortion)
void project_world_points_mean_pose_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    const float* scratch,
    cudaStream_t stream);

// Kernel 5b — project_world_points_shutter_pose backward (no external distortion)
void project_world_points_shutter_pose_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    float initial_relative_time,
    const bool* valid_flags,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    const float* scratch,
    cudaStream_t stream);

// Kernel 6b — image_points_to_world_rays_static_pose backward (no external distortion)
void image_points_to_world_rays_static_pose_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* image_points,
    const float* translation,
    const float* rotation,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_translation,
    float* grad_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    const float* scratch,
    cudaStream_t stream);

// Kernel 7b — image_points_to_world_rays_shutter_pose backward (no external distortion)
void image_points_to_world_rays_shutter_pose_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float* image_points,
    const float* start_rotation,
    const float* end_rotation,
    int64_t shutter_type,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_focal_length,
    float* grad_principal_point,
    float* grad_radial_coeffs,
    float* grad_tangential_coeffs,
    float* grad_thin_prism_coeffs,
    const float* scratch,
    cudaStream_t stream);

// =============================================================================
// FTheta launchers -- 12 forward + 12 backward (6 ops x 2 distortions each).
// Naming: <verb>_ftheta_<distortion>_<forward|backward>_launch.
// Per-op scratch strides are listed at the top of ftheta_kernel.cuh; all are
// gated by save_scratch in the host wrapper.
// =============================================================================

// ---- D1 (no_external) ------------------------------------------------------

void camera_rays_to_image_points_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float* camera_rays,
    float* image_points,
    bool* valid_flags,
    float* scratch,
    cudaStream_t stream);

void camera_rays_to_image_points_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float* camera_rays,
    const float* grad_image_points,
    float* grad_camera_rays,
    float* grad_principal_point,
    float* grad_fw_poly,
    float* grad_bw_poly,
    float* grad_A,
    float* grad_Ainv,
    const float* scratch,
    cudaStream_t stream);

// ---- D2 (no_external) ------------------------------------------------------

void image_points_to_camera_rays_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float* image_points,
    float* camera_rays,
    float* scratch,
    cudaStream_t stream);

void image_points_to_camera_rays_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float* image_points,
    const float* grad_camera_rays,
    float* grad_image_points,
    float* grad_principal_point,
    float* grad_fw_poly,
    float* grad_bw_poly,
    float* grad_A,
    float* grad_Ainv,
    const float* scratch,
    cudaStream_t stream);

// ---- D3 (bivariate_windshield) --------------------------------------------

void camera_rays_to_image_points_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* camera_rays,
    float* image_points,
    bool* valid_flags,
    float* scratch,
    cudaStream_t stream);

void camera_rays_to_image_points_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* camera_rays,
    const float* grad_image_points,
    float* grad_camera_rays,
    float* grad_principal_point,
    float* grad_fw_poly,
    float* grad_bw_poly,
    float* grad_A,
    float* grad_Ainv,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

void image_points_to_camera_rays_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    float* camera_rays,
    float* scratch,
    cudaStream_t stream);

void image_points_to_camera_rays_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* grad_camera_rays,
    float* grad_image_points,
    float* grad_principal_point,
    float* grad_fw_poly,
    float* grad_bw_poly,
    float* grad_A,
    float* grad_Ainv,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// ---- D4 (mean-pose, no_external + bivariate) ------------------------------

void project_world_points_mean_pose_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t mean_timestamp_us,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void project_world_points_mean_pose_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_fw_poly,
    float* grad_bw_poly,
    float* grad_A,
    float* grad_Ainv,
    const float* scratch,
    cudaStream_t stream);

void project_world_points_mean_pose_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t mean_timestamp_us,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void project_world_points_mean_pose_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_fw_poly,
    float* grad_bw_poly,
    float* grad_A,
    float* grad_Ainv,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// ---- D5 (static-pose image -> world rays, no_external + bivariate) --------

void image_points_to_world_rays_static_pose_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float* image_points,
    const float* translation,
    const float* rotation,
    int64_t timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void image_points_to_world_rays_static_pose_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float* image_points,
    const float* translation,
    const float* rotation,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_translation,
    float* grad_rotation,
    float* grad_principal_point,
    float* grad_fw_poly,
    float* grad_bw_poly,
    float* grad_A,
    float* grad_Ainv,
    const float* scratch,
    cudaStream_t stream);

void image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* translation,
    const float* rotation,
    int64_t timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* translation,
    const float* rotation,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_translation,
    float* grad_rotation,
    float* grad_principal_point,
    float* grad_fw_poly,
    float* grad_bw_poly,
    float* grad_A,
    float* grad_Ainv,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// ---- D6 (shutter-pose project_world_points, no_external + bivariate) ------

void project_world_points_shutter_pose_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void project_world_points_shutter_pose_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    float initial_relative_time,
    const bool* valid_flags,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_fw_poly,
    float* grad_bw_poly,
    float* grad_A,
    float* grad_Ainv,
    const float* scratch,
    cudaStream_t stream);

void project_world_points_shutter_pose_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void project_world_points_shutter_pose_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    float initial_relative_time,
    const bool* valid_flags,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_fw_poly,
    float* grad_bw_poly,
    float* grad_A,
    float* grad_Ainv,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// ---- D7 (shutter-pose image_points -> world rays, no_external + bivariate) -

void image_points_to_world_rays_shutter_pose_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float* image_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void image_points_to_world_rays_shutter_pose_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float* image_points,
    const float* start_rotation,
    const float* end_rotation,
    int64_t shutter_type,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_fw_poly,
    float* grad_bw_poly,
    float* grad_A,
    float* grad_Ainv,
    const float* scratch,
    cudaStream_t stream);

void image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* start_rotation,
    const float* end_rotation,
    int64_t shutter_type,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_fw_poly,
    float* grad_bw_poly,
    float* grad_A,
    float* grad_Ainv,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// =============================================================================
// OpenCV-fisheye launchers -- 12 forward + 12 backward (6 ops x 2 distortions).
// Naming: <verb>_opencv_fisheye_<distortion>_<forward|backward>_launch.
// Intrinsic grad outputs are grad_principal_point (2,), grad_focal_length (2,)
// and grad_forward_poly (4,); approx_backward_factor receives no gradient so it
// has no grad output slot.
// =============================================================================

// ---- D1 (no_external) ------------------------------------------------------

void camera_rays_to_image_points_opencv_fisheye_no_external_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* camera_rays,
    float* image_points,
    bool* valid_flags,
    float* scratch,
    cudaStream_t stream);

void camera_rays_to_image_points_opencv_fisheye_no_external_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* camera_rays,
    const float* grad_image_points,
    float* grad_camera_rays,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    const float* scratch,
    cudaStream_t stream);

// ---- D2 (no_external) ------------------------------------------------------

void image_points_to_camera_rays_opencv_fisheye_no_external_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* image_points,
    float* camera_rays,
    float* scratch,
    cudaStream_t stream);

void image_points_to_camera_rays_opencv_fisheye_no_external_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* image_points,
    const float* grad_camera_rays,
    float* grad_image_points,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    const float* scratch,
    cudaStream_t stream);

// ---- D1/D2 (bivariate_windshield) -----------------------------------------

void camera_rays_to_image_points_opencv_fisheye_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* camera_rays,
    float* image_points,
    bool* valid_flags,
    float* scratch,
    cudaStream_t stream);

void camera_rays_to_image_points_opencv_fisheye_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* camera_rays,
    const float* grad_image_points,
    float* grad_camera_rays,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

void image_points_to_camera_rays_opencv_fisheye_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    float* camera_rays,
    float* scratch,
    cudaStream_t stream);

void image_points_to_camera_rays_opencv_fisheye_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* grad_camera_rays,
    float* grad_image_points,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// ---- D3 (mean-pose, no_external + bivariate) ------------------------------

void project_world_points_mean_pose_opencv_fisheye_no_external_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t mean_timestamp_us,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void project_world_points_mean_pose_opencv_fisheye_no_external_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    const float* scratch,
    cudaStream_t stream);

void project_world_points_mean_pose_opencv_fisheye_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t mean_timestamp_us,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void project_world_points_mean_pose_opencv_fisheye_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// ---- D4 (shutter-pose project, no_external + bivariate) -------------------

void project_world_points_shutter_pose_opencv_fisheye_no_external_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void project_world_points_shutter_pose_opencv_fisheye_no_external_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    const float* scratch,
    cudaStream_t stream);

void project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* world_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float* image_points,
    bool* valid_flags,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_image_points,
    float* grad_world_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// ---- D5 (static-pose image -> world rays, no_external + bivariate) --------

void image_points_to_world_rays_static_pose_opencv_fisheye_no_external_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* image_points,
    const float* translation,
    const float* rotation,
    int64_t timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void image_points_to_world_rays_static_pose_opencv_fisheye_no_external_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* image_points,
    const float* translation,
    const float* rotation,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_translation,
    float* grad_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    const float* scratch,
    cudaStream_t stream);

void image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* translation,
    const float* rotation,
    int64_t timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* translation,
    const float* rotation,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_translation,
    float* grad_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);

// ---- D6 (shutter-pose image -> world rays, no_external + bivariate) -------

void image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* image_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    const float* image_points,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    const float* scratch,
    cudaStream_t stream);

void image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield_forward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* start_translation,
    const float* start_rotation,
    const float* end_translation,
    const float* end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float* world_rays,
    int64_t* timestamps_us,
    float* pose_translations,
    float* pose_rotations,
    float* scratch,
    cudaStream_t stream);

void image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVFisheyeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float* image_points,
    const float* start_rotation,
    const float* end_rotation,
    const float* grad_world_rays,
    float* grad_image_points,
    float* grad_start_translation,
    float* grad_end_translation,
    float* grad_start_rotation,
    float* grad_end_rotation,
    float* grad_principal_point,
    float* grad_focal_length,
    float* grad_forward_poly,
    float* grad_distortion_coeffs,
    const float* scratch,
    cudaStream_t stream);
