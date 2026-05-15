/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// Parameter pack consumed by every OpenCV pinhole projection kernel.
// Passed by value so the struct lands in registers / constant memory.
//
// Consumed by: camera_rays_to_image_points, image_points_to_camera_rays,
//              project_world_points_{mean,shutter}_pose,
//              image_points_to_world_rays_{static,shutter}_pose
//              (and their _backward variants), for both no_external and
//              bivariate_windshield distortion flavours.
struct OpenCVPinholeProjection_KernelParameters {
    const float* __restrict__ focal_length;      // (2,) float32 — [fx, fy]
    const float* __restrict__ principal_point;   // (2,) float32 — [cx, cy]
    const float* __restrict__ radial_coeffs;     // (6,) float32 — [k1..k6]
    const float* __restrict__ tangential_coeffs; // (2,) float32 — [p1, p2]
    const float* __restrict__ thin_prism_coeffs; // (4,) float32 — [s1..s4]
    int64_t width;  // sensor width in pixels; required for rolling-shutter kernels
    int64_t height; // sensor height in pixels; required for rolling-shutter kernels
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
