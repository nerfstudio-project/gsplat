/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// camera_torch.h — C++ host wrappers exposed to ext.cpp via TORCH_LIBRARY.
//
// Declares the CustomClassHolder types (OpenCVPinholeProjection,
// NoExternalDistortion, BivariateWindshieldDistortion) and the C++ wrapper
// functions that validate tensor shapes, allocate outputs, and dispatch to the
// CUDA _launch helpers declared in camera_params.h.
//
// All wrappers follow the same contract:
//   - Inputs are CUDA, float32, contiguous.
//   - A `scratch` tensor is returned alongside every forward result so that the
//     autograd graph can pass it to the matching _backward wrapper without
//     re-running the forward.  scratch is an empty (0,) tensor when no grad
//     is needed by any input.
//   - Nullable grad output pointers in the _backward wrappers signal "skip this
//     gradient"; the wrapper passes nullptr to the kernel and returns an empty
//     scratch tensor for that slot.

#pragma once

#include "camera_params.h"
#include "external_distortion_torch.h"

#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/custom_class.h>

#include <array>
#include <tuple>
#include <vector>

namespace gsplat_sensors {

// TorchScript custom class carrying the OpenCV pinhole camera intrinsics as
// per-component float32 tensors.  Shape contract:
//   focal_length       (2,) — [fx, fy]
//   principal_point    (2,) — [cx, cy]
//   radial_coeffs      (6,) — [k1..k6]
//   tangential_coeffs  (2,) — [p1, p2]
//   thin_prism_coeffs  (4,) — [s1..s4]
//   resolution         [width, height]
struct OpenCVPinholeProjection : public torch::CustomClassHolder {
    at::Tensor focal_length;
    at::Tensor principal_point;
    at::Tensor radial_coeffs;
    at::Tensor tangential_coeffs;
    at::Tensor thin_prism_coeffs;
    std::array<int64_t, 2> resolution;

    OpenCVPinholeProjection(
        at::Tensor focal_length,
        at::Tensor principal_point,
        at::Tensor radial_coeffs,
        at::Tensor tangential_coeffs,
        at::Tensor thin_prism_coeffs,
        std::array<int64_t, 2> resolution);

    OpenCVPinholeProjection_KernelParameters to_kernel_params() const;
};

// Validates projection shape and non-null; throws TORCH_CHECK on failure.
void check_projection(const c10::intrusive_ptr<OpenCVPinholeProjection>& projection);

// Returns (H, W, 2) float32 CUDA tensor of pixel-centre (x, y) coordinates.
at::Tensor generate_image_points(int64_t width, int64_t height, c10::Device device);

// ===========================================================================
// Kernel 2 — camera_rays_to_image_points (no external distortion)
// ===========================================================================
// forward: returns (image_points (N,2), valid_flags (N,) bool, scratch (N,6))
std::tuple<at::Tensor, at::Tensor, at::Tensor> camera_rays_to_image_points_opencv_pinhole_no_external(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& camera_rays);

// ===========================================================================
// Kernel 3 — image_points_to_camera_rays (no external distortion)
// ===========================================================================
// forward: returns (camera_rays (N,3), scratch (N,5))
std::tuple<at::Tensor, at::Tensor> image_points_to_camera_rays_opencv_pinhole_no_external(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points);

// ===========================================================================
// Kernel 4 — project_world_points_mean_pose (no external distortion)
// ===========================================================================
// forward: returns (image_points (N,2), valid_flags (N,), timestamps (N,) int64,
//                   pose_t (N,3), pose_r (N,4), scratch (N,9))
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_mean_pose_opencv_pinhole_no_external(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& world_points,
    const at::Tensor& start_translation,
    const at::Tensor& start_rotation,
    const at::Tensor& end_translation,
    const at::Tensor& end_rotation,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us);

// ===========================================================================
// Kernel 5 — project_world_points_shutter_pose (no external distortion)
// ===========================================================================
// forward: same 6-tuple as Kernel 4; scratch is (N,10)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_shutter_pose_opencv_pinhole_no_external(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& world_points,
    const at::Tensor& start_translation,
    const at::Tensor& start_rotation,
    const at::Tensor& end_translation,
    const at::Tensor& end_rotation,
    int64_t width,
    int64_t height,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    double stop_mean_error_px,
    double stop_delta_mean_error_px,
    double initial_relative_time);

// ===========================================================================
// Kernel 6 — image_points_to_world_rays_static_pose (no external distortion)
// ===========================================================================
// forward: returns (world_rays (N,6), timestamps (N,) int64,
//                   pose_t (N,3), pose_r (N,4), scratch (N,5))
// translations (M,3) and rotations (M,4) must each have M >= 1; first row used.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_static_pose_opencv_pinhole_no_external(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& translations,
    const at::Tensor& rotations,
    int64_t timestamp_us);

// ===========================================================================
// Kernel 7 — image_points_to_world_rays_shutter_pose (no external distortion)
// ===========================================================================
// forward: same 5-tuple as Kernel 6; scratch is (N,9)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& start_translation,
    const at::Tensor& start_rotation,
    const at::Tensor& end_translation,
    const at::Tensor& end_rotation,
    int64_t width,
    int64_t height,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us);

// ===========================================================================
// Kernels 2b–7b — backward passes (no external distortion)
// ===========================================================================
// Each backward returns its gradient tuple in the order
// (primary input grad, [pose grads,] intrinsic grads), where intrinsic grads
// are grad_focal_length (2,), grad_principal_point (2,), grad_radial_coeffs (6,),
// grad_tangential_coeffs (2,), grad_thin_prism_coeffs (4,); see each
// declaration's tuple arity for the per-kernel layout.
// Grad tensors for flags=false slots are empty (0,) scratch.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
camera_rays_to_image_points_opencv_pinhole_no_external_backward(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& camera_rays,
    const at::Tensor& grad_image_points,
    const at::Tensor& scratch,
    bool need_camera_ray_grad,
    bool need_focal_length_grad,
    bool need_principal_point_grad,
    bool need_radial_coeffs_grad,
    bool need_tangential_coeffs_grad,
    bool need_thin_prism_coeffs_grad);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_camera_rays_opencv_pinhole_no_external_backward(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& grad_camera_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_focal_length_grad,
    bool need_principal_point_grad,
    bool need_radial_coeffs_grad,
    bool need_tangential_coeffs_grad,
    bool need_thin_prism_coeffs_grad);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
project_world_points_mean_pose_opencv_pinhole_no_external_backward(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& world_points,
    const at::Tensor& start_rotation,
    const at::Tensor& end_rotation,
    const at::Tensor& grad_image_points,
    const at::Tensor& scratch,
    bool need_world_point_grad,
    bool need_start_translation_grad,
    bool need_end_translation_grad,
    bool need_start_rotation_grad,
    bool need_end_rotation_grad,
    bool need_focal_length_grad,
    bool need_principal_point_grad,
    bool need_radial_coeffs_grad,
    bool need_tangential_coeffs_grad,
    bool need_thin_prism_coeffs_grad);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
project_world_points_shutter_pose_opencv_pinhole_no_external_backward(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& world_points,
    const at::Tensor& start_rotation,
    const at::Tensor& end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    double initial_relative_time,
    const at::Tensor& valid_flags,
    const at::Tensor& grad_image_points,
    const at::Tensor& scratch,
    bool need_world_point_grad,
    bool need_start_translation_grad,
    bool need_end_translation_grad,
    bool need_start_rotation_grad,
    bool need_end_rotation_grad,
    bool need_focal_length_grad,
    bool need_principal_point_grad,
    bool need_radial_coeffs_grad,
    bool need_tangential_coeffs_grad,
    bool need_thin_prism_coeffs_grad);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_static_pose_opencv_pinhole_no_external_backward(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& translation,
    const at::Tensor& rotation,
    const at::Tensor& grad_world_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_translation_grad,
    bool need_rotation_grad,
    bool need_focal_length_grad,
    bool need_principal_point_grad,
    bool need_radial_coeffs_grad,
    bool need_tangential_coeffs_grad,
    bool need_thin_prism_coeffs_grad);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external_backward(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& start_rotation,
    const at::Tensor& end_rotation,
    int64_t shutter_type,
    const at::Tensor& grad_world_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_start_translation_grad,
    bool need_end_translation_grad,
    bool need_start_rotation_grad,
    bool need_end_rotation_grad,
    bool need_focal_length_grad,
    bool need_principal_point_grad,
    bool need_radial_coeffs_grad,
    bool need_tangential_coeffs_grad,
    bool need_thin_prism_coeffs_grad);

// ===========================================================================
// Kernels 8–13 — bivariate_windshield distortion variants (forward)
// ===========================================================================
// Same shapes as the no_external counterparts; scratch widths differ
// (see camera_params.h Kernel N comments for per-kernel scratch sizes).

// Kernel 8 — camera_rays_to_image_points (bivariate windshield, forward)
// Returns (image_points (N,2), valid_flags (N,) bool, scratch (N,10))
std::tuple<at::Tensor, at::Tensor, at::Tensor> camera_rays_to_image_points_opencv_pinhole_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& camera_rays);

// Kernel 9 — image_points_to_camera_rays (bivariate windshield, forward)
// Returns (camera_rays (N,3), scratch (N,9))
std::tuple<at::Tensor, at::Tensor> image_points_to_camera_rays_opencv_pinhole_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_mean_pose_opencv_pinhole_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& world_points,
    const at::Tensor& start_translation,
    const at::Tensor& start_rotation,
    const at::Tensor& end_translation,
    const at::Tensor& end_rotation,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& world_points,
    const at::Tensor& start_translation,
    const at::Tensor& start_rotation,
    const at::Tensor& end_translation,
    const at::Tensor& end_rotation,
    int64_t width,
    int64_t height,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    double stop_mean_error_px,
    double stop_delta_mean_error_px,
    double initial_relative_time);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& translations,
    const at::Tensor& rotations,
    int64_t timestamp_us);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& start_translation,
    const at::Tensor& start_rotation,
    const at::Tensor& end_translation,
    const at::Tensor& end_rotation,
    int64_t width,
    int64_t height,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us);

// ===========================================================================
// Kernels 8b–13b — backward passes (bivariate windshield)
// ===========================================================================
// Returns the same tuple as the no_external backward plus
// grad_distortion_coeffs (42,) as the final element.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_backward(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& camera_rays,
    const at::Tensor& grad_image_points,
    const at::Tensor& scratch,
    bool need_camera_ray_grad,
    bool need_focal_length_grad,
    bool need_principal_point_grad,
    bool need_radial_coeffs_grad,
    bool need_tangential_coeffs_grad,
    bool need_thin_prism_coeffs_grad,
    bool need_distortion_coeffs_grad);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_backward(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& grad_camera_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_focal_length_grad,
    bool need_principal_point_grad,
    bool need_radial_coeffs_grad,
    bool need_tangential_coeffs_grad,
    bool need_thin_prism_coeffs_grad,
    bool need_distortion_coeffs_grad);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_backward(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& world_points,
    const at::Tensor& start_rotation,
    const at::Tensor& end_rotation,
    const at::Tensor& grad_image_points,
    const at::Tensor& scratch,
    bool need_world_point_grad,
    bool need_start_translation_grad,
    bool need_end_translation_grad,
    bool need_start_rotation_grad,
    bool need_end_rotation_grad,
    bool need_focal_length_grad,
    bool need_principal_point_grad,
    bool need_radial_coeffs_grad,
    bool need_tangential_coeffs_grad,
    bool need_thin_prism_coeffs_grad,
    bool need_distortion_coeffs_grad);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_backward(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& world_points,
    const at::Tensor& start_rotation,
    const at::Tensor& end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    double initial_relative_time,
    const at::Tensor& valid_flags,
    const at::Tensor& grad_image_points,
    const at::Tensor& scratch,
    bool need_world_point_grad,
    bool need_start_translation_grad,
    bool need_end_translation_grad,
    bool need_start_rotation_grad,
    bool need_end_rotation_grad,
    bool need_focal_length_grad,
    bool need_principal_point_grad,
    bool need_radial_coeffs_grad,
    bool need_tangential_coeffs_grad,
    bool need_thin_prism_coeffs_grad,
    bool need_distortion_coeffs_grad);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_backward(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& translation,
    const at::Tensor& rotation,
    const at::Tensor& grad_world_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_translation_grad,
    bool need_rotation_grad,
    bool need_focal_length_grad,
    bool need_principal_point_grad,
    bool need_radial_coeffs_grad,
    bool need_tangential_coeffs_grad,
    bool need_thin_prism_coeffs_grad,
    bool need_distortion_coeffs_grad);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_backward(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& start_rotation,
    const at::Tensor& end_rotation,
    int64_t shutter_type,
    const at::Tensor& grad_world_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_start_translation_grad,
    bool need_end_translation_grad,
    bool need_start_rotation_grad,
    bool need_end_rotation_grad,
    bool need_focal_length_grad,
    bool need_principal_point_grad,
    bool need_radial_coeffs_grad,
    bool need_tangential_coeffs_grad,
    bool need_thin_prism_coeffs_grad,
    bool need_distortion_coeffs_grad);

} // namespace gsplat_sensors
