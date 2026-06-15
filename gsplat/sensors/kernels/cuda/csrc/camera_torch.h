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

// TorchScript custom class carrying the FTheta camera intrinsics as
// per-component float32 tensors.  Shape contract:
//   principal_point        (2,) — [cx, cy]
//   fw_poly                (6,) zero-padded; active terms = fw_poly_degree + 1
//   bw_poly                (6,) zero-padded; active terms = bw_poly_degree + 1
//   A                      (4,) row-major 2x2
//   resolution             [width, height]
//
// Ainv is computed on demand from A (closed-form 2x2 inverse); only A is a
// stored intrinsic.
struct FThetaProjection : public torch::CustomClassHolder {
    at::Tensor principal_point;
    at::Tensor fw_poly;
    at::Tensor bw_poly;
    at::Tensor A;
    // Backs the const float* Ainv pointer in FThetaProjection_KernelParameters;
    // owned by the struct to outlive the returned params.
    mutable at::Tensor Ainv_cache;
    std::array<int64_t, 2> resolution;
    int64_t reference_polynomial;
    int64_t fw_poly_degree;
    int64_t bw_poly_degree;
    int64_t newton_iterations;
    double max_angle;
    double min_2d_norm;

    FThetaProjection(
        at::Tensor principal_point,
        at::Tensor fw_poly,
        at::Tensor bw_poly,
        at::Tensor A,
        std::array<int64_t, 2> resolution,
        int64_t reference_polynomial,
        int64_t fw_poly_degree,
        int64_t bw_poly_degree,
        int64_t newton_iterations,
        double max_angle,
        double min_2d_norm);

    FThetaProjection_KernelParameters to_kernel_params() const;

    // Compute the 2x2 inverse of A on demand and return it as a flat (4,)
    // tensor with the same device/dtype/options as A.
    at::Tensor compute_ainv() const;
};

// Validates projection shape and non-null; throws TORCH_CHECK on failure.
void check_ftheta_projection(const c10::intrusive_ptr<FThetaProjection>& projection);

// TorchScript custom class carrying the OpenCV equidistant fisheye camera
// intrinsics as per-component float32 tensors.  Shape contract:
//   principal_point        (2,) — [cx, cy]
//   focal_length           (2,) — [fx, fy]
//   forward_poly           (4,) — [k1, k2, k3, k4]
//   approx_backward_factor (1,) — Newton initial-guess factor
//   resolution             [width, height]
//
// This is a reduced FThetaProjection: there is no A/Ainv affine, no backward
// polynomial, and no reference-polynomial selector.
struct OpenCVFisheyeProjection : public torch::CustomClassHolder {
    at::Tensor principal_point;
    at::Tensor focal_length;
    at::Tensor forward_poly;
    at::Tensor approx_backward_factor;
    std::array<int64_t, 2> resolution;
    int64_t newton_iterations;
    double max_angle;
    double min_2d_norm;

    OpenCVFisheyeProjection(
        at::Tensor principal_point,
        at::Tensor focal_length,
        at::Tensor forward_poly,
        at::Tensor approx_backward_factor,
        std::array<int64_t, 2> resolution,
        int64_t newton_iterations,
        double max_angle,
        double min_2d_norm);

    OpenCVFisheyeProjection_KernelParameters to_kernel_params() const;

    // Rescale the image-domain intrinsics for a new resolution. pp and focal
    // scale per-axis by (scale_u, scale_v); approx_backward_factor is recomputed from the new
    // focal (it is a derived initial-guess factor, not an independent
    // intrinsic). The three config scalars are preserved unscaled.
    c10::intrusive_ptr<OpenCVFisheyeProjection> transform(
        std::tuple<double, double> scale,
        std::tuple<double, double> offset,
        std::tuple<int64_t, int64_t> new_resolution) const;
};

// Validates projection shape and non-null; throws TORCH_CHECK on failure.
void check_opencv_fisheye_projection(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection);

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

// ===========================================================================
// FTheta forward passes (no_external and bivariate_windshield variants)
// ===========================================================================
// Same shapes as the OpenCVPinhole counterparts; scratch widths differ
// (see camera_params.h Kernel N comments for per-kernel scratch sizes).

// forward: returns (image_points (N,2), valid_flags (N,) bool, scratch (N,8))
std::tuple<at::Tensor, at::Tensor, at::Tensor>
camera_rays_to_image_points_ftheta_no_external(
    const c10::intrusive_ptr<FThetaProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& camera_rays);

// forward: returns (camera_rays (N,3), scratch (N,8))
std::tuple<at::Tensor, at::Tensor>
image_points_to_camera_rays_ftheta_no_external(
    const c10::intrusive_ptr<FThetaProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points);

// Returns (image_points (N,2), valid_flags (N,) bool, scratch (N,8))
std::tuple<at::Tensor, at::Tensor, at::Tensor>
camera_rays_to_image_points_ftheta_bivariate_windshield(
    const c10::intrusive_ptr<FThetaProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& camera_rays);

// Returns (camera_rays (N,3), scratch (N,12))
std::tuple<at::Tensor, at::Tensor>
image_points_to_camera_rays_ftheta_bivariate_windshield(
    const c10::intrusive_ptr<FThetaProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_mean_pose_ftheta_no_external(
    const c10::intrusive_ptr<FThetaProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& world_points,
    const at::Tensor& start_translation,
    const at::Tensor& start_rotation,
    const at::Tensor& end_translation,
    const at::Tensor& end_rotation,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_mean_pose_ftheta_bivariate_windshield(
    const c10::intrusive_ptr<FThetaProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& world_points,
    const at::Tensor& start_translation,
    const at::Tensor& start_rotation,
    const at::Tensor& end_translation,
    const at::Tensor& end_rotation,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_static_pose_ftheta_no_external(
    const c10::intrusive_ptr<FThetaProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& translations,
    const at::Tensor& rotations,
    int64_t timestamp_us);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_static_pose_ftheta_bivariate_windshield(
    const c10::intrusive_ptr<FThetaProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& translations,
    const at::Tensor& rotations,
    int64_t timestamp_us);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_shutter_pose_ftheta_no_external(
    const c10::intrusive_ptr<FThetaProjection>& projection,
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_shutter_pose_ftheta_bivariate_windshield(
    const c10::intrusive_ptr<FThetaProjection>& projection,
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
image_points_to_world_rays_shutter_pose_ftheta_no_external(
    const c10::intrusive_ptr<FThetaProjection>& projection,
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield(
    const c10::intrusive_ptr<FThetaProjection>& projection,
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
// FTheta backward passes (no_external and bivariate_windshield variants)
// ===========================================================================
// Each backward returns its gradient tuple in the order
// (primary input grad, [pose grads,] intrinsic grads), where intrinsic grads
// are grad_principal_point (2,), grad_fw_poly (6,),
// grad_bw_poly (6,), grad_A (4,), grad_Ainv (4,);
// bivariate_windshield variants append grad_distortion_coeffs (42,).
// Grad tensors for flags=false slots are empty (0,) scratch.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
camera_rays_to_image_points_ftheta_no_external_backward(
    const c10::intrusive_ptr<FThetaProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& camera_rays,
    const at::Tensor& grad_image_points,
    const at::Tensor& scratch,
    bool need_camera_ray_grad,
    bool need_principal_point_grad,
    bool need_fw_poly_grad,
    bool need_bw_poly_grad,
    bool need_A_grad,
    bool need_Ainv_grad);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_camera_rays_ftheta_no_external_backward(
    const c10::intrusive_ptr<FThetaProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& grad_camera_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_principal_point_grad,
    bool need_fw_poly_grad,
    bool need_bw_poly_grad,
    bool need_A_grad,
    bool need_Ainv_grad);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
camera_rays_to_image_points_ftheta_bivariate_windshield_backward(
    const c10::intrusive_ptr<FThetaProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& camera_rays,
    const at::Tensor& grad_image_points,
    const at::Tensor& scratch,
    bool need_camera_ray_grad,
    bool need_principal_point_grad,
    bool need_fw_poly_grad,
    bool need_bw_poly_grad,
    bool need_A_grad,
    bool need_Ainv_grad,
    bool need_distortion_coeffs_grad);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_camera_rays_ftheta_bivariate_windshield_backward(
    const c10::intrusive_ptr<FThetaProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& grad_camera_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_principal_point_grad,
    bool need_fw_poly_grad,
    bool need_bw_poly_grad,
    bool need_A_grad,
    bool need_Ainv_grad,
    bool need_distortion_coeffs_grad);

std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_mean_pose_ftheta_no_external_backward(
    const c10::intrusive_ptr<FThetaProjection>& projection,
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
    bool need_principal_point_grad,
    bool need_fw_poly_grad,
    bool need_bw_poly_grad,
    bool need_A_grad,
    bool need_Ainv_grad);

std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_mean_pose_ftheta_bivariate_windshield_backward(
    const c10::intrusive_ptr<FThetaProjection>& projection,
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
    bool need_principal_point_grad,
    bool need_fw_poly_grad,
    bool need_bw_poly_grad,
    bool need_A_grad,
    bool need_Ainv_grad,
    bool need_distortion_coeffs_grad);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_static_pose_ftheta_no_external_backward(
    const c10::intrusive_ptr<FThetaProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& translation,
    const at::Tensor& rotation,
    const at::Tensor& grad_world_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_translation_grad,
    bool need_rotation_grad,
    bool need_principal_point_grad,
    bool need_fw_poly_grad,
    bool need_bw_poly_grad,
    bool need_A_grad,
    bool need_Ainv_grad);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_backward(
    const c10::intrusive_ptr<FThetaProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& translation,
    const at::Tensor& rotation,
    const at::Tensor& grad_world_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_translation_grad,
    bool need_rotation_grad,
    bool need_principal_point_grad,
    bool need_fw_poly_grad,
    bool need_bw_poly_grad,
    bool need_A_grad,
    bool need_Ainv_grad,
    bool need_distortion_coeffs_grad);

std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_shutter_pose_ftheta_no_external_backward(
    const c10::intrusive_ptr<FThetaProjection>& projection,
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
    bool need_principal_point_grad,
    bool need_fw_poly_grad,
    bool need_bw_poly_grad,
    bool need_A_grad,
    bool need_Ainv_grad);

std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_shutter_pose_ftheta_bivariate_windshield_backward(
    const c10::intrusive_ptr<FThetaProjection>& projection,
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
    bool need_principal_point_grad,
    bool need_fw_poly_grad,
    bool need_bw_poly_grad,
    bool need_A_grad,
    bool need_Ainv_grad,
    bool need_distortion_coeffs_grad);

std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_shutter_pose_ftheta_no_external_backward(
    const c10::intrusive_ptr<FThetaProjection>& projection,
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
    bool need_principal_point_grad,
    bool need_fw_poly_grad,
    bool need_bw_poly_grad,
    bool need_A_grad,
    bool need_Ainv_grad);

std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_backward(
    const c10::intrusive_ptr<FThetaProjection>& projection,
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
    bool need_principal_point_grad,
    bool need_fw_poly_grad,
    bool need_bw_poly_grad,
    bool need_A_grad,
    bool need_Ainv_grad,
    bool need_distortion_coeffs_grad);

// ===========================================================================
// OpenCV-fisheye no_external ops (D1/D2/D3/D5). Intrinsic grad outputs are
// grad_principal_point (2,), grad_focal_length (2,) and grad_forward_poly (4,);
// approx_backward_factor receives no gradient.
// ===========================================================================

// D1 forward: returns (image_points (N,2), valid_flags (N,) bool, scratch (N,8)).
std::tuple<at::Tensor, at::Tensor, at::Tensor>
camera_rays_to_image_points_opencv_fisheye_no_external(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& camera_rays);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
camera_rays_to_image_points_opencv_fisheye_no_external_backward(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& camera_rays,
    const at::Tensor& grad_image_points,
    const at::Tensor& scratch,
    bool need_camera_ray_grad,
    bool need_principal_point_grad,
    bool need_focal_length_grad,
    bool need_forward_poly_grad);

// D2 forward: returns (camera_rays (N,3), scratch (N,8)).
std::tuple<at::Tensor, at::Tensor>
image_points_to_camera_rays_opencv_fisheye_no_external(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_camera_rays_opencv_fisheye_no_external_backward(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& grad_camera_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_principal_point_grad,
    bool need_focal_length_grad,
    bool need_forward_poly_grad);

// D3 forward: returns (image_points (N,2), valid_flags (N,), timestamps (N,)
// int64, pose_t (N,3), pose_r (N,4), scratch (N,14)).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_mean_pose_opencv_fisheye_no_external(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& world_points,
    const at::Tensor& start_translation,
    const at::Tensor& start_rotation,
    const at::Tensor& end_translation,
    const at::Tensor& end_rotation,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us);

std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor>
project_world_points_mean_pose_opencv_fisheye_no_external_backward(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
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
    bool need_principal_point_grad,
    bool need_focal_length_grad,
    bool need_forward_poly_grad);

// D5 forward: returns (world_rays (N,6), timestamps (N,) int64, pose_t (N,3),
// pose_r (N,4), scratch (N,8)).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_static_pose_opencv_fisheye_no_external(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& translations,
    const at::Tensor& rotations,
    int64_t timestamp_us);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_static_pose_opencv_fisheye_no_external_backward(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& translations,
    const at::Tensor& rotations,
    const at::Tensor& grad_world_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_translation_grad,
    bool need_rotation_grad,
    bool need_principal_point_grad,
    bool need_focal_length_grad,
    bool need_forward_poly_grad);

// ===========================================================================
// OpenCV-fisheye bivariate-windshield ops (D1/D2/D3/D5). Same intrinsic grad
// outputs as the no_external ops, plus a grad_distortion_coeffs output for the
// active bivariate-coeff slice. Scratch strides: D1=8, D2=12, D3=14, D5=12.
// ===========================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor>
camera_rays_to_image_points_opencv_fisheye_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& camera_rays);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
camera_rays_to_image_points_opencv_fisheye_bivariate_windshield_backward(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& camera_rays,
    const at::Tensor& grad_image_points,
    const at::Tensor& scratch,
    bool need_camera_ray_grad,
    bool need_principal_point_grad,
    bool need_focal_length_grad,
    bool need_forward_poly_grad,
    bool need_distortion_coeffs_grad);

std::tuple<at::Tensor, at::Tensor>
image_points_to_camera_rays_opencv_fisheye_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_camera_rays_opencv_fisheye_bivariate_windshield_backward(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& grad_camera_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_principal_point_grad,
    bool need_focal_length_grad,
    bool need_forward_poly_grad,
    bool need_distortion_coeffs_grad);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_mean_pose_opencv_fisheye_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& world_points,
    const at::Tensor& start_translation,
    const at::Tensor& start_rotation,
    const at::Tensor& end_translation,
    const at::Tensor& end_rotation,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us);

std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_mean_pose_opencv_fisheye_bivariate_windshield_backward(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
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
    bool need_principal_point_grad,
    bool need_focal_length_grad,
    bool need_forward_poly_grad,
    bool need_distortion_coeffs_grad);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& translations,
    const at::Tensor& rotations,
    int64_t timestamp_us);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield_backward(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& translations,
    const at::Tensor& rotations,
    const at::Tensor& grad_world_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_translation_grad,
    bool need_rotation_grad,
    bool need_principal_point_grad,
    bool need_focal_length_grad,
    bool need_forward_poly_grad,
    bool need_distortion_coeffs_grad);

// ===========================================================================
// OpenCV-fisheye shutter-pose ops (D4/D6). Same intrinsic grad outputs as the
// other fisheye ops, plus start/end pose grads. Scratch strides: D4=16 (both
// distortions; alpha at off+14), D6=12 (no_external) / 16 (bivariate). D4
// gates its backward on the converged NaN-sentinel alpha; D6 does not.
// ===========================================================================

// D4 forward: returns (image_points (N,2), valid_flags (N,) bool,
// timestamps (N,) int64, pose_t (N,3), pose_r (N,4), scratch (N,16)).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_shutter_pose_opencv_fisheye_no_external(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
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

// Unlike the FTheta/pinhole shutter-pose backward, this takes neither
// world_points/shutter_type/max_iterations nor valid_flags: the converged pose
// Jacobian and the per-point convergence mask (NaN-sentinel alpha, see above)
// are read back from scratch, so the backward needs only the endpoint rotations,
// the upstream grad, and scratch.
std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor>
project_world_points_shutter_pose_opencv_fisheye_no_external_backward(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& start_rotation,
    const at::Tensor& end_rotation,
    const at::Tensor& grad_image_points,
    const at::Tensor& scratch,
    bool need_world_point_grad,
    bool need_start_translation_grad,
    bool need_end_translation_grad,
    bool need_start_rotation_grad,
    bool need_end_rotation_grad,
    bool need_principal_point_grad,
    bool need_focal_length_grad,
    bool need_forward_poly_grad);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
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

std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor>
project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield_backward(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& start_rotation,
    const at::Tensor& end_rotation,
    const at::Tensor& grad_image_points,
    const at::Tensor& scratch,
    bool need_world_point_grad,
    bool need_start_translation_grad,
    bool need_end_translation_grad,
    bool need_start_rotation_grad,
    bool need_end_rotation_grad,
    bool need_principal_point_grad,
    bool need_focal_length_grad,
    bool need_forward_poly_grad,
    bool need_distortion_coeffs_grad);

// D6 forward: returns (world_rays (N,6), timestamps (N,) int64, pose_t (N,3),
// pose_r (N,4), scratch (N,12) no_external / (N,16) bivariate).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
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

std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external_backward(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& start_rotation,
    const at::Tensor& end_rotation,
    const at::Tensor& grad_world_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_start_translation_grad,
    bool need_end_translation_grad,
    bool need_start_rotation_grad,
    bool need_end_rotation_grad,
    bool need_principal_point_grad,
    bool need_focal_length_grad,
    bool need_forward_poly_grad);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
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

std::tuple<
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield_backward(
    const c10::intrusive_ptr<OpenCVFisheyeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& start_rotation,
    const at::Tensor& end_rotation,
    const at::Tensor& grad_world_rays,
    const at::Tensor& scratch,
    bool need_image_point_grad,
    bool need_start_translation_grad,
    bool need_end_translation_grad,
    bool need_start_rotation_grad,
    bool need_end_rotation_grad,
    bool need_principal_point_grad,
    bool need_focal_length_grad,
    bool need_forward_poly_grad,
    bool need_distortion_coeffs_grad);

} // namespace gsplat_sensors
