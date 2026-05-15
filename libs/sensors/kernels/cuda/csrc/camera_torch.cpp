/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// camera_torch.cpp — C++ wrapper layer between TORCH_LIBRARY (ext.cpp) and
// the CUDA _launch helpers (camera_params.h / camera_kernel.cu).
//
// Responsibilities:
//   1. Input validation — tensor dtype, contiguity, device, shape via TORCH_CHECK.
//   2. Output allocation — image_points, valid_flags, timestamps, pose tensors,
//      and the per-kernel scratch buffer that the autograd graph carries forward.
//   3. Dispatch — calls the appropriate _forward_launch or _backward_launch
//      function from camera_params.h.
//
// File layout:
//   [anonymous namespace]  — shared validation helpers and struct helpers
//   OpenCVPinholeProjection — constructor + to_kernel_params
//   check_projection / check_bivariate_windshield helpers
//   generate_image_points
//   camera_rays_to_image_points family (forward + backward)
//   image_points_to_camera_rays family (forward + backward)
//   project_world_points_mean_pose family (forward + backward)
//   project_world_points_shutter_pose family (forward + backward)
//   image_points_to_world_rays_static_pose family (forward + backward)
//   image_points_to_world_rays_shutter_pose family (forward + backward)
//   [each family appears twice: once for no_external, once for bivariate_windshield]

#include "camera_torch.h"
#include "shutter_type.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <utility>

namespace gsplat_sensors {
namespace {

// ===========================================================================
// Shared validation and allocation helpers (internal linkage)
// ===========================================================================

void check_component_shape(const at::Tensor& tensor, at::IntArrayRef shape, const char* name) {
    TORCH_CHECK(tensor.sizes() == shape, name, " must be ", shape);
}

void check_cuda_float_contiguous(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_matrix(const at::Tensor& tensor, int64_t cols, const char* name) {
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(1) == cols, name, " must be (N, ", cols, ")");
}

void check_component_vector(const at::Tensor& tensor, int64_t expected, const char* name) {
    if (tensor.dim() == 2) {
        TORCH_CHECK(
            tensor.size(0) == 1 && tensor.size(1) == expected,
            name,
            " must be (",
            expected,
            ",) or (1, ",
            expected,
            ")");
    } else {
        TORCH_CHECK(
            tensor.dim() == 1 && tensor.size(0) == expected,
            name,
            " must be (",
            expected,
            ",) or (1, ",
            expected,
            ")");
    }
}

void check_projection_for_device(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const at::Device& device) {
    check_projection(projection);
    check_cuda_float_contiguous(projection->focal_length, "focal_length");
    check_cuda_float_contiguous(projection->principal_point, "principal_point");
    check_cuda_float_contiguous(projection->radial_coeffs, "radial_coeffs");
    check_cuda_float_contiguous(projection->tangential_coeffs, "tangential_coeffs");
    check_cuda_float_contiguous(projection->thin_prism_coeffs, "thin_prism_coeffs");
    TORCH_CHECK(projection->focal_length.device() == device, "focal_length device must match inputs");
    TORCH_CHECK(projection->principal_point.device() == device, "principal_point device must match inputs");
    TORCH_CHECK(projection->radial_coeffs.device() == device, "radial_coeffs device must match inputs");
    TORCH_CHECK(projection->tangential_coeffs.device() == device, "tangential_coeffs device must match inputs");
    TORCH_CHECK(projection->thin_prism_coeffs.device() == device, "thin_prism_coeffs device must match inputs");
}

cudaStream_t current_stream(const at::Tensor& tensor) {
    return at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

at::Tensor empty_scratch(const at::TensorOptions& options) {
    return at::empty({0}, options);
}

at::Tensor zeros_like_shape(at::IntArrayRef shape, const at::Tensor& reference) {
    return at::zeros(shape, reference.options());
}

at::Tensor scratch_shape(int64_t count, int64_t stride, const at::Tensor& reference) {
    return at::empty({count, stride}, reference.options());
}

bool needs_projection_grad(const c10::intrusive_ptr<OpenCVPinholeProjection>& projection) {
    return projection->focal_length.requires_grad() ||
        projection->principal_point.requires_grad() ||
        projection->radial_coeffs.requires_grad() ||
        projection->tangential_coeffs.requires_grad() ||
        projection->thin_prism_coeffs.requires_grad();
}

struct IntrinsicGradOutputs {
    at::Tensor focal_length;
    at::Tensor principal_point;
    at::Tensor radial_coeffs;
    at::Tensor tangential_coeffs;
    at::Tensor thin_prism_coeffs;
};

at::Tensor maybe_intrinsic_grad(bool needed, at::IntArrayRef shape, const at::Tensor& reference) {
    return needed ? zeros_like_shape(shape, reference) : empty_scratch(reference.options());
}

float* maybe_data_ptr(at::Tensor& tensor, bool needed) {
    return needed ? tensor.data_ptr<float>() : nullptr;
}

IntrinsicGradOutputs make_intrinsic_grad_outputs(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    bool need_focal_length_grad,
    bool need_principal_point_grad,
    bool need_radial_coeffs_grad,
    bool need_tangential_coeffs_grad,
    bool need_thin_prism_coeffs_grad) {
    return {
        maybe_intrinsic_grad(need_focal_length_grad, {2}, projection->focal_length),
        maybe_intrinsic_grad(need_principal_point_grad, {2}, projection->principal_point),
        maybe_intrinsic_grad(need_radial_coeffs_grad, {6}, projection->radial_coeffs),
        maybe_intrinsic_grad(need_tangential_coeffs_grad, {2}, projection->tangential_coeffs),
        maybe_intrinsic_grad(need_thin_prism_coeffs_grad, {4}, projection->thin_prism_coeffs),
    };
}

OpenCVPinholeProjection_KernelParameters kernel_params_with_resolution(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    int64_t width,
    int64_t height) {
    auto params = projection->to_kernel_params();
    params.width = width;
    params.height = height;
    return params;
}

void check_optional_distortion(const c10::intrusive_ptr<NoExternalDistortion>& external_distortion) {
    TORCH_CHECK(external_distortion != nullptr, "external_distortion must be NoExternalDistortion");
}

void check_bivariate_windshield_for_device(
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Device& device) {
    check_bivariate_windshield_distortion(external_distortion);
    check_cuda_float_contiguous(external_distortion->distortion_coeffs, "distortion_coeffs");
    TORCH_CHECK(external_distortion->distortion_coeffs.device() == device, "distortion_coeffs device must match inputs");
}

bool needs_external_distortion_grad(
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion) {
    return external_distortion->distortion_coeffs.requires_grad();
}

at::Tensor make_distortion_grad_output(
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    bool needed) {
    return needed ? at::zeros_like(external_distortion->distortion_coeffs) :
        empty_scratch(external_distortion->distortion_coeffs.options());
}

void check_max_iterations(int64_t max_iterations) {
    TORCH_CHECK(max_iterations >= 1, "max_iterations must be at least 1");
    TORCH_CHECK(
        max_iterations <= kMaxRollingShutterIterations,
        "max_iterations must be <= ",
        kMaxRollingShutterIterations);
}

// Rejects any integer that does not map to a known ShutterType enumerator.
// Must be kept in sync with shutter_type.h.
void check_shutter_type(int64_t shutter_type) {
    switch (static_cast<ShutterType>(shutter_type)) {
        case ShutterType::ROLLING_TOP_TO_BOTTOM:
        case ShutterType::ROLLING_LEFT_TO_RIGHT:
        case ShutterType::ROLLING_BOTTOM_TO_TOP:
        case ShutterType::ROLLING_RIGHT_TO_LEFT:
        case ShutterType::GLOBAL:
            return;
    }
    TORCH_CHECK(
        false,
        "shutter_type=",
        shutter_type,
        " is not a valid gsplat_sensors::ShutterType (see shutter_type.h)");
}

// Ensures width and height are both positive; required by rolling-shutter
// kernels that use pixel position to derive per-point exposure timestamps.
void check_shutter_resolution(int64_t width, int64_t height) {
    TORCH_CHECK(
        width > 0 && height > 0,
        "resolution width and height must be positive for rolling-shutter kernels");
}

} // namespace

// ===========================================================================
// OpenCVPinholeProjection — constructor and kernel-params accessor
// ===========================================================================

OpenCVPinholeProjection::OpenCVPinholeProjection(
    at::Tensor focal_length_,
    at::Tensor principal_point_,
    at::Tensor radial_coeffs_,
    at::Tensor tangential_coeffs_,
    at::Tensor thin_prism_coeffs_,
    std::array<int64_t, 2> resolution_)
    : focal_length(std::move(focal_length_)),
      principal_point(std::move(principal_point_)),
      radial_coeffs(std::move(radial_coeffs_)),
      tangential_coeffs(std::move(tangential_coeffs_)),
      thin_prism_coeffs(std::move(thin_prism_coeffs_)),
      resolution(resolution_) {}

OpenCVPinholeProjection_KernelParameters OpenCVPinholeProjection::to_kernel_params() const {
    OpenCVPinholeProjection_KernelParameters params{};
    params.focal_length = focal_length.const_data_ptr<float>();
    params.principal_point = principal_point.const_data_ptr<float>();
    params.radial_coeffs = radial_coeffs.const_data_ptr<float>();
    params.tangential_coeffs = tangential_coeffs.const_data_ptr<float>();
    params.thin_prism_coeffs = thin_prism_coeffs.const_data_ptr<float>();
    params.width = resolution[0];
    params.height = resolution[1];
    return params;
}

void check_projection(const c10::intrusive_ptr<OpenCVPinholeProjection>& projection) {
    TORCH_CHECK(projection != nullptr, "projection must be OpenCVPinholeProjection");
    check_component_shape(projection->focal_length, {2}, "focal_length");
    check_component_shape(projection->principal_point, {2}, "principal_point");
    check_component_shape(projection->radial_coeffs, {6}, "radial_coeffs");
    check_component_shape(projection->tangential_coeffs, {2}, "tangential_coeffs");
    check_component_shape(projection->thin_prism_coeffs, {4}, "thin_prism_coeffs");
    TORCH_CHECK(projection->resolution[0] > 0 && projection->resolution[1] > 0, "resolution must be positive");
}

// ===========================================================================
// generate_image_points
// ===========================================================================

at::Tensor generate_image_points(int64_t width, int64_t height, c10::Device device) {
    TORCH_CHECK(width >= 0 && height >= 0, "resolution dimensions must be non-negative");
    TORCH_CHECK(device.is_cuda(), "generate_image_points requires a CUDA device");
    auto guard = c10::cuda::CUDAGuard(device);
    auto options = at::TensorOptions().device(device).dtype(at::kFloat);
    auto image_points = at::empty({height, width, 2}, options);
    generate_image_points_launch(width, height, image_points.data_ptr<float>(), at::cuda::getCurrentCUDAStream(device.index()).stream());
    return image_points;
}

// ===========================================================================
// camera_rays_to_image_points — no_external + bivariate_windshield
//   forward + backward
// ===========================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor> camera_rays_to_image_points_opencv_pinhole_no_external(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& camera_rays) {
    check_optional_distortion(external_distortion);
    const auto& rays = camera_rays;
    check_matrix(rays, 3, "camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_projection_for_device(projection, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto image_points = at::empty({rays.size(0), 2}, rays.options());
    auto valid_flags = at::empty({rays.size(0)}, rays.options().dtype(at::kBool));
    const bool save_scratch = rays.requires_grad() || needs_projection_grad(projection);
    auto scratch = save_scratch ?
        scratch_shape(rays.size(0), 6, rays) :
        empty_scratch(rays.options());
    camera_rays_to_image_points_forward_launch(
        rays.size(0),
        projection->to_kernel_params(),
        rays.data_ptr<float>(),
        image_points.data_ptr<float>(),
        valid_flags.data_ptr<bool>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(rays));
    return {image_points, valid_flags, scratch};
}

// ===========================================================================
// image_points_to_camera_rays — no_external + bivariate_windshield
//   forward + backward
// ===========================================================================

std::tuple<at::Tensor, at::Tensor> image_points_to_camera_rays_opencv_pinhole_no_external(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points) {
    check_optional_distortion(external_distortion);
    const auto& pts = image_points;
    check_matrix(pts, 2, "image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto camera_rays = at::empty({pts.size(0), 3}, pts.options());
    const bool save_scratch = pts.requires_grad() || needs_projection_grad(projection);
    auto scratch = save_scratch ?
        scratch_shape(pts.size(0), 5, pts) :
        empty_scratch(pts.options());
    image_points_to_camera_rays_forward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        camera_rays.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts));
    return {camera_rays, scratch};
}

// ===========================================================================
// project_world_points_mean_pose — no_external + bivariate_windshield
//   forward + backward
// ===========================================================================

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
    int64_t end_timestamp_us) {
    check_optional_distortion(external_distortion);
    const auto& pts = world_points;
    check_matrix(pts, 3, "world_points");
    check_component_vector(start_translation, 3, "start_translation");
    check_component_vector(end_translation, 3, "end_translation");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_cuda_float_contiguous(pts, "world_points");
    check_cuda_float_contiguous(start_translation, "start_translation");
    check_cuda_float_contiguous(end_translation, "end_translation");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    TORCH_CHECK(start_translation.device() == pts.device(), "start_translation device must match world_points");
    TORCH_CHECK(end_translation.device() == pts.device(), "end_translation device must match world_points");
    TORCH_CHECK(start_rotation.device() == pts.device(), "start_rotation device must match world_points");
    TORCH_CHECK(end_rotation.device() == pts.device(), "end_rotation device must match world_points");
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto image_points = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad() ||
        start_translation.requires_grad() ||
        end_translation.requires_grad() ||
        start_rotation.requires_grad() ||
        end_rotation.requires_grad() ||
        needs_projection_grad(projection);
    auto scratch = save_scratch ?
        scratch_shape(pts.size(0), 9, pts) :
        empty_scratch(pts.options());
    const auto mean_timestamp_us = static_cast<int64_t>(
        static_cast<double>(start_timestamp_us) +
        0.5 * static_cast<double>(end_timestamp_us - start_timestamp_us));
    project_world_points_mean_pose_forward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        start_translation.data_ptr<float>(),
        start_rotation.data_ptr<float>(),
        end_translation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        mean_timestamp_us,
        image_points.data_ptr<float>(),
        valid_flags.data_ptr<bool>(),
        timestamps.data_ptr<int64_t>(),
        pose_t.data_ptr<float>(),
        pose_r.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts));
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

// ===========================================================================
// project_world_points_shutter_pose — no_external + bivariate_windshield
//   forward + backward
// ===========================================================================

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
    double initial_relative_time) {
    check_optional_distortion(external_distortion);
    const auto& pts = world_points;
    check_matrix(pts, 3, "world_points");
    check_component_vector(start_translation, 3, "start_translation");
    check_component_vector(end_translation, 3, "end_translation");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_shutter_resolution(width, height);
    check_shutter_type(shutter_type);
    check_max_iterations(max_iterations);
    check_cuda_float_contiguous(pts, "world_points");
    check_cuda_float_contiguous(start_translation, "start_translation");
    check_cuda_float_contiguous(end_translation, "end_translation");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    TORCH_CHECK(start_translation.device() == pts.device(), "start_translation device must match world_points");
    TORCH_CHECK(end_translation.device() == pts.device(), "end_translation device must match world_points");
    TORCH_CHECK(start_rotation.device() == pts.device(), "start_rotation device must match world_points");
    TORCH_CHECK(end_rotation.device() == pts.device(), "end_rotation device must match world_points");
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto image_points = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad() ||
        start_translation.requires_grad() ||
        end_translation.requires_grad() ||
        start_rotation.requires_grad() ||
        end_rotation.requires_grad() ||
        needs_projection_grad(projection);
    auto scratch = save_scratch ?
        scratch_shape(pts.size(0), 10, pts) :
        empty_scratch(pts.options());
    project_world_points_shutter_pose_forward_launch(
        pts.size(0),
        kernel_params_with_resolution(projection, width, height),
        pts.data_ptr<float>(),
        start_translation.data_ptr<float>(),
        start_rotation.data_ptr<float>(),
        end_translation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        shutter_type,
        start_timestamp_us,
        end_timestamp_us,
        max_iterations,
        static_cast<float>(stop_mean_error_px),
        static_cast<float>(stop_delta_mean_error_px),
        static_cast<float>(initial_relative_time),
        image_points.data_ptr<float>(),
        valid_flags.data_ptr<bool>(),
        timestamps.data_ptr<int64_t>(),
        pose_t.data_ptr<float>(),
        pose_r.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts));
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

// ===========================================================================
// image_points_to_world_rays_static_pose — no_external + bivariate_windshield
//   forward + backward
// ===========================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_static_pose_opencv_pinhole_no_external(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<NoExternalDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& translations,
    const at::Tensor& rotations,
    int64_t timestamp_us) {
    check_optional_distortion(external_distortion);
    const auto& pts = image_points;
    const auto& trans = translations;
    const auto& rots = rotations;
    check_matrix(pts, 2, "image_points");
    check_matrix(trans, 3, "translations");
    check_matrix(rots, 4, "rotations");
    TORCH_CHECK(trans.size(0) >= 1 && rots.size(0) >= 1, "static pose requires one control pose");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(trans, "translations");
    check_cuda_float_contiguous(rots, "rotations");
    TORCH_CHECK(trans.device() == pts.device() && rots.device() == pts.device(), "pose tensors device must match image_points");
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto world_rays = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad() ||
        trans.requires_grad() ||
        rots.requires_grad() ||
        needs_projection_grad(projection);
    auto scratch = save_scratch ?
        scratch_shape(pts.size(0), 5, pts) :
        empty_scratch(pts.options());
    image_points_to_world_rays_static_pose_forward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        trans.data_ptr<float>(),
        rots.data_ptr<float>(),
        timestamp_us,
        world_rays.data_ptr<float>(),
        timestamps.data_ptr<int64_t>(),
        pose_t.data_ptr<float>(),
        pose_r.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts));
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

// ===========================================================================
// image_points_to_world_rays_shutter_pose — no_external + bivariate_windshield
//   forward + backward
// ===========================================================================

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
    int64_t end_timestamp_us) {
    check_optional_distortion(external_distortion);
    const auto& pts = image_points;
    check_matrix(pts, 2, "image_points");
    check_component_vector(start_translation, 3, "start_translation");
    check_component_vector(end_translation, 3, "end_translation");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_shutter_resolution(width, height);
    check_shutter_type(shutter_type);
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(start_translation, "start_translation");
    check_cuda_float_contiguous(end_translation, "end_translation");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    TORCH_CHECK(start_translation.device() == pts.device(), "start_translation device must match image_points");
    TORCH_CHECK(end_translation.device() == pts.device(), "end_translation device must match image_points");
    TORCH_CHECK(start_rotation.device() == pts.device(), "start_rotation device must match image_points");
    TORCH_CHECK(end_rotation.device() == pts.device(), "end_rotation device must match image_points");
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto world_rays = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad() ||
        start_translation.requires_grad() ||
        end_translation.requires_grad() ||
        start_rotation.requires_grad() ||
        end_rotation.requires_grad() ||
        needs_projection_grad(projection);
    auto scratch = save_scratch ?
        scratch_shape(pts.size(0), 9, pts) :
        empty_scratch(pts.options());
    image_points_to_world_rays_shutter_pose_forward_launch(
        pts.size(0),
        kernel_params_with_resolution(projection, width, height),
        pts.data_ptr<float>(),
        start_translation.data_ptr<float>(),
        start_rotation.data_ptr<float>(),
        end_translation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        shutter_type,
        start_timestamp_us,
        end_timestamp_us,
        world_rays.data_ptr<float>(),
        timestamps.data_ptr<int64_t>(),
        pose_t.data_ptr<float>(),
        pose_r.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts));
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

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
    bool need_thin_prism_coeffs_grad) {
    check_optional_distortion(external_distortion);
    const auto& rays = camera_rays;
    const auto& grad = grad_image_points;
    const auto& scratch_contig = scratch;
    check_matrix(rays, 3, "camera_rays");
    check_matrix(grad, 2, "grad_image_points");
    TORCH_CHECK(grad.size(0) == rays.size(0), "grad_image_points batch must match camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_cuda_float_contiguous(grad, "grad_image_points");
    check_cuda_float_contiguous(scratch_contig, "scratch");
    TORCH_CHECK(scratch_contig.numel() >= rays.size(0) * 6, "scratch too small for camera_rays_to_image_points backward");
    check_projection_for_device(projection, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto grad_rays = need_camera_ray_grad ? at::empty_like(rays) : empty_scratch(rays.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad);
    camera_rays_to_image_points_backward_launch(
        rays.size(0),
        projection->to_kernel_params(),
        rays.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_camera_ray_grad ? grad_rays.data_ptr<float>() : nullptr,
        maybe_data_ptr(intrinsic_grads.focal_length, need_focal_length_grad),
        maybe_data_ptr(intrinsic_grads.principal_point, need_principal_point_grad),
        maybe_data_ptr(intrinsic_grads.radial_coeffs, need_radial_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.tangential_coeffs, need_tangential_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.thin_prism_coeffs, need_thin_prism_coeffs_grad),
        scratch_contig.data_ptr<float>(),
        current_stream(rays));
    return {grad_rays, intrinsic_grads.focal_length, intrinsic_grads.principal_point, intrinsic_grads.radial_coeffs, intrinsic_grads.tangential_coeffs, intrinsic_grads.thin_prism_coeffs};
}

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
    bool need_thin_prism_coeffs_grad) {
    check_optional_distortion(external_distortion);
    const auto& pts = image_points;
    const auto& grad = grad_camera_rays;
    const auto& scratch_contig = scratch;
    check_matrix(pts, 2, "image_points");
    check_matrix(grad, 3, "grad_camera_rays");
    TORCH_CHECK(grad.size(0) == pts.size(0), "grad_camera_rays batch must match image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(grad, "grad_camera_rays");
    check_cuda_float_contiguous(scratch_contig, "scratch");
    TORCH_CHECK(scratch_contig.numel() >= pts.size(0) * 5, "scratch too small for image_points_to_camera_rays backward");
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad);
    image_points_to_camera_rays_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        maybe_data_ptr(intrinsic_grads.focal_length, need_focal_length_grad),
        maybe_data_ptr(intrinsic_grads.principal_point, need_principal_point_grad),
        maybe_data_ptr(intrinsic_grads.radial_coeffs, need_radial_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.tangential_coeffs, need_tangential_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.thin_prism_coeffs, need_thin_prism_coeffs_grad),
        scratch_contig.data_ptr<float>(),
        current_stream(pts));
    return {grad_pts, intrinsic_grads.focal_length, intrinsic_grads.principal_point, intrinsic_grads.radial_coeffs, intrinsic_grads.tangential_coeffs, intrinsic_grads.thin_prism_coeffs};
}

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
    bool need_thin_prism_coeffs_grad) {
    check_optional_distortion(external_distortion);
    const auto& pts = world_points;
    const auto& grad = grad_image_points;
    const auto& scratch_contig = scratch;
    check_matrix(pts, 3, "world_points");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_matrix(grad, 2, "grad_image_points");
    TORCH_CHECK(grad.size(0) == pts.size(0), "grad_image_points batch must match world_points");
    check_cuda_float_contiguous(pts, "world_points");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    check_cuda_float_contiguous(grad, "grad_image_points");
    check_cuda_float_contiguous(scratch_contig, "scratch");
    TORCH_CHECK(scratch_contig.numel() >= pts.size(0) * 9, "scratch too small for project_world_points_mean_pose backward");
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_world_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad
        ? at::zeros({3}, pts.options())
        : empty_scratch(pts.options());
    auto grad_end_t = need_end_translation_grad
        ? at::zeros({3}, pts.options())
        : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad
        ? at::zeros({4}, pts.options())
        : empty_scratch(pts.options());
    auto grad_end_r = need_end_rotation_grad
        ? at::zeros({4}, pts.options())
        : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad);
    project_world_points_mean_pose_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        start_rotation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_world_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_start_translation_grad ? grad_start_t.data_ptr<float>() : nullptr,
        need_end_translation_grad ? grad_end_t.data_ptr<float>() : nullptr,
        need_start_rotation_grad ? grad_start_r.data_ptr<float>() : nullptr,
        need_end_rotation_grad ? grad_end_r.data_ptr<float>() : nullptr,
        maybe_data_ptr(intrinsic_grads.focal_length, need_focal_length_grad),
        maybe_data_ptr(intrinsic_grads.principal_point, need_principal_point_grad),
        maybe_data_ptr(intrinsic_grads.radial_coeffs, need_radial_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.tangential_coeffs, need_tangential_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.thin_prism_coeffs, need_thin_prism_coeffs_grad),
        scratch_contig.data_ptr<float>(),
        current_stream(pts));
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intrinsic_grads.focal_length,
        intrinsic_grads.principal_point,
        intrinsic_grads.radial_coeffs,
        intrinsic_grads.tangential_coeffs,
        intrinsic_grads.thin_prism_coeffs};
}

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
    bool need_thin_prism_coeffs_grad) {
    check_optional_distortion(external_distortion);
    const auto& pts = world_points;
    const auto& valid = valid_flags;
    const auto& grad = grad_image_points;
    const auto& scratch_contig = scratch;
    check_matrix(pts, 3, "world_points");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_matrix(grad, 2, "grad_image_points");
    TORCH_CHECK(valid.dim() == 1 && valid.size(0) == pts.size(0), "valid_flags must be (N,)");
    check_shutter_type(shutter_type);
    check_max_iterations(max_iterations);
    check_cuda_float_contiguous(pts, "world_points");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    TORCH_CHECK(valid.is_cuda(), "valid_flags must be a CUDA tensor");
    TORCH_CHECK(valid.scalar_type() == at::kBool, "valid_flags must be bool");
    TORCH_CHECK(valid.is_contiguous(), "valid_flags must be contiguous");
    check_cuda_float_contiguous(grad, "grad_image_points");
    check_cuda_float_contiguous(scratch_contig, "scratch");
    TORCH_CHECK(scratch_contig.numel() >= pts.size(0) * 10, "scratch too small for project_world_points_shutter_pose backward");
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_world_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad
        ? at::zeros({3}, pts.options())
        : empty_scratch(pts.options());
    auto grad_end_t = need_end_translation_grad
        ? at::zeros({3}, pts.options())
        : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad
        ? at::zeros({4}, pts.options())
        : empty_scratch(pts.options());
    auto grad_end_r = need_end_rotation_grad
        ? at::zeros({4}, pts.options())
        : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad);
    project_world_points_shutter_pose_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        start_rotation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        shutter_type,
        max_iterations,
        static_cast<float>(initial_relative_time),
        valid.data_ptr<bool>(),
        grad.data_ptr<float>(),
        need_world_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_start_translation_grad ? grad_start_t.data_ptr<float>() : nullptr,
        need_end_translation_grad ? grad_end_t.data_ptr<float>() : nullptr,
        need_start_rotation_grad ? grad_start_r.data_ptr<float>() : nullptr,
        need_end_rotation_grad ? grad_end_r.data_ptr<float>() : nullptr,
        maybe_data_ptr(intrinsic_grads.focal_length, need_focal_length_grad),
        maybe_data_ptr(intrinsic_grads.principal_point, need_principal_point_grad),
        maybe_data_ptr(intrinsic_grads.radial_coeffs, need_radial_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.tangential_coeffs, need_tangential_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.thin_prism_coeffs, need_thin_prism_coeffs_grad),
        scratch_contig.data_ptr<float>(),
        current_stream(pts));
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intrinsic_grads.focal_length,
        intrinsic_grads.principal_point,
        intrinsic_grads.radial_coeffs,
        intrinsic_grads.tangential_coeffs,
        intrinsic_grads.thin_prism_coeffs};
}

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
    bool need_thin_prism_coeffs_grad) {
    check_optional_distortion(external_distortion);
    const auto& pts = image_points;
    const auto& grad = grad_world_rays;
    const auto& scratch_contig = scratch;
    check_matrix(pts, 2, "image_points");
    check_matrix(translation, 3, "translation");
    check_matrix(rotation, 4, "rotation");
    TORCH_CHECK(translation.size(0) >= 1 && rotation.size(0) >= 1, "static pose requires one control pose");
    check_matrix(grad, 6, "grad_world_rays");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(translation, "translation");
    check_cuda_float_contiguous(rotation, "rotation");
    check_cuda_float_contiguous(grad, "grad_world_rays");
    check_cuda_float_contiguous(scratch_contig, "scratch");
    TORCH_CHECK(scratch_contig.numel() >= pts.size(0) * 5, "scratch too small for image_points_to_world_rays_static_pose backward");
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_t = need_translation_grad ? at::zeros_like(translation) : empty_scratch(translation.options());
    auto grad_r = need_rotation_grad ? at::zeros_like(rotation) : empty_scratch(rotation.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad);
    image_points_to_world_rays_static_pose_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        translation.data_ptr<float>(),
        rotation.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_translation_grad ? grad_t.data_ptr<float>() : nullptr,
        need_rotation_grad ? grad_r.data_ptr<float>() : nullptr,
        maybe_data_ptr(intrinsic_grads.focal_length, need_focal_length_grad),
        maybe_data_ptr(intrinsic_grads.principal_point, need_principal_point_grad),
        maybe_data_ptr(intrinsic_grads.radial_coeffs, need_radial_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.tangential_coeffs, need_tangential_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.thin_prism_coeffs, need_thin_prism_coeffs_grad),
        scratch_contig.data_ptr<float>(),
        current_stream(pts));
    return {grad_pts, grad_t, grad_r, intrinsic_grads.focal_length, intrinsic_grads.principal_point, intrinsic_grads.radial_coeffs, intrinsic_grads.tangential_coeffs, intrinsic_grads.thin_prism_coeffs};
}

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
    bool need_thin_prism_coeffs_grad) {
    check_optional_distortion(external_distortion);
    const auto& pts = image_points;
    const auto& grad = grad_world_rays;
    const auto& scratch_contig = scratch;
    check_matrix(pts, 2, "image_points");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_matrix(grad, 6, "grad_world_rays");
    check_shutter_type(shutter_type);
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    check_cuda_float_contiguous(grad, "grad_world_rays");
    check_cuda_float_contiguous(scratch_contig, "scratch");
    TORCH_CHECK(scratch_contig.numel() >= pts.size(0) * 9, "scratch too small for image_points_to_world_rays_shutter_pose backward");
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad
        ? at::zeros({3}, pts.options())
        : empty_scratch(pts.options());
    auto grad_end_t = need_end_translation_grad
        ? at::zeros({3}, pts.options())
        : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad
        ? at::zeros({4}, pts.options())
        : empty_scratch(pts.options());
    auto grad_end_r = need_end_rotation_grad
        ? at::zeros({4}, pts.options())
        : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad);
    image_points_to_world_rays_shutter_pose_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        start_rotation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        shutter_type,
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_start_translation_grad ? grad_start_t.data_ptr<float>() : nullptr,
        need_end_translation_grad ? grad_end_t.data_ptr<float>() : nullptr,
        need_start_rotation_grad ? grad_start_r.data_ptr<float>() : nullptr,
        need_end_rotation_grad ? grad_end_r.data_ptr<float>() : nullptr,
        maybe_data_ptr(intrinsic_grads.focal_length, need_focal_length_grad),
        maybe_data_ptr(intrinsic_grads.principal_point, need_principal_point_grad),
        maybe_data_ptr(intrinsic_grads.radial_coeffs, need_radial_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.tangential_coeffs, need_tangential_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.thin_prism_coeffs, need_thin_prism_coeffs_grad),
        scratch_contig.data_ptr<float>(),
        current_stream(pts));
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intrinsic_grads.focal_length,
        intrinsic_grads.principal_point,
        intrinsic_grads.radial_coeffs,
        intrinsic_grads.tangential_coeffs,
        intrinsic_grads.thin_prism_coeffs};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> camera_rays_to_image_points_opencv_pinhole_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& camera_rays) {
    const auto& rays = camera_rays;
    check_matrix(rays, 3, "camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_projection_for_device(projection, rays.device());
    check_bivariate_windshield_for_device(external_distortion, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto image_points = at::empty({rays.size(0), 2}, rays.options());
    auto valid_flags = at::empty({rays.size(0)}, rays.options().dtype(at::kBool));
    const bool save_scratch = rays.requires_grad() ||
        needs_projection_grad(projection) ||
        needs_external_distortion_grad(external_distortion);
    auto scratch = save_scratch ?
        scratch_shape(rays.size(0), 10, rays) :
        empty_scratch(rays.options());
    camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_forward_launch(
        rays.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        rays.data_ptr<float>(),
        image_points.data_ptr<float>(),
        valid_flags.data_ptr<bool>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(rays));
    return {image_points, valid_flags, scratch};
}

std::tuple<at::Tensor, at::Tensor> image_points_to_camera_rays_opencv_pinhole_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points) {
    const auto& pts = image_points;
    check_matrix(pts, 2, "image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto camera_rays = at::empty({pts.size(0), 3}, pts.options());
    const bool save_scratch = pts.requires_grad() ||
        needs_projection_grad(projection) ||
        needs_external_distortion_grad(external_distortion);
    auto scratch = save_scratch ?
        scratch_shape(pts.size(0), 9, pts) :
        empty_scratch(pts.options());
    image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_forward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        camera_rays.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts));
    return {camera_rays, scratch};
}

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
    int64_t end_timestamp_us) {
    const auto& pts = world_points;
    check_matrix(pts, 3, "world_points");
    check_component_vector(start_translation, 3, "start_translation");
    check_component_vector(end_translation, 3, "end_translation");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_cuda_float_contiguous(pts, "world_points");
    check_cuda_float_contiguous(start_translation, "start_translation");
    check_cuda_float_contiguous(end_translation, "end_translation");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    TORCH_CHECK(start_translation.device() == pts.device(), "start_translation device must match world_points");
    TORCH_CHECK(end_translation.device() == pts.device(), "end_translation device must match world_points");
    TORCH_CHECK(start_rotation.device() == pts.device(), "start_rotation device must match world_points");
    TORCH_CHECK(end_rotation.device() == pts.device(), "end_rotation device must match world_points");
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto image_points = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad() ||
        start_translation.requires_grad() ||
        end_translation.requires_grad() ||
        start_rotation.requires_grad() ||
        end_rotation.requires_grad() ||
        needs_projection_grad(projection) ||
        needs_external_distortion_grad(external_distortion);
    auto scratch = save_scratch ?
        scratch_shape(pts.size(0), 9, pts) :
        empty_scratch(pts.options());
    const auto mean_timestamp_us = static_cast<int64_t>(
        static_cast<double>(start_timestamp_us) +
        0.5 * static_cast<double>(end_timestamp_us - start_timestamp_us));
    project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_forward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        start_translation.data_ptr<float>(),
        start_rotation.data_ptr<float>(),
        end_translation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        mean_timestamp_us,
        image_points.data_ptr<float>(),
        valid_flags.data_ptr<bool>(),
        timestamps.data_ptr<int64_t>(),
        pose_t.data_ptr<float>(),
        pose_r.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts));
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

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
    double initial_relative_time) {
    const auto& pts = world_points;
    check_matrix(pts, 3, "world_points");
    check_component_vector(start_translation, 3, "start_translation");
    check_component_vector(end_translation, 3, "end_translation");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_shutter_resolution(width, height);
    check_shutter_type(shutter_type);
    check_max_iterations(max_iterations);
    check_cuda_float_contiguous(pts, "world_points");
    check_cuda_float_contiguous(start_translation, "start_translation");
    check_cuda_float_contiguous(end_translation, "end_translation");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    TORCH_CHECK(start_translation.device() == pts.device(), "start_translation device must match world_points");
    TORCH_CHECK(end_translation.device() == pts.device(), "end_translation device must match world_points");
    TORCH_CHECK(start_rotation.device() == pts.device(), "start_rotation device must match world_points");
    TORCH_CHECK(end_rotation.device() == pts.device(), "end_rotation device must match world_points");
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto image_points = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad() ||
        start_translation.requires_grad() ||
        end_translation.requires_grad() ||
        start_rotation.requires_grad() ||
        end_rotation.requires_grad() ||
        needs_projection_grad(projection) ||
        needs_external_distortion_grad(external_distortion);
    auto scratch = save_scratch ?
        scratch_shape(pts.size(0), 10, pts) :
        empty_scratch(pts.options());
    project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_forward_launch(
        pts.size(0),
        kernel_params_with_resolution(projection, width, height),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        start_translation.data_ptr<float>(),
        start_rotation.data_ptr<float>(),
        end_translation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        shutter_type,
        start_timestamp_us,
        end_timestamp_us,
        max_iterations,
        static_cast<float>(stop_mean_error_px),
        static_cast<float>(stop_delta_mean_error_px),
        static_cast<float>(initial_relative_time),
        image_points.data_ptr<float>(),
        valid_flags.data_ptr<bool>(),
        timestamps.data_ptr<int64_t>(),
        pose_t.data_ptr<float>(),
        pose_r.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts));
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVPinholeProjection>& projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion>& external_distortion,
    const at::Tensor& image_points,
    const at::Tensor& translations,
    const at::Tensor& rotations,
    int64_t timestamp_us) {
    const auto& pts = image_points;
    const auto& trans = translations;
    const auto& rots = rotations;
    check_matrix(pts, 2, "image_points");
    check_matrix(trans, 3, "translations");
    check_matrix(rots, 4, "rotations");
    TORCH_CHECK(trans.size(0) >= 1 && rots.size(0) >= 1, "static pose requires one control pose");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(trans, "translations");
    check_cuda_float_contiguous(rots, "rotations");
    TORCH_CHECK(trans.device() == pts.device() && rots.device() == pts.device(), "pose tensors device must match image_points");
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto world_rays = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad() ||
        trans.requires_grad() ||
        rots.requires_grad() ||
        needs_projection_grad(projection) ||
        needs_external_distortion_grad(external_distortion);
    auto scratch = save_scratch ?
        scratch_shape(pts.size(0), 9, pts) :
        empty_scratch(pts.options());
    image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_forward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        trans.data_ptr<float>(),
        rots.data_ptr<float>(),
        timestamp_us,
        world_rays.data_ptr<float>(),
        timestamps.data_ptr<int64_t>(),
        pose_t.data_ptr<float>(),
        pose_r.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts));
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

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
    int64_t end_timestamp_us) {
    const auto& pts = image_points;
    check_matrix(pts, 2, "image_points");
    check_component_vector(start_translation, 3, "start_translation");
    check_component_vector(end_translation, 3, "end_translation");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_shutter_resolution(width, height);
    check_shutter_type(shutter_type);
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(start_translation, "start_translation");
    check_cuda_float_contiguous(end_translation, "end_translation");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    TORCH_CHECK(start_translation.device() == pts.device(), "start_translation device must match image_points");
    TORCH_CHECK(end_translation.device() == pts.device(), "end_translation device must match image_points");
    TORCH_CHECK(start_rotation.device() == pts.device(), "start_rotation device must match image_points");
    TORCH_CHECK(end_rotation.device() == pts.device(), "end_rotation device must match image_points");
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto world_rays = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad() ||
        start_translation.requires_grad() ||
        end_translation.requires_grad() ||
        start_rotation.requires_grad() ||
        end_rotation.requires_grad() ||
        needs_projection_grad(projection) ||
        needs_external_distortion_grad(external_distortion);
    auto scratch = save_scratch ?
        scratch_shape(pts.size(0), 12, pts) :
        empty_scratch(pts.options());
    image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_forward_launch(
        pts.size(0),
        kernel_params_with_resolution(projection, width, height),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        start_translation.data_ptr<float>(),
        start_rotation.data_ptr<float>(),
        end_translation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        shutter_type,
        start_timestamp_us,
        end_timestamp_us,
        world_rays.data_ptr<float>(),
        timestamps.data_ptr<int64_t>(),
        pose_t.data_ptr<float>(),
        pose_r.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts));
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

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
    bool need_distortion_coeffs_grad) {
    const auto& rays = camera_rays;
    const auto& grad = grad_image_points;
    check_matrix(rays, 3, "camera_rays");
    check_matrix(grad, 2, "grad_image_points");
    TORCH_CHECK(grad.size(0) == rays.size(0), "grad_image_points batch must match camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_cuda_float_contiguous(grad, "grad_image_points");
    check_cuda_float_contiguous(scratch, "scratch");
    TORCH_CHECK(scratch.numel() >= rays.size(0) * 10, "scratch too small for bivariate camera_rays_to_image_points backward");
    check_projection_for_device(projection, rays.device());
    check_bivariate_windshield_for_device(external_distortion, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto grad_rays = need_camera_ray_grad ? at::empty_like(rays) : empty_scratch(rays.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad);
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_backward_launch(
        rays.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        rays.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_camera_ray_grad ? grad_rays.data_ptr<float>() : nullptr,
        maybe_data_ptr(intrinsic_grads.focal_length, need_focal_length_grad),
        maybe_data_ptr(intrinsic_grads.principal_point, need_principal_point_grad),
        maybe_data_ptr(intrinsic_grads.radial_coeffs, need_radial_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.tangential_coeffs, need_tangential_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.thin_prism_coeffs, need_thin_prism_coeffs_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(rays));
    return {grad_rays, intrinsic_grads.focal_length, intrinsic_grads.principal_point, intrinsic_grads.radial_coeffs, intrinsic_grads.tangential_coeffs, intrinsic_grads.thin_prism_coeffs, grad_distortion};
}

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
    bool need_distortion_coeffs_grad) {
    const auto& pts = image_points;
    const auto& grad = grad_camera_rays;
    check_matrix(pts, 2, "image_points");
    check_matrix(grad, 3, "grad_camera_rays");
    TORCH_CHECK(grad.size(0) == pts.size(0), "grad_camera_rays batch must match image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(grad, "grad_camera_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    TORCH_CHECK(scratch.numel() >= pts.size(0) * 9, "scratch too small for bivariate image_points_to_camera_rays backward");
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad);
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        maybe_data_ptr(intrinsic_grads.focal_length, need_focal_length_grad),
        maybe_data_ptr(intrinsic_grads.principal_point, need_principal_point_grad),
        maybe_data_ptr(intrinsic_grads.radial_coeffs, need_radial_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.tangential_coeffs, need_tangential_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.thin_prism_coeffs, need_thin_prism_coeffs_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(pts));
    return {grad_pts, intrinsic_grads.focal_length, intrinsic_grads.principal_point, intrinsic_grads.radial_coeffs, intrinsic_grads.tangential_coeffs, intrinsic_grads.thin_prism_coeffs, grad_distortion};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
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
    bool need_distortion_coeffs_grad) {
    const auto& pts = world_points;
    const auto& grad = grad_image_points;
    check_matrix(pts, 3, "world_points");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_matrix(grad, 2, "grad_image_points");
    TORCH_CHECK(grad.size(0) == pts.size(0), "grad_image_points batch must match world_points");
    check_cuda_float_contiguous(pts, "world_points");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    check_cuda_float_contiguous(grad, "grad_image_points");
    check_cuda_float_contiguous(scratch, "scratch");
    TORCH_CHECK(scratch.numel() >= pts.size(0) * 9, "scratch too small for bivariate project_world_points_mean_pose backward");
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_world_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad);
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        start_rotation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_world_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_start_translation_grad ? grad_start_t.data_ptr<float>() : nullptr,
        need_end_translation_grad ? grad_end_t.data_ptr<float>() : nullptr,
        need_start_rotation_grad ? grad_start_r.data_ptr<float>() : nullptr,
        need_end_rotation_grad ? grad_end_r.data_ptr<float>() : nullptr,
        maybe_data_ptr(intrinsic_grads.focal_length, need_focal_length_grad),
        maybe_data_ptr(intrinsic_grads.principal_point, need_principal_point_grad),
        maybe_data_ptr(intrinsic_grads.radial_coeffs, need_radial_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.tangential_coeffs, need_tangential_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.thin_prism_coeffs, need_thin_prism_coeffs_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(pts));
    return {grad_pts, grad_start_t, grad_end_t, grad_start_r, grad_end_r, intrinsic_grads.focal_length, intrinsic_grads.principal_point, intrinsic_grads.radial_coeffs, intrinsic_grads.tangential_coeffs, intrinsic_grads.thin_prism_coeffs, grad_distortion};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
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
    bool need_distortion_coeffs_grad) {
    const auto& pts = world_points;
    const auto& valid = valid_flags;
    const auto& grad = grad_image_points;
    check_matrix(pts, 3, "world_points");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_matrix(grad, 2, "grad_image_points");
    TORCH_CHECK(valid.dim() == 1 && valid.size(0) == pts.size(0), "valid_flags must be (N,)");
    check_shutter_type(shutter_type);
    check_max_iterations(max_iterations);
    check_cuda_float_contiguous(pts, "world_points");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    TORCH_CHECK(valid.is_cuda(), "valid_flags must be a CUDA tensor");
    TORCH_CHECK(valid.scalar_type() == at::kBool, "valid_flags must be bool");
    TORCH_CHECK(valid.is_contiguous(), "valid_flags must be contiguous");
    check_cuda_float_contiguous(grad, "grad_image_points");
    check_cuda_float_contiguous(scratch, "scratch");
    TORCH_CHECK(scratch.numel() >= pts.size(0) * 10, "scratch too small for bivariate project_world_points_shutter_pose backward");
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_world_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad);
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        start_rotation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        shutter_type,
        max_iterations,
        static_cast<float>(initial_relative_time),
        valid.data_ptr<bool>(),
        grad.data_ptr<float>(),
        need_world_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_start_translation_grad ? grad_start_t.data_ptr<float>() : nullptr,
        need_end_translation_grad ? grad_end_t.data_ptr<float>() : nullptr,
        need_start_rotation_grad ? grad_start_r.data_ptr<float>() : nullptr,
        need_end_rotation_grad ? grad_end_r.data_ptr<float>() : nullptr,
        maybe_data_ptr(intrinsic_grads.focal_length, need_focal_length_grad),
        maybe_data_ptr(intrinsic_grads.principal_point, need_principal_point_grad),
        maybe_data_ptr(intrinsic_grads.radial_coeffs, need_radial_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.tangential_coeffs, need_tangential_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.thin_prism_coeffs, need_thin_prism_coeffs_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(pts));
    return {grad_pts, grad_start_t, grad_end_t, grad_start_r, grad_end_r, intrinsic_grads.focal_length, intrinsic_grads.principal_point, intrinsic_grads.radial_coeffs, intrinsic_grads.tangential_coeffs, intrinsic_grads.thin_prism_coeffs, grad_distortion};
}

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
    bool need_distortion_coeffs_grad) {
    const auto& pts = image_points;
    const auto& grad = grad_world_rays;
    check_matrix(pts, 2, "image_points");
    check_matrix(translation, 3, "translation");
    check_matrix(rotation, 4, "rotation");
    TORCH_CHECK(translation.size(0) >= 1 && rotation.size(0) >= 1, "static pose requires one control pose");
    check_matrix(grad, 6, "grad_world_rays");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(translation, "translation");
    check_cuda_float_contiguous(rotation, "rotation");
    check_cuda_float_contiguous(grad, "grad_world_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    TORCH_CHECK(scratch.numel() >= pts.size(0) * 9, "scratch too small for bivariate image_points_to_world_rays_static_pose backward");
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_t = need_translation_grad ? at::zeros_like(translation) : empty_scratch(translation.options());
    auto grad_r = need_rotation_grad ? at::zeros_like(rotation) : empty_scratch(rotation.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad);
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        translation.data_ptr<float>(),
        rotation.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_translation_grad ? grad_t.data_ptr<float>() : nullptr,
        need_rotation_grad ? grad_r.data_ptr<float>() : nullptr,
        maybe_data_ptr(intrinsic_grads.focal_length, need_focal_length_grad),
        maybe_data_ptr(intrinsic_grads.principal_point, need_principal_point_grad),
        maybe_data_ptr(intrinsic_grads.radial_coeffs, need_radial_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.tangential_coeffs, need_tangential_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.thin_prism_coeffs, need_thin_prism_coeffs_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(pts));
    return {grad_pts, grad_t, grad_r, intrinsic_grads.focal_length, intrinsic_grads.principal_point, intrinsic_grads.radial_coeffs, intrinsic_grads.tangential_coeffs, intrinsic_grads.thin_prism_coeffs, grad_distortion};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
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
    bool need_distortion_coeffs_grad) {
    const auto& pts = image_points;
    const auto& grad = grad_world_rays;
    check_matrix(pts, 2, "image_points");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_matrix(grad, 6, "grad_world_rays");
    check_shutter_type(shutter_type);
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    check_cuda_float_contiguous(grad, "grad_world_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    TORCH_CHECK(scratch.numel() >= pts.size(0) * 12, "scratch too small for bivariate image_points_to_world_rays_shutter_pose backward");
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad);
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        start_rotation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        shutter_type,
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_start_translation_grad ? grad_start_t.data_ptr<float>() : nullptr,
        need_end_translation_grad ? grad_end_t.data_ptr<float>() : nullptr,
        need_start_rotation_grad ? grad_start_r.data_ptr<float>() : nullptr,
        need_end_rotation_grad ? grad_end_r.data_ptr<float>() : nullptr,
        maybe_data_ptr(intrinsic_grads.focal_length, need_focal_length_grad),
        maybe_data_ptr(intrinsic_grads.principal_point, need_principal_point_grad),
        maybe_data_ptr(intrinsic_grads.radial_coeffs, need_radial_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.tangential_coeffs, need_tangential_coeffs_grad),
        maybe_data_ptr(intrinsic_grads.thin_prism_coeffs, need_thin_prism_coeffs_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(pts));
    return {grad_pts, grad_start_t, grad_end_t, grad_start_r, grad_end_r, intrinsic_grads.focal_length, intrinsic_grads.principal_point, intrinsic_grads.radial_coeffs, intrinsic_grads.tangential_coeffs, intrinsic_grads.thin_prism_coeffs, grad_distortion};
}

} // namespace gsplat_sensors
