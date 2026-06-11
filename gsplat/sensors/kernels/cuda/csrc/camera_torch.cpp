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

// Host wrappers validate tensors, allocate autograd scratch, and dispatch CUDA
// launch helpers while keeping TORCH_LIBRARY bindings thin.

#include "camera_torch.h"
#include "shutter_type.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <numbers>
#include <utility>

namespace gsplat_sensors
{
namespace
{
    // ===========================================================================
    // Shared validation and allocation helpers (internal linkage)
    // ===========================================================================

    void check_component_shape(const at::Tensor &tensor, at::IntArrayRef shape, const char *name)
    {
        TORCH_CHECK(tensor.sizes() == shape, name, " must be ", shape);
    }

    void check_cuda_float_contiguous(const at::Tensor &tensor, const char *name)
    {
        TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    }

    void check_tensor_device(const at::Tensor &tensor, const at::Device &device, const char *name)
    {
        TORCH_CHECK(tensor.device() == device, name, " device must match inputs");
    }

    void check_matrix(const at::Tensor &tensor, int64_t cols, const char *name)
    {
        TORCH_CHECK(tensor.dim() == 2 && tensor.size(1) == cols, name, " must be (N, ", cols, ")");
    }

    void check_component_vector(const at::Tensor &tensor, int64_t expected, const char *name)
    {
        if(tensor.dim() == 2)
        {
            TORCH_CHECK(
                tensor.size(0) == 1 && tensor.size(1) == expected,
                name,
                " must be (",
                expected,
                ",) or (1, ",
                expected,
                ")"
            );
        }
        else
        {
            TORCH_CHECK(
                tensor.dim() == 1 && tensor.size(0) == expected,
                name,
                " must be (",
                expected,
                ",) or (1, ",
                expected,
                ")"
            );
        }
    }

    void check_projection_for_device(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection, const at::Device &device
    )
    {
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

    cudaStream_t current_stream(const at::Tensor &tensor)
    {
        return at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    }

    at::Tensor empty_scratch(const at::TensorOptions &options)
    {
        return at::empty({0}, options);
    }

    at::Tensor zeros_like_shape(at::IntArrayRef shape, const at::Tensor &reference)
    {
        return at::zeros(shape, reference.options());
    }

    at::Tensor scratch_shape(int64_t count, int64_t stride, const at::Tensor &reference)
    {
        return at::empty({count, stride}, reference.options());
    }

    template<DistortionSensor Sensor, DistortionOpFamily Op, typename PolicyTag>
    at::Tensor scratch_shape_for_forward(int64_t count, const at::Tensor &reference)
    {
        constexpr int stride
            = DistortionScratchTraits<Sensor, Op, DistortionDirection::Forward, PolicyTag>::kScratchStride;
        return scratch_shape(count, stride, reference);
    }

    template<DistortionSensor Sensor, DistortionOpFamily Op, typename PolicyTag>
    void check_backward_scratch(const at::Tensor &scratch, int64_t count, const char *op_name)
    {
        constexpr int stride
            = DistortionScratchTraits<Sensor, Op, DistortionDirection::Backward, PolicyTag>::kScratchStride;
        TORCH_CHECK(scratch.numel() >= count * stride, "scratch too small for ", op_name, " backward");
    }

    bool needs_projection_grad(const c10::intrusive_ptr<OpenCVPinholeProjection> &projection)
    {
        return projection->focal_length.requires_grad()
            || projection->principal_point.requires_grad()
            || projection->radial_coeffs.requires_grad()
            || projection->tangential_coeffs.requires_grad()
            || projection->thin_prism_coeffs.requires_grad();
    }

    struct IntrinsicGradOutputs
    {
        at::Tensor focal_length;
        at::Tensor principal_point;
        at::Tensor radial_coeffs;
        at::Tensor tangential_coeffs;
        at::Tensor thin_prism_coeffs;
    };

    at::Tensor maybe_intrinsic_grad(bool needed, at::IntArrayRef shape, const at::Tensor &reference)
    {
        return needed ? zeros_like_shape(shape, reference) : empty_scratch(reference.options());
    }

    float *maybe_data_ptr(at::Tensor &tensor, bool needed)
    {
        return needed ? tensor.data_ptr<float>() : nullptr;
    }

    IntrinsicGradOutputs make_intrinsic_grad_outputs(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        bool need_focal_length_grad,
        bool need_principal_point_grad,
        bool need_radial_coeffs_grad,
        bool need_tangential_coeffs_grad,
        bool need_thin_prism_coeffs_grad
    )
    {
        return {
            maybe_intrinsic_grad(need_focal_length_grad, {2}, projection->focal_length),
            maybe_intrinsic_grad(need_principal_point_grad, {2}, projection->principal_point),
            maybe_intrinsic_grad(need_radial_coeffs_grad, {6}, projection->radial_coeffs),
            maybe_intrinsic_grad(need_tangential_coeffs_grad, {2}, projection->tangential_coeffs),
            maybe_intrinsic_grad(need_thin_prism_coeffs_grad, {4}, projection->thin_prism_coeffs),
        };
    }

    OpenCVPinholeProjection_KernelParameters kernel_params_with_resolution(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection, int64_t width, int64_t height
    )
    {
        auto params   = projection->to_kernel_params();
        params.width  = width;
        params.height = height;
        return params;
    }

    void check_optional_distortion(const c10::intrusive_ptr<NoExternalDistortion> &external_distortion)
    {
        TORCH_CHECK(external_distortion != nullptr, "external_distortion must be NoExternalDistortion");
    }

    void check_bivariate_windshield_for_device(
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion, const at::Device &device
    )
    {
        check_bivariate_windshield_distortion(external_distortion);
        check_cuda_float_contiguous(external_distortion->distortion_coeffs, "distortion_coeffs");
        TORCH_CHECK(
            external_distortion->distortion_coeffs.device() == device, "distortion_coeffs device must match inputs"
        );
    }

    bool needs_external_distortion_grad(const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion)
    {
        return external_distortion->distortion_coeffs.requires_grad();
    }

    at::Tensor make_distortion_grad_output(
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion, bool needed
    )
    {
        return needed ? at::zeros_like(external_distortion->distortion_coeffs)
                      : empty_scratch(external_distortion->distortion_coeffs.options());
    }

    void check_max_iterations(int64_t max_iterations)
    {
        TORCH_CHECK(max_iterations >= 1, "max_iterations must be at least 1");
        TORCH_CHECK(
            max_iterations <= kMaxRollingShutterIterations, "max_iterations must be <= ", kMaxRollingShutterIterations
        );
    }

    // Rejects any integer that does not map to a known ShutterType enumerator.
    // Must be kept in sync with shutter_type.h.
    void check_shutter_type(int64_t shutter_type)
    {
        switch(static_cast<ShutterType>(shutter_type))
        {
        case ShutterType::ROLLING_TOP_TO_BOTTOM:
        case ShutterType::ROLLING_LEFT_TO_RIGHT:
        case ShutterType::ROLLING_BOTTOM_TO_TOP:
        case ShutterType::ROLLING_RIGHT_TO_LEFT:
        case ShutterType::GLOBAL:                return;
        }
        TORCH_CHECK(
            false, "shutter_type=", shutter_type, " is not a valid gsplat_sensors::ShutterType (see shutter_type.h)"
        );
    }

    // Ensures width and height are both positive; required by rolling-shutter
    // kernels that use pixel position to derive per-point exposure timestamps.
    void check_shutter_resolution(int64_t width, int64_t height)
    {
        TORCH_CHECK(
            width > 0 && height > 0, "resolution width and height must be positive for rolling-shutter kernels"
        );
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
    std::array<int64_t, 2> resolution_
)
    : focal_length(std::move(focal_length_))
    , principal_point(std::move(principal_point_))
    , radial_coeffs(std::move(radial_coeffs_))
    , tangential_coeffs(std::move(tangential_coeffs_))
    , thin_prism_coeffs(std::move(thin_prism_coeffs_))
    , resolution(resolution_)
{
}

OpenCVPinholeProjection_KernelParameters OpenCVPinholeProjection::to_kernel_params() const
{
    OpenCVPinholeProjection_KernelParameters params{};
    params.focal_length      = focal_length.const_data_ptr<float>();
    params.principal_point   = principal_point.const_data_ptr<float>();
    params.radial_coeffs     = radial_coeffs.const_data_ptr<float>();
    params.tangential_coeffs = tangential_coeffs.const_data_ptr<float>();
    params.thin_prism_coeffs = thin_prism_coeffs.const_data_ptr<float>();
    params.width             = resolution[0];
    params.height            = resolution[1];
    return params;
}

void check_projection(const c10::intrusive_ptr<OpenCVPinholeProjection> &projection)
{
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

at::Tensor generate_image_points(int64_t width, int64_t height, c10::Device device)
{
    TORCH_CHECK(width >= 0 && height >= 0, "resolution dimensions must be non-negative");
    TORCH_CHECK(device.is_cuda(), "generate_image_points requires a CUDA device");
    auto guard        = c10::cuda::CUDAGuard(device);
    auto options      = at::TensorOptions().device(device).dtype(at::kFloat);
    auto image_points = at::empty({height, width, 2}, options);
    generate_image_points_launch(
        width, height, image_points.data_ptr<float>(), at::cuda::getCurrentCUDAStream(device.index()).stream()
    );
    return image_points;
}

// ===========================================================================
// camera_rays_to_image_points — no_external + bivariate_windshield
//   forward + backward
// ===========================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor> camera_rays_to_image_points_opencv_pinhole_no_external(
    const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
    const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
    const at::Tensor &camera_rays
)
{
    check_optional_distortion(external_distortion);
    const auto &rays = camera_rays;
    check_matrix(rays, 3, "camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_projection_for_device(projection, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto image_points       = at::empty({rays.size(0), 2}, rays.options());
    auto valid_flags        = at::empty({rays.size(0)}, rays.options().dtype(at::kBool));
    const bool save_scratch = rays.requires_grad() || needs_projection_grad(projection);
    auto scratch            = save_scratch ? scratch_shape(rays.size(0), 6, rays) : empty_scratch(rays.options());
    camera_rays_to_image_points_forward_launch(
        rays.size(0),
        projection->to_kernel_params(),
        rays.data_ptr<float>(),
        image_points.data_ptr<float>(),
        valid_flags.data_ptr<bool>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(rays)
    );
    return {image_points, valid_flags, scratch};
}

// ===========================================================================
// image_points_to_camera_rays — no_external + bivariate_windshield
//   forward + backward
// ===========================================================================

std::tuple<at::Tensor, at::Tensor> image_points_to_camera_rays_opencv_pinhole_no_external(
    const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
    const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
    const at::Tensor &image_points
)
{
    check_optional_distortion(external_distortion);
    const auto &pts = image_points;
    check_matrix(pts, 2, "image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto camera_rays        = at::empty({pts.size(0), 3}, pts.options());
    const bool save_scratch = pts.requires_grad() || needs_projection_grad(projection);
    auto scratch            = save_scratch ? scratch_shape(pts.size(0), 5, pts) : empty_scratch(pts.options());
    image_points_to_camera_rays_forward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        camera_rays.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts)
    );
    return {camera_rays, scratch};
}

// ===========================================================================
// project_world_points_mean_pose — no_external + bivariate_windshield
//   forward + backward
// ===========================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_mean_pose_opencv_pinhole_no_external(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts = world_points;
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

    auto image_points            = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags             = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps              = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t                  = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r                  = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch      = pts.requires_grad()
                                || start_translation.requires_grad()
                                || end_translation.requires_grad()
                                || start_rotation.requires_grad()
                                || end_rotation.requires_grad()
                                || needs_projection_grad(projection);
    auto scratch                 = save_scratch ? scratch_shape(pts.size(0), 9, pts) : empty_scratch(pts.options());
    const auto mean_timestamp_us = static_cast<int64_t>(
        static_cast<double>(start_timestamp_us) + 0.5 * static_cast<double>(end_timestamp_us - start_timestamp_us)
    );
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
        current_stream(pts)
    );
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

// ===========================================================================
// project_world_points_shutter_pose — no_external + bivariate_windshield
//   forward + backward
// ===========================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_shutter_pose_opencv_pinhole_no_external(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t width,
        int64_t height,
        int64_t shutter_type,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us,
        int64_t max_iterations,
        double stop_mean_error_px,
        double stop_delta_mean_error_px,
        double initial_relative_time
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts = world_points;
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

    auto image_points       = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags        = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || start_translation.requires_grad()
                           || end_translation.requires_grad()
                           || start_rotation.requires_grad()
                           || end_rotation.requires_grad()
                           || needs_projection_grad(projection);
    auto scratch            = save_scratch ? scratch_shape(pts.size(0), 10, pts) : empty_scratch(pts.options());
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
        current_stream(pts)
    );
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

// ===========================================================================
// image_points_to_world_rays_static_pose — no_external + bivariate_windshield
//   forward + backward
// ===========================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_static_pose_opencv_pinhole_no_external(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &translations,
        const at::Tensor &rotations,
        int64_t timestamp_us
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts   = image_points;
    const auto &trans = translations;
    const auto &rots  = rotations;
    check_matrix(pts, 2, "image_points");
    check_matrix(trans, 3, "translations");
    check_matrix(rots, 4, "rotations");
    TORCH_CHECK(trans.size(0) == 1 && rots.size(0) == 1, "static pose requires exactly one control pose");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(trans, "translations");
    check_cuda_float_contiguous(rots, "rotations");
    TORCH_CHECK(
        trans.device() == pts.device() && rots.device() == pts.device(), "pose tensors device must match image_points"
    );
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto world_rays = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t     = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r     = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch
        = pts.requires_grad() || trans.requires_grad() || rots.requires_grad() || needs_projection_grad(projection);
    auto scratch = save_scratch ? scratch_shape(pts.size(0), 5, pts) : empty_scratch(pts.options());
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
        current_stream(pts)
    );
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

// ===========================================================================
// image_points_to_world_rays_shutter_pose — no_external + bivariate_windshield
//   forward + backward
// ===========================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t width,
        int64_t height,
        int64_t shutter_type,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts = image_points;
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

    auto world_rays         = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || start_translation.requires_grad()
                           || end_translation.requires_grad()
                           || start_rotation.requires_grad()
                           || end_rotation.requires_grad()
                           || needs_projection_grad(projection);
    auto scratch            = save_scratch ? scratch_shape(pts.size(0), 9, pts) : empty_scratch(pts.options());
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
        current_stream(pts)
    );
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    camera_rays_to_image_points_opencv_pinhole_no_external_backward(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &camera_rays,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
        bool need_camera_ray_grad,
        bool need_focal_length_grad,
        bool need_principal_point_grad,
        bool need_radial_coeffs_grad,
        bool need_tangential_coeffs_grad,
        bool need_thin_prism_coeffs_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &rays           = camera_rays;
    const auto &grad           = grad_image_points;
    const auto &scratch_contig = scratch;
    check_matrix(rays, 3, "camera_rays");
    check_matrix(grad, 2, "grad_image_points");
    TORCH_CHECK(grad.size(0) == rays.size(0), "grad_image_points batch must match camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_cuda_float_contiguous(grad, "grad_image_points");
    check_cuda_float_contiguous(scratch_contig, "scratch");
    TORCH_CHECK(
        scratch_contig.numel() >= rays.size(0) * 6, "scratch too small for camera_rays_to_image_points backward"
    );
    check_projection_for_device(projection, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto grad_rays       = need_camera_ray_grad ? at::empty_like(rays) : empty_scratch(rays.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad
    );
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
        current_stream(rays)
    );
    return {
        grad_rays,
        intrinsic_grads.focal_length,
        intrinsic_grads.principal_point,
        intrinsic_grads.radial_coeffs,
        intrinsic_grads.tangential_coeffs,
        intrinsic_grads.thin_prism_coeffs
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_camera_rays_opencv_pinhole_no_external_backward(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &grad_camera_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_focal_length_grad,
        bool need_principal_point_grad,
        bool need_radial_coeffs_grad,
        bool need_tangential_coeffs_grad,
        bool need_thin_prism_coeffs_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts            = image_points;
    const auto &grad           = grad_camera_rays;
    const auto &scratch_contig = scratch;
    check_matrix(pts, 2, "image_points");
    check_matrix(grad, 3, "grad_camera_rays");
    TORCH_CHECK(grad.size(0) == pts.size(0), "grad_camera_rays batch must match image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(grad, "grad_camera_rays");
    check_cuda_float_contiguous(scratch_contig, "scratch");
    TORCH_CHECK(
        scratch_contig.numel() >= pts.size(0) * 5, "scratch too small for image_points_to_camera_rays backward"
    );
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts        = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad
    );
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
        current_stream(pts)
    );
    return {
        grad_pts,
        intrinsic_grads.focal_length,
        intrinsic_grads.principal_point,
        intrinsic_grads.radial_coeffs,
        intrinsic_grads.tangential_coeffs,
        intrinsic_grads.thin_prism_coeffs
    };
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
    at::Tensor
>
    project_world_points_mean_pose_opencv_pinhole_no_external_backward(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
        bool need_world_point_grad,
        bool need_start_translation_grad,
        bool need_end_translation_grad,
        bool need_start_rotation_grad,
        bool need_end_rotation_grad,
        bool need_focal_length_grad,
        bool need_principal_point_grad,
        bool need_radial_coeffs_grad,
        bool need_tangential_coeffs_grad,
        bool need_thin_prism_coeffs_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts            = world_points;
    const auto &grad           = grad_image_points;
    const auto &scratch_contig = scratch;
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
    TORCH_CHECK(
        scratch_contig.numel() >= pts.size(0) * 9, "scratch too small for project_world_points_mean_pose backward"
    );
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts        = need_world_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t    = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t      = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r    = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r      = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad
    );
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
        current_stream(pts)
    );
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
        intrinsic_grads.thin_prism_coeffs
    };
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
    at::Tensor
>
    project_world_points_shutter_pose_opencv_pinhole_no_external_backward(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        int64_t shutter_type,
        int64_t max_iterations,
        double initial_relative_time,
        const at::Tensor &valid_flags,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
        bool need_world_point_grad,
        bool need_start_translation_grad,
        bool need_end_translation_grad,
        bool need_start_rotation_grad,
        bool need_end_rotation_grad,
        bool need_focal_length_grad,
        bool need_principal_point_grad,
        bool need_radial_coeffs_grad,
        bool need_tangential_coeffs_grad,
        bool need_thin_prism_coeffs_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts            = world_points;
    const auto &valid          = valid_flags;
    const auto &grad           = grad_image_points;
    const auto &scratch_contig = scratch;
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
    TORCH_CHECK(
        scratch_contig.numel() >= pts.size(0) * 10, "scratch too small for project_world_points_shutter_pose backward"
    );
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts        = need_world_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t    = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t      = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r    = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r      = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad
    );
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
        current_stream(pts)
    );
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
        intrinsic_grads.thin_prism_coeffs
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_static_pose_opencv_pinhole_no_external_backward(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &translation,
        const at::Tensor &rotation,
        const at::Tensor &grad_world_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_translation_grad,
        bool need_rotation_grad,
        bool need_focal_length_grad,
        bool need_principal_point_grad,
        bool need_radial_coeffs_grad,
        bool need_tangential_coeffs_grad,
        bool need_thin_prism_coeffs_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts            = image_points;
    const auto &grad           = grad_world_rays;
    const auto &scratch_contig = scratch;
    check_matrix(pts, 2, "image_points");
    check_matrix(translation, 3, "translation");
    check_matrix(rotation, 4, "rotation");
    TORCH_CHECK(translation.size(0) == 1 && rotation.size(0) == 1, "static pose requires exactly one control pose");
    check_matrix(grad, 6, "grad_world_rays");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(translation, "translation");
    check_cuda_float_contiguous(rotation, "rotation");
    check_cuda_float_contiguous(grad, "grad_world_rays");
    check_cuda_float_contiguous(scratch_contig, "scratch");
    TORCH_CHECK(
        scratch_contig.numel() >= pts.size(0) * 5,
        "scratch too small for image_points_to_world_rays_static_pose backward"
    );
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts        = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_t          = need_translation_grad ? at::zeros_like(translation) : empty_scratch(translation.options());
    auto grad_r          = need_rotation_grad ? at::zeros_like(rotation) : empty_scratch(rotation.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad
    );
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
        current_stream(pts)
    );
    return {
        grad_pts,
        grad_t,
        grad_r,
        intrinsic_grads.focal_length,
        intrinsic_grads.principal_point,
        intrinsic_grads.radial_coeffs,
        intrinsic_grads.tangential_coeffs,
        intrinsic_grads.thin_prism_coeffs
    };
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
    at::Tensor
>
    image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external_backward(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        int64_t shutter_type,
        const at::Tensor &grad_world_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_start_translation_grad,
        bool need_end_translation_grad,
        bool need_start_rotation_grad,
        bool need_end_rotation_grad,
        bool need_focal_length_grad,
        bool need_principal_point_grad,
        bool need_radial_coeffs_grad,
        bool need_tangential_coeffs_grad,
        bool need_thin_prism_coeffs_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts            = image_points;
    const auto &grad           = grad_world_rays;
    const auto &scratch_contig = scratch;
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
    TORCH_CHECK(
        scratch_contig.numel() >= pts.size(0) * 9,
        "scratch too small for image_points_to_world_rays_shutter_pose backward"
    );
    check_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts        = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t    = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t      = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r    = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r      = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad
    );
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
        current_stream(pts)
    );
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
        intrinsic_grads.thin_prism_coeffs
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> camera_rays_to_image_points_opencv_pinhole_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
    const at::Tensor &camera_rays
)
{
    const auto &rays = camera_rays;
    check_matrix(rays, 3, "camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_projection_for_device(projection, rays.device());
    check_bivariate_windshield_for_device(external_distortion, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto image_points       = at::empty({rays.size(0), 2}, rays.options());
    auto valid_flags        = at::empty({rays.size(0)}, rays.options().dtype(at::kBool));
    const bool save_scratch = rays.requires_grad()
                           || needs_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape(rays.size(0), 10, rays) : empty_scratch(rays.options());
    camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_forward_launch(
        rays.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        rays.data_ptr<float>(),
        image_points.data_ptr<float>(),
        valid_flags.data_ptr<bool>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(rays)
    );
    return {image_points, valid_flags, scratch};
}

std::tuple<at::Tensor, at::Tensor> image_points_to_camera_rays_opencv_pinhole_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
    const at::Tensor &image_points
)
{
    const auto &pts = image_points;
    check_matrix(pts, 2, "image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto camera_rays        = at::empty({pts.size(0), 3}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || needs_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape(pts.size(0), 9, pts) : empty_scratch(pts.options());
    image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_forward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        camera_rays.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts)
    );
    return {camera_rays, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_mean_pose_opencv_pinhole_bivariate_windshield(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us
    )
{
    const auto &pts = world_points;
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

    auto image_points            = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags             = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps              = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t                  = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r                  = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch      = pts.requires_grad()
                                || start_translation.requires_grad()
                                || end_translation.requires_grad()
                                || start_rotation.requires_grad()
                                || end_rotation.requires_grad()
                                || needs_projection_grad(projection)
                                || needs_external_distortion_grad(external_distortion);
    auto scratch                 = save_scratch ? scratch_shape(pts.size(0), 9, pts) : empty_scratch(pts.options());
    const auto mean_timestamp_us = static_cast<int64_t>(
        static_cast<double>(start_timestamp_us) + 0.5 * static_cast<double>(end_timestamp_us - start_timestamp_us)
    );
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
        current_stream(pts)
    );
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t width,
        int64_t height,
        int64_t shutter_type,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us,
        int64_t max_iterations,
        double stop_mean_error_px,
        double stop_delta_mean_error_px,
        double initial_relative_time
    )
{
    const auto &pts = world_points;
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

    auto image_points       = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags        = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || start_translation.requires_grad()
                           || end_translation.requires_grad()
                           || start_rotation.requires_grad()
                           || end_rotation.requires_grad()
                           || needs_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape(pts.size(0), 10, pts) : empty_scratch(pts.options());
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
        current_stream(pts)
    );
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &translations,
        const at::Tensor &rotations,
        int64_t timestamp_us
    )
{
    const auto &pts   = image_points;
    const auto &trans = translations;
    const auto &rots  = rotations;
    check_matrix(pts, 2, "image_points");
    check_matrix(trans, 3, "translations");
    check_matrix(rots, 4, "rotations");
    TORCH_CHECK(trans.size(0) == 1 && rots.size(0) == 1, "static pose requires exactly one control pose");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(trans, "translations");
    check_cuda_float_contiguous(rots, "rotations");
    TORCH_CHECK(
        trans.device() == pts.device() && rots.device() == pts.device(), "pose tensors device must match image_points"
    );
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto world_rays         = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || trans.requires_grad()
                           || rots.requires_grad()
                           || needs_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape(pts.size(0), 9, pts) : empty_scratch(pts.options());
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
        current_stream(pts)
    );
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t width,
        int64_t height,
        int64_t shutter_type,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us
    )
{
    const auto &pts = image_points;
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

    auto world_rays         = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || start_translation.requires_grad()
                           || end_translation.requires_grad()
                           || start_rotation.requires_grad()
                           || end_rotation.requires_grad()
                           || needs_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape(pts.size(0), 12, pts) : empty_scratch(pts.options());
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
        current_stream(pts)
    );
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_backward(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &camera_rays,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
        bool need_camera_ray_grad,
        bool need_focal_length_grad,
        bool need_principal_point_grad,
        bool need_radial_coeffs_grad,
        bool need_tangential_coeffs_grad,
        bool need_thin_prism_coeffs_grad,
        bool need_distortion_coeffs_grad
    )
{
    const auto &rays = camera_rays;
    const auto &grad = grad_image_points;
    check_matrix(rays, 3, "camera_rays");
    check_matrix(grad, 2, "grad_image_points");
    TORCH_CHECK(grad.size(0) == rays.size(0), "grad_image_points batch must match camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_cuda_float_contiguous(grad, "grad_image_points");
    check_cuda_float_contiguous(scratch, "scratch");
    TORCH_CHECK(
        scratch.numel() >= rays.size(0) * 10, "scratch too small for bivariate camera_rays_to_image_points backward"
    );
    check_projection_for_device(projection, rays.device());
    check_bivariate_windshield_for_device(external_distortion, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto grad_rays       = need_camera_ray_grad ? at::empty_like(rays) : empty_scratch(rays.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad
    );
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
        current_stream(rays)
    );
    return {
        grad_rays,
        intrinsic_grads.focal_length,
        intrinsic_grads.principal_point,
        intrinsic_grads.radial_coeffs,
        intrinsic_grads.tangential_coeffs,
        intrinsic_grads.thin_prism_coeffs,
        grad_distortion
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_backward(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &grad_camera_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_focal_length_grad,
        bool need_principal_point_grad,
        bool need_radial_coeffs_grad,
        bool need_tangential_coeffs_grad,
        bool need_thin_prism_coeffs_grad,
        bool need_distortion_coeffs_grad
    )
{
    const auto &pts  = image_points;
    const auto &grad = grad_camera_rays;
    check_matrix(pts, 2, "image_points");
    check_matrix(grad, 3, "grad_camera_rays");
    TORCH_CHECK(grad.size(0) == pts.size(0), "grad_camera_rays batch must match image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(grad, "grad_camera_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    TORCH_CHECK(
        scratch.numel() >= pts.size(0) * 9, "scratch too small for bivariate image_points_to_camera_rays backward"
    );
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts        = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad
    );
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
        current_stream(pts)
    );
    return {
        grad_pts,
        intrinsic_grads.focal_length,
        intrinsic_grads.principal_point,
        intrinsic_grads.radial_coeffs,
        intrinsic_grads.tangential_coeffs,
        intrinsic_grads.thin_prism_coeffs,
        grad_distortion
    };
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
    at::Tensor,
    at::Tensor
>
    project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_backward(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
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
        bool need_distortion_coeffs_grad
    )
{
    const auto &pts  = world_points;
    const auto &grad = grad_image_points;
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
    TORCH_CHECK(
        scratch.numel() >= pts.size(0) * 9, "scratch too small for bivariate project_world_points_mean_pose backward"
    );
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts        = need_world_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t    = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t      = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r    = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r      = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad
    );
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
        current_stream(pts)
    );
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
        intrinsic_grads.thin_prism_coeffs,
        grad_distortion
    };
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
    at::Tensor,
    at::Tensor
>
    project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_backward(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        int64_t shutter_type,
        int64_t max_iterations,
        double initial_relative_time,
        const at::Tensor &valid_flags,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
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
        bool need_distortion_coeffs_grad
    )
{
    const auto &pts   = world_points;
    const auto &valid = valid_flags;
    const auto &grad  = grad_image_points;
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
    TORCH_CHECK(
        scratch.numel() >= pts.size(0) * 10,
        "scratch too small for bivariate project_world_points_shutter_pose backward"
    );
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts        = need_world_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t    = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t      = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r    = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r      = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad
    );
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
        current_stream(pts)
    );
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
        intrinsic_grads.thin_prism_coeffs,
        grad_distortion
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_backward(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &translation,
        const at::Tensor &rotation,
        const at::Tensor &grad_world_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_translation_grad,
        bool need_rotation_grad,
        bool need_focal_length_grad,
        bool need_principal_point_grad,
        bool need_radial_coeffs_grad,
        bool need_tangential_coeffs_grad,
        bool need_thin_prism_coeffs_grad,
        bool need_distortion_coeffs_grad
    )
{
    const auto &pts  = image_points;
    const auto &grad = grad_world_rays;
    check_matrix(pts, 2, "image_points");
    check_matrix(translation, 3, "translation");
    check_matrix(rotation, 4, "rotation");
    TORCH_CHECK(translation.size(0) == 1 && rotation.size(0) == 1, "static pose requires exactly one control pose");
    check_matrix(grad, 6, "grad_world_rays");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(translation, "translation");
    check_cuda_float_contiguous(rotation, "rotation");
    check_cuda_float_contiguous(grad, "grad_world_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    TORCH_CHECK(
        scratch.numel() >= pts.size(0) * 9,
        "scratch too small for bivariate image_points_to_world_rays_static_pose backward"
    );
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts        = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_t          = need_translation_grad ? at::zeros_like(translation) : empty_scratch(translation.options());
    auto grad_r          = need_rotation_grad ? at::zeros_like(rotation) : empty_scratch(rotation.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad
    );
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
        current_stream(pts)
    );
    return {
        grad_pts,
        grad_t,
        grad_r,
        intrinsic_grads.focal_length,
        intrinsic_grads.principal_point,
        intrinsic_grads.radial_coeffs,
        intrinsic_grads.tangential_coeffs,
        intrinsic_grads.thin_prism_coeffs,
        grad_distortion
    };
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
    at::Tensor,
    at::Tensor
>
    image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_backward(
        const c10::intrusive_ptr<OpenCVPinholeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        int64_t shutter_type,
        const at::Tensor &grad_world_rays,
        const at::Tensor &scratch,
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
        bool need_distortion_coeffs_grad
    )
{
    const auto &pts  = image_points;
    const auto &grad = grad_world_rays;
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
    TORCH_CHECK(
        scratch.numel() >= pts.size(0) * 12,
        "scratch too small for bivariate image_points_to_world_rays_shutter_pose backward"
    );
    check_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts        = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t    = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t      = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r    = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r      = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intrinsic_grads = make_intrinsic_grad_outputs(
        projection,
        need_focal_length_grad,
        need_principal_point_grad,
        need_radial_coeffs_grad,
        need_tangential_coeffs_grad,
        need_thin_prism_coeffs_grad
    );
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
        current_stream(pts)
    );
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
        intrinsic_grads.thin_prism_coeffs,
        grad_distortion
    };
}

// =============================================================================
// FThetaProjection host surface and validators.
// =============================================================================

FThetaProjection::FThetaProjection(
    at::Tensor principal_point_,
    at::Tensor fw_poly_,
    at::Tensor bw_poly_,
    at::Tensor A_,
    std::array<int64_t, 2> resolution_,
    int64_t reference_polynomial_,
    int64_t fw_poly_degree_,
    int64_t bw_poly_degree_,
    int64_t newton_iterations_,
    double max_angle_,
    double min_2d_norm_
)
    : principal_point(std::move(principal_point_))
    , fw_poly(std::move(fw_poly_))
    , bw_poly(std::move(bw_poly_))
    , A(std::move(A_))
    , resolution(resolution_)
    , reference_polynomial(reference_polynomial_)
    , fw_poly_degree(fw_poly_degree_)
    , bw_poly_degree(bw_poly_degree_)
    , newton_iterations(newton_iterations_)
    , max_angle(max_angle_)
    , min_2d_norm(min_2d_norm_)
{
}

at::Tensor FThetaProjection::compute_ainv() const
{
    // Compute on-device so kernel params can hold the result without a
    // host/device round trip.
    TORCH_CHECK(A.numel() == 4, "FThetaProjection: A must be a flat 4-element tensor");
    auto flat = A.reshape({4});
    auto a    = flat.select(0, 0);
    auto b    = flat.select(0, 1);
    auto c    = flat.select(0, 2);
    auto d    = flat.select(0, 3);
    auto det  = a * d - b * c;
    return at::stack({d / det, -b / det, -c / det, a / det}).contiguous();
}

FThetaProjection_KernelParameters FThetaProjection::to_kernel_params() const
{
    Ainv_cache = compute_ainv();
    FThetaProjection_KernelParameters params{};
    params.principal_point      = principal_point.const_data_ptr<float>();
    params.fw_poly              = fw_poly.const_data_ptr<float>();
    params.bw_poly              = bw_poly.const_data_ptr<float>();
    params.A                    = A.const_data_ptr<float>();
    params.Ainv                 = Ainv_cache.const_data_ptr<float>();
    params.width                = resolution[0];
    params.height               = resolution[1];
    params.reference_polynomial = reference_polynomial;
    params.fw_poly_degree       = fw_poly_degree;
    params.bw_poly_degree       = bw_poly_degree;
    params.newton_iterations    = newton_iterations;
    params.max_angle            = static_cast<float>(max_angle);
    params.min_2d_norm          = static_cast<float>(min_2d_norm);
    return params;
}

void check_ftheta_projection(const c10::intrusive_ptr<FThetaProjection> &projection)
{
    TORCH_CHECK(projection != nullptr, "projection must be FThetaProjection");
    check_component_shape(projection->principal_point, {2}, "principal_point");
    check_component_shape(projection->fw_poly, {kFThetaMaxPolynomialTerms}, "fw_poly");
    check_component_shape(projection->bw_poly, {kFThetaMaxPolynomialTerms}, "bw_poly");
    check_component_shape(projection->A, {4}, "A");
    TORCH_CHECK(projection->resolution[0] >= 0 && projection->resolution[1] >= 0, "resolution must be non-negative");
    TORCH_CHECK(
        projection->reference_polynomial == 0 || projection->reference_polynomial == 1,
        "reference_polynomial must be 0 (FORWARD) or 1 (BACKWARD)"
    );
    TORCH_CHECK(
        projection->fw_poly_degree >= 0 && projection->fw_poly_degree < kFThetaMaxPolynomialTerms,
        "fw_poly_degree must be in [0, kFThetaMaxPolynomialTerms)"
    );
    TORCH_CHECK(
        projection->bw_poly_degree >= 0 && projection->bw_poly_degree < kFThetaMaxPolynomialTerms,
        "bw_poly_degree must be in [0, kFThetaMaxPolynomialTerms)"
    );
    // Keep iterative caps centralized in camera_params.h.
    TORCH_CHECK(
        projection->newton_iterations >= 0 && projection->newton_iterations <= kFThetaMaxNewtonIterations,
        "newton_iterations must be in [0, ",
        kFThetaMaxNewtonIterations,
        "]"
    );
    // theta is bounded by pi; the range check also rejects NaN and infinities.
    TORCH_CHECK(
        projection->max_angle >= 0.0 && projection->max_angle <= std::numbers::pi_v<double>,
        "max_angle must be in [0, pi]"
    );
    // Zero can expose an on-axis 1/0 on device; +inf would clamp every ray to
    // the principal point, so positivity and finiteness are both required.
    TORCH_CHECK(projection->min_2d_norm > 0.0, "min_2d_norm must be strictly positive");
    TORCH_CHECK(std::isfinite(projection->min_2d_norm), "min_2d_norm must be finite");
    // Entry-value checks would force a CUDA->host sync; callers can opt into
    // them through gsplat_sensors.kernels.cameras.validate_camera_projection.
}

// =============================================================================
// OpenCVFisheyeProjection host surface and validators.
// =============================================================================

OpenCVFisheyeProjection::OpenCVFisheyeProjection(
    at::Tensor principal_point_,
    at::Tensor focal_length_,
    at::Tensor forward_poly_,
    at::Tensor approx_backward_factor_,
    std::array<int64_t, 2> resolution_,
    int64_t newton_iterations_,
    double max_angle_,
    double min_2d_norm_
)
    : principal_point(std::move(principal_point_))
    , focal_length(std::move(focal_length_))
    , forward_poly(std::move(forward_poly_))
    , approx_backward_factor(std::move(approx_backward_factor_))
    , resolution(resolution_)
    , newton_iterations(newton_iterations_)
    , max_angle(max_angle_)
    , min_2d_norm(min_2d_norm_)
{
}

OpenCVFisheyeProjection_KernelParameters OpenCVFisheyeProjection::to_kernel_params() const
{
    OpenCVFisheyeProjection_KernelParameters params{};
    params.principal_point        = principal_point.const_data_ptr<float>();
    params.focal_length           = focal_length.const_data_ptr<float>();
    params.forward_poly           = forward_poly.const_data_ptr<float>();
    params.approx_backward_factor = approx_backward_factor.const_data_ptr<float>();
    params.width                  = resolution[0];
    params.height                 = resolution[1];
    params.newton_iterations      = newton_iterations;
    params.max_angle              = static_cast<float>(max_angle);
    params.min_2d_norm            = static_cast<float>(min_2d_norm);
    return params;
}

c10::intrusive_ptr<OpenCVFisheyeProjection> OpenCVFisheyeProjection::transform(
    std::tuple<double, double> scale, std::tuple<double, double> offset, std::tuple<int64_t, int64_t> new_resolution
) const
{
    auto opts      = principal_point.options();
    double scale_u = std::get<0>(scale);
    double scale_v = std::get<1>(scale);

    auto scale_t             = at::tensor({scale_u, scale_v}, at::kDouble).to(opts);
    auto offset_t            = at::tensor({std::get<0>(offset), std::get<1>(offset)}, at::kDouble).to(opts);
    auto half_t              = at::tensor({0.5, 0.5}, at::kDouble).to(opts);
    auto new_principal_point = (principal_point + half_t) * scale_t - half_t - offset_t;

    auto new_focal_length = focal_length * scale_t;

    // approx_backward_factor is derived from intrinsics, not independent; after
    // scaling focal length, recompute it instead of scaling the cached value.
    int64_t new_w = std::get<0>(new_resolution);
    int64_t new_h = std::get<1>(new_resolution);
    auto res_t    = at::tensor({static_cast<double>(new_w), static_cast<double>(new_h)}, at::kDouble).to(opts);
    auto dist     = res_t / 2.0 / new_focal_length;
    auto new_ab   = (at::full({1}, max_angle, opts) / at::max(dist)).reshape({1});

    auto ptr = c10::make_intrusive<OpenCVFisheyeProjection>(
        new_principal_point,
        new_focal_length,
        forward_poly.clone(),
        new_ab,
        std::array<int64_t, 2>{new_w, new_h},
        newton_iterations,
        max_angle,
        min_2d_norm
    );
    gsplat_sensors::check_opencv_fisheye_projection(ptr);
    return ptr;
}

void check_opencv_fisheye_projection(const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection)
{
    TORCH_CHECK(projection != nullptr, "projection must be OpenCVFisheyeProjection");
    check_component_shape(projection->principal_point, {2}, "principal_point");
    check_component_shape(projection->focal_length, {2}, "focal_length");
    check_component_shape(projection->forward_poly, {kFisheyeForwardPolyTerms}, "forward_poly");
    check_component_shape(projection->approx_backward_factor, {1}, "approx_backward_factor");
    // Reject non-fp32 intrinsics before a stray fp64 tensor reaches
    // const_data_ptr<float>() in the kernel parameter path.
    TORCH_CHECK(projection->principal_point.scalar_type() == at::kFloat, "principal_point must be float32");
    TORCH_CHECK(projection->focal_length.scalar_type() == at::kFloat, "focal_length must be float32");
    TORCH_CHECK(projection->forward_poly.scalar_type() == at::kFloat, "forward_poly must be float32");
    TORCH_CHECK(
        projection->approx_backward_factor.scalar_type() == at::kFloat, "approx_backward_factor must be float32"
    );
    TORCH_CHECK(projection->resolution[0] >= 0 && projection->resolution[1] >= 0, "resolution must be non-negative");
    TORCH_CHECK(
        projection->newton_iterations >= 0 && projection->newton_iterations <= kFisheyeMaxNewtonIterations,
        "newton_iterations must be in [0, ",
        kFisheyeMaxNewtonIterations,
        "]"
    );
    // theta is bounded by pi; the range check also rejects NaN and infinities.
    TORCH_CHECK(
        projection->max_angle >= 0.0 && projection->max_angle <= std::numbers::pi_v<double>,
        "max_angle must be in [0, pi]"
    );
    // +inf would short-circuit every ray on device, so require finiteness too.
    TORCH_CHECK(projection->min_2d_norm > 0.0, "min_2d_norm must be strictly positive");
    TORCH_CHECK(std::isfinite(projection->min_2d_norm), "min_2d_norm must be finite");
    // Avoid CUDA->host sync here; value-level focal checks belong in opt-in
    // projection validation.
}

namespace
{
    void check_ftheta_projection_for_device(
        const c10::intrusive_ptr<FThetaProjection> &projection, const at::Device &device
    )
    {
        check_ftheta_projection(projection);
        check_cuda_float_contiguous(projection->principal_point, "principal_point");
        check_cuda_float_contiguous(projection->fw_poly, "fw_poly");
        check_cuda_float_contiguous(projection->bw_poly, "bw_poly");
        check_cuda_float_contiguous(projection->A, "A");
        TORCH_CHECK(projection->principal_point.device() == device, "principal_point device must match inputs");
        TORCH_CHECK(projection->fw_poly.device() == device, "fw_poly device must match inputs");
        TORCH_CHECK(projection->bw_poly.device() == device, "bw_poly device must match inputs");
        TORCH_CHECK(projection->A.device() == device, "A device must match inputs");
    }

    bool needs_ftheta_projection_grad(const c10::intrusive_ptr<FThetaProjection> &projection)
    {
        return projection->principal_point.requires_grad()
            || projection->fw_poly.requires_grad()
            || projection->bw_poly.requires_grad()
            || projection->A.requires_grad();
    }

    FThetaProjection_KernelParameters ftheta_kernel_params_with_resolution(
        const c10::intrusive_ptr<FThetaProjection> &projection, int64_t width, int64_t height
    )
    {
        auto params   = projection->to_kernel_params();
        params.width  = width;
        params.height = height;
        return params;
    }

    struct FThetaIntrinsicGradOutputs
    {
        at::Tensor principal_point;
        at::Tensor fw_poly;
        at::Tensor bw_poly;
        at::Tensor A;
        at::Tensor Ainv;
    };

    FThetaIntrinsicGradOutputs make_ftheta_intrinsic_grad_outputs(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        bool need_principal_point_grad,
        bool need_fw_poly_grad,
        bool need_bw_poly_grad,
        bool need_A_grad,
        bool need_Ainv_grad
    )
    {
        // Python folds grad_Ainv into grad_A with the inverse-matrix VJP.
        return {
            maybe_intrinsic_grad(need_principal_point_grad, {2}, projection->principal_point),
            maybe_intrinsic_grad(need_fw_poly_grad, {kFThetaMaxPolynomialTerms}, projection->fw_poly),
            maybe_intrinsic_grad(need_bw_poly_grad, {kFThetaMaxPolynomialTerms}, projection->bw_poly),
            maybe_intrinsic_grad(need_A_grad, {4}, projection->A),
            maybe_intrinsic_grad(need_Ainv_grad, {4}, projection->A),
        };
    }

    // ---------------------------------------------------------------------------
    // OpenCV-fisheye wrapper helpers. approx_backward_factor is derived, so it has
    // no gradient slot.
    // ---------------------------------------------------------------------------

    void check_opencv_fisheye_projection_for_device(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection, const at::Device &device
    )
    {
        check_opencv_fisheye_projection(projection);
        check_cuda_float_contiguous(projection->principal_point, "principal_point");
        check_cuda_float_contiguous(projection->focal_length, "focal_length");
        check_cuda_float_contiguous(projection->forward_poly, "forward_poly");
        check_cuda_float_contiguous(projection->approx_backward_factor, "approx_backward_factor");
        TORCH_CHECK(projection->principal_point.device() == device, "principal_point device must match inputs");
        TORCH_CHECK(projection->focal_length.device() == device, "focal_length device must match inputs");
        TORCH_CHECK(projection->forward_poly.device() == device, "forward_poly device must match inputs");
        TORCH_CHECK(
            projection->approx_backward_factor.device() == device, "approx_backward_factor device must match inputs"
        );
    }

    bool needs_opencv_fisheye_projection_grad(const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection)
    {
        return projection->principal_point.requires_grad()
            || projection->focal_length.requires_grad()
            || projection->forward_poly.requires_grad();
    }

    struct OpenCVFisheyeIntrinsicGradOutputs
    {
        at::Tensor principal_point;
        at::Tensor focal_length;
        at::Tensor forward_poly;
    };

    OpenCVFisheyeIntrinsicGradOutputs make_opencv_fisheye_intrinsic_grad_outputs(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        bool need_principal_point_grad,
        bool need_focal_length_grad,
        bool need_forward_poly_grad
    )
    {
        return {
            maybe_intrinsic_grad(need_principal_point_grad, {2}, projection->principal_point),
            maybe_intrinsic_grad(need_focal_length_grad, {2}, projection->focal_length),
            maybe_intrinsic_grad(need_forward_poly_grad, {kFisheyeForwardPolyTerms}, projection->forward_poly),
        };
    }

    OpenCVFisheyeProjection_KernelParameters opencv_fisheye_kernel_params_with_resolution(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection, int64_t width, int64_t height
    )
    {
        auto params   = projection->to_kernel_params();
        params.width  = width;
        params.height = height;
        return params;
    }
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> camera_rays_to_image_points_ftheta_no_external(
    const c10::intrusive_ptr<FThetaProjection> &projection,
    const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
    const at::Tensor &camera_rays
)
{
    check_optional_distortion(external_distortion);
    const auto &rays = camera_rays;
    check_matrix(rays, 3, "camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_ftheta_projection_for_device(projection, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto image_points       = at::empty({rays.size(0), 2}, rays.options());
    auto valid_flags        = at::empty({rays.size(0)}, rays.options().dtype(at::kBool));
    const bool save_scratch = rays.requires_grad() || needs_ftheta_projection_grad(projection);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::FTheta,
                                                 DistortionOpFamily::CameraRaysToImagePoints,
                                                 NoExternalDistortionPolicyTag
                                             >(rays.size(0), rays)
                                           : empty_scratch(rays.options());
    camera_rays_to_image_points_ftheta_no_external_forward_launch(
        rays.size(0),
        projection->to_kernel_params(),
        rays.data_ptr<float>(),
        image_points.data_ptr<float>(),
        valid_flags.data_ptr<bool>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(rays)
    );
    return {image_points, valid_flags, scratch};
}

std::tuple<at::Tensor, at::Tensor> image_points_to_camera_rays_ftheta_no_external(
    const c10::intrusive_ptr<FThetaProjection> &projection,
    const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
    const at::Tensor &image_points
)
{
    check_optional_distortion(external_distortion);
    const auto &pts = image_points;
    check_matrix(pts, 2, "image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_ftheta_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto camera_rays        = at::empty({pts.size(0), 3}, pts.options());
    const bool save_scratch = pts.requires_grad() || needs_ftheta_projection_grad(projection);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::FTheta,
                                                 DistortionOpFamily::ImagePointsToCameraRays,
                                                 NoExternalDistortionPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    image_points_to_camera_rays_ftheta_no_external_forward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        camera_rays.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts)
    );
    return {camera_rays, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> camera_rays_to_image_points_ftheta_bivariate_windshield(
    const c10::intrusive_ptr<FThetaProjection> &projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
    const at::Tensor &camera_rays
)
{
    const auto &rays = camera_rays;
    check_matrix(rays, 3, "camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_ftheta_projection_for_device(projection, rays.device());
    check_bivariate_windshield_for_device(external_distortion, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto image_points       = at::empty({rays.size(0), 2}, rays.options());
    auto valid_flags        = at::empty({rays.size(0)}, rays.options().dtype(at::kBool));
    const bool save_scratch = rays.requires_grad()
                           || needs_ftheta_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::FTheta,
                                                 DistortionOpFamily::CameraRaysToImagePoints,
                                                 BivariateWindshieldPolicyTag
                                             >(rays.size(0), rays)
                                           : empty_scratch(rays.options());
    camera_rays_to_image_points_ftheta_bivariate_windshield_forward_launch(
        rays.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        rays.data_ptr<float>(),
        image_points.data_ptr<float>(),
        valid_flags.data_ptr<bool>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(rays)
    );
    return {image_points, valid_flags, scratch};
}

std::tuple<at::Tensor, at::Tensor> image_points_to_camera_rays_ftheta_bivariate_windshield(
    const c10::intrusive_ptr<FThetaProjection> &projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
    const at::Tensor &image_points
)
{
    const auto &pts = image_points;
    check_matrix(pts, 2, "image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_ftheta_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto camera_rays        = at::empty({pts.size(0), 3}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || needs_ftheta_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::FTheta,
                                                 DistortionOpFamily::ImagePointsToCameraRays,
                                                 BivariateWindshieldPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    image_points_to_camera_rays_ftheta_bivariate_windshield_forward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        camera_rays.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts)
    );
    return {camera_rays, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_mean_pose_ftheta_no_external(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts = world_points;
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
    check_ftheta_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto image_points            = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags             = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps              = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t                  = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r                  = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch      = pts.requires_grad()
                                || start_translation.requires_grad()
                                || end_translation.requires_grad()
                                || start_rotation.requires_grad()
                                || end_rotation.requires_grad()
                                || needs_ftheta_projection_grad(projection);
    auto scratch                 = save_scratch ? scratch_shape_for_forward<
                                                      DistortionSensor::FTheta,
                                                      DistortionOpFamily::ProjectWorldPointsMeanPose,
                                                      NoExternalDistortionPolicyTag
                                                  >(pts.size(0), pts)
                                                : empty_scratch(pts.options());
    const auto mean_timestamp_us = static_cast<int64_t>(
        static_cast<double>(start_timestamp_us) + 0.5 * static_cast<double>(end_timestamp_us - start_timestamp_us)
    );
    project_world_points_mean_pose_ftheta_no_external_forward_launch(
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
        current_stream(pts)
    );
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_mean_pose_ftheta_bivariate_windshield(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us
    )
{
    const auto &pts = world_points;
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
    check_ftheta_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto image_points            = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags             = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps              = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t                  = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r                  = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch      = pts.requires_grad()
                                || start_translation.requires_grad()
                                || end_translation.requires_grad()
                                || start_rotation.requires_grad()
                                || end_rotation.requires_grad()
                                || needs_ftheta_projection_grad(projection)
                                || needs_external_distortion_grad(external_distortion);
    auto scratch                 = save_scratch ? scratch_shape_for_forward<
                                                      DistortionSensor::FTheta,
                                                      DistortionOpFamily::ProjectWorldPointsMeanPose,
                                                      BivariateWindshieldPolicyTag
                                                  >(pts.size(0), pts)
                                                : empty_scratch(pts.options());
    const auto mean_timestamp_us = static_cast<int64_t>(
        static_cast<double>(start_timestamp_us) + 0.5 * static_cast<double>(end_timestamp_us - start_timestamp_us)
    );
    project_world_points_mean_pose_ftheta_bivariate_windshield_forward_launch(
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
        current_stream(pts)
    );
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_static_pose_ftheta_no_external(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &translations,
        const at::Tensor &rotations,
        int64_t timestamp_us
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts   = image_points;
    const auto &trans = translations;
    const auto &rots  = rotations;
    check_matrix(pts, 2, "image_points");
    check_matrix(trans, 3, "translations");
    check_matrix(rots, 4, "rotations");
    TORCH_CHECK(trans.size(0) == 1 && rots.size(0) == 1, "static pose requires exactly one control pose");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(trans, "translations");
    check_cuda_float_contiguous(rots, "rotations");
    TORCH_CHECK(
        trans.device() == pts.device() && rots.device() == pts.device(), "pose tensors device must match image_points"
    );
    check_ftheta_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto world_rays         = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || trans.requires_grad()
                           || rots.requires_grad()
                           || needs_ftheta_projection_grad(projection);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::FTheta,
                                                 DistortionOpFamily::ImagePointsToWorldRaysStaticPose,
                                                 NoExternalDistortionPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    image_points_to_world_rays_static_pose_ftheta_no_external_forward_launch(
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
        current_stream(pts)
    );
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_static_pose_ftheta_bivariate_windshield(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &translations,
        const at::Tensor &rotations,
        int64_t timestamp_us
    )
{
    const auto &pts   = image_points;
    const auto &trans = translations;
    const auto &rots  = rotations;
    check_matrix(pts, 2, "image_points");
    check_matrix(trans, 3, "translations");
    check_matrix(rots, 4, "rotations");
    TORCH_CHECK(trans.size(0) == 1 && rots.size(0) == 1, "static pose requires exactly one control pose");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(trans, "translations");
    check_cuda_float_contiguous(rots, "rotations");
    TORCH_CHECK(
        trans.device() == pts.device() && rots.device() == pts.device(), "pose tensors device must match image_points"
    );
    check_ftheta_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto world_rays         = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || trans.requires_grad()
                           || rots.requires_grad()
                           || needs_ftheta_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::FTheta,
                                                 DistortionOpFamily::ImagePointsToWorldRaysStaticPose,
                                                 BivariateWindshieldPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_forward_launch(
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
        current_stream(pts)
    );
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_shutter_pose_ftheta_no_external(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t width,
        int64_t height,
        int64_t shutter_type,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us,
        int64_t max_iterations,
        double stop_mean_error_px,
        double stop_delta_mean_error_px,
        double initial_relative_time
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts = world_points;
    check_matrix(pts, 3, "world_points");
    check_component_vector(start_translation, 3, "start_translation");
    check_component_vector(end_translation, 3, "end_translation");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    TORCH_CHECK(width > 0 && height > 0, "resolution dimensions must be positive for shutter-pose ops");
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
    check_ftheta_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto image_points       = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags        = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || start_translation.requires_grad()
                           || end_translation.requires_grad()
                           || start_rotation.requires_grad()
                           || end_rotation.requires_grad()
                           || needs_ftheta_projection_grad(projection);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::FTheta,
                                                 DistortionOpFamily::ProjectWorldPointsShutterPose,
                                                 NoExternalDistortionPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    project_world_points_shutter_pose_ftheta_no_external_forward_launch(
        pts.size(0),
        ftheta_kernel_params_with_resolution(projection, width, height),
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
        current_stream(pts)
    );
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_shutter_pose_ftheta_bivariate_windshield(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t width,
        int64_t height,
        int64_t shutter_type,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us,
        int64_t max_iterations,
        double stop_mean_error_px,
        double stop_delta_mean_error_px,
        double initial_relative_time
    )
{
    const auto &pts = world_points;
    check_matrix(pts, 3, "world_points");
    check_component_vector(start_translation, 3, "start_translation");
    check_component_vector(end_translation, 3, "end_translation");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    TORCH_CHECK(width > 0 && height > 0, "resolution dimensions must be positive for shutter-pose ops");
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
    check_ftheta_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto image_points       = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags        = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || start_translation.requires_grad()
                           || end_translation.requires_grad()
                           || start_rotation.requires_grad()
                           || end_rotation.requires_grad()
                           || needs_ftheta_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::FTheta,
                                                 DistortionOpFamily::ProjectWorldPointsShutterPose,
                                                 BivariateWindshieldPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    project_world_points_shutter_pose_ftheta_bivariate_windshield_forward_launch(
        pts.size(0),
        ftheta_kernel_params_with_resolution(projection, width, height),
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
        current_stream(pts)
    );
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_shutter_pose_ftheta_no_external(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t width,
        int64_t height,
        int64_t shutter_type,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts = image_points;
    check_matrix(pts, 2, "image_points");
    check_component_vector(start_translation, 3, "start_translation");
    check_component_vector(end_translation, 3, "end_translation");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    TORCH_CHECK(width > 0 && height > 0, "resolution dimensions must be positive for shutter-pose ops");
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
    check_ftheta_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto world_rays         = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || start_translation.requires_grad()
                           || end_translation.requires_grad()
                           || start_rotation.requires_grad()
                           || end_rotation.requires_grad()
                           || needs_ftheta_projection_grad(projection);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::FTheta,
                                                 DistortionOpFamily::ImagePointsToWorldRaysShutterPose,
                                                 NoExternalDistortionPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    image_points_to_world_rays_shutter_pose_ftheta_no_external_forward_launch(
        pts.size(0),
        ftheta_kernel_params_with_resolution(projection, width, height),
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
        current_stream(pts)
    );
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t width,
        int64_t height,
        int64_t shutter_type,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us
    )
{
    const auto &pts = image_points;
    check_matrix(pts, 2, "image_points");
    check_component_vector(start_translation, 3, "start_translation");
    check_component_vector(end_translation, 3, "end_translation");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    TORCH_CHECK(width > 0 && height > 0, "resolution dimensions must be positive for shutter-pose ops");
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
    check_ftheta_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto world_rays         = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || start_translation.requires_grad()
                           || end_translation.requires_grad()
                           || start_rotation.requires_grad()
                           || end_rotation.requires_grad()
                           || needs_ftheta_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::FTheta,
                                                 DistortionOpFamily::ImagePointsToWorldRaysShutterPose,
                                                 BivariateWindshieldPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_forward_launch(
        pts.size(0),
        ftheta_kernel_params_with_resolution(projection, width, height),
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
        current_stream(pts)
    );
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

// =============================================================================
// FTheta backward host wrappers
//   Re-run the guards that the matching forward applied: shape and contiguity
//   of the grad/ray/scratch tensors, scratch.numel() against the per-op
//   element budget, device alignment, and the FThetaProjection invariants via
//   check_ftheta_projection_for_device. Shutter-pose variants additionally
//   re-check shutter_type. autograd hands us tensors the user can substitute,
//   so we cannot rely on forward having validated this exact call.
// =============================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    camera_rays_to_image_points_ftheta_no_external_backward(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &camera_rays,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
        bool need_camera_ray_grad,
        bool need_principal_point_grad,
        bool need_fw_poly_grad,
        bool need_bw_poly_grad,
        bool need_A_grad,
        bool need_Ainv_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &rays = camera_rays;
    const auto &grad = grad_image_points;
    check_matrix(rays, 3, "camera_rays");
    check_matrix(grad, 2, "grad_image_points");
    TORCH_CHECK(grad.size(0) == rays.size(0), "grad_image_points batch must match camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_cuda_float_contiguous(grad, "grad_image_points");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::FTheta,
        DistortionOpFamily::CameraRaysToImagePoints,
        NoExternalDistortionPolicyTag
    >(scratch, rays.size(0), "camera_rays_to_image_points_ftheta_no_external");
    check_ftheta_projection_for_device(projection, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto grad_rays = need_camera_ray_grad ? at::empty_like(rays) : empty_scratch(rays.options());
    auto intr      = make_ftheta_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_fw_poly_grad, need_bw_poly_grad, need_A_grad, need_Ainv_grad
    );
    camera_rays_to_image_points_ftheta_no_external_backward_launch(
        rays.size(0),
        projection->to_kernel_params(),
        rays.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_camera_ray_grad ? grad_rays.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.fw_poly, need_fw_poly_grad),
        maybe_data_ptr(intr.bw_poly, need_bw_poly_grad),
        maybe_data_ptr(intr.A, need_A_grad),
        maybe_data_ptr(intr.Ainv, need_Ainv_grad),
        scratch.data_ptr<float>(),
        current_stream(rays)
    );
    return {grad_rays, intr.principal_point, intr.fw_poly, intr.bw_poly, intr.A, intr.Ainv};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_camera_rays_ftheta_no_external_backward(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &grad_camera_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_principal_point_grad,
        bool need_fw_poly_grad,
        bool need_bw_poly_grad,
        bool need_A_grad,
        bool need_Ainv_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts  = image_points;
    const auto &grad = grad_camera_rays;
    check_matrix(pts, 2, "image_points");
    check_matrix(grad, 3, "grad_camera_rays");
    TORCH_CHECK(grad.size(0) == pts.size(0), "grad_camera_rays batch must match image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(grad, "grad_camera_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::FTheta,
        DistortionOpFamily::ImagePointsToCameraRays,
        NoExternalDistortionPolicyTag
    >(scratch, pts.size(0), "image_points_to_camera_rays_ftheta_no_external");
    check_ftheta_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto intr     = make_ftheta_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_fw_poly_grad, need_bw_poly_grad, need_A_grad, need_Ainv_grad
    );
    image_points_to_camera_rays_ftheta_no_external_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.fw_poly, need_fw_poly_grad),
        maybe_data_ptr(intr.bw_poly, need_bw_poly_grad),
        maybe_data_ptr(intr.A, need_A_grad),
        maybe_data_ptr(intr.Ainv, need_Ainv_grad),
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {grad_pts, intr.principal_point, intr.fw_poly, intr.bw_poly, intr.A, intr.Ainv};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    camera_rays_to_image_points_ftheta_bivariate_windshield_backward(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &camera_rays,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
        bool need_camera_ray_grad,
        bool need_principal_point_grad,
        bool need_fw_poly_grad,
        bool need_bw_poly_grad,
        bool need_A_grad,
        bool need_Ainv_grad,
        bool need_distortion_coeffs_grad
    )
{
    const auto &rays = camera_rays;
    const auto &grad = grad_image_points;
    check_matrix(rays, 3, "camera_rays");
    check_matrix(grad, 2, "grad_image_points");
    TORCH_CHECK(grad.size(0) == rays.size(0), "grad_image_points batch must match camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_cuda_float_contiguous(grad, "grad_image_points");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::FTheta,
        DistortionOpFamily::CameraRaysToImagePoints,
        BivariateWindshieldPolicyTag
    >(scratch, rays.size(0), "camera_rays_to_image_points_ftheta_bivariate_windshield");
    check_ftheta_projection_for_device(projection, rays.device());
    check_bivariate_windshield_for_device(external_distortion, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto grad_rays = need_camera_ray_grad ? at::empty_like(rays) : empty_scratch(rays.options());
    auto intr      = make_ftheta_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_fw_poly_grad, need_bw_poly_grad, need_A_grad, need_Ainv_grad
    );
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    camera_rays_to_image_points_ftheta_bivariate_windshield_backward_launch(
        rays.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        rays.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_camera_ray_grad ? grad_rays.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.fw_poly, need_fw_poly_grad),
        maybe_data_ptr(intr.bw_poly, need_bw_poly_grad),
        maybe_data_ptr(intr.A, need_A_grad),
        maybe_data_ptr(intr.Ainv, need_Ainv_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(rays)
    );
    return {grad_rays, intr.principal_point, intr.fw_poly, intr.bw_poly, intr.A, intr.Ainv, grad_distortion};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_camera_rays_ftheta_bivariate_windshield_backward(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &grad_camera_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_principal_point_grad,
        bool need_fw_poly_grad,
        bool need_bw_poly_grad,
        bool need_A_grad,
        bool need_Ainv_grad,
        bool need_distortion_coeffs_grad
    )
{
    const auto &pts  = image_points;
    const auto &grad = grad_camera_rays;
    check_matrix(pts, 2, "image_points");
    check_matrix(grad, 3, "grad_camera_rays");
    TORCH_CHECK(grad.size(0) == pts.size(0), "grad_camera_rays batch must match image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(grad, "grad_camera_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::FTheta,
        DistortionOpFamily::ImagePointsToCameraRays,
        BivariateWindshieldPolicyTag
    >(scratch, pts.size(0), "image_points_to_camera_rays_ftheta_bivariate_windshield");
    check_ftheta_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto intr     = make_ftheta_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_fw_poly_grad, need_bw_poly_grad, need_A_grad, need_Ainv_grad
    );
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    image_points_to_camera_rays_ftheta_bivariate_windshield_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.fw_poly, need_fw_poly_grad),
        maybe_data_ptr(intr.bw_poly, need_bw_poly_grad),
        maybe_data_ptr(intr.A, need_A_grad),
        maybe_data_ptr(intr.Ainv, need_Ainv_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {grad_pts, intr.principal_point, intr.fw_poly, intr.bw_poly, intr.A, intr.Ainv, grad_distortion};
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
    at::Tensor
>
    project_world_points_mean_pose_ftheta_no_external_backward(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
        bool need_world_point_grad,
        bool need_start_translation_grad,
        bool need_end_translation_grad,
        bool need_start_rotation_grad,
        bool need_end_rotation_grad,
        bool need_principal_point_grad,
        bool need_fw_poly_grad,
        bool need_bw_poly_grad,
        bool need_A_grad,
        bool need_Ainv_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts  = world_points;
    const auto &grad = grad_image_points;
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
    check_backward_scratch<
        DistortionSensor::FTheta,
        DistortionOpFamily::ProjectWorldPointsMeanPose,
        NoExternalDistortionPolicyTag
    >(scratch, pts.size(0), "project_world_points_mean_pose_ftheta_no_external");
    check_ftheta_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts     = need_world_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t   = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r   = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intr         = make_ftheta_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_fw_poly_grad, need_bw_poly_grad, need_A_grad, need_Ainv_grad
    );
    project_world_points_mean_pose_ftheta_no_external_backward_launch(
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
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.fw_poly, need_fw_poly_grad),
        maybe_data_ptr(intr.bw_poly, need_bw_poly_grad),
        maybe_data_ptr(intr.A, need_A_grad),
        maybe_data_ptr(intr.Ainv, need_Ainv_grad),
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intr.principal_point,
        intr.fw_poly,
        intr.bw_poly,
        intr.A,
        intr.Ainv
    };
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
    at::Tensor,
    at::Tensor
>
    project_world_points_mean_pose_ftheta_bivariate_windshield_backward(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
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
        bool need_distortion_coeffs_grad
    )
{
    const auto &pts  = world_points;
    const auto &grad = grad_image_points;
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
    check_backward_scratch<
        DistortionSensor::FTheta,
        DistortionOpFamily::ProjectWorldPointsMeanPose,
        BivariateWindshieldPolicyTag
    >(scratch, pts.size(0), "project_world_points_mean_pose_ftheta_bivariate_windshield");
    check_ftheta_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts     = need_world_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t   = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r   = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intr         = make_ftheta_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_fw_poly_grad, need_bw_poly_grad, need_A_grad, need_Ainv_grad
    );
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    project_world_points_mean_pose_ftheta_bivariate_windshield_backward_launch(
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
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.fw_poly, need_fw_poly_grad),
        maybe_data_ptr(intr.bw_poly, need_bw_poly_grad),
        maybe_data_ptr(intr.A, need_A_grad),
        maybe_data_ptr(intr.Ainv, need_Ainv_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intr.principal_point,
        intr.fw_poly,
        intr.bw_poly,
        intr.A,
        intr.Ainv,
        grad_distortion
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_static_pose_ftheta_no_external_backward(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &translation,
        const at::Tensor &rotation,
        const at::Tensor &grad_world_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_translation_grad,
        bool need_rotation_grad,
        bool need_principal_point_grad,
        bool need_fw_poly_grad,
        bool need_bw_poly_grad,
        bool need_A_grad,
        bool need_Ainv_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts  = image_points;
    const auto &grad = grad_world_rays;
    check_matrix(pts, 2, "image_points");
    check_matrix(translation, 3, "translation");
    check_matrix(rotation, 4, "rotation");
    TORCH_CHECK(translation.size(0) == 1 && rotation.size(0) == 1, "static pose requires exactly one control pose");
    check_matrix(grad, 6, "grad_world_rays");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(translation, "translation");
    check_cuda_float_contiguous(rotation, "rotation");
    check_cuda_float_contiguous(grad, "grad_world_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::FTheta,
        DistortionOpFamily::ImagePointsToWorldRaysStaticPose,
        NoExternalDistortionPolicyTag
    >(scratch, pts.size(0), "image_points_to_world_rays_static_pose_ftheta_no_external");
    check_ftheta_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_t   = need_translation_grad ? at::zeros_like(translation) : empty_scratch(translation.options());
    auto grad_r   = need_rotation_grad ? at::zeros_like(rotation) : empty_scratch(rotation.options());
    auto intr     = make_ftheta_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_fw_poly_grad, need_bw_poly_grad, need_A_grad, need_Ainv_grad
    );
    image_points_to_world_rays_static_pose_ftheta_no_external_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        translation.data_ptr<float>(),
        rotation.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_translation_grad ? grad_t.data_ptr<float>() : nullptr,
        need_rotation_grad ? grad_r.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.fw_poly, need_fw_poly_grad),
        maybe_data_ptr(intr.bw_poly, need_bw_poly_grad),
        maybe_data_ptr(intr.A, need_A_grad),
        maybe_data_ptr(intr.Ainv, need_Ainv_grad),
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {grad_pts, grad_t, grad_r, intr.principal_point, intr.fw_poly, intr.bw_poly, intr.A, intr.Ainv};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_backward(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &translation,
        const at::Tensor &rotation,
        const at::Tensor &grad_world_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_translation_grad,
        bool need_rotation_grad,
        bool need_principal_point_grad,
        bool need_fw_poly_grad,
        bool need_bw_poly_grad,
        bool need_A_grad,
        bool need_Ainv_grad,
        bool need_distortion_coeffs_grad
    )
{
    const auto &pts  = image_points;
    const auto &grad = grad_world_rays;
    check_matrix(pts, 2, "image_points");
    check_matrix(translation, 3, "translation");
    check_matrix(rotation, 4, "rotation");
    TORCH_CHECK(translation.size(0) == 1 && rotation.size(0) == 1, "static pose requires exactly one control pose");
    check_matrix(grad, 6, "grad_world_rays");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(translation, "translation");
    check_cuda_float_contiguous(rotation, "rotation");
    check_cuda_float_contiguous(grad, "grad_world_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::FTheta,
        DistortionOpFamily::ImagePointsToWorldRaysStaticPose,
        BivariateWindshieldPolicyTag
    >(scratch, pts.size(0), "image_points_to_world_rays_static_pose_ftheta_bivariate_windshield");
    check_ftheta_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_t   = need_translation_grad ? at::zeros_like(translation) : empty_scratch(translation.options());
    auto grad_r   = need_rotation_grad ? at::zeros_like(rotation) : empty_scratch(rotation.options());
    auto intr     = make_ftheta_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_fw_poly_grad, need_bw_poly_grad, need_A_grad, need_Ainv_grad
    );
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_backward_launch(
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
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.fw_poly, need_fw_poly_grad),
        maybe_data_ptr(intr.bw_poly, need_bw_poly_grad),
        maybe_data_ptr(intr.A, need_A_grad),
        maybe_data_ptr(intr.Ainv, need_Ainv_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {
        grad_pts, grad_t, grad_r, intr.principal_point, intr.fw_poly, intr.bw_poly, intr.A, intr.Ainv, grad_distortion
    };
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
    at::Tensor
>
    project_world_points_shutter_pose_ftheta_no_external_backward(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        int64_t shutter_type,
        int64_t max_iterations,
        double initial_relative_time,
        const at::Tensor &valid_flags,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
        bool need_world_point_grad,
        bool need_start_translation_grad,
        bool need_end_translation_grad,
        bool need_start_rotation_grad,
        bool need_end_rotation_grad,
        bool need_principal_point_grad,
        bool need_fw_poly_grad,
        bool need_bw_poly_grad,
        bool need_A_grad,
        bool need_Ainv_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts   = world_points;
    const auto &valid = valid_flags;
    const auto &grad  = grad_image_points;
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
    check_backward_scratch<
        DistortionSensor::FTheta,
        DistortionOpFamily::ProjectWorldPointsShutterPose,
        NoExternalDistortionPolicyTag
    >(scratch, pts.size(0), "project_world_points_shutter_pose_ftheta_no_external");
    check_ftheta_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts     = need_world_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t   = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r   = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intr         = make_ftheta_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_fw_poly_grad, need_bw_poly_grad, need_A_grad, need_Ainv_grad
    );
    project_world_points_shutter_pose_ftheta_no_external_backward_launch(
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
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.fw_poly, need_fw_poly_grad),
        maybe_data_ptr(intr.bw_poly, need_bw_poly_grad),
        maybe_data_ptr(intr.A, need_A_grad),
        maybe_data_ptr(intr.Ainv, need_Ainv_grad),
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intr.principal_point,
        intr.fw_poly,
        intr.bw_poly,
        intr.A,
        intr.Ainv
    };
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
    at::Tensor,
    at::Tensor
>
    project_world_points_shutter_pose_ftheta_bivariate_windshield_backward(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        int64_t shutter_type,
        int64_t max_iterations,
        double initial_relative_time,
        const at::Tensor &valid_flags,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
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
        bool need_distortion_coeffs_grad
    )
{
    const auto &pts   = world_points;
    const auto &valid = valid_flags;
    const auto &grad  = grad_image_points;
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
    check_backward_scratch<
        DistortionSensor::FTheta,
        DistortionOpFamily::ProjectWorldPointsShutterPose,
        BivariateWindshieldPolicyTag
    >(scratch, pts.size(0), "project_world_points_shutter_pose_ftheta_bivariate_windshield");
    check_ftheta_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts     = need_world_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t   = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r   = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intr         = make_ftheta_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_fw_poly_grad, need_bw_poly_grad, need_A_grad, need_Ainv_grad
    );
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    project_world_points_shutter_pose_ftheta_bivariate_windshield_backward_launch(
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
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.fw_poly, need_fw_poly_grad),
        maybe_data_ptr(intr.bw_poly, need_bw_poly_grad),
        maybe_data_ptr(intr.A, need_A_grad),
        maybe_data_ptr(intr.Ainv, need_Ainv_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intr.principal_point,
        intr.fw_poly,
        intr.bw_poly,
        intr.A,
        intr.Ainv,
        grad_distortion
    };
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
    at::Tensor
>
    image_points_to_world_rays_shutter_pose_ftheta_no_external_backward(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        int64_t shutter_type,
        const at::Tensor &grad_world_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_start_translation_grad,
        bool need_end_translation_grad,
        bool need_start_rotation_grad,
        bool need_end_rotation_grad,
        bool need_principal_point_grad,
        bool need_fw_poly_grad,
        bool need_bw_poly_grad,
        bool need_A_grad,
        bool need_Ainv_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts  = image_points;
    const auto &grad = grad_world_rays;
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
    check_backward_scratch<
        DistortionSensor::FTheta,
        DistortionOpFamily::ImagePointsToWorldRaysShutterPose,
        NoExternalDistortionPolicyTag
    >(scratch, pts.size(0), "image_points_to_world_rays_shutter_pose_ftheta_no_external");
    check_ftheta_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts     = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t   = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r   = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intr         = make_ftheta_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_fw_poly_grad, need_bw_poly_grad, need_A_grad, need_Ainv_grad
    );
    image_points_to_world_rays_shutter_pose_ftheta_no_external_backward_launch(
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
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.fw_poly, need_fw_poly_grad),
        maybe_data_ptr(intr.bw_poly, need_bw_poly_grad),
        maybe_data_ptr(intr.A, need_A_grad),
        maybe_data_ptr(intr.Ainv, need_Ainv_grad),
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intr.principal_point,
        intr.fw_poly,
        intr.bw_poly,
        intr.A,
        intr.Ainv
    };
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
    at::Tensor,
    at::Tensor
>
    image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_backward(
        const c10::intrusive_ptr<FThetaProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        int64_t shutter_type,
        const at::Tensor &grad_world_rays,
        const at::Tensor &scratch,
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
        bool need_distortion_coeffs_grad
    )
{
    const auto &pts  = image_points;
    const auto &grad = grad_world_rays;
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
    check_backward_scratch<
        DistortionSensor::FTheta,
        DistortionOpFamily::ImagePointsToWorldRaysShutterPose,
        BivariateWindshieldPolicyTag
    >(scratch, pts.size(0), "image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield");
    check_ftheta_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts     = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t   = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r   = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intr         = make_ftheta_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_fw_poly_grad, need_bw_poly_grad, need_A_grad, need_Ainv_grad
    );
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_backward_launch(
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
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.fw_poly, need_fw_poly_grad),
        maybe_data_ptr(intr.bw_poly, need_bw_poly_grad),
        maybe_data_ptr(intr.A, need_A_grad),
        maybe_data_ptr(intr.Ainv, need_Ainv_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intr.principal_point,
        intr.fw_poly,
        intr.bw_poly,
        intr.A,
        intr.Ainv,
        grad_distortion
    };
}

// ===========================================================================
// No-external D1/D2/D3/D5 wrappers use fixed scratch strides that backward
// validates before reading.
// ===========================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor> camera_rays_to_image_points_opencv_fisheye_no_external(
    const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
    const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
    const at::Tensor &camera_rays
)
{
    check_optional_distortion(external_distortion);
    const auto &rays = camera_rays;
    check_matrix(rays, 3, "camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_opencv_fisheye_projection_for_device(projection, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto image_points       = at::empty({rays.size(0), 2}, rays.options());
    auto valid_flags        = at::empty({rays.size(0)}, rays.options().dtype(at::kBool));
    const bool save_scratch = rays.requires_grad() || needs_opencv_fisheye_projection_grad(projection);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::OpenCVFisheye,
                                                 DistortionOpFamily::CameraRaysToImagePoints,
                                                 NoExternalDistortionPolicyTag
                                             >(rays.size(0), rays)
                                           : empty_scratch(rays.options());
    camera_rays_to_image_points_opencv_fisheye_no_external_forward_launch(
        rays.size(0),
        projection->to_kernel_params(),
        rays.data_ptr<float>(),
        image_points.data_ptr<float>(),
        valid_flags.data_ptr<bool>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(rays)
    );
    return {image_points, valid_flags, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    camera_rays_to_image_points_opencv_fisheye_no_external_backward(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &camera_rays,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
        bool need_camera_ray_grad,
        bool need_principal_point_grad,
        bool need_focal_length_grad,
        bool need_forward_poly_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &rays = camera_rays;
    const auto &grad = grad_image_points;
    check_matrix(rays, 3, "camera_rays");
    check_matrix(grad, 2, "grad_image_points");
    TORCH_CHECK(grad.size(0) == rays.size(0), "grad_image_points batch must match camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_cuda_float_contiguous(grad, "grad_image_points");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::OpenCVFisheye,
        DistortionOpFamily::CameraRaysToImagePoints,
        NoExternalDistortionPolicyTag
    >(scratch, rays.size(0), "camera_rays_to_image_points_opencv_fisheye_no_external");
    check_tensor_device(grad, rays.device(), "grad_image_points");
    check_tensor_device(scratch, rays.device(), "scratch");
    check_opencv_fisheye_projection_for_device(projection, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto grad_rays = need_camera_ray_grad ? at::empty_like(rays) : empty_scratch(rays.options());
    auto intr      = make_opencv_fisheye_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_focal_length_grad, need_forward_poly_grad
    );
    camera_rays_to_image_points_opencv_fisheye_no_external_backward_launch(
        rays.size(0),
        projection->to_kernel_params(),
        rays.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_camera_ray_grad ? grad_rays.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.focal_length, need_focal_length_grad),
        maybe_data_ptr(intr.forward_poly, need_forward_poly_grad),
        scratch.data_ptr<float>(),
        current_stream(rays)
    );
    return {grad_rays, intr.principal_point, intr.focal_length, intr.forward_poly};
}

std::tuple<at::Tensor, at::Tensor> image_points_to_camera_rays_opencv_fisheye_no_external(
    const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
    const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
    const at::Tensor &image_points
)
{
    check_optional_distortion(external_distortion);
    const auto &pts = image_points;
    check_matrix(pts, 2, "image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto camera_rays        = at::empty({pts.size(0), 3}, pts.options());
    const bool save_scratch = pts.requires_grad() || needs_opencv_fisheye_projection_grad(projection);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::OpenCVFisheye,
                                                 DistortionOpFamily::ImagePointsToCameraRays,
                                                 NoExternalDistortionPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    image_points_to_camera_rays_opencv_fisheye_no_external_forward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        camera_rays.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts)
    );
    return {camera_rays, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_camera_rays_opencv_fisheye_no_external_backward(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &grad_camera_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_principal_point_grad,
        bool need_focal_length_grad,
        bool need_forward_poly_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts  = image_points;
    const auto &grad = grad_camera_rays;
    check_matrix(pts, 2, "image_points");
    check_matrix(grad, 3, "grad_camera_rays");
    TORCH_CHECK(grad.size(0) == pts.size(0), "grad_camera_rays batch must match image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(grad, "grad_camera_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::OpenCVFisheye,
        DistortionOpFamily::ImagePointsToCameraRays,
        NoExternalDistortionPolicyTag
    >(scratch, pts.size(0), "image_points_to_camera_rays_opencv_fisheye_no_external");
    check_tensor_device(grad, pts.device(), "grad_camera_rays");
    check_tensor_device(scratch, pts.device(), "scratch");
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto intr     = make_opencv_fisheye_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_focal_length_grad, need_forward_poly_grad
    );
    image_points_to_camera_rays_opencv_fisheye_no_external_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.focal_length, need_focal_length_grad),
        maybe_data_ptr(intr.forward_poly, need_forward_poly_grad),
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {grad_pts, intr.principal_point, intr.focal_length, intr.forward_poly};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_mean_pose_opencv_fisheye_no_external(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts = world_points;
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
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto image_points            = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags             = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps              = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t                  = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r                  = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch      = pts.requires_grad()
                                || start_translation.requires_grad()
                                || end_translation.requires_grad()
                                || start_rotation.requires_grad()
                                || end_rotation.requires_grad()
                                || needs_opencv_fisheye_projection_grad(projection);
    auto scratch                 = save_scratch ? scratch_shape_for_forward<
                                                      DistortionSensor::OpenCVFisheye,
                                                      DistortionOpFamily::ProjectWorldPointsMeanPose,
                                                      NoExternalDistortionPolicyTag
                                                  >(pts.size(0), pts)
                                                : empty_scratch(pts.options());
    const auto mean_timestamp_us = static_cast<int64_t>(
        static_cast<double>(start_timestamp_us) + 0.5 * static_cast<double>(end_timestamp_us - start_timestamp_us)
    );
    project_world_points_mean_pose_opencv_fisheye_no_external_forward_launch(
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
        current_stream(pts)
    );
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_mean_pose_opencv_fisheye_no_external_backward(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
        bool need_world_point_grad,
        bool need_start_translation_grad,
        bool need_end_translation_grad,
        bool need_start_rotation_grad,
        bool need_end_rotation_grad,
        bool need_principal_point_grad,
        bool need_focal_length_grad,
        bool need_forward_poly_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts  = world_points;
    const auto &grad = grad_image_points;
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
    check_backward_scratch<
        DistortionSensor::OpenCVFisheye,
        DistortionOpFamily::ProjectWorldPointsMeanPose,
        NoExternalDistortionPolicyTag
    >(scratch, pts.size(0), "project_world_points_mean_pose_opencv_fisheye_no_external");
    check_tensor_device(start_rotation, pts.device(), "start_rotation");
    check_tensor_device(end_rotation, pts.device(), "end_rotation");
    check_tensor_device(grad, pts.device(), "grad_image_points");
    check_tensor_device(scratch, pts.device(), "scratch");
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts     = need_world_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t   = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r   = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intr         = make_opencv_fisheye_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_focal_length_grad, need_forward_poly_grad
    );
    project_world_points_mean_pose_opencv_fisheye_no_external_backward_launch(
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
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.focal_length, need_focal_length_grad),
        maybe_data_ptr(intr.forward_poly, need_forward_poly_grad),
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intr.principal_point,
        intr.focal_length,
        intr.forward_poly
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_static_pose_opencv_fisheye_no_external(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &translations,
        const at::Tensor &rotations,
        int64_t timestamp_us
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts   = image_points;
    const auto &trans = translations;
    const auto &rots  = rotations;
    check_matrix(pts, 2, "image_points");
    check_matrix(trans, 3, "translations");
    check_matrix(rots, 4, "rotations");
    TORCH_CHECK(trans.size(0) == 1 && rots.size(0) == 1, "static pose requires exactly one control pose");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(trans, "translations");
    check_cuda_float_contiguous(rots, "rotations");
    TORCH_CHECK(
        trans.device() == pts.device() && rots.device() == pts.device(), "pose tensors device must match image_points"
    );
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto world_rays         = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || trans.requires_grad()
                           || rots.requires_grad()
                           || needs_opencv_fisheye_projection_grad(projection);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::OpenCVFisheye,
                                                 DistortionOpFamily::ImagePointsToWorldRaysStaticPose,
                                                 NoExternalDistortionPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    image_points_to_world_rays_static_pose_opencv_fisheye_no_external_forward_launch(
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
        current_stream(pts)
    );
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_static_pose_opencv_fisheye_no_external_backward(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &translations,
        const at::Tensor &rotations,
        const at::Tensor &grad_world_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_translation_grad,
        bool need_rotation_grad,
        bool need_principal_point_grad,
        bool need_focal_length_grad,
        bool need_forward_poly_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts   = image_points;
    const auto &trans = translations;
    const auto &rots  = rotations;
    const auto &grad  = grad_world_rays;
    check_matrix(pts, 2, "image_points");
    check_matrix(trans, 3, "translations");
    check_matrix(rots, 4, "rotations");
    check_matrix(grad, 6, "grad_world_rays");
    TORCH_CHECK(trans.size(0) == 1 && rots.size(0) == 1, "static pose requires exactly one control pose");
    TORCH_CHECK(grad.size(0) == pts.size(0), "grad_world_rays batch must match image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(trans, "translations");
    check_cuda_float_contiguous(rots, "rotations");
    check_cuda_float_contiguous(grad, "grad_world_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    check_tensor_device(trans, pts.device(), "translations");
    check_tensor_device(rots, pts.device(), "rotations");
    check_tensor_device(grad, pts.device(), "grad_world_rays");
    check_tensor_device(scratch, pts.device(), "scratch");
    check_backward_scratch<
        DistortionSensor::OpenCVFisheye,
        DistortionOpFamily::ImagePointsToWorldRaysStaticPose,
        NoExternalDistortionPolicyTag
    >(scratch, pts.size(0), "image_points_to_world_rays_static_pose_opencv_fisheye_no_external");
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts   = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_trans = need_translation_grad ? at::zeros_like(trans) : empty_scratch(pts.options());
    auto grad_rot   = need_rotation_grad ? at::zeros_like(rots) : empty_scratch(pts.options());
    auto intr       = make_opencv_fisheye_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_focal_length_grad, need_forward_poly_grad
    );
    image_points_to_world_rays_static_pose_opencv_fisheye_no_external_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        trans.data_ptr<float>(),
        rots.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_translation_grad ? grad_trans.data_ptr<float>() : nullptr,
        need_rotation_grad ? grad_rot.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.focal_length, need_focal_length_grad),
        maybe_data_ptr(intr.forward_poly, need_forward_poly_grad),
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {grad_pts, grad_trans, grad_rot, intr.principal_point, intr.focal_length, intr.forward_poly};
}

// ===========================================================================
// Bivariate D1/D2/D3/D5 wrappers add coeff gradients and the larger inverse
// scratch needed by backproject-style ops.
// ===========================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor> camera_rays_to_image_points_opencv_fisheye_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
    const at::Tensor &camera_rays
)
{
    const auto &rays = camera_rays;
    check_matrix(rays, 3, "camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_opencv_fisheye_projection_for_device(projection, rays.device());
    check_bivariate_windshield_for_device(external_distortion, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto image_points       = at::empty({rays.size(0), 2}, rays.options());
    auto valid_flags        = at::empty({rays.size(0)}, rays.options().dtype(at::kBool));
    const bool save_scratch = rays.requires_grad()
                           || needs_opencv_fisheye_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::OpenCVFisheye,
                                                 DistortionOpFamily::CameraRaysToImagePoints,
                                                 BivariateWindshieldPolicyTag
                                             >(rays.size(0), rays)
                                           : empty_scratch(rays.options());
    camera_rays_to_image_points_opencv_fisheye_bivariate_windshield_forward_launch(
        rays.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        rays.data_ptr<float>(),
        image_points.data_ptr<float>(),
        valid_flags.data_ptr<bool>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(rays)
    );
    return {image_points, valid_flags, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    camera_rays_to_image_points_opencv_fisheye_bivariate_windshield_backward(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &camera_rays,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
        bool need_camera_ray_grad,
        bool need_principal_point_grad,
        bool need_focal_length_grad,
        bool need_forward_poly_grad,
        bool need_distortion_coeffs_grad
    )
{
    const auto &rays = camera_rays;
    const auto &grad = grad_image_points;
    check_matrix(rays, 3, "camera_rays");
    check_matrix(grad, 2, "grad_image_points");
    TORCH_CHECK(grad.size(0) == rays.size(0), "grad_image_points batch must match camera_rays");
    check_cuda_float_contiguous(rays, "camera_rays");
    check_cuda_float_contiguous(grad, "grad_image_points");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::OpenCVFisheye,
        DistortionOpFamily::CameraRaysToImagePoints,
        BivariateWindshieldPolicyTag
    >(scratch, rays.size(0), "camera_rays_to_image_points_opencv_fisheye_bivariate_windshield");
    check_tensor_device(grad, rays.device(), "grad_image_points");
    check_tensor_device(scratch, rays.device(), "scratch");
    check_opencv_fisheye_projection_for_device(projection, rays.device());
    check_bivariate_windshield_for_device(external_distortion, rays.device());
    auto guard = c10::cuda::CUDAGuard(rays.device());

    auto grad_rays = need_camera_ray_grad ? at::empty_like(rays) : empty_scratch(rays.options());
    auto intr      = make_opencv_fisheye_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_focal_length_grad, need_forward_poly_grad
    );
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    camera_rays_to_image_points_opencv_fisheye_bivariate_windshield_backward_launch(
        rays.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        rays.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_camera_ray_grad ? grad_rays.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.focal_length, need_focal_length_grad),
        maybe_data_ptr(intr.forward_poly, need_forward_poly_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(rays)
    );
    return {grad_rays, intr.principal_point, intr.focal_length, intr.forward_poly, grad_distortion};
}

std::tuple<at::Tensor, at::Tensor> image_points_to_camera_rays_opencv_fisheye_bivariate_windshield(
    const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
    const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
    const at::Tensor &image_points
)
{
    const auto &pts = image_points;
    check_matrix(pts, 2, "image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto camera_rays        = at::empty({pts.size(0), 3}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || needs_opencv_fisheye_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::OpenCVFisheye,
                                                 DistortionOpFamily::ImagePointsToCameraRays,
                                                 BivariateWindshieldPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    image_points_to_camera_rays_opencv_fisheye_bivariate_windshield_forward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        camera_rays.data_ptr<float>(),
        save_scratch ? scratch.data_ptr<float>() : nullptr,
        current_stream(pts)
    );
    return {camera_rays, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_camera_rays_opencv_fisheye_bivariate_windshield_backward(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &grad_camera_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_principal_point_grad,
        bool need_focal_length_grad,
        bool need_forward_poly_grad,
        bool need_distortion_coeffs_grad
    )
{
    const auto &pts  = image_points;
    const auto &grad = grad_camera_rays;
    check_matrix(pts, 2, "image_points");
    check_matrix(grad, 3, "grad_camera_rays");
    TORCH_CHECK(grad.size(0) == pts.size(0), "grad_camera_rays batch must match image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(grad, "grad_camera_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::OpenCVFisheye,
        DistortionOpFamily::ImagePointsToCameraRays,
        BivariateWindshieldPolicyTag
    >(scratch, pts.size(0), "image_points_to_camera_rays_opencv_fisheye_bivariate_windshield");
    check_tensor_device(grad, pts.device(), "grad_camera_rays");
    check_tensor_device(scratch, pts.device(), "scratch");
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto intr     = make_opencv_fisheye_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_focal_length_grad, need_forward_poly_grad
    );
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    image_points_to_camera_rays_opencv_fisheye_bivariate_windshield_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.focal_length, need_focal_length_grad),
        maybe_data_ptr(intr.forward_poly, need_forward_poly_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {grad_pts, intr.principal_point, intr.focal_length, intr.forward_poly, grad_distortion};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_mean_pose_opencv_fisheye_bivariate_windshield(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us
    )
{
    const auto &pts = world_points;
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
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto image_points            = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags             = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps              = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t                  = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r                  = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch      = pts.requires_grad()
                                || start_translation.requires_grad()
                                || end_translation.requires_grad()
                                || start_rotation.requires_grad()
                                || end_rotation.requires_grad()
                                || needs_opencv_fisheye_projection_grad(projection)
                                || needs_external_distortion_grad(external_distortion);
    auto scratch                 = save_scratch ? scratch_shape_for_forward<
                                                      DistortionSensor::OpenCVFisheye,
                                                      DistortionOpFamily::ProjectWorldPointsMeanPose,
                                                      BivariateWindshieldPolicyTag
                                                  >(pts.size(0), pts)
                                                : empty_scratch(pts.options());
    const auto mean_timestamp_us = static_cast<int64_t>(
        static_cast<double>(start_timestamp_us) + 0.5 * static_cast<double>(end_timestamp_us - start_timestamp_us)
    );
    project_world_points_mean_pose_opencv_fisheye_bivariate_windshield_forward_launch(
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
        current_stream(pts)
    );
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_mean_pose_opencv_fisheye_bivariate_windshield_backward(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
        bool need_world_point_grad,
        bool need_start_translation_grad,
        bool need_end_translation_grad,
        bool need_start_rotation_grad,
        bool need_end_rotation_grad,
        bool need_principal_point_grad,
        bool need_focal_length_grad,
        bool need_forward_poly_grad,
        bool need_distortion_coeffs_grad
    )
{
    const auto &pts  = world_points;
    const auto &grad = grad_image_points;
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
    check_backward_scratch<
        DistortionSensor::OpenCVFisheye,
        DistortionOpFamily::ProjectWorldPointsMeanPose,
        BivariateWindshieldPolicyTag
    >(scratch, pts.size(0), "project_world_points_mean_pose_opencv_fisheye_bivariate_windshield");
    check_tensor_device(start_rotation, pts.device(), "start_rotation");
    check_tensor_device(end_rotation, pts.device(), "end_rotation");
    check_tensor_device(grad, pts.device(), "grad_image_points");
    check_tensor_device(scratch, pts.device(), "scratch");
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts     = need_world_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t   = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r   = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intr         = make_opencv_fisheye_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_focal_length_grad, need_forward_poly_grad
    );
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    project_world_points_mean_pose_opencv_fisheye_bivariate_windshield_backward_launch(
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
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.focal_length, need_focal_length_grad),
        maybe_data_ptr(intr.forward_poly, need_forward_poly_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intr.principal_point,
        intr.focal_length,
        intr.forward_poly,
        grad_distortion
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &translations,
        const at::Tensor &rotations,
        int64_t timestamp_us
    )
{
    const auto &pts   = image_points;
    const auto &trans = translations;
    const auto &rots  = rotations;
    check_matrix(pts, 2, "image_points");
    check_matrix(trans, 3, "translations");
    check_matrix(rots, 4, "rotations");
    TORCH_CHECK(trans.size(0) == 1 && rots.size(0) == 1, "static pose requires exactly one control pose");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(trans, "translations");
    check_cuda_float_contiguous(rots, "rotations");
    TORCH_CHECK(
        trans.device() == pts.device() && rots.device() == pts.device(), "pose tensors device must match image_points"
    );
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto world_rays         = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || trans.requires_grad()
                           || rots.requires_grad()
                           || needs_opencv_fisheye_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::OpenCVFisheye,
                                                 DistortionOpFamily::ImagePointsToWorldRaysStaticPose,
                                                 BivariateWindshieldPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield_forward_launch(
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
        current_stream(pts)
    );
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield_backward(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &translations,
        const at::Tensor &rotations,
        const at::Tensor &grad_world_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_translation_grad,
        bool need_rotation_grad,
        bool need_principal_point_grad,
        bool need_focal_length_grad,
        bool need_forward_poly_grad,
        bool need_distortion_coeffs_grad
    )
{
    const auto &pts   = image_points;
    const auto &trans = translations;
    const auto &rots  = rotations;
    const auto &grad  = grad_world_rays;
    check_matrix(pts, 2, "image_points");
    check_matrix(trans, 3, "translations");
    check_matrix(rots, 4, "rotations");
    check_matrix(grad, 6, "grad_world_rays");
    TORCH_CHECK(trans.size(0) == 1 && rots.size(0) == 1, "static pose requires exactly one control pose");
    TORCH_CHECK(grad.size(0) == pts.size(0), "grad_world_rays batch must match image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(trans, "translations");
    check_cuda_float_contiguous(rots, "rotations");
    check_cuda_float_contiguous(grad, "grad_world_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::OpenCVFisheye,
        DistortionOpFamily::ImagePointsToWorldRaysStaticPose,
        BivariateWindshieldPolicyTag
    >(scratch, pts.size(0), "image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield");
    check_tensor_device(trans, pts.device(), "translations");
    check_tensor_device(rots, pts.device(), "rotations");
    check_tensor_device(grad, pts.device(), "grad_world_rays");
    check_tensor_device(scratch, pts.device(), "scratch");
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts   = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_trans = need_translation_grad ? at::zeros_like(trans) : empty_scratch(pts.options());
    auto grad_rot   = need_rotation_grad ? at::zeros_like(rots) : empty_scratch(pts.options());
    auto intr       = make_opencv_fisheye_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_focal_length_grad, need_forward_poly_grad
    );
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        trans.data_ptr<float>(),
        rots.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_translation_grad ? grad_trans.data_ptr<float>() : nullptr,
        need_rotation_grad ? grad_rot.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.focal_length, need_focal_length_grad),
        maybe_data_ptr(intr.forward_poly, need_forward_poly_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {
        grad_pts, grad_trans, grad_rot, intr.principal_point, intr.focal_length, intr.forward_poly, grad_distortion
    };
}

// =============================================================================
// Shutter wrappers keep D4/D6 separate because D4 has alpha convergence while
// D6 uses one pixel-derived alpha. Backward revalidates substituted tensors and
// scratch budgets before trusting saved state.
// =============================================================================

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_shutter_pose_opencv_fisheye_no_external(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t width,
        int64_t height,
        int64_t shutter_type,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us,
        int64_t max_iterations,
        double stop_mean_error_px,
        double stop_delta_mean_error_px,
        double initial_relative_time
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts = world_points;
    check_matrix(pts, 3, "world_points");
    check_component_vector(start_translation, 3, "start_translation");
    check_component_vector(end_translation, 3, "end_translation");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    TORCH_CHECK(width > 0 && height > 0, "resolution dimensions must be positive for shutter-pose ops");
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
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto image_points       = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags        = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || start_translation.requires_grad()
                           || end_translation.requires_grad()
                           || start_rotation.requires_grad()
                           || end_rotation.requires_grad()
                           || needs_opencv_fisheye_projection_grad(projection);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::OpenCVFisheye,
                                                 DistortionOpFamily::ProjectWorldPointsShutterPose,
                                                 NoExternalDistortionPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    project_world_points_shutter_pose_opencv_fisheye_no_external_forward_launch(
        pts.size(0),
        opencv_fisheye_kernel_params_with_resolution(projection, width, height),
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
        current_stream(pts)
    );
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_shutter_pose_opencv_fisheye_no_external_backward(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
        bool need_world_point_grad,
        bool need_start_translation_grad,
        bool need_end_translation_grad,
        bool need_start_rotation_grad,
        bool need_end_rotation_grad,
        bool need_principal_point_grad,
        bool need_focal_length_grad,
        bool need_forward_poly_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &grad = grad_image_points;
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_matrix(grad, 2, "grad_image_points");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    check_cuda_float_contiguous(grad, "grad_image_points");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::OpenCVFisheye,
        DistortionOpFamily::ProjectWorldPointsShutterPose,
        NoExternalDistortionPolicyTag
    >(scratch, grad.size(0), "project_world_points_shutter_pose_opencv_fisheye_no_external");
    check_tensor_device(start_rotation, grad.device(), "start_rotation");
    check_tensor_device(end_rotation, grad.device(), "end_rotation");
    check_tensor_device(scratch, grad.device(), "scratch");
    check_opencv_fisheye_projection_for_device(projection, grad.device());
    auto guard = c10::cuda::CUDAGuard(grad.device());

    auto grad_pts
        = need_world_point_grad ? at::empty({grad.size(0), 3}, grad.options()) : empty_scratch(grad.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, grad.options()) : empty_scratch(grad.options());
    auto grad_end_t   = need_end_translation_grad ? at::zeros({3}, grad.options()) : empty_scratch(grad.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, grad.options()) : empty_scratch(grad.options());
    auto grad_end_r   = need_end_rotation_grad ? at::zeros({4}, grad.options()) : empty_scratch(grad.options());
    auto intr         = make_opencv_fisheye_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_focal_length_grad, need_forward_poly_grad
    );
    project_world_points_shutter_pose_opencv_fisheye_no_external_backward_launch(
        grad.size(0),
        projection->to_kernel_params(),
        start_rotation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_world_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_start_translation_grad ? grad_start_t.data_ptr<float>() : nullptr,
        need_end_translation_grad ? grad_end_t.data_ptr<float>() : nullptr,
        need_start_rotation_grad ? grad_start_r.data_ptr<float>() : nullptr,
        need_end_rotation_grad ? grad_end_r.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.focal_length, need_focal_length_grad),
        maybe_data_ptr(intr.forward_poly, need_forward_poly_grad),
        scratch.data_ptr<float>(),
        current_stream(grad)
    );
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intr.principal_point,
        intr.focal_length,
        intr.forward_poly
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &world_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t width,
        int64_t height,
        int64_t shutter_type,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us,
        int64_t max_iterations,
        double stop_mean_error_px,
        double stop_delta_mean_error_px,
        double initial_relative_time
    )
{
    const auto &pts = world_points;
    check_matrix(pts, 3, "world_points");
    check_component_vector(start_translation, 3, "start_translation");
    check_component_vector(end_translation, 3, "end_translation");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    TORCH_CHECK(width > 0 && height > 0, "resolution dimensions must be positive for shutter-pose ops");
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
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto image_points       = at::empty({pts.size(0), 2}, pts.options());
    auto valid_flags        = at::empty({pts.size(0)}, pts.options().dtype(at::kBool));
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || start_translation.requires_grad()
                           || end_translation.requires_grad()
                           || start_rotation.requires_grad()
                           || end_rotation.requires_grad()
                           || needs_opencv_fisheye_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::OpenCVFisheye,
                                                 DistortionOpFamily::ProjectWorldPointsShutterPose,
                                                 BivariateWindshieldPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield_forward_launch(
        pts.size(0),
        opencv_fisheye_kernel_params_with_resolution(projection, width, height),
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
        current_stream(pts)
    );
    return {image_points, valid_flags, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield_backward(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        const at::Tensor &grad_image_points,
        const at::Tensor &scratch,
        bool need_world_point_grad,
        bool need_start_translation_grad,
        bool need_end_translation_grad,
        bool need_start_rotation_grad,
        bool need_end_rotation_grad,
        bool need_principal_point_grad,
        bool need_focal_length_grad,
        bool need_forward_poly_grad,
        bool need_distortion_coeffs_grad
    )
{
    const auto &grad = grad_image_points;
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_matrix(grad, 2, "grad_image_points");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    check_cuda_float_contiguous(grad, "grad_image_points");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::OpenCVFisheye,
        DistortionOpFamily::ProjectWorldPointsShutterPose,
        BivariateWindshieldPolicyTag
    >(scratch, grad.size(0), "project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield");
    check_tensor_device(start_rotation, grad.device(), "start_rotation");
    check_tensor_device(end_rotation, grad.device(), "end_rotation");
    check_tensor_device(scratch, grad.device(), "scratch");
    check_opencv_fisheye_projection_for_device(projection, grad.device());
    check_bivariate_windshield_for_device(external_distortion, grad.device());
    auto guard = c10::cuda::CUDAGuard(grad.device());

    auto grad_pts
        = need_world_point_grad ? at::empty({grad.size(0), 3}, grad.options()) : empty_scratch(grad.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, grad.options()) : empty_scratch(grad.options());
    auto grad_end_t   = need_end_translation_grad ? at::zeros({3}, grad.options()) : empty_scratch(grad.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, grad.options()) : empty_scratch(grad.options());
    auto grad_end_r   = need_end_rotation_grad ? at::zeros({4}, grad.options()) : empty_scratch(grad.options());
    auto intr         = make_opencv_fisheye_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_focal_length_grad, need_forward_poly_grad
    );
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield_backward_launch(
        grad.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        start_rotation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_world_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_start_translation_grad ? grad_start_t.data_ptr<float>() : nullptr,
        need_end_translation_grad ? grad_end_t.data_ptr<float>() : nullptr,
        need_start_rotation_grad ? grad_start_r.data_ptr<float>() : nullptr,
        need_end_rotation_grad ? grad_end_r.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.focal_length, need_focal_length_grad),
        maybe_data_ptr(intr.forward_poly, need_forward_poly_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(grad)
    );
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intr.principal_point,
        intr.focal_length,
        intr.forward_poly,
        grad_distortion
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t width,
        int64_t height,
        int64_t shutter_type,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts = image_points;
    check_matrix(pts, 2, "image_points");
    check_component_vector(start_translation, 3, "start_translation");
    check_component_vector(end_translation, 3, "end_translation");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    TORCH_CHECK(width > 0 && height > 0, "resolution dimensions must be positive for shutter-pose ops");
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
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto world_rays         = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || start_translation.requires_grad()
                           || end_translation.requires_grad()
                           || start_rotation.requires_grad()
                           || end_rotation.requires_grad()
                           || needs_opencv_fisheye_projection_grad(projection);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::OpenCVFisheye,
                                                 DistortionOpFamily::ImagePointsToWorldRaysShutterPose,
                                                 NoExternalDistortionPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external_forward_launch(
        pts.size(0),
        opencv_fisheye_kernel_params_with_resolution(projection, width, height),
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
        current_stream(pts)
    );
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external_backward(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<NoExternalDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        const at::Tensor &grad_world_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_start_translation_grad,
        bool need_end_translation_grad,
        bool need_start_rotation_grad,
        bool need_end_rotation_grad,
        bool need_principal_point_grad,
        bool need_focal_length_grad,
        bool need_forward_poly_grad
    )
{
    check_optional_distortion(external_distortion);
    const auto &pts  = image_points;
    const auto &grad = grad_world_rays;
    check_matrix(pts, 2, "image_points");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_matrix(grad, 6, "grad_world_rays");
    TORCH_CHECK(grad.size(0) == pts.size(0), "grad_world_rays batch must match image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    check_cuda_float_contiguous(grad, "grad_world_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::OpenCVFisheye,
        DistortionOpFamily::ImagePointsToWorldRaysShutterPose,
        NoExternalDistortionPolicyTag
    >(scratch, pts.size(0), "image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external");
    check_tensor_device(start_rotation, pts.device(), "start_rotation");
    check_tensor_device(end_rotation, pts.device(), "end_rotation");
    check_tensor_device(grad, pts.device(), "grad_world_rays");
    check_tensor_device(scratch, pts.device(), "scratch");
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts     = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t   = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r   = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intr         = make_opencv_fisheye_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_focal_length_grad, need_forward_poly_grad
    );
    image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        pts.data_ptr<float>(),
        start_rotation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_start_translation_grad ? grad_start_t.data_ptr<float>() : nullptr,
        need_end_translation_grad ? grad_end_t.data_ptr<float>() : nullptr,
        need_start_rotation_grad ? grad_start_r.data_ptr<float>() : nullptr,
        need_end_rotation_grad ? grad_end_r.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.focal_length, need_focal_length_grad),
        maybe_data_ptr(intr.forward_poly, need_forward_poly_grad),
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intr.principal_point,
        intr.focal_length,
        intr.forward_poly
    };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &start_translation,
        const at::Tensor &start_rotation,
        const at::Tensor &end_translation,
        const at::Tensor &end_rotation,
        int64_t width,
        int64_t height,
        int64_t shutter_type,
        int64_t start_timestamp_us,
        int64_t end_timestamp_us
    )
{
    const auto &pts = image_points;
    check_matrix(pts, 2, "image_points");
    check_component_vector(start_translation, 3, "start_translation");
    check_component_vector(end_translation, 3, "end_translation");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    TORCH_CHECK(width > 0 && height > 0, "resolution dimensions must be positive for shutter-pose ops");
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
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto world_rays         = at::empty({pts.size(0), 6}, pts.options());
    auto timestamps         = at::empty({pts.size(0)}, pts.options().dtype(at::kLong));
    auto pose_t             = at::empty({pts.size(0), 3}, pts.options());
    auto pose_r             = at::empty({pts.size(0), 4}, pts.options());
    const bool save_scratch = pts.requires_grad()
                           || start_translation.requires_grad()
                           || end_translation.requires_grad()
                           || start_rotation.requires_grad()
                           || end_rotation.requires_grad()
                           || needs_opencv_fisheye_projection_grad(projection)
                           || needs_external_distortion_grad(external_distortion);
    auto scratch            = save_scratch ? scratch_shape_for_forward<
                                                 DistortionSensor::OpenCVFisheye,
                                                 DistortionOpFamily::ImagePointsToWorldRaysShutterPose,
                                                 BivariateWindshieldPolicyTag
                                             >(pts.size(0), pts)
                                           : empty_scratch(pts.options());
    image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield_forward_launch(
        pts.size(0),
        opencv_fisheye_kernel_params_with_resolution(projection, width, height),
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
        current_stream(pts)
    );
    return {world_rays, timestamps, pose_t, pose_r, scratch};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield_backward(
        const c10::intrusive_ptr<OpenCVFisheyeProjection> &projection,
        const c10::intrusive_ptr<BivariateWindshieldDistortion> &external_distortion,
        const at::Tensor &image_points,
        const at::Tensor &start_rotation,
        const at::Tensor &end_rotation,
        const at::Tensor &grad_world_rays,
        const at::Tensor &scratch,
        bool need_image_point_grad,
        bool need_start_translation_grad,
        bool need_end_translation_grad,
        bool need_start_rotation_grad,
        bool need_end_rotation_grad,
        bool need_principal_point_grad,
        bool need_focal_length_grad,
        bool need_forward_poly_grad,
        bool need_distortion_coeffs_grad
    )
{
    const auto &pts  = image_points;
    const auto &grad = grad_world_rays;
    check_matrix(pts, 2, "image_points");
    check_component_vector(start_rotation, 4, "start_rotation");
    check_component_vector(end_rotation, 4, "end_rotation");
    check_matrix(grad, 6, "grad_world_rays");
    TORCH_CHECK(grad.size(0) == pts.size(0), "grad_world_rays batch must match image_points");
    check_cuda_float_contiguous(pts, "image_points");
    check_cuda_float_contiguous(start_rotation, "start_rotation");
    check_cuda_float_contiguous(end_rotation, "end_rotation");
    check_cuda_float_contiguous(grad, "grad_world_rays");
    check_cuda_float_contiguous(scratch, "scratch");
    check_backward_scratch<
        DistortionSensor::OpenCVFisheye,
        DistortionOpFamily::ImagePointsToWorldRaysShutterPose,
        BivariateWindshieldPolicyTag
    >(scratch, pts.size(0), "image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield");
    check_tensor_device(start_rotation, pts.device(), "start_rotation");
    check_tensor_device(end_rotation, pts.device(), "end_rotation");
    check_tensor_device(grad, pts.device(), "grad_world_rays");
    check_tensor_device(scratch, pts.device(), "scratch");
    check_opencv_fisheye_projection_for_device(projection, pts.device());
    check_bivariate_windshield_for_device(external_distortion, pts.device());
    auto guard = c10::cuda::CUDAGuard(pts.device());

    auto grad_pts     = need_image_point_grad ? at::empty_like(pts) : empty_scratch(pts.options());
    auto grad_start_t = need_start_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_t   = need_end_translation_grad ? at::zeros({3}, pts.options()) : empty_scratch(pts.options());
    auto grad_start_r = need_start_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto grad_end_r   = need_end_rotation_grad ? at::zeros({4}, pts.options()) : empty_scratch(pts.options());
    auto intr         = make_opencv_fisheye_intrinsic_grad_outputs(
        projection, need_principal_point_grad, need_focal_length_grad, need_forward_poly_grad
    );
    auto grad_distortion = make_distortion_grad_output(external_distortion, need_distortion_coeffs_grad);
    image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield_backward_launch(
        pts.size(0),
        projection->to_kernel_params(),
        external_distortion->to_kernel_params(),
        pts.data_ptr<float>(),
        start_rotation.data_ptr<float>(),
        end_rotation.data_ptr<float>(),
        grad.data_ptr<float>(),
        need_image_point_grad ? grad_pts.data_ptr<float>() : nullptr,
        need_start_translation_grad ? grad_start_t.data_ptr<float>() : nullptr,
        need_end_translation_grad ? grad_end_t.data_ptr<float>() : nullptr,
        need_start_rotation_grad ? grad_start_r.data_ptr<float>() : nullptr,
        need_end_rotation_grad ? grad_end_r.data_ptr<float>() : nullptr,
        maybe_data_ptr(intr.principal_point, need_principal_point_grad),
        maybe_data_ptr(intr.focal_length, need_focal_length_grad),
        maybe_data_ptr(intr.forward_poly, need_forward_poly_grad),
        need_distortion_coeffs_grad ? grad_distortion.data_ptr<float>() : nullptr,
        scratch.data_ptr<float>(),
        current_stream(pts)
    );
    return {
        grad_pts,
        grad_start_t,
        grad_end_t,
        grad_start_r,
        grad_end_r,
        intr.principal_point,
        intr.focal_length,
        intr.forward_poly,
        grad_distortion
    };
}
} // namespace gsplat_sensors
