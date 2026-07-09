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

#include "Config.h"

#if GSPLAT_BUILD_3DGUT

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda/std/optional>

#include "Common.h"
#include "ExternalDistortion.cuh"
#include "Projection.h"
#include "Utils.cuh"
#include "Cameras.cuh"
#include "Lidars.cuh"
#include "Sensors.cuh"
#include "Dispatch.h"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename SensorModel, typename scalar_t>
__global__ void projection_ut_3dgs_fused_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,     // [B, N, 3]
    const scalar_t *__restrict__ quats,     // [B, N, 4]
    const scalar_t *__restrict__ scales,    // [B, N, 3]
    const scalar_t *__restrict__ opacities, // [B, N] optional
    const scalar_t *__restrict__ viewmats0, // [B, C, 4, 4]
    const scalar_t *__restrict__ viewmats1, // [B, C, 4, 4] optional for rolling shutter
    // Image width and height are redundant, they are in the sensor model parameters
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool global_z_order,
    // uncented transform
    const UnscentedTransformParameters ut_params,    
    // sensor model parameters
    __grid_constant__ const typename SensorModel::KernelParameters sensor_model_params,
    // outputs
    int32_t *__restrict__ radii,         // [B, C, N, 2]
    scalar_t *__restrict__ means2d,      // [B, C, N, 2]
    scalar_t *__restrict__ depths,       // [B, C, N]
    scalar_t *__restrict__ conics,       // [B, C, N, 3]
    scalar_t *__restrict__ compensations // [B, C, N] optional
) {
    // parallelize over B * C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N) {
        return;
    }
    const uint32_t bid = idx / (C * N);  // batch id
    const uint32_t cid = (idx / N) % C;  // camera id 
    const uint32_t gid = idx % N;        // gaussian id

    // shift pointers to the current gaussian
    const glm::fvec3 mean = glm::make_vec3(means + bid * N * 3 + gid * 3);
    const glm::fvec3 scale = glm::make_vec3(scales + bid * N * 3 + gid * 3);
    glm::fquat quat = glm::fquat{
        quats[bid * N * 4 + gid * 4 + 0],
        quats[bid * N * 4 + gid * 4 + 1],
        quats[bid * N * 4 + gid * 4 + 2],
        quats[bid * N * 4 + gid * 4 + 3]};  // w,x,y,z quaternion
    // A zero-length quaternion has no defined orientation — the Gaussian is
    // degenerate. Skip it to avoid NaN from glm::normalize dividing by zero.
    // Under --use_fast_math, 0/0 silently gives 0 instead of NaN, causing
    // glm::mat3_cast to produce an identity rotation (wrong, not NaN).
    float quat_norm2 = glm::dot(quat, quat);
    if (is_near_zero(quat_norm2)) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // A Gaussian with near-zero scale along any axis has a divergent precision
    // matrix (1/scale → inf). The 3DGUT rasterization kernel evaluates
    // Gaussians in 3D using the precision matrix, so such Gaussians must not
    // enter the intersection list.  Cull them here.
    if (is_near_zero(scale[0]) || is_near_zero(scale[1]) || is_near_zero(scale[2])) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // Create rolling shutter parameter
    auto rs_params = RollingShutterParameters(
        viewmats0 + bid * C * 16 + cid * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + bid * C * 16 + cid * 16
    );

    // transform Gaussian center to camera space
    // Interpolate to *center* shutter pose as single per-Gaussian camera pose
    const auto shutter_pose = interpolate_shutter_pose(0.5f, rs_params);
    const vec3 mean_c = glm::rotate(shutter_pose.q, mean) + shutter_pose.t;

    const float cull_depth = global_z_order ? mean_c.z : glm::length(mean_c);
    if (cull_depth < near_plane || cull_depth > far_plane) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // Make sure the rotation quaternion is normalized
    quat *= rsqrtf(quat_norm2);

    // projection using unscented transform
    ImageGaussianReturn image_gaussian_return = world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
        SensorModel(sensor_model_params, bid * C + cid), rs_params, ut_params, mean, scale, quat);

    auto [mean2d, covar2d, valid_ut] = image_gaussian_return;
    if (!valid_ut) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // The UT center covariance weight can be very negative (e.g. ≈ -96 with
    // default alpha=0.1).  This means the UT covariance estimate is not
    // guaranteed positive-semidefinite — a diagonal entry can be negative even
    // when the determinant is positive.  Cull these: the UT has failed to
    // produce a valid 2D covariance for this Gaussian/camera combination, and
    // sqrtf(negative diagonal) below would produce NaN.
    if (covar2d[0][0] < 0.f || covar2d[1][1] < 0.f) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2 covar2d_inv = glm::inverse(covar2d);

    float extend = GAUSSIAN_EXTEND;
    // Note: the optimizations from StopThePop (https://arxiv.org/pdf/2402.00525) give identical
    // results when the Gaussian's contribution is evaluated in 2D from the projected 2D Gaussian.
    // However, with UT, the Gaussian's contribution is evaluated in 3D by intersecting the ray
    // with the 3D Gaussian.  As a result, the 2D culling criteria uses an approximation of what the
    // projected Gaussian would look like.  Therefore, this 2D-based culling can eliminate Gaussians
    // which would have a contribution in the 3D space and produce slightly different results,
    // especially with very non-linear projections / distortions.  Using these optimizations is
    // therefore a compromise between performance and quality, which is still reasonable for most cases.
    if (opacities != nullptr) {
        float opacity = opacities[bid * N + gid];
        opacity *= compensation;
        if (opacity < ALPHA_THRESHOLD) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }
        // Compute opacity-aware bounding box.
        // https://arxiv.org/pdf/2402.00525 Section B.2
        extend = min(GAUSSIAN_EXTEND, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
    }

    // compute tight rectangular bounding box (non differentiable)
    // https://arxiv.org/pdf/2402.00525
    float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    float tmp = sqrtf(max(0.01f, b * b - det));
    float v1 = b + tmp; // larger eigenvalue
    float r1 = extend * sqrtf(v1);
    float radius_x = ceilf(min(extend * sqrtf(covar2d[0][0]), r1));
    float radius_y = ceilf(min(extend * sqrtf(covar2d[1][1]), r1));

    if (radius_x <= radius_clip && radius_y <= radius_clip) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    if constexpr (is_lidar<SensorModel>::value) {
        // LIDAR culling is performed inside world_gaussian_to_image_gaussian_unscented_transform_shutter_pose,
        // so no additional 2D image-region masking is needed here.
    }
    else {
        // mask out gaussians outside the image region
        if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
            mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }
    }

    // Depth for sorting: z-depth (global_z_order=true) or Euclidean distance (global_z_order=false)
    float depth = global_z_order ? mean_c.z : glm::length(mean_c);

    // write to outputs
    radii[idx * 2] = (int32_t)radius_x;
    radii[idx * 2 + 1] = (int32_t)radius_y;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = depth;
    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
}

void launch_projection_ut_3dgs_fused_kernel(
    // inputs
    const at::Tensor means,                   // [..., N, 3]
    const at::Tensor quats,                   // [..., N, 4]
    const at::Tensor scales,                  // [..., N, 3]
    const at::optional<at::Tensor> opacities, // [..., N] optional
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    const bool global_z_order,
    // uncented transform
    const at::optional<c10::intrusive_ptr<UnscentedTransformParameters>>& ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const at::optional<c10::intrusive_ptr<FThetaCameraDistortionParameters>> &ftheta_coeffs, // shared parameters for all cameras
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params, // external distortion parameters
    // outputs
    at::Tensor radii,                      // [..., C, N, 2]
    at::Tensor means2d,                    // [..., C, N, 2]
    at::Tensor depths,                     // [..., C, N]
    at::Tensor conics,                     // [..., C, N, 3]
    at::optional<at::Tensor> compensations // [..., C, N] optional
) {
    uint32_t N = means.size(-2);          // number of gaussians
    uint32_t B = means.numel() / (N * 3); // number of batches
    uint32_t C = Ks.size(-3);             // number of cameras

    int64_t n_elements = B * C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // Set-up the sensor model parameters.
    // We do the dispatch between the runtime configurations and the actual concrete sensor model
    // by storing them in a std::variant, then calling the specialized kernel on the concrete type.
    const auto external_distortion_kernel_params = [&]() -> extdist::ExternalDistortionModelKernelParamsVariant {
        if (external_distortion_params.has_value()) {
            return extdist::BivariateWindshieldModel::KernelParameters{*external_distortion_params.value()};
        }
        else {
            return extdist::EmptyExternalDistortionModel::KernelParameters{};
        }
    }();

    const auto sensor_model_params = [&]() -> SensorModelKernelParamsVariant {
        if (camera_model == CameraModelType::PINHOLE) {
            if (!radial_coeffs.has_value() && !tangential_coeffs.has_value() && !thin_prism_coeffs.has_value()) {
                return to_sensor_model_kernel_params(
                    get_camera_model_kernel_params<PerfectPinholeCameraModel>(
                        {image_width, image_height},
                        rs_type,
                        external_distortion_kernel_params,
                        Ks.const_data_ptr<float>()
                    )
                );
            }
            else {
                return to_sensor_model_kernel_params(
                    get_camera_model_kernel_params<OpenCVPinholeCameraModel>(
                        {image_width, image_height},
                        rs_type,
                        external_distortion_kernel_params,
                        Ks.const_data_ptr<float>(),
                        radial_coeffs.has_value() ? radial_coeffs.value().const_data_ptr<float>() : nullptr,
                        tangential_coeffs.has_value() ? tangential_coeffs.value().const_data_ptr<float>() : nullptr,
                        thin_prism_coeffs.has_value() ? thin_prism_coeffs.value().const_data_ptr<float>() : nullptr
                    )
                );
            }
        }
        else if (camera_model == CameraModelType::FISHEYE) {
            return to_sensor_model_kernel_params(
                get_camera_model_kernel_params<OpenCVFisheyeCameraModel>(
                    {image_width, image_height},
                    rs_type,
                    external_distortion_kernel_params,
                    Ks.const_data_ptr<float>(),
                    radial_coeffs.has_value() ? radial_coeffs.value().const_data_ptr<float>() : nullptr
                )
            );
        }
        else if (camera_model == CameraModelType::FTHETA) {
            TORCH_CHECK(ftheta_coeffs.has_value(), "ftheta coefficients must be given for ftheta camera model");
            return to_sensor_model_kernel_params(
                get_camera_model_kernel_params<FThetaCameraModel>(
                    {image_width, image_height},
                    rs_type,
                    external_distortion_kernel_params,
                    Ks.const_data_ptr<float>(),
                    *ftheta_coeffs.value()
                )
            );
        }
        else if (camera_model == CameraModelType::LIDAR) {
            TORCH_CHECK(lidar_coeffs.has_value(), "Lidar coefficients must be given for lidar camera model");
            return RowOffsetStructuredSpinningLidarModel::KernelParameters{*lidar_coeffs.value()};
        }
        else {
            TORCH_CHECK(false, "Invalid camera model: only pinhole, fisheye, ftheta, and lidar camera models are supported");
        }
    }();

    // Default unscented-transform parameters when the caller leaves them unset.
    const auto ut = ut_params.has_value()
                        ? ut_params.value()
                        : c10::make_intrusive<UnscentedTransformParameters>();

    auto launch_kernel = [&]<typename SensorModel>() {
        projection_ut_3dgs_fused_kernel<SensorModel, float>
            <<<grid,
            threads,
            shmem_size,
            at::cuda::getCurrentCUDAStream()>>>(
                B,
                C,
                N,
                means.const_data_ptr<float>(),
                quats.const_data_ptr<float>(),
                scales.const_data_ptr<float>(),
                opacities.has_value() ? opacities.value().const_data_ptr<float>()
                                      : nullptr,
                viewmats0.const_data_ptr<float>(),
                viewmats1.has_value() ? viewmats1.value().const_data_ptr<float>()
                                      : nullptr,
                image_width,
                image_height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                global_z_order,
                // uncented transform
                *ut,
                std::get<typename SensorModel::KernelParameters>(sensor_model_params),
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<float>(),
                depths.data_ptr<float>(),
                conics.data_ptr<float>(),
                compensations.has_value() ? compensations.value().data_ptr<float>()
                                          : nullptr
            );
    };
    const bool dispatched = dispatch::dispatch(
        dispatch::make_mapped_type_param<SensorModelFromKernelParamsMap>(sensor_model_params),
        std::move(launch_kernel)
    );
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
    }

} // namespace gsplat

#endif
