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

// FTheta forward kernels. Scratch layouts are an ABI with the hand-written
// backward kernels; update DistortionScratchTraits and backward readers
// together when slots move.

#include "camera_kernel.cuh"
#include "external_distortion_kernel.cuh"
#include "ftheta_kernel.cuh"
#include "projection_forward_impl.cuh"

#include <c10/cuda/CUDAException.h>

namespace
{
constexpr int kThreads = 256;

template<DistortionOpFamily Op, typename PolicyTag>
using FThetaForwardScratch
    = DistortionScratchTraits<DistortionSensor::FTheta, Op, DistortionDirection::Forward, PolicyTag>;

// Returns a 1-D grid large enough to cover `count` threads at kThreads/block.
dim3 grid_for_count(int64_t count)
{
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

// =============================================================================
// D1 -- camera_rays_to_image_points_ftheta_{policy}
// =============================================================================

template<typename DistortionPolicy>
__global__ void camera_rays_to_image_points_ftheta_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ camera_rays,
    float *__restrict__ image_points,
    bool *__restrict__ valid_flags,
    float *__restrict__ scratch
)
{
    camera_rays_to_image_points_forward_impl<kThreads, FThetaProjectionPolicy, DistortionPolicy>(
        count, projection, distortion, camera_rays, image_points, valid_flags, scratch
    );
}

// =============================================================================
// D2 -- image_points_to_camera_rays_ftheta_{policy}
// =============================================================================

template<typename DistortionPolicy>
__global__ void image_points_to_camera_rays_ftheta_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ image_points,
    float *__restrict__ camera_rays,
    float *__restrict__ scratch
)
{
    image_points_to_camera_rays_forward_impl<kThreads, FThetaProjectionPolicy, DistortionPolicy>(
        count, projection, distortion, image_points, camera_rays, scratch
    );
}

// =============================================================================
// D3 -- project_world_points_mean_pose_ftheta_{policy}
// Mean-pose: thread 0 broadcasts the LERP/SLERP midpoint pose, intrinsics, and
// distortion parameters; each thread then projects its world point.
// =============================================================================

template<typename DistortionPolicy>
__global__ void project_world_points_mean_pose_ftheta_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ world_points,
    const float *__restrict__ start_translation,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_translation,
    const float *__restrict__ end_rotation,
    int64_t mean_timestamp_us,
    float *__restrict__ image_points,
    bool *__restrict__ valid_flags,
    int64_t *__restrict__ timestamps_us,
    float *__restrict__ pose_translations,
    float *__restrict__ pose_rotations,
    float *__restrict__ scratch
)
{
    project_world_points_mean_pose_forward_impl<kThreads, FThetaProjectionPolicy, DistortionPolicy>(
        count,
        projection,
        distortion,
        world_points,
        start_translation,
        start_rotation,
        end_translation,
        end_rotation,
        mean_timestamp_us,
        image_points,
        valid_flags,
        timestamps_us,
        pose_translations,
        pose_rotations,
        scratch
    );
}

// =============================================================================
// D5 -- image_points_to_world_rays_static_pose_ftheta_{policy}
// =============================================================================

template<typename DistortionPolicy>
__global__ void image_points_to_world_rays_static_pose_ftheta_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ translations,
    const float *__restrict__ rotations,
    int64_t timestamp_us,
    float *__restrict__ world_rays,
    int64_t *__restrict__ timestamps_us,
    float *__restrict__ pose_translations,
    float *__restrict__ pose_rotations,
    float *__restrict__ scratch
)
{
    image_points_to_world_rays_static_pose_forward_impl<kThreads, FThetaProjectionPolicy, DistortionPolicy>(
        count,
        projection,
        distortion,
        image_points,
        translations,
        rotations,
        timestamp_us,
        world_rays,
        timestamps_us,
        pose_translations,
        pose_rotations,
        scratch
    );
}

// =============================================================================
// D4 -- project_world_points_shutter_pose_ftheta_{policy}
// Shutter-pose: fixed-point convergence loop reprojects each world point under
// successive LERP/SLERP poses until the relative-time/pixel error settles.
// Global-shutter mode terminates after a single iteration.
// =============================================================================

template<typename DistortionPolicy>
__global__ void project_world_points_shutter_pose_ftheta_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ world_points,
    const float *__restrict__ start_translation,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_translation,
    const float *__restrict__ end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float *__restrict__ image_points,
    bool *__restrict__ valid_flags,
    int64_t *__restrict__ timestamps_us,
    float *__restrict__ pose_translations,
    float *__restrict__ pose_rotations,
    float *__restrict__ scratch
)
{
    using Scratch
        = FThetaForwardScratch<DistortionOpFamily::ProjectWorldPointsShutterPose, typename DistortionPolicy::Tag>;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    FThetaParams params         = load_ftheta_params(projection);
    auto distortion_params      = DistortionPolicy::load(distortion, Scratch::kIsUndistort);
    float3 world_point          = read_vec3(world_points, idx);
    float3 trans0               = read_vec3(start_translation, 0);
    float3 trans1               = read_vec3(end_translation, 0);
    float4 rot0                 = read_quat_xyzw_from_wxyz(start_rotation, 0);
    float4 rot1                 = read_quat_xyzw_from_wxyz(end_rotation, 0);
    float relative_time         = initial_relative_time;
    float alpha                 = 0.0f;
    float2 previous_image_point = make_float2(0.0f, 0.0f);
    float3 pose_t               = trans0;
    float4 pose_r_xyzw          = rot0;
    float3 p_rel                = make_float3(0.0f, 0.0f, 0.0f);
    float3 cam_pt               = make_float3(0.0f, 0.0f, 0.0f);
    FThetaProjectState state;
    state.ray_norm         = make_float3(0.0f, 0.0f, 0.0f);
    state.theta            = 0.0f;
    state.r                = 0.0f;
    state.xy_norm          = 0.0f;
    state.behind_camera    = false;
    state.angle_clamped    = false;
    state.min2d_clamped    = false;
    float2 image_point_out = make_float2(0.0f, 0.0f);
    bool valid_out         = false;
    for(int64_t i = 0; i < max_iterations; ++i)
    {
        alpha  = relative_time;
        pose_t = lerp3(trans0, trans1, alpha);
        float qx, qy, qz, qw;
        gsplat_geometry::quat_slerp_pair_fwd<float>(
            rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, alpha, &qx, &qy, &qz, &qw
        );
        pose_r_xyzw = make_float4(qx, qy, qz, qw);
        p_rel       = sub3(world_point, pose_t);
        cam_pt      = quat_inverse_rotate_xyzw_geom(pose_r_xyzw, p_rel);
        if(cam_pt.z <= 0.0f)
        {
            valid_out           = false;
            state.behind_camera = true;
            break;
        }
        const float3 projected_ray = DistortionPolicy::apply_fwd(cam_pt, distortion_params);
        bool fwd_valid             = false;
        FThetaProjectState iter_state;
        const float2 img = ftheta_project_ray(projected_ray, params, iter_state, fwd_valid);
        state            = iter_state;
        image_point_out  = img;
        valid_out        = fwd_valid;
        if(static_cast<gsplat_sensors::ShutterType>(shutter_type) == gsplat_sensors::ShutterType::GLOBAL || !fwd_valid)
        {
            break;
        }
        float next_relative_time
            = compute_relative_frame_time_opencv(img, projection.width, projection.height, shutter_type);
        float2 delta_px   = make_float2(img.x - previous_image_point.x, img.y - previous_image_point.y);
        float pixel_error = sqrtf(delta_px.x * delta_px.x + delta_px.y * delta_px.y);
        if(i > 0 && pixel_error < stop_delta_mean_error_px)
        {
            break;
        }
        float time_delta              = fabsf(next_relative_time - relative_time);
        float approximate_pixel_error = time_delta * static_cast<float>(std::max(projection.width, projection.height));
        if(approximate_pixel_error < stop_mean_error_px)
        {
            break;
        }
        previous_image_point = img;
        relative_time        = next_relative_time;
    }
    image_points[idx * 2 + 0] = image_point_out.x;
    image_points[idx * 2 + 1] = image_point_out.y;
    valid_flags[idx]          = valid_out && ftheta_image_point_in_frame(image_point_out, params);
    if(timestamps_us != nullptr)
    {
        timestamps_us[idx] = timestamp_from_relative_time(relative_time, start_timestamp_us, end_timestamp_us);
    }
    if(pose_translations != nullptr)
    {
        write_vec3(pose_translations, idx, pose_t);
    }
    if(pose_rotations != nullptr)
    {
        write_quat_wxyz_from_xyzw(pose_rotations, idx, pose_r_xyzw);
    }
    if(scratch != nullptr)
    {
        int64_t off       = idx * Scratch::kScratchStride;
        scratch[off + 0]  = p_rel.x;
        scratch[off + 1]  = p_rel.y;
        scratch[off + 2]  = p_rel.z;
        scratch[off + 3]  = cam_pt.x;
        scratch[off + 4]  = cam_pt.y;
        scratch[off + 5]  = cam_pt.z;
        scratch[off + 6]  = state.theta;
        scratch[off + 7]  = state.r;
        scratch[off + 8]  = state.xy_norm;
        scratch[off + 9]  = ftheta_pack_flags(state.behind_camera, state.angle_clamped, state.min2d_clamped);
        scratch[off + 10] = alpha;
    }
}

// =============================================================================
// D6 -- image_points_to_world_rays_shutter_pose_ftheta_{policy}
// Image-point shutter: relative_time is derived once from the pixel's row/col,
// then a single LERP/SLERP pose is used to lift the backprojected ray.
// =============================================================================

template<typename DistortionPolicy>
__global__ void image_points_to_world_rays_shutter_pose_ftheta_forward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ start_translation,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_translation,
    const float *__restrict__ end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float *__restrict__ world_rays,
    int64_t *__restrict__ timestamps_us,
    float *__restrict__ pose_translations,
    float *__restrict__ pose_rotations,
    float *__restrict__ scratch
)
{
    image_points_to_world_rays_shutter_pose_forward_impl<kThreads, FThetaProjectionPolicy, DistortionPolicy>(
        count,
        projection,
        distortion,
        image_points,
        start_translation,
        start_rotation,
        end_translation,
        end_rotation,
        shutter_type,
        start_timestamp_us,
        end_timestamp_us,
        world_rays,
        timestamps_us,
        pose_translations,
        pose_rotations,
        scratch
    );
}
} // namespace

// =============================================================================
// Public launchers keep the generated binding surface stable while the kernels
// share policy-templated implementations.
// =============================================================================

void camera_rays_to_image_points_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *camera_rays,
    float *image_points,
    bool *valid_flags,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    camera_rays_to_image_points_ftheta_forward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count, projection, NoExternalDistortion_KernelParameters{}, camera_rays, image_points, valid_flags, scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_camera_rays_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *image_points,
    float *camera_rays,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_camera_rays_ftheta_forward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count, projection, NoExternalDistortion_KernelParameters{}, image_points, camera_rays, scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void camera_rays_to_image_points_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *camera_rays,
    float *image_points,
    bool *valid_flags,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    camera_rays_to_image_points_ftheta_forward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count, projection, distortion, camera_rays, image_points, valid_flags, scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_camera_rays_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    float *camera_rays,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_camera_rays_ftheta_forward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count, projection, distortion, image_points, camera_rays, scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_mean_pose_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *world_points,
    const float *start_translation,
    const float *start_rotation,
    const float *end_translation,
    const float *end_rotation,
    int64_t mean_timestamp_us,
    float *image_points,
    bool *valid_flags,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_mean_pose_ftheta_forward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            NoExternalDistortion_KernelParameters{},
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            mean_timestamp_us,
            image_points,
            valid_flags,
            timestamps_us,
            pose_translations,
            pose_rotations,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_mean_pose_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *world_points,
    const float *start_translation,
    const float *start_rotation,
    const float *end_translation,
    const float *end_rotation,
    int64_t mean_timestamp_us,
    float *image_points,
    bool *valid_flags,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_mean_pose_ftheta_forward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            mean_timestamp_us,
            image_points,
            valid_flags,
            timestamps_us,
            pose_translations,
            pose_rotations,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_static_pose_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *image_points,
    const float *translation,
    const float *rotation,
    int64_t timestamp_us,
    float *world_rays,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_static_pose_ftheta_forward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            NoExternalDistortion_KernelParameters{},
            image_points,
            translation,
            rotation,
            timestamp_us,
            world_rays,
            timestamps_us,
            pose_translations,
            pose_rotations,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    const float *translation,
    const float *rotation,
    int64_t timestamp_us,
    float *world_rays,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_static_pose_ftheta_forward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            image_points,
            translation,
            rotation,
            timestamp_us,
            world_rays,
            timestamps_us,
            pose_translations,
            pose_rotations,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_shutter_pose_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *world_points,
    const float *start_translation,
    const float *start_rotation,
    const float *end_translation,
    const float *end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float *image_points,
    bool *valid_flags,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_shutter_pose_ftheta_forward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            NoExternalDistortion_KernelParameters{},
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            shutter_type,
            start_timestamp_us,
            end_timestamp_us,
            max_iterations,
            stop_mean_error_px,
            stop_delta_mean_error_px,
            initial_relative_time,
            image_points,
            valid_flags,
            timestamps_us,
            pose_translations,
            pose_rotations,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_shutter_pose_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *world_points,
    const float *start_translation,
    const float *start_rotation,
    const float *end_translation,
    const float *end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    int64_t max_iterations,
    float stop_mean_error_px,
    float stop_delta_mean_error_px,
    float initial_relative_time,
    float *image_points,
    bool *valid_flags,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_shutter_pose_ftheta_forward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            shutter_type,
            start_timestamp_us,
            end_timestamp_us,
            max_iterations,
            stop_mean_error_px,
            stop_delta_mean_error_px,
            initial_relative_time,
            image_points,
            valid_flags,
            timestamps_us,
            pose_translations,
            pose_rotations,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_shutter_pose_ftheta_no_external_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *image_points,
    const float *start_translation,
    const float *start_rotation,
    const float *end_translation,
    const float *end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float *world_rays,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_shutter_pose_ftheta_forward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            NoExternalDistortion_KernelParameters{},
            image_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            shutter_type,
            start_timestamp_us,
            end_timestamp_us,
            world_rays,
            timestamps_us,
            pose_translations,
            pose_rotations,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_forward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    const float *start_translation,
    const float *start_rotation,
    const float *end_translation,
    const float *end_rotation,
    int64_t shutter_type,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us,
    float *world_rays,
    int64_t *timestamps_us,
    float *pose_translations,
    float *pose_rotations,
    float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_shutter_pose_ftheta_forward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            image_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            shutter_type,
            start_timestamp_us,
            end_timestamp_us,
            world_rays,
            timestamps_us,
            pose_translations,
            pose_rotations,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
