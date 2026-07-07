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

#pragma once

#include "projection_forward_impl.cuh"

#include <type_traits>

template<int BlockThreads>
__device__ __forceinline__ void reduce_projection_static_pose_grads(
    float3 d_translation,
    float4 d_rotation_xyzw,
    float *__restrict__ grad_translation,
    float *__restrict__ grad_rotation
)
{
    const float tx = block_sum<BlockThreads>(d_translation.x);
    const float ty = block_sum<BlockThreads>(d_translation.y);
    const float tz = block_sum<BlockThreads>(d_translation.z);
    const float rx = block_sum<BlockThreads>(d_rotation_xyzw.x);
    const float ry = block_sum<BlockThreads>(d_rotation_xyzw.y);
    const float rz = block_sum<BlockThreads>(d_rotation_xyzw.z);
    const float rw = block_sum<BlockThreads>(d_rotation_xyzw.w);

    if(threadIdx.x == 0)
    {
        if(grad_translation != nullptr)
        {
            atomicAdd(&grad_translation[0], tx);
            atomicAdd(&grad_translation[1], ty);
            atomicAdd(&grad_translation[2], tz);
        }
        if(grad_rotation != nullptr)
        {
            atomicAdd(&grad_rotation[0], rw);
            atomicAdd(&grad_rotation[1], rx);
            atomicAdd(&grad_rotation[2], ry);
            atomicAdd(&grad_rotation[3], rz);
        }
    }
}

template<int BlockThreads>
__device__ __forceinline__ void reduce_projection_dynamic_pose_grads(
    float3 d_start_translation,
    float3 d_end_translation,
    float4 d_start_rotation_xyzw,
    float4 d_end_rotation_xyzw,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation
)
{
    const float t0x = block_sum<BlockThreads>(d_start_translation.x);
    const float t0y = block_sum<BlockThreads>(d_start_translation.y);
    const float t0z = block_sum<BlockThreads>(d_start_translation.z);
    const float t1x = block_sum<BlockThreads>(d_end_translation.x);
    const float t1y = block_sum<BlockThreads>(d_end_translation.y);
    const float t1z = block_sum<BlockThreads>(d_end_translation.z);
    const float r0x = block_sum<BlockThreads>(d_start_rotation_xyzw.x);
    const float r0y = block_sum<BlockThreads>(d_start_rotation_xyzw.y);
    const float r0z = block_sum<BlockThreads>(d_start_rotation_xyzw.z);
    const float r0w = block_sum<BlockThreads>(d_start_rotation_xyzw.w);
    const float r1x = block_sum<BlockThreads>(d_end_rotation_xyzw.x);
    const float r1y = block_sum<BlockThreads>(d_end_rotation_xyzw.y);
    const float r1z = block_sum<BlockThreads>(d_end_rotation_xyzw.z);
    const float r1w = block_sum<BlockThreads>(d_end_rotation_xyzw.w);

    if(threadIdx.x == 0)
    {
        if(grad_start_translation != nullptr)
        {
            atomicAdd(&grad_start_translation[0], t0x);
            atomicAdd(&grad_start_translation[1], t0y);
            atomicAdd(&grad_start_translation[2], t0z);
        }
        if(grad_end_translation != nullptr)
        {
            atomicAdd(&grad_end_translation[0], t1x);
            atomicAdd(&grad_end_translation[1], t1y);
            atomicAdd(&grad_end_translation[2], t1z);
        }
        if(grad_start_rotation != nullptr)
        {
            atomicAdd(&grad_start_rotation[0], r0w);
            atomicAdd(&grad_start_rotation[1], r0x);
            atomicAdd(&grad_start_rotation[2], r0y);
            atomicAdd(&grad_start_rotation[3], r0z);
        }
        if(grad_end_rotation != nullptr)
        {
            atomicAdd(&grad_end_rotation[0], r1w);
            atomicAdd(&grad_end_rotation[1], r1x);
            atomicAdd(&grad_end_rotation[2], r1y);
            atomicAdd(&grad_end_rotation[3], r1z);
        }
    }
}

template<int BlockThreads, typename ProjectionPolicy, typename DistortionPolicy>
__device__ __forceinline__ void camera_rays_to_image_points_backward_impl(
    int64_t count,
    const typename ProjectionPolicy::KernelParameters &projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ camera_rays,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_camera_rays,
    const typename ProjectionPolicy::IntrinsicGradOutputs &intrinsic_outputs,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    validate_projection_block_size<BlockThreads>();
    static_assert(std::is_trivially_copyable_v<typename ProjectionPolicy::IntrinsicGradOutputs>);
    constexpr DistortionOpFamily kOp = DistortionOpFamily::CameraRaysToImagePoints;
    using Scratch = typename ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::Backward;
    ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::validate();

    const int64_t idx                              = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const typename ProjectionPolicy::Params params = ProjectionPolicy::load(projection);
    const typename DistortionPolicy::Params distortion_params
        = DistortionPolicy::load(distortion, Scratch::kIsUndistort);
    typename ProjectionPolicy::ParamGrads d_projection{};
    typename DistortionPolicy::ParamGrads d_distortion{};

    if(idx < count)
    {
        typename ProjectionPolicy::ProjectState state;
        ProjectionPolicy::template ScratchIO<kOp>::template load_backward<Scratch>(
            scratch, idx * Scratch::kScratchStride, state
        );
        const float3 ray           = read_vec3(camera_rays, idx);
        const float3 projected_ray = DistortionPolicy::apply_fwd(ray, distortion_params);
        const float2 d_image_point = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);
        float3 d_projected_ray     = make_float3(0.0f, 0.0f, 0.0f);
        ProjectionPolicy::project_bwd(projected_ray, params, state, d_image_point, d_projected_ray, d_projection);
        float3 d_ray = make_float3(0.0f, 0.0f, 0.0f);
        DistortionPolicy::apply_bwd(ray, distortion_params, d_projected_ray, d_ray, d_distortion);
        if(grad_camera_rays != nullptr)
        {
            write_vec3(grad_camera_rays, idx, d_ray);
        }
    }

    ProjectionPolicy::template reduce_intrinsic_grads<BlockThreads>(d_projection, intrinsic_outputs);
    DistortionPolicy::template reduce_param_grads<BlockThreads>(
        d_distortion, distortion, Scratch::kIsUndistort, grad_distortion_coeffs
    );
}

template<int BlockThreads, typename ProjectionPolicy, typename DistortionPolicy>
__device__ __forceinline__ void image_points_to_camera_rays_backward_impl(
    int64_t count,
    const typename ProjectionPolicy::KernelParameters &projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ grad_camera_rays,
    float *__restrict__ grad_image_points,
    const typename ProjectionPolicy::IntrinsicGradOutputs &intrinsic_outputs,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    validate_projection_block_size<BlockThreads>();
    static_assert(std::is_trivially_copyable_v<typename ProjectionPolicy::IntrinsicGradOutputs>);
    constexpr DistortionOpFamily kOp = DistortionOpFamily::ImagePointsToCameraRays;
    using Scratch = typename ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::Backward;
    ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::validate();

    const int64_t idx                              = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const typename ProjectionPolicy::Params params = ProjectionPolicy::load(projection);
    const typename DistortionPolicy::Params distortion_params
        = DistortionPolicy::load(distortion, Scratch::kIsUndistort);
    typename ProjectionPolicy::ParamGrads d_projection{};
    typename DistortionPolicy::ParamGrads d_distortion{};

    if(idx < count)
    {
        const int64_t off = idx * Scratch::kScratchStride;
        typename ProjectionPolicy::BackprojectState state;
        ProjectionPolicy::template ScratchIO<kOp>::template load_backward<Scratch>(scratch, off, state);
        const float2 image_point   = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        const float3 projected_ray = ProjectionPolicy::backproject_output(state);
        const float3 inverse_primal
            = DistortionPolicy::inverse_bwd_input(projected_ray, scratch, off + Scratch::kInverseStashOffset);
        float3 d_inverse = read_vec3(grad_camera_rays, idx);
        if constexpr(DistortionPolicy::kHasDistortion)
        {
            d_inverse = normalize3_bwd(inverse_primal, d_inverse);
        }
        float3 d_projected_ray = make_float3(0.0f, 0.0f, 0.0f);
        DistortionPolicy::apply_bwd(projected_ray, distortion_params, d_inverse, d_projected_ray, d_distortion);
        float2 d_image_point = make_float2(0.0f, 0.0f);
        ProjectionPolicy::backproject_bwd(image_point, params, state, d_projected_ray, d_image_point, d_projection);
        if(grad_image_points != nullptr)
        {
            grad_image_points[idx * 2 + 0] = d_image_point.x;
            grad_image_points[idx * 2 + 1] = d_image_point.y;
        }
    }

    ProjectionPolicy::template reduce_intrinsic_grads<BlockThreads>(d_projection, intrinsic_outputs);
    DistortionPolicy::template reduce_param_grads<BlockThreads>(
        d_distortion, distortion, Scratch::kIsUndistort, grad_distortion_coeffs
    );
}

template<int BlockThreads, typename ProjectionPolicy, typename DistortionPolicy>
__device__ __forceinline__ void project_world_points_mean_pose_backward_impl(
    int64_t count,
    const typename ProjectionPolicy::KernelParameters &projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ world_points,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_world_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    const typename ProjectionPolicy::IntrinsicGradOutputs &intrinsic_outputs,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    validate_projection_block_size<BlockThreads>();
    static_assert(std::is_trivially_copyable_v<typename ProjectionPolicy::Params>);
    static_assert(std::is_trivially_copyable_v<typename DistortionPolicy::Params>);
    static_assert(std::is_trivially_copyable_v<typename ProjectionPolicy::IntrinsicGradOutputs>);
    constexpr DistortionOpFamily kOp = DistortionOpFamily::ProjectWorldPointsMeanPose;
    using Scratch = typename ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::Backward;
    ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::validate();

    (void)world_points;
    const int64_t idx                              = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const typename ProjectionPolicy::Params params = ProjectionPolicy::load(projection);
    const typename DistortionPolicy::Params distortion_params
        = DistortionPolicy::load(distortion, Scratch::kIsUndistort);
    typename ProjectionPolicy::ParamGrads d_projection{};
    typename DistortionPolicy::ParamGrads d_distortion{};
    float4 d_start_rotation_xyzw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_end_rotation_xyzw   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_start_translation   = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_end_translation     = make_float3(0.0f, 0.0f, 0.0f);

    if(idx < count)
    {
        const int64_t off = idx * Scratch::kScratchStride;
        float3 p_rel;
        float3 camera_point;
        typename ProjectionPolicy::ProjectState state;
        ProjectionPolicy::template ScratchIO<kOp>::template load_backward<Scratch>(
            scratch, off, p_rel, camera_point, state
        );

        if(ProjectionPolicy::pose_project_backward_enabled(state))
        {
            const float2 d_image_point = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);
            float3 camera_ray          = camera_point;
            if constexpr(ProjectionPolicy::kNormalizePoseProjectInput)
            {
                camera_ray = normalize3(camera_point);
            }
            const float3 projected_ray = DistortionPolicy::apply_fwd(camera_ray, distortion_params);
            typename ProjectionPolicy::ProjectState replayed_state = state;
            ProjectionPolicy::prepare_pose_project_state_for_backward(replayed_state, projected_ray);
            float3 d_projected_ray = make_float3(0.0f, 0.0f, 0.0f);
            ProjectionPolicy::project_bwd(
                projected_ray, params, replayed_state, d_image_point, d_projected_ray, d_projection
            );
            float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
            DistortionPolicy::apply_bwd(camera_ray, distortion_params, d_projected_ray, d_camera_ray, d_distortion);
            float3 d_camera_point = d_camera_ray;
            if constexpr(ProjectionPolicy::kNormalizePoseProjectInput)
            {
                d_camera_point = normalize3_bwd(camera_point, d_camera_ray);
            }

            const float4 start_rotation_xyzw = read_quat_xyzw_from_wxyz(start_rotation, 0);
            const float4 end_rotation_xyzw   = read_quat_xyzw_from_wxyz(end_rotation, 0);
            float rx, ry, rz, rw;
            gsplat_geometry::quat_slerp_pair_fwd<float>(
                start_rotation_xyzw.x,
                start_rotation_xyzw.y,
                start_rotation_xyzw.z,
                start_rotation_xyzw.w,
                end_rotation_xyzw.x,
                end_rotation_xyzw.y,
                end_rotation_xyzw.z,
                end_rotation_xyzw.w,
                0.5f,
                &rx,
                &ry,
                &rz,
                &rw
            );
            const float4 midpoint_rotation_xyzw = make_float4(rx, ry, rz, rw);
            float4 d_midpoint_rotation_xyzw     = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel                      = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(
                midpoint_rotation_xyzw, p_rel, d_camera_point, d_midpoint_rotation_xyzw, d_p_rel
            );

            if(grad_world_points != nullptr)
            {
                write_vec3(grad_world_points, idx, d_p_rel);
            }
            const float3 d_midpoint_translation = scale3(d_p_rel, -1.0f);
            d_start_translation                 = scale3(d_midpoint_translation, 0.5f);
            d_end_translation                   = scale3(d_midpoint_translation, 0.5f);

            float gq0x, gq0y, gq0z, gq0w;
            float gq1x, gq1y, gq1z, gq1w;
            gsplat_geometry::quat_slerp_pair_bwd_no_time_grad<float>(
                start_rotation_xyzw.x,
                start_rotation_xyzw.y,
                start_rotation_xyzw.z,
                start_rotation_xyzw.w,
                end_rotation_xyzw.x,
                end_rotation_xyzw.y,
                end_rotation_xyzw.z,
                end_rotation_xyzw.w,
                0.5f,
                rx,
                ry,
                rz,
                rw,
                d_midpoint_rotation_xyzw.x,
                d_midpoint_rotation_xyzw.y,
                d_midpoint_rotation_xyzw.z,
                d_midpoint_rotation_xyzw.w,
                &gq0x,
                &gq0y,
                &gq0z,
                &gq0w,
                &gq1x,
                &gq1y,
                &gq1z,
                &gq1w
            );
            d_start_rotation_xyzw = make_float4(gq0x, gq0y, gq0z, gq0w);
            d_end_rotation_xyzw   = make_float4(gq1x, gq1y, gq1z, gq1w);
        }
        else if(grad_world_points != nullptr)
        {
            write_vec3(grad_world_points, idx, make_float3(0.0f, 0.0f, 0.0f));
        }
    }

    reduce_projection_dynamic_pose_grads<BlockThreads>(
        d_start_translation,
        d_end_translation,
        d_start_rotation_xyzw,
        d_end_rotation_xyzw,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation
    );
    ProjectionPolicy::template reduce_intrinsic_grads<BlockThreads>(d_projection, intrinsic_outputs);
    DistortionPolicy::template reduce_param_grads<BlockThreads>(
        d_distortion, distortion, Scratch::kIsUndistort, grad_distortion_coeffs
    );
}

template<int BlockThreads, typename ProjectionPolicy, typename DistortionPolicy>
__device__ __forceinline__ void image_points_to_world_rays_static_pose_backward_impl(
    int64_t count,
    const typename ProjectionPolicy::KernelParameters &projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ translation,
    const float *__restrict__ rotation,
    const float *__restrict__ grad_world_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_translation,
    float *__restrict__ grad_rotation,
    const typename ProjectionPolicy::IntrinsicGradOutputs &intrinsic_outputs,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    validate_projection_block_size<BlockThreads>();
    static_assert(std::is_trivially_copyable_v<typename ProjectionPolicy::IntrinsicGradOutputs>);
    constexpr DistortionOpFamily kOp = DistortionOpFamily::ImagePointsToWorldRaysStaticPose;
    using Scratch = typename ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::Backward;
    ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::validate();

    (void)translation;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const typename DistortionPolicy::Params distortion_params
        = DistortionPolicy::load(distortion, Scratch::kIsUndistort);
    typename ProjectionPolicy::ParamGrads d_projection{};
    typename DistortionPolicy::ParamGrads d_distortion{};
    float4 d_rotation_xyzw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_translation   = make_float3(0.0f, 0.0f, 0.0f);

    if(idx < count)
    {
        const int64_t off = idx * Scratch::kScratchStride;
        typename ProjectionPolicy::BackprojectState state;
        ProjectionPolicy::template ScratchIO<kOp>::template load_backward<Scratch>(scratch, off, state);
        const float3 d_origin
            = make_float3(grad_world_rays[idx * 6 + 0], grad_world_rays[idx * 6 + 1], grad_world_rays[idx * 6 + 2]);
        const float3 d_direction
            = make_float3(grad_world_rays[idx * 6 + 3], grad_world_rays[idx * 6 + 4], grad_world_rays[idx * 6 + 5]);
        d_translation = d_origin;

        const float3 projected_ray = ProjectionPolicy::backproject_output(state);
        const float3 inverse_primal
            = DistortionPolicy::inverse_bwd_input(projected_ray, scratch, off + Scratch::kInverseStashOffset);
        float3 camera_ray = inverse_primal;
        if constexpr(DistortionPolicy::kHasDistortion)
        {
            camera_ray = normalize3(inverse_primal);
        }
        const float4 rotation_xyzw = read_quat_xyzw_from_wxyz(rotation, 0);
        float3 d_camera_ray        = make_float3(0.0f, 0.0f, 0.0f);
        quat_rotate_bwd_xyzw_geom(rotation_xyzw, camera_ray, d_direction, d_rotation_xyzw, d_camera_ray);

        float3 d_inverse = d_camera_ray;
        if constexpr(DistortionPolicy::kHasDistortion)
        {
            d_inverse = normalize3_bwd(inverse_primal, d_camera_ray);
        }
        float3 d_projected_ray = make_float3(0.0f, 0.0f, 0.0f);
        DistortionPolicy::apply_bwd(projected_ray, distortion_params, d_inverse, d_projected_ray, d_distortion);
        const float2 image_point = make_float2(image_points[idx * 2 + 0], image_points[idx * 2 + 1]);
        float2 d_image_point     = make_float2(0.0f, 0.0f);
        ProjectionPolicy::backproject_bwd_from_kernel_parameters(
            projection, image_point, state, d_projected_ray, d_image_point, d_projection
        );
        if(grad_image_points != nullptr)
        {
            grad_image_points[idx * 2 + 0] = d_image_point.x;
            grad_image_points[idx * 2 + 1] = d_image_point.y;
        }
    }

    reduce_projection_static_pose_grads<BlockThreads>(d_translation, d_rotation_xyzw, grad_translation, grad_rotation);
    ProjectionPolicy::template reduce_intrinsic_grads<BlockThreads>(d_projection, intrinsic_outputs);
    DistortionPolicy::template reduce_param_grads<BlockThreads>(
        d_distortion, distortion, Scratch::kIsUndistort, grad_distortion_coeffs
    );
}

template<int BlockThreads, typename ProjectionPolicy, typename DistortionPolicy>
__device__ __forceinline__ void image_points_to_world_rays_shutter_pose_backward_impl(
    int64_t count,
    const typename ProjectionPolicy::KernelParameters &projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    const float *__restrict__ grad_world_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    const typename ProjectionPolicy::IntrinsicGradOutputs &intrinsic_outputs,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    validate_projection_block_size<BlockThreads>();
    static_assert(std::is_trivially_copyable_v<typename ProjectionPolicy::IntrinsicGradOutputs>);
    constexpr DistortionOpFamily kOp = DistortionOpFamily::ImagePointsToWorldRaysShutterPose;
    using Scratch = typename ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::Backward;
    ProjectionScratchContract<ProjectionPolicy, kOp, DistortionPolicy>::validate();

    const int64_t idx                              = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const typename ProjectionPolicy::Params params = ProjectionPolicy::load(projection);
    const typename DistortionPolicy::Params distortion_params
        = DistortionPolicy::load(distortion, Scratch::kIsUndistort);
    typename ProjectionPolicy::ParamGrads d_projection{};
    typename DistortionPolicy::ParamGrads d_distortion{};
    float4 d_start_rotation_xyzw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_end_rotation_xyzw   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_start_translation   = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_end_translation     = make_float3(0.0f, 0.0f, 0.0f);

    if(idx < count)
    {
        const int64_t off = idx * Scratch::kScratchStride;
        typename ProjectionPolicy::BackprojectState state;
        float alpha = 0.0f;
        ProjectionPolicy::template ScratchIO<kOp>::template load_backward<Scratch>(scratch, off, state, alpha);
        const float2 image_point
            = load_shutter_image_point<ProjectionPolicy, DistortionPolicy>(image_points, idx, projection.width);
        const float3 d_origin
            = make_float3(grad_world_rays[idx * 6 + 0], grad_world_rays[idx * 6 + 1], grad_world_rays[idx * 6 + 2]);
        const float3 d_direction
            = make_float3(grad_world_rays[idx * 6 + 3], grad_world_rays[idx * 6 + 4], grad_world_rays[idx * 6 + 5]);

        d_start_translation = scale3(d_origin, 1.0f - alpha);
        d_end_translation   = scale3(d_origin, alpha);

        const float4 start_rotation_xyzw = read_quat_xyzw_from_wxyz(start_rotation, 0);
        const float4 end_rotation_xyzw   = read_quat_xyzw_from_wxyz(end_rotation, 0);
        float rx, ry, rz, rw;
        gsplat_geometry::quat_slerp_pair_fwd<float>(
            start_rotation_xyzw.x,
            start_rotation_xyzw.y,
            start_rotation_xyzw.z,
            start_rotation_xyzw.w,
            end_rotation_xyzw.x,
            end_rotation_xyzw.y,
            end_rotation_xyzw.z,
            end_rotation_xyzw.w,
            alpha,
            &rx,
            &ry,
            &rz,
            &rw
        );
        const float4 pose_rotation_xyzw = make_float4(rx, ry, rz, rw);

        const float3 projected_ray = ProjectionPolicy::backproject_output(state);
        const float3 inverse_primal
            = DistortionPolicy::inverse_bwd_input(projected_ray, scratch, off + Scratch::kInverseStashOffset);
        float3 camera_ray = inverse_primal;
        if constexpr(DistortionPolicy::kHasDistortion)
        {
            camera_ray = normalize3(inverse_primal);
        }
        float4 d_pose_rotation_xyzw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float3 d_camera_ray         = make_float3(0.0f, 0.0f, 0.0f);
        quat_rotate_bwd_xyzw_geom(pose_rotation_xyzw, camera_ray, d_direction, d_pose_rotation_xyzw, d_camera_ray);

        float gq0x, gq0y, gq0z, gq0w;
        float gq1x, gq1y, gq1z, gq1w;
        gsplat_geometry::quat_slerp_pair_bwd_no_time_grad<float>(
            start_rotation_xyzw.x,
            start_rotation_xyzw.y,
            start_rotation_xyzw.z,
            start_rotation_xyzw.w,
            end_rotation_xyzw.x,
            end_rotation_xyzw.y,
            end_rotation_xyzw.z,
            end_rotation_xyzw.w,
            alpha,
            rx,
            ry,
            rz,
            rw,
            d_pose_rotation_xyzw.x,
            d_pose_rotation_xyzw.y,
            d_pose_rotation_xyzw.z,
            d_pose_rotation_xyzw.w,
            &gq0x,
            &gq0y,
            &gq0z,
            &gq0w,
            &gq1x,
            &gq1y,
            &gq1z,
            &gq1w
        );
        d_start_rotation_xyzw = make_float4(gq0x, gq0y, gq0z, gq0w);
        d_end_rotation_xyzw   = make_float4(gq1x, gq1y, gq1z, gq1w);

        float3 d_inverse = d_camera_ray;
        if constexpr(DistortionPolicy::kHasDistortion)
        {
            d_inverse = normalize3_bwd(inverse_primal, d_camera_ray);
        }
        float3 d_projected_ray = make_float3(0.0f, 0.0f, 0.0f);
        DistortionPolicy::apply_bwd(projected_ray, distortion_params, d_inverse, d_projected_ray, d_distortion);
        float2 d_image_point = make_float2(0.0f, 0.0f);
        ProjectionPolicy::backproject_bwd(image_point, params, state, d_projected_ray, d_image_point, d_projection);
        if(image_points != nullptr && grad_image_points != nullptr)
        {
            grad_image_points[idx * 2 + 0] = d_image_point.x;
            grad_image_points[idx * 2 + 1] = d_image_point.y;
        }
    }

    reduce_projection_dynamic_pose_grads<BlockThreads>(
        d_start_translation,
        d_end_translation,
        d_start_rotation_xyzw,
        d_end_rotation_xyzw,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation
    );
    ProjectionPolicy::template reduce_intrinsic_grads<BlockThreads>(d_projection, intrinsic_outputs);
    DistortionPolicy::template reduce_param_grads<BlockThreads>(
        d_distortion, distortion, Scratch::kIsUndistort, grad_distortion_coeffs
    );
}
