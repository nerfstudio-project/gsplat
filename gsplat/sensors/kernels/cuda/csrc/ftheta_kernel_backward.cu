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

// Backward CUDA kernels for FTheta camera models. Implements the VJPs for each
// forward kernel in ftheta_kernel.cu, chaining through the FTheta IFT adjoint
// helpers in ftheta_kernel.cuh and the shared pose/quaternion adapters in
// camera_kernel.cuh.

#include "camera_kernel.cuh"
#include "external_distortion_kernel.cuh"
#include "ftheta_kernel.cuh"
#include "projection_backward_impl.cuh"

#include <c10/cuda/CUDAException.h>

namespace
{
constexpr int kThreads = 256;

template<DistortionOpFamily Op, typename PolicyTag>
using FThetaBackwardScratch
    = DistortionScratchTraits<DistortionSensor::FTheta, Op, DistortionDirection::Backward, PolicyTag>;

dim3 grid_for_count(int64_t count)
{
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

// Block-reduces each FThetaParamGrads slot via block_sum then atomicAdds the
// result from thread 0. Called once per kernel after all per-ray gradient work.
__device__ __forceinline__ void reduce_ftheta_intrinsic_grads(
    const FThetaParamGrads &local,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv
)
{
    float pp0 = block_sum<kThreads>(local.pp[0]);
    float pp1 = block_sum<kThreads>(local.pp[1]);
    float fw0 = block_sum<kThreads>(local.fw_poly[0]);
    float fw1 = block_sum<kThreads>(local.fw_poly[1]);
    float fw2 = block_sum<kThreads>(local.fw_poly[2]);
    float fw3 = block_sum<kThreads>(local.fw_poly[3]);
    float fw4 = block_sum<kThreads>(local.fw_poly[4]);
    float fw5 = block_sum<kThreads>(local.fw_poly[5]);
    float bw0 = block_sum<kThreads>(local.bw_poly[0]);
    float bw1 = block_sum<kThreads>(local.bw_poly[1]);
    float bw2 = block_sum<kThreads>(local.bw_poly[2]);
    float bw3 = block_sum<kThreads>(local.bw_poly[3]);
    float bw4 = block_sum<kThreads>(local.bw_poly[4]);
    float bw5 = block_sum<kThreads>(local.bw_poly[5]);
    float A0  = block_sum<kThreads>(local.A[0]);
    float A1  = block_sum<kThreads>(local.A[1]);
    float A2  = block_sum<kThreads>(local.A[2]);
    float A3  = block_sum<kThreads>(local.A[3]);
    float Ai0 = block_sum<kThreads>(local.Ainv[0]);
    float Ai1 = block_sum<kThreads>(local.Ainv[1]);
    float Ai2 = block_sum<kThreads>(local.Ainv[2]);
    float Ai3 = block_sum<kThreads>(local.Ainv[3]);

    if(threadIdx.x == 0)
    {
        if(grad_principal_point != nullptr)
        {
            atomicAdd(&grad_principal_point[0], pp0);
            atomicAdd(&grad_principal_point[1], pp1);
        }
        if(grad_fw_poly != nullptr)
        {
            atomicAdd(&grad_fw_poly[0], fw0);
            atomicAdd(&grad_fw_poly[1], fw1);
            atomicAdd(&grad_fw_poly[2], fw2);
            atomicAdd(&grad_fw_poly[3], fw3);
            atomicAdd(&grad_fw_poly[4], fw4);
            atomicAdd(&grad_fw_poly[5], fw5);
        }
        if(grad_bw_poly != nullptr)
        {
            atomicAdd(&grad_bw_poly[0], bw0);
            atomicAdd(&grad_bw_poly[1], bw1);
            atomicAdd(&grad_bw_poly[2], bw2);
            atomicAdd(&grad_bw_poly[3], bw3);
            atomicAdd(&grad_bw_poly[4], bw4);
            atomicAdd(&grad_bw_poly[5], bw5);
        }
        if(grad_A != nullptr)
        {
            atomicAdd(&grad_A[0], A0);
            atomicAdd(&grad_A[1], A1);
            atomicAdd(&grad_A[2], A2);
            atomicAdd(&grad_A[3], A3);
        }
        if(grad_Ainv != nullptr)
        {
            atomicAdd(&grad_Ainv[0], Ai0);
            atomicAdd(&grad_Ainv[1], Ai1);
            atomicAdd(&grad_Ainv[2], Ai2);
            atomicAdd(&grad_Ainv[3], Ai3);
        }
    }
}

// =============================================================================
// camera_rays_to_image_points backward
// =============================================================================

template<typename DistortionPolicy>
__global__ void camera_rays_to_image_points_ftheta_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ camera_rays,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_camera_rays,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    camera_rays_to_image_points_backward_impl<kThreads, FThetaProjectionPolicy, DistortionPolicy>(
        count,
        projection,
        distortion,
        camera_rays,
        grad_image_points,
        grad_camera_rays,
        FThetaProjectionPolicy::IntrinsicGradOutputs{
            grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv
        },
        grad_distortion_coeffs,
        scratch
    );
}

// =============================================================================
// image_points_to_camera_rays backward
// =============================================================================

template<typename DistortionPolicy>
__global__ void image_points_to_camera_rays_ftheta_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ grad_camera_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    image_points_to_camera_rays_backward_impl<kThreads, FThetaProjectionPolicy, DistortionPolicy>(
        count,
        projection,
        distortion,
        image_points,
        grad_camera_rays,
        grad_image_points,
        FThetaProjectionPolicy::IntrinsicGradOutputs{
            grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv
        },
        grad_distortion_coeffs,
        scratch
    );
}

// Unpacks the 10-slot mean/shutter forward scratch
// ([p_rel(3), cam_pt(3), theta, r, xy_norm, flags]) into FThetaProjectState
// plus p_rel and cam_pt. ray_norm is left zero; the caller recomputes it
// from the distortion policy's projected ray (apply_fwd of cam_pt).
__device__ __forceinline__ void ftheta_load_meanpose_10(
    const float *__restrict__ scratch, int64_t off, float3 &p_rel, float3 &cam_pt, FThetaProjectState &state
)
{
    p_rel         = make_float3(scratch[off + 0], scratch[off + 1], scratch[off + 2]);
    cam_pt        = make_float3(scratch[off + 3], scratch[off + 4], scratch[off + 5]);
    state.theta   = scratch[off + 6];
    state.r       = scratch[off + 7];
    state.xy_norm = scratch[off + 8];
    ftheta_unpack_flags(scratch[off + 9], state.behind_camera, state.angle_clamped, state.min2d_clamped);
    state.ray_norm = make_float3(0.0f, 0.0f, 0.0f);
}

// =============================================================================
// project_world_points_mean_pose backward
// =============================================================================

template<typename DistortionPolicy>
__global__ void project_world_points_mean_pose_ftheta_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
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
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    project_world_points_mean_pose_backward_impl<kThreads, FThetaProjectionPolicy, DistortionPolicy>(
        count,
        projection,
        distortion,
        world_points,
        start_rotation,
        end_rotation,
        grad_image_points,
        grad_world_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        FThetaProjectionPolicy::IntrinsicGradOutputs{
            grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv
        },
        grad_distortion_coeffs,
        scratch
    );
}

// =============================================================================
// image_points_to_world_rays_static_pose backward
// =============================================================================

template<typename DistortionPolicy>
__global__ void image_points_to_world_rays_static_pose_ftheta_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ translation,
    const float *__restrict__ rotation,
    const float *__restrict__ grad_world_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_translation,
    float *__restrict__ grad_rotation,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    image_points_to_world_rays_static_pose_backward_impl<kThreads, FThetaProjectionPolicy, DistortionPolicy>(
        count,
        projection,
        distortion,
        image_points,
        translation,
        rotation,
        grad_world_rays,
        grad_image_points,
        grad_translation,
        grad_rotation,
        FThetaProjectionPolicy::IntrinsicGradOutputs{
            grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv
        },
        grad_distortion_coeffs,
        scratch
    );
}

// =============================================================================
// project_world_points_shutter_pose backward
// =============================================================================

template<typename DistortionPolicy>
__global__ void project_world_points_shutter_pose_ftheta_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ world_points,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    float initial_relative_time,
    const bool *__restrict__ valid_flags,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_world_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    using Scratch
        = FThetaBackwardScratch<DistortionOpFamily::ProjectWorldPointsShutterPose, typename DistortionPolicy::Tag>;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    (void)max_iterations;
    (void)initial_relative_time;
    (void)shutter_type;
    FThetaParams params    = load_ftheta_params(projection);
    auto distortion_params = DistortionPolicy::load(distortion, Scratch::kIsUndistort);
    FThetaParamGrads d_params{};
    typename DistortionPolicy::ParamGrads d_distortion{};
    float4 d_rot0   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_rot1   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 d_trans0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_trans1 = make_float3(0.0f, 0.0f, 0.0f);

    if(idx < count)
    {
        bool valid = valid_flags[idx];
        if(valid)
        {
            int64_t off = idx * Scratch::kScratchStride;
            float3 p_rel;
            float3 cam_pt;
            FThetaProjectState state;
            ftheta_load_meanpose_10(scratch, off, p_rel, cam_pt, state);
            float alpha = scratch[off + 10];

            float2 d_img = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);

            float3 projected_ray        = DistortionPolicy::apply_fwd(cam_pt, distortion_params);
            FThetaProjectState replayed = state;
            replayed.ray_norm           = normalize3(projected_ray);
            float3 d_projected_ray      = make_float3(0.0f, 0.0f, 0.0f);
            ftheta_project_ray_bwd(projected_ray, params, replayed, d_img, d_projected_ray, d_params);
            float3 d_cam_pt = make_float3(0.0f, 0.0f, 0.0f);
            DistortionPolicy::apply_bwd(cam_pt, distortion_params, d_projected_ray, d_cam_pt, d_distortion);

            float4 rot0 = read_quat_xyzw_from_wxyz(start_rotation, 0);
            float4 rot1 = read_quat_xyzw_from_wxyz(end_rotation, 0);
            float rx, ry, rz, rw;
            gsplat_geometry::quat_slerp_pair_fwd<float>(
                rot0.x, rot0.y, rot0.z, rot0.w, rot1.x, rot1.y, rot1.z, rot1.w, alpha, &rx, &ry, &rz, &rw
            );
            float4 rot_alpha_xyzw = make_float4(rx, ry, rz, rw);
            float4 d_rot_alpha    = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel        = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(rot_alpha_xyzw, p_rel, d_cam_pt, d_rot_alpha, d_p_rel);

            if(grad_world_points != nullptr)
            {
                write_vec3(grad_world_points, idx, d_p_rel);
            }
            float3 d_pose_t  = scale3(d_p_rel, -1.0f);
            d_trans0.x      += (1.0f - alpha) * d_pose_t.x;
            d_trans0.y      += (1.0f - alpha) * d_pose_t.y;
            d_trans0.z      += (1.0f - alpha) * d_pose_t.z;
            d_trans1.x      += alpha * d_pose_t.x;
            d_trans1.y      += alpha * d_pose_t.y;
            d_trans1.z      += alpha * d_pose_t.z;

            float gq0x, gq0y, gq0z, gq0w, gq1x, gq1y, gq1z, gq1w;
            gsplat_geometry::quat_slerp_pair_bwd_no_time_grad<float>(
                rot0.x,
                rot0.y,
                rot0.z,
                rot0.w,
                rot1.x,
                rot1.y,
                rot1.z,
                rot1.w,
                alpha,
                rx,
                ry,
                rz,
                rw,
                d_rot_alpha.x,
                d_rot_alpha.y,
                d_rot_alpha.z,
                d_rot_alpha.w,
                &gq0x,
                &gq0y,
                &gq0z,
                &gq0w,
                &gq1x,
                &gq1y,
                &gq1z,
                &gq1w
            );
            float4 d_r0  = make_float4(gq0x, gq0y, gq0z, gq0w);
            float4 d_r1  = make_float4(gq1x, gq1y, gq1z, gq1w);
            d_rot0.x    += d_r0.x;
            d_rot0.y    += d_r0.y;
            d_rot0.z    += d_r0.z;
            d_rot0.w    += d_r0.w;
            d_rot1.x    += d_r1.x;
            d_rot1.y    += d_r1.y;
            d_rot1.z    += d_r1.z;
            d_rot1.w    += d_r1.w;
        }
        else
        {
            if(grad_world_points != nullptr)
            {
                write_vec3(grad_world_points, idx, make_float3(0.0f, 0.0f, 0.0f));
            }
        }
    }

    float t0x = block_sum<kThreads>(d_trans0.x);
    float t0y = block_sum<kThreads>(d_trans0.y);
    float t0z = block_sum<kThreads>(d_trans0.z);
    float t1x = block_sum<kThreads>(d_trans1.x);
    float t1y = block_sum<kThreads>(d_trans1.y);
    float t1z = block_sum<kThreads>(d_trans1.z);
    float r0x = block_sum<kThreads>(d_rot0.x);
    float r0y = block_sum<kThreads>(d_rot0.y);
    float r0z = block_sum<kThreads>(d_rot0.z);
    float r0w = block_sum<kThreads>(d_rot0.w);
    float r1x = block_sum<kThreads>(d_rot1.x);
    float r1y = block_sum<kThreads>(d_rot1.y);
    float r1z = block_sum<kThreads>(d_rot1.z);
    float r1w = block_sum<kThreads>(d_rot1.w);

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
    reduce_ftheta_intrinsic_grads(d_params, grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv);
    DistortionPolicy::template reduce_param_grads<kThreads>(
        d_distortion, distortion, Scratch::kIsUndistort, grad_distortion_coeffs
    );
}

// =============================================================================
// image_points_to_world_rays_shutter_pose backward
// =============================================================================

template<typename DistortionPolicy>
__global__ void image_points_to_world_rays_shutter_pose_ftheta_backward_kernel(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    int64_t shutter_type,
    const float *__restrict__ grad_world_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_fw_poly,
    float *__restrict__ grad_bw_poly,
    float *__restrict__ grad_A,
    float *__restrict__ grad_Ainv,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    (void)shutter_type;
    image_points_to_world_rays_shutter_pose_backward_impl<kThreads, FThetaProjectionPolicy, DistortionPolicy>(
        count,
        projection,
        distortion,
        image_points,
        start_rotation,
        end_rotation,
        grad_world_rays,
        grad_image_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        FThetaProjectionPolicy::IntrinsicGradOutputs{
            grad_principal_point, grad_fw_poly, grad_bw_poly, grad_A, grad_Ainv
        },
        grad_distortion_coeffs,
        scratch
    );
}
} // namespace

// =============================================================================
// Backward host launchers (12 total).
// =============================================================================

void camera_rays_to_image_points_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *camera_rays,
    const float *grad_image_points,
    float *grad_camera_rays,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    camera_rays_to_image_points_ftheta_backward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            NoExternalDistortion_KernelParameters{},
            camera_rays,
            grad_image_points,
            grad_camera_rays,
            grad_principal_point,
            grad_fw_poly,
            grad_bw_poly,
            grad_A,
            grad_Ainv,
            nullptr,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_camera_rays_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *image_points,
    const float *grad_camera_rays,
    float *grad_image_points,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_camera_rays_ftheta_backward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            NoExternalDistortion_KernelParameters{},
            image_points,
            grad_camera_rays,
            grad_image_points,
            grad_principal_point,
            grad_fw_poly,
            grad_bw_poly,
            grad_A,
            grad_Ainv,
            nullptr,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void camera_rays_to_image_points_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *camera_rays,
    const float *grad_image_points,
    float *grad_camera_rays,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    camera_rays_to_image_points_ftheta_backward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            camera_rays,
            grad_image_points,
            grad_camera_rays,
            grad_principal_point,
            grad_fw_poly,
            grad_bw_poly,
            grad_A,
            grad_Ainv,
            grad_distortion_coeffs,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_camera_rays_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    const float *grad_camera_rays,
    float *grad_image_points,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_camera_rays_ftheta_backward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            image_points,
            grad_camera_rays,
            grad_image_points,
            grad_principal_point,
            grad_fw_poly,
            grad_bw_poly,
            grad_A,
            grad_Ainv,
            grad_distortion_coeffs,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_mean_pose_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *world_points,
    const float *start_rotation,
    const float *end_rotation,
    const float *grad_image_points,
    float *grad_world_points,
    float *grad_start_translation,
    float *grad_end_translation,
    float *grad_start_rotation,
    float *grad_end_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_mean_pose_ftheta_backward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            NoExternalDistortion_KernelParameters{},
            world_points,
            start_rotation,
            end_rotation,
            grad_image_points,
            grad_world_points,
            grad_start_translation,
            grad_end_translation,
            grad_start_rotation,
            grad_end_rotation,
            grad_principal_point,
            grad_fw_poly,
            grad_bw_poly,
            grad_A,
            grad_Ainv,
            nullptr,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_mean_pose_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *world_points,
    const float *start_rotation,
    const float *end_rotation,
    const float *grad_image_points,
    float *grad_world_points,
    float *grad_start_translation,
    float *grad_end_translation,
    float *grad_start_rotation,
    float *grad_end_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_mean_pose_ftheta_backward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            world_points,
            start_rotation,
            end_rotation,
            grad_image_points,
            grad_world_points,
            grad_start_translation,
            grad_end_translation,
            grad_start_rotation,
            grad_end_rotation,
            grad_principal_point,
            grad_fw_poly,
            grad_bw_poly,
            grad_A,
            grad_Ainv,
            grad_distortion_coeffs,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_static_pose_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *image_points,
    const float *translation,
    const float *rotation,
    const float *grad_world_rays,
    float *grad_image_points,
    float *grad_translation,
    float *grad_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_static_pose_ftheta_backward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            NoExternalDistortion_KernelParameters{},
            image_points,
            translation,
            rotation,
            grad_world_rays,
            grad_image_points,
            grad_translation,
            grad_rotation,
            grad_principal_point,
            grad_fw_poly,
            grad_bw_poly,
            grad_A,
            grad_Ainv,
            nullptr,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    const float *translation,
    const float *rotation,
    const float *grad_world_rays,
    float *grad_image_points,
    float *grad_translation,
    float *grad_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_static_pose_ftheta_backward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            image_points,
            translation,
            rotation,
            grad_world_rays,
            grad_image_points,
            grad_translation,
            grad_rotation,
            grad_principal_point,
            grad_fw_poly,
            grad_bw_poly,
            grad_A,
            grad_Ainv,
            grad_distortion_coeffs,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_shutter_pose_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *world_points,
    const float *start_rotation,
    const float *end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    float initial_relative_time,
    const bool *valid_flags,
    const float *grad_image_points,
    float *grad_world_points,
    float *grad_start_translation,
    float *grad_end_translation,
    float *grad_start_rotation,
    float *grad_end_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_shutter_pose_ftheta_backward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            NoExternalDistortion_KernelParameters{},
            world_points,
            start_rotation,
            end_rotation,
            shutter_type,
            max_iterations,
            initial_relative_time,
            valid_flags,
            grad_image_points,
            grad_world_points,
            grad_start_translation,
            grad_end_translation,
            grad_start_rotation,
            grad_end_rotation,
            grad_principal_point,
            grad_fw_poly,
            grad_bw_poly,
            grad_A,
            grad_Ainv,
            nullptr,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_world_points_shutter_pose_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *world_points,
    const float *start_rotation,
    const float *end_rotation,
    int64_t shutter_type,
    int64_t max_iterations,
    float initial_relative_time,
    const bool *valid_flags,
    const float *grad_image_points,
    float *grad_world_points,
    float *grad_start_translation,
    float *grad_end_translation,
    float *grad_start_rotation,
    float *grad_end_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_shutter_pose_ftheta_backward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            world_points,
            start_rotation,
            end_rotation,
            shutter_type,
            max_iterations,
            initial_relative_time,
            valid_flags,
            grad_image_points,
            grad_world_points,
            grad_start_translation,
            grad_end_translation,
            grad_start_rotation,
            grad_end_rotation,
            grad_principal_point,
            grad_fw_poly,
            grad_bw_poly,
            grad_A,
            grad_Ainv,
            grad_distortion_coeffs,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_shutter_pose_ftheta_no_external_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    const float *image_points,
    const float *start_rotation,
    const float *end_rotation,
    int64_t shutter_type,
    const float *grad_world_rays,
    float *grad_image_points,
    float *grad_start_translation,
    float *grad_end_translation,
    float *grad_start_rotation,
    float *grad_end_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_shutter_pose_ftheta_backward_kernel<NoExternalDistortionPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            NoExternalDistortion_KernelParameters{},
            image_points,
            start_rotation,
            end_rotation,
            shutter_type,
            grad_world_rays,
            grad_image_points,
            grad_start_translation,
            grad_end_translation,
            grad_start_rotation,
            grad_end_rotation,
            grad_principal_point,
            grad_fw_poly,
            grad_bw_poly,
            grad_A,
            grad_Ainv,
            nullptr,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_backward_launch(
    int64_t count,
    FThetaProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    const float *start_rotation,
    const float *end_rotation,
    int64_t shutter_type,
    const float *grad_world_rays,
    float *grad_image_points,
    float *grad_start_translation,
    float *grad_end_translation,
    float *grad_start_rotation,
    float *grad_end_rotation,
    float *grad_principal_point,
    float *grad_fw_poly,
    float *grad_bw_poly,
    float *grad_A,
    float *grad_Ainv,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_world_rays_shutter_pose_ftheta_backward_kernel<BivariateWindshieldPolicy>
        <<<grid_for_count(count), kThreads, 0, stream>>>(
            count,
            projection,
            distortion,
            image_points,
            start_rotation,
            end_rotation,
            shutter_type,
            grad_world_rays,
            grad_image_points,
            grad_start_translation,
            grad_end_translation,
            grad_start_rotation,
            grad_end_rotation,
            grad_principal_point,
            grad_fw_poly,
            grad_bw_poly,
            grad_A,
            grad_Ainv,
            grad_distortion_coeffs,
            scratch
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
