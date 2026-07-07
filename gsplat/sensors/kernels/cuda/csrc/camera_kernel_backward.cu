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

// Native CUDA VJPs for OpenCV pinhole camera models, called by the PyTorch
// autograd wrappers for each forward kernel in camera_kernel.cu. Shared
// quaternion VJP adapters live in camera_kernel.cuh and delegate to
// gsplat/geometry.

#include "projection_backward_impl.cuh"

#include <c10/cuda/CUDAException.h>

namespace
{
// ===========================================================================
// Launch utilities
// ===========================================================================

constexpr int kThreads = 256;

// Returns the 1-D grid required to cover `count` elements at kThreads per block.
dim3 grid_for_count(int64_t count)
{
    return dim3(static_cast<unsigned int>((count + kThreads - 1) / kThreads));
}

// ===========================================================================
// K1 backward — camera_rays_to_image_points (no external distortion)
//
// Backward: pinhole project camera rays → image points (no external distortion).
// Consumes forward scratch [x, y, inv_z, r2, icD, den] (stride 6) and
// reconstructs the VJP w.r.t. intrinsics + camera_rays. Threads where
// inv_z <= 0 (ray behind camera) zero their camera_ray gradient but still
// participate in block-reduction shuffles.
// ===========================================================================
__global__ void camera_rays_to_image_points_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float *__restrict__ camera_rays,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_camera_rays,
    float *__restrict__ grad_focal_length,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_radial_coeffs,
    float *__restrict__ grad_tangential_coeffs,
    float *__restrict__ grad_thin_prism_coeffs,
    const float *__restrict__ scratch
)
{
    camera_rays_to_image_points_backward_impl<kThreads, PinholeEarlyExitPolicy, NoExternalDistortionPolicy>(
        count,
        projection,
        NoExternalDistortion_KernelParameters{},
        camera_rays,
        grad_image_points,
        grad_camera_rays,
        OpenCVPinholeIntrinsicGradOutputs{
            grad_focal_length, grad_principal_point, grad_radial_coeffs, grad_tangential_coeffs, grad_thin_prism_coeffs
        },
        nullptr,
        scratch
    );
}

// ===========================================================================
// K2 backward — image_points_to_camera_rays (no external distortion)
//
// Backward: backproject image points → camera rays (no external distortion).
// Consumes forward scratch [xs, ys, r2, icD, den] (stride 5) and applies the
// implicit fixed-point VJP.
// ===========================================================================
__global__ void image_points_to_camera_rays_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float *__restrict__ image_points,
    const float *__restrict__ grad_camera_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_focal_length,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_radial_coeffs,
    float *__restrict__ grad_tangential_coeffs,
    float *__restrict__ grad_thin_prism_coeffs,
    const float *__restrict__ scratch
)
{
    image_points_to_camera_rays_backward_impl<kThreads, PinholeEarlyExitPolicy, NoExternalDistortionPolicy>(
        count,
        projection,
        NoExternalDistortion_KernelParameters{},
        image_points,
        grad_camera_rays,
        grad_image_points,
        OpenCVPinholeIntrinsicGradOutputs{
            grad_focal_length, grad_principal_point, grad_radial_coeffs, grad_tangential_coeffs, grad_thin_prism_coeffs
        },
        nullptr,
        scratch
    );
}

// ===========================================================================
// K3 backward — project_world_points_mean_pose (no external distortion)
//
// Backward: project world points → image points with mean shutter pose
// (slerp alpha = 0.5, no external distortion). Scratch layout (stride 9):
// [0..2]=p_rel, [3..5]=cam_pt, [6]=r2, [7]=icD, [8]=den.
// ===========================================================================
__global__ void project_world_points_mean_pose_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float *__restrict__ world_points,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_world_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    float *__restrict__ grad_focal_length,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_radial_coeffs,
    float *__restrict__ grad_tangential_coeffs,
    float *__restrict__ grad_thin_prism_coeffs,
    const float *__restrict__ scratch
)
{
    project_world_points_mean_pose_backward_impl<kThreads, PinholeEarlyExitPolicy, NoExternalDistortionPolicy>(
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
        OpenCVPinholeIntrinsicGradOutputs{
            grad_focal_length, grad_principal_point, grad_radial_coeffs, grad_tangential_coeffs, grad_thin_prism_coeffs
        },
        nullptr,
        scratch
    );
}

// ===========================================================================
// K4 backward — pinhole-local project_world_points_shutter_pose body.
//
// Both external-distortion variants use this body. Scratch layout (stride 10):
// same as K3 but slot [9]=alpha. d_trans splits (1-alpha)/(alpha) between
// start/end.
// valid_flags == nullptr means all rays are active.
// ===========================================================================
template<typename DistortionPolicy>
inline __device__ void project_world_points_shutter_pose_pinhole_backward_impl(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    typename DistortionPolicy::KernelParameters distortion,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    const bool *__restrict__ valid_flags,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_world_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    OpenCVPinholeIntrinsicGradOutputs intrinsic_outputs,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    constexpr DistortionOpFamily kOp = DistortionOpFamily::ProjectWorldPointsShutterPose;
    using ForwardScratch             = DistortionScratchTraits<
        DistortionSensor::OpenCVPinhole,
        kOp,
        DistortionDirection::Forward,
        typename DistortionPolicy::Tag
    >;
    using BackwardScratch = DistortionScratchTraits<
        DistortionSensor::OpenCVPinhole,
        kOp,
        DistortionDirection::Backward,
        typename DistortionPolicy::Tag
    >;
    static_assert(ForwardScratch::kScratchStride == BackwardScratch::kScratchStride);
    static_assert(ForwardScratch::kInverseStashOffset == BackwardScratch::kInverseStashOffset);
    static_assert(ForwardScratch::kIsUndistort == BackwardScratch::kIsUndistort);
    OpenCVPinholeScratchIO<kOp>::validate<ForwardScratch>();
    OpenCVPinholeScratchIO<kOp>::validate<BackwardScratch>();

    const int64_t idx                = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const OpenCVPinholeParams params = load_opencv_pinhole_params(projection);
    const typename DistortionPolicy::Params distortion_params
        = DistortionPolicy::load(distortion, BackwardScratch::kIsUndistort);
    OpenCVPinholeParamGrads d_projection{};
    typename DistortionPolicy::ParamGrads d_distortion{};
    float3 d_start_translation   = make_float3(0.0f, 0.0f, 0.0f);
    float3 d_end_translation     = make_float3(0.0f, 0.0f, 0.0f);
    float4 d_start_rotation_xyzw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d_end_rotation_xyzw   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if(idx < count)
    {
        const int64_t off = idx * BackwardScratch::kScratchStride;
        float3 p_rel;
        float3 camera_point;
        OpenCVPinholeProjectState state;
        float alpha = 0.0f;
        OpenCVPinholeScratchIO<kOp>::load_backward<BackwardScratch>(scratch, off, p_rel, camera_point, state, alpha);
        float3 d_world_point = make_float3(0.0f, 0.0f, 0.0f);
        if(valid_flags == nullptr || valid_flags[idx])
        {
            const float3 camera_ray    = normalize3(camera_point);
            const float3 projected_ray = DistortionPolicy::apply_fwd(camera_ray, distortion_params);
            state.projection_input     = projected_ray;
            state.projection_forward   = projected_ray.z > 0.0f;
            state.inv_z                = 1.0f / projected_ray.z;
            state.x                    = projected_ray.x * state.inv_z;
            state.y                    = projected_ray.y * state.inv_z;
            const float2 d_image_point = make_float2(grad_image_points[idx * 2 + 0], grad_image_points[idx * 2 + 1]);
            float3 d_projected_ray     = make_float3(0.0f, 0.0f, 0.0f);
            opencv_pinhole_project_bwd(projected_ray, params, state, d_image_point, d_projected_ray, d_projection);
            float3 d_camera_ray = make_float3(0.0f, 0.0f, 0.0f);
            DistortionPolicy::apply_bwd(camera_ray, distortion_params, d_projected_ray, d_camera_ray, d_distortion);
            const float3 d_camera_point = normalize3_bwd(camera_point, d_camera_ray);

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
            float4 d_pose_rotation_xyzw     = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float3 d_p_rel                  = make_float3(0.0f, 0.0f, 0.0f);
            quat_inverse_rotate_bwd_xyzw_geom(pose_rotation_xyzw, p_rel, d_camera_point, d_pose_rotation_xyzw, d_p_rel);
            d_world_point                   = d_p_rel;
            const float3 d_pose_translation = scale3(d_p_rel, -1.0f);
            d_start_translation             = scale3(d_pose_translation, 1.0f - alpha);
            d_end_translation               = scale3(d_pose_translation, alpha);

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
        }
        if(grad_world_points != nullptr)
        {
            write_vec3(grad_world_points, idx, d_world_point);
        }
    }

    reduce_projection_dynamic_pose_grads<kThreads>(
        d_start_translation,
        d_end_translation,
        d_start_rotation_xyzw,
        d_end_rotation_xyzw,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation
    );
    reduce_opencv_pinhole_grads<kThreads>(d_projection, intrinsic_outputs);
    DistortionPolicy::template reduce_param_grads<kThreads>(
        d_distortion, distortion, BackwardScratch::kIsUndistort, grad_distortion_coeffs
    );
}

__global__ void project_world_points_shutter_pose_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    const bool *__restrict__ valid_flags,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_world_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    float *__restrict__ grad_focal_length,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_radial_coeffs,
    float *__restrict__ grad_tangential_coeffs,
    float *__restrict__ grad_thin_prism_coeffs,
    const float *__restrict__ scratch
)
{
    project_world_points_shutter_pose_pinhole_backward_impl<NoExternalDistortionPolicy>(
        count,
        projection,
        NoExternalDistortion_KernelParameters{},
        start_rotation,
        end_rotation,
        valid_flags,
        grad_image_points,
        grad_world_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        OpenCVPinholeIntrinsicGradOutputs{
            grad_focal_length, grad_principal_point, grad_radial_coeffs, grad_tangential_coeffs, grad_thin_prism_coeffs
        },
        nullptr,
        scratch
    );
}

// ===========================================================================
// K5 backward — image_points_to_world_rays_static_pose (no external distortion)
//
// Backward: backproject image points → world rays with static pose (no external
// distortion). Scratch layout (stride 5): [0]=xs, [1]=ys, [2]=r2, [3]=icD,
// [4]=den. The origin gradient passes directly to translation; the direction
// gradient composes camera backprojection with the static rotation.
// ===========================================================================
__global__ void image_points_to_world_rays_static_pose_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float *__restrict__ image_points,
    const float *__restrict__ rotation,
    const float *__restrict__ grad_world_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_translation,
    float *__restrict__ grad_rotation,
    float *__restrict__ grad_focal_length,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_radial_coeffs,
    float *__restrict__ grad_tangential_coeffs,
    float *__restrict__ grad_thin_prism_coeffs,
    const float *__restrict__ scratch
)
{
    image_points_to_world_rays_static_pose_backward_impl<kThreads, PinholeEarlyExitPolicy, NoExternalDistortionPolicy>(
        count,
        projection,
        NoExternalDistortion_KernelParameters{},
        image_points,
        nullptr,
        rotation,
        grad_world_rays,
        grad_image_points,
        grad_translation,
        grad_rotation,
        OpenCVPinholeIntrinsicGradOutputs{
            grad_focal_length, grad_principal_point, grad_radial_coeffs, grad_tangential_coeffs, grad_thin_prism_coeffs
        },
        nullptr,
        scratch
    );
}

// ===========================================================================
// K6 backward — image_points_to_world_rays_shutter_pose (no external distortion)
//
// Backward: backproject image points → world rays with rolling-shutter pose
// (no external distortion). Scratch layout (stride 9): [0]=xs, [1]=ys,
// [2]=r2, [3]=icD, [4]=den, [5]=alpha, [6..8]=placeholder (forward writes
// (0.0, 1.0, 0.0); this backward does not consume slots [6..8]).
// d_origin splits (1-alpha)/(alpha) to start/end translation.
// ===========================================================================
__global__ void image_points_to_world_rays_shutter_pose_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float *__restrict__ image_points,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    const float *__restrict__ grad_world_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    float *__restrict__ grad_focal_length,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_radial_coeffs,
    float *__restrict__ grad_tangential_coeffs,
    float *__restrict__ grad_thin_prism_coeffs,
    const float *__restrict__ scratch
)
{
    image_points_to_world_rays_shutter_pose_backward_impl<kThreads, PinholeEarlyExitPolicy, NoExternalDistortionPolicy>(
        count,
        projection,
        NoExternalDistortion_KernelParameters{},
        image_points,
        start_rotation,
        end_rotation,
        grad_world_rays,
        grad_image_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        OpenCVPinholeIntrinsicGradOutputs{
            grad_focal_length, grad_principal_point, grad_radial_coeffs, grad_tangential_coeffs, grad_thin_prism_coeffs
        },
        nullptr,
        scratch
    );
}

// ===========================================================================
// K7 backward — camera_rays_to_image_points (bivariate windshield distortion)
//
// Backward: project camera rays → image points with bivariate distortion.
// Scratch layout (stride 10): [0..2]=distorted_ray (unused here; kept for
// stride consistency), [3]=x, [4]=y, [5]=inv_z, [6]=r2, [7]=icD, [8]=den,
// [9]=post-bivariate front-facing flag.
// ===========================================================================
__global__ void camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ camera_rays,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_camera_rays,
    float *__restrict__ grad_focal_length,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_radial_coeffs,
    float *__restrict__ grad_tangential_coeffs,
    float *__restrict__ grad_thin_prism_coeffs,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    camera_rays_to_image_points_backward_impl<kThreads, PinholeFixedTenPolicy, BivariateWindshieldPolicy>(
        count,
        projection,
        distortion,
        camera_rays,
        grad_image_points,
        grad_camera_rays,
        OpenCVPinholeIntrinsicGradOutputs{
            grad_focal_length, grad_principal_point, grad_radial_coeffs, grad_tangential_coeffs, grad_thin_prism_coeffs
        },
        grad_distortion_coeffs,
        scratch
    );
}

// ===========================================================================
// K8 backward — image_points_to_camera_rays (bivariate windshield distortion)
//
// Backward: backproject image points → camera rays with bivariate distortion.
// Scratch layout (stride 9): [0]=xs, [1]=ys, [2]=r2, [3]=icD, [4]=den,
// [5..7]=camera_ray_pre_norm, [8]=unused.
// ===========================================================================
__global__ void image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ grad_camera_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_focal_length,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_radial_coeffs,
    float *__restrict__ grad_tangential_coeffs,
    float *__restrict__ grad_thin_prism_coeffs,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    image_points_to_camera_rays_backward_impl<kThreads, PinholeFixedTenPolicy, BivariateWindshieldPolicy>(
        count,
        projection,
        distortion,
        image_points,
        grad_camera_rays,
        grad_image_points,
        OpenCVPinholeIntrinsicGradOutputs{
            grad_focal_length, grad_principal_point, grad_radial_coeffs, grad_tangential_coeffs, grad_thin_prism_coeffs
        },
        grad_distortion_coeffs,
        scratch
    );
}

// ===========================================================================
// K9 backward — project_world_points_mean_pose (bivariate windshield)
//
// Backward: project world points → image points with mean pose and bivariate
// distortion. Scratch layout (stride 9): same as K3. The raw camera-point and
// post-bivariate front-facing gates remain distinct.
// ===========================================================================
__global__ void project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ world_points,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_world_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    float *__restrict__ grad_focal_length,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_radial_coeffs,
    float *__restrict__ grad_tangential_coeffs,
    float *__restrict__ grad_thin_prism_coeffs,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    project_world_points_mean_pose_backward_impl<kThreads, PinholeFixedTenPolicy, BivariateWindshieldPolicy>(
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
        OpenCVPinholeIntrinsicGradOutputs{
            grad_focal_length, grad_principal_point, grad_radial_coeffs, grad_tangential_coeffs, grad_thin_prism_coeffs
        },
        grad_distortion_coeffs,
        scratch
    );
}

// ===========================================================================
// K10 backward — project_world_points_shutter_pose (bivariate windshield)
//
// Backward: project world points → image points with rolling-shutter pose and
// bivariate distortion. Scratch layout (stride 10): same as K4.
// valid_flags == nullptr means all rays are active.
// ===========================================================================
__global__ void project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    const bool *__restrict__ valid_flags,
    const float *__restrict__ grad_image_points,
    float *__restrict__ grad_world_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    float *__restrict__ grad_focal_length,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_radial_coeffs,
    float *__restrict__ grad_tangential_coeffs,
    float *__restrict__ grad_thin_prism_coeffs,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    project_world_points_shutter_pose_pinhole_backward_impl<BivariateWindshieldPolicy>(
        count,
        projection,
        distortion,
        start_rotation,
        end_rotation,
        valid_flags,
        grad_image_points,
        grad_world_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        OpenCVPinholeIntrinsicGradOutputs{
            grad_focal_length, grad_principal_point, grad_radial_coeffs, grad_tangential_coeffs, grad_thin_prism_coeffs
        },
        grad_distortion_coeffs,
        scratch
    );
}

// ===========================================================================
// K11 backward — image_points_to_world_rays_static_pose (bivariate windshield)
//
// Backward: backproject image points → world rays with static pose and bivariate
// distortion. Scratch layout (stride 9): [0]=xs, [1]=ys, [2]=r2, [3]=icD,
// [4]=den, [5..7]=camera_ray_pre_norm.
// ===========================================================================
__global__ void image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ rotation,
    const float *__restrict__ grad_world_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_translation,
    float *__restrict__ grad_rotation,
    float *__restrict__ grad_focal_length,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_radial_coeffs,
    float *__restrict__ grad_tangential_coeffs,
    float *__restrict__ grad_thin_prism_coeffs,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    image_points_to_world_rays_static_pose_backward_impl<kThreads, PinholeFixedTenPolicy, BivariateWindshieldPolicy>(
        count,
        projection,
        distortion,
        image_points,
        nullptr,
        rotation,
        grad_world_rays,
        grad_image_points,
        grad_translation,
        grad_rotation,
        OpenCVPinholeIntrinsicGradOutputs{
            grad_focal_length, grad_principal_point, grad_radial_coeffs, grad_tangential_coeffs, grad_thin_prism_coeffs
        },
        grad_distortion_coeffs,
        scratch
    );
}

// ===========================================================================
// K12 backward — image_points_to_world_rays_shutter_pose (bivariate windshield)
//
// Backward: backproject image points → world rays with rolling-shutter pose
// and bivariate distortion. Scratch layout (stride 12): [0]=xs, [1]=ys,
// [2]=r2, [3]=icD, [4]=den, [5..7]=camera_ray_pre_norm, [8]=alpha,
// [9..11]=compatibility placeholders.
// ===========================================================================
__global__ void image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_backward_kernel(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *__restrict__ image_points,
    const float *__restrict__ start_rotation,
    const float *__restrict__ end_rotation,
    const float *__restrict__ grad_world_rays,
    float *__restrict__ grad_image_points,
    float *__restrict__ grad_start_translation,
    float *__restrict__ grad_end_translation,
    float *__restrict__ grad_start_rotation,
    float *__restrict__ grad_end_rotation,
    float *__restrict__ grad_focal_length,
    float *__restrict__ grad_principal_point,
    float *__restrict__ grad_radial_coeffs,
    float *__restrict__ grad_tangential_coeffs,
    float *__restrict__ grad_thin_prism_coeffs,
    float *__restrict__ grad_distortion_coeffs,
    const float *__restrict__ scratch
)
{
    image_points_to_world_rays_shutter_pose_backward_impl<kThreads, PinholeFixedTenPolicy, BivariateWindshieldPolicy>(
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
        OpenCVPinholeIntrinsicGradOutputs{
            grad_focal_length, grad_principal_point, grad_radial_coeffs, grad_tangential_coeffs, grad_thin_prism_coeffs
        },
        grad_distortion_coeffs,
        scratch
    );
}
} // namespace

// ===========================================================================
// Host wrappers — launch the __global__ kernels above
// ===========================================================================

// Launches K1 backward kernel.
void camera_rays_to_image_points_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float *camera_rays,
    const float *grad_image_points,
    float *grad_camera_rays,
    float *grad_focal_length,
    float *grad_principal_point,
    float *grad_radial_coeffs,
    float *grad_tangential_coeffs,
    float *grad_thin_prism_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    camera_rays_to_image_points_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        camera_rays,
        grad_image_points,
        grad_camera_rays,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K2 backward kernel.
void image_points_to_camera_rays_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float *image_points,
    const float *grad_camera_rays,
    float *grad_image_points,
    float *grad_focal_length,
    float *grad_principal_point,
    float *grad_radial_coeffs,
    float *grad_tangential_coeffs,
    float *grad_thin_prism_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_camera_rays_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        image_points,
        grad_camera_rays,
        grad_image_points,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K3 backward kernel.
void project_world_points_mean_pose_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float *world_points,
    const float *start_rotation,
    const float *end_rotation,
    const float *grad_image_points,
    float *grad_world_points,
    float *grad_start_translation,
    float *grad_end_translation,
    float *grad_start_rotation,
    float *grad_end_rotation,
    float *grad_focal_length,
    float *grad_principal_point,
    float *grad_radial_coeffs,
    float *grad_tangential_coeffs,
    float *grad_thin_prism_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_mean_pose_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        world_points,
        start_rotation,
        end_rotation,
        grad_image_points,
        grad_world_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K4 backward kernel.
void project_world_points_shutter_pose_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    float *grad_focal_length,
    float *grad_principal_point,
    float *grad_radial_coeffs,
    float *grad_tangential_coeffs,
    float *grad_thin_prism_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    (void)world_points;
    (void)shutter_type;
    (void)max_iterations;
    (void)initial_relative_time;
    project_world_points_shutter_pose_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        start_rotation,
        end_rotation,
        valid_flags,
        grad_image_points,
        grad_world_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K5 backward kernel.
void image_points_to_world_rays_static_pose_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    const float *image_points,
    const float *translation,
    const float *rotation,
    const float *grad_world_rays,
    float *grad_image_points,
    float *grad_translation,
    float *grad_rotation,
    float *grad_focal_length,
    float *grad_principal_point,
    float *grad_radial_coeffs,
    float *grad_tangential_coeffs,
    float *grad_thin_prism_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    (void)translation;
    image_points_to_world_rays_static_pose_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        image_points,
        rotation,
        grad_world_rays,
        grad_image_points,
        grad_translation,
        grad_rotation,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K6 backward kernel.
void image_points_to_world_rays_shutter_pose_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    float *grad_focal_length,
    float *grad_principal_point,
    float *grad_radial_coeffs,
    float *grad_tangential_coeffs,
    float *grad_thin_prism_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    (void)shutter_type;
    image_points_to_world_rays_shutter_pose_backward_kernel<<<grid_for_count(count), kThreads, 0, stream>>>(
        count,
        projection,
        image_points,
        start_rotation,
        end_rotation,
        grad_world_rays,
        grad_image_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        scratch
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K7 backward kernel.
void camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *camera_rays,
    const float *grad_image_points,
    float *grad_camera_rays,
    float *grad_focal_length,
    float *grad_principal_point,
    float *grad_radial_coeffs,
    float *grad_tangential_coeffs,
    float *grad_thin_prism_coeffs,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        distortion,
        camera_rays,
        grad_image_points,
        grad_camera_rays,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K8 backward kernel.
void image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    const float *grad_camera_rays,
    float *grad_image_points,
    float *grad_focal_length,
    float *grad_principal_point,
    float *grad_radial_coeffs,
    float *grad_tangential_coeffs,
    float *grad_thin_prism_coeffs,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        distortion,
        image_points,
        grad_camera_rays,
        grad_image_points,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K9 backward kernel.
void project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    float *grad_focal_length,
    float *grad_principal_point,
    float *grad_radial_coeffs,
    float *grad_tangential_coeffs,
    float *grad_thin_prism_coeffs,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
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
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K10 backward kernel.
void project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    float *grad_focal_length,
    float *grad_principal_point,
    float *grad_radial_coeffs,
    float *grad_tangential_coeffs,
    float *grad_thin_prism_coeffs,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    (void)world_points;
    (void)shutter_type;
    (void)max_iterations;
    (void)initial_relative_time;
    project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        distortion,
        start_rotation,
        end_rotation,
        valid_flags,
        grad_image_points,
        grad_world_points,
        grad_start_translation,
        grad_end_translation,
        grad_start_rotation,
        grad_end_rotation,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K11 backward kernel.
void image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
    BivariateWindshieldDistortion_KernelParameters distortion,
    const float *image_points,
    const float *translation,
    const float *rotation,
    const float *grad_world_rays,
    float *grad_image_points,
    float *grad_translation,
    float *grad_rotation,
    float *grad_focal_length,
    float *grad_principal_point,
    float *grad_radial_coeffs,
    float *grad_tangential_coeffs,
    float *grad_thin_prism_coeffs,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    (void)translation;
    image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
        projection,
        distortion,
        image_points,
        rotation,
        grad_world_rays,
        grad_image_points,
        grad_translation,
        grad_rotation,
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches K12 backward kernel.
void image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_backward_launch(
    int64_t count,
    OpenCVPinholeProjection_KernelParameters projection,
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
    float *grad_focal_length,
    float *grad_principal_point,
    float *grad_radial_coeffs,
    float *grad_tangential_coeffs,
    float *grad_thin_prism_coeffs,
    float *grad_distortion_coeffs,
    const float *scratch,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    (void)shutter_type;
    image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_backward_kernel<<<
        grid_for_count(count),
        kThreads,
        0,
        stream
    >>>(count,
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
        grad_focal_length,
        grad_principal_point,
        grad_radial_coeffs,
        grad_tangential_coeffs,
        grad_thin_prism_coeffs,
        grad_distortion_coeffs,
        scratch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
