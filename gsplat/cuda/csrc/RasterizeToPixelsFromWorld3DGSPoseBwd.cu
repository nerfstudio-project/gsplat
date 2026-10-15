/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#    include <ATen/core/Tensor.h>
#    include <ATen/cuda/Atomic.cuh>
#    include <c10/cuda/CUDAStream.h>
#    include <cstdint>
#    include <cuda/std/optional>

#    include "Common.h"
#    include "ExternalDistortion.cuh"
#    include "RasterizeToPixelsFromWorld3DGS.h"
#    include "RasterizeToPixelsFromWorld3DGS.cuh"
#    include "TorchUtils.h"
#    include "Utils.cuh"
#    include "Cameras.cuh"
#    include "Lidars.cuh"

namespace gsplat
{
namespace
{
    // Backpropagate world-ray gradients into shutter poses, reduced per image.
    // Ray generation uses o = R^T (o_cam - t), d = R^T c.
    // Recover o_cam and c from the forward ray to support all camera models.
    constexpr uint32_t POSE_BWD_BLOCK = 256;

    // Coefficients for a rotation vector v with |v| = theta:
    //   Exp(v) x   = x + s (v x x) + a v x (v x x)
    //   Jr(v) x    = x - a (v x x) + b v x (v x x)
    //   Jr^-1(v) x = x + 1/2 (v x x) + e v x (v x x)
    struct So3Coeffs
    {
        float s, a, b, e;
    };

    __device__ __forceinline__ So3Coeffs so3_coeffs(const vec3 &v)
    {
        const float th2 = glm::dot(v, v);
        So3Coeffs k;
        if(th2 < 0.04f)
        {
            // Use Taylor expansions below 0.2 rad to avoid float32 cancellation.
            k.s = 1.f - th2 / 6.f * (1.f - th2 / 20.f);
            k.a = 0.5f - th2 / 24.f * (1.f - th2 / 30.f);
            k.b = 1.f / 6.f - th2 / 120.f * (1.f - th2 / 42.f);
            k.e = 1.f / 12.f + th2 / 720.f * (1.f + th2 / 42.f);
        }
        else
        {
            const float th = sqrtf(th2);
            const float sn = sinf(th);
            const float cs = cosf(th);
            k.s            = sn / th;
            k.a            = (1.f - cs) / th2;
            k.b            = (th - sn) / (th2 * th);
            k.e            = 1.f / th2 - (1.f + cs) / (2.f * th * sn);
        }
        return k;
    }

    // x + c1 (v x x) + c2 v x (v x x)
    __device__ __forceinline__ vec3 so3_apply(const vec3 &v, const float c1, const float c2, const vec3 &x)
    {
        const vec3 vx = glm::cross(v, x);
        return x + c1 * vx + c2 * glm::cross(v, vx);
    }

    struct RayAndTime
    {
        WorldRay ray;
        float tau;
    };

    template<typename scalar_t>
    __global__ void rasterize_to_pixels_from_world_3dgs_pose_bwd_kernel(
        const uint32_t image_width,
        const uint32_t image_height,
        const scalar_t *__restrict__ viewmats0, // [I, 4, 4]
        const scalar_t *__restrict__ viewmats1, // [I, 4, 4] optional for rolling shutter
        const scalar_t *__restrict__ Ks,        // [I, 3, 3]
        const CameraModelType camera_model_type,
        const ShutterType rs_type,
        const scalar_t *__restrict__ radial_coeffs,                    // [I, 6] or [I, 4] optional
        const scalar_t *__restrict__ tangential_coeffs,                // [I, 2] optional
        const scalar_t *__restrict__ thin_prism_coeffs,                // [I, 4] optional
        const FThetaCameraDistortionDeviceParams ftheta_device_coeffs, // shared parameters for all cameras
        const cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_device_coeffs,
        const cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> external_distortion_device_params,
        const scalar_t *__restrict__ v_rays, // [I, image_height, image_width, 6]
        double *__restrict__ v_pose          // [I, POSE_BWD_ACC]
    )
    {
        const uint32_t iid        = blockIdx.y;
        const uint32_t image_area = image_width * image_height;
        const uint32_t pix_id     = blockIdx.x * blockDim.x + threadIdx.x;
        const bool inside         = pix_id < image_area;
        const bool rolling        = viewmats1 != nullptr;

        float acc[POSE_BWD_ACC] = {};

        if(inside)
        {
            const scalar_t *vr = v_rays + (static_cast<int64_t>(iid) * image_area + pix_id) * 6;
            const vec3 g_o     = {vr[0], vr[1], vr[2]};
            const vec3 g_d     = {vr[3], vr[4], vr[5]};

            // Avoid regenerating rays for zero gradient contributions.
            if(g_o != vec3(0.f) || g_d != vec3(0.f))
            {
                const auto rs_params
                    = RollingShutterParameters(viewmats0 + iid * 16, rolling ? viewmats1 + iid * 16 : nullptr);
                const uint32_t i    = pix_id / image_width;
                const uint32_t j    = pix_id - i * image_width;
                // Same camera model and pixel mapping as compute_world_ray.
                const RayAndTime rt = visit_camera_model<RayAndTime>(
                    iid,
                    Ks,
                    image_width,
                    image_height,
                    camera_model_type,
                    rs_type,
                    radial_coeffs,
                    tangential_coeffs,
                    thin_prism_coeffs,
                    ftheta_device_coeffs,
                    lidar_device_coeffs,
                    external_distortion_device_params,
                    [&](const auto &camera_model)
                    {
                        const auto image_point = camera_model.element_to_image_point(j, i);
                        return RayAndTime{
                            camera_model.image_point_to_world_ray_shutter_pose(image_point, rs_params),
                            camera_model.shutter_relative_frame_time(image_point),
                        };
                    }
                );

                if(rt.ray.valid_flag)
                {
                    const vec3 &o = rt.ray.ray_org;
                    const vec3 &d = rt.ray.ray_dir;
                    if(!rolling)
                    {
                        // Accumulate M = c g_d^T + o_cam g_o^T and sum g_o.
                        // finalize_viewmat_grads applies the translation terms.
                        const mat3 R    = glm::mat3_cast(rs_params.q_start);
                        const vec3 c    = R * d;
                        const vec3 ocam = R * o + rs_params.t_start;
#    pragma unroll
                        for(int r = 0; r < 3; ++r)
                        {
#    pragma unroll
                            for(int k = 0; k < 3; ++k)
                            {
                                acc[r * 3 + k] = c[r] * g_d[k] + ocam[r] * g_o[k];
                            }
                        }
                        acc[9]  = g_o.x;
                        acc[10] = g_o.y;
                        acc[11] = g_o.z;
                    }
                    else
                    {
                        // Endpoint Jacobians for right perturbations R <- R Exp(xi).
                        // J1 = tau Jr(tau phi) Jr^-1(phi); J0 = Exp(-tau phi) - J1 Exp(-phi).
                        const float tau     = rt.tau;
                        const ShutterPose p = interpolate_shutter_pose(tau, rs_params);
                        const vec3 g_xi     = glm::cross(g_d, d) + glm::cross(g_o, o);
                        const vec3 v_t      = -(glm::mat3_cast(p.q) * g_o);

                        // phi = Log(R0^T R1) on the short arc, as glm::slerp takes it.
                        glm::fquat q1 = rs_params.q_end;
                        if(glm::dot(rs_params.q_start, q1) < 0.f)
                        {
                            q1 = -q1;
                        }
                        const glm::fquat r = glm::inverse(rs_params.q_start) * q1;
                        const vec3 rv      = {r.x, r.y, r.z};
                        const float rn     = glm::length(rv);
                        const vec3 phi     = rn > 1e-8f ? (2.f * atan2f(rn, r.w) / rn) * rv : (2.f / r.w) * rv;
                        const vec3 tphi    = tau * phi;

                        const So3Coeffs kp = so3_coeffs(phi);
                        const So3Coeffs kt = so3_coeffs(tphi);
                        // J1^T g = tau Jr^-1(phi)^T Jr(tau phi)^T g
                        const vec3 g1      = tau * so3_apply(phi, -0.5f, kp.e, so3_apply(tphi, kt.a, kt.b, g_xi));
                        // J0^T g = Exp(tau phi) g - Exp(phi) J1^T g
                        const vec3 g0      = so3_apply(tphi, kt.s, kt.a, g_xi) - so3_apply(phi, kp.s, kp.a, g1);

                        const vec3 vt0 = (1.f - tau) * v_t;
                        const vec3 vt1 = tau * v_t;
#    pragma unroll
                        for(int k = 0; k < 3; ++k)
                        {
                            acc[k]     = g0[k];
                            acc[3 + k] = vt0[k];
                            acc[6 + k] = g1[k];
                            acc[9 + k] = vt1[k];
                        }
                    }
                }
            }
        }

        // Reduce within each block, then accumulate image totals in float64.
        const uint32_t lane = threadIdx.x & 31u;
        const uint32_t warp = threadIdx.x >> 5;
#    pragma unroll
        for(int k = 0; k < POSE_BWD_ACC; ++k)
        {
#    pragma unroll
            for(int offset = 16; offset > 0; offset >>= 1)
            {
                acc[k] += __shfl_down_sync(0xFFFFFFFFu, acc[k], offset);
            }
        }
        __shared__ float warp_sums[POSE_BWD_BLOCK / 32][POSE_BWD_ACC];
        if(lane == 0)
        {
#    pragma unroll
            for(int k = 0; k < POSE_BWD_ACC; ++k)
            {
                warp_sums[warp][k] = acc[k];
            }
        }
        __syncthreads();
        if(warp == 0)
        {
            constexpr uint32_t n_warps = POSE_BWD_BLOCK / 32;
#    pragma unroll
            for(int k = 0; k < POSE_BWD_ACC; ++k)
            {
                float v = lane < n_warps ? warp_sums[lane][k] : 0.f;
#    pragma unroll
                for(int offset = 16; offset > 0; offset >>= 1)
                {
                    v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
                }
                if(lane == 0 && v != 0.f)
                {
                    gpuAtomicAdd(v_pose + iid * POSE_BWD_ACC + k, static_cast<double>(v));
                }
            }
        }
    }
} // namespace

void launch_rasterize_to_pixels_from_world_3dgs_pose_bwd_kernel(
    const uint32_t image_width,
    const uint32_t image_height,
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const CameraModelType camera_model,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,                              // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs,                          // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs,                          // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs, // shared parameters for all cameras
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    const at::Tensor v_rays, // [..., C, image_height, image_width, 6]
    at::Tensor v_pose        // [I, POSE_BWD_ACC] float64, zero-initialized
)
{
    const uint32_t I = static_cast<uint32_t>(viewmats0.numel() / 16);
    TORCH_CHECK(
        v_rays.is_contiguous() && v_rays.numel() == static_cast<int64_t>(I) * image_height * image_width * 6,
        "v_rays must be a contiguous [..., C, H, W, 6] tensor"
    );
    TORCH_CHECK(
        v_pose.scalar_type() == at::kDouble
            && v_pose.is_contiguous()
            && v_pose.numel() == static_cast<int64_t>(I) * POSE_BWD_ACC,
        "v_pose must be a contiguous float64 [I, ",
        POSE_BWD_ACC,
        "] tensor"
    );
    TORCH_CHECK(I <= 65535u, "too many images for the eval3d pose backward grid: ", I);
    if(I == 0 || image_width == 0 || image_height == 0)
    {
        return;
    }

    FThetaCameraDistortionDeviceParams ftheta_device_coeffs(gsplat::checked_deref(ftheta_coeffs, "ftheta_coeffs"));
    cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> external_distortion_device_params
        = cuda::std::nullopt;
    if(external_distortion_params.has_value())
    {
        const auto &params = gsplat::checked_deref(external_distortion_params.value(), "external_distortion_params");
        external_distortion_device_params = extdist::BivariateWindshieldModelDeviceParams(params);
    }
    cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_device_coeffs
        = cuda::std::nullopt;
    if(lidar_coeffs.has_value())
    {
        lidar_device_coeffs = *lidar_coeffs.value();
    }

    const uint32_t image_area = image_width * image_height;
    const dim3 grid           = {(image_area + POSE_BWD_BLOCK - 1) / POSE_BWD_BLOCK, I, 1};
    rasterize_to_pixels_from_world_3dgs_pose_bwd_kernel<float>
        <<<grid, POSE_BWD_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            image_width,
            image_height,
            viewmats0.const_data_ptr<float>(),
            data_ptr_or_null<const float>(viewmats1),
            Ks.const_data_ptr<float>(),
            camera_model,
            rs_type,
            data_ptr_or_null<const float>(radial_coeffs),
            data_ptr_or_null<const float>(tangential_coeffs),
            data_ptr_or_null<const float>(thin_prism_coeffs),
            ftheta_device_coeffs,
            lidar_device_coeffs,
            external_distortion_device_params,
            v_rays.const_data_ptr<float>(),
            v_pose.data_ptr<double>()
        );
}
} // namespace gsplat

#endif
