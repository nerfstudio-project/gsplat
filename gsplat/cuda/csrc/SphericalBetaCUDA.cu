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

#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/ops/zeros.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "SphericalBeta.h"
#include "SphericalHarmonics.cuh"
#include "SphericalHarmonics.h"
#include "Utils.cuh"

namespace gsplat
{
namespace cg = cooperative_groups;

// A Spherical Beta lobe is a bounded, Phong-like specular kernel: an amplitude
// weighting (R . V)^(4 exp(beta)) over the hemisphere facing the lobe axis R,
// which is parameterised by the spherical angles (theta, phi). Storing the
// exponent in log space keeps it positive and spans several orders of magnitude
// with a well-conditioned gradient. Unlike a spherical Gaussian the kernel
// reaches exactly zero at the horizon, so no truncation is needed. See
// "Deformable Beta Splatting", Liu et al., arXiv:2501.18630, Eq. 8.
struct SphericalBetaLobe
{
    vec3 axis;       // unit lobe direction R
    float sin_theta; // reused by the angular gradients
    float cos_theta;
    float sin_phi;
    float cos_phi;
    float dot;      // R . V, negative on the far hemisphere
    float exponent; // 4 exp(beta)
    float weight;   // (R . V)^exponent, or 0 on the far hemisphere
};

__device__ __forceinline__ SphericalBetaLobe
    spherical_beta_lobe(const vec3 &dir_n, const float theta, const float phi, const float beta)
{
    SphericalBetaLobe lobe;
    sincosf(theta, &lobe.sin_theta, &lobe.cos_theta);
    sincosf(phi, &lobe.sin_phi, &lobe.cos_phi);
    lobe.axis     = vec3(lobe.sin_theta * lobe.cos_phi, lobe.sin_theta * lobe.sin_phi, lobe.cos_theta);
    lobe.dot      = glm::dot(dir_n, lobe.axis);
    lobe.exponent = 4.f * expf(beta);
    lobe.weight   = lobe.dot > 0.f ? powf(lobe.dot, lobe.exponent) : 0.f;
    return lobe;
}

// One thread per output element; a thread evaluates every lobe and writes all
// three channels, because the lobe geometry is shared across channels.
template<typename scalar_t, typename opmath_t>
__global__ void spherical_beta_fwd_kernel(
    const int64_t n_elements,
    const uint32_t C,
    const uint32_t N,
    const uint32_t M,
    const uint32_t lobes_to_use,
    const float *__restrict__ means,
    const float *__restrict__ viewmats,
    const float *__restrict__ camera_offsets,
    const opmath_t *__restrict__ base_colors, // [..., N, 3]
    const scalar_t *__restrict__ coeffs,      // [N, M, 6]
    const bool *__restrict__ masks,           // [..., N]
    const int64_t *__restrict__ batch_ids,
    const int64_t *__restrict__ camera_ids,
    const int64_t *__restrict__ gaussian_ids,
    opmath_t *__restrict__ colors // [..., N, 3]
)
{
    // parallelize over B * C * N (dense) or nnz (packed)
    const int64_t output_id = cg::this_grid().thread_rank();
    if(output_id >= n_elements)
    {
        return;
    }
    const bool packed = batch_ids != nullptr;

    // Masked-out elements still pass the base colour through, so the output is
    // defined everywhere and callers need no separate copy.
    const opmath_t *base = base_colors + output_id * SB_NUM_CHANNELS;
    opmath_t out[SB_NUM_CHANNELS];
#pragma unroll
    for(int c = 0; c < SB_NUM_CHANNELS; ++c)
    {
        out[c] = base[c];
    }

    if(masks == nullptr || masks[output_id])
    {
        const int64_t batch_id    = packed ? batch_ids[output_id] : output_id / (static_cast<int64_t>(C) * N);
        const int64_t camera_id   = packed ? camera_ids[output_id] : (output_id / N) % C;
        const int64_t gaussian_id = packed ? gaussian_ids[output_id] : output_id % N;
        const int64_t coeff_id    = packed ? output_id : gaussian_id;

        const int64_t image_offset = batch_id * C + camera_id;
        const float *mean          = means + (batch_id * N + gaussian_id) * 3;
        const vec3 dir             = view_direction_from_camera_data(mean, viewmats, camera_offsets, image_offset);
        const vec3 dir_n           = dir * rsqrtf(glm::dot(dir, dir));

        const scalar_t *lobe_ptr = coeffs + coeff_id * M * SB_LOBE_WIDTH;
        for(uint32_t m = 0; m < lobes_to_use; ++m, lobe_ptr += SB_LOBE_WIDTH)
        {
            const SphericalBetaLobe lobe = spherical_beta_lobe(
                dir_n,
                static_cast<opmath_t>(lobe_ptr[3]),
                static_cast<opmath_t>(lobe_ptr[4]),
                static_cast<opmath_t>(lobe_ptr[5])
            );
#pragma unroll
            for(int c = 0; c < SB_NUM_CHANNELS; ++c)
            {
                out[c] += lobe.weight * static_cast<opmath_t>(lobe_ptr[c]);
            }
        }
    }

#pragma unroll
    for(int c = 0; c < SB_NUM_CHANNELS; ++c)
    {
        colors[output_id * SB_NUM_CHANNELS + c] = out[c];
    }
}

void launch_spherical_beta_fwd_kernel(
    // inputs
    const uint32_t lobes_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::optional<at::Tensor> viewmats_rs,
    const at::Tensor base_colors,         // [..., N, 3]
    const at::Tensor coeffs,              // [N, M, 6]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    // outputs
    at::Tensor colors // [..., N, 3]
)
{
    const uint32_t M = coeffs.size(-2);
    const uint32_t N = means.size(-2);
    const uint32_t C = viewmats.size(-3);
    const uint32_t B = c10::multiply_integers(means.sizes().slice(0, means.dim() - 2));

    int64_t n_elements = batch_ids.has_value() ? coeffs.size(0) : static_cast<int64_t>(B) * C * N;
    if(n_elements == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    constexpr unsigned int threads = 256;
    unsigned int blocks            = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));
    auto stream                    = at::cuda::getCurrentCUDAStream();

    at::Tensor camera_offsets;
    if(viewmats_rs.has_value())
    {
        camera_offsets = precompute_spherical_harmonics_camera_offsets(viewmats, viewmats_rs);
    }

    // Dispatch on the coeff dtype (fp16/fp32); colours use opmath_t (float).
    AT_DISPATCH_V2(
        coeffs.scalar_type(),
        "spherical_beta_fwd_kernel",
        AT_WRAP(
            [&]()
            {
                using opmath_t  = at::opmath_type<scalar_t>;
                auto *masks_ptr = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;

                spherical_beta_fwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
                    n_elements,
                    C,
                    N,
                    M,
                    lobes_to_use,
                    means.const_data_ptr<float>(),
                    viewmats.const_data_ptr<float>(),
                    camera_offsets.defined() ? camera_offsets.const_data_ptr<float>() : nullptr,
                    base_colors.const_data_ptr<opmath_t>(),
                    coeffs.const_data_ptr<scalar_t>(),
                    masks_ptr,
                    batch_ids.has_value() ? batch_ids.value().const_data_ptr<int64_t>() : nullptr,
                    camera_ids.has_value() ? camera_ids.value().const_data_ptr<int64_t>() : nullptr,
                    gaussian_ids.has_value() ? gaussian_ids.value().const_data_ptr<int64_t>() : nullptr,
                    colors.data_ptr<opmath_t>()
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        ),
        at::kFloat,
        at::kHalf
    );
}

// One thread per (coefficient row, lobe). A lobe owns exactly six gradients, so
// they accumulate in registers across all images without a per-degree template
// or a scratch buffer. Threads sharing a coefficient row are adjacent, which
// lets them combine their view-direction gradients before touching means.
template<typename scalar_t, typename opmath_t>
__global__ void spherical_beta_bwd_kernel(
    const int64_t coefficient_rows,
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t M,
    const uint32_t lobes_to_use,
    const float *__restrict__ means,
    const float *__restrict__ viewmats,
    const float *__restrict__ camera_offsets,
    const scalar_t *__restrict__ coeffs, // [N, M, 6]
    const bool *__restrict__ masks,      // [..., N]
    const int64_t *__restrict__ batch_ids,
    const int64_t *__restrict__ camera_ids,
    const int64_t *__restrict__ gaussian_ids,
    const opmath_t *__restrict__ v_colors, // [..., N, 3]
    scalar_t *__restrict__ v_coeffs,       // [N, M, 6] (coeff dtype)
    float *__restrict__ v_means,
    float *__restrict__ v_viewdirs // [..., N, 3] optional
)
{
    const int64_t idx   = cg::this_grid().thread_rank();
    const int64_t count = coefficient_rows * M;
    if(idx >= count)
    {
        return;
    }
    const bool packed      = batch_ids != nullptr;
    const int64_t coeff_id = idx / M;
    const uint32_t m       = static_cast<uint32_t>(idx % M);

    scalar_t *v_lobe = v_coeffs + idx * SB_LOBE_WIDTH;
    if(m >= lobes_to_use)
    {
        // Stored lobes beyond the active count do not contribute; zero them so
        // the at::empty output is fully defined.
#pragma unroll
        for(int i = 0; i < SB_LOBE_WIDTH; ++i)
        {
            v_lobe[i] = static_cast<scalar_t>(0.f);
        }
        return;
    }

    const scalar_t *lobe_ptr = coeffs + idx * SB_LOBE_WIDTH;
    const opmath_t theta     = static_cast<opmath_t>(lobe_ptr[3]);
    const opmath_t phi       = static_cast<opmath_t>(lobe_ptr[4]);
    const opmath_t beta      = static_cast<opmath_t>(lobe_ptr[5]);

    opmath_t v_rgb[SB_NUM_CHANNELS] = {};
    opmath_t v_theta                = 0;
    opmath_t v_phi                  = 0;
    opmath_t v_beta                 = 0;

    const uint32_t image_count = packed ? 1 : B * C;
    for(uint32_t dense_image_id = 0; dense_image_id < image_count; ++dense_image_id)
    {
        const int64_t output_id = packed ? coeff_id : static_cast<int64_t>(dense_image_id) * N + coeff_id;
        if(masks != nullptr && !masks[output_id])
        {
            continue;
        }

        const int64_t batch_id     = packed ? batch_ids[output_id] : dense_image_id / C;
        const int64_t camera_id    = packed ? camera_ids[output_id] : dense_image_id % C;
        const int64_t gaussian_id  = packed ? gaussian_ids[output_id] : coeff_id;
        const int64_t image_offset = batch_id * C + camera_id;
        const float *mean          = means + (batch_id * N + gaussian_id) * 3;
        const vec3 dir             = view_direction_from_camera_data(mean, viewmats, camera_offsets, image_offset);
        const float inorm          = rsqrtf(glm::dot(dir, dir));
        const vec3 dir_n           = dir * inorm;

        const SphericalBetaLobe lobe = spherical_beta_lobe(dir_n, theta, phi, beta);
        const opmath_t *v_color      = v_colors + output_id * SB_NUM_CHANNELS;
        vec3 v_dir                   = {0.f, 0.f, 0.f};

        // The amplitude gradient is defined even on the far hemisphere, where
        // the weight is zero; the angular gradients are not.
        opmath_t v_weight = 0;
#pragma unroll
        for(int c = 0; c < SB_NUM_CHANNELS; ++c)
        {
            v_rgb[c] += v_color[c] * lobe.weight;
            v_weight += v_color[c] * static_cast<opmath_t>(lobe_ptr[c]);
        }

        if(lobe.dot > 0.f)
        {
            const opmath_t v_dot  = v_weight * lobe.exponent * powf(lobe.dot, lobe.exponent - 1.f);
            v_theta              += v_dot
                                  * (dir_n.x * lobe.cos_theta * lobe.cos_phi
                                     + dir_n.y * lobe.cos_theta * lobe.sin_phi
                                     - dir_n.z * lobe.sin_theta);
            v_phi                += v_dot * lobe.sin_theta * (dir_n.y * lobe.cos_phi - dir_n.x * lobe.sin_phi);
            // d(weight)/d(beta) folds in d(exponent)/d(beta) = exponent.
            v_beta               += v_weight * lobe.weight * logf(lobe.dot) * lobe.exponent;

            if(v_means != nullptr || v_viewdirs != nullptr)
            {
                // Chain through the direction normalisation.
                const vec3 v_dir_n = v_dot * lobe.axis;
                v_dir              = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;
            }
        }

        // Adjacent threads handle lobes of the same coefficient row. Combine
        // their direction gradients before touching the geometry outputs.
        if((v_means != nullptr || v_viewdirs != nullptr) && reduce_view_direction_channels(coeff_id, v_dir))
        {
            if(v_means != nullptr)
            {
                float *v_mean = v_means + (batch_id * N + gaussian_id) * 3;
                gpuAtomicAdd(v_mean, v_dir.x);
                gpuAtomicAdd(v_mean + 1, v_dir.y);
                gpuAtomicAdd(v_mean + 2, v_dir.z);
            }
            if(v_viewdirs != nullptr)
            {
                float *v_dir_out = v_viewdirs + output_id * 3;
                gpuAtomicAdd(v_dir_out, v_dir.x);
                gpuAtomicAdd(v_dir_out + 1, v_dir.y);
                gpuAtomicAdd(v_dir_out + 2, v_dir.z);
            }
        }
    }

#pragma unroll
    for(int c = 0; c < SB_NUM_CHANNELS; ++c)
    {
        v_lobe[c] = static_cast<scalar_t>(v_rgb[c]);
    }
    v_lobe[3] = static_cast<scalar_t>(v_theta);
    v_lobe[4] = static_cast<scalar_t>(v_phi);
    v_lobe[5] = static_cast<scalar_t>(v_beta);
}

void launch_spherical_beta_bwd_kernel(
    // inputs
    const uint32_t lobes_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::optional<at::Tensor> viewmats_rs,
    const at::Tensor coeffs,              // [N, M, 6]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    const at::Tensor v_colors, // [..., N, 3]
    // outputs
    at::Tensor v_coeffs,
    at::optional<at::Tensor> v_means,
    at::optional<at::Tensor> v_viewmats,
    at::optional<at::Tensor> v_viewmats_rs
)
{
    const uint32_t M = coeffs.size(-2);
    const uint32_t N = means.size(-2);
    const uint32_t C = viewmats.size(-3);
    const uint32_t B = c10::multiply_integers(means.sizes().slice(0, means.dim() - 2));

    const bool needs_v_viewdirs = v_viewmats.has_value() || v_viewmats_rs.has_value();

    // One thread per (coefficient row, lobe). Dense mode loops over B * C images
    // internally; packed mode has one coefficient row per output.
    const int64_t coefficient_rows = batch_ids.has_value() ? coeffs.size(0) : N;
    int64_t n_elements             = coefficient_rows * M;
    if(n_elements == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    constexpr unsigned int threads = 256;
    unsigned int blocks            = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));
    auto stream                    = at::cuda::getCurrentCUDAStream();

    at::Tensor v_viewdirs;
    if(needs_v_viewdirs)
    {
        v_viewdirs = at::zeros(v_colors.sizes(), means.options());
    }

    at::Tensor camera_offsets;
    if(viewmats_rs.has_value())
    {
        camera_offsets = precompute_spherical_harmonics_camera_offsets(viewmats, viewmats_rs);
    }

    // Dispatch on the coeff dtype (fp16/fp32). v_coeffs is written in the coeff
    // dtype after fp32 register accumulation; v_colors uses opmath_t (float).
    AT_DISPATCH_V2(
        coeffs.scalar_type(),
        "spherical_beta_bwd_kernel",
        AT_WRAP(
            [&]()
            {
                using opmath_t  = at::opmath_type<scalar_t>;
                auto *masks_ptr = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;

                spherical_beta_bwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
                    coefficient_rows,
                    B,
                    C,
                    N,
                    M,
                    lobes_to_use,
                    means.const_data_ptr<float>(),
                    viewmats.const_data_ptr<float>(),
                    camera_offsets.defined() ? camera_offsets.const_data_ptr<float>() : nullptr,
                    coeffs.const_data_ptr<scalar_t>(),
                    masks_ptr,
                    batch_ids.has_value() ? batch_ids.value().const_data_ptr<int64_t>() : nullptr,
                    camera_ids.has_value() ? camera_ids.value().const_data_ptr<int64_t>() : nullptr,
                    gaussian_ids.has_value() ? gaussian_ids.value().const_data_ptr<int64_t>() : nullptr,
                    v_colors.const_data_ptr<opmath_t>(),
                    v_coeffs.data_ptr<scalar_t>(),
                    v_means.has_value() ? v_means.value().data_ptr<float>() : nullptr,
                    needs_v_viewdirs ? v_viewdirs.data_ptr<float>() : nullptr
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        ),
        at::kFloat,
        at::kHalf
    );

    if(needs_v_viewdirs)
    {
        launch_spherical_harmonics_view_direction_vjp_reduction<ViewmatGradientUpdate::Assign>(
            N, 0, N, viewmats, v_viewdirs, v_viewmats, viewmats_rs, batch_ids, camera_ids, v_viewmats_rs
        );
    }
}
} // namespace gsplat
