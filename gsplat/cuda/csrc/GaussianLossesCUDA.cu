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

#include "Config.h"

#if GSPLAT_BUILD_LOSSES

#    include <algorithm>

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAException.h>
#    include <c10/cuda/CUDAStream.h>

namespace gsplat
{
// The four members own independent row counts (heterogeneous regularization
// domains): scales [N_scales, 3], densities [N_densities], z_scales
// [N_z_scales], positions/cuboid_dims [N_oob, 3]. Any count may be zero; a
// single launch covers the largest member and each member guards on its own
// count. Callers whose members share one gaussian cloud simply pass equal
// counts and take exactly the pre-generalization path.
//
// `preactivation` switches the member math to log-space inputs:
//
// - scales/z_scales arrive as log-space pre-activations with exp() fused
//   here. Per-segment weights are folded caller-side as `+log(w)` on the
//   pre-activations; a non-positive weight arrives as a -inf fill, which
//   exp() maps to an exact zero (value and gradient). Folding by
//   multiplication after exp() would instead risk 0 * inf = NaN when a
//   pre-activation overflows. The `+log(w)` fold is exact only for the
//   scale member, whose loss is LINEAR in exp(s); it must NOT be applied
//   to z-scale pre-activations — the threshold is subtracted after exp(),
//   so relu(exp(z + log w) - T) = relu(w * exp(z) - T) != w * relu(exp(z)
//   - T) for any w outside {exact-zero branch, 1}. See
//   gsplat.losses.fold_log_space_weight for the contract.
// - A zero visibility lane takes the same exact-zero branch as a folded
//   non-positive weight (forward and backward), and the backward
//   additionally branches on a zero upstream — both rule out the same
//   inf * 0 = NaN hazard without reading/exponentiating the lane.
// - densities stay post-activation but are treated as signed magnitudes
//   (callers fold per-segment weights by plain multiplication), so |.| is
//   applied to keep exact abs semantics.
//
// With `preactivation == false` every member computes the pre-existing
// post-activation math, bit for bit.

namespace
{
    template<typename scalar_t>
    inline __device__ scalar_t sign_of(scalar_t x)
    {
        return x > scalar_t(0) ? scalar_t(1) : (x < scalar_t(0) ? scalar_t(-1) : scalar_t(0));
    }
} // namespace

// ---------------------------------------------------------------------------
// Forward kernel: fuses scale_reg, density_reg, z_scale_reg, out_of_bound
// ---------------------------------------------------------------------------
template<typename scalar_t>
__global__ void gaussian_losses_fwd_kernel(
    const int64_t N_scales,
    const int64_t N_densities,
    const int64_t N_z_scales,
    const int64_t N_oob,
    const scalar_t *__restrict__ scales,      // [N_scales, 3]
    const scalar_t *__restrict__ densities,   // [N_densities]
    const scalar_t *__restrict__ z_scales,    // [N_z_scales]
    const scalar_t *__restrict__ positions,   // [N_oob, 3]
    const scalar_t *__restrict__ cuboid_dims, // [N_oob, 3]
    const scalar_t *__restrict__ visibility,  // [max(N_scales, N_densities)] or nullptr
    const scalar_t z_scale_threshold,
    const bool preactivation,
    scalar_t *__restrict__ loss_scale,   // [N_scales, 3]
    scalar_t *__restrict__ loss_density, // [N_densities]
    scalar_t *__restrict__ loss_z_scale, // [N_z_scales]
    scalar_t *__restrict__ loss_oob      // [N_oob, 3]
)
{
    const int64_t idx  = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t idx3 = idx * 3;

    // gaussian_scale_reg: abs(scales) * visibility
    if(idx < N_scales)
    {
        const scalar_t vis = (visibility != nullptr) ? visibility[idx] : static_cast<scalar_t>(1);
        if(preactivation)
        {
            // exp(preact) >= 0, abs is identity; exp(-inf) == 0 realizes a
            // folded non-positive segment weight as an exact zero. A zero
            // visibility lane takes the same exact-zero no-read branch:
            // multiplying after the exp() would risk inf * 0 = NaN when the
            // pre-activation overflows, so never evaluate exp() here.
            if(vis == static_cast<scalar_t>(0))
            {
                loss_scale[idx3 + 0] = static_cast<scalar_t>(0);
                loss_scale[idx3 + 1] = static_cast<scalar_t>(0);
                loss_scale[idx3 + 2] = static_cast<scalar_t>(0);
            }
            else
            {
                loss_scale[idx3 + 0] = exp(scales[idx3 + 0]) * vis;
                loss_scale[idx3 + 1] = exp(scales[idx3 + 1]) * vis;
                loss_scale[idx3 + 2] = exp(scales[idx3 + 2]) * vis;
            }
        }
        else
        {
            // Since scales are post-activation (>= 0), abs is identity.
            loss_scale[idx3 + 0] = scales[idx3 + 0] * vis;
            loss_scale[idx3 + 1] = scales[idx3 + 1] * vis;
            loss_scale[idx3 + 2] = scales[idx3 + 2] * vis;
        }
    }

    // gaussian_density_reg: abs(densities) * visibility
    if(idx < N_densities)
    {
        const scalar_t vis     = (visibility != nullptr) ? visibility[idx] : static_cast<scalar_t>(1);
        const scalar_t density = densities[idx];
        // Post-activation contract (>= 0) makes abs an identity on the legacy
        // path; pre-activation mode applies it so folded signed weights keep
        // exact abs semantics.
        loss_density[idx]      = (preactivation ? abs(density) : density) * vis;
    }

    // gaussian_z_scale_reg: relu(z_scales - threshold)
    if(idx < N_z_scales)
    {
        const scalar_t z      = preactivation ? exp(z_scales[idx]) : z_scales[idx];
        const scalar_t z_diff = z - z_scale_threshold;
        loss_z_scale[idx]     = z_diff > static_cast<scalar_t>(0) ? z_diff : static_cast<scalar_t>(0);
    }

    // out_of_bound_loss: relu(|positions| - cuboid_dims / 2)
    if(idx < N_oob)
    {
        for(int d = 0; d < 3; d++)
        {
            const scalar_t abs_pos  = abs(positions[idx3 + d]);
            const scalar_t half_dim = cuboid_dims[idx3 + d] * static_cast<scalar_t>(0.5);
            const scalar_t diff     = abs_pos - half_dim;
            loss_oob[idx3 + d]      = diff > static_cast<scalar_t>(0) ? diff : static_cast<scalar_t>(0);
        }
    }
}

// ---------------------------------------------------------------------------
// Backward kernel
// ---------------------------------------------------------------------------
template<typename scalar_t>
__global__ void gaussian_losses_bwd_kernel(
    const int64_t N_scales,
    const int64_t N_densities,
    const int64_t N_z_scales,
    const int64_t N_oob,
    const scalar_t *__restrict__ scales,      // [N_scales, 3]
    const scalar_t *__restrict__ densities,   // [N_densities]
    const scalar_t *__restrict__ z_scales,    // [N_z_scales]
    const scalar_t *__restrict__ positions,   // [N_oob, 3]
    const scalar_t *__restrict__ cuboid_dims, // [N_oob, 3]
    const scalar_t *__restrict__ visibility,  // [max(N_scales, N_densities)] or nullptr
    const scalar_t z_scale_threshold,
    const bool preactivation,
    const scalar_t *__restrict__ v_loss_scale,   // [N_scales, 3]
    const scalar_t *__restrict__ v_loss_density, // [N_densities]
    const scalar_t *__restrict__ v_loss_z_scale, // [N_z_scales]
    const scalar_t *__restrict__ v_loss_oob,     // [N_oob, 3]
    scalar_t *__restrict__ v_scales,             // [N_scales, 3]
    scalar_t *__restrict__ v_densities,          // [N_densities]
    scalar_t *__restrict__ v_z_scales,           // [N_z_scales]
    scalar_t *__restrict__ v_positions           // [N_oob, 3]
)
{
    const int64_t idx  = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t idx3 = idx * 3;

    if(idx < N_scales)
    {
        const scalar_t vis = (visibility != nullptr) ? visibility[idx] : static_cast<scalar_t>(1);
        if(preactivation)
        {
            // d(exp(s) * vis)/ds = exp(s) * vis; exp(-inf) == 0 keeps folded
            // non-positive segment weights at an exact-zero gradient. exp()
            // may overflow to inf, so a zero factor — a zero visibility lane
            // or a zero upstream (e.g. a None cotangent normalized to zeros,
            // or a lane the objective masked out) — must branch to an exact
            // zero without evaluating exp(): inf * 0 = NaN otherwise.
            if(vis == static_cast<scalar_t>(0))
            {
                v_scales[idx3 + 0] = static_cast<scalar_t>(0);
                v_scales[idx3 + 1] = static_cast<scalar_t>(0);
                v_scales[idx3 + 2] = static_cast<scalar_t>(0);
            }
            else
            {
                for(int d = 0; d < 3; d++)
                {
                    const scalar_t up  = v_loss_scale[idx3 + d];
                    v_scales[idx3 + d] = (up == static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0)
                                                                          : exp(scales[idx3 + d]) * vis * up;
                }
            }
        }
        else
        {
            // d(scale_reg)/d(scales) = vis, relying on the post-activation
            // contract (scales >= 0 ⇒ forward is scales * vis, no abs). This
            // keeps fwd/bwd consistent; if a caller violates the contract the
            // result is wrong in the same direction on both paths.
            v_scales[idx3 + 0] = v_loss_scale[idx3 + 0] * vis;
            v_scales[idx3 + 1] = v_loss_scale[idx3 + 1] * vis;
            v_scales[idx3 + 2] = v_loss_scale[idx3 + 2] * vis;
        }
    }

    if(idx < N_densities)
    {
        const scalar_t vis = (visibility != nullptr) ? visibility[idx] : static_cast<scalar_t>(1);
        // d(|d| * vis)/dd = sign(d) * vis (sign(0) == 0, matching abs's
        // subgradient choice in the reference); the legacy path keeps the
        // post-activation contract d/dd = vis.
        v_densities[idx]
            = preactivation ? sign_of(densities[idx]) * vis * v_loss_density[idx] : v_loss_density[idx] * vis;
    }

    // d(z_scale_reg)/d(z) = (z_post > threshold) ? d(z_post)/dz : 0, with
    // z_post = exp(z) in pre-activation mode (d(z_post)/dz = exp(z)) and
    // z_post = z otherwise (d(z_post)/dz = 1).
    if(idx < N_z_scales)
    {
        if(preactivation)
        {
            // Same zero-upstream gate as the scale member: exp(z) can
            // overflow to inf and inf * 0 = NaN, so a zero upstream takes an
            // exact-zero branch without evaluating exp().
            const scalar_t up = v_loss_z_scale[idx];
            if(up == static_cast<scalar_t>(0))
            {
                v_z_scales[idx] = static_cast<scalar_t>(0);
            }
            else
            {
                const scalar_t z = exp(z_scales[idx]);
                v_z_scales[idx]  = (z > z_scale_threshold) ? z * up : static_cast<scalar_t>(0);
            }
        }
        else
        {
            const scalar_t z_diff = z_scales[idx] - z_scale_threshold;
            v_z_scales[idx] = (z_diff > static_cast<scalar_t>(0)) ? v_loss_z_scale[idx] : static_cast<scalar_t>(0);
        }
    }

    // d(oob)/d(positions) = (|pos| > half_dim) ? sign(pos) : 0
    if(idx < N_oob)
    {
        for(int d = 0; d < 3; d++)
        {
            const scalar_t pos      = positions[idx3 + d];
            const scalar_t abs_pos  = abs(pos);
            const scalar_t half_dim = cuboid_dims[idx3 + d] * static_cast<scalar_t>(0.5);
            if(abs_pos > half_dim)
            {
                const scalar_t sign
                    = (pos > static_cast<scalar_t>(0)) ? static_cast<scalar_t>(1) : static_cast<scalar_t>(-1);
                v_positions[idx3 + d] = v_loss_oob[idx3 + d] * sign;
            }
            else
            {
                v_positions[idx3 + d] = static_cast<scalar_t>(0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

void launch_gaussian_losses_fwd_kernel(
    const at::Tensor &scales,
    const at::Tensor &densities,
    const at::Tensor &z_scales,
    const at::Tensor &positions,
    const at::Tensor &cuboid_dims,
    const at::Tensor *visibility,
    float z_scale_threshold,
    bool preactivation,
    at::Tensor &loss_scale,
    at::Tensor &loss_density,
    at::Tensor &loss_z_scale,
    at::Tensor &loss_oob
)
{
    // Public contract: a negative threshold would make relu(0 - T) > 0, so a
    // disabled (w <= 0 folded) or empty z-scale member would no longer be an
    // exact zero.
    TORCH_CHECK(z_scale_threshold >= 0.0f, "z_scale_threshold must be non-negative, got ", z_scale_threshold);
    const int64_t N_scales    = scales.size(0);
    const int64_t N_densities = densities.size(0);
    const int64_t N_z_scales  = z_scales.size(0);
    const int64_t N_oob       = positions.size(0);
    const int64_t N           = std::max(std::max(N_scales, N_densities), std::max(N_z_scales, N_oob));
    if(N == 0)
    {
        return;
    }

    dim3 threads(256);
    dim3 grid(static_cast<uint32_t>((N + threads.x - 1) / threads.x));

    AT_DISPATCH_FLOATING_TYPES(
        scales.scalar_type(),
        "gaussian_losses_fwd",
        [&]()
        {
            const scalar_t *vis_ptr = (visibility != nullptr) ? visibility->data_ptr<scalar_t>() : nullptr;
            gaussian_losses_fwd_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                N_scales,
                N_densities,
                N_z_scales,
                N_oob,
                scales.data_ptr<scalar_t>(),
                densities.data_ptr<scalar_t>(),
                z_scales.data_ptr<scalar_t>(),
                positions.data_ptr<scalar_t>(),
                cuboid_dims.data_ptr<scalar_t>(),
                vis_ptr,
                static_cast<scalar_t>(z_scale_threshold),
                preactivation,
                loss_scale.data_ptr<scalar_t>(),
                loss_density.data_ptr<scalar_t>(),
                loss_z_scale.data_ptr<scalar_t>(),
                loss_oob.data_ptr<scalar_t>()
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gaussian_losses_bwd_kernel(
    const at::Tensor &scales,
    const at::Tensor &densities,
    const at::Tensor &z_scales,
    const at::Tensor &positions,
    const at::Tensor &cuboid_dims,
    const at::Tensor *visibility,
    float z_scale_threshold,
    bool preactivation,
    const at::Tensor &v_loss_scale,
    const at::Tensor &v_loss_density,
    const at::Tensor &v_loss_z_scale,
    const at::Tensor &v_loss_oob,
    at::Tensor &v_scales,
    at::Tensor &v_densities,
    at::Tensor &v_z_scales,
    at::Tensor &v_positions
)
{
    // Public contract: a negative threshold would make relu(0 - T) > 0, so a
    // disabled (w <= 0 folded) or empty z-scale member would no longer be an
    // exact zero.
    TORCH_CHECK(z_scale_threshold >= 0.0f, "z_scale_threshold must be non-negative, got ", z_scale_threshold);
    const int64_t N_scales    = scales.size(0);
    const int64_t N_densities = densities.size(0);
    const int64_t N_z_scales  = z_scales.size(0);
    const int64_t N_oob       = positions.size(0);
    const int64_t N           = std::max(std::max(N_scales, N_densities), std::max(N_z_scales, N_oob));
    if(N == 0)
    {
        return;
    }

    dim3 threads(256);
    dim3 grid(static_cast<uint32_t>((N + threads.x - 1) / threads.x));

    AT_DISPATCH_FLOATING_TYPES(
        scales.scalar_type(),
        "gaussian_losses_bwd",
        [&]()
        {
            const scalar_t *vis_ptr = (visibility != nullptr) ? visibility->data_ptr<scalar_t>() : nullptr;
            gaussian_losses_bwd_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                N_scales,
                N_densities,
                N_z_scales,
                N_oob,
                scales.data_ptr<scalar_t>(),
                densities.data_ptr<scalar_t>(),
                z_scales.data_ptr<scalar_t>(),
                positions.data_ptr<scalar_t>(),
                cuboid_dims.data_ptr<scalar_t>(),
                vis_ptr,
                static_cast<scalar_t>(z_scale_threshold),
                preactivation,
                v_loss_scale.data_ptr<scalar_t>(),
                v_loss_density.data_ptr<scalar_t>(),
                v_loss_z_scale.data_ptr<scalar_t>(),
                v_loss_oob.data_ptr<scalar_t>(),
                v_scales.data_ptr<scalar_t>(),
                v_densities.data_ptr<scalar_t>(),
                v_z_scales.data_ptr<scalar_t>(),
                v_positions.data_ptr<scalar_t>()
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace gsplat

#endif
