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

#include "GSplatBuildConfig.h"

#if GSPLAT_BUILD_LOSSES

#    include <algorithm>
#    include <initializer_list>
#    include <utility>

#    include <ATen/Functions.h>
#    include <ATen/TensorUtils.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAGuard.h>
#    include <torch/library.h>

#    include "GaussianLosses.h"
#    include "DeformLosses.h"
#    include "Common.h"

namespace gsplat
{
// Output tensors are allocated by the caller (Python autograd wrapper) and
// passed in as mutable arguments — keeps memory lifetime explicit on the
// Python side so it can be reused by torch's caching allocator across training
// steps.
//
// The gaussian dispatch is grouped: one host entry
// launching a group of kernels. Besides the four per-gaussian regularizers,
// the entry optionally launches the deform-smoothness kernels (implemented in
// DeformLosses.*, shipped with this dispatch and reached through its launch
// helpers) when the optional deform arguments are present.
//
// The four regularizer members own independent row counts (heterogeneous
// regularization domains): scales [N_scales, 3], densities [N_densities],
// z_scales [N_z_scales], positions/cuboid_dims [N_oob, 3]. Any count may be
// zero; callers whose members share one gaussian cloud pass equal counts and
// take exactly the pre-generalization path. `visibility` spans the scale and
// density members, so it must be [max(N_scales, N_densities)] when present.
//
// The trailing `preactivation` flag (schema default false) switches the
// member math to log-space inputs: scales/z_scales arrive as log-space
// pre-activations with exp() fused in-kernel (per-segment weights fold
// caller-side as +log(w); non-positive weights arrive as -inf fills that
// exp() maps to exact zeros), and densities are treated as signed magnitudes
// (|.| applied). The +log(w) fold is valid ONLY for the scale member (linear
// in exp(s)); it must NOT be pre-applied to z-scale inputs — the z member
// subtracts its threshold after the exp(), so relu(exp(z + log w) - T) =
// relu(w * exp(z) - T) != w * relu(exp(z) - T). The op has no segment-weight
// arguments for z; see gsplat.losses.fold_log_space_weight for the caller
// contract. With the flag off, every member computes the post-activation
// math bit for bit.

namespace
{
    // Presence contract shared by the fwd/bwd entries: the deform group member
    // rides along iff `deformation` is present, in which case its companion
    // tensors must all be present too (and the mask only ever accompanies an
    // enabled deform member).
    inline void check_deform_group_presence(
        bool has_deformation, bool has_mask, std::initializer_list<std::pair<bool, const char *>> companions
    )
    {
        for(const auto &companion: companions)
        {
            TORCH_CHECK(
                companion.first == has_deformation,
                "deformation and ",
                companion.second,
                " must either both be present or both be absent"
            );
        }
        TORCH_CHECK(has_deformation || !has_mask, "deform_mask requires deformation to be present");
    }
} // namespace

void gaussian_losses_fwd(
    const at::Tensor &scales,
    const at::Tensor &densities,
    const at::Tensor &z_scales,
    const at::Tensor &positions,
    const at::Tensor &cuboid_dims,
    const at::optional<at::Tensor> &visibility,
    double z_scale_threshold,
    at::Tensor loss_scale,
    at::Tensor loss_density,
    at::Tensor loss_z_scale,
    at::Tensor loss_oob,
    const at::optional<at::Tensor> &deformation,
    const at::optional<at::Tensor> &deform_mask,
    at::optional<at::Tensor> deform_sums,
    at::optional<at::Tensor> deform_loss,
    bool preactivation
)
{
    DEVICE_GUARD(scales);
    CHECK_INPUT(scales);
    CHECK_INPUT(densities);
    CHECK_INPUT(z_scales);
    CHECK_INPUT(positions);
    CHECK_INPUT(cuboid_dims);
    CHECK_INPUT(loss_scale);
    CHECK_INPUT(loss_density);
    CHECK_INPUT(loss_z_scale);
    CHECK_INPUT(loss_oob);

    // Independent per-member row counts; only positions/cuboid_dims share one.
    const int64_t N_scales    = scales.size(0);
    const int64_t N_densities = densities.size(0);
    const int64_t N_oob       = positions.size(0);
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3, "scales must be [N_scales, 3]");
    TORCH_CHECK(densities.dim() == 1, "densities must be [N_densities]");
    TORCH_CHECK(z_scales.dim() == 1, "z_scales must be [N_z_scales]");
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3, "positions must be [N_oob, 3]");
    TORCH_CHECK(
        cuboid_dims.dim() == 2 && cuboid_dims.size(0) == N_oob && cuboid_dims.size(1) == 3,
        "cuboid_dims must be [N_oob, 3] matching positions"
    );
    TORCH_CHECK(
        loss_scale.sizes() == scales.sizes(),
        "loss_scale must have same shape as scales, got ",
        loss_scale.sizes(),
        " vs ",
        scales.sizes()
    );
    TORCH_CHECK(
        loss_density.sizes() == densities.sizes(),
        "loss_density must have same shape as densities, got ",
        loss_density.sizes(),
        " vs ",
        densities.sizes()
    );
    TORCH_CHECK(
        loss_z_scale.sizes() == z_scales.sizes(),
        "loss_z_scale must have same shape as z_scales, got ",
        loss_z_scale.sizes(),
        " vs ",
        z_scales.sizes()
    );
    TORCH_CHECK(
        loss_oob.sizes() == positions.sizes(),
        "loss_oob must have same shape as positions, got ",
        loss_oob.sizes(),
        " vs ",
        positions.sizes()
    );

    const at::Tensor *vis_ptr = nullptr;
    if(visibility.has_value())
    {
        CHECK_INPUT(visibility.value());
        TORCH_CHECK(
            visibility.value().dim() == 1 && visibility.value().size(0) == std::max(N_scales, N_densities),
            "visibility must be [max(N_scales, N_densities)]"
        );
        vis_ptr = &visibility.value();
    }

    // Invariant: an early return for empty gaussian work must not skip
    // later group members' launches. The gaussian launch helper returns early
    // internally when every member count is 0, so control always reaches the
    // deform member below — the deform launch is gated only on argument
    // presence, never on the gaussian point counts.
    launch_gaussian_losses_fwd_kernel(
        scales,
        densities,
        z_scales,
        positions,
        cuboid_dims,
        vis_ptr,
        static_cast<float>(z_scale_threshold),
        preactivation,
        loss_scale,
        loss_density,
        loss_z_scale,
        loss_oob
    );

    // --- Grouped deform-smoothness member ----------------------------------
    // Reuses the deform port's launch helpers and validation contract
    // (DeformLosses.*), sharing this entry's device guard and current stream.
    check_deform_group_presence(
        deformation.has_value(),
        deform_mask.has_value(),
        {
            {deform_sums.has_value(), "deform_sums"},
            {deform_loss.has_value(), "deform_loss"}
    }
    );
    if(deformation.has_value())
    {
        const at::Tensor &deform = deformation.value();
        at::Tensor &sums         = deform_sums.value();
        at::Tensor &loss         = deform_loss.value();
        CHECK_INPUT(deform);
        CHECK_INPUT(sums);
        CHECK_INPUT(loss);
        TORCH_CHECK(deform.dim() == 2, "deformation must be [N, D], got ", deform.sizes());
        TORCH_CHECK(sums.numel() == 2, "deform_sums must be a [2] tensor, got ", sums.sizes());
        TORCH_CHECK(loss.numel() == 1, "deform_loss must be a scalar tensor, got ", loss.sizes());
        TORCH_CHECK(
            deform.device() == scales.device(),
            "deformation must be on the same device as scales (",
            scales.device(),
            "), got ",
            deform.device()
        );
        deform_losses_check_matches_deformation(deform, sums, "deform_sums");
        deform_losses_check_matches_deformation(deform, loss, "deform_loss");

        const at::Tensor *mask_ptr = nullptr;
        int64_t stride_row         = 0;
        int64_t stride_col         = 0;
        if(deform_mask.has_value())
        {
            CHECK_INPUT(deform_mask.value());
            deform_losses_check_matches_deformation(deform, deform_mask.value(), "deform_mask");
            deform_losses_resolve_mask_broadcast(deform, deform_mask.value(), stride_row, stride_col);
            mask_ptr = &deform_mask.value();
        }

        launch_deform_losses_fwd_kernel(deform, mask_ptr, stride_row, stride_col, sums, loss);
    }
}

void gaussian_losses_bwd(
    const at::Tensor &scales,
    const at::Tensor &densities,
    const at::Tensor &z_scales,
    const at::Tensor &positions,
    const at::Tensor &cuboid_dims,
    const at::optional<at::Tensor> &visibility,
    double z_scale_threshold,
    const at::Tensor &v_loss_scale,
    const at::Tensor &v_loss_density,
    const at::Tensor &v_loss_z_scale,
    const at::Tensor &v_loss_oob,
    at::Tensor v_scales,
    at::Tensor v_densities,
    at::Tensor v_z_scales,
    at::Tensor v_positions,
    const at::optional<at::Tensor> &deformation,
    const at::optional<at::Tensor> &deform_mask,
    const at::optional<at::Tensor> &deform_sums,
    const at::optional<at::Tensor> &v_deform_loss,
    at::optional<at::Tensor> v_deformation,
    at::optional<at::Tensor> v_deform_mask,
    bool preactivation
)
{
    DEVICE_GUARD(scales);
    CHECK_INPUT(scales);
    CHECK_INPUT(densities);
    CHECK_INPUT(z_scales);
    CHECK_INPUT(positions);
    CHECK_INPUT(cuboid_dims);
    CHECK_INPUT(v_loss_scale);
    CHECK_INPUT(v_loss_density);
    CHECK_INPUT(v_loss_z_scale);
    CHECK_INPUT(v_loss_oob);
    CHECK_INPUT(v_scales);
    CHECK_INPUT(v_densities);
    CHECK_INPUT(v_z_scales);
    CHECK_INPUT(v_positions);
    TORCH_CHECK(
        v_loss_scale.sizes() == scales.sizes(),
        "v_loss_scale must have same shape as scales, got ",
        v_loss_scale.sizes(),
        " vs ",
        scales.sizes()
    );
    TORCH_CHECK(
        v_loss_density.sizes() == densities.sizes(),
        "v_loss_density must have same shape as densities, got ",
        v_loss_density.sizes(),
        " vs ",
        densities.sizes()
    );
    TORCH_CHECK(
        v_loss_z_scale.sizes() == z_scales.sizes(),
        "v_loss_z_scale must have same shape as z_scales, got ",
        v_loss_z_scale.sizes(),
        " vs ",
        z_scales.sizes()
    );
    TORCH_CHECK(
        v_loss_oob.sizes() == positions.sizes(),
        "v_loss_oob must have same shape as positions, got ",
        v_loss_oob.sizes(),
        " vs ",
        positions.sizes()
    );
    TORCH_CHECK(
        v_scales.sizes() == scales.sizes(),
        "v_scales must have same shape as scales, got ",
        v_scales.sizes(),
        " vs ",
        scales.sizes()
    );
    TORCH_CHECK(
        v_densities.sizes() == densities.sizes(),
        "v_densities must have same shape as densities, got ",
        v_densities.sizes(),
        " vs ",
        densities.sizes()
    );
    TORCH_CHECK(
        v_z_scales.sizes() == z_scales.sizes(),
        "v_z_scales must have same shape as z_scales, got ",
        v_z_scales.sizes(),
        " vs ",
        z_scales.sizes()
    );
    TORCH_CHECK(
        v_positions.sizes() == positions.sizes(),
        "v_positions must have same shape as positions, got ",
        v_positions.sizes(),
        " vs ",
        positions.sizes()
    );

    const at::Tensor *vis_ptr = nullptr;
    if(visibility.has_value())
    {
        CHECK_INPUT(visibility.value());
        TORCH_CHECK(
            visibility.value().dim() == 1 && visibility.value().size(0) == std::max(scales.size(0), densities.size(0)),
            "visibility must be [max(N_scales, N_densities)]"
        );
        vis_ptr = &visibility.value();
    }

    // Same invariant as the forward: the gaussian launch helper's internal
    // all-members-empty early return must not — and does not — skip the
    // deform group member's backward launch below.
    launch_gaussian_losses_bwd_kernel(
        scales,
        densities,
        z_scales,
        positions,
        cuboid_dims,
        vis_ptr,
        static_cast<float>(z_scale_threshold),
        preactivation,
        v_loss_scale,
        v_loss_density,
        v_loss_z_scale,
        v_loss_oob,
        v_scales,
        v_densities,
        v_z_scales,
        v_positions
    );

    // --- Grouped deform-smoothness member ----------------------------------
    // Mirrors the forward member's validation contract before reusing the
    // deform backward launch helper on this entry's device guard and current
    // stream.
    check_deform_group_presence(
        deformation.has_value(),
        deform_mask.has_value(),
        {
            {  deform_sums.has_value(),   "deform_sums"},
            {v_deform_loss.has_value(), "v_deform_loss"},
            {v_deformation.has_value(), "v_deformation"}
    }
    );
    TORCH_CHECK(
        deform_mask.has_value() == v_deform_mask.has_value(),
        "deform_mask and v_deform_mask must either both be present or both be absent"
    );
    if(deformation.has_value())
    {
        const at::Tensor &deform = deformation.value();
        const at::Tensor &sums   = deform_sums.value();
        const at::Tensor &v_loss = v_deform_loss.value();
        at::Tensor &v_deform     = v_deformation.value();
        CHECK_INPUT(deform);
        CHECK_INPUT(sums);
        CHECK_INPUT(v_loss);
        CHECK_INPUT(v_deform);
        TORCH_CHECK(deform.dim() == 2, "deformation must be [N, D], got ", deform.sizes());
        TORCH_CHECK(sums.numel() == 2, "deform_sums must be a [2] tensor, got ", sums.sizes());
        TORCH_CHECK(v_loss.numel() == 1, "v_deform_loss must be a scalar tensor, got ", v_loss.sizes());
        TORCH_CHECK(
            v_deform.sizes() == deform.sizes(),
            "v_deformation must have same shape as deformation, got ",
            v_deform.sizes(),
            " vs ",
            deform.sizes()
        );
        TORCH_CHECK(
            deform.device() == scales.device(),
            "deformation must be on the same device as scales (",
            scales.device(),
            "), got ",
            deform.device()
        );
        deform_losses_check_matches_deformation(deform, sums, "deform_sums");
        deform_losses_check_matches_deformation(deform, v_loss, "v_deform_loss");
        deform_losses_check_matches_deformation(deform, v_deform, "v_deformation");

        const at::Tensor *mask_ptr = nullptr;
        at::Tensor *v_mask_ptr     = nullptr;
        int64_t stride_row         = 0;
        int64_t stride_col         = 0;
        if(deform_mask.has_value())
        {
            CHECK_INPUT(deform_mask.value());
            CHECK_INPUT(v_deform_mask.value());
            deform_losses_check_matches_deformation(deform, deform_mask.value(), "deform_mask");
            deform_losses_check_matches_deformation(deform, v_deform_mask.value(), "v_deform_mask");
            deform_losses_resolve_mask_broadcast(deform, deform_mask.value(), stride_row, stride_col);
            TORCH_CHECK(
                v_deform_mask.value().sizes() == deform_mask.value().sizes(),
                "v_deform_mask must have same shape as deform_mask, got ",
                v_deform_mask.value().sizes(),
                " vs ",
                deform_mask.value().sizes()
            );
            mask_ptr   = &deform_mask.value();
            v_mask_ptr = &v_deform_mask.value();
        }

        launch_deform_losses_bwd_kernel(deform, mask_ptr, stride_row, stride_col, sums, v_loss, v_deform, v_mask_ptr);
    }
}

void register_gaussian_losses_cuda_impl(torch::Library &m)
{
    m.impl("gaussian_losses_fwd", &gaussian_losses_fwd);
    m.impl("gaussian_losses_bwd", &gaussian_losses_bwd);
}
} // namespace gsplat

#endif
