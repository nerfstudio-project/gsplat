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

#    include <ATen/Functions.h>
#    include <ATen/TensorUtils.h>
#    include <ATen/core/Tensor.h>
#    include <torch/library.h>

#    include "Common.h"
#    include "SsimLosses.h"

namespace gsplat
{
namespace
{
    void check_float32_same_device(const char *name, const at::Tensor &primary, const at::Tensor &other)
    {
        TORCH_CHECK(other.scalar_type() == at::kFloat, name, " must be float32, got ", other.scalar_type());
        TORCH_CHECK(
            other.device() == primary.device(), name, " must be on ", primary.device(), ", got ", other.device()
        );
    }

    void check_ssim_common(
        const at::Tensor &flags,
        const at::Tensor &pred,
        const at::Tensor &target,
        const at::Tensor &loss,
        const at::Tensor &dm_dmu1,
        const at::Tensor &dm_dsigma1_sq,
        const at::Tensor &dm_dsigma12
    )
    {
        TORCH_CHECK(pred.dim() == 4, "pred must be [B, H, W, C], got ", pred.sizes());
        const int64_t B = pred.size(0);
        const int64_t H = pred.size(1);
        const int64_t W = pred.size(2);
        const int64_t C = pred.size(3);

        TORCH_CHECK(pred.scalar_type() == at::kFloat, "pred must be float32, got ", pred.scalar_type());
        TORCH_CHECK(
            target.sizes() == pred.sizes(),
            "target must have same shape as pred, got ",
            target.sizes(),
            " vs ",
            pred.sizes()
        );
        check_float32_same_device("target", pred, target);

        TORCH_CHECK(flags.dim() == 3, "flags must be [B, H, W], got ", flags.sizes());
        TORCH_CHECK(flags.scalar_type() == at::kInt, "flags must be int32, got ", flags.scalar_type());
        TORCH_CHECK(flags.device() == pred.device(), "flags must be on ", pred.device(), ", got ", flags.device());
        TORCH_CHECK(flags.size(0) == B && flags.size(1) == H && flags.size(2) == W, "flags must be [B, H, W]");

        TORCH_CHECK(
            loss.dim() == 4 && loss.size(0) == B && loss.size(1) == H && loss.size(2) == W && loss.size(3) == 1,
            "loss must be [B, H, W, 1], got ",
            loss.sizes()
        );
        check_float32_same_device("loss", pred, loss);

        TORCH_CHECK(
            dm_dmu1.dim() == 4
                && dm_dmu1.size(0) == B
                && dm_dmu1.size(1) == C
                && dm_dmu1.size(2) == H
                && dm_dmu1.size(3) == W,
            "dm_dmu1 must be [B, C, H, W], got ",
            dm_dmu1.sizes()
        );
        TORCH_CHECK(
            dm_dsigma1_sq.dim() == 4
                && dm_dsigma1_sq.size(0) == B
                && dm_dsigma1_sq.size(1) == C
                && dm_dsigma1_sq.size(2) == H
                && dm_dsigma1_sq.size(3) == W,
            "dm_dsigma1_sq must be [B, C, H, W], got ",
            dm_dsigma1_sq.sizes()
        );
        TORCH_CHECK(
            dm_dsigma12.dim() == 4
                && dm_dsigma12.size(0) == B
                && dm_dsigma12.size(1) == C
                && dm_dsigma12.size(2) == H
                && dm_dsigma12.size(3) == W,
            "dm_dsigma12 must be [B, C, H, W], got ",
            dm_dsigma12.sizes()
        );
        check_float32_same_device("dm_dmu1", pred, dm_dmu1);
        check_float32_same_device("dm_dsigma1_sq", pred, dm_dsigma1_sq);
        check_float32_same_device("dm_dsigma12", pred, dm_dsigma12);
    }
} // namespace

void ssim_losses_fwd(
    const at::Tensor &flags,
    const at::Tensor &pred,
    const at::Tensor &target,
    double factor,
    bool mask_mode_target,
    double constant_mask_value,
    at::Tensor loss,
    at::Tensor dm_dmu1,
    at::Tensor dm_dsigma1_sq,
    at::Tensor dm_dsigma12
)
{
    DEVICE_GUARD(pred);
    CHECK_INPUT(flags);
    CHECK_INPUT(pred);
    CHECK_INPUT(target);
    CHECK_INPUT(loss);
    CHECK_INPUT(dm_dmu1);
    CHECK_INPUT(dm_dsigma1_sq);
    CHECK_INPUT(dm_dsigma12);
    check_ssim_common(flags, pred, target, loss, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);

    launch_ssim_losses_fwd_kernel(
        flags,
        pred,
        target,
        static_cast<float>(factor),
        mask_mode_target,
        static_cast<float>(constant_mask_value),
        loss,
        dm_dmu1,
        dm_dsigma1_sq,
        dm_dsigma12
    );
}

void ssim_losses_bwd(
    const at::Tensor &flags,
    const at::Tensor &pred,
    const at::Tensor &target,
    double factor,
    bool mask_mode_target,
    double constant_mask_value,
    const at::Tensor &v_loss,
    const at::Tensor &dm_dmu1,
    const at::Tensor &dm_dsigma1_sq,
    const at::Tensor &dm_dsigma12,
    at::Tensor v_pred
)
{
    DEVICE_GUARD(pred);
    CHECK_INPUT(flags);
    CHECK_INPUT(pred);
    CHECK_INPUT(target);
    CHECK_INPUT(v_loss);
    CHECK_INPUT(dm_dmu1);
    CHECK_INPUT(dm_dsigma1_sq);
    CHECK_INPUT(dm_dsigma12);
    CHECK_INPUT(v_pred);
    check_ssim_common(flags, pred, target, v_loss, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
    TORCH_CHECK(v_pred.sizes() == pred.sizes(), "v_pred must have same shape as pred, got ", v_pred.sizes());
    check_float32_same_device("v_pred", pred, v_pred);

    launch_ssim_losses_bwd_kernel(
        flags,
        pred,
        target,
        static_cast<float>(factor),
        mask_mode_target,
        static_cast<float>(constant_mask_value),
        v_loss,
        dm_dmu1,
        dm_dsigma1_sq,
        dm_dsigma12,
        v_pred
    );
}

void register_ssim_losses_cuda_impl(torch::Library &m)
{
    m.impl("ssim_losses_fwd", &ssim_losses_fwd);
    m.impl("ssim_losses_bwd", &ssim_losses_bwd);
}
} // namespace gsplat

#endif
