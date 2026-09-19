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

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "Common.h"        // where all the macros are defined
#include "SphericalBeta.h" // where the launch function is declared
#include "TorchUtils.h"

namespace gsplat
{
namespace
{
    void check_spherical_beta_inputs(
        int64_t lobes_to_use,
        const at::Tensor &means,
        const at::Tensor &viewmats,
        const at::Tensor &coeffs,
        const at::optional<at::Tensor> &masks,
        const at::optional<at::Tensor> &batch_ids,
        const at::optional<at::Tensor> &camera_ids,
        const at::optional<at::Tensor> &gaussian_ids,
        const at::optional<at::Tensor> &viewmats_rs = c10::nullopt
    )
    {
        TORCH_CHECK(
            lobes_to_use >= 0 && lobes_to_use <= SB_MAX_LOBES,
            "lobes_to_use must be between 0 and ",
            SB_MAX_LOBES,
            ", got ",
            lobes_to_use
        );
        TORCH_CHECK(means.dim() >= 2 && means.size(-1) == 3, "means must have shape [..., N, 3], got ", means.sizes());
        TORCH_CHECK(
            viewmats.dim() == means.dim() + 1 && viewmats.size(-2) == 4 && viewmats.size(-1) == 4,
            "viewmats must have shape [..., C, 4, 4], got ",
            viewmats.sizes()
        );
        TORCH_CHECK(
            means.sizes().slice(0, means.dim() - 2) == viewmats.sizes().slice(0, viewmats.dim() - 3),
            "means and viewmats batch dimensions must match"
        );
        TORCH_CHECK(means.scalar_type() == at::kFloat, "means must be float32");
        TORCH_CHECK(viewmats.scalar_type() == at::kFloat, "viewmats must be float32");
        TORCH_CHECK(coeffs.dim() == 3, "coeffs must have shape [N, M, 6] or [nnz, M, 6], got ", coeffs.sizes());
        TORCH_CHECK(
            coeffs.size(-1) == SB_LOBE_WIDTH,
            "coeffs last dim must be ",
            SB_LOBE_WIDTH,
            " ([r, g, b, theta, phi, beta]), got ",
            coeffs.size(-1)
        );
        TORCH_CHECK(
            lobes_to_use <= coeffs.size(-2),
            "lobes_to_use requires more lobes than provided; lobes_to_use ",
            lobes_to_use,
            ", coeffs shape ",
            coeffs.sizes()
        );
        const bool packed = batch_ids.has_value() || camera_ids.has_value() || gaussian_ids.has_value();
        TORCH_CHECK(
            !packed || (batch_ids.has_value() && camera_ids.has_value() && gaussian_ids.has_value()),
            "batch_ids, camera_ids, and gaussian_ids must either all be provided or all be None"
        );
        if(packed)
        {
            const int64_t nnz = coeffs.size(0);
            for(const auto &ids: {batch_ids.value(), camera_ids.value(), gaussian_ids.value()})
            {
                TORCH_CHECK(ids.dim() == 1 && ids.numel() == nnz, "packed ID tensors must have shape [nnz]");
                TORCH_CHECK(ids.scalar_type() == at::kLong, "packed ID tensors must be int64");
                CHECK_INPUT(ids);
            }
            if(masks.has_value())
            {
                TORCH_CHECK(
                    masks.value().dim() == 1 && masks.value().numel() == nnz, "packed masks must have shape [nnz]"
                );
            }
        }
        else
        {
            TORCH_CHECK(means.size(-2) == coeffs.size(0), "means N must match coeffs N in dense mode");
            if(masks.has_value())
            {
                at::DimVector mask_shape(viewmats.sizes().slice(0, viewmats.dim() - 2));
                mask_shape.push_back(means.size(-2));
                TORCH_CHECK(masks.value().sizes() == mask_shape, "dense masks must have shape [..., C, N]");
            }
        }
        CHECK_INPUT(means);
        CHECK_INPUT(viewmats);
        if(viewmats_rs.has_value())
        {
            TORCH_CHECK(viewmats_rs.value().sizes() == viewmats.sizes(), "viewmats_rs must match viewmats shape");
            TORCH_CHECK(viewmats_rs.value().scalar_type() == at::kFloat, "viewmats_rs must be float32");
            CHECK_INPUT(viewmats_rs.value());
        }
        CHECK_INPUT(coeffs);
        if(masks.has_value())
        {
            CHECK_INPUT(masks.value());
        }
    }

    // The base colour doubles as the output shape and layout: the kernel copies
    // it through and adds the lobes on top, so it must already be broadcast to
    // one row per output element.
    void check_spherical_beta_colors(
        const at::Tensor &colors,
        const at::Tensor &means,
        const at::Tensor &viewmats,
        const bool packed,
        const int64_t rows,
        const char *name
    )
    {
        TORCH_CHECK(colors.scalar_type() == at::kFloat, name, " must be float32");
        if(packed)
        {
            TORCH_CHECK(
                colors.dim() == 2 && colors.size(0) == rows && colors.size(1) == SB_NUM_CHANNELS,
                name,
                " must have shape [nnz, ",
                SB_NUM_CHANNELS,
                "], got ",
                colors.sizes()
            );
        }
        else
        {
            at::DimVector shape(viewmats.sizes().slice(0, viewmats.dim() - 2));
            shape.push_back(means.size(-2));
            shape.push_back(SB_NUM_CHANNELS);
            TORCH_CHECK(
                colors.sizes() == at::IntArrayRef(shape),
                name,
                " must have shape [..., C, N, ",
                SB_NUM_CHANNELS,
                "], got ",
                colors.sizes()
            );
        }
        CHECK_INPUT(colors);
    }
} // namespace

// Spherical Beta
//
// Adds view-dependent specular lobes onto a base colour. Each lobe stores
// [r, g, b, theta, phi, beta]: an amplitude, a direction in spherical
// coordinates, and a log-space sharpness. This is the appearance model of
// "Deformable Beta Splatting" (Liu et al., arXiv:2501.18630) and is an
// alternative to the l>=1 spherical harmonic bands, which it may also be
// combined with.
at::Tensor spherical_beta_fwd(
    int64_t lobes_to_use,
    const at::Tensor &means,       // [..., N, 3]
    const at::Tensor &viewmats,    // [..., C, 4, 4]
    const at::Tensor &base_colors, // [..., C, N, 3] or [nnz, 3]
    const at::Tensor &coeffs,      // [N, M, 6] or [nnz, M, 6]
    const at::optional<at::Tensor> &masks,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    const at::optional<at::Tensor> &viewmats_rs
)
{
    DEVICE_GUARD(means);
    check_spherical_beta_inputs(
        lobes_to_use, means, viewmats, coeffs, masks, batch_ids, camera_ids, gaussian_ids, viewmats_rs
    );
    check_spherical_beta_colors(base_colors, means, viewmats, batch_ids.has_value(), coeffs.size(0), "base_colors");

    at::Tensor colors = at::empty(base_colors.sizes(), base_colors.options());

    launch_spherical_beta_fwd_kernel(
        lobes_to_use,
        means,
        viewmats,
        viewmats_rs,
        base_colors,
        coeffs,
        masks,
        batch_ids,
        camera_ids,
        gaussian_ids,
        colors
    );
    return colors;
}

std::tuple<at::Tensor, at::optional<at::Tensor>, at::optional<at::Tensor>, at::optional<at::Tensor>> spherical_beta_bwd(
    int64_t lobes_to_use,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &masks,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    const at::optional<at::Tensor> &viewmats_rs,
    const at::Tensor &v_colors,
    bool compute_v_means,
    bool compute_v_viewmats,
    bool compute_v_viewmats_rs
)
{
    DEVICE_GUARD(means);
    check_spherical_beta_inputs(
        lobes_to_use, means, viewmats, coeffs, masks, batch_ids, camera_ids, gaussian_ids, viewmats_rs
    );
    CHECK_DENSE(v_colors);
    at::Tensor grad_colors = v_colors.contiguous();
    check_spherical_beta_colors(grad_colors, means, viewmats, batch_ids.has_value(), coeffs.size(0), "v_colors");

    // The kernel reduces each lobe's gradient over all images in fp32 registers
    // and writes v_coeffs directly in the coeff dtype with a single store, so no
    // fp32 scratch buffer, zero-init, or fp32->coeff cast is required.
    at::Tensor v_coeffs = at::empty(coeffs.sizes(), coeffs.options());
    at::Tensor v_means, v_viewmats, v_viewmats_rs;
    if(compute_v_means)
    {
        v_means = at::zeros_like(means);
    }
    if(compute_v_viewmats)
    {
        v_viewmats = at::zeros_like(viewmats);
    }
    if(compute_v_viewmats_rs)
    {
        TORCH_INTERNAL_ASSERT(viewmats_rs.has_value());
        v_viewmats_rs = at::zeros_like(viewmats_rs.value());
    }

    launch_spherical_beta_bwd_kernel(
        lobes_to_use,
        means,
        viewmats,
        viewmats_rs,
        coeffs,
        masks,
        batch_ids,
        camera_ids,
        gaussian_ids,
        grad_colors,
        v_coeffs,
        as_optional_tensor(v_means),
        as_optional_tensor(v_viewmats),
        as_optional_tensor(v_viewmats_rs)
    );

    return std::make_tuple(
        v_coeffs, as_optional_tensor(v_means), as_optional_tensor(v_viewmats), as_optional_tensor(v_viewmats_rs)
    );
}

void register_spherical_beta_cuda_impl(torch::Library &m)
{
    m.impl("spherical_beta", to_torch_op<&spherical_beta_fwd>);
    m.impl("spherical_beta_bwd", to_torch_op<&spherical_beta_bwd>);
}
} // namespace gsplat
