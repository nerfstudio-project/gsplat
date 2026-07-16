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

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#if defined(__GNUC__) && !defined(__clang__)
// GCC PR110498 can diagnose std::vector<bool>::reserve inside
// torch::autograd::Function::apply as either of these warnings.
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Warray-bounds"
#    pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
#include <torch/csrc/autograd/custom_function.h>
#if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic pop
#endif
#include <torch/library.h>

#include "Common.h"             // where all the macros are defined
#include "SphericalHarmonics.h" // where the launch function is declared
#include "TorchUtils.h"

namespace gsplat
{
namespace
{
    constexpr double SH_C0 = 0.2820947917738781;

    void check_spherical_harmonics_inputs(
        int64_t degrees_to_use,
        const at::Tensor &means,
        const at::Tensor &viewmats,
        const at::Tensor &coeffs,
        const at::optional<at::Tensor> &masks,
        const at::optional<at::Tensor> &batch_ids,
        const at::optional<at::Tensor> &camera_ids,
        const at::optional<at::Tensor> &gaussian_ids,
        const at::optional<at::Tensor> &viewmats_rs = c10::nullopt,
        bool omit_l0                                = false
    )
    {
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
        TORCH_CHECK(coeffs.dim() == 3, "coeffs must have shape [N, K, D] or [nnz, K, D], got ", coeffs.sizes());
        TORCH_CHECK(coeffs.size(-1) >= 1, "coeffs last dim D must be >= 1, got ", coeffs.size(-1));
        TORCH_CHECK(
            (degrees_to_use + 1) * (degrees_to_use + 1) - (omit_l0 ? 1 : 0) <= coeffs.size(-2),
            "degrees_to_use requires more SH coefficients than provided; degree ",
            degrees_to_use,
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

    void check_spherical_harmonics_l1_plus_inputs(
        int64_t degrees_to_use,
        const at::Tensor &means,
        const at::Tensor &viewmats,
        const at::Tensor &shN,
        const at::optional<at::Tensor> &masks,
        const at::optional<at::Tensor> &batch_ids,
        const at::optional<at::Tensor> &camera_ids,
        const at::optional<at::Tensor> &gaussian_ids,
        const at::optional<at::Tensor> &viewmats_rs = c10::nullopt
    )
    {
        TORCH_CHECK(shN.dim() == 3, "shN must have shape [N, K - 1, D], got ", shN.sizes());
        check_spherical_harmonics_inputs(
            degrees_to_use, means, viewmats, shN, masks, batch_ids, camera_ids, gaussian_ids, viewmats_rs, true
        );
    }

    void check_spherical_harmonics_l0_inputs(const at::Tensor &sh0)
    {
        TORCH_CHECK(sh0.dim() == 3, "sh0 must have shape [N, 1, D], got ", sh0.sizes());
        TORCH_CHECK(sh0.size(-2) == 1, "sh0 must contain exactly one SH coefficient, got ", sh0.sizes());
        TORCH_CHECK(sh0.size(-1) >= 1, "sh0 last dim D must be >= 1, got ", sh0.size(-1));
        CHECK_INPUT(sh0);
    }
} // namespace

// Spherical harmonics
struct SphericalHarmonicsFwdResult
{
    at::Tensor colors;
};

template<>
struct TorchArgDef<SphericalHarmonicsFwdResult>
{
    static auto to(const SphericalHarmonicsFwdResult &r)
    {
        return to_torch_args(r.colors);
    }

    template<class TT>
    static SphericalHarmonicsFwdResult from(TT &&t)
    {
        return {.colors = t};
    }
};

struct SphericalHarmonicsBwdResult
{
    at::Tensor v_coeffs;
    at::optional<at::Tensor> v_means;
    at::optional<at::Tensor> v_viewmats;
    at::optional<at::Tensor> v_viewmats_rs;
};

template<>
struct TorchArgDef<SphericalHarmonicsBwdResult>
{
    static auto to(const SphericalHarmonicsBwdResult &r)
    {
        return to_torch_args(r.v_coeffs, r.v_means, r.v_viewmats, r.v_viewmats_rs);
    }
};

// Gradients of the differentiable forward outputs.
struct SphericalHarmonicsGrad
{
    static constexpr bool is_grad_bundle = true;
    at::Tensor colors; // [..., N, D]
};

template<>
struct TorchArgDef<SphericalHarmonicsGrad>
{
    static auto to(const SphericalHarmonicsGrad &g)
    {
        return to_torch_args(g.colors);
    }

    template<class TT>
    static SphericalHarmonicsGrad from(TT &&t)
    {
        return {.colors = t};
    }
};

SphericalHarmonicsFwdResult spherical_harmonics_fwd(
    int64_t degrees_to_use,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &masks,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    const at::optional<at::Tensor> &viewmats_rs
)
{
    DEVICE_GUARD(means);
    check_spherical_harmonics_inputs(
        degrees_to_use, means, viewmats, coeffs, masks, batch_ids, camera_ids, gaussian_ids, viewmats_rs
    );

    const bool packed = batch_ids.has_value();
    auto out_shape    = packed ? std::vector<int64_t>{coeffs.size(0), coeffs.size(-1)} : viewmats.sizes().vec();
    if(!packed)
    {
        out_shape.pop_back();
        out_shape.pop_back();
        out_shape.push_back(means.size(-2));
        out_shape.push_back(coeffs.size(-1));
    }
    at::Tensor colors = at::empty(out_shape, means.options());

    launch_spherical_harmonics_fwd_kernel(
        degrees_to_use, means, viewmats, viewmats_rs, coeffs, masks, batch_ids, camera_ids, gaussian_ids, colors
    );
    return SphericalHarmonicsFwdResult{
        .colors = colors,
    };
}

at::Tensor spherical_harmonics_l0_fwd(const at::Tensor &sh0)
{
    DEVICE_GUARD(sh0);
    check_spherical_harmonics_l0_inputs(sh0);

    // The full SH kernel evaluates coefficients in fp32, including when the
    // stored coefficients are fp16. Match that contract for split evaluation.
    return sh0.select(-2, 0).to(at::kFloat).mul(SH_C0); // [N, D]
}

at::Tensor spherical_harmonics_l0_bwd(const at::Tensor &sh0, const at::Tensor &v_colors)
{
    DEVICE_GUARD(sh0);
    check_spherical_harmonics_l0_inputs(sh0);
    CHECK_DENSE(v_colors);
    TORCH_CHECK(
        v_colors.dim() == 2 && v_colors.size(0) == sh0.size(0) && v_colors.size(1) == sh0.size(2),
        "v_colors must have shape [N, D] matching sh0; got v_colors ",
        v_colors.sizes(),
        " and sh0 ",
        sh0.sizes()
    );
    CHECK_DEVICE(v_colors);

    return v_colors.unsqueeze(-2).mul(SH_C0).to(sh0.scalar_type()); // [N, 1, D]
}

SphericalHarmonicsFwdResult spherical_harmonics_l1_plus_fwd(
    int64_t degrees_to_use,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::Tensor &shN,
    const at::optional<at::Tensor> &masks,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    const at::optional<at::Tensor> &viewmats_rs
)
{
    DEVICE_GUARD(means);
    check_spherical_harmonics_l1_plus_inputs(
        degrees_to_use, means, viewmats, shN, masks, batch_ids, camera_ids, gaussian_ids, viewmats_rs
    );

    const bool packed = batch_ids.has_value();
    auto out_shape    = packed ? std::vector<int64_t>{shN.size(0), shN.size(-1)} : viewmats.sizes().vec();
    if(!packed)
    {
        out_shape.pop_back();
        out_shape.pop_back();
        out_shape.push_back(means.size(-2));
        out_shape.push_back(shN.size(-1));
    }
    at::Tensor colors = at::empty(out_shape, means.options());

    launch_spherical_harmonics_l1_plus_fwd_kernel(
        degrees_to_use, means, viewmats, viewmats_rs, shN, masks, batch_ids, camera_ids, gaussian_ids, colors
    );
    return SphericalHarmonicsFwdResult{
        .colors = colors,
    };
}

SphericalHarmonicsFwdResult spherical_harmonics_fwd_privateuseone(
    int64_t degrees_to_use,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &masks,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    const at::optional<at::Tensor> &viewmats_rs
)
{
    DEVICE_GUARD(means);
    check_spherical_harmonics_inputs(
        degrees_to_use, means, viewmats, coeffs, masks, batch_ids, camera_ids, gaussian_ids, viewmats_rs
    );
    TORCH_CHECK(!viewmats_rs.has_value(), "rolling-shutter spherical_harmonics is not supported on PrivateUse1");
    TORCH_CHECK(!batch_ids.has_value(), "packed spherical_harmonics is not supported on PrivateUse1");
    auto out_shape = viewmats.sizes().vec();
    out_shape.pop_back();
    out_shape.pop_back();
    out_shape.push_back(means.size(-2));
    out_shape.push_back(coeffs.size(-1));
    at::Tensor colors = at::empty(out_shape, means.options());
    launch_spherical_harmonics_fwd_kernels(
        degrees_to_use, means, viewmats, coeffs, masks, batch_ids, camera_ids, gaussian_ids, colors
    );
    return SphericalHarmonicsFwdResult{
        .colors = colors,
    };
}

SphericalHarmonicsFwdResult spherical_harmonics_l1_plus_fwd_privateuseone(
    int64_t degrees_to_use,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::Tensor &shN,
    const at::optional<at::Tensor> &masks,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    const at::optional<at::Tensor> &viewmats_rs
)
{
    DEVICE_GUARD(means);
    check_spherical_harmonics_l1_plus_inputs(
        degrees_to_use, means, viewmats, shN, masks, batch_ids, camera_ids, gaussian_ids, viewmats_rs
    );
    TORCH_CHECK(
        !viewmats_rs.has_value(), "rolling-shutter spherical_harmonics_l1_plus is not supported on PrivateUse1"
    );
    TORCH_CHECK(!batch_ids.has_value(), "packed spherical_harmonics_l1_plus is not supported on PrivateUse1");
    auto out_shape = viewmats.sizes().vec();
    out_shape.pop_back();
    out_shape.pop_back();
    out_shape.push_back(means.size(-2));
    out_shape.push_back(shN.size(-1));
    at::Tensor colors = at::empty(out_shape, means.options());
    launch_spherical_harmonics_l1_plus_fwd_kernels(
        degrees_to_use, means, viewmats, shN, masks, batch_ids, camera_ids, gaussian_ids, colors
    );
    return SphericalHarmonicsFwdResult{
        .colors = colors,
    };
}

// Full backward for spherical_harmonics.
SphericalHarmonicsBwdResult spherical_harmonics_bwd(
    int64_t degrees_to_use,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &masks,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    const at::optional<at::Tensor> &viewmats_rs,
    const SphericalHarmonicsGrad &grad,
    bool compute_v_means,
    bool compute_v_viewmats,
    bool compute_v_viewmats_rs
)
{
    DEVICE_GUARD(means);
    check_spherical_harmonics_inputs(
        degrees_to_use, means, viewmats, coeffs, masks, batch_ids, camera_ids, gaussian_ids, viewmats_rs
    );
    TORCH_INTERNAL_ASSERT(grad.colors.defined());
    CHECK_DENSE(grad.colors);
    at::Tensor grad_colors = grad.colors.contiguous();
    CHECK_INPUT(grad_colors);
    TORCH_CHECK(
        grad_colors.size(-1) == coeffs.size(-1),
        "v_colors last dim (",
        grad_colors.size(-1),
        ") must match coeffs last dim (",
        coeffs.size(-1),
        ")"
    );

    // Always accumulate v_coeffs in fp32 to avoid precision loss when multiple
    // (batch, gaussian) elements atomic-add into the same slot. For fp32
    // coeffs the accumulator IS the output; for fp16 we cast at the end.
    at::Tensor v_coeffs_accum = at::zeros(coeffs.sizes(), coeffs.options().dtype(at::kFloat));
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

    launch_spherical_harmonics_bwd_kernel(
        degrees_to_use,
        means,
        viewmats,
        viewmats_rs,
        coeffs,
        masks,
        batch_ids,
        camera_ids,
        gaussian_ids,
        grad_colors,
        v_coeffs_accum,
        as_optional_tensor(v_means),
        as_optional_tensor(v_viewmats),
        as_optional_tensor(v_viewmats_rs)
    );

    at::Tensor v_coeffs
        = (coeffs.scalar_type() == at::kFloat) ? v_coeffs_accum : v_coeffs_accum.to(coeffs.scalar_type());
    return SphericalHarmonicsBwdResult{
        .v_coeffs      = v_coeffs,
        .v_means       = as_optional_tensor(v_means),
        .v_viewmats    = as_optional_tensor(v_viewmats),
        .v_viewmats_rs = as_optional_tensor(v_viewmats_rs),
    };
}

SphericalHarmonicsBwdResult spherical_harmonics_l1_plus_bwd(
    int64_t degrees_to_use,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::Tensor &shN,
    const at::optional<at::Tensor> &masks,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    const at::optional<at::Tensor> &viewmats_rs,
    const SphericalHarmonicsGrad &grad,
    bool compute_v_means,
    bool compute_v_viewmats,
    bool compute_v_viewmats_rs
)
{
    DEVICE_GUARD(means);
    check_spherical_harmonics_l1_plus_inputs(
        degrees_to_use, means, viewmats, shN, masks, batch_ids, camera_ids, gaussian_ids, viewmats_rs
    );
    TORCH_INTERNAL_ASSERT(grad.colors.defined());
    CHECK_DENSE(grad.colors);
    at::Tensor grad_colors = grad.colors.contiguous();
    CHECK_INPUT(grad_colors);
    TORCH_CHECK(
        grad_colors.size(-1) == shN.size(-1),
        "v_colors last dim (",
        grad_colors.size(-1),
        ") must match shN last dim (",
        shN.size(-1),
        ")"
    );

    at::Tensor v_shN_accum = at::zeros(shN.sizes(), shN.options().dtype(at::kFloat));
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

    launch_spherical_harmonics_l1_plus_bwd_kernel(
        degrees_to_use,
        means,
        viewmats,
        viewmats_rs,
        shN,
        masks,
        batch_ids,
        camera_ids,
        gaussian_ids,
        grad_colors,
        v_shN_accum,
        as_optional_tensor(v_means),
        as_optional_tensor(v_viewmats),
        as_optional_tensor(v_viewmats_rs)
    );

    at::Tensor v_shN = (shN.scalar_type() == at::kFloat) ? v_shN_accum : v_shN_accum.to(shN.scalar_type());
    return SphericalHarmonicsBwdResult{
        .v_coeffs      = v_shN,
        .v_means       = as_optional_tensor(v_means),
        .v_viewmats    = as_optional_tensor(v_viewmats),
        .v_viewmats_rs = as_optional_tensor(v_viewmats_rs),
    };
}

// The C++ autograd operator. Dispatches through the registered
// `gsplat::spherical_harmonics` op so the Python-side register_autograd kernel
// makes it differentiable.
at::Tensor spherical_harmonics(
    int64_t degrees_to_use,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &masks,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    const at::optional<at::Tensor> &viewmats_rs
)
{
    return call_torch_op<&spherical_harmonics_fwd>(
               "gsplat::spherical_harmonics",
               degrees_to_use,
               means,
               viewmats,
               coeffs,
               masks,
               batch_ids,
               camera_ids,
               gaussian_ids,
               viewmats_rs
    )
        .colors;
}

// Fused forward assembly of proj_features = [SH colors | extra | (depth)] for
// the unpacked rasterization path. Writes each complete output row in one
// coalesced pass, replacing the SH-eval + cat(color, extra) + cat(.., depth)
// chain. Forward-only; backward is handled by the autograd wrapper in C++.
void assemble_proj_features_unpacked_fwd(
    int64_t degrees_to_use,
    int64_t B,
    int64_t C,
    int64_t N,
    int64_t Dc,
    int64_t E,
    int64_t color_post,
    int64_t extra_post,
    bool has_depth,
    bool depth_is_zero,
    bool extra_has_c,
    const at::Tensor &means,                     // [B, N, 3]
    const at::Tensor &viewmats,                  // [B, C, 4, 4]
    const at::optional<at::Tensor> &viewmats_rs, // [B, C, 4, 4]
    const at::Tensor &coeffs,                    // [N, K, Dc]
    const at::optional<at::Tensor> &extra,       // [B, C, N, E] or [B, N, E]
    const at::optional<at::Tensor> &depths,      // [B, C, N]
    const at::optional<at::Tensor> &masks,       // [B, C, N]
    at::Tensor &out,                             // [B, C, N, Dc + E + has_depth]
    const at::optional<at::Tensor> &relu_mask    // [B, C, N, Dc]
)
{
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(viewmats);
    if(viewmats_rs.has_value())
    {
        CHECK_INPUT(viewmats_rs.value());
        TORCH_CHECK(viewmats_rs.value().sizes() == viewmats.sizes(), "viewmats_rs must match viewmats shape");
        TORCH_CHECK(viewmats_rs.value().scalar_type() == at::kFloat, "viewmats_rs must be float32");
    }
    CHECK_INPUT(coeffs);
    CHECK_DEVICE(out);
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(means.scalar_type() == at::kFloat, "means must be float32");
    TORCH_CHECK(viewmats.scalar_type() == at::kFloat, "viewmats must be float32");
    TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
    TORCH_CHECK(means.size(-1) == 3, "means must have last dim 3");
    TORCH_CHECK(viewmats.size(-2) == 4 && viewmats.size(-1) == 4, "viewmats must have last dims 4x4");
    TORCH_CHECK(coeffs.dim() == 3, "coeffs must be [N, K, Dc], got ", coeffs.sizes());
    TORCH_CHECK(coeffs.size(-3) == N, "coeffs N (", coeffs.size(-3), ") != N (", N, ")");
    TORCH_CHECK(coeffs.size(-1) == Dc, "coeffs Dc (", coeffs.size(-1), ") != Dc (", Dc, ")");
    const int64_t lead = B * C * N;
    TORCH_CHECK(means.numel() == B * N * 3, "means numel mismatch");
    TORCH_CHECK(viewmats.numel() == B * C * 16, "viewmats numel mismatch");
    const int64_t width = Dc + E + (has_depth ? 1 : 0);
    TORCH_CHECK(out.size(-1) == width, "out width (", out.size(-1), ") != ", width);
    TORCH_CHECK(out.numel() == lead * width, "out numel mismatch");
    TORCH_CHECK(color_post >= 0 && color_post <= 2, "bad color_post ", color_post);
    TORCH_CHECK(extra_post >= 0 && extra_post <= 2, "bad extra_post ", extra_post);

    if(E > 0)
    {
        TORCH_CHECK(extra.has_value(), "extra required when E > 0");
        CHECK_INPUT(extra.value());
        TORCH_CHECK(extra.value().scalar_type() == at::kFloat, "extra must be float32");
        const int64_t want = (extra_has_c ? lead : B * N) * E;
        TORCH_CHECK(extra.value().numel() == want, "extra numel ", extra.value().numel(), " != ", want);
    }
    if(has_depth && !depth_is_zero)
    {
        TORCH_CHECK(depths.has_value(), "depths required when has_depth and !depth_is_zero");
        CHECK_INPUT(depths.value());
        TORCH_CHECK(depths.value().scalar_type() == at::kFloat, "depths must be float32");
        TORCH_CHECK(depths.value().numel() == lead, "depths numel mismatch");
    }
    if(masks.has_value())
    {
        CHECK_INPUT(masks.value());
        TORCH_CHECK(masks.value().numel() == lead, "masks numel mismatch");
    }
    if(relu_mask.has_value())
    {
        CHECK_DEVICE(relu_mask.value());
        TORCH_CHECK(relu_mask.value().is_contiguous(), "relu_mask must be contiguous");
        TORCH_CHECK(relu_mask.value().scalar_type() == at::kBool, "relu_mask must be bool");
        TORCH_CHECK(relu_mask.value().numel() == lead * Dc, "relu_mask numel mismatch");
        TORCH_CHECK(color_post == 2, "relu_mask only valid with color_post=shift_relu");
    }

    launch_assemble_proj_features_unpacked_fwd_kernel(
        static_cast<uint32_t>(B),
        static_cast<uint32_t>(C),
        static_cast<uint32_t>(N),
        static_cast<uint32_t>(degrees_to_use),
        static_cast<uint32_t>(Dc),
        static_cast<uint32_t>(E),
        static_cast<uint32_t>(color_post),
        static_cast<uint32_t>(extra_post),
        has_depth,
        depth_is_zero,
        extra_has_c,
        means,
        viewmats,
        viewmats_rs,
        coeffs,
        (E > 0) ? extra : c10::nullopt,
        (has_depth && !depth_is_zero) ? depths : c10::nullopt,
        masks,
        out,
        relu_mask
    );
}

namespace
{
    // Autograd wrapper around the fused assembler. Forward runs the single-kernel
    // assembly; backward routes the (relu-masked) color-slice gradient through the
    // SH backward kernel and passes extra/depth gradients straight through. Mirrors
    // the reference path (maybe_evaluate_feature_sh + cat) so the composed
    // rasterization autograd graph is numerically identical.
    class AssembleProjFeaturesAutograd : public torch::autograd::Function<AssembleProjFeaturesAutograd>
    {
    public:
        static at::Tensor forward(
            torch::autograd::AutogradContext *ctx,
            int64_t degrees_to_use,
            int64_t B,
            int64_t C,
            int64_t N,
            int64_t Dc,
            int64_t E,
            int64_t color_post,
            int64_t extra_post,
            bool has_depth,
            bool depth_is_zero,
            bool extra_has_c,
            const at::Tensor &means,
            const at::Tensor &viewmats,
            const at::optional<at::Tensor> &viewmats_rs,
            const at::Tensor &coeffs,
            const at::optional<at::Tensor> &extra,
            const at::optional<at::Tensor> &depths,
            const at::optional<at::Tensor> &masks
        )
        {
            at::Tensor means_c    = means.contiguous();
            at::Tensor viewmats_c = viewmats.contiguous();
            at::optional<at::Tensor> viewmats_rs_c
                = viewmats_rs.has_value() ? at::optional<at::Tensor>(viewmats_rs.value().contiguous()) : c10::nullopt;
            at::Tensor coeffs_c = coeffs.contiguous();
            at::optional<at::Tensor> extra_c
                = (extra.has_value() && E > 0) ? at::optional<at::Tensor>(extra.value().contiguous()) : c10::nullopt;
            at::optional<at::Tensor> depths_c = (has_depth && !depth_is_zero && depths.has_value())
                                                  ? at::optional<at::Tensor>(depths.value().contiguous())
                                                  : c10::nullopt;
            at::optional<at::Tensor> masks_c
                = masks.has_value() ? at::optional<at::Tensor>(masks.value().contiguous()) : c10::nullopt;

            const int64_t width = Dc + E + (has_depth ? 1 : 0);
            // out layout: means leading batch dims, then (C, N, width).
            std::vector<int64_t> out_shape(means_c.sizes().begin(), means_c.sizes().end() - 2);
            out_shape.push_back(C);
            out_shape.push_back(N);
            out_shape.push_back(width);
            at::Tensor out = at::empty(out_shape, means_c.options());

            // Capture grad requirements now (the contiguous() copies above were made
            // under the Function's no-grad forward, so their requires_grad is false).
            const bool need_means       = means.requires_grad();
            const bool need_viewmats    = viewmats.requires_grad();
            const bool need_viewmats_rs = viewmats_rs.has_value() && viewmats_rs.value().requires_grad();
            const bool need_coeffs      = coeffs.requires_grad();
            const bool need_extra       = extra.has_value() && extra.value().requires_grad();
            const bool need_depths      = depths.has_value() && depths.value().requires_grad();

            // relu Jacobian mask (needed in backward for shift_relu) is emitted by
            // the kernel when any color input requires grad.
            at::optional<at::Tensor> relu_mask;
            if(color_post == 2 && (need_means || need_viewmats || need_viewmats_rs || need_coeffs))
            {
                std::vector<int64_t> rm_shape(means_c.sizes().begin(), means_c.sizes().end() - 2);
                rm_shape.push_back(C);
                rm_shape.push_back(N);
                rm_shape.push_back(Dc);
                relu_mask = at::empty(rm_shape, means_c.options().dtype(at::kBool));
            }

            launch_assemble_proj_features_unpacked_fwd_kernel(
                static_cast<uint32_t>(B),
                static_cast<uint32_t>(C),
                static_cast<uint32_t>(N),
                static_cast<uint32_t>(degrees_to_use),
                static_cast<uint32_t>(Dc),
                static_cast<uint32_t>(E),
                static_cast<uint32_t>(color_post),
                static_cast<uint32_t>(extra_post),
                has_depth,
                depth_is_zero,
                extra_has_c,
                means_c,
                viewmats_c,
                viewmats_rs_c,
                coeffs_c,
                extra_c,
                depths_c,
                masks_c,
                out,
                relu_mask
            );

            ctx->save_for_backward(
                {means_c,
                 viewmats_c,
                 viewmats_rs_c.has_value() ? viewmats_rs_c.value() : at::Tensor(),
                 coeffs_c,
                 masks_c.has_value() ? masks_c.value() : at::Tensor(),
                 relu_mask.has_value() ? relu_mask.value() : at::Tensor()}
            );
            ctx->saved_data["degrees_to_use"]   = degrees_to_use;
            ctx->saved_data["Dc"]               = Dc;
            ctx->saved_data["E"]                = E;
            ctx->saved_data["color_post"]       = color_post;
            ctx->saved_data["has_depth"]        = has_depth;
            ctx->saved_data["depth_is_zero"]    = depth_is_zero;
            ctx->saved_data["extra_has_c"]      = extra_has_c;
            ctx->saved_data["need_means"]       = need_means;
            ctx->saved_data["need_viewmats"]    = need_viewmats;
            ctx->saved_data["need_viewmats_rs"] = need_viewmats_rs;
            ctx->saved_data["need_coeffs"]      = need_coeffs;
            ctx->saved_data["need_extra"]       = need_extra;
            ctx->saved_data["need_depths"]      = need_depths;
            return out;
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs
        )
        {
            const auto saved                = ctx->get_saved_variables();
            const at::Tensor &means_c       = saved[0];
            const at::Tensor &viewmats_c    = saved[1];
            const at::Tensor &viewmats_rs_t = saved[2];
            const at::Tensor &coeffs_c      = saved[3];
            const at::Tensor &masks_t       = saved[4];
            const at::Tensor &relu_mask     = saved[5];
            at::optional<at::Tensor> viewmats_rs
                = viewmats_rs_t.defined() ? at::optional<at::Tensor>(viewmats_rs_t) : c10::nullopt;
            at::optional<at::Tensor> masks = masks_t.defined() ? at::optional<at::Tensor>(masks_t) : c10::nullopt;

            const int64_t degrees_to_use = ctx->saved_data["degrees_to_use"].toInt();
            const int64_t Dc             = ctx->saved_data["Dc"].toInt();
            const int64_t E              = ctx->saved_data["E"].toInt();
            const int64_t color_post     = ctx->saved_data["color_post"].toInt();
            const bool has_depth         = ctx->saved_data["has_depth"].toBool();
            const bool depth_is_zero     = ctx->saved_data["depth_is_zero"].toBool();
            const bool extra_has_c       = ctx->saved_data["extra_has_c"].toBool();

            at::Tensor g = grad_outputs[0].contiguous();

            const bool need_means       = ctx->saved_data["need_means"].toBool();
            const bool need_viewmats    = ctx->saved_data["need_viewmats"].toBool();
            const bool need_viewmats_rs = ctx->saved_data["need_viewmats_rs"].toBool();
            const bool need_coeffs      = ctx->saved_data["need_coeffs"].toBool();
            const bool need_extra       = ctx->saved_data["need_extra"].toBool();
            const bool need_depths      = ctx->saved_data["need_depths"].toBool();
            at::Tensor v_means, v_viewmats, v_viewmats_rs, v_coeffs, v_extra, v_depths;

            if(need_means || need_viewmats || need_viewmats_rs || need_coeffs)
            {
                at::Tensor g_color = g.narrow(-1, 0, Dc);
                if(color_post == 2 && relu_mask.defined())
                {
                    g_color = g_color.mul(relu_mask);
                }
                g_color = g_color.contiguous();

                at::Tensor v_coeffs_accum = at::zeros(coeffs_c.sizes(), coeffs_c.options().dtype(at::kFloat));
                if(need_means)
                {
                    v_means = at::zeros_like(means_c);
                }
                if(need_viewmats)
                {
                    v_viewmats = at::zeros_like(viewmats_c);
                }
                if(need_viewmats_rs)
                {
                    TORCH_INTERNAL_ASSERT(viewmats_rs.has_value());
                    v_viewmats_rs = at::zeros_like(viewmats_rs.value());
                }
                launch_spherical_harmonics_bwd_kernel(
                    static_cast<uint32_t>(degrees_to_use),
                    means_c,
                    viewmats_c,
                    viewmats_rs,
                    coeffs_c,
                    masks,
                    c10::nullopt,
                    c10::nullopt,
                    c10::nullopt,
                    g_color,
                    v_coeffs_accum,
                    need_means ? at::optional<at::Tensor>(v_means) : c10::nullopt,
                    need_viewmats ? at::optional<at::Tensor>(v_viewmats) : c10::nullopt,
                    need_viewmats_rs ? at::optional<at::Tensor>(v_viewmats_rs) : c10::nullopt
                );
                if(need_coeffs)
                {
                    v_coeffs = (coeffs_c.scalar_type() == at::kFloat) ? v_coeffs_accum
                                                                      : v_coeffs_accum.to(coeffs_c.scalar_type());
                }
            }

            if(E > 0 && need_extra)
            {
                at::Tensor g_extra = g.narrow(-1, Dc, E);
                v_extra
                    = extra_has_c ? g_extra.contiguous() : g_extra.sum(means_c.dim() - 2); // reduce broadcast over C
            }

            if(has_depth && !depth_is_zero && need_depths)
            {
                v_depths = g.narrow(-1, Dc + E, 1).squeeze(-1).contiguous();
            }

            // grads in forward-input order (non-tensor inputs -> undefined Tensor).
            torch::autograd::variable_list grads(18);
            grads[11] = v_means;
            grads[12] = v_viewmats;
            grads[13] = v_viewmats_rs;
            grads[14] = v_coeffs;
            grads[15] = v_extra;
            grads[16] = v_depths;
            return grads;
        }
    };
} // namespace

at::Tensor assemble_proj_features(
    int64_t degrees_to_use,
    int64_t B,
    int64_t C,
    int64_t N,
    int64_t Dc,
    int64_t E,
    int64_t color_post,
    int64_t extra_post,
    bool has_depth,
    bool depth_is_zero,
    bool extra_has_c,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::optional<at::Tensor> &viewmats_rs,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &extra,
    const at::optional<at::Tensor> &depths,
    const at::optional<at::Tensor> &masks
)
{
    return AssembleProjFeaturesAutograd::apply(
        degrees_to_use,
        B,
        C,
        N,
        Dc,
        E,
        color_post,
        extra_post,
        has_depth,
        depth_is_zero,
        extra_has_c,
        means,
        viewmats,
        viewmats_rs,
        coeffs,
        extra,
        depths,
        masks
    );
}

SphericalHarmonicsBwdResult spherical_harmonics_bwd_privateuseone(
    int64_t degrees_to_use,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &masks,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    const at::optional<at::Tensor> &viewmats_rs,
    const SphericalHarmonicsGrad &grad,
    bool compute_v_means,
    bool compute_v_viewmats,
    bool compute_v_viewmats_rs
)
{
    DEVICE_GUARD(means);
    check_spherical_harmonics_inputs(
        degrees_to_use, means, viewmats, coeffs, masks, batch_ids, camera_ids, gaussian_ids, viewmats_rs
    );
    TORCH_CHECK(!viewmats_rs.has_value(), "rolling-shutter spherical_harmonics is not supported on PrivateUse1");
    TORCH_INTERNAL_ASSERT(!compute_v_viewmats_rs);
    TORCH_CHECK(!batch_ids.has_value(), "packed spherical_harmonics is not supported on PrivateUse1");
    TORCH_INTERNAL_ASSERT(grad.colors.defined());
    CHECK_DENSE(grad.colors);
    at::Tensor grad_colors = grad.colors.contiguous();
    CHECK_INPUT(grad_colors);
    TORCH_CHECK(
        grad_colors.size(-1) == coeffs.size(-1),
        "v_colors last dim (",
        grad_colors.size(-1),
        ") must match coeffs last dim (",
        coeffs.size(-1),
        ")"
    );

    at::Tensor v_coeffs_accum = at::empty(coeffs.sizes(), coeffs.options().dtype(at::kFloat));
    at::Tensor v_means, v_viewmats;
    if(compute_v_means)
    {
        v_means = at::empty_like(means);
    }
    if(compute_v_viewmats)
    {
        v_viewmats = at::zeros_like(viewmats);
    }

    launch_spherical_harmonics_bwd_kernels(
        degrees_to_use,
        means,
        viewmats,
        coeffs,
        masks,
        batch_ids,
        camera_ids,
        gaussian_ids,
        grad_colors,
        v_coeffs_accum,
        as_optional_tensor(v_means),
        as_optional_tensor(v_viewmats)
    );

    at::Tensor v_coeffs
        = (coeffs.scalar_type() == at::kFloat) ? v_coeffs_accum : v_coeffs_accum.to(coeffs.scalar_type());
    return SphericalHarmonicsBwdResult{
        .v_coeffs      = v_coeffs,
        .v_means       = as_optional_tensor(v_means),
        .v_viewmats    = as_optional_tensor(v_viewmats),
        .v_viewmats_rs = c10::nullopt,
    };
}

SphericalHarmonicsBwdResult spherical_harmonics_l1_plus_bwd_privateuseone(
    int64_t degrees_to_use,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::Tensor &shN,
    const at::optional<at::Tensor> &masks,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    const at::optional<at::Tensor> &viewmats_rs,
    const SphericalHarmonicsGrad &grad,
    bool compute_v_means,
    bool compute_v_viewmats,
    bool compute_v_viewmats_rs
)
{
    DEVICE_GUARD(means);
    check_spherical_harmonics_l1_plus_inputs(
        degrees_to_use, means, viewmats, shN, masks, batch_ids, camera_ids, gaussian_ids, viewmats_rs
    );
    TORCH_CHECK(
        !viewmats_rs.has_value(), "rolling-shutter spherical_harmonics_l1_plus is not supported on PrivateUse1"
    );
    TORCH_INTERNAL_ASSERT(!compute_v_viewmats_rs);
    TORCH_CHECK(!batch_ids.has_value(), "packed spherical_harmonics_l1_plus is not supported on PrivateUse1");
    TORCH_INTERNAL_ASSERT(grad.colors.defined());
    CHECK_DENSE(grad.colors);
    at::Tensor grad_colors = grad.colors.contiguous();
    CHECK_INPUT(grad_colors);
    TORCH_CHECK(
        grad_colors.size(-1) == shN.size(-1),
        "v_colors last dim (",
        grad_colors.size(-1),
        ") must match shN last dim (",
        shN.size(-1),
        ")"
    );

    at::Tensor v_shN_accum = at::zeros(shN.sizes(), shN.options().dtype(at::kFloat));
    at::Tensor v_means, v_viewmats;
    if(compute_v_means)
    {
        v_means = at::zeros_like(means);
    }
    if(compute_v_viewmats)
    {
        v_viewmats = at::zeros_like(viewmats);
    }

    launch_spherical_harmonics_l1_plus_bwd_kernels(
        degrees_to_use,
        means,
        viewmats,
        shN,
        masks,
        batch_ids,
        camera_ids,
        gaussian_ids,
        grad_colors,
        v_shN_accum,
        as_optional_tensor(v_means),
        as_optional_tensor(v_viewmats)
    );

    at::Tensor v_shN = (shN.scalar_type() == at::kFloat) ? v_shN_accum : v_shN_accum.to(shN.scalar_type());
    return SphericalHarmonicsBwdResult{
        .v_coeffs      = v_shN,
        .v_means       = as_optional_tensor(v_means),
        .v_viewmats    = as_optional_tensor(v_viewmats),
        .v_viewmats_rs = c10::nullopt,
    };
}

void register_spherical_harmonics_cuda_impl(torch::Library &m)
{
    m.impl("spherical_harmonics", to_torch_op<&spherical_harmonics_fwd>);
    m.impl("spherical_harmonics_bwd", to_torch_op<&spherical_harmonics_bwd>);
    m.impl("spherical_harmonics_l0", to_torch_op<&spherical_harmonics_l0_fwd>);
    m.impl("spherical_harmonics_l0_bwd", to_torch_op<&spherical_harmonics_l0_bwd>);
    m.impl("spherical_harmonics_l1_plus", to_torch_op<&spherical_harmonics_l1_plus_fwd>);
    m.impl("spherical_harmonics_l1_plus_bwd", to_torch_op<&spherical_harmonics_l1_plus_bwd>);
    m.impl("assemble_proj_features_unpacked_fwd", &assemble_proj_features_unpacked_fwd);
}

void register_spherical_harmonics_privateuseone_impl(torch::Library &m)
{
    m.impl("spherical_harmonics", to_torch_op<&spherical_harmonics_fwd_privateuseone>);
    m.impl("spherical_harmonics_bwd", to_torch_op<&spherical_harmonics_bwd_privateuseone>);
    m.impl("spherical_harmonics_l0", to_torch_op<&spherical_harmonics_l0_fwd>);
    m.impl("spherical_harmonics_l0_bwd", to_torch_op<&spherical_harmonics_l0_bwd>);
    m.impl("spherical_harmonics_l1_plus", to_torch_op<&spherical_harmonics_l1_plus_fwd_privateuseone>);
    m.impl("spherical_harmonics_l1_plus_bwd", to_torch_op<&spherical_harmonics_l1_plus_bwd_privateuseone>);
}
} // namespace gsplat
