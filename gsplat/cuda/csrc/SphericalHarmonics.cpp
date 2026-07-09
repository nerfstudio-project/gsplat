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
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "Common.h"             // where all the macros are defined
#include "SphericalHarmonics.h" // where the launch function is declared
#include "TorchUtils.h"

namespace gsplat
{
namespace
{
    void check_spherical_harmonics_inputs(
        int64_t degrees_to_use, const at::Tensor &dirs, const at::Tensor &coeffs, const at::optional<at::Tensor> &masks
    )
    {
        TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
        TORCH_CHECK(dirs.dim() >= 2, "dirs must have shape [..., N, 3], got ", dirs.sizes());
        TORCH_CHECK(coeffs.dim() == 3, "coeffs must have shape [N, K, D], got ", coeffs.sizes());
        TORCH_CHECK(coeffs.size(-1) >= 1, "coeffs last dim D must be >= 1, got ", coeffs.size(-1));
        TORCH_CHECK(
            dirs.size(-2) == coeffs.size(-3), "dirs N (", dirs.size(-2), ") must match coeffs N (", coeffs.size(-3), ")"
        );
        TORCH_CHECK(
            (degrees_to_use + 1) * (degrees_to_use + 1) <= coeffs.size(-2),
            "degrees_to_use requires more SH coefficients than provided; degree ",
            degrees_to_use,
            ", coeffs shape ",
            coeffs.sizes()
        );
        if(masks.has_value())
        {
            at::DimVector mask_shape(dirs.sizes().slice(0, dirs.dim() - 1));
            TORCH_CHECK(
                masks.value().sizes() == mask_shape,
                "masks must match dirs.shape[:-1]; got masks ",
                masks.value().sizes(),
                " and dirs ",
                dirs.sizes()
            );
        }
        CHECK_INPUT(dirs);
        CHECK_INPUT(coeffs);
        if(masks.has_value())
        {
            CHECK_INPUT(masks.value());
        }
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
    at::optional<at::Tensor> v_dirs;
};

template<>
struct TorchArgDef<SphericalHarmonicsBwdResult>
{
    static auto to(const SphericalHarmonicsBwdResult &r)
    {
        return to_torch_args(r.v_coeffs, r.v_dirs);
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
    const at::Tensor &dirs,               // [..., N, 3]
    const at::Tensor &coeffs,             // [N, K, D]
    const at::optional<at::Tensor> &masks // [..., N]
)
{
    DEVICE_GUARD(dirs);
    check_spherical_harmonics_inputs(degrees_to_use, dirs, coeffs, masks);

    // colors dtype follows dirs; the kernel converts fp16 coeffs to fp32 on read.
    auto out_shape    = dirs.sizes().vec();
    out_shape.back()  = coeffs.size(-1);
    at::Tensor colors = at::empty(out_shape, dirs.options()); // [..., N, D]

    launch_spherical_harmonics_fwd_kernel(degrees_to_use, dirs, coeffs, masks, colors);
    return SphericalHarmonicsFwdResult{
        .colors = colors,
    };
}

at::Tensor spherical_harmonics_fwd_privateuseone(
    int64_t degrees_to_use,
    const at::Tensor &dirs,               // [..., N, 3]
    const at::Tensor &coeffs,             // [N, K, D]
    const at::optional<at::Tensor> &masks // [..., N]
)
{
    DEVICE_GUARD(dirs);
    check_spherical_harmonics_inputs(degrees_to_use, dirs, coeffs, masks);

    auto out_shape    = dirs.sizes().vec();
    out_shape.back()  = coeffs.size(-1);
    at::Tensor colors = at::empty(out_shape, dirs.options()); // [..., N, D]

    launch_spherical_harmonics_fwd_kernels(degrees_to_use, dirs, coeffs, masks, colors);
    return colors; // [..., N, D]
}

// Full backward for spherical_harmonics.
// `compute_v_dirs` selects whether to compute the direction gradient. It is an
// explicit argument rather than something inferred from a tensor's
// requires_grad, so this functional op's behavior follows only from its inputs.
SphericalHarmonicsBwdResult spherical_harmonics_bwd(
    int64_t degrees_to_use,
    const at::Tensor &dirs,                // [..., N, 3]
    const at::Tensor &coeffs,              // [N, K, D]
    const at::optional<at::Tensor> &masks, // [..., N]
    const SphericalHarmonicsGrad &grad,
    bool compute_v_dirs
)
{
    DEVICE_GUARD(dirs);
    check_spherical_harmonics_inputs(degrees_to_use, dirs, coeffs, masks);
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
    at::Tensor v_dirs;
    if(compute_v_dirs)
    {
        v_dirs = at::zeros_like(dirs);
    }

    launch_spherical_harmonics_bwd_kernel(
        degrees_to_use, dirs, coeffs, masks, grad_colors, v_coeffs_accum, as_optional_tensor(v_dirs)
    );

    at::Tensor v_coeffs
        = (coeffs.scalar_type() == at::kFloat) ? v_coeffs_accum : v_coeffs_accum.to(coeffs.scalar_type());
    return SphericalHarmonicsBwdResult{
        .v_coeffs = v_coeffs,
        .v_dirs   = as_optional_tensor(v_dirs),
    };
}

// The C++ autograd operator. Dispatches through the registered
// `gsplat::spherical_harmonics` op so the Python-side register_autograd kernel
// makes it differentiable.
at::Tensor spherical_harmonics(
    int64_t degrees_to_use, const at::Tensor &dirs, const at::Tensor &coeffs, const at::optional<at::Tensor> &masks
)
{
    return call_torch_op<&spherical_harmonics_fwd>("gsplat::spherical_harmonics", degrees_to_use, dirs, coeffs, masks)
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
    const at::Tensor &means,                  // [B, N, 3]
    const at::Tensor &campos,                 // [B, C, 3]
    const at::Tensor &coeffs,                 // [N, K, Dc]
    const at::optional<at::Tensor> &extra,    // [B, C, N, E] or [B, N, E]
    const at::optional<at::Tensor> &depths,   // [B, C, N]
    const at::optional<at::Tensor> &masks,    // [B, C, N]
    at::Tensor &out,                          // [B, C, N, Dc + E + has_depth]
    const at::optional<at::Tensor> &relu_mask // [B, C, N, Dc]
)
{
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(campos);
    CHECK_INPUT(coeffs);
    CHECK_DEVICE(out);
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(means.scalar_type() == at::kFloat, "means must be float32");
    TORCH_CHECK(campos.scalar_type() == at::kFloat, "campos must be float32");
    TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
    TORCH_CHECK(means.size(-1) == 3, "means must have last dim 3");
    TORCH_CHECK(campos.size(-1) == 3, "campos must have last dim 3");
    TORCH_CHECK(coeffs.dim() == 3, "coeffs must be [N, K, Dc], got ", coeffs.sizes());
    TORCH_CHECK(coeffs.size(-3) == N, "coeffs N (", coeffs.size(-3), ") != N (", N, ")");
    TORCH_CHECK(coeffs.size(-1) == Dc, "coeffs Dc (", coeffs.size(-1), ") != Dc (", Dc, ")");
    const int64_t lead = B * C * N;
    TORCH_CHECK(means.numel() == B * N * 3, "means numel mismatch");
    TORCH_CHECK(campos.numel() == B * C * 3, "campos numel mismatch");
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
        campos,
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
            const at::Tensor &campos,
            const at::Tensor &coeffs,
            const at::optional<at::Tensor> &extra,
            const at::optional<at::Tensor> &depths,
            const at::optional<at::Tensor> &masks
        )
        {
            at::Tensor means_c  = means.contiguous();
            at::Tensor campos_c = campos.contiguous();
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
            const bool need_means  = means.requires_grad();
            const bool need_campos = campos.requires_grad();
            const bool need_coeffs = coeffs.requires_grad();
            const bool need_extra  = extra.has_value() && extra.value().requires_grad();
            const bool need_depths = depths.has_value() && depths.value().requires_grad();

            // relu Jacobian mask (needed in backward for shift_relu) is emitted by
            // the kernel when any color input requires grad.
            at::optional<at::Tensor> relu_mask;
            if(color_post == 2 && (need_means || need_campos || need_coeffs))
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
                campos_c,
                coeffs_c,
                extra_c,
                depths_c,
                masks_c,
                out,
                relu_mask
            );

            ctx->save_for_backward(
                {means_c,
                 campos_c,
                 coeffs_c,
                 masks_c.has_value() ? masks_c.value() : at::Tensor(),
                 relu_mask.has_value() ? relu_mask.value() : at::Tensor()}
            );
            ctx->saved_data["degrees_to_use"] = degrees_to_use;
            ctx->saved_data["Dc"]             = Dc;
            ctx->saved_data["E"]              = E;
            ctx->saved_data["color_post"]     = color_post;
            ctx->saved_data["has_depth"]      = has_depth;
            ctx->saved_data["depth_is_zero"]  = depth_is_zero;
            ctx->saved_data["extra_has_c"]    = extra_has_c;
            ctx->saved_data["need_means"]     = need_means;
            ctx->saved_data["need_campos"]    = need_campos;
            ctx->saved_data["need_coeffs"]    = need_coeffs;
            ctx->saved_data["need_extra"]     = need_extra;
            ctx->saved_data["need_depths"]    = need_depths;
            return out;
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs
        )
        {
            const auto saved               = ctx->get_saved_variables();
            const at::Tensor &means_c      = saved[0];
            const at::Tensor &campos_c     = saved[1];
            const at::Tensor &coeffs_c     = saved[2];
            const at::Tensor &masks_t      = saved[3];
            const at::Tensor &relu_mask    = saved[4];
            at::optional<at::Tensor> masks = masks_t.defined() ? at::optional<at::Tensor>(masks_t) : c10::nullopt;

            const int64_t degrees_to_use = ctx->saved_data["degrees_to_use"].toInt();
            const int64_t Dc             = ctx->saved_data["Dc"].toInt();
            const int64_t E              = ctx->saved_data["E"].toInt();
            const int64_t color_post     = ctx->saved_data["color_post"].toInt();
            const bool has_depth         = ctx->saved_data["has_depth"].toBool();
            const bool depth_is_zero     = ctx->saved_data["depth_is_zero"].toBool();
            const bool extra_has_c       = ctx->saved_data["extra_has_c"].toBool();

            at::Tensor g = grad_outputs[0].contiguous();

            const bool need_means  = ctx->saved_data["need_means"].toBool();
            const bool need_campos = ctx->saved_data["need_campos"].toBool();
            const bool need_coeffs = ctx->saved_data["need_coeffs"].toBool();
            const bool need_extra  = ctx->saved_data["need_extra"].toBool();
            const bool need_depths = ctx->saved_data["need_depths"].toBool();
            const bool need_dir    = need_means || need_campos;

            at::Tensor v_means, v_campos, v_coeffs, v_extra, v_depths;

            if(need_means || need_campos || need_coeffs)
            {
                const int64_t batch_ndim = means_c.dim() - 2;
                // dirs = means[..., None, :, :] - campos[..., None, :] -> [*b, C, N, 3]
                at::Tensor dirs          = means_c.unsqueeze(batch_ndim).sub(campos_c.unsqueeze(-2)).contiguous();
                at::Tensor g_color       = g.narrow(-1, 0, Dc);
                if(color_post == 2 && relu_mask.defined())
                {
                    g_color = g_color.mul(relu_mask);
                }
                g_color = g_color.contiguous();

                at::Tensor v_coeffs_accum = at::zeros(coeffs_c.sizes(), coeffs_c.options().dtype(at::kFloat));
                at::Tensor v_dirs;
                if(need_dir)
                {
                    v_dirs = at::zeros_like(dirs);
                }
                launch_spherical_harmonics_bwd_kernel(
                    static_cast<uint32_t>(degrees_to_use),
                    dirs,
                    coeffs_c,
                    masks,
                    g_color,
                    v_coeffs_accum,
                    need_dir ? at::optional<at::Tensor>(v_dirs) : c10::nullopt
                );
                if(need_coeffs)
                {
                    v_coeffs = (coeffs_c.scalar_type() == at::kFloat) ? v_coeffs_accum
                                                                      : v_coeffs_accum.to(coeffs_c.scalar_type());
                }
                if(need_dir)
                {
                    if(need_means)
                    {
                        v_means = v_dirs.sum(batch_ndim); // reduce over C
                    }
                    if(need_campos)
                    {
                        v_campos = v_dirs.sum(-2).neg(); // reduce over N
                    }
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
            torch::autograd::variable_list grads(17);
            grads[11] = v_means;
            grads[12] = v_campos;
            grads[13] = v_coeffs;
            grads[14] = v_extra;
            grads[15] = v_depths;
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
    const at::Tensor &campos,
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
        campos,
        coeffs,
        extra,
        depths,
        masks
    );
}

std::tuple<at::Tensor, at::optional<at::Tensor>> spherical_harmonics_bwd_privateuseone(
    int64_t degrees_to_use,
    const at::Tensor &dirs,                // [..., N, 3]
    const at::Tensor &coeffs,              // [N, K, D]
    const at::optional<at::Tensor> &masks, // [..., N]
    const at::Tensor &v_colors,            // [..., N, D]
    bool compute_v_dirs
)
{
    DEVICE_GUARD(dirs);
    check_spherical_harmonics_inputs(degrees_to_use, dirs, coeffs, masks);
    CHECK_INPUT(v_colors);
    TORCH_CHECK(
        v_colors.size(-1) == coeffs.size(-1),
        "v_colors last dim (",
        v_colors.size(-1),
        ") must match coeffs last dim (",
        coeffs.size(-1),
        ")"
    );

    at::Tensor v_coeffs_accum = at::zeros(coeffs.sizes(), coeffs.options().dtype(at::kFloat));
    at::Tensor v_dirs;
    if(compute_v_dirs)
    {
        v_dirs = at::zeros_like(dirs);
    }

    launch_spherical_harmonics_bwd_kernels(
        degrees_to_use,
        dirs,
        coeffs,
        masks,
        v_colors,
        v_coeffs_accum,
        v_dirs.defined() ? at::optional<at::Tensor>(v_dirs) : c10::nullopt
    );

    at::Tensor v_coeffs
        = (coeffs.scalar_type() == at::kFloat) ? v_coeffs_accum : v_coeffs_accum.to(coeffs.scalar_type());
    return std::make_tuple(v_coeffs, as_optional_tensor(v_dirs));
}

void register_spherical_harmonics_cuda_impl(torch::Library &m)
{
    m.impl("spherical_harmonics", to_torch_op<&spherical_harmonics_fwd>);
    m.impl("spherical_harmonics_bwd", to_torch_op<&spherical_harmonics_bwd>);
    m.impl("assemble_proj_features_unpacked_fwd", &assemble_proj_features_unpacked_fwd);
}

void register_spherical_harmonics_privateuseone_impl(torch::Library &m)
{
    m.impl("spherical_harmonics", to_torch_op<&spherical_harmonics_fwd_privateuseone>);
    m.impl("spherical_harmonics_bwd", to_torch_op<&spherical_harmonics_bwd_privateuseone>);
}
} // namespace gsplat
