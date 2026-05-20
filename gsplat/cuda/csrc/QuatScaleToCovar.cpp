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

#include "Config.h"

#if GSPLAT_BUILD_3DGS

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD

// TODO: replacing the following with per-operation kernels might make compile
// faster.
// https://github.com/pytorch/pytorch/blob/740ce0fa5f8c7e9e51422b614f8187ab93a60b8b/aten/src/ATen/native/cuda/ScanKernels.cpp#L8-L17
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "Common.h"           // where all the macros are defined
#include "QuatScaleToCovar.h" // where the launch function is declared
#include "TorchUtils.h"

namespace gsplat {

namespace {

void check_quat_scale_to_covar_preci_inputs(
    const at::Tensor &quats, const at::Tensor &scales
) {
    TORCH_CHECK(
        quats.dim() >= 1,
        "quats must have shape [..., 4], got ",
        quats.sizes()
    );
    TORCH_CHECK(
        scales.dim() == quats.dim(),
        "scales must have shape [..., 3], got ",
        scales.sizes()
    );

    const int64_t batch_ndim = quats.dim() - 1;
    TORCH_CHECK(
        quats.size(batch_ndim) == 4,
        "quats must have shape [..., 4], got ",
        quats.sizes()
    );

    at::DimVector scales_shape(quats.sizes().slice(0, batch_ndim));
    scales_shape.append({3});
    TORCH_CHECK(
        scales.sizes() == scales_shape,
        "scales must have shape [..., 3], got ",
        scales.sizes()
    );
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
}

} // namespace

// Converts quaternion/scale parameterization into covariance and/or precision
// matrices, optionally returning only the upper-triangular representation.
struct QuatScaleToCovarPreciFwdResult {
    at::optional<at::Tensor> covars;
    at::optional<at::Tensor> precis;
};

template <> struct TorchArgDef<QuatScaleToCovarPreciFwdResult> {
    static auto to(const QuatScaleToCovarPreciFwdResult &r) { return to_torch_args(r.covars, r.precis); }
};

// Public result is the raw forward output for this op.
using QuatScaleToCovarPreciResult = QuatScaleToCovarPreciFwdResult;

QuatScaleToCovarPreciFwdResult quat_scale_to_covar_preci_fwd(
    const at::Tensor &quats,  // [..., 4]
    const at::Tensor &scales, // [..., 3]
    bool compute_covar,
    bool compute_preci,
    bool triu
) {
    DEVICE_GUARD(quats);

    check_quat_scale_to_covar_preci_inputs(quats, scales);

    auto opt = quats.options();
    at::DimVector out_shape(quats.sizes().slice(0, quats.dim() - 1));
    if (triu) {
        out_shape.append({6});
    } else {
        out_shape.append({3, 3});
    }

    at::Tensor covars, precis;
    if (compute_covar) covars = at::empty(out_shape, opt);
    if (compute_preci) precis = at::empty(out_shape, opt);

    launch_quat_scale_to_covar_preci_fwd_kernel(
        quats,
        scales,
        triu,
        compute_covar ? at::optional<at::Tensor>(covars) : at::nullopt,
        compute_preci ? at::optional<at::Tensor>(precis) : at::nullopt
    );

    return QuatScaleToCovarPreciFwdResult{
        .covars = compute_covar ? at::optional<at::Tensor>(covars) : at::nullopt,
        .precis = compute_preci ? at::optional<at::Tensor>(precis) : at::nullopt,
    };
}

struct QuatScaleToCovarPreciBwdResult {
    at::Tensor v_quats;
    at::Tensor v_scales;
};

// Gradients of the differentiable forward outputs.
struct QuatScaleToCovarPreciGrad {
    static constexpr bool is_grad_bundle = true;
    at::optional<at::Tensor> covars;
    at::optional<at::Tensor> precis;
};

QuatScaleToCovarPreciBwdResult quat_scale_to_covar_preci_bwd(
    const at::Tensor &quats,  // [..., 4]
    const at::Tensor &scales, // [..., 3]
    bool triu,
    const QuatScaleToCovarPreciGrad &grad
) {
    DEVICE_GUARD(quats);
    check_quat_scale_to_covar_preci_inputs(quats, scales);

    if (grad.covars.has_value()) {
        CHECK_INPUT(grad.covars.value());
    }
    if (grad.precis.has_value()) {
        CHECK_INPUT(grad.precis.value());
    }

    // kernel with directly write values into these tensors so we could empty
    // init them.
    at::Tensor v_scales = at::empty_like(scales);
    at::Tensor v_quats = at::empty_like(quats);

    if (grad.covars.has_value() || grad.precis.has_value()) {
        launch_quat_scale_to_covar_preci_bwd_kernel(
            quats, scales, triu, grad.covars, grad.precis, v_quats, v_scales
        );
    } else {
        // if no gradients are provided, just zero out the tensors.
        v_scales.zero_();
        v_quats.zero_();
    }

    return QuatScaleToCovarPreciBwdResult{
        .v_quats = v_quats,
        .v_scales = v_scales,
    };
}

namespace {

at::optional<at::Tensor> dense_contiguous(const at::optional<at::Tensor> &grad) {
    if (!grad.has_value()) {
        return at::nullopt;
    }
    const at::Tensor &g = grad.value();
    return (g.is_sparse() ? g.to_dense() : g).contiguous();
}

class QuatScaleToCovarPreciAutograd
    : public torch::autograd::Function<QuatScaleToCovarPreciAutograd> {
public:
    // Forward-input positions; COUNT sizes the returned grad list.
    struct FwdInput {
        enum {
            QUATS, SCALES,
            COMPUTE_COVAR, COMPUTE_PRECI, TRIU,
            COUNT,
        };
    };

    // Forward-output positions, in forward()'s return order.
    struct FwdOutput {
        enum { COVARS, PRECIS, COUNT };
    };

    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        const at::Tensor &quats, const at::Tensor &scales,
        bool compute_covar, bool compute_preci, bool triu
    ) {
        static_assert(
            FwdInput::COUNT == fwd_input_count<&forward>(),
            "FwdInput must have one enumerator per forward input"
        );

        // Keep absent output grads undefined (not materialized zeros) so the
        // backward correctly skips disabled slots:
        // - disabled outputs (compute_covar / compute_preci == false) are
        //   returned as zero-length non-differentiable sentinels;
        // - default materialize_grads would hand backward a defined [0] grad
        //   for them, which the CUDA kernel then indexes as [N, 3, 3] / [N, 6]
        //   per gaussian, reading past the empty buffer (out-of-bounds).
        ctx->set_materialize_grads(false);

        // --- Run forward --------------------------------------------------
        QuatScaleToCovarPreciFwdResult outputs = quat_scale_to_covar_preci_fwd(
            quats, scales, compute_covar, compute_preci, triu
        );

        // --- Save state for backward --------------------------------------
        ctx_save<&quat_scale_to_covar_preci_bwd>(ctx, quats, scales, triu);

        // --- Normalize optional outputs for autograd ----------------------
        // The dispatcher-facing wrapper returns optionals, but
        // torch::autograd::Function::apply returns a variable_list. Keep
        // disabled outputs as non-differentiable sentinels inside the custom
        // Function, then restore nullopt in quat_scale_to_covar_preci().
        at::Tensor covars_output = compute_covar
            ? outputs.covars.value()
            : at::empty({0}, quats.options());
        at::Tensor precis_output = compute_preci
            ? outputs.precis.value()
            : at::empty({0}, quats.options());
        torch::autograd::variable_list non_differentiable;
        if (!compute_covar) {
            non_differentiable.push_back(covars_output);
        }
        if (!compute_preci) {
            non_differentiable.push_back(precis_output);
        }
        if (!non_differentiable.empty()) {
            ctx->mark_non_differentiable(non_differentiable);
        }

        torch::autograd::variable_list out(FwdOutput::COUNT);
        out[FwdOutput::COVARS] = covars_output;
        out[FwdOutput::PRECIS] = precis_output;
        return out;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        QuatScaleToCovarPreciGrad grad{
            .covars = dense_contiguous(as_optional_tensor(grad_outputs[FwdOutput::COVARS])),
            .precis = dense_contiguous(as_optional_tensor(grad_outputs[FwdOutput::PRECIS])),
        };
        QuatScaleToCovarPreciBwdResult g = apply_bwd<&quat_scale_to_covar_preci_bwd>(ctx, grad);

        torch::autograd::variable_list grads(FwdInput::COUNT);
        grads[FwdInput::QUATS]  = g.v_quats;
        grads[FwdInput::SCALES] = g.v_scales;
        return grads;
    }

    // Forwards to apply() and packs the variable_list into the typed result.
    static QuatScaleToCovarPreciResult call(
        const at::Tensor &quats, const at::Tensor &scales,
        bool compute_covar, bool compute_preci, bool triu
    ) {
        torch::autograd::variable_list outputs =
            apply(quats, scales, compute_covar, compute_preci, triu);
        at::optional<at::Tensor> covars;
        if (compute_covar) {
            covars = outputs[FwdOutput::COVARS];
        }
        at::optional<at::Tensor> precis;
        if (compute_preci) {
            precis = outputs[FwdOutput::PRECIS];
        }
        return {.covars = covars, .precis = precis};
    }
};

} // namespace

void register_quat_scale_to_covar_cuda_impl(torch::Library &m) {
    m.impl("quat_scale_to_covar_preci", to_torch_op<&quat_scale_to_covar_preci_fwd>);
}

void register_quat_scale_to_covar_autograd_cuda_impl(torch::Library &m) {
    // No fwd-only guard: the custom forward is light, and apply() already
    // avoids recording a backward node when autograd is inactive.
    m.impl("quat_scale_to_covar_preci", to_torch_op<&QuatScaleToCovarPreciAutograd::call>);
}

} // namespace gsplat

#endif
