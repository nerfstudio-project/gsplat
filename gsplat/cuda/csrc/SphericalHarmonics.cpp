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

namespace gsplat {

namespace {

void check_spherical_harmonics_inputs(
    int64_t degrees_to_use,
    const at::Tensor &dirs,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &masks
) {
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
    TORCH_CHECK(dirs.dim() >= 2, "dirs must have shape [..., N, 3], got ", dirs.sizes());
    TORCH_CHECK(
        coeffs.dim() == 3,
        "coeffs must have shape [N, K, D], got ",
        coeffs.sizes()
    );
    TORCH_CHECK(
        coeffs.size(-1) >= 1,
        "coeffs last dim D must be >= 1, got ",
        coeffs.size(-1)
    );
    TORCH_CHECK(
        dirs.size(-2) == coeffs.size(-3),
        "dirs N (",
        dirs.size(-2),
        ") must match coeffs N (",
        coeffs.size(-3),
        ")"
    );
    TORCH_CHECK(
        (degrees_to_use + 1) * (degrees_to_use + 1) <= coeffs.size(-2),
        "degrees_to_use requires more SH coefficients than provided; degree ",
        degrees_to_use,
        ", coeffs shape ",
        coeffs.sizes()
    );
    if (masks.has_value()) {
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
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
}

} // namespace

// Spherical harmonics
struct SphericalHarmonicsFwdResult {
    at::Tensor colors;
};
template <> struct TorchArgDef<SphericalHarmonicsFwdResult> {
    static auto to(const SphericalHarmonicsFwdResult &r) { return to_torch_args(r.colors); }
};

struct SphericalHarmonicsBwdResult {
    at::Tensor v_coeffs;
    at::Tensor v_dirs;
};

// Gradients of the differentiable forward outputs.
struct SphericalHarmonicsGrad {
    static constexpr bool is_grad_bundle = true;
    at::Tensor colors; // [..., N, D]
};

SphericalHarmonicsFwdResult spherical_harmonics_fwd(
    int64_t degrees_to_use,
    const at::Tensor &dirs,               // [..., N, 3]
    const at::Tensor &coeffs,             // [N, K, D]
    const at::optional<at::Tensor> &masks // [..., N]
) {
    DEVICE_GUARD(dirs);
    check_spherical_harmonics_inputs(degrees_to_use, dirs, coeffs, masks);

    // colors dtype follows dirs; the kernel converts fp16 coeffs to fp32 on read.
    auto out_shape = dirs.sizes().vec();
    out_shape.back() = coeffs.size(-1);
    at::Tensor colors = at::empty(out_shape, dirs.options()); // [..., N, D]

    launch_spherical_harmonics_fwd_kernel(
        degrees_to_use, dirs, coeffs, masks, colors
    );
    return SphericalHarmonicsFwdResult{
        .colors = colors,
    };
}

SphericalHarmonicsBwdResult spherical_harmonics_bwd(
    int64_t degrees_to_use,
    const at::Tensor &dirs,                // [..., N, 3]
    const at::Tensor &coeffs,              // [N, K, D]
    const at::optional<at::Tensor> &masks, // [..., N]
    const SphericalHarmonicsGrad &grad
) {
    DEVICE_GUARD(dirs);
    check_spherical_harmonics_inputs(degrees_to_use, dirs, coeffs, masks);
    TORCH_INTERNAL_ASSERT(grad.colors.defined());
    CHECK_INPUT(grad.colors);
    TORCH_CHECK(
        grad.colors.size(-1) == coeffs.size(-1),
        "v_colors last dim (",
        grad.colors.size(-1),
        ") must match coeffs last dim (",
        coeffs.size(-1),
        ")"
    );

    // dirs gradients are only computed when the saved dirs tensor requires them.
    const bool compute_v_dirs = dirs.requires_grad();

    // Always accumulate v_coeffs in fp32 to avoid precision loss when multiple
    // (batch, gaussian) elements atomic-add into the same slot. For fp32
    // coeffs the accumulator IS the output; for fp16 we cast at the end.
    at::Tensor v_coeffs_accum =
        at::zeros(coeffs.sizes(), coeffs.options().dtype(at::kFloat));
    at::Tensor v_dirs;
    if (compute_v_dirs) {
        v_dirs = at::zeros_like(dirs);
    }

    launch_spherical_harmonics_bwd_kernel(
        degrees_to_use,
        dirs,
        coeffs,
        masks,
        grad.colors,
        v_coeffs_accum,
        v_dirs.defined() ? at::optional<at::Tensor>(v_dirs) : c10::nullopt
    );

    at::Tensor v_coeffs = (coeffs.scalar_type() == at::kFloat)
        ? v_coeffs_accum
        : v_coeffs_accum.to(coeffs.scalar_type());
    return SphericalHarmonicsBwdResult{
        .v_coeffs = v_coeffs,
        .v_dirs = v_dirs,
    };
}

namespace {

class SphericalHarmonicsAutograd
    : public torch::autograd::Function<SphericalHarmonicsAutograd> {
public:
    // Forward-input positions; COUNT sizes the returned grad list.
    struct FwdInput {
        enum {
            DEGREES_TO_USE,
            DIRS, COEFFS, MASKS,
            COUNT,
        };
    };

    // Forward-output positions, in forward()'s return order.
    struct FwdOutput {
        enum { COLORS, COUNT };
    };

    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        int64_t degrees_to_use,
        const at::Tensor &dirs, const at::Tensor &coeffs,
        const at::optional<at::Tensor> &masks
    ) {
        static_assert(
            FwdInput::COUNT == fwd_input_count<&forward>(),
            "FwdInput must have one enumerator per forward input"
        );

        // --- Run forward --------------------------------------------------
        SphericalHarmonicsFwdResult outputs = spherical_harmonics_fwd(
            degrees_to_use, dirs, coeffs, masks
        );

        // --- Save state for backward --------------------------------------
        ctx_save<&spherical_harmonics_bwd>(
            ctx, degrees_to_use, dirs, coeffs, masks
        );

        torch::autograd::variable_list out(FwdOutput::COUNT);
        out[FwdOutput::COLORS] = outputs.colors;
        return out;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        SphericalHarmonicsGrad grad{
            .colors = grad_outputs[FwdOutput::COLORS].contiguous(),
        };
        SphericalHarmonicsBwdResult g = apply_bwd<&spherical_harmonics_bwd>(ctx, grad);

        torch::autograd::variable_list grads(FwdInput::COUNT);
        grads[FwdInput::DIRS]   = g.v_dirs;
        grads[FwdInput::COEFFS] = g.v_coeffs;
        return grads;
    }

    // Forwards to apply() and unpacks the single color output.
    static at::Tensor call(
        int64_t degrees_to_use,
        const at::Tensor &dirs, const at::Tensor &coeffs,
        const at::optional<at::Tensor> &masks
    ) {
        torch::autograd::variable_list outputs =
            apply(degrees_to_use, dirs, coeffs, masks);
        return outputs[FwdOutput::COLORS];
    }
};

} // namespace

at::Tensor spherical_harmonics(
    int64_t degrees_to_use,
    const at::Tensor &dirs, const at::Tensor &coeffs,
    const at::optional<at::Tensor> &masks
) {
    // No fwd-only guard here: the custom forward is light, and apply() already
    // avoids recording a backward node when autograd is inactive.
    return SphericalHarmonicsAutograd::call(degrees_to_use, dirs, coeffs, masks);
}

void register_spherical_harmonics_cuda_impl(torch::Library &m) {
    m.impl("spherical_harmonics", to_torch_op<&spherical_harmonics_fwd>);
}

void register_spherical_harmonics_autograd_cuda_impl(torch::Library &m) {
    m.impl("spherical_harmonics", to_torch_op<&spherical_harmonics>);
}

} // namespace gsplat
