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

#include "gaussian_scene_pack.cuh"

#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == at::kFloat, #x " must have dtype float32")
#define DEVICE_GUARD(x) const at::cuda::OptionalCUDAGuard device_guard(device_of(x))

namespace gsplat {
namespace scene {

enum class ShCompressionMode : int64_t {
    None = 0,
    Packed32B = 1,
    Packed16B = 2,
};

std::tuple<at::Tensor, at::Tensor, at::Tensor> pack_gaussian_inference_scene_cuda(
    const at::Tensor &means,      // [N, 3] float32
    const at::Tensor &quats,      // [N, 4] float32
    const at::Tensor &scales,     // [N, 3] float32
    const at::Tensor &opacities,  // [N]    float32
    const at::Tensor &colors,     // [N, 3] or [N, K, 3] float32
    int64_t sh_degree,            // -1 for RGB, 0-3 for SH
    int64_t sh_compression_mode   // 0=none, 1=32b, 2=16b
) {
    // Disable grad tracking for all ATen ops inside this function.
    // Without this, intermediate ops like .t() and .contiguous() on
    // grad-tracked inputs would propagate CloneBackward0 to the outputs.
    torch::NoGradGuard no_grad;

    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(opacities);
    CHECK_INPUT(colors);

    CHECK_FLOAT(means);
    CHECK_FLOAT(quats);
    CHECK_FLOAT(scales);
    CHECK_FLOAT(opacities);
    CHECK_FLOAT(colors);

    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3,
        "means must be [N, 3]; got ", means.sizes());
    TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4,
        "quats must be [N, 4]; got ", quats.sizes());
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3,
        "scales must be [N, 3]; got ", scales.sizes());
    TORCH_CHECK(opacities.dim() == 1,
        "opacities must be 1-D [N]; got dim=", opacities.dim());

    int64_t N = means.size(0);
    TORCH_CHECK(quats.size(0) == N,
        "quats.size(0) must match means.size(0)=", N, "; got ", quats.size(0));
    TORCH_CHECK(scales.size(0) == N,
        "scales.size(0) must match means.size(0)=", N, "; got ", scales.size(0));
    TORCH_CHECK(opacities.size(0) == N,
        "opacities.size(0) must match means.size(0)=", N, "; got ", opacities.size(0));
    TORCH_CHECK(colors.size(0) == N,
        "colors.size(0) must match means.size(0)=", N, "; got ", colors.size(0));

    TORCH_CHECK(sh_degree >= -1 && sh_degree <= 3,
        "sh_degree must be in [-1, 3]; got ", sh_degree);

    if (sh_degree >= 0) {
        int64_t K_expected = (sh_degree + 1) * (sh_degree + 1);
        TORCH_CHECK(colors.dim() == 3 && colors.size(1) == K_expected && colors.size(2) == 3,
            "For sh_degree=", sh_degree, ", colors must be [N, ",
            K_expected, ", 3]; got ", colors.sizes());
    } else {
        TORCH_CHECK(colors.dim() == 2 && colors.size(1) == 3,
            "For sh_degree=-1, colors must be [N, 3]; got ", colors.sizes());
    }

    TORCH_CHECK(
        sh_compression_mode >= 0 && sh_compression_mode <= 2,
        "sh_compression_mode must be one of [0, 1, 2]; got ",
        sh_compression_mode
    );
    TORCH_CHECK(
        sh_compression_mode == 0 || sh_degree == 3,
        "sh_compression_mode=",
        sh_compression_mode,
        " requires sh_degree=3; got sh_degree=",
        sh_degree
    );
    const auto sh_compression = static_cast<ShCompressionMode>(sh_compression_mode);

    auto opts_h = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);

    // means_planar: [N, 3] -> [3, N]
    auto means_planar = means.t().contiguous();

    // qso_packed [N, 8] fp16 with clamping for finite fp16 range
    constexpr float fp16_max = 65504.0f;
    auto qso_packed = at::empty({N, 8}, opts_h);
    // quats are unit-norm, values in [-1, 1], no clamping needed
    qso_packed.narrow(1, 0, 4).copy_(quats.to(at::kHalf));
    // scales: positive but may be large; clamp to fp16 range
    qso_packed.narrow(1, 4, 3).copy_(scales.clamp(-fp16_max, fp16_max).to(at::kHalf));
    // opacities are sigmoid-bounded [0, 1], no clamping needed
    qso_packed.narrow(1, 7, 1).copy_(opacities.unsqueeze(1).to(at::kHalf));

    // Pack colors based on SH degree and compression mode.
    int64_t K_sh = (sh_degree >= 0 && colors.dim() == 3) ? colors.size(1) : 0;
    at::Tensor colors_packed;
    if (sh_degree >= 0 && K_sh == 16) {
        if (sh_compression == ShCompressionMode::None) {
            // SH3 "none": preserve the raw SH layout in fp16 lanes.
            colors_packed = colors.clamp(-fp16_max, fp16_max).to(at::kHalf);
        } else if (sh_compression == ShCompressionMode::Packed32B) {
            // SH3 "32b": pack into contiguous 32-bit coefficient lanes.
            colors_packed = colors.contiguous().view({N, 48});
        } else {
            // SH3 "16b": pack into contiguous fp16 coefficient lanes.
            colors_packed = colors.clamp(-fp16_max, fp16_max).to(at::kHalf).view({N, 48});
        }
    } else if (sh_degree >= 0 && K_sh > 0) {
        // Lower-degree SH (0-2): keep float32
        colors_packed = colors.contiguous();
    } else {
        // RGB: [N, 3] -> [N, 4] fp16 {R, G, B, 0} with padding for alignment
        auto c = at::zeros({N, 4}, opts_h);
        c.narrow(1, 0, 3).copy_(colors.clamp(-fp16_max, fp16_max).to(at::kHalf));
        colors_packed = c;
    }

    return std::make_tuple(means_planar, qso_packed, colors_packed);
}

} // namespace scene
} // namespace gsplat
