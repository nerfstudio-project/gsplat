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

// Config.h is resolved via extra_include_paths set to gsplat/cuda/csrc/
#include "Config.h"

#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/ones_like.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <utility>

#include "Common.h"    // CHECK_INPUT, DEVICE_GUARD, CameraModelType
#include "GaussianRenderInferenceScene.h"

// Viewer kernel headers
#include "Projection.h"
#include "SphericalHarmonics.h"
#include "IntersectMTFused.h"
#include "SHCompression.h"
#include "InferenceTypes.h"

namespace gsplat {
namespace gaussian_render_inference_scene {

static constexpr float SH_ACTIVATION_SCALE = 0.5f;
static constexpr float SH_ACTIVATION_SHIFT = 0.0f;

InferenceRenderState::~InferenceRenderState() = default;

// ==========================================================================
// create_gaussian_render_inference_scene_state
// ==========================================================================
std::unique_ptr<InferenceRenderState> create_gaussian_render_inference_scene_state(
    const InferenceRenderScene& scene,
    int64_t sh_compression_mode
) {
    DEVICE_GUARD(scene.means_planar);

    auto state = std::make_unique<InferenceRenderState>();

    state->num_gaussians = static_cast<uint32_t>(scene.means_planar.size(1));  // [3, N]
    if (state->num_gaussians == 0) return state;

    auto opts_f = at::TensorOptions().dtype(at::kFloat).device(scene.means_planar.device());
    auto opts_h = at::TensorOptions().dtype(at::kHalf).device(scene.means_planar.device());
    auto opts_i = at::TensorOptions().dtype(at::kInt).device(scene.means_planar.device());

    // ---- Determine K from shape ----
    if (scene.sh_degree >= 0 && scene.colors_packed.dim() == 3) {
        state->sh_coeffs_per_channel = static_cast<uint32_t>(scene.colors_packed.size(1));
    } else {
        state->sh_coeffs_per_channel = 0;
    }
    state->maxShDegree = (state->sh_coeffs_per_channel > 0)
        ? static_cast<uint32_t>(std::sqrt(static_cast<double>(state->sh_coeffs_per_channel))) - 1
        : 0;

    // ---- SH compression at creation time for K==16 ----
    // sh_compression_mode: 0=NONE (skip), 1=COMPRESS_32B only, 2=COMPRESS_16B only.
    TORCH_CHECK(
        sh_compression_mode >= 0 && sh_compression_mode <= 2,
        "sh_compression_mode must be 0 (NONE), 1 (COMPRESS_32B), or 2 (COMPRESS_16B); got ",
        sh_compression_mode);
    state->shCompressionMode = sh_compression_mode;
    TORCH_CHECK(
        sh_compression_mode == 0 || state->sh_coeffs_per_channel == 16,
        "sh_compression_mode requires SH degree 3 (K==16); got K=",
        state->sh_coeffs_per_channel);
    if (state->sh_coeffs_per_channel == 16 && sh_compression_mode != 0) {
        auto sh_float = scene.colors_packed.contiguous().to(at::kFloat);
        auto opacities_float = scene.qso_packed.contiguous().narrow(1, 7, 1).squeeze(1).to(at::kFloat);
        auto compressed = higs::launch_sh_compress(
            sh_float, opacities_float, static_cast<SHCompressionMode>(sh_compression_mode));
        state->shCompressed = compressed.packed;
        state->shDecodeParams = std::make_unique<higs::SHDecodeParams>(compressed.decode_params);
        torch::cuda::synchronize();
    }

    // ---- Pre-allocate per-frame intermediates ----
    int64_t num_gaussians = static_cast<int64_t>(state->num_gaussians);
    state->visible = at::zeros({(num_gaussians + 31) / 32}, opts_i);
    state->means2d = at::empty({1, 1, num_gaussians, 2}, opts_f);
    state->depths  = at::empty({1, 1, num_gaussians}, opts_f);
    state->conics  = at::empty({1, 1, num_gaussians, 4}, opts_h);
    state->colors  = at::zeros({num_gaussians, 4}, opts_h);

    // ---- Pre-allocate default black background: [R, G, B, T] ----
    state->background = at::zeros({1, 4}, opts_h);
    state->background.select(1, 3).fill_(at::Half(1.0f));

    // ---- Create intersection stage ----
    state->isect = std::make_unique<IntersectMTFused>();

    return state;
}

// ==========================================================================
// release_gaussian_render_inference_scene_state
// ==========================================================================
void release_gaussian_render_inference_scene_state(InferenceRenderState& state) {
    InferenceRenderState empty;
    std::swap(state, empty);
}

// ==========================================================================
// render_gaussian_inference_scene
// ==========================================================================
at::Tensor render_gaussian_inference_scene(
    InferenceRenderState& state,
    const InferenceRenderScene& scene,
    const at::Tensor& viewmat,
    const at::Tensor& K,
    int64_t width, int64_t height,
    int64_t tile_size,
    double near_plane, double far_plane,
    double radius_clip, double eps2d,
    int64_t sh_degree,
    int64_t sh_compression_mode,
    const at::optional<at::Tensor>& background,
    const at::optional<at::Tensor>& out_rgbt
) {
    // NOTE: No c10::NoGradGuard here -- the Python caller already enforces
    // torch.inference_mode() (via check_inference_grad_mode), so adding a
    // redundant guard would cost ~2 us per frame in thread-local toggles.
    DEVICE_GUARD(scene.means_planar);

    auto opts_h = at::TensorOptions().dtype(at::kHalf).device(scene.means_planar.device());

    // ---- Validate out_rgbt if provided ----
    if (out_rgbt.has_value()) {
        const auto& buf = out_rgbt.value();
        TORCH_CHECK(buf.is_cuda(), "out_rgbt must be a CUDA tensor");
        TORCH_CHECK(buf.dtype() == at::kHalf, "out_rgbt must be float16");
        TORCH_CHECK(buf.is_contiguous(), "out_rgbt must be contiguous");
        TORCH_CHECK(buf.dim() == 4, "out_rgbt must be 4D");
        TORCH_CHECK(
            buf.size(0) == 1 && buf.size(1) == height &&
                buf.size(2) == width && buf.size(3) == 4,
            "out_rgbt must have shape [1, ", height, ", ", width,
            ", 4], got [", buf.size(0), ", ", buf.size(1), ", ",
            buf.size(2), ", ", buf.size(3), "]");
    }

    // ---- Early return for empty scene ----
    if (state.num_gaussians == 0) {
        at::Tensor rgbt = out_rgbt.has_value()
            ? out_rgbt.value() : at::zeros({1, height, width, 4}, opts_h);
        rgbt.zero_();
        rgbt.select(3, 3).fill_(at::Half(1.0f));
        if (background.has_value()) {
            rgbt.narrow(3, 0, 3).copy_(
                background.value().reshape({1, 1, 1, 3}).to(at::kHalf));
        }
        return rgbt;
    }

    // ---- Tile grid dimensions ----
    uint32_t tw = (static_cast<uint32_t>(width) + static_cast<uint32_t>(tile_size) - 1)
                  / static_cast<uint32_t>(tile_size);
    uint32_t th = (static_cast<uint32_t>(height) + static_cast<uint32_t>(tile_size) - 1)
                  / static_cast<uint32_t>(tile_size);

    // ---- Clamp SH degree ----
    TORCH_CHECK(
        sh_compression_mode == state.shCompressionMode,
        "render sh_compression_mode (", sh_compression_mode,
        ") must match the mode used to create the inference render state (",
        state.shCompressionMode, ")");
    auto compression = static_cast<SHCompressionMode>(state.shCompressionMode);
    if (state.sh_coeffs_per_channel > 0) {
        sh_degree = std::min(sh_degree, static_cast<int64_t>(state.maxShDegree));
    }

    // ---- Get CUDA stream ----
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ---- Camera views for kernels; reshape is metadata-only for valid inputs. ----
    TORCH_CHECK(viewmat.is_contiguous(), "viewmat must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    auto viewmat_4d = viewmat.reshape({1, 1, 4, 4});
    auto K_4d = K.reshape({1, 1, 3, 3});

    // ---- Set background only when it changes from the cached default. ----
    if (background.has_value()) {
        state.background.narrow(1, 0, 3).copy_(
            background.value().unsqueeze(0).to(at::kHalf));
        state.backgroundIsDefault = false;
    } else if (!state.backgroundIsDefault) {
        state.background.zero_();
        state.background.select(1, 3).fill_(at::Half(1.0f));
        state.backgroundIsDefault = true;
    }

    // ---- Select output framebuffer ----
    // Python callers normally pass a pre-allocated out tensor; the allocation
    // fallback keeps the C++ helper usable for direct calls and the stateless
    // compatibility wrapper.  Output is always half (fp16); callers that need
    // float32 convert after the call.
    at::Tensor rgbt = out_rgbt.has_value()
        ? out_rgbt.value()
        : at::empty({1, height, width, 4}, opts_h);

    // ==================================================================
    // Projection + SH evaluation
    // ==================================================================
    const auto& means = scene.means_planar;
    const auto& qso_packed = scene.qso_packed;
    const auto& colors_packed = scene.colors_packed;
    const at::Tensor* raster_colors = &state.colors;

    if (state.sh_coeffs_per_channel > 0 && compression != SHCompressionMode::NONE) {
        // ---- Compressed SH — fused projection + compressed SH decode (K==16 validated at creation) ----
        const higs::SHDecodeParams* decode_params = state.shDecodeParams.get();
        TORCH_CHECK(
            decode_params != nullptr,
            "compressed SH decode parameters are missing for sh_compression_mode=",
            state.shCompressionMode);

        higs::launch_projection_sh_fused_kernel(
            means, {}, qso_packed, viewmat_4d, K_4d,
            static_cast<uint32_t>(width), static_cast<uint32_t>(height),
            static_cast<float>(eps2d), static_cast<float>(near_plane),
            static_cast<float>(far_plane), static_cast<float>(radius_clip),
            gsplat::CameraModelType::PINHOLE,
            static_cast<int32_t>(sh_degree), state.shCompressed, SH_ACTIVATION_SCALE, SH_ACTIVATION_SHIFT,
            compression, decode_params,
            state.visible, state.means2d, state.depths, state.conics, state.colors, {});

    } else if (state.sh_coeffs_per_channel > 0) {
        // ---- Uncompressed SH — fused projection + SH for any degree ----
        // The fused kernel handles all SH degrees; only compressed mode
        // is restricted to degree 3.
        higs::launch_projection_sh_fused_kernel(
            means, {}, qso_packed, viewmat_4d, K_4d,
            static_cast<uint32_t>(width), static_cast<uint32_t>(height),
            static_cast<float>(eps2d), static_cast<float>(near_plane),
            static_cast<float>(far_plane), static_cast<float>(radius_clip),
            gsplat::CameraModelType::PINHOLE,
            static_cast<int32_t>(sh_degree), colors_packed, SH_ACTIVATION_SCALE, SH_ACTIVATION_SHIFT,
            SHCompressionMode::NONE, nullptr,
            state.visible, state.means2d, state.depths, state.conics, state.colors, {});

    } else {
        // ---- Pre-activated RGB: projection only + copy colors ----
        higs::launch_projection_fwd_kernel(
            means, {}, qso_packed, viewmat_4d, K_4d,
            static_cast<uint32_t>(width), static_cast<uint32_t>(height),
            static_cast<float>(eps2d), static_cast<float>(near_plane),
            static_cast<float>(far_plane), static_cast<float>(radius_clip),
            gsplat::CameraModelType::PINHOLE,
            state.visible, state.means2d, state.depths, state.conics, {});

        // colors_packed is already [N, 4] half {R, G, B, 0}
        raster_colors = &colors_packed;
    }

    // ==================================================================
    // Intersection + Rasterization (fused macro-tile path)
    // ==================================================================
    state.isect->execute(
        state.means2d, state.depths, state.conics, state.visible,
        static_cast<int32_t>(tile_size),
        static_cast<int32_t>(tw), static_cast<int32_t>(th),
        stream);

    state.isect->rasterize(
        state.means2d, state.conics, *raster_colors, state.background,
        static_cast<uint32_t>(width), static_cast<uint32_t>(height),
        rgbt);

    // ==================================================================
    // Return native half4 RGBT output
    // ==================================================================
    return rgbt;
}

// ==========================================================================
// GaussianInferenceRenderer — pybind wrapper
// ==========================================================================

GaussianInferenceRenderer::~GaussianInferenceRenderer() = default;

GaussianInferenceRenderer::GaussianInferenceRenderer(
    const at::Tensor& means_planar,
    const at::Tensor& qso_packed,
    const at::Tensor& colors_packed,
    int64_t sh_degree,
    int64_t sh_compression_mode
) {
    // Normalize SH3 pre-packed 2-D (N, 48) tensors to 3-D (N, 16, 3) once at
    // construction time.  The cached view is reused in render() to avoid a
    // per-frame reshape + contiguous check.
    int64_t N = means_planar.size(1);
    if (sh_degree == 3 && colors_packed.dim() == 2 && colors_packed.size(1) == 48) {
        colors_normalized_ = colors_packed.reshape({N, 16, 3}).contiguous();
    } else {
        colors_normalized_ = colors_packed;
    }

    InferenceRenderScene scene;
    scene.means_planar = means_planar;
    scene.qso_packed = qso_packed;
    scene.colors_packed = colors_normalized_;
    scene.sh_degree = sh_degree;
    scene.sh_compression_mode = sh_compression_mode;

    state_ = create_gaussian_render_inference_scene_state(scene, sh_compression_mode);
}

at::Tensor GaussianInferenceRenderer::render(
    const at::Tensor& means_planar,
    const at::Tensor& qso_packed,
    const at::Tensor& colors_packed,
    const at::Tensor& viewmat,
    const at::Tensor& K,
    int64_t width, int64_t height,
    int64_t tile_size,
    double near_plane, double far_plane,
    double radius_clip, double eps2d,
    int64_t sh_degree,
    int64_t sh_compression_mode,
    const at::optional<at::Tensor>& background,
    const at::optional<at::Tensor>& out_rgbt
) {
    // Use the colors tensor normalized once at construction time (colors_normalized_)
    // instead of re-normalizing every frame.  The means_planar and qso_packed are
    // passed through as-is since they don't need reshaping.
    InferenceRenderScene scene;
    scene.means_planar = means_planar;
    scene.qso_packed = qso_packed;
    scene.colors_packed = colors_normalized_;
    scene.sh_degree = sh_degree;
    scene.sh_compression_mode = sh_compression_mode;

    return render_gaussian_inference_scene(*state_, scene, viewmat, K,
                      width, height, tile_size,
                      near_plane, far_plane, radius_clip, eps2d,
                      sh_degree, sh_compression_mode,
                      background, out_rgbt);
}

void GaussianInferenceRenderer::release() {
    if (state_) {
        release_gaussian_render_inference_scene_state(*state_);
    }
}

// ==========================================================================
// gaussian_render_inference_only — torch.ops compatibility wrapper
// ==========================================================================
std::tuple<at::Tensor, at::Tensor> gaussian_render_inference_only(
    const at::Tensor& means_planar,
    const at::Tensor& qso_packed,
    const at::Tensor& colors_packed,
    const at::Tensor& viewmat,
    const at::Tensor& K,
    int64_t width,
    int64_t height,
    int64_t sh_degree,
    int64_t tile_size,
    double near_plane,
    double far_plane,
    double radius_clip,
    double eps2d,
    int64_t sh_compression_mode,
    const at::optional<at::Tensor>& background,
    const at::optional<at::Tensor>& out_renders,
    const at::optional<at::Tensor>& out_alphas
) {
    // NOTE: No c10::NoGradGuard -- Python callers enforce inference_mode().
    DEVICE_GUARD(means_planar);
    CHECK_INPUT(means_planar);
    CHECK_INPUT(qso_packed);
    CHECK_INPUT(colors_packed);
    CHECK_INPUT(viewmat);
    CHECK_INPUT(K);
    if (background.has_value()) {
        CHECK_INPUT(background.value());
    }
    if (out_renders.has_value()) {
        const auto& buf = out_renders.value();
        TORCH_CHECK(buf.is_cuda(), "out_renders must be a CUDA tensor");
        TORCH_CHECK(buf.dtype() == at::kFloat, "out_renders must be float32");
        TORCH_CHECK(buf.is_contiguous(), "out_renders must be contiguous");
        TORCH_CHECK(buf.dim() == 3 && buf.size(0) == height && buf.size(1) == width && buf.size(2) == 3,
            "out_renders must have shape [", height, ", ", width, ", 3], got [",
            buf.size(0), ", ", buf.size(1), ", ", buf.size(2), "]");
    }
    if (out_alphas.has_value()) {
        const auto& buf = out_alphas.value();
        TORCH_CHECK(buf.is_cuda(), "out_alphas must be a CUDA tensor");
        TORCH_CHECK(buf.dtype() == at::kFloat, "out_alphas must be float32");
        TORCH_CHECK(buf.is_contiguous(), "out_alphas must be contiguous");
        TORCH_CHECK(buf.dim() == 3 && buf.size(0) == height && buf.size(1) == width && buf.size(2) == 1,
            "out_alphas must have shape [", height, ", ", width, ", 1], got [",
            buf.size(0), ", ", buf.size(1), ", ", buf.size(2), "]");
    }

    int64_t N = means_planar.size(1);  // [3, N]
    auto opts_f = at::TensorOptions().dtype(at::kFloat).device(means_planar.device());
    auto opts_h = at::TensorOptions().dtype(at::kHalf).device(means_planar.device());

    // ---- Handle N==0 case ----
    if (N == 0) {
        at::Tensor renders;
        at::Tensor alphas;
        if (out_renders.has_value()) {
            renders = out_renders.value();
            renders.zero_();
        } else {
            renders = at::zeros({height, width, 3}, opts_f);
        }
        if (out_alphas.has_value()) {
            alphas = out_alphas.value();
            alphas.zero_();
        } else {
            alphas = at::zeros({height, width, 1}, opts_f);
        }
        if (background.has_value()) {
            renders.add_(background.value().reshape({1, 1, 3}));
        }
        return std::make_tuple(renders, alphas);
    }

    // ---- Normalize colors_packed for SH degree 3 pre-packed formats ----
    // The scene packer may store SH3 compressed data as 2-D (N, 48) tensors.
    // Internal rendering code expects 3-D (N, 16, 3) for K detection and SH
    // evaluation.  Reshape here so downstream code sees the canonical layout.
    at::Tensor colors_normalized = colors_packed;
    if (sh_degree == 3 && colors_packed.dim() == 2 && colors_packed.size(1) == 48) {
        colors_normalized = colors_packed.reshape({N, 16, 3}).contiguous();
    }

    // ---- Construct scene and create state for exactly the requested mode ----
    InferenceRenderScene scene;
    scene.means_planar = means_planar;
    scene.qso_packed = qso_packed;
    scene.colors_packed = colors_normalized;
    scene.sh_degree = sh_degree;
    scene.sh_compression_mode = sh_compression_mode;

    auto state = create_gaussian_render_inference_scene_state(scene, sh_compression_mode);

    // ---- Render via render_gaussian_inference_scene ----
    at::Tensor rgbt = render_gaussian_inference_scene(
        *state, scene, viewmat, K,
        width, height, tile_size,
        near_plane, far_plane, radius_clip, eps2d,
        sh_degree, sh_compression_mode,
        background, at::nullopt);

    // ---- Extract RGB and alpha from RGBT output ----
    // rgbt: [1, H, W, 4] half {R, G, B, T}
    auto output = rgbt.squeeze(0);           // [H, W, 4]
    auto rgb = output.narrow(2, 0, 3);       // [H, W, 3] -- render colors in half
    auto T = output.narrow(2, 3, 1);         // [H, W, 1] -- transmittance
    auto alphas_h = at::ones_like(T).sub_(T);  // alpha = 1 - T

    // Write results into pre-allocated buffers or allocate new ones
    at::Tensor renders_f;
    at::Tensor alphas_f;

    if (out_renders.has_value()) {
        renders_f = out_renders.value();
        renders_f.copy_(rgb);  // half -> float copy into pre-allocated buffer
    } else {
        renders_f = rgb.to(at::kFloat);     // [H, W, 3]
    }

    if (out_alphas.has_value()) {
        alphas_f = out_alphas.value();
        alphas_f.copy_(alphas_h);  // half -> float copy into pre-allocated buffer
    } else {
        alphas_f = alphas_h.to(at::kFloat);   // [H, W, 1]
    }

    // ---- Release temporary state ----
    release_gaussian_render_inference_scene_state(*state);

    return std::make_tuple(renders_f, alphas_f);
}

} // namespace gaussian_render_inference_scene
} // namespace gsplat
