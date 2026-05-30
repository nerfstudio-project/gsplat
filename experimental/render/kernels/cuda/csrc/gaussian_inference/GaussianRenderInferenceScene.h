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

#pragma once

#include <ATen/core/Tensor.h>
#include <cstdint>
#include <memory>
#include <tuple>

class IntersectMTFused;

namespace higs {
struct SHDecodeParams;
} // namespace higs

namespace gsplat {
namespace gaussian_render_inference_scene {

// ---------------------------------------------------------------------------
// InferenceRenderScene — non-owning view of packed scene tensors
// ---------------------------------------------------------------------------
struct InferenceRenderScene {
    at::Tensor means_planar;        // [3, N] float32
    at::Tensor qso_packed;          // [N, 8] half
    at::Tensor colors_packed;       // [N, K, 3] half/float (SH) or [N, 4] half (pre-activated)
    int64_t sh_degree;              // maximum SH degree (-1 for pre-activated RGB)
    int64_t sh_compression_mode;    // 0=NONE, 1=COMPRESS_32B, 2=COMPRESS_16B
};

// ---------------------------------------------------------------------------
// InferenceRenderState — owns render-cache buffers
// Not thread-safe: shared buffers assume single-stream, single-thread usage.
// ---------------------------------------------------------------------------
struct InferenceRenderState {
    ~InferenceRenderState();
    InferenceRenderState() = default;
    InferenceRenderState(InferenceRenderState&&) = default;
    InferenceRenderState& operator=(InferenceRenderState&&) = default;

    // Scene metadata (derived at creation)
    uint32_t num_gaussians = 0;             // total number of gaussians in the scene
    uint32_t maxShDegree = 0;               // maximum SH degree from scene data
    uint32_t sh_coeffs_per_channel = 0;     // number of SH coefficients per color channel (K)

    // SH compression product selected at creation time for K==16.
    int64_t shCompressionMode = 0;
    at::Tensor shCompressed;
    std::unique_ptr<higs::SHDecodeParams> shDecodeParams;

    // Per-frame intermediates (persistent, sized once at creation)
    at::Tensor colors;              // [N, 4] half
    at::Tensor visible;             // [(N+31)/32] int32
    at::Tensor means2d;             // [1, 1, N, 2] float
    at::Tensor depths;              // [1, 1, N] float
    at::Tensor conics;              // [1, 1, N, 4] half

    // Intersection stage (fused macro-tile)
    std::unique_ptr<IntersectMTFused> isect;

    // Cached background buffer.
    at::Tensor background;          // [1, 4] half
    bool backgroundIsDefault = true;
};

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Allocate and initialise an InferenceRenderState from the given scene.
/// @param sh_compression_mode  Controls eager SH precomputation for K==16 scenes:
///   - 0 (NONE):         skip SH precomputation entirely.
///   - 1 (COMPRESS_32B): precompute only the 32-byte codec.
///   - 2 (COMPRESS_16B): precompute only the 16-byte codec.
///   When a mode is requested the call includes a host-GPU sync.
std::unique_ptr<InferenceRenderState> create_gaussian_render_inference_scene_state(
    const InferenceRenderScene& scene,
    int64_t sh_compression_mode = 0);

/// Release all GPU resources held by the state (reset tensors/ptrs, set num_gaussians=0).
void release_gaussian_render_inference_scene_state(InferenceRenderState& state);

/// Core render function.  Scene tensors are read from `scene`; cached
/// intermediates live in `state`.  Writes into `out_rgbt` when provided and
/// returns [1,H,W,4] float16 RGBT.
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
    const at::optional<at::Tensor>& out_rgbt);

// ---------------------------------------------------------------------------
// GaussianInferenceRenderer — pybind wrapper around state + free functions
// ---------------------------------------------------------------------------
class GaussianInferenceRenderer {
public:
    /// Construct from pre-packed scene tensors.  Creates internal
    /// InferenceRenderScene, calls create_gaussian_render_inference_scene_state, stores result.
    GaussianInferenceRenderer(
        const at::Tensor& means_planar,
        const at::Tensor& qso_packed,
        const at::Tensor& colors_packed,
        int64_t sh_degree,
        int64_t sh_compression_mode);

    ~GaussianInferenceRenderer();

    // Non-copyable, non-movable
    GaussianInferenceRenderer(const GaussianInferenceRenderer&) = delete;
    GaussianInferenceRenderer& operator=(const GaussianInferenceRenderer&) = delete;
    GaussianInferenceRenderer(GaussianInferenceRenderer&&) = delete;
    GaussianInferenceRenderer& operator=(GaussianInferenceRenderer&&) = delete;

    /// Render a frame.  Scene tensors are passed per-call (NOT stored).
    at::Tensor render(
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
        const at::optional<at::Tensor>& out_rgbt);

    /// Release all GPU resources.
    void release();

    /// Number of gaussians in the loaded scene (0 after release).
    uint32_t numGaussians() const { return state_ ? state_->num_gaussians : 0; }

    /// Whether release() has been called.
    bool isReleased() const { return !state_ || state_->num_gaussians == 0; }

private:
    std::unique_ptr<InferenceRenderState> state_;
    at::Tensor colors_normalized_;  // cached [N,K,3] view (avoids per-frame reshape)
};

// ---------------------------------------------------------------------------
// Stateless compatibility wrapper (torch.ops)
// ---------------------------------------------------------------------------

/// C++ compatibility wrapper for torch.ops.experimental.gaussian_render_inference_only.
/// Keeps the EXACT same signature as the original GaussianRenderInferenceSceneViewer implementation.
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
    const at::optional<at::Tensor>& out_alphas);

} // namespace gaussian_render_inference_scene
} // namespace gsplat
