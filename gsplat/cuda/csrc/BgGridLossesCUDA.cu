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

#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAException.h>
#    include <c10/cuda/CUDAStream.h>
#    include <cuda_runtime.h>

#    include "BgGridLosses.h"

namespace gsplat
{
namespace
{
    // A sub-loss is disabled only by a strictly negative factor. Spelled
    // `!(factor < 0)` rather than `factor >= 0` so a NaN factor stays enabled
    // and propagates (matching the pure-PyTorch reference); `factor >= 0` would
    // silently disable NaN. Centralized here so the policy is stated once.
    __device__ __forceinline__ bool is_loss_enabled(float factor)
    {
        return !(factor < 0.0f);
    }

    // ---------------------------------------------------------------------------
    // Sky env-map TV forward: for each input element at flat index ti of
    // [B*D, H, W, C] (with B*D rolled into the first dim), accumulate the sum of
    // squared one-sided forward differences along the D, H, W axes and multiply
    // by `factor`. No count normalisation is applied here; the caller-provided
    // factor carries any outer normalisation (e.g. a mean-reduction `1/N`).
    //
    // Planar envmap (D == 1): no depth neighbour. Cubemap (D == 6): ordered
    // adjacent-face differences along the flattened D axis (face d paired with
    // d+1), with no wraparound.
    __device__ __forceinline__ float sky_tv_fwd_element(
        int ti, int /* B_tex unused */, int D, int H, int W, int C, const float *__restrict__ bg_tex, float factor
    )
    {
        const int N_per_face = H * W * C; // elements per (b, d) slice
        const int bd         = ti / N_per_face;
        const int rem        = ti - bd * N_per_face;
        const int h          = rem / (W * C);
        const int wc         = rem - h * (W * C);
        const int w          = wc / C;
        // channel index not needed — we index ti directly

        const float v = bg_tex[ti];
        float acc     = 0.0f;

        // Depth neighbour: one-sided d -> d+1 where possible
        if(D > 1 && (bd % D) < (D - 1))
        {
            const float nd    = bg_tex[ti + N_per_face];
            const float diff  = v - nd;
            acc              += diff * diff;
        }
        if(h < H - 1)
        {
            const float nh    = bg_tex[ti + W * C];
            const float diff  = v - nh;
            acc              += diff * diff;
        }
        if(w < W - 1)
        {
            const float nw    = bg_tex[ti + C];
            const float diff  = v - nw;
            acc              += diff * diff;
        }
        return acc * factor;
    }

    // Sky env-map TV backward: accumulates d(loss)/d(bg_tex[ti]) from both the
    // center-as-source and center-as-target terms. Because the backward kernel
    // dispatches one thread per element, we can avoid atomics by computing the
    // full gradient at ti from its own neighbours (v - left, v - up, v - prev_d
    // on the trailing side; v - right, v - down, v - next_d on the leading
    // side — all contribute +/- 2*(v - n)*factor).
    __device__ __forceinline__ float sky_tv_bwd_element(
        int ti,
        int /* B_tex unused */,
        int D,
        int H,
        int W,
        int C,
        const float *__restrict__ bg_tex,
        const float *__restrict__ v_loss,
        float factor
    )
    {
        const int N_per_face = H * W * C;
        const int bd         = ti / N_per_face;
        const int rem        = ti - bd * N_per_face;
        const int h          = rem / (W * C);
        const int wc         = rem - h * (W * C);
        const int w          = wc / C;
        const int d_in_face  = bd % D;

        const float v = bg_tex[ti];
        float grad    = 0.0f;

        // Forward term where this element is the "source" (center)
        // Backward term where this element is the "target" (neighbour)
        if(D > 1)
        {
            if(d_in_face < D - 1)
            {
                const int nid     = ti + N_per_face;
                const float diff  = v - bg_tex[nid];
                grad             += 2.0f * diff * factor * v_loss[ti];
            }
            if(d_in_face > 0)
            {
                const int nid     = ti - N_per_face;
                const float diff  = bg_tex[nid] - v;
                grad             -= 2.0f * diff * factor * v_loss[nid];
            }
        }
        if(h < H - 1)
        {
            const int nid     = ti + W * C;
            const float diff  = v - bg_tex[nid];
            grad             += 2.0f * diff * factor * v_loss[ti];
        }
        if(h > 0)
        {
            const int nid     = ti - W * C;
            const float diff  = bg_tex[nid] - v;
            grad             -= 2.0f * diff * factor * v_loss[nid];
        }
        if(w < W - 1)
        {
            const int nid     = ti + C;
            const float diff  = v - bg_tex[nid];
            grad             += 2.0f * diff * factor * v_loss[ti];
        }
        if(w > 0)
        {
            const int nid     = ti - C;
            const float diff  = bg_tex[nid] - v;
            grad             -= 2.0f * diff * factor * v_loss[nid];
        }
        return grad;
    }

    // ---------------------------------------------------------------------------
    // Grid drift forward: per (b, d, h, w) cell, read the 12 channel entries
    // (b*12 + c, d, h, w), compute Frobenius distance from identity, multiply
    // by factor. Returns one scalar per cell.
    __device__ __forceinline__ float grid_drift_fwd_cell(
        int ci, int /* B unused */, int D, int H, int W, const float *__restrict__ grid, float factor
    )
    {
        const int DHW = D * H * W;
        const int b   = ci / DHW;
        const int rem = ci - b * DHW;
        const int dhw = rem; // d * H * W + h * W + w

        constexpr int NCH  = LOSSES_GRID_NUM_CHANNELS;
        const int stride_c = DHW;
        const int base     = b * NCH * DHW + dhw;

        float sum_sq = 0.0f;
#    pragma unroll
        for(int c = 0; c < NCH; ++c)
        {
            const int r        = c / LOSSES_GRID_NUM_COLS;
            const int col      = c - r * LOSSES_GRID_NUM_COLS;
            const float v      = grid[base + c * stride_c];
            const float ident  = (r == col) ? 1.0f : 0.0f;
            const float diff   = v - ident;
            sum_sq            += diff * diff;
        }
        return sqrtf(sum_sq) * factor;
    }

    // Grid TV spatial forward: sum of squared forward differences along D, H, W
    // across all 12 channels, divided by channel count. Factor carries outer
    // normalization.
    __device__ __forceinline__ float grid_tv_fwd_cell(
        int ci, int /* B unused */, int D, int H, int W, const float *__restrict__ grid, float factor
    )
    {
        const int DHW = D * H * W;
        const int b   = ci / DHW;
        const int rem = ci - b * DHW;
        const int d   = rem / (H * W);
        const int hw  = rem - d * (H * W);
        const int h   = hw / W;
        const int w   = hw - h * W;

        constexpr int NCH  = LOSSES_GRID_NUM_CHANNELS;
        const int stride_c = DHW;
        const int base     = b * NCH * DHW + d * (H * W) + h * W + w;

        float acc = 0.0f;
#    pragma unroll
        for(int c = 0; c < NCH; ++c)
        {
            const int base_c = base + c * stride_c;
            const float v    = grid[base_c];
            if(d < D - 1)
            {
                const float n     = grid[base_c + H * W];
                const float diff  = v - n;
                acc              += diff * diff;
            }
            if(h < H - 1)
            {
                const float n     = grid[base_c + W];
                const float diff  = v - n;
                acc              += diff * diff;
            }
            if(w < W - 1)
            {
                const float n     = grid[base_c + 1];
                const float diff  = v - n;
                acc              += diff * diff;
            }
        }
        return (acc / static_cast<float>(NCH)) * factor;
    }

    // Fused grid drift + TV backward for a single (b, d, h, w) cell, gather form.
    // Each output cell computes its complete gradient and writes v_grid once per
    // channel -- no atomics and no pre-zeroed output buffer. The TV gradient at a
    // cell is the sum of (a) its own forward D/H/W diffs, weighted by this cell's
    // TV cotangent, and (b) the backward diffs from the up-to-three source cells
    // that list it as their forward neighbour, each weighted by that source
    // cell's TV cotangent (read from v_tv_loss). Drift is purely per-cell. Writes
    // are unconditional (zero when both factors are disabled) so every entry of a
    // present grid is set, which is why the caller no longer pre-zeros v_grid.
    //
    // Optimizations:
    //   - Shared (b, d, h, w) decomposition done once, reused for both losses.
    //   - Cache this cell's 12 channel values in registers (single read, reused
    //     for the drift sum_sq and the TV self term).
    //   - Per-cell TV weights (self + 3 backward-source cells) computed once.
    __device__ __forceinline__ void grid_drift_and_tv_bwd_cell(
        int ci,
        int /* B unused */,
        int D,
        int H,
        int W,
        float drift_factor,
        float tv_factor,
        const float *__restrict__ grid,
        float v_drift,
        const float *__restrict__ v_tv_loss,
        float *__restrict__ v_grid
    )
    {
        const int DHW = D * H * W;
        const int b   = ci / DHW;
        const int rem = ci - b * DHW;
        const int d   = rem / (H * W);
        const int hw  = rem - d * (H * W);
        const int h   = hw / W;
        const int w   = hw - h * W;

        constexpr int NCH   = LOSSES_GRID_NUM_CHANNELS;
        constexpr int NCOLS = LOSSES_GRID_NUM_COLS;
        const int stride_c  = DHW;
        const int base      = b * NCH * DHW + d * (H * W) + h * W + w;

        // Cache this cell's 12 channel values once; accumulate the drift sum_sq.
        const bool drift_on = is_loss_enabled(drift_factor);
        float cached_m[NCH];
        float sum_sq = 0.0f;
#    pragma unroll
        for(int c = 0; c < NCH; ++c)
        {
            const float v = grid[base + c * stride_c];
            cached_m[c]   = v;
            if(drift_on)
            {
                const int r       = c / NCOLS;
                const int col     = c - r * NCOLS;
                const float diff  = v - ((r == col) ? 1.0f : 0.0f);
                sum_sq           += diff * diff;
            }
        }
        const bool drift_active = drift_on && sum_sq > 1e-12f;
        const float drift_scale = drift_active ? (drift_factor * v_drift / sqrtf(sum_sq)) : 0.0f;

        // Per-cell TV weights: self (forward diffs) + up-to-3 backward-source cells.
        const bool tv_on = is_loss_enabled(tv_factor);
        float gb_self = 0.0f, gb_dm = 0.0f, gb_hm = 0.0f, gb_wm = 0.0f;
        if(tv_on)
        {
            const float k = 2.0f * tv_factor / static_cast<float>(NCH);
            gb_self       = k * v_tv_loss[ci];
            if(d > 0)
            {
                gb_dm = k * v_tv_loss[ci - H * W];
            }
            if(h > 0)
            {
                gb_hm = k * v_tv_loss[ci - W];
            }
            if(w > 0)
            {
                gb_wm = k * v_tv_loss[ci - 1];
            }
        }
        const bool has_df = d < D - 1, has_hf = h < H - 1, has_wf = w < W - 1;
        const bool has_db = d > 0, has_hb = h > 0, has_wb = w > 0;

#    pragma unroll
        for(int c = 0; c < NCH; ++c)
        {
            const int idx = base + c * stride_c;
            const float v = cached_m[c];
            float grad    = 0.0f;

            if(tv_on)
            {
                // Forward diffs: this cell is the source i in (v_i - v_{i+dir}).
                if(has_df)
                {
                    grad += (v - grid[idx + H * W]) * gb_self;
                }
                if(has_hf)
                {
                    grad += (v - grid[idx + W]) * gb_self;
                }
                if(has_wf)
                {
                    grad += (v - grid[idx + 1]) * gb_self;
                }
                // Backward diffs: this cell is neighbour i+dir of source i = idx-dir.
                if(has_db)
                {
                    grad += (v - grid[idx - H * W]) * gb_dm;
                }
                if(has_hb)
                {
                    grad += (v - grid[idx - W]) * gb_hm;
                }
                if(has_wb)
                {
                    grad += (v - grid[idx - 1]) * gb_wm;
                }
            }

            if(drift_active)
            {
                const int r        = c / NCOLS;
                const int col      = c - r * NCOLS;
                const float ident  = (r == col) ? 1.0f : 0.0f;
                grad              += drift_scale * (v - ident);
            }

            v_grid[idx] = grad;
        }
    }

    // ---------------------------------------------------------------------------
    // Forward kernel: one thread per element up to max(numel_bg_tex, numel_gc_cells, numel_gf_cells).
    // ---------------------------------------------------------------------------
    __global__ void bg_grid_losses_fwd_kernel(
        int B_tex,
        int D_tex,
        int H_tex,
        int W_tex,
        int C_tex,
        int B_gc,
        int D_gc,
        int H_gc,
        int W_gc,
        int B_gf,
        int D_gf,
        int H_gf,
        int W_gf,
        float bg_tex_factor,
        float grid_drift_camera_factor,
        float grid_drift_frame_factor,
        float grid_camera_tv_factor,
        float grid_frame_tv_factor,
        const float *__restrict__ bg_tex,
        const float *__restrict__ grids_camera,
        const float *__restrict__ grids_frame,
        float *__restrict__ bg_tex_loss,
        float *__restrict__ grids_drift_loss,    // [numel_gc_cells + numel_gf_cells]
        float *__restrict__ grid_camera_tv_loss, // [numel_gc_cells]
        float *__restrict__ grid_frame_tv_loss,  // [numel_gf_cells]
        int numel_bg_tex,
        int numel_gc_cells,
        int numel_gf_cells
    )
    {
        const int ti = blockIdx.x * blockDim.x + threadIdx.x;

        // Sky env-map TV
        if(is_loss_enabled(bg_tex_factor) && ti < numel_bg_tex)
        {
            bg_tex_loss[ti] = sky_tv_fwd_element(ti, B_tex, D_tex, H_tex, W_tex, C_tex, bg_tex, bg_tex_factor);
        }
        else if(ti < numel_bg_tex)
        {
            bg_tex_loss[ti] = 0.0f;
        }

        // Grid drift (camera) + grid TV (camera)
        if(ti < numel_gc_cells)
        {
            if(is_loss_enabled(grid_drift_camera_factor))
            {
                grids_drift_loss[ti]
                    = grid_drift_fwd_cell(ti, B_gc, D_gc, H_gc, W_gc, grids_camera, grid_drift_camera_factor);
            }
            else
            {
                grids_drift_loss[ti] = 0.0f;
            }
            if(is_loss_enabled(grid_camera_tv_factor))
            {
                grid_camera_tv_loss[ti]
                    = grid_tv_fwd_cell(ti, B_gc, D_gc, H_gc, W_gc, grids_camera, grid_camera_tv_factor);
            }
            else
            {
                grid_camera_tv_loss[ti] = 0.0f;
            }
        }

        // Grid drift (frame) + grid TV (frame)
        if(ti < numel_gf_cells)
        {
            if(is_loss_enabled(grid_drift_frame_factor))
            {
                grids_drift_loss[numel_gc_cells + ti]
                    = grid_drift_fwd_cell(ti, B_gf, D_gf, H_gf, W_gf, grids_frame, grid_drift_frame_factor);
            }
            else
            {
                grids_drift_loss[numel_gc_cells + ti] = 0.0f;
            }
            if(is_loss_enabled(grid_frame_tv_factor))
            {
                grid_frame_tv_loss[ti]
                    = grid_tv_fwd_cell(ti, B_gf, D_gf, H_gf, W_gf, grids_frame, grid_frame_tv_factor);
            }
            else
            {
                grid_frame_tv_loss[ti] = 0.0f;
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Backward kernel. Gather form: each thread writes every channel of its own
    // v_bg_tex / v_grids_camera / v_grids_frame entry exactly once (no atomics),
    // so the Python side no longer has to pre-zero these grad buffers.
    // ---------------------------------------------------------------------------
    __global__ void bg_grid_losses_bwd_kernel(
        int B_tex,
        int D_tex,
        int H_tex,
        int W_tex,
        int C_tex,
        int B_gc,
        int D_gc,
        int H_gc,
        int W_gc,
        int B_gf,
        int D_gf,
        int H_gf,
        int W_gf,
        float bg_tex_factor,
        float grid_drift_camera_factor,
        float grid_drift_frame_factor,
        float grid_camera_tv_factor,
        float grid_frame_tv_factor,
        const float *__restrict__ bg_tex,
        const float *__restrict__ grids_camera,
        const float *__restrict__ grids_frame,
        const float *__restrict__ v_bg_tex_loss,
        const float *__restrict__ v_grids_drift_loss,
        const float *__restrict__ v_grid_camera_tv_loss,
        const float *__restrict__ v_grid_frame_tv_loss,
        float *__restrict__ v_bg_tex,
        float *__restrict__ v_grids_camera,
        float *__restrict__ v_grids_frame,
        int numel_bg_tex,
        int numel_gc_cells,
        int numel_gf_cells
    )
    {
        const int ti = blockIdx.x * blockDim.x + threadIdx.x;

        if(ti < numel_bg_tex)
        {
            // sky_tv_bwd_element gathers: each element's gradient depends on the
            // upstream gradient of its own cell (as the source) and its
            // neighbours' cells (as the target), so it writes v_bg_tex once
            // without atomics. Write unconditionally -- a disabled bg-tex loss
            // still zeros the grad buffer, so the caller need not pre-zero it.
            v_bg_tex[ti]
                = is_loss_enabled(bg_tex_factor)
                    ? sky_tv_bwd_element(ti, B_tex, D_tex, H_tex, W_tex, C_tex, bg_tex, v_bg_tex_loss, bg_tex_factor)
                    : 0.0f;
        }

        // Gather backward writes every cell of a present grid exactly once (zero
        // when both factors are disabled), so the guard is just the bounds check.
        if(ti < numel_gc_cells)
        {
            const float v_drift = is_loss_enabled(grid_drift_camera_factor) ? v_grids_drift_loss[ti] : 0.0f;
            grid_drift_and_tv_bwd_cell(
                ti,
                B_gc,
                D_gc,
                H_gc,
                W_gc,
                grid_drift_camera_factor,
                grid_camera_tv_factor,
                grids_camera,
                v_drift,
                v_grid_camera_tv_loss,
                v_grids_camera
            );
        }

        if(ti < numel_gf_cells)
        {
            const float v_drift
                = is_loss_enabled(grid_drift_frame_factor) ? v_grids_drift_loss[numel_gc_cells + ti] : 0.0f;
            grid_drift_and_tv_bwd_cell(
                ti,
                B_gf,
                D_gf,
                H_gf,
                W_gf,
                grid_drift_frame_factor,
                grid_frame_tv_factor,
                grids_frame,
                v_drift,
                v_grid_frame_tv_loss,
                v_grids_frame
            );
        }
    }
} // namespace

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

void launch_bg_grid_losses_fwd_kernel(
    int B_tex,
    int D_tex,
    int H_tex,
    int W_tex,
    int C_tex,
    int B_gc,
    int D_gc,
    int H_gc,
    int W_gc,
    int B_gf,
    int D_gf,
    int H_gf,
    int W_gf,
    float bg_tex_factor,
    float grid_drift_camera_factor,
    float grid_drift_frame_factor,
    float grid_camera_tv_factor,
    float grid_frame_tv_factor,
    const at::Tensor *bg_tex,
    const at::Tensor *grids_camera,
    const at::Tensor *grids_frame,
    at::Tensor &bg_tex_loss,
    at::Tensor &grids_drift_loss,
    at::Tensor &grid_camera_tv_loss,
    at::Tensor &grid_frame_tv_loss
)
{
    const int numel_bg_tex   = B_tex * D_tex * H_tex * W_tex * C_tex;
    const int numel_gc_cells = B_gc * D_gc * H_gc * W_gc;
    const int numel_gf_cells = B_gf * D_gf * H_gf * W_gf;

    const int N = std::max({numel_bg_tex, numel_gc_cells, numel_gf_cells});
    if(N == 0)
    {
        return;
    }

    dim3 threads(256);
    dim3 blocks((N + threads.x - 1) / threads.x);

    bg_grid_losses_fwd_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        B_tex,
        D_tex,
        H_tex,
        W_tex,
        C_tex,
        B_gc,
        D_gc,
        H_gc,
        W_gc,
        B_gf,
        D_gf,
        H_gf,
        W_gf,
        bg_tex_factor,
        grid_drift_camera_factor,
        grid_drift_frame_factor,
        grid_camera_tv_factor,
        grid_frame_tv_factor,
        bg_tex ? bg_tex->data_ptr<float>() : nullptr,
        grids_camera ? grids_camera->data_ptr<float>() : nullptr,
        grids_frame ? grids_frame->data_ptr<float>() : nullptr,
        bg_tex_loss.data_ptr<float>(),
        grids_drift_loss.data_ptr<float>(),
        grid_camera_tv_loss.data_ptr<float>(),
        grid_frame_tv_loss.data_ptr<float>(),
        numel_bg_tex,
        numel_gc_cells,
        numel_gf_cells
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_bg_grid_losses_bwd_kernel(
    int B_tex,
    int D_tex,
    int H_tex,
    int W_tex,
    int C_tex,
    int B_gc,
    int D_gc,
    int H_gc,
    int W_gc,
    int B_gf,
    int D_gf,
    int H_gf,
    int W_gf,
    float bg_tex_factor,
    float grid_drift_camera_factor,
    float grid_drift_frame_factor,
    float grid_camera_tv_factor,
    float grid_frame_tv_factor,
    const at::Tensor *bg_tex,
    const at::Tensor *grids_camera,
    const at::Tensor *grids_frame,
    const at::Tensor &v_bg_tex_loss,
    const at::Tensor &v_grids_drift_loss,
    const at::Tensor &v_grid_camera_tv_loss,
    const at::Tensor &v_grid_frame_tv_loss,
    at::Tensor &v_bg_tex,
    at::Tensor &v_grids_camera,
    at::Tensor &v_grids_frame
)
{
    const int numel_bg_tex   = B_tex * D_tex * H_tex * W_tex * C_tex;
    const int numel_gc_cells = B_gc * D_gc * H_gc * W_gc;
    const int numel_gf_cells = B_gf * D_gf * H_gf * W_gf;

    const int N = std::max({numel_bg_tex, numel_gc_cells, numel_gf_cells});
    if(N == 0)
    {
        return;
    }

    dim3 threads(256);
    dim3 blocks((N + threads.x - 1) / threads.x);

    bg_grid_losses_bwd_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        B_tex,
        D_tex,
        H_tex,
        W_tex,
        C_tex,
        B_gc,
        D_gc,
        H_gc,
        W_gc,
        B_gf,
        D_gf,
        H_gf,
        W_gf,
        bg_tex_factor,
        grid_drift_camera_factor,
        grid_drift_frame_factor,
        grid_camera_tv_factor,
        grid_frame_tv_factor,
        bg_tex ? bg_tex->data_ptr<float>() : nullptr,
        grids_camera ? grids_camera->data_ptr<float>() : nullptr,
        grids_frame ? grids_frame->data_ptr<float>() : nullptr,
        v_bg_tex_loss.data_ptr<float>(),
        v_grids_drift_loss.data_ptr<float>(),
        v_grid_camera_tv_loss.data_ptr<float>(),
        v_grid_frame_tv_loss.data_ptr<float>(),
        v_bg_tex.data_ptr<float>(),
        v_grids_camera.data_ptr<float>(),
        v_grids_frame.data_ptr<float>(),
        numel_bg_tex,
        numel_gc_cells,
        numel_gf_cells
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace gsplat

#endif // GSPLAT_BUILD_LOSSES
