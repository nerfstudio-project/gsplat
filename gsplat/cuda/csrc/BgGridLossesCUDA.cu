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

#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAException.h>
#    include <c10/cuda/CUDAStream.h>
#    include <cuda_runtime.h>

#    include "BgGridLosses.h"

namespace gsplat
{
namespace
{
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

    // Fused grid drift + TV backward for a single (b, d, h, w) cell.
    // Computes d(drift_loss)/d(grid) + d(tv_loss)/d(grid) for all 12 channels
    // of that cell and writes to v_grid via atomicAdd (since each cell
    // participates in multiple backward computations through its neighbours).
    //
    // Optimizations:
    //   - Shared (b, d, h, w) decomposition done once, reused for both losses.
    //   - Drift: cache 12 m_ij entries + 12 flat indices in registers, single
    //     read, reuse for both `sum_sq` and per-entry atomicAdd.
    //   - TV: per channel, read v once, sweep 3 neighbour dirs (d+1, h+1, w+1),
    //     accumulate self-gradient in a register, write one atomicAdd per
    //     channel instead of 3.
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
        float v_tv,
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

        // -- Drift bwd: cache 12 entries + indices, compute once, scatter once.
        float cached_m[NCH];
        int cached_idx[NCH];
        float sum_sq = 0.0f;
        if(!(drift_factor < 0.0f))
        {
#    pragma unroll
            for(int c = 0; c < NCH; ++c)
            {
                const int idx      = base + c * stride_c;
                const float v      = grid[idx];
                cached_m[c]        = v;
                cached_idx[c]      = idx;
                const int r        = c / NCOLS;
                const int col      = c - r * NCOLS;
                const float ident  = (r == col) ? 1.0f : 0.0f;
                const float diff   = v - ident;
                sum_sq            += diff * diff;
            }
        }

// -- TV bwd: per channel, accumulate self-grad in register.
#    pragma unroll
        for(int c = 0; c < NCH; ++c)
        {
            float grad_self    = 0.0f;
            const int idx_self = (!(drift_factor < 0.0f)) ? cached_idx[c] : base + c * stride_c;
            const float v      = (!(drift_factor < 0.0f)) ? cached_m[c] : grid[idx_self];

            if(!(tv_factor < 0.0f))
            {
                const float grad_base = 2.0f * tv_factor / static_cast<float>(NCH) * v_tv;
                if(d < D - 1)
                {
                    const int idx_n   = idx_self + H * W;
                    const float diff  = v - grid[idx_n];
                    const float g     = diff * grad_base;
                    grad_self        += g;
                    atomicAdd(&v_grid[idx_n], -g);
                }
                if(h < H - 1)
                {
                    const int idx_n   = idx_self + W;
                    const float diff  = v - grid[idx_n];
                    const float g     = diff * grad_base;
                    grad_self        += g;
                    atomicAdd(&v_grid[idx_n], -g);
                }
                if(w < W - 1)
                {
                    const int idx_n   = idx_self + 1;
                    const float diff  = v - grid[idx_n];
                    const float g     = diff * grad_base;
                    grad_self        += g;
                    atomicAdd(&v_grid[idx_n], -g);
                }
            }

            // Drift self-grad (scales with 1/sqrt(sum_sq))
            if(!(drift_factor < 0.0f) && sum_sq > 1e-12f)
            {
                const int r             = c / NCOLS;
                const int col           = c - r * NCOLS;
                const float ident       = (r == col) ? 1.0f : 0.0f;
                const float grad_drift  = drift_factor * v_drift / sqrtf(sum_sq) * (v - ident);
                grad_self              += grad_drift;
            }

            if(grad_self != 0.0f)
            {
                atomicAdd(&v_grid[idx_self], grad_self);
            }
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
        if(!(bg_tex_factor < 0.0f) && ti < numel_bg_tex)
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
            if(!(grid_drift_camera_factor < 0.0f))
            {
                grids_drift_loss[ti]
                    = grid_drift_fwd_cell(ti, B_gc, D_gc, H_gc, W_gc, grids_camera, grid_drift_camera_factor);
            }
            else
            {
                grids_drift_loss[ti] = 0.0f;
            }
            if(!(grid_camera_tv_factor < 0.0f))
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
            if(!(grid_drift_frame_factor < 0.0f))
            {
                grids_drift_loss[numel_gc_cells + ti]
                    = grid_drift_fwd_cell(ti, B_gf, D_gf, H_gf, W_gf, grids_frame, grid_drift_frame_factor);
            }
            else
            {
                grids_drift_loss[numel_gc_cells + ti] = 0.0f;
            }
            if(!(grid_frame_tv_factor < 0.0f))
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
    // Backward kernel. v_bg_tex / v_grids_camera / v_grids_frame are assumed
    // zero-initialised on the Python side (used with atomicAdd).
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

        if(!(bg_tex_factor < 0.0f) && ti < numel_bg_tex)
        {
            // Note: sky_tv_bwd_element reads multiple v_bg_tex_loss positions;
            // each element's gradient depends on the upstream gradient of both
            // its own cell (when it's the source) and its neighbours' cells
            // (when it's the target), so this handles all of those in one go
            // without needing atomics on v_bg_tex.
            v_bg_tex[ti]
                = sky_tv_bwd_element(ti, B_tex, D_tex, H_tex, W_tex, C_tex, bg_tex, v_bg_tex_loss, bg_tex_factor);
        }

        if((!(grid_drift_camera_factor < 0.0f) || !(grid_camera_tv_factor < 0.0f)) && ti < numel_gc_cells)
        {
            const float v_drift = (!(grid_drift_camera_factor < 0.0f)) ? v_grids_drift_loss[ti] : 0.0f;
            const float v_tv    = (!(grid_camera_tv_factor < 0.0f)) ? v_grid_camera_tv_loss[ti] : 0.0f;
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
                v_tv,
                v_grids_camera
            );
        }

        if((!(grid_drift_frame_factor < 0.0f) || !(grid_frame_tv_factor < 0.0f)) && ti < numel_gf_cells)
        {
            const float v_drift = (!(grid_drift_frame_factor < 0.0f)) ? v_grids_drift_loss[numel_gc_cells + ti] : 0.0f;
            const float v_tv    = (!(grid_frame_tv_factor < 0.0f)) ? v_grid_frame_tv_loss[ti] : 0.0f;
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
                v_tv,
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
