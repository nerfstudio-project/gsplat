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

// Device-side environment-map background sampling for scene CUDA. Paired with
// env_map_sample.cu: this header holds __device__ helpers (bilinear filtering,
// equirectangular / cubemap projection and per-ray fwd/bwd bodies); the .cu file
// holds __global__ kernels, launches, and exported *_cuda symbols. The sampling
// reproduces F.grid_sample(mode="bilinear", align_corners=False,
// padding_mode="border") as used in env_map_background.py.
#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

namespace at
{
class Tensor;
}

namespace gsplat
{
namespace scene
{
    // Clamp an integer index into the inclusive range [lo, hi].
    __forceinline__ __device__ int env_map_clampi(int v, int lo, int hi)
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    // Bilinear axis setup for align_corners=False over an axis of size S.
    // Maps a normalized coordinate c in [-1, 1] to the two neighboring texel
    // indices and their interpolation weights. Border padding is realized by
    // clamping the integer READ indices to [0, S-1]; the weights are computed
    // from the UNclamped continuous coordinate (matches env_map_background.py).
    // Outputs: *i0/*i1 clamped lower/upper indices, *w0/*w1 their weights.
    template<typename scalar_t>
    __forceinline__ __device__ void env_map_bilinear_axis(
        scalar_t c, int S, int *i0, int *i1, scalar_t *w0, scalar_t *w1
    )
    {
        const scalar_t ix  = ((c + scalar_t(1)) * scalar_t(S) - scalar_t(1)) * scalar_t(0.5);
        const scalar_t x0f = floor(ix);
        const scalar_t wx1 = ix - x0f;
        *w1                = wx1;
        *w0                = scalar_t(1) - wx1;
        const int x0       = static_cast<int>(x0f);
        const int x1       = x0 + 1;
        *i0                = env_map_clampi(x0, 0, S - 1);
        *i1                = env_map_clampi(x1, 0, S - 1);
    }

    // Compute the 4 bilinear corner texel offsets and weights for an
    // equirectangular texture of shape [1, H, W, 3] given a unit ray direction.
    // Each base[k] is a flat offset into the contiguous texture that addresses
    // texel (row, col); add the channel c in {0,1,2} to read/write a component.
    template<typename scalar_t>
    __forceinline__ __device__ void env_map_equirect_corners(
        scalar_t x, scalar_t y, scalar_t z, int H, int W, int64_t base[4], scalar_t w[4]
    )
    {
        const scalar_t PI = scalar_t(3.14159265358979323846);

        // Azimuth in (-pi, pi]; nudge x at the poles (x == y == 0) so atan2 has a
        // finite gradient. Polar angle clamped strictly inside [0, pi].
        const bool pole     = (x == scalar_t(0)) && (y == scalar_t(0));
        const scalar_t xd   = pole ? x + scalar_t(1e-6) : x;
        const scalar_t phi  = atan2(y, xd);
        const scalar_t zc   = fmin(fmax(z, scalar_t(-1) + scalar_t(1e-6)), scalar_t(1) - scalar_t(1e-6));
        const scalar_t th   = acos(zc);
        const scalar_t frac = (phi + PI) / (scalar_t(2) * PI);

        // Horizontal axis is padded by one column on each side (wrap-around), so
        // the sampled coordinate lives in a texture of width Wp = W + 2.
        const scalar_t Wf = scalar_t(W);
        const int Wp      = W + 2;
        const scalar_t u  = (scalar_t(2) * frac * Wf + scalar_t(2)) / (Wf + scalar_t(2)) - scalar_t(1);
        const scalar_t v  = th / PI * scalar_t(2) - scalar_t(1);

        int px0, px1, py0, py1;
        scalar_t wx0, wx1, wy0, wy1;
        env_map_bilinear_axis(u, Wp, &px0, &px1, &wx0, &wx1);
        env_map_bilinear_axis(v, H, &py0, &py1, &wy0, &wy1);

        // Map padded column p (in [0, Wp-1]) back to a real texture column:
        // p=0 -> W-1, p=Wp-1 -> 0, interior p -> p-1.
        const int rx0 = ((px0 - 1) % W + W) % W;
        const int rx1 = ((px1 - 1) % W + W) % W;

        base[0] = (static_cast<int64_t>(py0) * W + rx0) * 3;
        base[1] = (static_cast<int64_t>(py0) * W + rx1) * 3;
        base[2] = (static_cast<int64_t>(py1) * W + rx0) * 3;
        base[3] = (static_cast<int64_t>(py1) * W + rx1) * 3;
        w[0]    = wy0 * wx0;
        w[1]    = wy0 * wx1;
        w[2]    = wy1 * wx0;
        w[3]    = wy1 * wx1;
    }

    // Compute the 4 bilinear corner texel offsets and weights for a cubemap
    // texture of shape [1, 6, H, W, 3] (W == H) given a unit ray direction.
    // Replicates the OpenGL dominant-axis face routing of
    // _dominant_axis_to_face_uv (priority x > y > z; >= for the major-axis test,
    // strict > 0 for the sign, so the <= 0 branch selects the negative face).
    template<typename scalar_t>
    __forceinline__ __device__ void env_map_cubemap_corners(
        scalar_t x, scalar_t y, scalar_t z, int H, int W, int64_t base[4], scalar_t w[4]
    )
    {
        const scalar_t EPS_MA = scalar_t(1.1920929e-7);

        const scalar_t ax  = fabs(x);
        const scalar_t ay  = fabs(y);
        const scalar_t az  = fabs(z);
        const bool x_major = (ax >= ay) && (ax >= az);
        const bool y_major = (!x_major) && (ay >= az);

        int face;
        scalar_t sc, tc, ma;
        if(x_major)
        {
            if(x > scalar_t(0))
            {
                face = 0; // +X
                sc   = -z;
                tc   = -y;
                ma   = ax;
            }
            else
            {
                face = 1; // -X
                sc   = z;
                tc   = -y;
                ma   = ax;
            }
        }
        else if(y_major)
        {
            if(y > scalar_t(0))
            {
                face = 2; // +Y
                sc   = x;
                tc   = z;
                ma   = ay;
            }
            else
            {
                face = 3; // -Y
                sc   = x;
                tc   = -z;
                ma   = ay;
            }
        }
        else
        {
            if(z > scalar_t(0))
            {
                face = 4; // +Z
                sc   = x;
                tc   = -y;
                ma   = az;
            }
            else
            {
                face = 5; // -Z
                sc   = -x;
                tc   = -y;
                ma   = az;
            }
        }

        ma               = fmax(ma, EPS_MA);
        const scalar_t u = sc / ma;
        const scalar_t v = tc / ma;

        int px0, px1, py0, py1;
        scalar_t wx0, wx1, wy0, wy1;
        env_map_bilinear_axis(u, W, &px0, &px1, &wx0, &wx1);
        env_map_bilinear_axis(v, H, &py0, &py1, &wy0, &wy1);

        const int64_t face_row0 = static_cast<int64_t>(face) * H;
        base[0]                 = ((face_row0 + py0) * W + px0) * 3;
        base[1]                 = ((face_row0 + py0) * W + px1) * 3;
        base[2]                 = ((face_row0 + py1) * W + px0) * 3;
        base[3]                 = ((face_row0 + py1) * W + px1) * 3;
        w[0]                    = wy0 * wx0;
        w[1]                    = wy0 * wx1;
        w[2]                    = wy1 * wx0;
        w[3]                    = wy1 * wx1;
    }

    // Per-ray forward: sample the equirectangular texture into out[i] (N,3).
    template<typename scalar_t>
    __forceinline__ __device__ void env_map_equirect_fwd_device(
        int64_t i,
        const scalar_t *__restrict__ rays_d,
        const scalar_t *__restrict__ textures,
        int H,
        int W,
        scalar_t *__restrict__ out
    )
    {
        const scalar_t x = rays_d[i * 3 + 0];
        const scalar_t y = rays_d[i * 3 + 1];
        const scalar_t z = rays_d[i * 3 + 2];
        int64_t base[4];
        scalar_t w[4];
        env_map_equirect_corners(x, y, z, H, W, base, w);
#pragma unroll
        for(int c = 0; c < 3; ++c)
        {
            scalar_t acc = scalar_t(0);
#pragma unroll
            for(int k = 0; k < 4; ++k)
            {
                acc += w[k] * textures[base[k] + c];
            }
            out[i * 3 + c] = acc;
        }
    }

    // Per-ray backward: scatter grad_out[i] (N,3) into grad_textures via atomicAdd.
    template<typename scalar_t>
    __forceinline__ __device__ void env_map_equirect_bwd_device(
        int64_t i,
        const scalar_t *__restrict__ rays_d,
        int H,
        int W,
        const scalar_t *__restrict__ grad_out,
        scalar_t *__restrict__ grad_textures
    )
    {
        const scalar_t x = rays_d[i * 3 + 0];
        const scalar_t y = rays_d[i * 3 + 1];
        const scalar_t z = rays_d[i * 3 + 2];
        int64_t base[4];
        scalar_t w[4];
        env_map_equirect_corners(x, y, z, H, W, base, w);
#pragma unroll
        for(int c = 0; c < 3; ++c)
        {
            const scalar_t go = grad_out[i * 3 + c];
#pragma unroll
            for(int k = 0; k < 4; ++k)
            {
                atomicAdd(&grad_textures[base[k] + c], w[k] * go);
            }
        }
    }

    // Per-ray forward: sample the cubemap texture into out[i] (N,3).
    template<typename scalar_t>
    __forceinline__ __device__ void env_map_cubemap_fwd_device(
        int64_t i,
        const scalar_t *__restrict__ rays_d,
        const scalar_t *__restrict__ textures,
        int H,
        int W,
        scalar_t *__restrict__ out
    )
    {
        const scalar_t x = rays_d[i * 3 + 0];
        const scalar_t y = rays_d[i * 3 + 1];
        const scalar_t z = rays_d[i * 3 + 2];
        int64_t base[4];
        scalar_t w[4];
        env_map_cubemap_corners(x, y, z, H, W, base, w);
#pragma unroll
        for(int c = 0; c < 3; ++c)
        {
            scalar_t acc = scalar_t(0);
#pragma unroll
            for(int k = 0; k < 4; ++k)
            {
                acc += w[k] * textures[base[k] + c];
            }
            out[i * 3 + c] = acc;
        }
    }

    // Per-ray backward: scatter grad_out[i] (N,3) into grad_textures via atomicAdd.
    template<typename scalar_t>
    __forceinline__ __device__ void env_map_cubemap_bwd_device(
        int64_t i,
        const scalar_t *__restrict__ rays_d,
        int H,
        int W,
        const scalar_t *__restrict__ grad_out,
        scalar_t *__restrict__ grad_textures
    )
    {
        const scalar_t x = rays_d[i * 3 + 0];
        const scalar_t y = rays_d[i * 3 + 1];
        const scalar_t z = rays_d[i * 3 + 2];
        int64_t base[4];
        scalar_t w[4];
        env_map_cubemap_corners(x, y, z, H, W, base, w);
#pragma unroll
        for(int c = 0; c < 3; ++c)
        {
            const scalar_t go = grad_out[i * 3 + c];
#pragma unroll
            for(int k = 0; k < 4; ++k)
            {
                atomicAdd(&grad_textures[base[k] + c], w[k] * go);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Host-side CUDA launchers (out-param style, defined in env_map_sample.cu).
    // rays_d [N, 3] is assumed unit-normalized. Equirect textures are [1, H, W, 3];
    // cubemap textures are [1, 6, H, W, 3] with W == H. Backward grad_textures must
    // be zero-initialized (the kernels accumulate with atomicAdd).
    // -------------------------------------------------------------------------
    void sample_env_map_equirect_fwd_cuda(const at::Tensor &rays_d, const at::Tensor &textures, at::Tensor &out);
    void sample_env_map_equirect_bwd_cuda(
        const at::Tensor &rays_d, const at::Tensor &textures, const at::Tensor &grad_out, at::Tensor &grad_textures
    );
    void sample_env_map_cubemap_fwd_cuda(const at::Tensor &rays_d, const at::Tensor &textures, at::Tensor &out);
    void sample_env_map_cubemap_bwd_cuda(
        const at::Tensor &rays_d, const at::Tensor &textures, const at::Tensor &grad_out, at::Tensor &grad_textures
    );
} // namespace scene
} // namespace gsplat
