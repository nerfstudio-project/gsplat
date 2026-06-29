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
 *
 * Self-contained device-side helpers for fused NHT inference.
 *
 * Provides warp-cooperative MMA types whose "native memory" weight layout
 * matches TCNN's FullyFusedMLP, enabling direct use of trained weights with
 * no reformatting.  Avoids pulling in TCNN's ministd.h (which re-typedefs
 * stdint types and conflicts with <cstdint> already in scope from ATen).
 *
 * Expected weight layout in the passed buffer (TCNN "native" fragment format,
 * i.e. the layout produced by tcnn mma_mat::into_native_memory — fragments
 * serialized warp-wide, 16 bytes per lane per fragment):
 *   [0 .. ENC*H)            Layer-0 weights (ENC_DIM × H)
 *   [ENC*H .. ENC*H + H*H)  Layer-1 weights (H × H)
 *   [.. + H*16)             Output weights  (H × 16)
 * where ENC_DIM = round_up16(FEAT_OUT + 9) and H = MLP_HIDDEN (typically 64).
 *
 * NOTE: tcnn's PyTorch `backbone.params` buffer stores each matrix in LINEAR
 * column-major order (= row-major [n_out, n_in]) and must be converted before
 * use here — see gsplat.nht.convert_mlp_params_to_fused_native().
 *
 * All 32 threads of the warp must call nht_fused_shade() together.
 */
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace nht_mlp {

static constexpr uint32_t WARP  = 32u;
static constexpr uint32_t SH3_N = 9u;  // SH degree-3: l=0,1,2 → 9 basis funcs

// True hardware lane id. Do NOT derive this from threadIdx.x: the rasterizer
// launches 2-D {tile_size, tile_size} blocks, where threadIdx.x only spans the
// tile width and is not the lane index.
__device__ __forceinline__ uint32_t lane_id() {
    uint32_t lid;
    asm("mov.u32 %0, %%laneid;" : "=r"(lid));
    return lid;
}
constexpr uint32_t round_up16(uint32_t x) { return (x + 15u) & ~15u; }

// ── tvec<T,N>: per-thread array ───────────────────────────────────────────────
template <typename T, uint32_t N>
struct tvec {
    T d[N];
    __device__ T&       operator[](uint32_t i)       { return d[i]; }
    __device__ const T& operator[](uint32_t i) const { return d[i]; }
    __device__ T&       operator()(uint32_t i)       { return d[i]; }  // sh_enc compat
};

// ── Matrix layouts ────────────────────────────────────────────────────────────
enum class ML { CM, RM };

// ── mma_frag: one 16×16 fp16 WMMA fragment (128 bits per thread) ─────────────
// regs[0..3] encode two half-precision values each via uint32 (= __half2 bits).
// The mma.sync PTX instruction processes 16×16 tiles via two m16n8k16 calls:
//   first produces columns 0..7, second produces columns 8..15 of a 16-wide result.
template <ML LAY> struct mma_frag { uint32_t r[4]; };

// Accumulate: acc_frag(RM) += A_frag(RM) × B_frag(CM)  [16×16 = 2 × m16n8k16]
__device__ __forceinline__ void mma_frag_accum(
    mma_frag<ML::RM>& acc,
    const mma_frag<ML::RM>& a,
    const mma_frag<ML::CM>& b
) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(acc.r[0]), "+r"(acc.r[1])
        : "r"(a.r[0]), "r"(a.r[1]), "r"(a.r[2]), "r"(a.r[3]),
          "r"(b.r[0]), "r"(b.r[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(acc.r[2]), "+r"(acc.r[3])
        : "r"(a.r[0]), "r"(a.r[1]), "r"(a.r[2]), "r"(a.r[3]),
          "r"(b.r[2]), "r"(b.r[3]));
#endif
}

__device__ __forceinline__ void frag_relu(mma_frag<ML::RM>& f) {
    const __half2 zero = __float2half2_rn(0.f);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        __half2 v; memcpy(&v, &f.r[k], 4);
        v = __hmax2(v, zero);
        memcpy(&f.r[k], &v, 4);
    }
}

__device__ __forceinline__ void frag_sigmoid(mma_frag<ML::RM>& f) {
    const __half2 one = __float2half2_rn(1.f);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        __half2 v; memcpy(&v, &f.r[k], 4);
        __half2 e = h2exp(__hneg2(v));
        v = __h2div(one, __hadd2(one, e));
        memcpy(&f.r[k], &v, 4);
    }
}

// ── mma_mat<M,N,LAY>: M×N fp16 matrix in TCNN "native" format ────────────────
// Native format: each fragment fid is stored as 32 consecutive mma_frag structs
// (one per warp lane) at byte offset fid * 32 in the weight array.
template <uint32_t M_, uint32_t N_, ML LAY_ = ML::CM>
struct mma_mat {
    static constexpr uint32_t M = M_, N = N_;
    static constexpr ML LAY = LAY_;
    static_assert(M % 16 == 0 && N % 16 == 0, "M, N must be multiples of 16");

    static constexpr uint32_t NR = (M / 16) * (N / 16);   // number of 16×16 frags
    // Fragment stride: for RM = N/16 frags per row of tiles; for CM = M/16 per col
    static constexpr uint32_t FS = (LAY == ML::RM) ? (N / 16) : (M / 16);

    mma_frag<LAY> frags[NR];

    __device__ mma_frag<LAY>& frag(uint32_t r16, uint32_t c16) {
        return frags[(LAY == ML::RM) ? (r16 * FS + c16) : (c16 * FS + r16)];
    }
    __device__ const mma_frag<LAY>& frag(uint32_t r16, uint32_t c16) const {
        return frags[(LAY == ML::RM) ? (r16 * FS + c16) : (c16 * FS + r16)];
    }

    // Load weight matrix from TCNN native format (each lane loads its frag entry)
    __device__ static mma_mat from_native(const __half* __restrict__ data) {
        mma_mat r;
        const uint32_t lid = lane_id();
        #pragma unroll
        for (uint32_t i = 0; i < NR; ++i)
            r.frags[i] = ((const mma_frag<LAY>*)data)[i * WARP + lid];
        return r;
    }

    __device__ static mma_mat zero() {
        mma_mat r; memset(&r, 0, sizeof(r)); return r;
    }

    // Activate all fragments in-place
    __device__ void relu() {
        #pragma unroll
        for (uint32_t i = 0; i < NR; ++i) frag_relu(frags[i]);
    }
    __device__ void sigmoid() {
        #pragma unroll
        for (uint32_t i = 0; i < NR; ++i) frag_sigmoid(frags[i]);
    }
};

template <uint32_t N> using mma_vec = mma_mat<32, N, ML::RM>;

// (M×K RM) × (K×N CM) → (M×N RM)
template <uint32_t M, uint32_t K, uint32_t N>
__device__ __forceinline__
mma_mat<M, N, ML::RM> matmul(const mma_mat<M, K, ML::RM>& A,
                              const mma_mat<K, N, ML::CM>& B) {
    auto C = mma_mat<M, N, ML::RM>::zero();
    #pragma unroll
    for (uint32_t fr = 0; fr < M/16; ++fr)
        #pragma unroll
        for (uint32_t fc = 0; fc < N/16; ++fc)
            #pragma unroll
            for (uint32_t fk = 0; fk < K/16; ++fk)
                mma_frag_accum(C.frag(fr, fc), A.frag(fr, fk), B.frag(fk, fc));
    return C;
}

// (M×K RM) × (K×N CM weights in native memory) → (M×N RM)
//
// Streams the weight fragments from memory inside the accumulation loop
// instead of materializing the whole mma_mat<K, N> in registers first.
// For a 128×128 layer the materialized form costs 256 regs/thread (forced
// spills); streaming keeps exactly one 4-register fragment live at a time.
// Loads stay fully coalesced (16 B per lane per fragment, lane-contiguous).
template <uint32_t M, uint32_t K, uint32_t N>
__device__ __forceinline__
mma_mat<M, N, ML::RM> matmul_native(const mma_mat<M, K, ML::RM>& A,
                                     const __half* __restrict__ params) {
    constexpr uint32_t FS = K / 16;   // CM fragment stride
    const uint32_t lid = lane_id();
    auto C = mma_mat<M, N, ML::RM>::zero();
    #pragma unroll
    for (uint32_t fc = 0; fc < N/16; ++fc)
        #pragma unroll
        for (uint32_t fk = 0; fk < K/16; ++fk) {
            const mma_frag<ML::CM> b =
                ((const mma_frag<ML::CM>*)params)[(fc * FS + fk) * WARP + lid];
            #pragma unroll
            for (uint32_t fr = 0; fr < M/16; ++fr)
                mma_frag_accum(C.frag(fr, fc), A.frag(fr, fk), b);
        }
    return C;
}

// Build mma_vec<N> (32×N RM, one row per lane) from a per-thread tvec<half, N>
// using warp shuffles only — no shared memory. Mirrors tcnn's
// mma_mat(const tvec&) constructor exactly (see tcnn mma.h).
template <uint32_t N>
__device__ __forceinline__
mma_vec<N> vec_to_mma(const tvec<__half, N>& v) {
    constexpr uint32_t FS_ = N / 16;
    mma_vec<N> r;
    const uint32_t lid  = lane_id();
    const uint32_t xlid = lid % 4;
    const uint32_t ylid = lid / 4;

    #pragma unroll
    for (uint32_t i = 0; i < N; i += 2) {
        __half2 val;
        val.x = v[i];
        val.y = (i + 1 < N) ? v[i + 1] : __float2half(0.f);
        uint32_t src; memcpy(&src, &val, 4);

        const uint32_t xi = (i / 2) % 4;
        #pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t paired_lid = j * 8 + ylid;
            // reg_from_xy(x=i, y=j*8) for RM:
            const uint32_t x   = i, y = j * 8;
            const uint32_t fid = x / 16 + (y / 16) * FS_;
            const uint32_t bx  = y / 8;
            const uint32_t by  = x / 8;
            const uint32_t bid = (bx % 2) + (by % 2) * 2;
            const uint32_t other = __shfl_sync(0xFFFFFFFFu, src, paired_lid);
            if (xlid == xi) r.frags[fid].r[bid] = other;
        }
    }
    return r;
}

// Extract N_OUT per-thread half values from mma_vec<N> using TCNN's shfl method.
//
// Uses __shfl_sync to gather each thread's output row without shared memory.
// This mirrors TCNN mma_mat::vec<N_OUT>() exactly:
//   xlid = (lid % 8) * 4       — selects which 4-thread group within each 8-thread block
//   ylid = lid / 8              — selects which 8-thread block (j index)
//   paired_lid = xlid + (i/2)%4 — the lane that holds column i of this thread's row
//
// No smem argument required (smem pointer kept for interface compatibility but unused).
template <uint32_t N, uint32_t N_OUT>
__device__ __forceinline__
tvec<__half, N_OUT> mma_to_vec(const mma_vec<N>& v) {
    static_assert(N_OUT <= N, "N_OUT must be <= N");
    constexpr uint32_t FS = N / 16u;  // fragment stride for RM layout
    tvec<__half, N_OUT> r;
    const uint32_t lid  = lane_id();
    const uint32_t xlid = (lid % 8u) * 4u;
    const uint32_t ylid = lid / 8u;

    #pragma unroll
    for (uint32_t i = 0; i < N; i += 2) {
        const uint32_t paired_lid = xlid + (i / 2u) % 4u;
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            // reg_from_xy(x=i, y=j*8) for RM:
            const uint32_t x   = i, y = j * 8u;
            const uint32_t fid = x / 16u + (y / 16u) * FS;
            const uint32_t bx  = y / 8u;    // for RM: bx = y/8
            const uint32_t by  = x / 8u;    // for RM: by = x/8
            const uint32_t bid = (bx % 2u) + (by % 2u) * 2u;
            const uint32_t src = v.frags[fid].r[bid];
            const uint32_t other = __shfl_sync(0xFFFFFFFFu, src, paired_lid);
            if (ylid == j) {
                __half2 h2; memcpy(&h2, &other, 4);
                if (i + 0u < N_OUT) r[i + 0u] = h2.x;
                if (i + 1u < N_OUT) r[i + 1u] = h2.y;
            }
        }
    }
    return r;
}

// ── SH degree-3 encoding ──────────────────────────────────────────────────────
// Fills positions 0..8 (l=0,1,2) and pads 9..SH_DIM-1 with 1.0f.
// (x,y,z) must be in [-1,1] range.
template <uint32_t SH_DIM>
__device__ __forceinline__
void sh3_enc(float x, float y, float z, tvec<__half, SH_DIM>& out) {
    static_assert(SH_DIM >= SH3_N, "SH_DIM must be >= 9");
    const float xy = x*y, xz = x*z, yz = y*z, x2 = x*x, y2 = y*y, z2 = z*z;
    out[0] = __float2half( 0.28209479177387814f);
    out[1] = __float2half(-0.48860251190291987f * y);
    out[2] = __float2half( 0.48860251190291987f * z);
    out[3] = __float2half(-0.48860251190291987f * x);
    out[4] = __float2half( 1.0925484305920792f  * xy);
    out[5] = __float2half(-1.0925484305920792f  * yz);
    out[6] = __float2half( 0.94617469575755997f * z2 - 0.31539156525251999f);
    out[7] = __float2half(-1.0925484305920792f  * xz);
    out[8] = __float2half( 0.54627421529603959f * (x2 - y2));
    #pragma unroll
    for (uint32_t i = SH3_N; i < SH_DIM; ++i) out[i] = __float2half(1.0f);
}

// ── nht_fused_shade<FEAT_OUT, ENC_DIM, MLP_HIDDEN, N_HIDDEN_LAYERS> ─────────
//
// Warp-cooperative fused shade matching TCNN FullyFusedMLP + Composite encoding.
// Network layout (weight buffer offset 0):
//   Layer 0 : ENC_DIM  × MLP_HIDDEN  (input → first hidden)
//   Layers 1…N_HIDDEN_LAYERS-1: MLP_HIDDEN × MLP_HIDDEN  (hidden-to-hidden)
//   Output  : MLP_HIDDEN × 16         (last hidden → output, sigmoid, take [0:3])
//
// N_HIDDEN_LAYERS matches TCNN's n_hidden_layers parameter: the total number of
// hidden ReLU layers (= loop iterations + 1 for the first layer).
//
// ALL 32 threads of the warp must enter together.
//
template <uint32_t FEAT_OUT, uint32_t ENC_DIM,
          uint32_t MLP_HIDDEN, uint32_t N_HIDDEN_LAYERS>
__device__ __forceinline__
tvec<__half, 3> nht_fused_shade(
    const float* __restrict__ pix_feat,
    float rx, float ry, float rz,
    const __half* __restrict__ params
) {
    static_assert(ENC_DIM % 16 == 0,    "ENC_DIM must be divisible by 16");
    static_assert(MLP_HIDDEN % 16 == 0, "MLP_HIDDEN must be divisible by 16");
    static_assert(ENC_DIM >= FEAT_OUT + SH3_N, "ENC_DIM too small");
    static_assert(N_HIDDEN_LAYERS >= 1, "N_HIDDEN_LAYERS must be >= 1");

    constexpr uint32_t SH_DIM  = ENC_DIM - FEAT_OUT;
    constexpr uint32_t OUT_PAD = 16u;

    // ── 1. Build encoded tvec<half, ENC_DIM> ───────────────────────────────
    tvec<__half, ENC_DIM> enc;
    #pragma unroll
    for (uint32_t i = 0; i < FEAT_OUT; ++i) enc[i] = __float2half(pix_feat[i]);
    {
        tvec<__half, SH_DIM> sh;
        sh3_enc<SH_DIM>(fmaf(2.f,rx,-1.f), fmaf(2.f,ry,-1.f), fmaf(2.f,rz,-1.f), sh);
        #pragma unroll
        for (uint32_t i = 0; i < SH_DIM; ++i) enc[FEAT_OUT + i] = sh[i];
    }

    // ── 2. Layer 0: ENC_DIM → MLP_HIDDEN, ReLU ────────────────────────────
    auto h = matmul_native<32, ENC_DIM, MLP_HIDDEN>(
        vec_to_mma<ENC_DIM>(enc), params);
    params += ENC_DIM * MLP_HIDDEN;
    h.relu();

    // ── 3. Hidden-to-hidden: N_HIDDEN_LAYERS − 1 more layers, ReLU ─────────
    #pragma unroll
    for (uint32_t l = 0; l < N_HIDDEN_LAYERS - 1; ++l) {
        h = matmul_native<32, MLP_HIDDEN, MLP_HIDDEN>(h, params);
        params += MLP_HIDDEN * MLP_HIDDEN;
        h.relu();
    }

    // ── 4. Output: MLP_HIDDEN → OUT_PAD, Sigmoid ──────────────────────────
    auto out = matmul_native<32, MLP_HIDDEN, OUT_PAD>(h, params);
    out.sigmoid();
    return mma_to_vec<OUT_PAD, 3>(out);
}

// Compute ENC_DIM from FEAT_OUT at compile time
template <uint32_t FEAT_OUT>
static constexpr uint32_t enc_dim_v = round_up16(FEAT_OUT + SH3_N);

////////////////////////////////////////////////////////////////
// Backward pass building blocks
////////////////////////////////////////////////////////////////

// Transpose one 8x8 b16 sub-block held in a register (warp-cooperative).
__device__ __forceinline__ uint32_t transpose_reg(uint32_t reg) {
#if __CUDA_ARCH__ >= 800
    asm("movmatrix.sync.trans.aligned.m8n8.b16 %0, %0;" : "+r"(reg));
#endif
    return reg;
}

// Transpose a 16x16 fragment (same layout label; the 2x2 sub-block grid is
// col-major independent of layout, hence the index swap — mirrors tcnn).
template <ML LAY>
__device__ __forceinline__ mma_frag<LAY> frag_transpose(const mma_frag<LAY>& f) {
    mma_frag<LAY> r;
    #pragma unroll
    for (uint32_t rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (uint32_t rc = 0; rc < 2; ++rc)
            r.r[rr * 2 + rc] = transpose_reg(f.r[rr + rc * 2]);
    return r;
}

// Reinterpret an RM fragment as the CM fragment of the same 16x16 values
// (per-register movmatrix, register order unchanged — mirrors tcnn flip_layout).
__device__ __forceinline__ mma_frag<ML::CM> frag_flip_to_cm(const mma_frag<ML::RM>& f) {
    mma_frag<ML::CM> r;
    #pragma unroll
    for (uint32_t k = 0; k < 4; ++k) r.r[k] = transpose_reg(f.r[k]);
    return r;
}

// dz = dh ⊙ (h > 0)   (ReLU backward, h = post-activation = pre-act sign proxy)
__device__ __forceinline__ void frag_relu_bwd(
    mma_frag<ML::RM>& d, const mma_frag<ML::RM>& h
) {
    const __half2 zero = __float2half2_rn(0.f);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        __half2 dv, hv;
        memcpy(&dv, &d.r[k], 4);
        memcpy(&hv, &h.r[k], 4);
        dv = __hmul2(dv, __hgt2(hv, zero));
        memcpy(&d.r[k], &dv, 4);
    }
}

// dz = dy ⊙ y ⊙ (1 − y)   (sigmoid backward, y = recomputed sigmoid output)
__device__ __forceinline__ void frag_sigmoid_bwd(
    mma_frag<ML::RM>& d, const mma_frag<ML::RM>& y
) {
    const __half2 one = __float2half2_rn(1.f);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        __half2 dv, yv;
        memcpy(&dv, &d.r[k], 4);
        memcpy(&yv, &y.r[k], 4);
        dv = __hmul2(dv, __hmul2(yv, __hsub2(one, yv)));
        memcpy(&d.r[k], &dv, 4);
    }
}

template <uint32_t M, uint32_t N>
__device__ __forceinline__ void mat_relu_bwd(
    mma_mat<M, N, ML::RM>& d, const mma_mat<M, N, ML::RM>& h
) {
    #pragma unroll
    for (uint32_t i = 0; i < mma_mat<M, N, ML::RM>::NR; ++i)
        frag_relu_bwd(d.frags[i], h.frags[i]);
}

// dX(M×K) = dY(M×N) × Wᵀ, with W a (K×N) matrix in native CM fragment layout.
// Streams + transposes (movmatrix) the weight fragments on the fly.
template <uint32_t M, uint32_t N_IN, uint32_t K_OUT>
__device__ __forceinline__
mma_mat<M, K_OUT, ML::RM> matmul_native_T(const mma_mat<M, N_IN, ML::RM>& dY,
                                           const __half* __restrict__ params) {
    constexpr uint32_t FS = K_OUT / 16;   // W's CM fragment stride
    const uint32_t lid = lane_id();
    auto C = mma_mat<M, K_OUT, ML::RM>::zero();
    #pragma unroll
    for (uint32_t fc = 0; fc < K_OUT/16; ++fc)
        #pragma unroll
        for (uint32_t tn = 0; tn < N_IN/16; ++tn) {
            // W.frag(k_tile=fc, n_tile=tn), transposed → Wᵀ tile (tn, fc)
            mma_frag<ML::CM> w =
                ((const mma_frag<ML::CM>*)params)[(tn * FS + fc) * WARP + lid];
            w = frag_transpose(w);
            #pragma unroll
            for (uint32_t fr = 0; fr < M/16; ++fr)
                mma_frag_accum(C.frag(fr, fc), dY.frag(fr, tn), w);
        }
    return C;
}

// dW(K×N) += Xᵀ(K×M) @ dY(M×N) accumulated into a shared fp16 slab kept in
// NATIVE fragment layout (16 B per lane per fragment — coalesced, conflict
// free, vectorized half2 adds). The flush converts to tcnn linear layout.
//
// Warps are serialized by the caller-supplied loop (block-uniform!): warp w
// adds its outer product on iteration w; everyone else waits at the barrier.
// One streamed 16×16 fragment is live at a time (no dW materialization).
template <uint32_t M, uint32_t K, uint32_t N>
__device__ __forceinline__ void outer_product_into_slab(
    const mma_mat<M, K, ML::RM>& X,
    const mma_mat<M, N, ML::RM>& dY,
    __half* __restrict__ slab,
    uint32_t warp_id,
    uint32_t num_warps
) {
    constexpr uint32_t FS  = K / 16;
    constexpr uint32_t NRT = (K / 16) * (N / 16);
    // Process fragments in register-resident chunks: every warp computes its
    // chunk in parallel (mma + movmatrix), then only the cheap slab adds are
    // serialized across warps. Keeps at most CHUNK fragments (4 regs each)
    // live per thread.
    constexpr uint32_t CHUNK = NRT < 8u ? NRT : 8u;
    const uint32_t lid = lane_id();

    // Fully unrolled: base/fr/fc must be compile-time constants, otherwise the
    // fragment register arrays (X, dY, accs) spill to local memory.
    #pragma unroll
    for (uint32_t base = 0; base < NRT; base += CHUNK) {
        mma_frag<ML::CM> accs[CHUNK];
        #pragma unroll
        for (uint32_t c = 0; c < CHUNK; ++c) {
            const uint32_t fid = base + c;     // CM ordering: fid = fc*FS + fr
            if (fid >= NRT) break;
            const uint32_t fr = fid % FS;
            const uint32_t fc = fid / FS;
            mma_frag<ML::RM> acc = {0u, 0u, 0u, 0u};
            #pragma unroll
            for (uint32_t kb = 0; kb < M/16; ++kb) {
                // Xᵀ tile (fr, kb) = transpose of X tile (kb, fr)
                mma_frag<ML::RM> a = frag_transpose(X.frag(kb, fr));
                mma_frag<ML::CM> b = frag_flip_to_cm(dY.frag(kb, fc));
                mma_frag_accum(acc, a, b);
            }
            accs[c] = frag_flip_to_cm(acc);
        }
        for (uint32_t w = 0; w < num_warps; ++w) {
            if (w == warp_id) {
                #pragma unroll
                for (uint32_t c = 0; c < CHUNK; ++c) {
                    const uint32_t fid = base + c;
                    if (fid >= NRT) break;
                    // One 16-byte lane-private entry, 4 vectorized half2 adds.
                    __half2* entry = (__half2*)&slab[(fid * WARP + lid) * 8u];
                    #pragma unroll
                    for (uint32_t r = 0; r < 4; ++r) {
                        __half2 v; memcpy(&v, &accs[c].r[r], 4);
                        entry[r] = __hadd2(entry[r], v);
                    }
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void slab_zero(
    __half* slab, uint32_t n, uint32_t tid, uint32_t nthreads
) {
    // n is always a multiple of 2 here (K*N with K,N multiples of 16)
    __half2* s2 = (__half2*)slab;
    const __half2 z = __float2half2_rn(0.f);
    for (uint32_t i = tid; i < n / 2; i += nthreads) s2[i] = z;
}

// Flush the block's fp16 native-fragment partial into the global fp32
// gradient buffer in tcnn LINEAR (column-major, k + n*K) order.
// Each half2 covers rows (k, k+1) of one column n → two consecutive linear
// elements, i.e. two coalesced-ish fp32 atomics.
template <uint32_t K>
__device__ __forceinline__ void slab_flush_atomic(
    const __half* slab, uint32_t n_halves, float* __restrict__ g,
    uint32_t tid, uint32_t nthreads
) {
    constexpr uint32_t FS = K / 16;
    const __half2* s2 = (const __half2*)slab;
    for (uint32_t i = tid; i < n_halves / 2; i += nthreads) {
        const uint32_t p    = i * 2;
        const uint32_t fid  = p >> 8;          // 256 halves per fragment
        const uint32_t rem  = p & 255u;
        const uint32_t lane = rem >> 3;
        const uint32_t reg  = (rem >> 1) & 3u;
        // CM fragment → (k, n): rx = reg%2 (row block), ry = reg/2 (col block)
        const uint32_t k = (fid % FS) * 16 + (lane % 4) * 2 + (reg % 2) * 8;
        const uint32_t n = (fid / FS) * 16 + (lane / 4) + (reg / 2) * 8;
        const __half2 v = s2[i];
        const float vx = __half2float(v.x);
        const float vy = __half2float(v.y);
        float* dst = g + n * K + k;
        if (vx != 0.f) atomicAdd(dst,     vx);
        if (vy != 0.f) atomicAdd(dst + 1, vy);
    }
}

// ── nht_fused_shade_bwd ──────────────────────────────────────────────────────
//
// Warp-cooperative backward of nht_fused_shade. Recomputes the forward pass
// from the saved per-pixel accumulated features, then backpropagates the
// (loss-scaled) RGB gradient through sigmoid → output layer → hidden layers →
// input encoding. Produces:
//   - return value: dL/d pix_feat (fp32, unscaled) for the rasterizer backward
//   - v_params:     dL/d weights accumulated via fp32 atomics in tcnn LINEAR
//                   layout (still multiplied by loss_scale; divide on host)
//
// Block-uniform: ALL threads of the block must call this together (the dW
// slab reduction serializes warps with __syncthreads).
//
// dw_slab must hold MLP_HIDDEN*MLP_HIDDEN halves (largest layer).
//
template <uint32_t FEAT_OUT, uint32_t ENC_DIM,
          uint32_t MLP_HIDDEN, uint32_t N_HIDDEN_LAYERS>
__device__ __forceinline__
tvec<float, FEAT_OUT> nht_fused_shade_bwd(
    const float* __restrict__ pix_feat,
    float rx, float ry, float rz,
    const float v_rgb[3],            // upstream dL/dRGB (fp32, unscaled)
    const float loss_scale,
    const __half* __restrict__ params,      // native fragment layout
    float* __restrict__ v_params,           // [n_params] fp32, linear layout
    __half* __restrict__ dw_slab,            // shared, H*H halves, block-shared
    const uint32_t warp_id,
    const uint32_t num_warps,
    const uint32_t tid,
    const uint32_t nthreads
) {
    static_assert(ENC_DIM % 16 == 0,    "ENC_DIM must be divisible by 16");
    static_assert(MLP_HIDDEN % 16 == 0, "MLP_HIDDEN must be divisible by 16");
    static_assert(N_HIDDEN_LAYERS >= 1, "N_HIDDEN_LAYERS must be >= 1");
    constexpr uint32_t SH_DIM  = ENC_DIM - FEAT_OUT;
    constexpr uint32_t OUT_PAD = 16u;
    constexpr uint32_t W0_OFF  = 0;
    constexpr uint32_t WOUT_OFF =
        ENC_DIM * MLP_HIDDEN + (N_HIDDEN_LAYERS - 1) * MLP_HIDDEN * MLP_HIDDEN;

    // ── Forward recompute (identical to nht_fused_shade) ───────────────────
    tvec<__half, ENC_DIM> enc;
    #pragma unroll
    for (uint32_t i = 0; i < FEAT_OUT; ++i) enc[i] = __float2half(pix_feat[i]);
    {
        tvec<__half, SH_DIM> sh;
        sh3_enc<SH_DIM>(fmaf(2.f,rx,-1.f), fmaf(2.f,ry,-1.f), fmaf(2.f,rz,-1.f), sh);
        #pragma unroll
        for (uint32_t i = 0; i < SH_DIM; ++i) enc[FEAT_OUT + i] = sh[i];
    }
    const auto enc_mma = vec_to_mma<ENC_DIM>(enc);

    mma_vec<MLP_HIDDEN> h[N_HIDDEN_LAYERS];   // post-ReLU activations
    h[0] = matmul_native<32, ENC_DIM, MLP_HIDDEN>(enc_mma, params + W0_OFF);
    h[0].relu();
    #pragma unroll
    for (uint32_t l = 1; l < N_HIDDEN_LAYERS; ++l) {
        h[l] = matmul_native<32, MLP_HIDDEN, MLP_HIDDEN>(
            h[l - 1], params + ENC_DIM * MLP_HIDDEN + (l - 1) * MLP_HIDDEN * MLP_HIDDEN);
        h[l].relu();
    }
    auto y = matmul_native<32, MLP_HIDDEN, OUT_PAD>(
        h[N_HIDDEN_LAYERS - 1], params + WOUT_OFF);
    y.sigmoid();

    // ── Sigmoid backward: dz_out = (dy * S) ⊙ y ⊙ (1−y), zero-padded ───────
    tvec<__half, OUT_PAD> dy;
    #pragma unroll
    for (uint32_t i = 0; i < OUT_PAD; ++i)
        dy[i] = (i < 3) ? __float2half(v_rgb[i] * loss_scale) : __float2half(0.f);
    auto dz = vec_to_mma<OUT_PAD>(dy);
    #pragma unroll
    for (uint32_t i = 0; i < mma_vec<OUT_PAD>::NR; ++i)
        frag_sigmoid_bwd(dz.frags[i], y.frags[i]);

    // dW accumulation can be skipped entirely (frozen MLP / ablation) by
    // passing v_params == nullptr. The pointer is block-uniform, so the
    // __syncthreads inside the slab helpers stay block-uniform too.
    const bool want_dw = (v_params != nullptr);

    // ── Output layer: dW_out += h_lastᵀ @ dz; dh = dz × W_outᵀ ─────────────
    if (want_dw) {
        slab_zero(dw_slab, MLP_HIDDEN * OUT_PAD, tid, nthreads);
        __syncthreads();
        outer_product_into_slab<32, MLP_HIDDEN, OUT_PAD>(
            h[N_HIDDEN_LAYERS - 1], dz, dw_slab, warp_id, num_warps);
        slab_flush_atomic<MLP_HIDDEN>(dw_slab, MLP_HIDDEN * OUT_PAD,
                                      v_params + WOUT_OFF, tid, nthreads);
    }

    auto dh = matmul_native_T<32, OUT_PAD, MLP_HIDDEN>(dz, params + WOUT_OFF);
    mat_relu_bwd(dh, h[N_HIDDEN_LAYERS - 1]);

    // ── Hidden layers, back to front ───────────────────────────────────────
    #pragma unroll
    for (uint32_t li = N_HIDDEN_LAYERS - 1; li >= 1; --li) {
        const uint32_t w_off =
            ENC_DIM * MLP_HIDDEN + (li - 1) * MLP_HIDDEN * MLP_HIDDEN;
        if (want_dw) {
            __syncthreads();
            slab_zero(dw_slab, MLP_HIDDEN * MLP_HIDDEN, tid, nthreads);
            __syncthreads();
            outer_product_into_slab<32, MLP_HIDDEN, MLP_HIDDEN>(
                h[li - 1], dh, dw_slab, warp_id, num_warps);
            slab_flush_atomic<MLP_HIDDEN>(dw_slab, MLP_HIDDEN * MLP_HIDDEN,
                                          v_params + w_off, tid, nthreads);
        }

        auto dh_prev = matmul_native_T<32, MLP_HIDDEN, MLP_HIDDEN>(dh, params + w_off);
        mat_relu_bwd(dh_prev, h[li - 1]);
        dh = dh_prev;
    }

    // ── Input layer: dW0 += encᵀ @ dh; dEnc = dh × W0ᵀ ─────────────────────
    if (want_dw) {
        __syncthreads();
        slab_zero(dw_slab, ENC_DIM * MLP_HIDDEN, tid, nthreads);
        __syncthreads();
        outer_product_into_slab<32, ENC_DIM, MLP_HIDDEN>(
            enc_mma, dh, dw_slab, warp_id, num_warps);
        slab_flush_atomic<ENC_DIM>(dw_slab, ENC_DIM * MLP_HIDDEN,
                                   v_params + W0_OFF, tid, nthreads);
    }

    auto denc = matmul_native_T<32, MLP_HIDDEN, ENC_DIM>(dh, params + W0_OFF);
    auto dfeat_h = mma_to_vec<ENC_DIM, FEAT_OUT>(denc);

    const float inv_scale = 1.f / loss_scale;
    tvec<float, FEAT_OUT> out;
    #pragma unroll
    for (uint32_t i = 0; i < FEAT_OUT; ++i)
        out[i] = __half2float(dfeat_h[i]) * inv_scale;
    return out;
}

} // namespace nht_mlp
