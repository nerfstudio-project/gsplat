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

#include <cuda_fp16.h>
#include <cstdint>
#include "SHCompression.h"
#include "Utils.h"

namespace higs {

// input mode for the unified K=16 SH evaluation kernel
enum class SHInputMode
{
    RAW_FLOAT,
    RAW_HALF,
    COMPRESS_32B,
    COMPRESS_16B
};

// 32-byte SH coefficient compression for K=16 (degree 3).
//
// Packed layout (256 bits = 8 × uint32):
//
//   block[0] bits  0..15: DC_Y fp16 (16 bits, lossless)
//   block[0] bits 16..30: DC_Co fp15 (fp16 >> 1, drop mantissa LSB)
//   block[0] bit  31:     DC_Cg fp15 bit 14 (MSB)
//
//   CoCg interleaved region (uint32-aligned, k=2..15, unsigned bias U1 encoding):
//     block[1]: [Co2|Cg2, Co3|Cg3, Co4|Cg4, Co5|Cg5]   each byte = (qCo+8)<<4 | (qCg+8)
//     block[2]: [Co6|Cg6, Co7|Cg7, Co8|Cg8, Co9|Cg9]
//     block[3]: [Co10|Cg10, Co11|Cg11, Co12|Cg12, Co13|Cg13]
//     block[4] bits 0..15: [Co14|Cg14, Co15|Cg15]
//
//   Remaining region (112 bits in block[4] bits 16..31 + block[5..7]):
//     bits  0..13:  DC_Cg fp15 low 14 bits (total Cg = 1 + 14 = 15 bits)
//     bits 14..19:  k=1 Y  (6-bit unsigned bias: qY+32)
//     bits 20..23:  k=1 Co (4-bit unsigned bias: qCo+8)
//     bits 24..27:  k=1 Cg (4-bit unsigned bias: qCg+8)
//     bits 28..33:  Y2  (6-bit unsigned bias: qY+32) ─┐
//     bits 34..39:  Y3                                  │ 14 × 6-bit Y values
//     bits 40..45:  Y4                                  │ for k=2..15
//     ...                                               │ (funnel shift extraction)
//     bits 106..111: Y15                               ─┘
//
//   HO quantization: signed q ∈ [-(max_q+1), max_q] stored as unsigned (q + max_q + 1).
//     Y:    6-bit, max_q=31, stored as (q+32) ∈ [0,63], spread to uint8 as (q+32)<<2
//     CoCg: 4-bit, max_q=7,  stored as (q+8)  ∈ [0,15], spread to uint8 as (q+8)<<4
//   Decoder uses __uint8x4_to_half4 (no sign correction needed), then FMA:
//     normalized = val * inv_norm + offset   (offset = -(max_q+1)/max_q)
//
//   Per-basis scaling: each SH basis k has its own Y/Co/Cg scale factor (p99.99 percentile).
//   Scales are stored in SHDecodeParams (half2[8] per channel) for the decoder and
//   SHEncodeParams (float[16] per channel) for the encoder.
//
//   Bit budget: 32 + 96 + 16 + 14 + 14 + 84 = 256 bits ✓
//
// Constants for SH degree-0 basis and 3DGS color bias.
static constexpr float SH_Y00  = 0.2820947917738781f;
static constexpr float SH_BIAS = 0.5f;

// Convert any element type to float (works under __CUDA_NO_HALF_CONVERSIONS__).
template<typename T>
__device__ __forceinline__ float _sh_to_float(T v) { return static_cast<float>(v); }
template<>
__device__ __forceinline__ float _sh_to_float<__half>(__half v) { return __half2float(v); }

// Encode K=16 SH coefficients into 32-byte packed representation.
// T = float or __half.
// coeffs:      input [3][16] — channel-first (R/G/B × 16 bases), caller loads.
// block:       output [8] uint32 — caller stores.
// inv_scales:  per-basis inverse scales (float arrays, indexed by basis k).
// Fixed gamma = 1/2 — matches decoder's square_signed_h2 in Phase 2.
template<typename T>
__device__ void sh_encode_32b(const T (&coeffs)[3][16], uint32_t (&block)[8], const float (&inv_scales)[3][16])
{
    // Linear RGB → YCoCg-R-style luma/chroma basis used by the codec.
    auto rgb_to_ycocg = [](float R, float G, float B, float &Y, float &Co, float &Cg) {
        Y  = 0.25f * R + 0.5f * G + 0.25f * B;
        Co = 0.5f * R - 0.5f * B;
        Cg = -0.25f * R + 0.5f * G - 0.25f * B;
    };

    // Quantize to the float domain consumed by __cvt_pack_sat_u8_f32.
    // Full two's complement range: q ∈ [-(max_q+1), max_q], stored as q + (max_q+1) ∈ [0, 2*max_q+1].
    // Matches the decoder: normalized = q/max_q via FMA(stored*spread, inv_norm, offset).
    // `roundf` is used explicitly (round half away from zero) because __cvt_pack_sat_u8_f32
    // does cvt.rni (round half to even) which tiebreaks differently at bin boundaries and
    // can shift reconstructed values by ~0.1 near the edge of range (where post-square bin
    // width is large). Pre-rounding yields an integer-valued float so the hardware rni is
    // idempotent; saturation in the pack still clamps to [0, 255] as a safety net.
    auto quantize_u1f = [](float val, float inv_scale, float max_q) -> float {
        float norm       = val * inv_scale;
        float compressed = copysignf(sqrtf(fabsf(norm)), norm);
        float q          = fminf(fmaxf(roundf(compressed * max_q), -(max_q + 1.0f)), max_q);
        return q + (max_q + 1.0f);
    };

    // --- DC (k=0) : YCoCg in fp16 / fp15 / fp15 ---
    float R0 = _sh_to_float(coeffs[0][0]) * SH_Y00 + SH_BIAS;
    float G0 = _sh_to_float(coeffs[1][0]) * SH_Y00 + SH_BIAS;
    float B0 = _sh_to_float(coeffs[2][0]) * SH_Y00 + SH_BIAS;

    float dc_Y, dc_Co, dc_Cg;
    rgb_to_ycocg(R0, G0, B0, dc_Y, dc_Co, dc_Cg);

    uint16_t y_bits;
    AssignAs<uint16_t>(y_bits, __float2half(dc_Y));
    const __half2 tr_cocg = RoundToNearest<9, false>(__floats2half2_rn(dc_Co, dc_Cg));
    const uint32_t co15   = (reinterpret_cast<const uint32_t &>(tr_cocg) >> 1) & 0x7FFF;
    const uint32_t cg15   = (reinterpret_cast<const uint32_t &>(tr_cocg) >> 17) & 0x7FFF;
    block[0]              = static_cast<uint32_t>(y_bits) | (co15 << 16) | ((cg15 >> 14) << 31);

    // --- k=1 (HO head): 6-bit Y + 4-bit Co + 4-bit Cg, all biased ---
    // One __cvt_pack_sat_u8_f32 gives us {uY, uCo, uCg, 0} in a u8x4.
    float Y1, Co1, Cg1;
    {
        float R = _sh_to_float(coeffs[0][1]);
        float G = _sh_to_float(coeffs[1][1]);
        float B = _sh_to_float(coeffs[2][1]);
        rgb_to_ycocg(R, G, B, Y1, Co1, Cg1);
    }
    const uint32_t k1_u8x4 =
        __cvt_pack_sat_u8_f32(quantize_u1f(Y1, inv_scales[0][1], 31.f), quantize_u1f(Co1, inv_scales[1][1], 7.f),
                              quantize_u1f(Cg1, inv_scales[2][1], 7.f), 0.f);

    // --- HO groups g=0..3 spanning k=2..15 ---
    // Each group produces one CoCg output word + one 24-bit Y chunk.
    // Group 3 only has k=14,15 live; the rest is zero-padded and the CoCg tail
    // uses pack_4x8_to_4x4 into the low 16 bits of block[4].
    uint32_t y_packed[4];
    uint32_t cocg_tail16 = 0;
#pragma unroll
    for (int g = 0; g < 4; g++)
    {
        float Yg[4]  = {0.f, 0.f, 0.f, 0.f};
        float Cog[4] = {0.f, 0.f, 0.f, 0.f};
        float Cgg[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            const int k = 2 + g * 4 + j;
            if (k <= 15)
            {
                float R = _sh_to_float(coeffs[0][k]);
                float G = _sh_to_float(coeffs[1][k]);
                float B = _sh_to_float(coeffs[2][k]);
                rgb_to_ycocg(R, G, B, Yg[j], Cog[j], Cgg[j]);
            }
        }

        // Compress u8x4 of 4 Y lanes (each ≤ 0x3F) into a 24-bit field:
        //   Y0 | (Y1<<6) | (Y2<<12) | (Y3<<18). Top 2 bits of each byte are dropped.
        const int k_base    = 2 + g * 4;
        const uint32_t Y_u8 = __cvt_pack_sat_u8_f32(
            quantize_u1f(Yg[0], inv_scales[0][k_base], 31.f), quantize_u1f(Yg[1], inv_scales[0][k_base + 1], 31.f),
            quantize_u1f(Yg[2], inv_scales[0][k_base + 2], 31.f), quantize_u1f(Yg[3], inv_scales[0][k_base + 3], 31.f));
        y_packed[g] =
            (Y_u8 & 0x3Fu) | ((Y_u8 & 0x3F00u) >> 2) | ((Y_u8 & 0x3F0000u) >> 4) | ((Y_u8 & 0x3F000000u) >> 6);

        if (g < 3)
        {
            // Bytes will be (qCo+8)<<4 | (qCg+8) as the decoder expects.
            const uint32_t Co_u8 = __cvt_pack_sat_u8_f32(quantize_u1f(Cog[0], inv_scales[1][k_base], 7.f),
                                                         quantize_u1f(Cog[1], inv_scales[1][k_base + 1], 7.f),
                                                         quantize_u1f(Cog[2], inv_scales[1][k_base + 2], 7.f),
                                                         quantize_u1f(Cog[3], inv_scales[1][k_base + 3], 7.f));
            const uint32_t Cg_u8 = __cvt_pack_sat_u8_f32(quantize_u1f(Cgg[0], inv_scales[2][k_base], 7.f),
                                                         quantize_u1f(Cgg[1], inv_scales[2][k_base + 1], 7.f),
                                                         quantize_u1f(Cgg[2], inv_scales[2][k_base + 2], 7.f),
                                                         quantize_u1f(Cgg[3], inv_scales[2][k_base + 3], 7.f));
            block[1 + g]         = (Co_u8 << 4) | Cg_u8;
        }
        else
        {
            // Tail: k=14,15 live in lanes 0,1; interleave {Cg14, Co14, Cg15, Co15} and
            // compress u8x4 (each lane ≤ 0xF) into a 16-bit field by dropping the top
            // nibble of every byte, yielding bytes [(Co14<<4)|Cg14, (Co15<<4)|Cg15].
            const uint32_t tail_u8 = __cvt_pack_sat_u8_f32(quantize_u1f(Cgg[0], inv_scales[2][k_base], 7.f),
                                                           quantize_u1f(Cog[0], inv_scales[1][k_base], 7.f),
                                                           quantize_u1f(Cgg[1], inv_scales[2][k_base + 1], 7.f),
                                                           quantize_u1f(Cog[1], inv_scales[1][k_base + 1], 7.f));
            cocg_tail16            = (tail_u8 & 0xFu) | ((tail_u8 & 0xF00u) >> 4) | ((tail_u8 & 0xF0000u) >> 8) |
                          ((tail_u8 & 0xF000000u) >> 12);
        }
    }

    // --- Remaining 112-bit stream: block[4] hi16 + block[5..7] ---
    // 28-bit header packs cg15_lo14 | uY1(6) | uCo1(4) | uCg1(4).
    // k1_u8x4 byte layout: [uY1, uCo1, uCg1, 0] → shift each into place.
    const uint32_t hdr28 = (cg15 & 0x3FFFu)                    // bits  0..13
                           | ((k1_u8x4 & 0x3Fu) << 14)         // bits 14..19: uY (6 bits)
                           | (((k1_u8x4 >> 8) & 0xFu) << 20)   // bits 20..23: uCo (4 bits)
                           | (((k1_u8x4 >> 16) & 0xFu) << 24); // bits 24..27: uCg (4 bits)

    // Four 24-bit Y chunks placed contiguously starting at bit 28 of the 128-bit stream.
    // Simple shifts+OR here (not funnel shifts): y_packed[] are 24-bit payloads with
    // zero high 8 bits, so naive funnel shift would inject those zeros into the seam.
    uint32_t rem[4];
    rem[0] = hdr28 | (y_packed[0] << 28);
    rem[1] = (y_packed[0] >> 4) | (y_packed[1] << 20);
    rem[2] = (y_packed[1] >> 12) | (y_packed[2] << 12);
    rem[3] = (y_packed[2] >> 20) | (y_packed[3] << 4);

    // Final 32-bit stitch between rem[] words and the pre-filled block[4] low 16
    // is a plain 16-bit funnel shift since rem[] are fully populated.
    block[4] = cocg_tail16 | (rem[0] << 16);
    block[5] = __funnelshift_l(rem[0], rem[1], 16);
    block[6] = __funnelshift_l(rem[1], rem[2], 16);
    block[7] = __funnelshift_l(rem[2], rem[3], 16);
}

// Decode 32-byte packed SH coefficients to fp16 SoA.
// block:      input [8] uint32 — caller loads.
// out_coeffs: output [3][8] half2 — [R/G/B][pair0..7], 16 bases as 8 half2 pairs.
//             Pair 0 = {k=0, k=1}, pair 1 = {k=2, k=3}, ..., pair 7 = {k=14, k=15}.
// scales:     per-basis decoder scales (half2[8] per channel, pair 0 low = 1.0 for DC).
//
// Two-phase design:
//   Phase 1: extract all U1-biased quantized values → __uint8x4_to_half4 → fp16 arrays
//   Phase 2: uniform half2: FMA(val, inv_norm, offset) → x^2 → scale → YCoCg→RGB
// Fixed inv_gamma = 2 (gamma = 1/2): x^2 with sign copy, pure half2 SIMD, no transcendentals.
__device__ inline void sh_decode_32b(const uint32_t (&block)[8], __half2 (&out_coeffs)[3][8],
                                     const __half2 (&scales)[3][8])
{
    // ===== Phase 1: Extract all values into fp16 arrays =====

    __half Y[16], Co[16], Cg[16];

    // --- DC (k=0): Y fp16 (16) + Co fp15 (15) + Cg fp15 (15) = 46 bits ---
    // Y: full fp16 from block[0] bits 0..15
    AssignAs<uint16_t>(Y[0], block[0]);

    // Co fp15: block[0] bits 17..31
    AssignAs<uint16_t>(Co[0], (block[0] >> 15) & 0xFFFEu);

    // Cg fp15: bit 15 from block[0] bit 31, bits 1..14 from remaining region
    AssignAs<uint16_t>(Cg[0], ((block[0] >> 16) & 0x8000u) | ((block[4] >> 15) & 0x7FFEu));

    // --- k=1: extract from remaining region ---
    const uint32_t k1_raw = __funnelshift_rc(block[4], block[5], 30);
    const __half2 k1_cocg = __uint8x2_to_half2(((k1_raw >> 2) & 0xF0u) | ((k1_raw << 2) & 0xF000u));

    Y[1]  = __uint2half_rn((k1_raw << 2) & 0xFFu);
    Co[1] = __low2half(k1_cocg);
    Cg[1] = __high2half(k1_cocg);

    // --- CoCg k=2..15: 3 clean uint32 reads + 1 partial ---
    // Bytes are (qCo+8)<<4 | (qCg+8). Upper nibble = Co uint4, lower = Cg uint4.
    // After masking: upper nibble already in uint8 position for __uint8x4_to_half4.
#pragma unroll
    for (int i = 0; i < 3; i++)
    {
        const uint32_t co_u8 = block[1 + i] & 0xF0F0F0F0u;        // Co nibbles in upper nibble of each byte
        const uint32_t cg_u8 = (block[1 + i] & 0x0F0F0F0Fu) << 4; // Cg nibbles shifted to upper nibble
        __uint8x4_to_half4(&Co[i * 4 + 2], co_u8);
        __uint8x4_to_half4(&Cg[i * 4 + 2], cg_u8);
    }
    // partial pair decode for k=14, k=15
    AssignAs<__half2>(Co[14], __uint8x2_to_half2(block[4] & 0xF0F0u));
    AssignAs<__half2>(Cg[14], __uint8x2_to_half2((block[4] & 0x0F0Fu) << 4));

    // --- Y k=2..15: funnel shift extraction (14×6-bit unsigned bias) ---
    // Y data at rem bit 28 → absolute block[5] bit 12.
    auto spread_6bit_to_uint8x4 = [](uint32_t packed_6x4) -> uint32_t {
        const uint32_t lo = __select(0x003FC0u, packed_6x4 << 2, packed_6x4);
        const uint32_t hi = __select(0xFF0000u, packed_6x4, packed_6x4 >> 2);
        return __prmt_idx(lo << 2, hi, 0x6510) & 0xFCFCFCFCu;
    };

    // k=2..13: 3 full batches of 4 at half2-aligned offsets
#pragma unroll
    for (int batch = 0; batch < 3; batch++)
    {
        const int bit_offset = 12 + batch * 24;
        const int word_idx   = 5 + bit_offset / 32;
        const int shift      = bit_offset % 32;

        __uint8x4_to_half4(&Y[2 + batch * 4],
                           spread_6bit_to_uint8x4(__funnelshift_rc(block[word_idx], block[word_idx + 1], shift)));
    }

    // k=14..15: pair write to avoid out-of-bounds Y[16]
    AssignAs<__half2>(Y[14], __uint8x2_to_half2(((block[7] >> 18) & 0xFCu) | ((block[7] >> 16) & 0xFC00u)));

    // ===== Phase 2: Uniform half2 arithmetic =====
    //
    // U1 decode: val * inv_norm gives (q + max_q + 1) / max_q.
    // FMA with offset (-(max_q+1)/max_q) gives q / max_q.
    // Then x^2 with sign copy (fixed inv_gamma=2), then per-basis scale.

    constexpr float INV_NORM_Y    = 1.0f / (4.0f * 31.0f);
    constexpr float INV_NORM_COCG = 1.0f / (16.0f * 7.0f);

    // Y:    uint8 = (q+32)<<2 → fp16 = (q+32)*4 → *(1/(4*31)) - 32/31 = q/31
    // CoCg: uint8 = (q+8)<<4  → fp16 = (q+8)*16 → *(1/(16*7)) - 8/7  = q/7

    // x^2 with sign copy in half2: pure SIMD, no float conversion, no transcendentals.
    auto square_signed_h2 = [](__half2 v) -> __half2 { return __h2_copy_sign(__hmul2(v, v), v); };
    auto square_signed_h  = [](__half v) -> __half { return __h_copy_sign(__hmul(v, v), v); };

    __half2 *Y_h2  = reinterpret_cast<__half2 *>(Y);
    __half2 *Co_h2 = reinterpret_cast<__half2 *>(Co);
    __half2 *Cg_h2 = reinterpret_cast<__half2 *>(Cg);

    // Normalize + x^2 + scale for HO (pairs 1..7 fully, pair 0 high lane = k=1)
    // Pair 0 low lane is DC YCoCg — not int-N quantized, skip normalization.
    // Handle k=1 separately, then pairs 1..7 uniformly.
    const __half2 inv_norm_y    = __float2half2_rn(INV_NORM_Y);
    const __half2 inv_norm_cocg = __float2half2_rn(INV_NORM_COCG);
    // FMA offsets: -(max_q+1)/max_q to account for the asymmetric bias (stored = q + max_q + 1).
    constexpr float OFFSET_Y    = -32.0f / 31.0f;
    constexpr float OFFSET_COCG = -8.0f / 7.0f;
    const __half2 offset_y      = __float2half2_rn(OFFSET_Y);
    const __half2 offset_cocg   = __float2half2_rn(OFFSET_COCG);

    // Normalize + inv-gamma for k=1 (scalar, high lane of pair 0)
    Y[1]          = square_signed_h(__hfma(Y[1], __float2half_rn(INV_NORM_Y), __float2half_rn(OFFSET_Y)));
    __half2 cocg1 = __highs2half2(Co_h2[0], Cg_h2[0]);
    cocg1         = square_signed_h2(__hfma2(cocg1, inv_norm_cocg, offset_cocg));
    Co[1]         = __low2half(cocg1);
    Cg[1]         = __high2half(cocg1);

    // Normalize + inv-gamma for pairs 1..7
#pragma unroll
    for (int p = 1; p < 8; p++)
    {
        Y_h2[p]  = square_signed_h2(__hfma2(Y_h2[p], inv_norm_y, offset_y));
        Co_h2[p] = square_signed_h2(__hfma2(Co_h2[p], inv_norm_cocg, offset_cocg));
        Cg_h2[p] = square_signed_h2(__hfma2(Cg_h2[p], inv_norm_cocg, offset_cocg));
    }

#pragma unroll
    for (int p = 0; p < 8; p++)
    {
        // Apply per-basis scales (pair 0 DC lane = 1.0, so DC is unmodified)
        Y_h2[p]  = __hmul2(Y_h2[p], scales[0][p]);
        Co_h2[p] = __hmul2(Co_h2[p], scales[1][p]);
        Cg_h2[p] = __hmul2(Cg_h2[p], scales[2][p]);

        // YCoCg → RGB for all 8 pairs (DC included — DC is in YCoCg color space)
        const __half2 Y_sub_Cg = __hsub2(Y_h2[p], Cg_h2[p]);
        out_coeffs[0][p]       = __hadd2(Y_sub_Cg, Co_h2[p]);
        out_coeffs[1][p]       = __hadd2(Y_h2[p], Cg_h2[p]);
        out_coeffs[2][p]       = __hsub2(Y_sub_Cg, Co_h2[p]);
    }

    // DC fixup (pair 0 low lane only): YCoCg→RGB produced the color, undo color transform.
    // coeff = color * (1/Y_00) + (-SH_BIAS/Y_00).  k=1 (high lane): color * 1.0 + 0.0.
    const __half2 dc_mul = __halves2half2(__float2half(1.0f / SH_Y00), __float2half(1.0f));
    const __half2 dc_add = __halves2half2(__float2half(-SH_BIAS / SH_Y00), __float2half(0.0f));
    out_coeffs[0][0]     = __hfma2(out_coeffs[0][0], dc_mul, dc_add);
    out_coeffs[1][0]     = __hfma2(out_coeffs[1][0], dc_mul, dc_add);
    out_coeffs[2][0]     = __hfma2(out_coeffs[2][0], dc_mul, dc_add);
}

// 16-byte SH coefficient compression for K=16 (degree 3).
//
// Packed layout (128 bits = 4 × uint32):
//
//   block[0] bits  0..15:  DC_Y fp16 (16 bits, lossless)
//   block[0] bits 16..26:  DC_Co fp11 (fp16 >> 5, drop 5 mantissa LSBs)
//   block[0] bits 27..31:  DC_Cg fp11 high 5 bits
//
//   block[1] bits  0..5:   DC_Cg fp11 low 6 bits  (total Cg = 5 + 6 = 11)
//   block[1] bits  6..31:  Y1..Y4 packed 6-bit ─┐
//   block[2] bits  0..31:  Y5..Y9 + Y10 lo      │ 15 × 6-bit Y values for k=1..15
//   block[3] bits  0..31:  Y10 hi..Y15          ─┘
//
//   HO CoCg entirely dropped (zero on decode).
//
//   Bit budget: 16 + 11 + 11 + 15*6 = 128 bits ✓

// Encode K=16 SH coefficients into 16-byte packed representation.
// T = float or __half.
// coeffs:      input [3][16] — channel-first (R/G/B × 16 bases), caller loads.
// block:       output [4] uint32 — caller stores.
// inv_scales:  per-basis inverse scales (only y[] used for HO; co[]/cg[] ignored).
template<typename T>
__device__ void sh_encode_16b(const T (&coeffs)[3][16], uint32_t (&block)[4], const float (&inv_scales)[3][16])
{
    auto rgb_to_ycocg = [](float R, float G, float B, float &Y, float &Co, float &Cg) {
        Y  = 0.25f * R + 0.5f * G + 0.25f * B;
        Co = 0.5f * R - 0.5f * B;
        Cg = -0.25f * R + 0.5f * G - 0.25f * B;
    };

    // HO only needs luma — no CoCg in 16B format
    auto rgb_to_y = [](float R, float G, float B) -> float { return 0.25f * R + 0.5f * G + 0.25f * B; };

    auto quantize_u1f = [](float val, float inv_scale, float max_q) -> float {
        float norm       = val * inv_scale;
        float compressed = copysignf(sqrtf(fabsf(norm)), norm);
        float q          = fminf(fmaxf(roundf(compressed * max_q), -(max_q + 1.0f)), max_q);
        return q + (max_q + 1.0f);
    };

    // --- DC (k=0) : Y@fp16 + Co@fp11 + Cg@fp11 ---
    float R0 = _sh_to_float(coeffs[0][0]) * SH_Y00 + SH_BIAS;
    float G0 = _sh_to_float(coeffs[1][0]) * SH_Y00 + SH_BIAS;
    float B0 = _sh_to_float(coeffs[2][0]) * SH_Y00 + SH_BIAS;

    float dc_Y, dc_Co, dc_Cg;
    rgb_to_ycocg(R0, G0, B0, dc_Y, dc_Co, dc_Cg);

    uint16_t y_bits;
    AssignAs<uint16_t>(y_bits, __float2half(dc_Y));

    // fp11 = fp16 rounded to 5 mantissa bits (drop 5 LSBs with rounding), then >> 5
    const __half2 tr_cocg = RoundToNearest<5, false>(__floats2half2_rn(dc_Co, dc_Cg));
    const uint32_t co11   = (reinterpret_cast<const uint32_t &>(tr_cocg) >> 5) & 0x7FFu;
    const uint32_t cg11   = (reinterpret_cast<const uint32_t &>(tr_cocg) >> 21) & 0x7FFu;

    block[0] = static_cast<uint32_t>(y_bits) | (co11 << 16) | (cg11 << 27);

    // --- HO Y k=1..15: 4 groups of 4 (last group zero-pads k=16) ---
    // Each group packs 4 Y values via __cvt_pack_sat_u8_f32, then compresses
    // the u8x4 into a 24-bit chunk by dropping the top 2 bits of each byte.
    uint32_t y_packed[4];
#pragma unroll
    for (int g = 0; g < 4; g++)
    {
        float Yg[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            const int k = 1 + g * 4 + j;
            if (k <= 15)
            {
                Yg[j] = rgb_to_y(_sh_to_float(coeffs[0][k]), _sh_to_float(coeffs[1][k]),
                                 _sh_to_float(coeffs[2][k]));
            }
        }

        const int k_base    = 1 + g * 4;
        const uint32_t Y_u8 = __cvt_pack_sat_u8_f32(
            quantize_u1f(Yg[0], inv_scales[0][k_base], 31.f), quantize_u1f(Yg[1], inv_scales[0][k_base + 1], 31.f),
            quantize_u1f(Yg[2], inv_scales[0][k_base + 2], 31.f), quantize_u1f(Yg[3], inv_scales[0][k_base + 3], 31.f));
        y_packed[g] =
            (Y_u8 & 0x3Fu) | ((Y_u8 & 0x3F00u) >> 2) | ((Y_u8 & 0x3F0000u) >> 4) | ((Y_u8 & 0x3F000000u) >> 6);
    }

    // Stitch: 6-bit cg11_hi header + four 24-bit Y chunks → block[1..3].
    // Simple shifts+OR (not funnel shifts) because y_packed[] have zero high-8 bits.
    const uint32_t cg11_hi = (cg11 >> 5) & 0x3Fu;
    block[1]               = cg11_hi | (y_packed[0] << 6) | (y_packed[1] << 30);
    block[2]               = (y_packed[1] >> 2) | (y_packed[2] << 22);
    block[3]               = (y_packed[2] >> 10) | (y_packed[3] << 14);
}

// Decode 16-byte packed SH coefficients to fp16 SoA.
// block:      input [4] uint32 — caller loads.
// out_coeffs: output [3][8] half2 — same format as sh_decode_32b.
//             HO CoCg is zero (not encoded in 16B format).
// scales:     per-basis decoder scales (only y[] used for HO).
__device__ inline void sh_decode_16b(const uint32_t (&block)[4], __half2 (&out_coeffs)[3][8],
                                     const __half2 (&scales)[3][8])
{
    // ===== Phase 1: Extract Y[0..15] and DC CoCg =====

    __half Y[16];

    // --- DC (k=0): Y fp16 (16) + Co fp11 (11) + Cg fp11 (11) = 38 bits ---
    AssignAs<uint16_t>(Y[0], block[0]);

    __half dc_Co, dc_Cg;
    AssignAs<uint16_t>(dc_Co, ((block[0] >> 16) & 0x7FFu) << 5);
    const uint32_t cg11 = (block[0] >> 27) | ((block[1] & 0x3Fu) << 5);
    AssignAs<uint16_t>(dc_Cg, (cg11 & 0x7FFu) << 5);

    // --- HO Y k=1..15: 15 × 6-bit unsigned bias, starting at block[1] bit 6 ---
    // Reconstruct the 90-bit Y stream aligned to bit 0.
    const uint32_t s0 = (block[1] >> 6) | (block[2] << 26);
    const uint32_t s1 = (block[2] >> 6) | (block[3] << 26);
    const uint32_t s2 = block[3] >> 6;

    // k=1: scalar extract (keeps half2 pair alignment for subsequent writes)
    Y[1] = __uint2half_rn((s0 & 0x3Fu) << 2);

    // k=2..13: 3 full batches of 4 at half2-aligned offsets
    auto spread_6bit_to_uint8x4 = [](uint32_t packed_6x4) -> uint32_t {
        const uint32_t lo = __select(0x003FC0u, packed_6x4 << 2, packed_6x4);
        const uint32_t hi = __select(0xFF0000u, packed_6x4, packed_6x4 >> 2);
        return __prmt_idx(lo << 2, hi, 0x6510) & 0xFCFCFCFCu;
    };

    __uint8x4_to_half4(&Y[2], spread_6bit_to_uint8x4(__funnelshift_rc(s0, s1, 6)));
    __uint8x4_to_half4(&Y[6], spread_6bit_to_uint8x4(__funnelshift_rc(s0, s1, 30)));
    __uint8x4_to_half4(&Y[10], spread_6bit_to_uint8x4(__funnelshift_rc(s1, s2, 22)));

    // k=14..15: pair write via __uint8x2_to_half2 (avoids out-of-bounds Y[16])
    AssignAs<__half2>(Y[14], __uint8x2_to_half2(((s2 >> 12) & 0xFCu) | (((s2 >> 18) & 0xFCu) << 8)));

    // ===== Phase 2: Normalize, inv-gamma, scale, YCoCg → RGB =====

    constexpr float INV_NORM_Y = 1.0f / (4.0f * 31.0f);
    constexpr float OFFSET_Y   = -32.0f / 31.0f;

    auto square_signed_h2 = [](__half2 v) -> __half2 { return __h2_copy_sign(__hmul2(v, v), v); };
    auto square_signed_h  = [](__half v) -> __half { return __h_copy_sign(__hmul(v, v), v); };

    __half2 *Y_h2 = reinterpret_cast<__half2 *>(Y);

    // Normalize + inv-gamma for k=1 (scalar, high lane of pair 0)
    Y[1] = square_signed_h(__hfma(Y[1], __float2half_rn(INV_NORM_Y), __float2half_rn(OFFSET_Y)));

    // Scale k=1 by its per-basis Y scale (high lane of scales.y[0])
    const __half k1_scaled = __hmul(Y[1], __high2half(scales[0][0]));

    // DC (pair 0 low lane): full YCoCg → RGB, then undo SH color transform
    const __half dc_Y_sub_Cg = __hsub(Y[0], dc_Cg);
    const __half dc_inv_y00  = __float2half(1.0f / SH_Y00);
    const __half dc_bias     = __float2half(-SH_BIAS / SH_Y00);
    const __half dc_R        = __hfma(__hadd(dc_Y_sub_Cg, dc_Co), dc_inv_y00, dc_bias);
    const __half dc_G        = __hfma(__hadd(Y[0], dc_Cg), dc_inv_y00, dc_bias);
    const __half dc_B        = __hfma(__hsub(dc_Y_sub_Cg, dc_Co), dc_inv_y00, dc_bias);

    out_coeffs[0][0] = __halves2half2(dc_R, k1_scaled);
    out_coeffs[1][0] = __halves2half2(dc_G, k1_scaled);
    out_coeffs[2][0] = __halves2half2(dc_B, k1_scaled);

    // HO pairs 1..7: CoCg = 0 → R = G = B = Y * scale
    const __half2 inv_norm_y = __float2half2_rn(INV_NORM_Y);
    const __half2 offset_y   = __float2half2_rn(OFFSET_Y);

#pragma unroll
    for (int p = 1; p < 8; p++)
    {
        const __half2 y_scaled = __hmul2(square_signed_h2(__hfma2(Y_h2[p], inv_norm_y, offset_y)), scales[0][p]);
        out_coeffs[0][p]       = y_scaled;
        out_coeffs[1][p]       = y_scaled;
        out_coeffs[2][p]       = y_scaled;
    }
}

// Compute normalized view direction from camera position to gaussian center.
// means is in [3, N] planar layout.
__device__ __forceinline__ float3 GetViewDir(const float *__restrict__ means, uint32_t N, uint32_t idx, float cam_x,
                                             float cam_y, float cam_z)
{
    const float dx    = means[idx] - cam_x;
    const float dy    = means[N + idx] - cam_y;
    const float dz    = means[2 * N + idx] - cam_z;
    const float inorm = rsqrtf(dx * dx + dy * dy + dz * dz);
    return make_float3(dx * inorm, dy * inorm, dz * inorm);
}

// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation

__device__ __forceinline__ void EvaluteSHCoeffs(const uint32_t degree, // degree of SH to be evaluated
                                                const uint32_t c,      // color channel
                                                const float3 &dir,     // pre-normalized direction
                                                const float *coeffs,   // [K, 3]
                                                float bias, float min_value, float &color_out)
{
    float result = 0.2820947917738781f * coeffs[c];
    if (degree >= 1)
    {
        const float x = dir.x;
        const float y = dir.y;
        const float z = dir.z;

        result += 0.48860251190292f * (-y * coeffs[1 * 3 + c] + z * coeffs[2 * 3 + c] - x * coeffs[3 * 3 + c]);
        if (degree >= 2)
        {
            const float z2 = z * z;

            const float fTmp0B = -1.092548430592079f * z;
            const float fC1    = x * x - y * y;
            const float fS1    = 2.f * x * y;
            const float pSH6   = (0.9461746957575601f * z2 - 0.3153915652525201f);
            const float pSH7   = fTmp0B * x;
            const float pSH5   = fTmp0B * y;
            const float pSH8   = 0.5462742152960395f * fC1;
            const float pSH4   = 0.5462742152960395f * fS1;

            result += pSH4 * coeffs[4 * 3 + c] + pSH5 * coeffs[5 * 3 + c] + pSH6 * coeffs[6 * 3 + c] +
                      pSH7 * coeffs[7 * 3 + c] + pSH8 * coeffs[8 * 3 + c];
            if (degree >= 3)
            {
                const float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                const float fTmp1B = 1.445305721320277f * z;
                const float fC2    = x * fC1 - y * fS1;
                const float fS2    = x * fS1 + y * fC1;
                const float pSH12  = z * (1.865881662950577f * z2 - 1.119528997770346f);
                const float pSH13  = fTmp0C * x;
                const float pSH11  = fTmp0C * y;
                const float pSH14  = fTmp1B * fC1;
                const float pSH10  = fTmp1B * fS1;
                const float pSH15  = -0.5900435899266435f * fC2;
                const float pSH9   = -0.5900435899266435f * fS2;

                result += pSH9 * coeffs[9 * 3 + c] + pSH10 * coeffs[10 * 3 + c] + pSH11 * coeffs[11 * 3 + c] +
                          pSH12 * coeffs[12 * 3 + c] + pSH13 * coeffs[13 * 3 + c] + pSH14 * coeffs[14 * 3 + c] +
                          pSH15 * coeffs[15 * 3 + c];

                if (degree >= 4)
                {
                    const float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
                    const float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                    const float fTmp2B = -1.770130769779931f * z;
                    const float fC3    = x * fC2 - y * fS2;
                    const float fS3    = x * fS2 + y * fC2;
                    const float pSH20  = (1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6);
                    const float pSH21  = fTmp0D * x;
                    const float pSH19  = fTmp0D * y;
                    const float pSH22  = fTmp1C * fC1;
                    const float pSH18  = fTmp1C * fS1;
                    const float pSH23  = fTmp2B * fC2;
                    const float pSH17  = fTmp2B * fS2;
                    const float pSH24  = 0.6258357354491763f * fC3;
                    const float pSH16  = 0.6258357354491763f * fS3;

                    result += pSH16 * coeffs[16 * 3 + c] + pSH17 * coeffs[17 * 3 + c] + pSH18 * coeffs[18 * 3 + c] +
                              pSH19 * coeffs[19 * 3 + c] + pSH20 * coeffs[20 * 3 + c] + pSH21 * coeffs[21 * 3 + c] +
                              pSH22 * coeffs[22 * 3 + c] + pSH23 * coeffs[23 * 3 + c] + pSH24 * coeffs[24 * 3 + c];
                }
            }
        }
    }

    color_out = max(result + bias, min_value);
}

// Vectorized coefficient load: reads N_VECS × VEC from src, converts each element to float.
template<typename T, typename VEC, int N_VECS>
__device__ __forceinline__ void LoadSHCoeffs(const T *src, float *dst)
{
    constexpr int ELEMS_PER_VEC = sizeof(VEC) / sizeof(T);
#pragma unroll
    for (int i = 0; i < N_VECS; ++i)
    {
        T tmp[ELEMS_PER_VEC];
        reinterpret_cast<VEC *>(tmp)[0] = reinterpret_cast<const VEC *>(src)[i];
#pragma unroll
        for (int j = 0; j < ELEMS_PER_VEC; ++j)
        {
            dst[i * ELEMS_PER_VEC + j] = _sh_to_float(tmp[j]);
        }
    }
}

// Evaluate SH for 3 color channels at the given degree and write packed half4 output pixel.
// DEGREE is a template parameter so sh_coeffs_to_color_fast receives a compile-time constant.
template<int DEGREE>
__device__ __forceinline__ void EvalSHAndPack(const float3 &dir_n, const float *sh_coeffs, float bias, float min_value,
                                              __half *pixel)
{
    float out_color[3];
    EvaluteSHCoeffs(DEGREE, 0, dir_n, sh_coeffs, bias, min_value, out_color[0]);
    EvaluteSHCoeffs(DEGREE, 1, dir_n, sh_coeffs, bias, min_value, out_color[1]);
    EvaluteSHCoeffs(DEGREE, 2, dir_n, sh_coeffs, bias, min_value, out_color[2]);
    __half2 packed_out[2];
    packed_out[0] = __floats2half2_rn(out_color[0], out_color[1]);
    packed_out[1] = __floats2half2_rn(out_color[2], 0.f);
    AssignAs<uint2>(pixel[0], packed_out);
}

// Load SH coefficients for gaussian idx, evaluate all 3 color channels, write packed half4
// output pixel. MODE selects the coefficient source at compile time. dir_n must be the
// pre-computed normalized view direction for this gaussian.
template<SHInputMode MODE>
__device__ __forceinline__ void EvalSHForGaussian(const float3 &dir_n, int idx, int N, int degrees_to_use,
                                                  const void *__restrict__ input_data, float bias, float min_value,
                                                  const SHDecodeParams &decode_params, __half *__restrict__ colors)
{
    float sh_coeffs[16 * 3];
    __half *pixel = colors + idx * 4;

    if constexpr (MODE == SHInputMode::RAW_FLOAT || MODE == SHInputMode::RAW_HALF)
    {
        using T          = std::conditional_t<MODE == SHInputMode::RAW_FLOAT, float, __half>;
        using VEC_SMALL  = std::conditional_t<MODE == SHInputMode::RAW_FLOAT, uint4, ushort4>;
        constexpr int N2 = MODE == SHInputMode::RAW_FLOAT ? 7 : 4;  // uint4 loads for degree 2
        constexpr int N3 = MODE == SHInputMode::RAW_FLOAT ? 12 : 6; // uint4 loads for degree 3

        const T *elem_coeffs = static_cast<const T *>(input_data) + static_cast<int64_t>(idx) * (16 * 3);

        switch (degrees_to_use)
        {
        case 0:
            LoadSHCoeffs<T, VEC_SMALL, 1>(elem_coeffs, sh_coeffs);
            EvalSHAndPack<0>(dir_n, sh_coeffs, bias, min_value, pixel);
            return;
        case 1:
            LoadSHCoeffs<T, VEC_SMALL, 3>(elem_coeffs, sh_coeffs);
            EvalSHAndPack<1>(dir_n, sh_coeffs, bias, min_value, pixel);
            return;
        case 2:
            LoadSHCoeffs<T, uint4, N2>(elem_coeffs, sh_coeffs);
            EvalSHAndPack<2>(dir_n, sh_coeffs, bias, min_value, pixel);
            return;
        default:
            LoadSHCoeffs<T, uint4, N3>(elem_coeffs, sh_coeffs);
            EvalSHAndPack<3>(dir_n, sh_coeffs, bias, min_value, pixel);
            return;
        }
    }

    if constexpr (MODE == SHInputMode::COMPRESS_32B || MODE == SHInputMode::COMPRESS_16B)
    {
        const auto *packed = static_cast<const uint4 *>(input_data);
        __half2 out_coeffs[3][8];

        if constexpr (MODE == SHInputMode::COMPRESS_32B)
        {
            // load 2 × uint4 at stride N
            uint32_t block[8];
            uint4 b0 = packed[idx];
            uint4 b1 = packed[static_cast<int64_t>(N) + idx];
            AssignAs<uint4>(block[0], b0);
            AssignAs<uint4>(block[4], b1);
            sh_decode_32b(block, out_coeffs, decode_params.scales);
        }
        else
        {
            // load 1 × uint4 per gaussian
            uint32_t block[4];
            uint4 b = packed[idx];
            AssignAs<uint4>(block[0], b);
            sh_decode_16b(block, out_coeffs, decode_params.scales);
        }

        // flatten to [K, 3] float layout expected by sh_coeffs_to_color_fast
#pragma unroll
        for (int c = 0; c < 3; c++)
        {
#pragma unroll
            for (int p = 0; p < 8; p++)
            {
                float2 pair                    = __half22float2(out_coeffs[c][p]);
                sh_coeffs[(p * 2 + 0) * 3 + c] = pair.x;
                sh_coeffs[(p * 2 + 1) * 3 + c] = pair.y;
            }
        }

        EvalSHAndPack<3>(dir_n, sh_coeffs, bias, min_value, pixel);
    }
}

} // namespace higs
