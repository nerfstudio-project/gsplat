/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <cuda_fp16.h>

#include "Common.h"  // GSPLAT_HOST_DEVICE

namespace gsplat::priming {

////////////////////////////////////////////////////////////////
// priming_state[pix] uint32 packed encoding.
//
// Consumer invariant: decoded T is the T_init for decoded batch K.
// Writers at batch b publish `(T_after_b, b + 1)` because their post-walk
// transmittance becomes the start transmittance for the next batch.
//
// Bit layout, from most to least significant:
//   bit 31      : !sat, with inverted polarity.
//                 1 = not saturated, 0 = saturated.
//                 Saturated entries sort lower under unsigned atomicMin, so
//                 one saturation flag cannot be overwritten by a later
//                 non-saturated writer.
//   bits 16..30 : fp16 T magnitude without the sign bit.
//                 T is always non-negative, so the fp16 sign bit is available
//                 for the !sat flag above.
//   bits 0..15  : uint16_t(-K).
//                 For equal T, unsigned atomicMin prefers the smaller
//                 negated key, which is the deeper writer.
//
// Initial value: T_init for batch 0 is 1.0 and the chain is not saturated.
////////////////////////////////////////////////////////////////

namespace detail {

// Low 16 bits carry uint16_t(-K). The adjacent 15 bits carry the non-negative
// fp16 transmittance magnitude; the fp16 sign bit is repurposed as !sat.
constexpr uint32_t K_BITS = 16u;
constexpr uint32_t T_BITS = 15u;
constexpr uint32_t T_SHIFT = K_BITS;
constexpr uint32_t SAT_SHIFT = T_SHIFT + T_BITS;

// K_MASK bounds the wrapped negated batch key. K_VALUE_COUNT is the modulus
// used to recover positive K from that uint16_t representation.
constexpr uint32_t K_VALUE_COUNT = uint32_t{1} << K_BITS;
constexpr uint32_t K_MASK = K_VALUE_COUNT - 1u;

// T_MAGNITUDE_MASK intentionally strips the fp16 sign bit. T_MASK names the
// same field after shifting it into the packed priming word.
constexpr uint32_t T_MAGNITUDE_MASK = (uint32_t{1} << T_BITS) - 1u;
constexpr uint32_t T_MASK = T_MAGNITUDE_MASK << T_SHIFT;

// Inverted saturation bit: non-saturated states keep this bit set so
// saturated states sort lower under unsigned atomicMin.
constexpr uint32_t SAT_BIT_INV = uint32_t{1} << SAT_SHIFT;

// Host code enforces max_batches_per_tile < K_LIMIT before launching the
// ParallelBatch forward kernels. K == 0 is reserved for the initial value.
constexpr int32_t K_LIMIT = static_cast<int32_t>(K_MASK);

// Exact fp16 power-of-two encodings without embedding raw half-precision bit
// patterns in tests. These constants assume positive, normalized fp16 values.
constexpr uint16_t FP16_EXPONENT_SHIFT = 10u;
constexpr int32_t FP16_EXPONENT_BIAS = 15;

GSPLAT_HOST_DEVICE constexpr uint16_t fp16_power_of_two_bits(int32_t exponent)
{
    return static_cast<uint16_t>(
        static_cast<uint16_t>(exponent + FP16_EXPONENT_BIAS)
        << FP16_EXPONENT_SHIFT);
}

}  // namespace detail

GSPLAT_HOST_DEVICE constexpr uint32_t encode_from_fp16_bits(
    uint16_t T_fp16_bits,
    int32_t K,
    bool saturated
) {
    // Host tests need deterministic forced-order packed states without
    // reimplementing this layout. The runtime float encoder below uses this
    // same helper after choosing the rounded fp16 magnitude.
    const uint32_t T_bits =
        static_cast<uint32_t>(T_fp16_bits) & detail::T_MAGNITUDE_MASK;
    const uint32_t neg_K =
        static_cast<uint32_t>(static_cast<uint16_t>(-K));
    const uint32_t sat_inv = saturated ? 0u : detail::SAT_BIT_INV;
    return sat_inv | (T_bits << detail::T_SHIFT) | neg_K;
}

constexpr uint32_t INIT_PACKED =
    encode_from_fp16_bits(detail::fp16_power_of_two_bits(0), 0, false);

static_assert((INIT_PACKED & detail::SAT_BIT_INV) != 0u,
              "init must have !sat=1");
static_assert((INIT_PACKED & detail::K_MASK) == 0u,
              "init K must be 0");
static_assert(((INIT_PACKED & detail::T_MASK) >> detail::T_SHIFT) ==
                  detail::fp16_power_of_two_bits(0),
              "init T_fp16 magnitude must be 1.0");

// Partials can reject an in-bounds pixel when the camera/lidar model cannot
// produce a valid world ray. Reserve the absolute minimum packed value as that
// cross-kernel invalid-ray sentinel:
//   - saturated=true sorts below any non-saturated state under atomicMin;
//   - T=0 sorts below any positive transmittance;
//   - K=0 is otherwise reserved for INIT_PACKED.
//
// Valid partial writers publish K=b+1, so a valid saturated ray with T=0 still
// cannot collide with this full packed value.
constexpr uint32_t INVALID_RAY_PACKED =
    encode_from_fp16_bits(0u, 0, true);

static_assert(INVALID_RAY_PACKED == 0u,
              "invalid ray sentinel must remain the atomicMin floor");

#ifdef __CUDACC__

struct DecodeForBatchResult {
    float T_init_used;
    int32_t stored_K;
    bool stored_sat;
    bool chain_saturated;
    bool use_stored_state;
};

__device__ __forceinline__ uint32_t encode(
    float T,
    int32_t K,
    bool saturated
) {
    assert(T >= 0.0f && T <= 1.0f);
    assert(K >= 0 && K < detail::K_LIMIT);

    // Decoded T is used as an upper bound when tightening
    // TRANSMITTANCE_THRESHOLD / T. Rounding down would make that threshold
    // too large, so encode with upward rounding and strip the fp16 sign bit.
    const uint32_t T_bits =
        static_cast<uint32_t>(__half_as_ushort(__float2half_ru(T))) &
        detail::T_MAGNITUDE_MASK;
    return encode_from_fp16_bits(
        static_cast<uint16_t>(T_bits), K, saturated);
}

__device__ __forceinline__ void decode(
    uint32_t packed,
    float &T,
    int32_t &K,
    bool &saturated
) {
    saturated = (packed & detail::SAT_BIT_INV) == 0u;

    const uint16_t T_bits =
        static_cast<uint16_t>((packed & detail::T_MASK) >> detail::T_SHIFT);
    T = __half2float(__ushort_as_half(T_bits));

    const uint32_t neg_K = packed & detail::K_MASK;
    K = (neg_K == 0u)
        ? 0
        : static_cast<int32_t>(detail::K_VALUE_COUNT - neg_K);

    assert(T >= 0.0f && T <= 1.0f);
    assert(K >= 0 && K < detail::K_LIMIT);
}

// Decode the state exactly as a partials CTA for `batch_id` consumes it.
//
// The stored_K > batch_id case represents a downstream writer reaching the
// atomic state first. That value is not a valid seed for this batch, so the
// reader falls back to the unprimed state.
__device__ __forceinline__ DecodeForBatchResult
decode_for_batch(uint32_t packed, int32_t batch_id) {
    float stored_T;
    int32_t stored_K;
    bool stored_sat;
    decode(packed, stored_T, stored_K, stored_sat);

    DecodeForBatchResult result{
        1.0f,
        stored_K,
        stored_sat,
        false,
        false,
    };
    if (stored_K <= batch_id) {
        result.T_init_used = stored_T;
        result.chain_saturated = stored_sat;
        result.use_stored_state = true;
    }
    assert(result.T_init_used >= 0.0f && result.T_init_used <= 1.0f);
    return result;
}

__device__ __forceinline__ bool is_invalid_ray(uint32_t packed) {
    return packed == INVALID_RAY_PACKED;
}

__device__ __forceinline__ bool is_saturated(uint32_t packed) {
    return (packed & detail::SAT_BIT_INV) == 0u;
}

__device__ __forceinline__ int32_t get_K(uint32_t packed) {
    const uint32_t neg_K = packed & detail::K_MASK;
    const int32_t K =
        (neg_K == 0u)
            ? 0
            : static_cast<int32_t>(detail::K_VALUE_COUNT - neg_K);
    assert(K >= 0 && K < detail::K_LIMIT);
    return K;
}

__device__ __forceinline__ float get_T(uint32_t packed) {
    const uint16_t T_bits =
        static_cast<uint16_t>((packed & detail::T_MASK) >> detail::T_SHIFT);
    const float T = __half2float(__ushort_as_half(T_bits));
    assert(T >= 0.0f && T <= 1.0f);
    return T;
}

#endif  // __CUDACC__

}  // namespace gsplat::priming
