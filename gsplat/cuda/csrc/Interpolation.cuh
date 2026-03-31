/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GSPLAT_INTERPOLATION_CUH
#define GSPLAT_INTERPOLATION_CUH

#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// Tetrahedral Feature Interpolation
// ============================================================================
namespace gsplat {
namespace interp {

// ============================================================================
// float3 arithmetic used by barycentric backward
// ============================================================================

__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3 operator*(float s, float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

// ============================================================================
// Precomputed tetrahedron geometry (inradius=1)
// ============================================================================
//
// Tetrahedron vertices:
//   p0 = (sqrt6, -sqrt2, -1),  p1 = (-sqrt6, -sqrt2, -1)
//   p2 = (0, 2*sqrt2, -1),     p3 = (0, 0, 3)
//
// The constants below were derived from compute_tetrahedron_geometry()
// (retained below for documentation). All heights are 4 (regular tet).

// Face normals (inward-pointing, unit length)
constexpr float TET_N0_X =  0.8164965809277261f;  // sqrt(6)/3
constexpr float TET_N0_Y = -0.4714045207910317f;  // -sqrt(2)/3
constexpr float TET_N0_Z = -0.3333333333333333f;  // -1/3

constexpr float TET_N1_X = -0.8164965809277261f;  // -sqrt(6)/3
constexpr float TET_N1_Y = -0.4714045207910317f;  // -sqrt(2)/3
constexpr float TET_N1_Z = -0.3333333333333333f;  // -1/3

constexpr float TET_N2_X =  0.0f;
constexpr float TET_N2_Y =  0.9428090415820634f;  // 2*sqrt(2)/3
constexpr float TET_N2_Z = -0.3333333333333333f;  // -1/3

constexpr float TET_N3_X =  0.0f;
constexpr float TET_N3_Y =  0.0f;
constexpr float TET_N3_Z =  1.0f;

constexpr float TET_INV_H = 0.25f;  // 1/4 (all heights = 4)

// Reference points on each face (for distance computation):
//   face 0: p1 = (-sqrt6, -sqrt2, -1)
//   face 1,2,3: p0 = (sqrt6, -sqrt2, -1)
constexpr float TET_P0_X =  2.449489742783178f;   // sqrt(6)
constexpr float TET_P0_Y = -1.4142135623730951f;  // -sqrt(2)
constexpr float TET_P0_Z = -1.0f;

constexpr float TET_P1_X = -2.449489742783178f;   // -sqrt(6)
constexpr float TET_P1_Y = -1.4142135623730951f;  // -sqrt(2)
constexpr float TET_P1_Z = -1.0f;

// Documentation only: shows how the constexpr values above were derived.
//
//   void compute_tetrahedron_geometry(p0, p1, p2, p3, n0..n3, h0..h3) {
//       n0_raw = cross(p2-p1, p3-p1); n0_raw *= copysign(1, dot(n0_raw, p0-p1)); n0 = normalize(n0_raw);
//       n1_raw = cross(p3-p0, p2-p0); n1_raw *= copysign(1, dot(n1_raw, p1-p0)); n1 = normalize(n1_raw);
//       n2_raw = cross(p1-p0, p3-p0); n2_raw *= copysign(1, dot(n2_raw, p2-p0)); n2 = normalize(n2_raw);
//       n3_raw = cross(p2-p0, p1-p0); n3_raw *= copysign(1, dot(n3_raw, p3-p0)); n3 = normalize(n3_raw);
//       h0 = dot(n0, p0-p1); h1 = dot(n1, p1-p0); h2 = dot(n2, p2-p0); h3 = dot(n3, p3-p0);
//   }

// ============================================================================
// Barycentric weights using precomputed tetrahedron geometry
// ============================================================================

__device__ __forceinline__ void tet_barycentric_weights(
    float3 sample_pos,
    float& w0, float& w1, float& w2, float& w3
) {
    // d_i = dot(n_i, sample_pos - ref_point_on_face_i)
    // w_i = d_i / h_i  =  d_i * TET_INV_H
    const float dx0 = sample_pos.x - TET_P1_X;
    const float dy0 = sample_pos.y - TET_P1_Y;
    const float dz0 = sample_pos.z - TET_P1_Z;
    w0 = (TET_N0_X * dx0 + TET_N0_Y * dy0 + TET_N0_Z * dz0) * TET_INV_H;

    const float dx1 = sample_pos.x - TET_P0_X;
    const float dy1 = sample_pos.y - TET_P0_Y;
    const float dz1 = sample_pos.z - TET_P0_Z;
    w1 = (TET_N1_X * dx1 + TET_N1_Y * dy1 + TET_N1_Z * dz1) * TET_INV_H;
    w2 = (TET_N2_X * dx1 + TET_N2_Y * dy1 + TET_N2_Z * dz1) * TET_INV_H;
    w3 = (TET_N3_Z * dz1) * TET_INV_H;  // n3 = (0, 0, 1)
}

// ============================================================================
// Barycentric Interpolation (Forward) - precomputed geometry
// ============================================================================

template<int CHANNELS>
__device__ __forceinline__ void barycentric_interpolate_fwd(
    float3 sample_pos,
    const float* __restrict__ v0,
    const float* __restrict__ v1,
    const float* __restrict__ v2,
    const float* __restrict__ v3,
    float* __restrict__ result
) {
    float w0, w1, w2, w3;
    tet_barycentric_weights(sample_pos, w0, w1, w2, w3);

    #pragma unroll
    for (int i = 0; i < CHANNELS; ++i) {
        result[i] = w0 * v0[i] + w1 * v1[i] + w2 * v2[i] + w3 * v3[i];
    }
}

// ============================================================================
// Barycentric Interpolation (Backward) - precomputed geometry
// ============================================================================

template<int CHANNELS>
__device__ __forceinline__ void barycentric_interpolate_bwd(
    float3 sample_pos,
    const float* __restrict__ v0,
    const float* __restrict__ v1,
    const float* __restrict__ v2,
    const float* __restrict__ v3,
    const float* __restrict__ v_result,
    float3& v_sample_pos,
    float* __restrict__ v_v0,
    float* __restrict__ v_v1,
    float* __restrict__ v_v2,
    float* __restrict__ v_v3
) {
    float w0, w1, w2, w3;
    tet_barycentric_weights(sample_pos, w0, w1, w2, w3);

    #pragma unroll
    for (int i = 0; i < CHANNELS; ++i) {
        v_v0[i] = w0 * v_result[i];
        v_v1[i] = w1 * v_result[i];
        v_v2[i] = w2 * v_result[i];
        v_v3[i] = w3 * v_result[i];
    }

    float v_w0 = 0.0f, v_w1 = 0.0f, v_w2 = 0.0f, v_w3 = 0.0f;
    #pragma unroll
    for (int i = 0; i < CHANNELS; ++i) {
        v_w0 += v0[i] * v_result[i];
        v_w1 += v1[i] * v_result[i];
        v_w2 += v2[i] * v_result[i];
        v_w3 += v3[i] * v_result[i];
    }

    // v_sample_pos = sum_i(v_w_i * n_i * TET_INV_H)
    const float3 n0 = make_float3(TET_N0_X, TET_N0_Y, TET_N0_Z);
    const float3 n1 = make_float3(TET_N1_X, TET_N1_Y, TET_N1_Z);
    const float3 n2 = make_float3(TET_N2_X, TET_N2_Y, TET_N2_Z);
    const float3 n3 = make_float3(TET_N3_X, TET_N3_Y, TET_N3_Z);

    v_sample_pos = (n0 * v_w0 + n1 * v_w1 + n2 * v_w2 + n3 * v_w3) * TET_INV_H;
}

} // namespace interp
} // namespace gsplat

// ============================================================================
// Global wrapper functions
// ============================================================================

template<int CHANNELS = 4>
__device__ __forceinline__ void barycentric_interpolate_cuda_fwd(
    float3 sample_pos,
    const float* v0, const float* v1, const float* v2, const float* v3,
    float* result
) {
    gsplat::interp::barycentric_interpolate_fwd<CHANNELS>(
        sample_pos, v0, v1, v2, v3, result);
}

template<int CHANNELS = 4>
__device__ __forceinline__ void barycentric_interpolate_cuda_bwd(
    float3 sample_pos,
    const float* v0, const float* v1, const float* v2, const float* v3,
    const float* v_result,
    float3* v_sample_pos,
    float* v_v0, float* v_v1, float* v_v2, float* v_v3
) {
    gsplat::interp::barycentric_interpolate_bwd<CHANNELS>(
        sample_pos, v0, v1, v2, v3, v_result,
        *v_sample_pos, v_v0, v_v1, v_v2, v_v3);
}

__device__ __forceinline__ void tetrahedron_barycentric_weights(
    float3 sample_pos,
    float& w0, float& w1, float& w2, float& w3
) {
    gsplat::interp::tet_barycentric_weights(sample_pos, w0, w1, w2, w3);
}

// ============================================================================
// Harmonic Encoding (sin + cos to separate channels)
//   FREQUENCY_SCALE_EXPONENTIAL: 0=linear (k+1), 1=exponential (2^k)
// ============================================================================

__forceinline__ __device__ float get_encoding_frequency(int freq_idx)
{
#if FREQUENCY_SCALE_EXPONENTIAL
    return ldexpf(1.0f, freq_idx);
#else
    return (float)(freq_idx + 1);
#endif
}

inline __device__ void harmonic_encoding_fwd(
    const float base, const uint32_t freq_idx,
    float &out_sin, float &out_cos)
{
    __sincosf(base * get_encoding_frequency(freq_idx), &out_sin, &out_cos);
}

inline __device__ void harmonic_encoding_bwd(
    const float base, const uint32_t freq_idx,
    float &out_d_sin, float &out_d_cos)
{
    const float freq = get_encoding_frequency(freq_idx);
    float s, c;
    __sincosf(base * freq, &s, &c);
    out_d_sin =  c * freq;
    out_d_cos = -s * freq;
}

#endif // GSPLAT_INTERPOLATION_CUH
