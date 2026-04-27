/*
 * SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace gsplat::gemm_raster {

// Shared low-level helpers for the experimental GEMM raster backends. This
// header is intentionally limited to math, packing, and Tensor Core utilities;
// backend-selection policy stays in the calling kernels.

constexpr uint32_t kGemmTileSize = 16;
constexpr uint32_t kGemmBlockSize = kGemmTileSize * kGemmTileSize;
constexpr uint32_t kGemmBatchSize = 16;
constexpr uint32_t kGemmVecLen = 8;
constexpr uint32_t kGemmPowerPadding = 8;

// Store two fp16 values in one 32-bit register. The GEMM kernels use this for
// Tensor Core inputs and compact temporary feature buffers.
__forceinline__ __device__ uint half22uint(half2 x) {
    return *(reinterpret_cast<unsigned int *>(&x));
}

__forceinline__ __device__ half2 uint2half2(uint x) {
    return *(reinterpret_cast<half2 *>(&x));
}

__forceinline__ __device__ uint pack_two_fp32_to_half2_uint(float x, float y) {
    float2 temp_f = make_float2(x, y);
    half2 temp_h = __float22half2_rn(temp_f);
    return half22uint(temp_h);
}

// Keep the exponent evaluation on the fp16 Tensor Core path. If higher
// precision is needed, the calling kernels should adjust their accumulation
// strategy rather than changing this shared helper.
__forceinline__ __device__ __half fast_exp2_f16(__half x) {
    __half y;
    asm volatile(
        "ex2.approx.f16 %0, %1;\n"
        : "=h"(*(reinterpret_cast<unsigned short *>(&(y))))
        : "h"(*(reinterpret_cast<unsigned short *>(&(x))))
    );
    return y;
}

__forceinline__ __device__ __half log2e_half() {
    return __ushort_as_half(static_cast<unsigned short>(0x3dc5));
}

__forceinline__ __device__ void load_matrix_x4(
    uint32_t &reg0,
    uint32_t &reg1,
    uint32_t &reg2,
    uint32_t &reg3,
    const size_t smem_addr
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(reg0), "=r"(reg1), "=r"(reg2), "=r"(reg3)
        : "l"(smem_addr)
    );
#else
    reg0 = 0;
    reg1 = 0;
    reg2 = 0;
    reg3 = 0;
    (void)smem_addr;
#endif
}

__forceinline__ __device__ void load_matrix_x2(
    uint32_t &reg0,
    uint32_t &reg1,
    const size_t smem_addr
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(reg0), "=r"(reg1)
        : "l"(smem_addr)
    );
#else
    reg0 = 0;
    reg1 = 0;
    (void)smem_addr;
#endif
}

__forceinline__ __device__ void mma_16x8x8_fp16(
    uint32_t &regD0,
    uint32_t &regD1,
    const uint32_t regA0,
    const uint32_t regA1,
    const uint32_t regB,
    const uint32_t regC0,
    const uint32_t regC1
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
        : "=r"(regD0), "=r"(regD1)
        : "r"(regA0), "r"(regA1), "r"(regB), "r"(regC0), "r"(regC1)
    );
#else
    regD0 = regC0;
    regD1 = regC1;
    (void)regA0;
    (void)regA1;
    (void)regB;
#endif
}

inline bool is_gemm_raster_supported(const uint32_t tile_size) {
    // The current GEMM kernels are specialized for 16x16 tiles and require
    // SM75+ for the ldmatrix/mma instructions below.
    if (tile_size != kGemmTileSize) {
        return false;
    }
    const cudaDeviceProp *props = at::cuda::getCurrentDeviceProperties();
    return (props->major > 7) || (props->major == 7 && props->minor >= 5);
}

} // namespace gsplat::gemm_raster
