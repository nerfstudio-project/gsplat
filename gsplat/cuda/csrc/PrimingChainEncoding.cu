/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Config.h"

#include <ATen/ATen.h>

#include "PrimingChainEncoding.h"

#if GSPLAT_BUILD_3DGUT

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "PrimingChainEncoding.cuh"

namespace gsplat {

namespace {

__global__ void priming_decode_for_batch_kernel(
    const int32_t *__restrict__ packed,
    const int32_t *__restrict__ batch_ids,
    float *__restrict__ T_init_used,
    int32_t *__restrict__ stored_K,
    bool *__restrict__ stored_sat,
    bool *__restrict__ chain_saturated,
    bool *__restrict__ use_stored_state,
    int64_t n
) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const gsplat::priming::DecodeForBatchResult decoded =
        gsplat::priming::decode_for_batch(
            static_cast<uint32_t>(packed[idx]), batch_ids[idx]);
    T_init_used[idx] = decoded.T_init_used;
    stored_K[idx] = decoded.stored_K;
    stored_sat[idx] = decoded.stored_sat;
    chain_saturated[idx] = decoded.chain_saturated;
    use_stored_state[idx] = decoded.use_stored_state;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
launch_priming_decode_for_batch(
    const at::Tensor &packed,
    const at::Tensor &batch_ids
) {
    TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor");
    TORCH_CHECK(batch_ids.is_cuda(), "batch_ids must be a CUDA tensor");
    TORCH_CHECK(
        packed.device() == batch_ids.device(),
        "packed and batch_ids must be on the same CUDA device"
    );
    TORCH_CHECK(packed.scalar_type() == at::kInt,
                "packed must have dtype int32");
    TORCH_CHECK(batch_ids.scalar_type() == at::kInt,
                "batch_ids must have dtype int32");
    TORCH_CHECK(packed.sizes() == batch_ids.sizes(),
                "packed and batch_ids must have identical shapes");
    TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");
    TORCH_CHECK(batch_ids.is_contiguous(), "batch_ids must be contiguous");

    c10::cuda::CUDAGuard device_guard(packed.device());

    at::Tensor T_init_used =
        at::empty(packed.sizes(), packed.options().dtype(at::kFloat));
    at::Tensor stored_K = at::empty_like(batch_ids);
    at::Tensor stored_sat =
        at::empty(packed.sizes(), packed.options().dtype(at::kBool));
    at::Tensor chain_saturated =
        at::empty(packed.sizes(), packed.options().dtype(at::kBool));
    at::Tensor use_stored_state =
        at::empty(packed.sizes(), packed.options().dtype(at::kBool));

    const int64_t n = packed.numel();
    if (n == 0) {
        return std::make_tuple(
            T_init_used, stored_K, stored_sat, chain_saturated,
            use_stored_state);
    }

    constexpr int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    priming_decode_for_batch_kernel<<<
        static_cast<unsigned int>(blocks),
        threads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
            packed.const_data_ptr<int32_t>(),
            batch_ids.const_data_ptr<int32_t>(),
            T_init_used.data_ptr<float>(),
            stored_K.data_ptr<int32_t>(),
            stored_sat.data_ptr<bool>(),
            chain_saturated.data_ptr<bool>(),
            use_stored_state.data_ptr<bool>(),
            n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(
        T_init_used, stored_K, stored_sat, chain_saturated,
        use_stored_state);
}

}  // namespace gsplat

#else

namespace gsplat {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
launch_priming_decode_for_batch(
    const at::Tensor &packed,
    const at::Tensor &batch_ids
) {
    (void)packed;
    (void)batch_ids;
    TORCH_CHECK(
        false,
        "launch_priming_decode_for_batch requires GSPLAT_BUILD_3DGUT=1"
    );
    return {};
}

}  // namespace gsplat

#endif  // GSPLAT_BUILD_3DGUT
