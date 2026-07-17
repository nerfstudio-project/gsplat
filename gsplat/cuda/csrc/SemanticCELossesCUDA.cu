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

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAException.h>
#    include <c10/cuda/CUDAStream.h>
#    include <c10/macros/Macros.h>

#    include <cstdint>
#    include <limits>

#    include "KernelUtils.cuh"
#    include "cores/SemanticCECore.cuh"

namespace gsplat
{
// Masked semantic cross-entropy: row gating comes from a caller-provided
// boolean `valid` row mask (no hardcoded per-ray flag predicate), and
// the row loss is the numerically stable logsumexp form
//     CE_i = log(sum_j exp(x_ij - m_i)) + m_i - x_i[t_i],   m_i = max_j x_ij.
//
// The per-element / per-row arithmetic lives in cores/SemanticCECore.cuh
// (shared with the fused all-losses kernel); the kernels below are shells
// that own the warp-per-row layout, the warp/block reductions, and the
// scalar-accumulator atomics.
//
// Denominator contract: every valid row counts toward the denominator — rows
// whose target equals `ignore_index` contribute zero to the numerator only.
// The finalize kernel divides by the count when it is positive and returns an
// exact zero otherwise (equivalent to clamping the denominator to >= 1, since
// the numerator is also zero when no row is valid).
//
// Out-of-range targets: a valid row whose target is outside [0, n_classes)
// (and not equal to `ignore_index`) is a caller bug. Both kernels fire a loud
// device assert on assert-enabled builds; on assert-free builds the row is
// deterministically skipped exactly like an ignored row — zero numerator
// contribution, still counted in the denominator, exact-zero gradients, and
// its logits are never read.
//
// Backward: d(loss)/d(x_ij) = (softmax_ij - [j == t_i]) * v_loss / count for
// valid, non-ignored rows; exact zeros for invalid, ignored, and (assert-free)
// out-of-range rows.

namespace
{
    constexpr int SCE_WARP_SIZE       = 32;
    constexpr int SCE_BLOCK_THREADS   = 256;
    constexpr int SCE_WARPS_PER_BLOCK = SCE_BLOCK_THREADS / SCE_WARP_SIZE;
    // Keep the two scalar accumulators from receiving one atomic update per
    // block on production-sized images: once this occupancy-sized cap is
    // reached, each warp grid-strides over additional rows and the per-block
    // partials are flushed with a single pair of atomics.
    constexpr int SCE_MAX_BLOCKS      = 1024;

    static_assert(SCE_BLOCK_THREADS % SCE_WARP_SIZE == 0, "block must be whole warps");

    // The kernels lay out lanes by SCE_WARP_SIZE and pass it as the explicit
    // reduction width at every warp_reduce_*<SCE_WARP_SIZE> call site, so the
    // tree can no longer diverge from the layout the way the native-warpSize
    // shared helper did on a 64-lane wave. Layout, the 32-bit FULL_WARP_MASK
    // below, and the width are all 32-bound; a wave64 port must convert all
    // three together (see the wave64 audit) before this can widen.
    static_assert(SCE_WARP_SIZE == 32, "lane layout, masks, and the reduction width all assume 32");
    constexpr unsigned FULL_WARP_MASK = 0xFFFFFFFFu;

    // -----------------------------------------------------------------------
    // Forward: one warp per row (lanes stride the classes), warp partials
    // reduced per block in shared memory, one atomic pair per block.
    //
    // Row/class indexing is int64: the host validation admits dims up to
    // INT_MAX, where the 32-bit loop updates (`row += row_stride`,
    // `class_idx += SCE_WARP_SIZE`) would overflow past the final iteration
    // and wrap the index negative (reading before the row pointer).
    // -----------------------------------------------------------------------
    template<typename scalar_t, typename target_t>
    __global__ void semantic_ce_fwd_kernel(
        const int64_t n_rows,
        const int64_t n_classes,
        const scalar_t *__restrict__ logits,  // [n_rows, n_classes]
        const target_t *__restrict__ targets, // [n_rows]
        const bool *__restrict__ valid,       // [n_rows]
        const int64_t ignore_index,
        scalar_t *__restrict__ loss_sum,  // [1] accumulator (zero-initialized)
        int32_t *__restrict__ valid_count // [1] accumulator (zero-initialized)
    )
    {
        __shared__ scalar_t shared_warp_loss[SCE_WARPS_PER_BLOCK];
        __shared__ int32_t shared_warp_count[SCE_WARPS_PER_BLOCK];

        const int lane_idx       = threadIdx.x & (SCE_WARP_SIZE - 1);
        const int warp_idx       = threadIdx.x / SCE_WARP_SIZE;
        const int64_t first_row  = static_cast<int64_t>(blockIdx.x) * SCE_WARPS_PER_BLOCK + warp_idx;
        const int64_t row_stride = static_cast<int64_t>(gridDim.x) * SCE_WARPS_PER_BLOCK;

        scalar_t warp_loss = 0;
        int32_t warp_count = 0;

        for(int64_t row = first_row; row < n_rows; row += row_stride)
        {
            // All lanes load the same element — a broadcast from one cache
            // line, so every lane holds identical (warp-uniform) values and
            // the branches below do not diverge within the warp.
            const bool row_valid = valid[row];
            const int64_t target = row_valid ? semantic_ce_classify_target(targets[row], n_classes, ignore_index) : 0;
            // A valid row with a target outside [0, n_classes) (and not equal
            // to ignore_index) is a caller bug: assert loudly when device
            // asserts are enabled; when they are compiled out, the
            // `contributing` predicate deterministically skips the row for the
            // numerator (it still counts in the denominator below), same as an
            // ignored row.
            CUDA_KERNEL_ASSERT(!row_valid || target != kSemanticCeTargetInvalid);
            const bool contributing = row_valid && target >= 0;

            const scalar_t *row_ptr = logits + row * n_classes;

            // Numerically stable logsumexp: subtract the row max before exp.
            scalar_t row_max = -std::numeric_limits<scalar_t>::infinity();
            if(contributing)
            {
                for(int64_t class_idx = lane_idx; class_idx < n_classes; class_idx += SCE_WARP_SIZE)
                {
                    const scalar_t value = row_ptr[class_idx];
                    row_max              = value > row_max ? value : row_max;
                }
            }
            warp_reduce_max_all_lanes<SCE_WARP_SIZE>(FULL_WARP_MASK, row_max);

            scalar_t exp_sum      = 0;
            scalar_t target_logit = 0;
            if(contributing)
            {
                for(int64_t class_idx = lane_idx; class_idx < n_classes; class_idx += SCE_WARP_SIZE)
                {
                    const scalar_t value  = row_ptr[class_idx];
                    exp_sum              += semantic_ce_exp_shifted(value, row_max);
                    if(class_idx == target)
                    {
                        target_logit = value;
                    }
                }
            }
            // Exactly one lane saw the target class; summing broadcasts it.
            warp_reduce_sum_all_lanes<SCE_WARP_SIZE>(FULL_WARP_MASK, exp_sum, target_logit);

            if(lane_idx == 0 && row_valid)
            {
                warp_loss += semantic_ce_row_loss(contributing, exp_sum, row_max, target_logit);
                // Every valid row counts toward the denominator, including
                // ignore_index (and assert-free out-of-range) rows.
                ++warp_count;
            }
        }

        if(lane_idx == 0)
        {
            shared_warp_loss[warp_idx]  = warp_loss;
            shared_warp_count[warp_idx] = warp_count;
        }
        __syncthreads();

        if(threadIdx.x == 0)
        {
            scalar_t block_loss = 0;
            int32_t block_count = 0;
#    pragma unroll
            for(int w = 0; w < SCE_WARPS_PER_BLOCK; w++)
            {
                block_loss  += shared_warp_loss[w];
                block_count += shared_warp_count[w];
            }
            if(block_count > 0)
            {
                atomicAdd(loss_sum, block_loss);
                atomicAdd(valid_count, block_count);
            }
        }
    }

    // Finalize the masked mean (clamped-denominator contract; see
    // semantic_ce_masked_mean in the core header).
    template<typename scalar_t>
    __global__ void semantic_ce_finalize_kernel(
        const scalar_t *__restrict__ loss_sum, const int32_t *__restrict__ valid_count, scalar_t *__restrict__ loss
    )
    {
        loss[0] = semantic_ce_masked_mean(loss_sum[0], valid_count[0]);
    }

    // -----------------------------------------------------------------------
    // Backward: same warp-per-row layout (and the same int64 row/class
    // indexing as the forward). The grid-stride loop writes every
    // (row, class) element of v_logits exactly once — softmax-minus-onehot
    // gradients for contributing rows, exact zeros for invalid, ignored, and
    // (assert-free) out-of-range rows — so the buffer needs no pre-zeroing:
    // the autograd wrapper allocates it with torch.empty_like.
    // -----------------------------------------------------------------------
    template<typename scalar_t, typename target_t>
    __global__ void semantic_ce_bwd_kernel(
        const int64_t n_rows,
        const int64_t n_classes,
        const scalar_t *__restrict__ logits,  // [n_rows, n_classes]
        const target_t *__restrict__ targets, // [n_rows]
        const bool *__restrict__ valid,       // [n_rows]
        const int64_t ignore_index,
        const int32_t *__restrict__ valid_count, // [1] from forward
        const scalar_t *__restrict__ v_loss,     // [] upstream gradient
        scalar_t *__restrict__ v_logits          // [n_rows, n_classes]
    )
    {
        const int lane_idx       = threadIdx.x & (SCE_WARP_SIZE - 1);
        const int warp_idx       = threadIdx.x / SCE_WARP_SIZE;
        const int64_t first_row  = static_cast<int64_t>(blockIdx.x) * SCE_WARPS_PER_BLOCK + warp_idx;
        const int64_t row_stride = static_cast<int64_t>(gridDim.x) * SCE_WARPS_PER_BLOCK;

        const int32_t count  = valid_count[0];
        const scalar_t scale = semantic_ce_masked_mean(v_loss[0], count);

        for(int64_t row = first_row; row < n_rows; row += row_stride)
        {
            const bool row_valid = valid[row];
            const int64_t target = row_valid ? semantic_ce_classify_target(targets[row], n_classes, ignore_index) : 0;
            // Same out-of-range contract as the forward: loud on
            // assert-enabled builds, deterministic skip (exact-zero gradient
            // row, written below) when asserts are compiled out.
            CUDA_KERNEL_ASSERT(!row_valid || target != kSemanticCeTargetInvalid);

            scalar_t *grad_row = v_logits + row * n_classes;
            if(!row_valid || target < 0 || count <= 0)
            {
                // Exact zeros: masked-out, ignored, and (assert-free)
                // out-of-range rows contribute nothing.
                // (count <= 0 cannot co-occur with a valid row after a well-
                // formed forward; the guard protects direct op calls.)
                for(int64_t class_idx = lane_idx; class_idx < n_classes; class_idx += SCE_WARP_SIZE)
                {
                    grad_row[class_idx] = scalar_t(0);
                }
                continue;
            }

            const scalar_t *row_ptr = logits + row * n_classes;

            scalar_t row_max = -std::numeric_limits<scalar_t>::infinity();
            for(int64_t class_idx = lane_idx; class_idx < n_classes; class_idx += SCE_WARP_SIZE)
            {
                const scalar_t value = row_ptr[class_idx];
                row_max              = value > row_max ? value : row_max;
            }
            warp_reduce_max_all_lanes<SCE_WARP_SIZE>(FULL_WARP_MASK, row_max);

            scalar_t exp_sum = 0;
            for(int64_t class_idx = lane_idx; class_idx < n_classes; class_idx += SCE_WARP_SIZE)
            {
                exp_sum += semantic_ce_exp_shifted(row_ptr[class_idx], row_max);
            }
            warp_reduce_sum_all_lanes<SCE_WARP_SIZE>(FULL_WARP_MASK, exp_sum);
            const scalar_t inverse_sum = scalar_t(1) / exp_sum;

            for(int64_t class_idx = lane_idx; class_idx < n_classes; class_idx += SCE_WARP_SIZE)
            {
                grad_row[class_idx]
                    = semantic_ce_grad_element(row_ptr[class_idx], row_max, inverse_sum, class_idx == target, scale);
            }
        }
    }
} // namespace

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

void launch_semantic_ce_fwd_kernel(
    const at::Tensor &logits,
    const at::Tensor &targets,
    const at::Tensor &valid,
    int64_t ignore_index,
    at::Tensor &loss_sum,
    at::Tensor &valid_count,
    at::Tensor &loss
)
{
    const int64_t n_rows    = logits.size(0);
    const int64_t n_classes = logits.size(1);
    const auto stream       = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        logits.scalar_type(),
        "semantic_ce_fwd",
        [&]()
        {
            if(n_rows > 0)
            {
                // 64-bit ceil-div (an int32 `n_rows + 7` would overflow at
                // INT_MAX rows); the SCE_MAX_BLOCKS cap keeps the grid dim
                // 32-bit-safe.
                const int64_t required_blocks = (n_rows + SCE_WARPS_PER_BLOCK - 1) / SCE_WARPS_PER_BLOCK;
                const dim3 grid(
                    static_cast<uint32_t>(required_blocks < SCE_MAX_BLOCKS ? required_blocks : SCE_MAX_BLOCKS)
                );
                const dim3 threads(SCE_BLOCK_THREADS);
                if(targets.scalar_type() == at::kByte)
                {
                    semantic_ce_fwd_kernel<scalar_t, uint8_t><<<grid, threads, 0, stream>>>(
                        n_rows,
                        n_classes,
                        logits.data_ptr<scalar_t>(),
                        targets.data_ptr<uint8_t>(),
                        valid.data_ptr<bool>(),
                        ignore_index,
                        loss_sum.data_ptr<scalar_t>(),
                        valid_count.data_ptr<int32_t>()
                    );
                }
                else if(targets.scalar_type() == at::kLong)
                {
                    semantic_ce_fwd_kernel<scalar_t, int64_t><<<grid, threads, 0, stream>>>(
                        n_rows,
                        n_classes,
                        logits.data_ptr<scalar_t>(),
                        targets.data_ptr<int64_t>(),
                        valid.data_ptr<bool>(),
                        ignore_index,
                        loss_sum.data_ptr<scalar_t>(),
                        valid_count.data_ptr<int32_t>()
                    );
                }
                else
                {
                    TORCH_CHECK(false, "unsupported semantic target dtype ", targets.scalar_type());
                }
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            // Always finalize so `loss` is written even when n_rows == 0.
            semantic_ce_finalize_kernel<scalar_t><<<1, 1, 0, stream>>>(
                loss_sum.data_ptr<scalar_t>(), valid_count.data_ptr<int32_t>(), loss.data_ptr<scalar_t>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
}

void launch_semantic_ce_bwd_kernel(
    const at::Tensor &logits,
    const at::Tensor &targets,
    const at::Tensor &valid,
    int64_t ignore_index,
    const at::Tensor &valid_count,
    const at::Tensor &v_loss,
    at::Tensor &v_logits
)
{
    const int64_t n_rows    = logits.size(0);
    const int64_t n_classes = logits.size(1);
    if(n_rows == 0)
    {
        return;
    }

    // 64-bit ceil-div (an int32 `n_rows + 7` would overflow at INT_MAX rows);
    // the SCE_MAX_BLOCKS cap keeps the grid dim 32-bit-safe.
    const int64_t required_blocks = (n_rows + SCE_WARPS_PER_BLOCK - 1) / SCE_WARPS_PER_BLOCK;
    const dim3 grid(static_cast<uint32_t>(required_blocks < SCE_MAX_BLOCKS ? required_blocks : SCE_MAX_BLOCKS));
    const dim3 threads(SCE_BLOCK_THREADS);
    const auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        logits.scalar_type(),
        "semantic_ce_bwd",
        [&]()
        {
            if(targets.scalar_type() == at::kByte)
            {
                semantic_ce_bwd_kernel<scalar_t, uint8_t><<<grid, threads, 0, stream>>>(
                    n_rows,
                    n_classes,
                    logits.data_ptr<scalar_t>(),
                    targets.data_ptr<uint8_t>(),
                    valid.data_ptr<bool>(),
                    ignore_index,
                    valid_count.data_ptr<int32_t>(),
                    v_loss.data_ptr<scalar_t>(),
                    v_logits.data_ptr<scalar_t>()
                );
            }
            else if(targets.scalar_type() == at::kLong)
            {
                semantic_ce_bwd_kernel<scalar_t, int64_t><<<grid, threads, 0, stream>>>(
                    n_rows,
                    n_classes,
                    logits.data_ptr<scalar_t>(),
                    targets.data_ptr<int64_t>(),
                    valid.data_ptr<bool>(),
                    ignore_index,
                    valid_count.data_ptr<int32_t>(),
                    v_loss.data_ptr<scalar_t>(),
                    v_logits.data_ptr<scalar_t>()
                );
            }
            else
            {
                TORCH_CHECK(false, "unsupported semantic target dtype ", targets.scalar_type());
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
}
} // namespace gsplat

#endif
