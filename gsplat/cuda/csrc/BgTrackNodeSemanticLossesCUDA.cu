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

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAException.h>
#    include <c10/cuda/CUDAGuard.h>
#    include <c10/cuda/CUDAStream.h>
#    include <c10/macros/Macros.h>

#    include <algorithm>
#    include <cstdint>
#    include <limits>

#    include "BgTrackNodeSemanticLosses.h"
#    include "Common.h"
#    include "cores/BgTrackNodeSemanticCore.cuh"

namespace gsplat
{
namespace
{
    // ------------------------------------------------------------------------
    // Constants and shared conventions
    //
    // The per-point selection predicates and BCE value/gradient math live in
    // cores/BgTrackNodeSemanticCore.cuh (shared with the future fused
    // all-losses kernel); the kernels below are shells owning iteration,
    // track tiling, reductions, and atomics.
    // ------------------------------------------------------------------------

    constexpr int BLOCK_THREADS       = 256;
    constexpr int WARP_SIZE           = 32;
    constexpr int WARPS_PER_BLOCK     = BLOCK_THREADS / WARP_SIZE;
    constexpr unsigned FULL_WARP_MASK = 0xFFFFFFFFu;

    constexpr int TRACK_TILE_SIZE = 64;

    // Selection-bit, semantic mask-word, and track-box layout conventions are
    // defined once in the core header.
    using bg_track_node_semantic::BACKGROUND_IN_TRACK_SELECTION_BIT;
    using bg_track_node_semantic::NODE_SEMANTIC_SELECTION_BIT;
    using bg_track_node_semantic::SEMANTIC_CLASS_BITS;
    using bg_track_node_semantic::SEMANTIC_CLASS_COUNT;
    using bg_track_node_semantic::SEMANTIC_CLASS_MASK;
    using bg_track_node_semantic::TRACK_BOX_VALUES;

    // In-kernel reduction workspace, caller-allocated as an int32[8] tensor
    // (32 bytes, 16-byte aligned) so allocation is dtype-independent: the fp32
    // layout occupies 16 bytes, the fp64 layout the full 32 bytes.
    constexpr int64_t WORKSPACE_INT32_WORDS = 8;

    template<typename scalar_t>
    struct alignas(16) BgTrackNodeSemanticWorkspace
    {
        scalar_t background_loss_sum;
        scalar_t node_loss_sum;
        int32_t background_selected_count;
        int32_t node_selected_count;
    };

    static_assert(sizeof(BgTrackNodeSemanticWorkspace<float>) == 16);
    static_assert(sizeof(BgTrackNodeSemanticWorkspace<double>) <= WORKSPACE_INT32_WORDS * sizeof(int32_t));
    static_assert(alignof(BgTrackNodeSemanticWorkspace<float>) == 16);
    static_assert(alignof(BgTrackNodeSemanticWorkspace<double>) == 16);

    inline uint32_t div_round_up(uint32_t val, uint32_t divisor)
    {
        return (val + divisor - 1) / divisor;
    }

    inline void check_same_device(const at::Device device, const at::Tensor &tensor, const char *name)
    {
        TORCH_CHECK(tensor.device() == device, name, " must be on ", device, ", got ", tensor.device());
    }

    // ------------------------------------------------------------------------
    // Device helpers
    // ------------------------------------------------------------------------

    // Warp-wide sum reduction across all WARP_SIZE lanes for any number of
    // accumulators. Every lane must participate; lane 0 receives the sums.
    template<typename... Ts>
    __device__ __forceinline__ void warp_reduce_sum_all(Ts &...values)
    {
#    pragma unroll
        for(int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        {
            ((values += __shfl_down_sync(FULL_WARP_MASK, values, offset)), ...);
        }
    }

    // Binary search over sorted timestamps returning the "right" bound index:
    // (none or data[ret-1]) <= val < data[ret], assuming val <= data[length-1]
    // was handled by the caller wrapper below.
    __device__ __forceinline__ int32_t
        binary_search_unsafe(const int64_t val, const int64_t *__restrict__ data, const int32_t length)
    {
        int32_t count = length;
        int32_t first = 0;
        while(count > 0)
        {
            int32_t it          = first;
            const int32_t step  = count / 2;
            it                 += step;
            if(data[it] < val)
            {
                first  = it + 1;
                count -= step + 1;
            }
            else
            {
                count = step;
            }
        }
        return first;
    }

    // Same as a plain right-bound binary search but returns length - 1 when
    // val == data[length - 1] (left-open interval at the range end), so the
    // caller can always interpolate between ret - 1 and ret for in-range
    // queries. Ported from ku::binary_search_interp.
    __device__ __forceinline__ int32_t
        binary_search_interp(const int64_t val, const int64_t *__restrict__ data, const int32_t length)
    {
        if(length <= 0)
        {
            return 0;
        }
        if(val == data[0])
        {
            return 1;
        }
        if(val == data[length - 1])
        {
            return length - 1;
        }
        if(val > data[length - 1])
        {
            return length;
        }
        return binary_search_unsafe(val, data, length);
    }

    // Quaternion (x, y, z, w) slerp with shortest-arc flip and a linear
    // fallback for nearby quaternions. Ported from ku::unitquat_slerp.
    template<typename scalar_t>
    __device__ __forceinline__ void unitquat_slerp(
        const scalar_t q_start[4], const scalar_t q_end_in[4], const scalar_t t, scalar_t out[4]
    )
    {
        scalar_t q_end[4] = {q_end_in[0], q_end_in[1], q_end_in[2], q_end_in[3]};
        scalar_t cos_omega
            = q_start[0] * q_end[0] + q_start[1] * q_end[1] + q_start[2] * q_end[2] + q_start[3] * q_end[3];
        if(cos_omega < scalar_t(0))
        {
            cos_omega = -cos_omega;
#    pragma unroll
            for(int i = 0; i < 4; i++)
            {
                q_end[i] = -q_end[i];
            }
        }

        const bool nearby_quaternions = cos_omega > scalar_t(1) - scalar_t(1e-3);
        const scalar_t omega          = acos(cos_omega);
        const scalar_t alpha          = nearby_quaternions ? (scalar_t(1) - t) : sin((scalar_t(1) - t) * omega);
        const scalar_t beta           = nearby_quaternions ? t : sin(t * omega);

        scalar_t norm2 = scalar_t(0);
#    pragma unroll
        for(int i = 0; i < 4; i++)
        {
            out[i]  = alpha * q_start[i] + beta * q_end[i];
            norm2  += out[i] * out[i];
        }
        const scalar_t inv_norm = rsqrt(norm2);
#    pragma unroll
        for(int i = 0; i < 4; i++)
        {
            out[i] *= inv_norm;
        }
    }

    // Unit quaternion (x, y, z, w) to a row-major SO3 rotation matrix.
    // Ported from ku::unitquat_rotmatrix.
    template<typename scalar_t>
    __device__ __forceinline__ void unitquat_rotmatrix(const scalar_t q[4], scalar_t R[3][3])
    {
        const scalar_t x = q[0];
        const scalar_t y = q[1];
        const scalar_t z = q[2];
        const scalar_t w = q[3];

        const scalar_t xx = x * x;
        const scalar_t yy = y * y;
        const scalar_t zz = z * z;
        const scalar_t ww = w * w;

        R[0][0] = xx - yy - zz + ww;
        R[0][1] = scalar_t(2) * (x * y - z * w);
        R[0][2] = scalar_t(2) * (x * z + y * w);
        R[1][0] = scalar_t(2) * (x * y + z * w);
        R[1][1] = -xx + yy - zz + ww;
        R[1][2] = scalar_t(2) * (y * z - x * w);
        R[2][0] = scalar_t(2) * (x * z - y * w);
        R[2][1] = scalar_t(2) * (y * z + x * w);
        R[2][2] = -xx - yy + zz + ww;
    }

    struct RangeSemanticFilter
    {
        static constexpr bool USES_SEMANTICS = true;

        int32_t point_begin;
        int32_t mask_word;
        uint32_t class_mask;

        __device__ __forceinline__ int point_index(const int local_point_idx) const
        {
            return point_begin + local_point_idx;
        }

        __device__ __forceinline__ bool is_allowed(const int argmax_idx) const
        {
            return bg_track_node_semantic::mask_word_allows(argmax_idx, mask_word, class_mask);
        }
    };

    // Include-all filter: the kernel skips the semantic argmax and mask check
    // entirely when USES_SEMANTICS is false, so no is_allowed is needed.
    struct UnfilteredRange
    {
        static constexpr bool USES_SEMANTICS = false;

        int32_t point_begin;

        __device__ __forceinline__ int point_index(const int local_point_idx) const
        {
            return point_begin + local_point_idx;
        }
    };

    // ------------------------------------------------------------------------
    // Track-box precompute
    // ------------------------------------------------------------------------

    template<typename scalar_t>
    __global__ void interpolate_track_boxes_kernel(
        const int n_tracks,
        const int n_total_poses,
        const int64_t *__restrict__ camera_timestamps_startend_us,
        const int32_t *__restrict__ tracks_packinfo,
        const scalar_t *__restrict__ tracks_poses,
        const int64_t *__restrict__ tracks_timestamps_us,
        const scalar_t *__restrict__ cuboids_dims,
        scalar_t *__restrict__ track_boxes
    )
    {
        const int track_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(track_idx >= n_tracks)
        {
            return;
        }

        scalar_t *const box   = track_boxes + static_cast<int64_t>(track_idx) * TRACK_BOX_VALUES;
        const auto invalidate = [box]()
        {
            box[12] = box[13] = box[14] = box[15] = scalar_t(0);
        };

        // Only row 0 of camera_timestamps_startend_us (the reference camera)
        // is consumed; the host validation guarantees B >= 1. The midpoint
        // matches torch.mean([start, end], float64).to(int64) for positive
        // microsecond Unix timestamps, without a GPU mean/cast launch.
        const int64_t timestamp_start = camera_timestamps_startend_us[0];
        const int64_t timestamp_end   = camera_timestamps_startend_us[1];
        const int64_t timestamp       = timestamp_start + (timestamp_end - timestamp_start) / 2;

        const int32_t track_start_idx = tracks_packinfo[track_idx * 2];
        const int32_t n_track_poses   = tracks_packinfo[track_idx * 2 + 1];
        // tracks_packinfo contents live on the device and are consumed as
        // gather indices without host validation (see the trusted-input note
        // at the launch sites); this guard turns a corrupt pack table into a
        // loud device-side assert instead of an out-of-bounds read.
        CUDA_KERNEL_ASSERT(
            track_start_idx >= 0
            && n_track_poses >= 0
            && static_cast<int64_t>(track_start_idx) + n_track_poses <= n_total_poses
        );
        if(n_track_poses <= 1)
        {
            invalidate();
            return;
        }

        const int64_t *const local_timestamps = tracks_timestamps_us + track_start_idx;
        if(timestamp < local_timestamps[0] || timestamp > local_timestamps[n_track_poses - 1])
        {
            invalidate();
            return;
        }

        const int32_t interpolation_end_idx   = binary_search_interp(timestamp, local_timestamps, n_track_poses);
        const int32_t interpolation_start_idx = interpolation_end_idx - 1;
        const int64_t t0                      = local_timestamps[interpolation_start_idx];
        const int64_t t1                      = local_timestamps[interpolation_end_idx];
        if(t1 == t0)
        {
            invalidate();
            return;
        }

        const scalar_t alpha = static_cast<scalar_t>(timestamp - t0) / static_cast<scalar_t>(t1 - t0);
        const scalar_t *const pose0
            = tracks_poses + static_cast<int64_t>(track_start_idx + interpolation_start_idx) * 7;
        const scalar_t *const pose1 = tracks_poses + static_cast<int64_t>(track_start_idx + interpolation_end_idx) * 7;

        scalar_t center[3];
#    pragma unroll
        for(int i = 0; i < 3; i++)
        {
            center[i] = (scalar_t(1) - alpha) * pose0[i] + alpha * pose1[i];
        }
        const scalar_t quat0[4] = {pose0[3], pose0[4], pose0[5], pose0[6]};
        const scalar_t quat1[4] = {pose1[3], pose1[4], pose1[5], pose1[6]};
        scalar_t quat[4];
        unitquat_slerp(quat0, quat1, alpha, quat);
        scalar_t R[3][3];
        unitquat_rotmatrix(quat, R);

        // world-to-local rows are the transpose of the local-to-world rotation.
#    pragma unroll
        for(int row = 0; row < 3; row++)
        {
#    pragma unroll
            for(int col = 0; col < 3; col++)
            {
                box[row * 4 + col] = R[col][row];
            }
            box[row * 4 + 3] = center[row];
        }

        const scalar_t *const dims = cuboids_dims + static_cast<int64_t>(track_idx) * 3;
        box[12]                    = dims[0] * scalar_t(0.5);
        box[13]                    = dims[1] * scalar_t(0.5);
        box[14]                    = dims[2] * scalar_t(0.5);
        box[15]                    = scalar_t(1);
    }

    // ------------------------------------------------------------------------
    // Generic segmented background-in-track kernels
    // ------------------------------------------------------------------------

    template<typename scalar_t, typename SemanticFilter>
    __global__ void background_in_track_forward_kernel(
        const int n_launch_points,
        const int n_tracks,
        const int n_semantic_classes,
        const scalar_t density_logits_min,
        const scalar_t *__restrict__ positions,
        const scalar_t *__restrict__ density_logits,
        const scalar_t *__restrict__ semantic_logits,
        const scalar_t *__restrict__ track_boxes,
        uint8_t *__restrict__ selection_bits,
        int32_t *__restrict__ selected_count,
        scalar_t *__restrict__ loss_sum,
        const SemanticFilter semantic_filter
    )
    {
        __shared__ scalar_t shared_boxes[TRACK_TILE_SIZE * TRACK_BOX_VALUES];
        __shared__ scalar_t shared_warp_loss[WARPS_PER_BLOCK];
        __shared__ int32_t shared_warp_count[WARPS_PER_BLOCK];

        const int local_point_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int point_idx       = semantic_filter.point_index(local_point_idx);
        bool candidate            = local_point_idx < n_launch_points;
        scalar_t density_logit    = scalar_t(0);
        scalar_t point[3]         = {scalar_t(0), scalar_t(0), scalar_t(0)};

        if(candidate)
        {
            density_logit = density_logits[point_idx];
            candidate     = density_logit > density_logits_min;
            // Compute the semantic argmax once per point and hand it to the
            // filter instead of recomputing it inside is_allowed for every
            // mask-word check.
            if constexpr(SemanticFilter::USES_SEMANTICS)
            {
                if(candidate)
                {
                    const int argmax_idx
                        = bg_track_node_semantic::semantic_argmax(point_idx, n_semantic_classes, semantic_logits);
                    candidate = semantic_filter.is_allowed(argmax_idx);
                }
            }
            if(candidate)
            {
                const int64_t i3 = static_cast<int64_t>(point_idx) * 3;
                point[0]         = positions[i3 + 0];
                point[1]         = positions[i3 + 1];
                point[2]         = positions[i3 + 2];
            }
        }

        bool selected                = false;
        bool any_remaining_candidate = __syncthreads_or(candidate);
        for(int tile_start = 0; tile_start < n_tracks && any_remaining_candidate; tile_start += TRACK_TILE_SIZE)
        {
            const int tile_tracks = min(TRACK_TILE_SIZE, n_tracks - tile_start);
            const int tile_values = tile_tracks * TRACK_BOX_VALUES;
            for(int value_idx = threadIdx.x; value_idx < tile_values; value_idx += blockDim.x)
            {
                shared_boxes[value_idx] = track_boxes[tile_start * TRACK_BOX_VALUES + value_idx];
            }
            __syncthreads();

            if(candidate && !selected)
            {
                for(int local_track_idx = 0; local_track_idx < tile_tracks; ++local_track_idx)
                {
                    const scalar_t *const box = shared_boxes + local_track_idx * TRACK_BOX_VALUES;
                    if(box[15] > scalar_t(0) && bg_track_node_semantic::point_inside_box(point, box))
                    {
                        selected = true;
                        break;
                    }
                }
            }
            // This vote is also the barrier that protects the next tile
            // overwrite. It skips all track traffic for sparse blocks and stops
            // once every candidate in the block has intersected a track.
            any_remaining_candidate = __syncthreads_or(candidate && !selected);
        }

        if(local_point_idx < n_launch_points)
        {
            if(selected)
            {
                selection_bits[point_idx] = BACKGROUND_IN_TRACK_SELECTION_BIT;
            }
        }

        scalar_t thread_loss = selected ? bg_track_node_semantic::stable_softplus(density_logit) : scalar_t(0);
        int32_t thread_count = selected ? 1 : 0;
        const int lane_idx   = threadIdx.x & (WARP_SIZE - 1);
        const int warp_idx   = threadIdx.x / WARP_SIZE;
        warp_reduce_sum_all(thread_loss, thread_count);
        if(lane_idx == 0)
        {
            shared_warp_loss[warp_idx]  = thread_loss;
            shared_warp_count[warp_idx] = thread_count;
        }
        __syncthreads();

        if(warp_idx == 0)
        {
            scalar_t block_loss = lane_idx < WARPS_PER_BLOCK ? shared_warp_loss[lane_idx] : scalar_t(0);
            int32_t block_count = lane_idx < WARPS_PER_BLOCK ? shared_warp_count[lane_idx] : 0;
            warp_reduce_sum_all(block_loss, block_count);
            if(lane_idx == 0 && block_count > 0)
            {
                atomicAdd(loss_sum, block_loss);
                atomicAdd(selected_count, block_count);
            }
        }
    }

    template<typename scalar_t>
    __global__ void finalize_background_in_track_loss_kernel(
        const BgTrackNodeSemanticWorkspace<scalar_t> *__restrict__ workspace,
        const scalar_t background_lambda,
        scalar_t *__restrict__ loss,
        scalar_t *__restrict__ weighted_loss
    )
    {
        const int32_t count  = workspace->background_selected_count;
        const scalar_t value = count > 0 ? workspace->background_loss_sum / static_cast<scalar_t>(count) : scalar_t(0);
        loss[0]              = value;
        weighted_loss[0]     = background_lambda * value;
    }

    template<typename scalar_t>
    __global__ void background_in_track_backward_kernel(
        const int n_points,
        const scalar_t *__restrict__ density_logits,
        const uint8_t *__restrict__ selection_bits,
        const int32_t *__restrict__ selected_count,
        const scalar_t *__restrict__ grad_unweighted_loss,
        const scalar_t *__restrict__ grad_weighted_loss,
        const scalar_t background_lambda,
        scalar_t *__restrict__ grad_density_logits
    )
    {
        const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(point_idx >= n_points)
        {
            return;
        }

        const int32_t count = selected_count[0];
        // Absent upstream gradients arrive as nullptr and count as zero (the
        // op-level gradients are optional).
        const scalar_t upstream
            = (grad_unweighted_loss != nullptr ? grad_unweighted_loss[0] : scalar_t(0))
            + background_lambda * (grad_weighted_loss != nullptr ? grad_weighted_loss[0] : scalar_t(0));
        grad_density_logits[point_idx]
            = (selection_bits[point_idx] & BACKGROUND_IN_TRACK_SELECTION_BIT) != 0 && count > 0
                ? upstream
                      * bg_track_node_semantic::stable_sigmoid(density_logits[point_idx])
                      / static_cast<scalar_t>(count)
                : scalar_t(0);
    }

    // ------------------------------------------------------------------------
    // Packed node-semantic kernels
    // ------------------------------------------------------------------------

    template<typename scalar_t, bool MULTIWORD_SEMANTICS>
    __global__ void node_semantic_forward_kernel(
        const int n_points,
        const int n_semantic_classes,
        const int semantic_mask_word,
        const scalar_t *__restrict__ density_logits,
        const scalar_t *__restrict__ semantic_logits,
        const uint32_t class_mask,
        const bool select_matches,
        uint8_t *__restrict__ selection_bits,
        BgTrackNodeSemanticWorkspace<scalar_t> *__restrict__ workspace
    )
    {
        __shared__ scalar_t shared_warp_loss[WARPS_PER_BLOCK];
        __shared__ int32_t shared_warp_count[WARPS_PER_BLOCK];

        const int point_idx    = blockIdx.x * blockDim.x + threadIdx.x;
        bool selected          = false;
        bool owns_class        = false;
        scalar_t density_logit = scalar_t(0);
        if(point_idx < n_points)
        {
            const int semantic_class
                = bg_track_node_semantic::semantic_argmax(point_idx, n_semantic_classes, semantic_logits);
            owns_class
                = bg_track_node_semantic::owns_mask_word<MULTIWORD_SEMANTICS>(semantic_class, semantic_mask_word);
            if(owns_class)
            {
                // Polarity is subtle (see the core's contract): an empty class
                // mask with select_matches=true penalizes nothing; with
                // select_matches=false it penalizes everything.
                selected = bg_track_node_semantic::selection_matches<MULTIWORD_SEMANTICS>(
                    semantic_class, class_mask, select_matches
                );
                selection_bits[point_idx] = selected ? NODE_SEMANTIC_SELECTION_BIT : uint8_t{0};
                density_logit             = density_logits[point_idx];
            }
        }

        scalar_t thread_loss = selected ? bg_track_node_semantic::stable_softplus(density_logit) : scalar_t(0);
        int32_t thread_count = selected ? 1 : 0;
        const int lane_idx   = threadIdx.x & (WARP_SIZE - 1);
        const int warp_idx   = threadIdx.x / WARP_SIZE;
        warp_reduce_sum_all(thread_loss, thread_count);
        if(lane_idx == 0)
        {
            shared_warp_loss[warp_idx]  = thread_loss;
            shared_warp_count[warp_idx] = thread_count;
        }
        __syncthreads();

        if(warp_idx == 0)
        {
            scalar_t block_loss = lane_idx < WARPS_PER_BLOCK ? shared_warp_loss[lane_idx] : scalar_t(0);
            int32_t block_count = lane_idx < WARPS_PER_BLOCK ? shared_warp_count[lane_idx] : 0;
            warp_reduce_sum_all(block_loss, block_count);
            if(lane_idx == 0 && block_count > 0)
            {
                atomicAdd(&workspace->node_loss_sum, block_loss);
                atomicAdd(&workspace->node_selected_count, block_count);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Joint two-node kernel: shared primary domain, bg-track containment test
    // + node-semantic selection in one pass
    // ------------------------------------------------------------------------

    template<typename scalar_t, bool MULTIWORD_SEMANTICS>
    __global__ void background_in_track_node_semantic_two_node_forward_kernel(
        const int n_background_points,
        const int n_background_blocks,
        const int n_background_semantic_classes,
        const int n_other_node_points,
        const int n_other_node_semantic_classes,
        const int semantic_mask_word,
        const int n_tracks,
        const scalar_t background_lambda,
        const scalar_t density_logits_min,
        const scalar_t *__restrict__ background_positions,
        const scalar_t *__restrict__ background_density_logits,
        const scalar_t *__restrict__ background_semantic_logits,
        const scalar_t *__restrict__ other_node_density_logits,
        const scalar_t *__restrict__ other_node_semantic_logits,
        const uint32_t background_allowed_class_mask,
        const uint32_t node_background_class_mask,
        const bool node_background_select,
        const uint32_t node_other_class_mask,
        const bool node_other_select,
        const scalar_t *__restrict__ track_boxes,
        uint8_t *__restrict__ selection_bits,
        BgTrackNodeSemanticWorkspace<scalar_t> *__restrict__ workspace
    )
    {
        __shared__ scalar_t shared_boxes[TRACK_TILE_SIZE * TRACK_BOX_VALUES];
        __shared__ scalar_t shared_warp_background_loss[WARPS_PER_BLOCK];
        __shared__ int32_t shared_warp_background_count[WARPS_PER_BLOCK];
        __shared__ scalar_t shared_warp_node_loss[WARPS_PER_BLOCK];
        __shared__ int32_t shared_warp_node_count[WARPS_PER_BLOCK];

        const bool is_background = static_cast<int>(blockIdx.x) < n_background_blocks;
        const int node_block
            = is_background ? static_cast<int>(blockIdx.x) : static_cast<int>(blockIdx.x) - n_background_blocks;
        const int point_idx = node_block * blockDim.x + threadIdx.x;

        bool background_selected = false;
        bool node_selected       = false;
        bool owns_semantic_class = false;
        scalar_t density_logit   = scalar_t(0);
        if(is_background)
        {
            bool track_candidate = false;
            scalar_t point[3]    = {scalar_t(0), scalar_t(0), scalar_t(0)};
            if(point_idx < n_background_points)
            {
                density_logit            = background_density_logits[point_idx];
                const int semantic_class = bg_track_node_semantic::semantic_argmax(
                    point_idx, n_background_semantic_classes, background_semantic_logits
                );
                owns_semantic_class
                    = bg_track_node_semantic::owns_mask_word<MULTIWORD_SEMANTICS>(semantic_class, semantic_mask_word);
                if(owns_semantic_class)
                {
                    node_selected = bg_track_node_semantic::selection_matches<MULTIWORD_SEMANTICS>(
                        semantic_class, node_background_class_mask, node_background_select
                    );
                    const bool background_class_allowed
                        = background_lambda >= scalar_t(0)
                       && bg_track_node_semantic::class_mask_matches<MULTIWORD_SEMANTICS>(
                              semantic_class, background_allowed_class_mask
                       );
                    track_candidate = density_logit > density_logits_min && background_class_allowed;
                    if(track_candidate)
                    {
                        const int64_t i3 = static_cast<int64_t>(point_idx) * 3;
                        point[0]         = background_positions[i3 + 0];
                        point[1]         = background_positions[i3 + 1];
                        point[2]         = background_positions[i3 + 2];
                    }
                }
            }

            bool any_remaining_candidate = __syncthreads_or(track_candidate);
            for(int tile_start = 0; tile_start < n_tracks && any_remaining_candidate; tile_start += TRACK_TILE_SIZE)
            {
                const int tile_tracks = min(TRACK_TILE_SIZE, n_tracks - tile_start);
                const int tile_values = tile_tracks * TRACK_BOX_VALUES;
                for(int value_idx = threadIdx.x; value_idx < tile_values; value_idx += blockDim.x)
                {
                    shared_boxes[value_idx] = track_boxes[tile_start * TRACK_BOX_VALUES + value_idx];
                }
                __syncthreads();

                if(track_candidate && !background_selected)
                {
                    for(int local_track_idx = 0; local_track_idx < tile_tracks; ++local_track_idx)
                    {
                        const scalar_t *const box = shared_boxes + local_track_idx * TRACK_BOX_VALUES;
                        if(box[15] > scalar_t(0) && bg_track_node_semantic::point_inside_box(point, box))
                        {
                            background_selected = true;
                            break;
                        }
                    }
                }
                any_remaining_candidate = __syncthreads_or(track_candidate && !background_selected);
            }

            if(point_idx < n_background_points && owns_semantic_class)
            {
                selection_bits[point_idx] = (background_selected ? BACKGROUND_IN_TRACK_SELECTION_BIT : uint8_t{0})
                                          | (node_selected ? NODE_SEMANTIC_SELECTION_BIT : uint8_t{0});
            }
        }
        else if(point_idx < n_other_node_points)
        {
            density_logit            = other_node_density_logits[point_idx];
            const int semantic_class = bg_track_node_semantic::semantic_argmax(
                point_idx, n_other_node_semantic_classes, other_node_semantic_logits
            );
            owns_semantic_class
                = bg_track_node_semantic::owns_mask_word<MULTIWORD_SEMANTICS>(semantic_class, semantic_mask_word);
            if(owns_semantic_class)
            {
                node_selected = bg_track_node_semantic::selection_matches<MULTIWORD_SEMANTICS>(
                    semantic_class, node_other_class_mask, node_other_select
                );
                selection_bits[n_background_points + point_idx]
                    = node_selected ? NODE_SEMANTIC_SELECTION_BIT : uint8_t{0};
            }
        }

        const scalar_t selected_loss    = background_selected || node_selected
                                            ? bg_track_node_semantic::stable_softplus(density_logit)
                                            : scalar_t(0);
        scalar_t background_thread_loss = background_selected ? selected_loss : scalar_t(0);
        int32_t background_thread_count = background_selected ? 1 : 0;
        scalar_t node_thread_loss       = node_selected ? selected_loss : scalar_t(0);
        int32_t node_thread_count       = node_selected ? 1 : 0;
        const int lane_idx              = threadIdx.x & (WARP_SIZE - 1);
        const int warp_idx              = threadIdx.x / WARP_SIZE;
        warp_reduce_sum_all(background_thread_loss, background_thread_count, node_thread_loss, node_thread_count);
        if(lane_idx == 0)
        {
            shared_warp_background_loss[warp_idx]  = background_thread_loss;
            shared_warp_background_count[warp_idx] = background_thread_count;
            shared_warp_node_loss[warp_idx]        = node_thread_loss;
            shared_warp_node_count[warp_idx]       = node_thread_count;
        }
        __syncthreads();

        if(warp_idx == 0)
        {
            scalar_t background_block_loss
                = lane_idx < WARPS_PER_BLOCK ? shared_warp_background_loss[lane_idx] : scalar_t(0);
            int32_t background_block_count = lane_idx < WARPS_PER_BLOCK ? shared_warp_background_count[lane_idx] : 0;
            scalar_t node_block_loss       = lane_idx < WARPS_PER_BLOCK ? shared_warp_node_loss[lane_idx] : scalar_t(0);
            int32_t node_block_count       = lane_idx < WARPS_PER_BLOCK ? shared_warp_node_count[lane_idx] : 0;
            warp_reduce_sum_all(background_block_loss, background_block_count, node_block_loss, node_block_count);
            if(lane_idx == 0)
            {
                if(background_block_count > 0)
                {
                    atomicAdd(&workspace->background_loss_sum, background_block_loss);
                    atomicAdd(&workspace->background_selected_count, background_block_count);
                }
                if(node_block_count > 0)
                {
                    atomicAdd(&workspace->node_loss_sum, node_block_loss);
                    atomicAdd(&workspace->node_selected_count, node_block_count);
                }
            }
        }
    }

    template<typename scalar_t>
    __global__ void finalize_background_in_track_node_semantic_two_node_kernel(
        const BgTrackNodeSemanticWorkspace<scalar_t> *__restrict__ workspace,
        const scalar_t background_lambda,
        const scalar_t node_lambda,
        scalar_t *__restrict__ background_unweighted_loss,
        scalar_t *__restrict__ background_weighted_loss,
        scalar_t *__restrict__ node_unweighted_loss,
        scalar_t *__restrict__ node_weighted_loss
    )
    {
        const int32_t background_count  = workspace->background_selected_count;
        const scalar_t background_value = background_count > 0
                                            ? workspace->background_loss_sum / static_cast<scalar_t>(background_count)
                                            : scalar_t(0);
        const int32_t node_count        = workspace->node_selected_count;
        const scalar_t node_value
            = node_count > 0 ? workspace->node_loss_sum / static_cast<scalar_t>(node_count) : scalar_t(0);
        background_unweighted_loss[0] = background_value;
        background_weighted_loss[0]   = background_lambda * background_value;
        node_unweighted_loss[0]       = node_value;
        node_weighted_loss[0]         = node_lambda * node_value;
    }

    template<typename scalar_t>
    __global__ void finalize_node_semantic_loss_kernel(
        const BgTrackNodeSemanticWorkspace<scalar_t> *__restrict__ workspace,
        const scalar_t node_lambda,
        scalar_t *__restrict__ node_unweighted_loss,
        scalar_t *__restrict__ node_weighted_loss
    )
    {
        const int32_t node_count = workspace->node_selected_count;
        const scalar_t node_value
            = node_count > 0 ? workspace->node_loss_sum / static_cast<scalar_t>(node_count) : scalar_t(0);
        node_unweighted_loss[0] = node_value;
        node_weighted_loss[0]   = node_lambda * node_value;
    }

    template<typename scalar_t>
    __global__ void background_in_track_node_semantic_two_node_backward_kernel(
        const int n_background_points,
        const int n_background_blocks,
        const scalar_t *__restrict__ background_density_logits,
        scalar_t *__restrict__ grad_background_density_logits,
        const int n_other_node_points,
        const scalar_t *__restrict__ other_node_density_logits,
        scalar_t *__restrict__ grad_other_node_density_logits,
        const uint8_t *__restrict__ selection_bits,
        const BgTrackNodeSemanticWorkspace<scalar_t> *__restrict__ workspace,
        const scalar_t *__restrict__ grad_background_unweighted,
        const scalar_t *__restrict__ grad_background_weighted,
        const scalar_t *__restrict__ grad_node_unweighted,
        const scalar_t *__restrict__ grad_node_weighted,
        const scalar_t background_lambda,
        const scalar_t node_lambda
    )
    {
        const bool is_background = static_cast<int>(blockIdx.x) < n_background_blocks;
        const int node_block
            = is_background ? static_cast<int>(blockIdx.x) : static_cast<int>(blockIdx.x) - n_background_blocks;
        const int point_idx = node_block * blockDim.x + threadIdx.x;
        const int n_points  = is_background ? n_background_points : n_other_node_points;
        if(point_idx >= n_points)
        {
            return;
        }

        scalar_t *const node_gradient = is_background ? grad_background_density_logits : grad_other_node_density_logits;
        if(node_gradient == nullptr)
        {
            return;
        }

        const uint8_t selected = selection_bits[is_background ? point_idx : n_background_points + point_idx];
        scalar_t contribution  = scalar_t(0);
        if(background_lambda >= scalar_t(0)
           && is_background
           && (selected & BACKGROUND_IN_TRACK_SELECTION_BIT) != 0
           && workspace->background_selected_count > 0)
        {
            const scalar_t background_upstream
                = (grad_background_unweighted != nullptr ? grad_background_unweighted[0] : scalar_t(0))
                + background_lambda * (grad_background_weighted != nullptr ? grad_background_weighted[0] : scalar_t(0));
            contribution += background_upstream / static_cast<scalar_t>(workspace->background_selected_count);
        }
        if((selected & NODE_SEMANTIC_SELECTION_BIT) != 0 && workspace->node_selected_count > 0)
        {
            const scalar_t node_upstream
                = (grad_node_unweighted != nullptr ? grad_node_unweighted[0] : scalar_t(0))
                + node_lambda * (grad_node_weighted != nullptr ? grad_node_weighted[0] : scalar_t(0));
            contribution += node_upstream / static_cast<scalar_t>(workspace->node_selected_count);
        }

        // Points with no upstream contribution (unselected, disabled member,
        // or zero-count) get an exact zero without their logit being read:
        // sigmoid(non-finite) * 0 would otherwise poison the gradient of a
        // point that never contributed to the loss.
        if(contribution == scalar_t(0))
        {
            node_gradient[point_idx] = scalar_t(0);
            return;
        }
        const scalar_t density_logit
            = is_background ? background_density_logits[point_idx] : other_node_density_logits[point_idx];
        node_gradient[point_idx] = bg_track_node_semantic::stable_sigmoid(density_logit) * contribution;
    }

    // ------------------------------------------------------------------------
    // Host helpers
    // ------------------------------------------------------------------------

    template<typename scalar_t, typename SemanticFilter>
    void launch_background_in_track_forward(
        const int n_launch_points,
        const int n_tracks,
        const int n_semantic_classes,
        const scalar_t density_logits_min,
        const scalar_t *positions,
        const scalar_t *density_logits,
        const scalar_t *semantic_logits,
        const scalar_t *track_boxes,
        uint8_t *selection_bits,
        int32_t *selected_count,
        scalar_t *loss_sum,
        const SemanticFilter semantic_filter,
        const cudaStream_t stream
    )
    {
        if(n_launch_points == 0)
        {
            return;
        }
        const dim3 threads(BLOCK_THREADS);
        const dim3 blocks(div_round_up(n_launch_points, BLOCK_THREADS));
        background_in_track_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            n_launch_points,
            n_tracks,
            n_semantic_classes,
            density_logits_min,
            positions,
            density_logits,
            semantic_logits,
            track_boxes,
            selection_bits,
            selected_count,
            loss_sum,
            semantic_filter
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Multi-word semantic class masks: SEMANTIC_CLASS_BITS = 5, so class IDs
    // beyond 31 land in subsequent uint32 mask words. Out-of-range IDs are
    // ignored. class_ids is a non-owning view into the op's argument storage
    // (see the segment aliases in BgTrackNodeSemanticLosses.h); it is fully
    // consumed here, synchronously on the host, before any launch returns.
    std::vector<uint32_t> make_semantic_class_masks(
        const at::IntArrayRef class_ids, const int n_semantic_classes, const size_t mask_word_count
    )
    {
        std::vector<uint32_t> class_masks(mask_word_count, uint32_t{0});
        for(const int64_t class_id: class_ids)
        {
            if(class_id < 0 || class_id >= n_semantic_classes)
            {
                continue;
            }
            class_masks[static_cast<size_t>(class_id) >> SEMANTIC_CLASS_BITS] |= uint32_t{1}
                                                                              << (class_id & SEMANTIC_CLASS_MASK);
        }
        return class_masks;
    }

    template<typename scalar_t>
    void check_workspace(const at::Tensor &workspace)
    {
        TORCH_CHECK(workspace.scalar_type() == at::kInt, "workspace must be int32");
        TORCH_CHECK(
            workspace.numel() == WORKSPACE_INT32_WORDS
                && reinterpret_cast<std::uintptr_t>(workspace.data_ptr<int32_t>())
                           % alignof(BgTrackNodeSemanticWorkspace<scalar_t>)
                       == 0,
            "workspace must be a 16-byte-aligned int32[",
            WORKSPACE_INT32_WORDS,
            "] tensor"
        );
    }

    // ------------------------------------------------------------------------
    // Generic segmented forward (background member alone)
    // ------------------------------------------------------------------------

    template<typename scalar_t>
    void background_in_track_loss_forward_generic(
        const scalar_t density_logits_min,
        const at::Tensor &positions,
        const at::Tensor &density_logits,
        const int64_t n_semantic_points_value,
        const std::vector<at::Tensor> &semantic_logits,
        const std::vector<BackgroundInTrackSemanticSegment> &semantic_segments,
        const at::Tensor &camera_timestamps_startend_us,
        const at::Tensor &tracks_packinfo,
        const at::Tensor &tracks_poses,
        const at::Tensor &tracks_timestamps_us,
        const at::Tensor &cuboids_dims,
        const scalar_t background_lambda,
        const at::Tensor &track_boxes,
        const at::Tensor &selection_bits,
        const at::Tensor &workspace,
        const at::Tensor &loss,
        const at::Tensor &weighted_loss
    )
    {
        CHECK_INPUT(positions);
        CHECK_INPUT(density_logits);
        CHECK_INPUT(camera_timestamps_startend_us);
        CHECK_INPUT(tracks_packinfo);
        CHECK_INPUT(tracks_poses);
        CHECK_INPUT(tracks_timestamps_us);
        CHECK_INPUT(cuboids_dims);
        CHECK_INPUT(track_boxes);
        CHECK_INPUT(selection_bits);
        CHECK_INPUT(workspace);
        CHECK_INPUT(loss);
        CHECK_INPUT(weighted_loss);

        const auto device = positions.device();
        check_same_device(device, density_logits, "density_logits");
        check_same_device(device, camera_timestamps_startend_us, "camera_timestamps_startend_us");
        check_same_device(device, tracks_packinfo, "tracks_packinfo");
        check_same_device(device, tracks_poses, "tracks_poses");
        check_same_device(device, tracks_timestamps_us, "tracks_timestamps_us");
        check_same_device(device, cuboids_dims, "cuboids_dims");
        check_same_device(device, track_boxes, "track_boxes");
        check_same_device(device, selection_bits, "selection_bits");
        check_same_device(device, workspace, "workspace");
        check_same_device(device, loss, "loss");
        check_same_device(device, weighted_loss, "weighted_loss");

        const auto dtype = density_logits.scalar_type();
        TORCH_CHECK(positions.scalar_type() == dtype, "positions must match density_logits dtype");
        TORCH_CHECK(camera_timestamps_startend_us.scalar_type() == at::kLong, "camera timestamps must be int64");
        TORCH_CHECK(tracks_packinfo.scalar_type() == at::kInt, "tracks_packinfo must be int32");
        TORCH_CHECK(tracks_poses.scalar_type() == dtype, "tracks_poses must match density_logits dtype");
        TORCH_CHECK(tracks_timestamps_us.scalar_type() == at::kLong, "track timestamps must be int64");
        TORCH_CHECK(cuboids_dims.scalar_type() == dtype, "cuboids_dims must match density_logits dtype");
        TORCH_CHECK(track_boxes.scalar_type() == dtype, "track_boxes must match density_logits dtype");
        TORCH_CHECK(selection_bits.scalar_type() == at::kByte, "selection_bits must be uint8");
        TORCH_CHECK(loss.scalar_type() == dtype, "loss must match density_logits dtype");
        TORCH_CHECK(weighted_loss.scalar_type() == dtype, "weighted_loss must match density_logits dtype");
        check_workspace<scalar_t>(workspace);

        TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3, "positions must have shape [N,3]");
        TORCH_CHECK(density_logits.dim() == 1, "density_logits must have shape [N]");
        TORCH_CHECK(
            tracks_packinfo.dim() == 2 && tracks_packinfo.size(1) == 2, "tracks_packinfo must have shape [T,2]"
        );
        TORCH_CHECK(tracks_poses.dim() == 2 && tracks_poses.size(1) == 7, "tracks_poses must have shape [P,7]");

        TORCH_CHECK(
            n_semantic_points_value >= 0 && n_semantic_points_value <= std::numeric_limits<int>::max(),
            "n_semantic_points must fit a nonnegative int"
        );
        TORCH_CHECK(positions.size(0) <= std::numeric_limits<int>::max(), "positions contains too many points");
        TORCH_CHECK(
            tracks_packinfo.size(0) <= std::numeric_limits<int>::max(), "tracks_packinfo contains too many tracks"
        );
        TORCH_CHECK(tracks_poses.size(0) <= std::numeric_limits<int>::max(), "tracks_poses contains too many poses");

        const int n_points          = static_cast<int>(positions.size(0));
        const int n_tracks          = static_cast<int>(tracks_packinfo.size(0));
        const int n_total_poses     = static_cast<int>(tracks_poses.size(0));
        const int n_semantic_points = static_cast<int>(n_semantic_points_value);

        TORCH_CHECK(density_logits.size(0) == n_points, "density_logits must have shape [N]");
        TORCH_CHECK(n_semantic_points <= n_points, "semantic_logits cannot contain more rows than positions");
        TORCH_CHECK(
            semantic_logits.size() == semantic_segments.size(),
            "semantic tensors and segments must have the same length"
        );
        // Only row 0 (the reference camera) is consumed by the interpolation
        // kernel; extra rows are accepted for caller convenience but ignored,
        // matching the pure-PyTorch fallback. B == 0 would make the kernel
        // read out of bounds, so reject it loudly here.
        TORCH_CHECK(
            camera_timestamps_startend_us.dim() == 2
                && camera_timestamps_startend_us.size(0) >= 1
                && camera_timestamps_startend_us.size(1) == 2,
            "camera_timestamps_startend_us must have shape [B,2] with B >= 1 "
            "(B == 0 is rejected); only row 0 (the reference camera) is read"
        );
        TORCH_CHECK(
            tracks_timestamps_us.dim() == 1 && tracks_timestamps_us.size(0) == n_total_poses,
            "tracks_timestamps_us must have shape [P]"
        );
        TORCH_CHECK(
            cuboids_dims.dim() == 2 && cuboids_dims.size(0) == n_tracks && cuboids_dims.size(1) == 3,
            "cuboids_dims must have shape [T,3]"
        );
        TORCH_CHECK(
            track_boxes.dim() == 2 && track_boxes.size(0) == n_tracks && track_boxes.size(1) == TRACK_BOX_VALUES,
            "track_boxes must have shape [T,16]"
        );
        TORCH_CHECK(
            reinterpret_cast<std::uintptr_t>(track_boxes.data_ptr<scalar_t>()) % 16 == 0,
            "track_boxes must be 16-byte aligned"
        );
        TORCH_CHECK(
            selection_bits.dim() == 1 && selection_bits.size(0) >= n_points,
            "selection_bits must have at least N elements"
        );
        TORCH_CHECK(loss.numel() == 1, "loss must contain one element");
        TORCH_CHECK(weighted_loss.numel() == 1, "weighted_loss must contain one element");

        if(n_semantic_points == 0)
        {
            TORCH_CHECK(semantic_segments.empty(), "semantic segments must be empty when n_semantic_points is zero");
        }
        else
        {
            TORCH_CHECK(!semantic_segments.empty(), "semantic segments are required when semantic logits are present");
        }
        int64_t previous_segment_end = 0;
        for(size_t segment_idx = 0; segment_idx < semantic_segments.size(); ++segment_idx)
        {
            const int64_t segment_end          = semantic_segments[segment_idx].first;
            const at::Tensor &segment_semantic = semantic_logits[segment_idx];
            CHECK_INPUT(segment_semantic);
            check_same_device(device, segment_semantic, "semantic_logits");
            TORCH_CHECK(
                segment_end >= previous_segment_end && segment_end <= n_semantic_points,
                "semantic segment ends must be nondecreasing and bounded by n_semantic_points"
            );
            TORCH_CHECK(
                segment_semantic.scalar_type() == dtype
                    && segment_semantic.dim() == 2
                    && segment_semantic.size(0) == segment_end - previous_segment_end
                    && segment_semantic.size(1) > 0
                    && segment_semantic.size(1) <= std::numeric_limits<int>::max(),
                "each semantic tensor must be a [Nsegment,C] tensor matching the density dtype, 1 <= C <= INT_MAX"
            );
            previous_segment_end = segment_end;
        }
        TORCH_CHECK(
            previous_segment_end == n_semantic_points, "the final semantic segment end must equal n_semantic_points"
        );

        DEVICE_GUARD(positions);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        auto *const workspace_ptr
            = reinterpret_cast<BgTrackNodeSemanticWorkspace<scalar_t> *>(workspace.data_ptr<int32_t>());
        int32_t *const selected_count_ptr = &workspace_ptr->background_selected_count;
        C10_CUDA_CHECK(cudaMemsetAsync(workspace_ptr, 0, sizeof(BgTrackNodeSemanticWorkspace<scalar_t>), stream));
        if(selection_bits.numel() > 0)
        {
            C10_CUDA_CHECK(cudaMemsetAsync(selection_bits.data_ptr<uint8_t>(), 0, selection_bits.numel(), stream));
        }

        if(n_points == 0 || n_tracks == 0)
        {
            finalize_background_in_track_loss_kernel<scalar_t><<<1, 1, 0, stream>>>(
                workspace_ptr, background_lambda, loss.data_ptr<scalar_t>(), weighted_loss.data_ptr<scalar_t>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }

        const dim3 threads(BLOCK_THREADS);
        const dim3 track_blocks(div_round_up(n_tracks, BLOCK_THREADS));
        // tracks_packinfo is trusted input: its shape/dtype are validated
        // above, but its (start, count) contents are device-resident gather
        // indices — checking them here would need a synchronizing D2H copy.
        // The kernel instead range-guards every row with CUDA_KERNEL_ASSERT
        // (start >= 0 && start + count <= P).
        interpolate_track_boxes_kernel<scalar_t><<<track_blocks, threads, 0, stream>>>(
            n_tracks,
            n_total_poses,
            camera_timestamps_startend_us.data_ptr<int64_t>(),
            tracks_packinfo.data_ptr<int32_t>(),
            tracks_poses.data_ptr<scalar_t>(),
            tracks_timestamps_us.data_ptr<int64_t>(),
            cuboids_dims.data_ptr<scalar_t>(),
            track_boxes.data_ptr<scalar_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const scalar_t *const positions_ptr      = positions.data_ptr<scalar_t>();
        const scalar_t *const density_logits_ptr = density_logits.data_ptr<scalar_t>();
        const scalar_t *const track_boxes_ptr    = track_boxes.data_ptr<scalar_t>();
        uint8_t *const selection_bits_ptr        = selection_bits.data_ptr<uint8_t>();
        scalar_t *const loss_ptr                 = &workspace_ptr->background_loss_sum;

        int point_begin = 0;
        for(size_t segment_idx = 0; segment_idx < semantic_segments.size(); ++segment_idx)
        {
            const int point_end                = static_cast<int>(semantic_segments[segment_idx].first);
            const int n_segment_points         = point_end - point_begin;
            const at::Tensor &segment_semantic = semantic_logits[segment_idx];
            const int n_semantic_classes       = static_cast<int>(segment_semantic.size(1));
            const size_t mask_word_count
                = (static_cast<size_t>(n_semantic_classes) + SEMANTIC_CLASS_COUNT - 1) / SEMANTIC_CLASS_COUNT;
            const std::vector<uint32_t> class_masks
                = make_semantic_class_masks(semantic_segments[segment_idx].second, n_semantic_classes, mask_word_count);
            for(size_t mask_word = 0; mask_word < mask_word_count; ++mask_word)
            {
                if(class_masks[mask_word] == 0)
                {
                    continue;
                }
                const RangeSemanticFilter semantic_filter{
                    0,
                    static_cast<int32_t>(mask_word),
                    class_masks[mask_word],
                };
                launch_background_in_track_forward(
                    n_segment_points,
                    n_tracks,
                    n_semantic_classes,
                    density_logits_min,
                    positions_ptr + static_cast<int64_t>(point_begin) * 3,
                    density_logits_ptr + point_begin,
                    segment_semantic.data_ptr<scalar_t>(),
                    track_boxes_ptr,
                    selection_bits_ptr + point_begin,
                    selected_count_ptr,
                    loss_ptr,
                    semantic_filter,
                    stream
                );
            }
            point_begin = point_end;
        }
        if(n_points > n_semantic_points)
        {
            const UnfilteredRange unfiltered_suffix{0};
            launch_background_in_track_forward(
                n_points - n_semantic_points,
                n_tracks,
                1,
                density_logits_min,
                positions_ptr + static_cast<int64_t>(n_semantic_points) * 3,
                density_logits_ptr + n_semantic_points,
                static_cast<const scalar_t *>(nullptr),
                track_boxes_ptr,
                selection_bits_ptr + n_semantic_points,
                selected_count_ptr,
                loss_ptr,
                unfiltered_suffix,
                stream
            );
        }

        finalize_background_in_track_loss_kernel<scalar_t><<<1, 1, 0, stream>>>(
            workspace_ptr, background_lambda, loss.data_ptr<scalar_t>(), weighted_loss.data_ptr<scalar_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ------------------------------------------------------------------------
    // Generic backward (background member alone)
    // ------------------------------------------------------------------------

    template<typename scalar_t>
    void background_in_track_loss_backward_generic(
        const at::Tensor &density_logits,
        const at::Tensor &selection_bits,
        const at::Tensor &workspace,
        const std::optional<at::Tensor> &grad_unweighted_loss,
        const std::optional<at::Tensor> &grad_weighted_loss,
        const scalar_t background_lambda,
        const at::Tensor &grad_density_logits
    )
    {
        CHECK_INPUT(density_logits);
        CHECK_INPUT(selection_bits);
        CHECK_INPUT(workspace);
        CHECK_INPUT(grad_density_logits);

        const auto device = density_logits.device();
        check_same_device(device, selection_bits, "selection_bits");
        check_same_device(device, workspace, "workspace");
        check_same_device(device, grad_density_logits, "grad_density_logits");
        const auto dtype = density_logits.scalar_type();
        TORCH_CHECK(selection_bits.scalar_type() == at::kByte, "selection_bits must be uint8");
        check_workspace<scalar_t>(workspace);
        TORCH_CHECK(grad_density_logits.scalar_type() == dtype, "grad_density_logits must match density_logits dtype");
        TORCH_CHECK(density_logits.dim() == 1, "density_logits must be 1-D");
        // Mirror the generic forward's windowed acceptance (>= N, background
        // bits in the leading N entries): the autograd wrapper allocates one
        // shared uint8 buffer sized Nbackground + Nother even when only the
        // background member is enabled, and that buffer must round-trip into
        // this backward unchanged.
        TORCH_CHECK(
            selection_bits.dim() == 1 && selection_bits.size(0) >= density_logits.numel(),
            "selection_bits must have at least N elements"
        );
        TORCH_CHECK(
            grad_density_logits.sizes() == density_logits.sizes(), "grad_density_logits must match density_logits"
        );
        const auto validate_upstream = [&](const std::optional<at::Tensor> &gradient, const char *name)
        {
            if(!gradient.has_value())
            {
                return;
            }
            CHECK_INPUT(gradient.value());
            check_same_device(device, gradient.value(), name);
            TORCH_CHECK(
                gradient->scalar_type() == dtype && gradient->numel() == 1,
                name,
                " must be a scalar matching the density dtype"
            );
        };
        validate_upstream(grad_unweighted_loss, "grad_unweighted_loss");
        validate_upstream(grad_weighted_loss, "grad_weighted_loss");
        TORCH_CHECK(density_logits.numel() <= std::numeric_limits<int>::max(), "too many density points");

        const int n_points = static_cast<int>(density_logits.numel());
        if(n_points == 0)
        {
            return;
        }

        DEVICE_GUARD(density_logits);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const dim3 threads(BLOCK_THREADS);
        const dim3 blocks(div_round_up(n_points, BLOCK_THREADS));
        const auto *const workspace_ptr
            = reinterpret_cast<const BgTrackNodeSemanticWorkspace<scalar_t> *>(workspace.data_ptr<int32_t>());
        background_in_track_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            n_points,
            density_logits.data_ptr<scalar_t>(),
            selection_bits.data_ptr<uint8_t>(),
            &workspace_ptr->background_selected_count,
            grad_unweighted_loss.has_value() ? grad_unweighted_loss->data_ptr<scalar_t>() : nullptr,
            grad_weighted_loss.has_value() ? grad_weighted_loss->data_ptr<scalar_t>() : nullptr,
            background_lambda,
            grad_density_logits.data_ptr<scalar_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ------------------------------------------------------------------------
    // Packed node-semantic forward (independent point domain)
    // ------------------------------------------------------------------------

    template<typename scalar_t>
    void node_semantic_loss_forward_packed(
        const int64_t selection_offset,
        const at::Tensor &density_logits,
        const std::vector<at::Tensor> &semantic_logits,
        const std::vector<NodeSemanticSegment> &segments,
        const scalar_t node_lambda,
        const at::Tensor &selection_bits,
        const at::Tensor &workspace,
        const at::Tensor &unweighted_loss,
        const at::Tensor &weighted_loss
    )
    {
        CHECK_INPUT(density_logits);
        CHECK_INPUT(selection_bits);
        CHECK_INPUT(workspace);
        CHECK_INPUT(unweighted_loss);
        CHECK_INPUT(weighted_loss);

        const at::Device device = density_logits.device();
        check_same_device(device, selection_bits, "selection_bits");
        check_same_device(device, workspace, "workspace");
        check_same_device(device, unweighted_loss, "unweighted_loss");
        check_same_device(device, weighted_loss, "weighted_loss");
        const auto dtype = density_logits.scalar_type();
        TORCH_CHECK(density_logits.dim() == 1, "packed node density logits must be a [N] tensor");
        TORCH_CHECK(
            semantic_logits.size() == segments.size(), "node semantic tensors and segments must have the same length"
        );
        TORCH_CHECK(
            selection_offset >= 0 && selection_offset + density_logits.numel() == selection_bits.numel(),
            "packed node selection range must end at the selection buffer boundary"
        );
        TORCH_CHECK(
            selection_bits.scalar_type() == at::kByte && selection_bits.dim() == 1,
            "selection_bits must be uint8 [Nbackground + Nnode]"
        );
        check_workspace<scalar_t>(workspace);
        TORCH_CHECK(
            unweighted_loss.scalar_type() == dtype
                && weighted_loss.scalar_type() == dtype
                && unweighted_loss.numel() == 1
                && weighted_loss.numel() == 1,
            "node semantic outputs must be scalars matching the density dtype"
        );
        TORCH_CHECK(
            density_logits.numel() <= std::numeric_limits<int32_t>::max(),
            "packed node density contains too many points"
        );

        int64_t point_begin = 0;
        for(size_t segment_idx = 0; segment_idx < segments.size(); ++segment_idx)
        {
            const at::Tensor &segment_semantic = semantic_logits[segment_idx];
            CHECK_INPUT(segment_semantic);
            check_same_device(device, segment_semantic, "node semantic logits");
            const int64_t point_end = std::get<0>(segments[segment_idx]);
            TORCH_CHECK(
                point_end >= point_begin && point_end <= density_logits.numel(),
                "node semantic segment endpoints must be ordered and in range"
            );
            TORCH_CHECK(
                segment_semantic.scalar_type() == dtype
                    && segment_semantic.dim() == 2
                    && segment_semantic.size(0) == point_end - point_begin
                    && segment_semantic.size(1) > 0
                    && segment_semantic.size(1) <= std::numeric_limits<int>::max(),
                "each node semantic tensor must be a [Nsegment,C] tensor matching the density dtype, 1 <= C <= INT_MAX"
            );
            point_begin = point_end;
        }
        TORCH_CHECK(
            point_begin == density_logits.numel(), "node semantic segments must cover every packed density row"
        );

        DEVICE_GUARD(density_logits);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        auto *const workspace_ptr
            = reinterpret_cast<BgTrackNodeSemanticWorkspace<scalar_t> *>(workspace.data_ptr<int32_t>());

        const dim3 threads(BLOCK_THREADS);
        point_begin = 0;
        for(size_t segment_idx = 0; segment_idx < segments.size(); ++segment_idx)
        {
            const int64_t point_end            = std::get<0>(segments[segment_idx]);
            const int n_segment_points         = static_cast<int>(point_end - point_begin);
            const at::Tensor &segment_semantic = semantic_logits[segment_idx];
            const int n_semantic_classes       = static_cast<int>(segment_semantic.size(1));
            const size_t mask_word_count
                = (static_cast<size_t>(n_semantic_classes) + SEMANTIC_CLASS_COUNT - 1) / SEMANTIC_CLASS_COUNT;
            const std::vector<uint32_t> class_masks
                = make_semantic_class_masks(std::get<1>(segments[segment_idx]), n_semantic_classes, mask_word_count);
            bool multiword_semantics = false;
            for(size_t mask_word = 1; mask_word < mask_word_count; ++mask_word)
            {
                multiword_semantics |= class_masks[mask_word] != 0;
            }

            if(n_segment_points > 0)
            {
                const dim3 blocks(div_round_up(n_segment_points, BLOCK_THREADS));
                const scalar_t *const segment_density = density_logits.data_ptr<scalar_t>() + point_begin;
                uint8_t *const segment_selection = selection_bits.data_ptr<uint8_t>() + selection_offset + point_begin;
                if(!multiword_semantics)
                {
                    node_semantic_forward_kernel<scalar_t, false><<<blocks, threads, 0, stream>>>(
                        n_segment_points,
                        n_semantic_classes,
                        0,
                        segment_density,
                        segment_semantic.data_ptr<scalar_t>(),
                        class_masks[0],
                        std::get<2>(segments[segment_idx]),
                        segment_selection,
                        workspace_ptr
                    );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }
                else
                {
                    for(size_t mask_word = 0; mask_word < mask_word_count; ++mask_word)
                    {
                        node_semantic_forward_kernel<scalar_t, true><<<blocks, threads, 0, stream>>>(
                            n_segment_points,
                            n_semantic_classes,
                            static_cast<int>(mask_word),
                            segment_density,
                            segment_semantic.data_ptr<scalar_t>(),
                            class_masks[mask_word],
                            std::get<2>(segments[segment_idx]),
                            segment_selection,
                            workspace_ptr
                        );
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    }
                }
            }
            point_begin = point_end;
        }

        finalize_node_semantic_loss_kernel<scalar_t><<<1, 1, 0, stream>>>(
            workspace_ptr, node_lambda, unweighted_loss.data_ptr<scalar_t>(), weighted_loss.data_ptr<scalar_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ------------------------------------------------------------------------
    // Joint forward (shared primary domain)
    // ------------------------------------------------------------------------

    template<typename scalar_t>
    void background_in_track_loss_forward_joint(
        const at::Tensor &bg_positions,
        const at::Tensor &bg_density_logits,
        const at::Tensor &bg_semantic_logits,
        const at::Tensor &other_density_logits,
        const std::vector<at::Tensor> &other_semantic_logits,
        const std::vector<NodeSemanticSegment> &other_segments,
        const at::Tensor &camera_timestamps_startend_us,
        const at::Tensor &tracks_packinfo,
        const at::Tensor &tracks_poses,
        const at::Tensor &tracks_timestamps_us,
        const at::Tensor &cuboids_dims,
        const at::IntArrayRef bg_allowed_class_ids,
        const at::IntArrayRef node_primary_class_ids,
        const bool node_primary_select,
        const scalar_t density_logits_min,
        const scalar_t bg_lambda,
        const scalar_t node_lambda,
        const at::Tensor &track_boxes,
        const at::Tensor &selection_bits,
        const at::Tensor &workspace,
        const at::Tensor &bg_unweighted_loss,
        const at::Tensor &bg_weighted_loss,
        const at::Tensor &node_unweighted_loss,
        const at::Tensor &node_weighted_loss
    )
    {
        const bool background_in_track_enabled = bg_lambda >= scalar_t(0);
        CHECK_INPUT(bg_positions);
        CHECK_INPUT(bg_density_logits);
        CHECK_INPUT(bg_semantic_logits);
        CHECK_INPUT(other_density_logits);
        CHECK_INPUT(camera_timestamps_startend_us);
        CHECK_INPUT(tracks_packinfo);
        CHECK_INPUT(tracks_poses);
        CHECK_INPUT(tracks_timestamps_us);
        CHECK_INPUT(cuboids_dims);
        CHECK_INPUT(track_boxes);
        CHECK_INPUT(selection_bits);
        CHECK_INPUT(workspace);
        CHECK_INPUT(bg_unweighted_loss);
        CHECK_INPUT(bg_weighted_loss);
        CHECK_INPUT(node_unweighted_loss);
        CHECK_INPUT(node_weighted_loss);

        const at::Device device = bg_density_logits.device();
        check_same_device(device, bg_positions, "bg_positions");
        check_same_device(device, bg_semantic_logits, "bg_semantic_logits");
        check_same_device(device, other_density_logits, "other_density_logits");
        if(background_in_track_enabled)
        {
            check_same_device(device, camera_timestamps_startend_us, "camera_timestamps_startend_us");
            check_same_device(device, tracks_packinfo, "tracks_packinfo");
            check_same_device(device, tracks_poses, "tracks_poses");
            check_same_device(device, tracks_timestamps_us, "tracks_timestamps_us");
            check_same_device(device, cuboids_dims, "cuboids_dims");
        }
        check_same_device(device, track_boxes, "track_boxes");
        check_same_device(device, selection_bits, "selection_bits");
        check_same_device(device, workspace, "workspace");
        check_same_device(device, bg_unweighted_loss, "bg_unweighted_loss");
        check_same_device(device, bg_weighted_loss, "bg_weighted_loss");
        check_same_device(device, node_unweighted_loss, "node_unweighted_loss");
        check_same_device(device, node_weighted_loss, "node_weighted_loss");

        const auto dtype = bg_density_logits.scalar_type();
        TORCH_CHECK(bg_positions.scalar_type() == dtype, "bg_positions must match the density dtype");
        TORCH_CHECK(bg_semantic_logits.scalar_type() == dtype, "bg_semantic_logits must match the density dtype");
        TORCH_CHECK(other_density_logits.scalar_type() == dtype, "other_density_logits must match the density dtype");
        TORCH_CHECK(
            camera_timestamps_startend_us.scalar_type() == at::kLong, "camera_timestamps_startend_us must be int64"
        );
        TORCH_CHECK(tracks_packinfo.scalar_type() == at::kInt, "tracks_packinfo must be int32");
        TORCH_CHECK(tracks_poses.scalar_type() == dtype, "tracks_poses must match the density dtype");
        TORCH_CHECK(tracks_timestamps_us.scalar_type() == at::kLong, "tracks_timestamps_us must be int64");
        TORCH_CHECK(cuboids_dims.scalar_type() == dtype, "cuboids_dims must match the density dtype");
        TORCH_CHECK(track_boxes.scalar_type() == dtype, "track_boxes must match the density dtype");
        TORCH_CHECK(selection_bits.scalar_type() == at::kByte, "selection_bits must be uint8");

        TORCH_CHECK(bg_positions.dim() == 2 && bg_positions.size(1) == 3, "bg_positions must have shape [Nbg,3]");
        TORCH_CHECK(bg_density_logits.dim() == 1, "bg_density_logits must have shape [Nbg]");
        TORCH_CHECK(
            bg_semantic_logits.dim() == 2
                && bg_semantic_logits.size(1) > 0
                && bg_semantic_logits.size(1) <= std::numeric_limits<int>::max(),
            "bg_semantic_logits must have shape [Nbg,Cbg], 1 <= Cbg <= INT_MAX"
        );
        TORCH_CHECK(other_density_logits.dim() == 1, "other_density_logits must have shape [Nother]");
        TORCH_CHECK(
            bg_positions.size(0) == bg_density_logits.size(0)
                && bg_semantic_logits.size(0) == bg_density_logits.size(0),
            "background positions, densities, and semantics must have matching rows"
        );
        TORCH_CHECK(
            other_semantic_logits.size() == other_segments.size(),
            "other semantic tensors and segments must have the same length"
        );
        int64_t other_point_begin = 0;
        for(size_t segment_idx = 0; segment_idx < other_segments.size(); ++segment_idx)
        {
            const at::Tensor &segment_semantic = other_semantic_logits[segment_idx];
            CHECK_INPUT(segment_semantic);
            check_same_device(device, segment_semantic, "other_semantic_logits");
            const int64_t point_end = std::get<0>(other_segments[segment_idx]);
            TORCH_CHECK(
                point_end >= other_point_begin && point_end <= other_density_logits.numel(),
                "other node segment endpoints must be ordered and in range"
            );
            TORCH_CHECK(
                segment_semantic.scalar_type() == dtype
                    && segment_semantic.dim() == 2
                    && segment_semantic.size(0) == point_end - other_point_begin
                    && segment_semantic.size(1) > 0
                    && segment_semantic.size(1) <= std::numeric_limits<int>::max(),
                "each other semantic tensor must be a [Nsegment,C] tensor matching the density dtype, 1 <= C <= "
                "INT_MAX"
            );
            other_point_begin = point_end;
        }
        TORCH_CHECK(
            other_point_begin == other_density_logits.numel(), "other node segments must cover every packed density row"
        );
        // Row-0 (reference camera) semantics; see the matching check in
        // background_in_track_loss_forward_generic.
        TORCH_CHECK(
            camera_timestamps_startend_us.dim() == 2
                && camera_timestamps_startend_us.size(0) >= 1
                && camera_timestamps_startend_us.size(1) == 2,
            "camera_timestamps_startend_us must have shape [B,2] with B >= 1 "
            "(B == 0 is rejected); only row 0 (the reference camera) is read"
        );
        TORCH_CHECK(
            tracks_packinfo.dim() == 2 && tracks_packinfo.size(1) == 2, "tracks_packinfo must have shape [T,2]"
        );
        TORCH_CHECK(tracks_poses.dim() == 2 && tracks_poses.size(1) == 7, "tracks_poses must have shape [P,7]");
        TORCH_CHECK(
            tracks_timestamps_us.dim() == 1 && tracks_timestamps_us.size(0) == tracks_poses.size(0),
            "tracks_timestamps_us must have shape [P]"
        );
        TORCH_CHECK(
            cuboids_dims.dim() == 2 && cuboids_dims.size(0) == tracks_packinfo.size(0) && cuboids_dims.size(1) == 3,
            "cuboids_dims must have shape [T,3]"
        );
        TORCH_CHECK(
            track_boxes.dim() == 2
                && track_boxes.size(0) == tracks_packinfo.size(0)
                && track_boxes.size(1) == TRACK_BOX_VALUES,
            "track_boxes must have shape [T,16]"
        );
        TORCH_CHECK(
            reinterpret_cast<std::uintptr_t>(track_boxes.data_ptr<scalar_t>()) % 16 == 0,
            "track_boxes must be 16-byte aligned"
        );

        TORCH_CHECK(
            bg_density_logits.numel() <= std::numeric_limits<int>::max(), "bg_density_logits contains too many points"
        );
        TORCH_CHECK(
            other_density_logits.numel() <= std::numeric_limits<int>::max(),
            "other_density_logits contains too many points"
        );
        TORCH_CHECK(
            tracks_packinfo.size(0) <= std::numeric_limits<int>::max(), "tracks_packinfo contains too many tracks"
        );
        TORCH_CHECK(tracks_poses.size(0) <= std::numeric_limits<int>::max(), "tracks_poses contains too many poses");
        const int64_t total_points = bg_density_logits.numel() + other_density_logits.numel();
        TORCH_CHECK(
            total_points <= std::numeric_limits<int32_t>::max(), "the combined selected-point count must fit int32"
        );
        TORCH_CHECK(
            selection_bits.dim() == 1 && selection_bits.numel() == total_points,
            "selection_bits must contain Nbg + Nother elements"
        );
        check_workspace<scalar_t>(workspace);

        const auto validate_output = [&dtype](const at::Tensor &output, const char *name)
        {
            TORCH_CHECK(
                output.scalar_type() == dtype && output.numel() == 1,
                name,
                " must be a scalar matching the density dtype"
            );
        };
        validate_output(bg_unweighted_loss, "bg_unweighted_loss");
        validate_output(bg_weighted_loss, "bg_weighted_loss");
        validate_output(node_unweighted_loss, "node_unweighted_loss");
        validate_output(node_weighted_loss, "node_weighted_loss");

        const int n_background_points           = static_cast<int>(bg_density_logits.numel());
        const int n_background_semantic_classes = static_cast<int>(bg_semantic_logits.size(1));
        const int n_first_other_points
            = other_segments.empty() ? 0 : static_cast<int>(std::get<0>(other_segments.front()));
        const int n_first_other_semantic_classes
            = other_segments.empty() ? 1 : static_cast<int>(other_semantic_logits.front().size(1));
        const int n_tracks = static_cast<int>(tracks_packinfo.size(0));
        const size_t mask_word_count
            = (static_cast<size_t>(std::max(n_background_semantic_classes, n_first_other_semantic_classes))
               + SEMANTIC_CLASS_COUNT
               - 1)
            / SEMANTIC_CLASS_COUNT;
        const std::vector<uint32_t> bg_allowed_class_masks
            = make_semantic_class_masks(bg_allowed_class_ids, n_background_semantic_classes, mask_word_count);
        const std::vector<uint32_t> node_primary_class_masks
            = make_semantic_class_masks(node_primary_class_ids, n_background_semantic_classes, mask_word_count);
        const at::IntArrayRef first_other_class_ids
            = other_segments.empty() ? at::IntArrayRef{} : std::get<1>(other_segments.front());
        const bool first_other_select = !other_segments.empty() && std::get<2>(other_segments.front());
        const std::vector<uint32_t> first_other_class_masks
            = make_semantic_class_masks(first_other_class_ids, n_first_other_semantic_classes, mask_word_count);
        bool multiword_semantics = false;
        for(size_t mask_word = 1; mask_word < mask_word_count; ++mask_word)
        {
            multiword_semantics |= bg_allowed_class_masks[mask_word] != 0
                                || node_primary_class_masks[mask_word] != 0
                                || first_other_class_masks[mask_word] != 0;
        }
        const int n_background_blocks  = static_cast<int>(div_round_up(n_background_points, BLOCK_THREADS));
        const int n_first_other_blocks = static_cast<int>(div_round_up(n_first_other_points, BLOCK_THREADS));

        DEVICE_GUARD(bg_density_logits);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        auto *const workspace_ptr
            = reinterpret_cast<BgTrackNodeSemanticWorkspace<scalar_t> *>(workspace.data_ptr<int32_t>());
        C10_CUDA_CHECK(cudaMemsetAsync(workspace_ptr, 0, sizeof(BgTrackNodeSemanticWorkspace<scalar_t>), stream));

        const dim3 threads(BLOCK_THREADS);
        if(background_in_track_enabled && n_tracks > 0)
        {
            const dim3 track_blocks(div_round_up(n_tracks, BLOCK_THREADS));
            // tracks_packinfo contents are trusted device-side gather indices;
            // the kernel range-guards them (see the note at the generic-path
            // launch site).
            interpolate_track_boxes_kernel<scalar_t><<<track_blocks, threads, 0, stream>>>(
                n_tracks,
                static_cast<int>(tracks_poses.size(0)),
                camera_timestamps_startend_us.data_ptr<int64_t>(),
                tracks_packinfo.data_ptr<int32_t>(),
                tracks_poses.data_ptr<scalar_t>(),
                tracks_timestamps_us.data_ptr<int64_t>(),
                cuboids_dims.data_ptr<scalar_t>(),
                track_boxes.data_ptr<scalar_t>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        const scalar_t *const first_other_semantic_ptr = other_segments.empty()
                                                           ? bg_semantic_logits.data_ptr<scalar_t>()
                                                           : other_semantic_logits.front().data_ptr<scalar_t>();
        if(n_background_blocks + n_first_other_blocks > 0)
        {
            const dim3 blocks(n_background_blocks + n_first_other_blocks);
            if(!multiword_semantics)
            {
                background_in_track_node_semantic_two_node_forward_kernel<scalar_t, false>
                    <<<blocks, threads, 0, stream>>>(
                        n_background_points,
                        n_background_blocks,
                        n_background_semantic_classes,
                        n_first_other_points,
                        n_first_other_semantic_classes,
                        0,
                        n_tracks,
                        bg_lambda,
                        density_logits_min,
                        bg_positions.data_ptr<scalar_t>(),
                        bg_density_logits.data_ptr<scalar_t>(),
                        bg_semantic_logits.data_ptr<scalar_t>(),
                        other_density_logits.data_ptr<scalar_t>(),
                        first_other_semantic_ptr,
                        bg_allowed_class_masks[0],
                        node_primary_class_masks[0],
                        node_primary_select,
                        first_other_class_masks[0],
                        first_other_select,
                        track_boxes.data_ptr<scalar_t>(),
                        selection_bits.data_ptr<uint8_t>(),
                        workspace_ptr
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
            else
            {
                for(size_t mask_word = 0; mask_word < mask_word_count; ++mask_word)
                {
                    background_in_track_node_semantic_two_node_forward_kernel<scalar_t, true>
                        <<<blocks, threads, 0, stream>>>(
                            n_background_points,
                            n_background_blocks,
                            n_background_semantic_classes,
                            n_first_other_points,
                            n_first_other_semantic_classes,
                            static_cast<int>(mask_word),
                            n_tracks,
                            bg_lambda,
                            density_logits_min,
                            bg_positions.data_ptr<scalar_t>(),
                            bg_density_logits.data_ptr<scalar_t>(),
                            bg_semantic_logits.data_ptr<scalar_t>(),
                            other_density_logits.data_ptr<scalar_t>(),
                            first_other_semantic_ptr,
                            bg_allowed_class_masks[mask_word],
                            node_primary_class_masks[mask_word],
                            node_primary_select,
                            first_other_class_masks[mask_word],
                            first_other_select,
                            track_boxes.data_ptr<scalar_t>(),
                            selection_bits.data_ptr<uint8_t>(),
                            workspace_ptr
                        );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }
            }
        }

        int64_t point_begin = n_first_other_points;
        for(size_t segment_idx = 1; segment_idx < other_segments.size(); ++segment_idx)
        {
            const int64_t point_end            = std::get<0>(other_segments[segment_idx]);
            const int n_segment_points         = static_cast<int>(point_end - point_begin);
            const at::Tensor &segment_semantic = other_semantic_logits[segment_idx];
            const int n_semantic_classes       = static_cast<int>(segment_semantic.size(1));
            const size_t segment_mask_word_count
                = (static_cast<size_t>(n_semantic_classes) + SEMANTIC_CLASS_COUNT - 1) / SEMANTIC_CLASS_COUNT;
            const std::vector<uint32_t> segment_class_masks = make_semantic_class_masks(
                std::get<1>(other_segments[segment_idx]), n_semantic_classes, segment_mask_word_count
            );
            bool segment_multiword = false;
            for(size_t mask_word = 1; mask_word < segment_mask_word_count; ++mask_word)
            {
                segment_multiword |= segment_class_masks[mask_word] != 0;
            }

            if(n_segment_points > 0)
            {
                const dim3 segment_blocks(div_round_up(n_segment_points, BLOCK_THREADS));
                const scalar_t *const segment_density = other_density_logits.data_ptr<scalar_t>() + point_begin;
                uint8_t *const segment_selection
                    = selection_bits.data_ptr<uint8_t>() + n_background_points + point_begin;
                if(!segment_multiword)
                {
                    node_semantic_forward_kernel<scalar_t, false><<<segment_blocks, threads, 0, stream>>>(
                        n_segment_points,
                        n_semantic_classes,
                        0,
                        segment_density,
                        segment_semantic.data_ptr<scalar_t>(),
                        segment_class_masks[0],
                        std::get<2>(other_segments[segment_idx]),
                        segment_selection,
                        workspace_ptr
                    );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }
                else
                {
                    for(size_t mask_word = 0; mask_word < segment_mask_word_count; ++mask_word)
                    {
                        node_semantic_forward_kernel<scalar_t, true><<<segment_blocks, threads, 0, stream>>>(
                            n_segment_points,
                            n_semantic_classes,
                            static_cast<int>(mask_word),
                            segment_density,
                            segment_semantic.data_ptr<scalar_t>(),
                            segment_class_masks[mask_word],
                            std::get<2>(other_segments[segment_idx]),
                            segment_selection,
                            workspace_ptr
                        );
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    }
                }
            }
            point_begin = point_end;
        }

        finalize_background_in_track_node_semantic_two_node_kernel<scalar_t><<<1, 1, 0, stream>>>(
            workspace_ptr,
            bg_lambda,
            node_lambda,
            bg_unweighted_loss.data_ptr<scalar_t>(),
            bg_weighted_loss.data_ptr<scalar_t>(),
            node_unweighted_loss.data_ptr<scalar_t>(),
            node_weighted_loss.data_ptr<scalar_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ------------------------------------------------------------------------
    // Joint backward (also serves packed mode: bg bits first, node bits after)
    // ------------------------------------------------------------------------

    template<typename scalar_t>
    void background_in_track_loss_backward_joint(
        const at::Tensor &bg_density_logits,
        const at::Tensor &other_node_density_logits,
        const at::Tensor &selection_bits,
        const at::Tensor &workspace,
        const std::optional<at::Tensor> &grad_bg_unweighted,
        const std::optional<at::Tensor> &grad_bg_weighted,
        const std::optional<at::Tensor> &grad_node_unweighted,
        const std::optional<at::Tensor> &grad_node_weighted,
        const scalar_t bg_lambda,
        const scalar_t node_lambda,
        const std::optional<at::Tensor> &grad_bg_density_logits,
        const std::optional<at::Tensor> &grad_other_node_density_logits
    )
    {
        CHECK_INPUT(bg_density_logits);
        CHECK_INPUT(other_node_density_logits);
        CHECK_INPUT(selection_bits);
        CHECK_INPUT(workspace);
        const at::Device device = bg_density_logits.device();
        check_same_device(device, other_node_density_logits, "other_node_density_logits");
        check_same_device(device, selection_bits, "selection_bits");
        check_same_device(device, workspace, "workspace");

        const auto dtype = bg_density_logits.scalar_type();
        TORCH_CHECK(bg_density_logits.dim() == 1, "bg_density_logits must be a [Nbg] tensor");
        TORCH_CHECK(
            other_node_density_logits.scalar_type() == dtype && other_node_density_logits.dim() == 1,
            "other_node_density_logits must be a [Nother] tensor matching the density dtype"
        );
        const int64_t total_points = bg_density_logits.numel() + other_node_density_logits.numel();
        TORCH_CHECK(
            selection_bits.scalar_type() == at::kByte
                && selection_bits.dim() == 1
                && selection_bits.numel() == total_points,
            "selection_bits must be uint8 with Nbg + Nother elements"
        );
        check_workspace<scalar_t>(workspace);
        TORCH_CHECK(
            bg_density_logits.numel() <= std::numeric_limits<int>::max(), "bg_density_logits contains too many points"
        );
        TORCH_CHECK(
            other_node_density_logits.numel() <= std::numeric_limits<int>::max(),
            "other_node_density_logits contains too many points"
        );
        TORCH_CHECK(
            total_points <= std::numeric_limits<int32_t>::max(), "the combined selected-point count must fit int32"
        );

        const auto validate_output_gradient = [&](const std::optional<at::Tensor> &gradient, const char *name)
        {
            if(!gradient.has_value())
            {
                return;
            }
            CHECK_INPUT(gradient.value());
            check_same_device(device, gradient.value(), name);
            TORCH_CHECK(
                gradient->scalar_type() == dtype && gradient->numel() == 1,
                name,
                " must be a scalar matching the density dtype"
            );
        };
        validate_output_gradient(grad_bg_unweighted, "grad_bg_unweighted");
        validate_output_gradient(grad_bg_weighted, "grad_bg_weighted");
        validate_output_gradient(grad_node_unweighted, "grad_node_unweighted");
        validate_output_gradient(grad_node_weighted, "grad_node_weighted");

        const auto validate_density_gradient
            = [&](const std::optional<at::Tensor> &gradient, const at::Tensor &density, const char *name)
        {
            if(!gradient.has_value())
            {
                return;
            }
            CHECK_INPUT(gradient.value());
            check_same_device(device, gradient.value(), name);
            TORCH_CHECK(
                gradient->scalar_type() == dtype && gradient->sizes() == density.sizes(),
                name,
                " must match its density input's dtype and shape"
            );
        };
        validate_density_gradient(grad_bg_density_logits, bg_density_logits, "grad_bg_density_logits");
        validate_density_gradient(
            grad_other_node_density_logits, other_node_density_logits, "grad_other_node_density_logits"
        );

        const int n_background_points = static_cast<int>(bg_density_logits.numel());
        const int n_other_node_points = static_cast<int>(other_node_density_logits.numel());
        const int n_background_blocks = grad_bg_density_logits.has_value()
                                          ? static_cast<int>(div_round_up(n_background_points, BLOCK_THREADS))
                                          : 0;
        const int n_other_node_blocks = grad_other_node_density_logits.has_value()
                                          ? static_cast<int>(div_round_up(n_other_node_points, BLOCK_THREADS))
                                          : 0;
        if(n_background_blocks + n_other_node_blocks == 0)
        {
            return;
        }

        DEVICE_GUARD(bg_density_logits);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const dim3 threads(BLOCK_THREADS);
        const dim3 blocks(n_background_blocks + n_other_node_blocks);
        background_in_track_node_semantic_two_node_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            n_background_points,
            n_background_blocks,
            bg_density_logits.data_ptr<scalar_t>(),
            grad_bg_density_logits.has_value() ? grad_bg_density_logits->data_ptr<scalar_t>() : nullptr,
            n_other_node_points,
            other_node_density_logits.data_ptr<scalar_t>(),
            grad_other_node_density_logits.has_value() ? grad_other_node_density_logits->data_ptr<scalar_t>() : nullptr,
            selection_bits.data_ptr<uint8_t>(),
            reinterpret_cast<const BgTrackNodeSemanticWorkspace<scalar_t> *>(workspace.data_ptr<int32_t>()),
            grad_bg_unweighted.has_value() ? grad_bg_unweighted->data_ptr<scalar_t>() : nullptr,
            grad_bg_weighted.has_value() ? grad_bg_weighted->data_ptr<scalar_t>() : nullptr,
            grad_node_unweighted.has_value() ? grad_node_unweighted->data_ptr<scalar_t>() : nullptr,
            grad_node_weighted.has_value() ? grad_node_weighted->data_ptr<scalar_t>() : nullptr,
            bg_lambda,
            node_lambda
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
} // namespace

// ----------------------------------------------------------------------------
// Exported host entries (mode dispatch)
// ----------------------------------------------------------------------------

void launch_bg_track_node_semantic_losses_fwd_kernel(
    const double density_logits_min,
    const at::Tensor &positions,
    const at::Tensor &density_logits,
    const at::Tensor &semantic_logits,
    const std::optional<int64_t> &n_semantic_points,
    const std::vector<at::Tensor> &background_semantic_logits,
    const std::vector<BackgroundInTrackSemanticSegment> &semantic_segments,
    const at::Tensor &other_density_logits,
    const std::vector<at::Tensor> &other_semantic_logits,
    const std::vector<NodeSemanticSegment> &other_segments,
    const at::Tensor &camera_timestamps_startend_us,
    const at::Tensor &tracks_packinfo,
    const at::Tensor &tracks_poses,
    const at::Tensor &tracks_timestamps_us,
    const at::Tensor &cuboids_dims,
    const at::IntArrayRef background_allowed_class_ids,
    const std::optional<NodeSemanticPredicate> &node_primary_predicate,
    const double background_lambda,
    const double node_lambda,
    const at::Tensor &track_boxes,
    const at::Tensor &selection,
    const at::Tensor &workspace,
    const at::Tensor &background_unweighted_loss,
    const at::Tensor &background_weighted_loss,
    const at::Tensor &node_unweighted_loss,
    const at::Tensor &node_weighted_loss
)
{
    const bool background_in_track_enabled = background_lambda >= 0.0;
    const bool node_semantic_enabled       = node_lambda >= 0.0;
    TORCH_CHECK(
        background_in_track_enabled || node_semantic_enabled,
        "at least one background-in-track / node-semantic loss must be enabled"
    );
    if(node_semantic_enabled && !n_semantic_points.has_value())
    {
        TORCH_CHECK(
            node_primary_predicate.has_value(),
            "the joint node-semantic path requires an explicit primary-domain predicate; "
            "pass (class_ids, select_matches): empty ids with select_matches=true penalize "
            "no primary point, with select_matches=false they penalize all of them"
        );
        const at::IntArrayRef node_primary_class_ids = node_primary_predicate->first;
        const bool node_primary_select               = node_primary_predicate->second;
        AT_DISPATCH_FLOATING_TYPES(
            density_logits.scalar_type(),
            "bg_track_node_semantic_losses_fwd_joint",
            [&]()
            {
                background_in_track_loss_forward_joint<scalar_t>(
                    positions,
                    density_logits,
                    semantic_logits,
                    other_density_logits,
                    other_semantic_logits,
                    other_segments,
                    camera_timestamps_startend_us,
                    tracks_packinfo,
                    tracks_poses,
                    tracks_timestamps_us,
                    cuboids_dims,
                    background_allowed_class_ids,
                    node_primary_class_ids,
                    node_primary_select,
                    static_cast<scalar_t>(density_logits_min),
                    static_cast<scalar_t>(background_lambda),
                    static_cast<scalar_t>(node_lambda),
                    track_boxes,
                    selection,
                    workspace,
                    background_unweighted_loss,
                    background_weighted_loss,
                    node_unweighted_loss,
                    node_weighted_loss
                );
            }
        );
        return;
    }

    if(node_semantic_enabled)
    {
        TORCH_CHECK(
            background_in_track_enabled && n_semantic_points.has_value(),
            "packed node mode requires independent background metadata"
        );
        TORCH_CHECK(
            other_density_logits.scalar_type() == density_logits.scalar_type(),
            "other_density_logits must match density_logits dtype"
        );
        AT_DISPATCH_FLOATING_TYPES(
            density_logits.scalar_type(),
            "bg_track_node_semantic_losses_fwd_packed",
            [&]()
            {
                background_in_track_loss_forward_generic<scalar_t>(
                    static_cast<scalar_t>(density_logits_min),
                    positions,
                    density_logits,
                    *n_semantic_points,
                    background_semantic_logits,
                    semantic_segments,
                    camera_timestamps_startend_us,
                    tracks_packinfo,
                    tracks_poses,
                    tracks_timestamps_us,
                    cuboids_dims,
                    static_cast<scalar_t>(background_lambda),
                    track_boxes,
                    selection,
                    workspace,
                    background_unweighted_loss,
                    background_weighted_loss
                );
                node_semantic_loss_forward_packed<scalar_t>(
                    density_logits.numel(),
                    other_density_logits,
                    other_semantic_logits,
                    other_segments,
                    static_cast<scalar_t>(node_lambda),
                    selection,
                    workspace,
                    node_unweighted_loss,
                    node_weighted_loss
                );
            }
        );
        return;
    }

    TORCH_CHECK(
        background_in_track_enabled && n_semantic_points.has_value(),
        "background-in-track generic mode requires semantic metadata"
    );
    AT_DISPATCH_FLOATING_TYPES(
        density_logits.scalar_type(),
        "bg_track_node_semantic_losses_fwd_generic",
        [&]()
        {
            background_in_track_loss_forward_generic<scalar_t>(
                static_cast<scalar_t>(density_logits_min),
                positions,
                density_logits,
                *n_semantic_points,
                background_semantic_logits,
                semantic_segments,
                camera_timestamps_startend_us,
                tracks_packinfo,
                tracks_poses,
                tracks_timestamps_us,
                cuboids_dims,
                static_cast<scalar_t>(background_lambda),
                track_boxes,
                selection,
                workspace,
                background_unweighted_loss,
                background_weighted_loss
            );
        }
    );
}

void launch_bg_track_node_semantic_losses_bwd_kernel(
    const at::Tensor &background_density_logits,
    const at::Tensor &other_density_logits,
    const at::Tensor &selection,
    const at::Tensor &workspace,
    const std::optional<at::Tensor> &grad_background_unweighted,
    const std::optional<at::Tensor> &grad_background_weighted,
    const std::optional<at::Tensor> &grad_node_unweighted,
    const std::optional<at::Tensor> &grad_node_weighted,
    const double background_lambda,
    const double node_lambda,
    const std::optional<at::Tensor> &grad_background_density_logits,
    const std::optional<at::Tensor> &grad_other_density_logits
)
{
    const bool background_in_track_enabled = background_lambda >= 0.0;
    const bool node_semantic_enabled       = node_lambda >= 0.0;
    TORCH_CHECK(
        background_in_track_enabled || node_semantic_enabled,
        "at least one background-in-track / node-semantic loss must be enabled"
    );
    if(node_semantic_enabled)
    {
        AT_DISPATCH_FLOATING_TYPES(
            background_density_logits.scalar_type(),
            "bg_track_node_semantic_losses_bwd_joint",
            [&]()
            {
                background_in_track_loss_backward_joint<scalar_t>(
                    background_density_logits,
                    other_density_logits,
                    selection,
                    workspace,
                    grad_background_unweighted,
                    grad_background_weighted,
                    grad_node_unweighted,
                    grad_node_weighted,
                    static_cast<scalar_t>(background_lambda),
                    static_cast<scalar_t>(node_lambda),
                    grad_background_density_logits,
                    grad_other_density_logits
                );
            }
        );
        return;
    }

    TORCH_CHECK(background_in_track_enabled, "background-in-track must be enabled in generic mode");
    if(!grad_background_density_logits.has_value())
    {
        return;
    }
    AT_DISPATCH_FLOATING_TYPES(
        background_density_logits.scalar_type(),
        "bg_track_node_semantic_losses_bwd_generic",
        [&]()
        {
            background_in_track_loss_backward_generic<scalar_t>(
                background_density_logits,
                selection,
                workspace,
                grad_background_unweighted,
                grad_background_weighted,
                static_cast<scalar_t>(background_lambda),
                grad_background_density_logits.value()
            );
        }
    );
}
} // namespace gsplat

#endif
