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

#include <cstdint>

namespace gsplat::bg_track_node_semantic
{
// =============================================================================
// Background-in-track + node-semantic BCE losses: per-element device cores.
//
// Two-layer layout (device cores + kernel shells): these __device__
// __forceinline__ functions are the single source of the per-point selection
// predicates and BCE value/gradient arithmetic. They are consumed by the
// per-dispatch kernel shells in BgTrackNodeSemanticLossesCUDA.cu today and by
// the fused all-losses kernel later, so the math lives here exactly once.
//
// Contracts:
//  * Header-only and torch-free: plain scalars and raw pointers only.
//  * scalar_t is float or double (AT_DISPATCH_FLOATING_TYPES in the shells).
//  * Cores are pure per-element math over one point. Warp/block reductions,
//    shared-memory track tiling, atomics, selection-buffer publication, and
//    the track-box interpolation precompute stay in the kernel shells.
// =============================================================================

// Per-point flag bits written into the uint8 selection buffer by the forward
// kernels and consumed by the backward kernels.
constexpr uint8_t BACKGROUND_IN_TRACK_SELECTION_BIT = uint8_t{1} << 0;
constexpr uint8_t NODE_SEMANTIC_SELECTION_BIT       = uint8_t{1} << 1;

// Semantic class IDs are packed into 32-bit mask words: word index is
// class_id >> SEMANTIC_CLASS_BITS, bit index is class_id & SEMANTIC_CLASS_MASK.
constexpr int SEMANTIC_CLASS_BITS      = 5;
constexpr int SEMANTIC_CLASS_COUNT     = 1 << SEMANTIC_CLASS_BITS;
constexpr uint32_t SEMANTIC_CLASS_MASK = SEMANTIC_CLASS_COUNT - 1;

// A track box stores three world-to-local rotation rows, with the
// world-space center in their fourth slot, followed by (half_dim_x,
// half_dim_y, half_dim_z, valid): 16 scalars per track. Keeping the center
// separate preserves the exact dot(R, point - center) arithmetic at
// inclusive cuboid boundaries.
constexpr int TRACK_BOX_VALUES = 16;

// Argmax over the [n_semantic_classes] logits row of point_idx; ties keep the
// lowest class index (strict > scan from class 0).
template<typename scalar_t>
__device__ __forceinline__ int semantic_argmax(
    const int point_idx, const int n_semantic_classes, const scalar_t *__restrict__ semantic_logits
)
{
    const scalar_t *const logits = semantic_logits + static_cast<int64_t>(point_idx) * n_semantic_classes;
    int argmax_idx               = 0;
    scalar_t max_logit           = logits[0];
    for(int class_idx = 1; class_idx < n_semantic_classes; ++class_idx)
    {
        const scalar_t candidate = logits[class_idx];
        if(candidate > max_logit)
        {
            max_logit  = candidate;
            argmax_idx = class_idx;
        }
    }
    return argmax_idx;
}

// Multiword word-ownership check: with more than 32 semantic classes the mask
// spans several uint32 words and each kernel launch handles exactly one word,
// so a point is owned by (and its selection bit written from) the launch
// whose word contains its argmax class. Single-word launches
// (MULTIWORD_SEMANTICS == false) own every point.
template<bool MULTIWORD_SEMANTICS>
__device__ __forceinline__ bool owns_mask_word(const int semantic_class, const int semantic_mask_word)
{
    return !MULTIWORD_SEMANTICS || (semantic_class >> SEMANTIC_CLASS_BITS) == semantic_mask_word;
}

// True when semantic_class's bit is set in this launch's class_mask word.
// Callers must already have established word ownership via owns_mask_word.
// In single-word launches classes >= SEMANTIC_CLASS_COUNT never match (their
// bit lives in a word that was never built).
template<bool MULTIWORD_SEMANTICS>
__device__ __forceinline__ bool class_mask_matches(const int semantic_class, const uint32_t class_mask)
{
    const uint32_t class_bit = uint32_t{1} << (semantic_class & SEMANTIC_CLASS_MASK);
    return (MULTIWORD_SEMANTICS || semantic_class < SEMANTIC_CLASS_COUNT) && (class_mask & class_bit) != 0;
}

// Node-semantic selection predicate. Polarity is subtle:
// selected = class_matches == select_matches. An empty class mask with
// select_matches=true penalizes nothing; with select_matches=false it
// penalizes everything (in its mask word).
template<bool MULTIWORD_SEMANTICS>
__device__ __forceinline__ bool selection_matches(
    const int semantic_class, const uint32_t class_mask, const bool select_matches
)
{
    return class_mask_matches<MULTIWORD_SEMANTICS>(semantic_class, class_mask) == select_matches;
}

// Allow-list predicate of the segmented background-in-track filter: the
// argmax class must live in this launch's mask word and have its bit set.
__device__ __forceinline__ bool mask_word_allows(
    const int argmax_idx, const int32_t mask_word, const uint32_t class_mask
)
{
    return argmax_idx / SEMANTIC_CLASS_COUNT == mask_word
        && (class_mask & (uint32_t{1} << (argmax_idx & SEMANTIC_CLASS_MASK))) != 0;
}

// Track-box containment test. Box layout per track (TRACK_BOX_VALUES == 16):
//   [ 0.. 2] world-to-local rotation row 0, [ 3] center.x
//   [ 4.. 6] world-to-local rotation row 1, [ 7] center.y
//   [ 8..10] world-to-local rotation row 2, [11] center.z
//   [12..14] half dims,                     [15] valid flag
// Boundaries are inclusive; callers gate on the valid flag (box[15] > 0).
template<typename scalar_t>
__device__ __forceinline__ bool point_inside_box(const scalar_t point[3], const scalar_t *__restrict__ box)
{
    const scalar_t cx      = point[0] - box[3];
    const scalar_t cy      = point[1] - box[7];
    const scalar_t cz      = point[2] - box[11];
    const scalar_t local_x = box[0] * cx + box[1] * cy + box[2] * cz;
    const scalar_t local_y = box[4] * cx + box[5] * cy + box[6] * cz;
    const scalar_t local_z = box[8] * cx + box[9] * cy + box[10] * cz;
    return local_x >= -box[12]
        && local_x <= box[12]
        && local_y >= -box[13]
        && local_y <= box[13]
        && local_z >= -box[14]
        && local_z <= box[14];
}

// Softplus BCE value core: log(1 + exp(x)) in its overflow-safe form, the
// per-selected-point loss value of both the background-in-track and
// node-semantic BCE terms.
template<typename scalar_t>
__device__ __forceinline__ scalar_t stable_softplus(const scalar_t x)
{
    const scalar_t relu = x > scalar_t(0) ? x : scalar_t(0);
    return relu + log1p(exp(-fabs(x)));
}

// Sigmoid gradient core: d/dx softplus(x) = 1/(1+exp(-x)). Branchless with
// the same overflow safety as the two-branch exp form, expressed through
// tanh.
template<typename scalar_t>
__device__ __forceinline__ scalar_t stable_sigmoid(const scalar_t x)
{
    return scalar_t(0.5) * (tanh(scalar_t(0.5) * x) + scalar_t(1));
}
} // namespace gsplat::bg_track_node_semantic
