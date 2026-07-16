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
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>

namespace gsplat
{
// Background-in-track + node-semantic grouped BCE losses.
//
// Single entry point per direction. The forward computes the complete masked
// BCE-with-logits mean used to suppress background Gaussians that lie inside
// any dynamic cuboid track. When node_lambda is nonnegative, the same call
// also computes node-semantic BCE for every configured Gaussian node. When
// their point domains align, the primary node shares semantic classification
// with the background member; otherwise generic background and packed node
// kernels accumulate into the same workspace. Either member may be enabled
// independently (lambda >= 0 enables); only density logits are
// differentiable.
//
// Filtered semantic layers are packed first in positions/density_logits, and
// their semantic rows share that point ordering. Each segment owns its
// semantic tensor, so layers may use independent class counts. Every segment
// stores its exclusive end followed by its configured class IDs. Points after
// n_semantic_points are processed without semantic filtering. A missing
// n_semantic_points selects the shared primary domain and requires an
// explicit node_primary_predicate. An empty predicate remains valid:
// select_matches=true penalizes no primary point, select_matches=false
// penalizes all of them.
//
// Buffer conventions (allocated dtype-independently by the caller):
//   - selection: uint8, one byte per point; bit 0 marks background-in-track
//     selection, bit 1 marks node-semantic selection. Sized >= N_background in
//     background-only mode and exactly N_background + N_other whenever the
//     node member is enabled.
//   - workspace: int32[8], 16-byte aligned. Interpreted in-kernel as the
//     dtype-templated reduction struct {loss sums (scalar_t) + selected
//     counts (int32)}; 8 int32 words (32 bytes) cover both the fp32 (16-byte)
//     and fp64 (32-byte) layouts, so allocation does not depend on dtype.
//     Forward-to-backward contract: ONLY the forward writes the selected
//     counts, and the backward divides by them. The exact workspace instance
//     the forward filled must reach the backward unmodified — re-zeroing,
//     pooling, or reusing it between the two passes silently corrupts every
//     gradient of that step.
//   - track_boxes: [n_tracks, 16] scalar_t, 16-byte aligned. Precomputed by
//     the forward: rows 0-2 hold the world-to-local rotation rows with the
//     world center in their fourth slot, row 3 holds (half_dim_x, half_dim_y,
//     half_dim_z, valid).
//
// Timestamps stay int64 (microseconds), tracks_packinfo stays int32.
// Floating tensors are templated on scalar_t (fp32/fp64) and must all share
// density_logits' dtype.
//
// The class-id lists inside the segment/predicate aliases are non-owning
// at::IntArrayRef views into the op's int[] argument storage (kept alive by
// the dispatcher for the duration of the op call). The launchers consume
// them synchronously on the host — packed into uint32 masks before any
// kernel launch returns — so nothing may retain these views past the call.
using BackgroundInTrackSemanticSegment = std::pair<int64_t, at::IntArrayRef>;
using NodeSemanticSegment              = std::tuple<int64_t, at::IntArrayRef, bool>;
using NodeSemanticPredicate            = std::pair<at::IntArrayRef, bool>;

void launch_bg_track_node_semantic_losses_fwd_kernel(
    double density_logits_min,
    const at::Tensor &positions,       // [Nbg, 3]
    const at::Tensor &density_logits,  // [Nbg]
    const at::Tensor &semantic_logits, // [Nbg, Cbg] (joint mode only; dummy otherwise)
    const std::optional<int64_t> &n_semantic_points,
    const std::vector<at::Tensor> &background_semantic_logits, // per-segment [Nseg, C]
    const std::vector<BackgroundInTrackSemanticSegment> &semantic_segments,
    const at::Tensor &other_density_logits, // [Nother]
    const std::vector<at::Tensor> &other_semantic_logits,
    const std::vector<NodeSemanticSegment> &other_segments,
    const at::Tensor &camera_timestamps_startend_us, // [B, 2] int64, B >= 1; only row 0
                                                     // (the reference camera) is read
    const at::Tensor &tracks_packinfo,               // [T, 2] int32 (start idx, n poses)
    const at::Tensor &tracks_poses,                  // [P, 7] (tx,ty,tz,qx,qy,qz,qw)
    const at::Tensor &tracks_timestamps_us,          // [P] int64
    const at::Tensor &cuboids_dims,                  // [T, 3]
    at::IntArrayRef background_allowed_class_ids,
    const std::optional<NodeSemanticPredicate> &node_primary_predicate,
    double background_lambda,
    double node_lambda,
    const at::Tensor &track_boxes,                // [T, 16] (output workspace)
    const at::Tensor &selection,                  // uint8 (output, see conventions)
    const at::Tensor &workspace,                  // int32[8] (output, see conventions)
    const at::Tensor &background_unweighted_loss, // [] scalar (output)
    const at::Tensor &background_weighted_loss,   // [] scalar (output)
    const at::Tensor &node_unweighted_loss,       // [] scalar (output)
    const at::Tensor &node_weighted_loss          // [] scalar (output)
);

// Backward: re-reads the selection bits and reduction workspace written by
// the forward and scatters d(mean softplus)/d(logit) = upstream / count *
// sigmoid(logit) to the selected density logits (both members accumulate into
// points selected by both). The selected counts it divides by are written
// only by the forward: the same workspace instance must arrive here
// unmodified (no re-zeroing or pooling between the passes). Absent upstream
// gradients count as zero; absent gradient outputs skip that member's point
// domain entirely.
void launch_bg_track_node_semantic_losses_bwd_kernel(
    const at::Tensor &background_density_logits,                 // [Nbg]
    const at::Tensor &other_density_logits,                      // [Nother]
    const at::Tensor &selection,                                 // uint8 (from forward)
    const at::Tensor &workspace,                                 // int32[8] (from forward)
    const std::optional<at::Tensor> &grad_background_unweighted, // [] scalar
    const std::optional<at::Tensor> &grad_background_weighted,   // [] scalar
    const std::optional<at::Tensor> &grad_node_unweighted,       // [] scalar
    const std::optional<at::Tensor> &grad_node_weighted,         // [] scalar
    double background_lambda,
    double node_lambda,
    const std::optional<at::Tensor> &grad_background_density_logits, // [Nbg] (output)
    const std::optional<at::Tensor> &grad_other_density_logits       // [Nother] (output)
);
} // namespace gsplat
