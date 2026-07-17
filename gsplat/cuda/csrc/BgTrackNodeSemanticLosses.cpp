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

#include <ATen/core/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/OptionalArrayRef.h>
#include <torch/library.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "BgTrackNodeSemanticLosses.h"

namespace gsplat
{
// Outputs (track_boxes, selection, workspace, four loss scalars, gradients)
// are allocated by the caller (Python autograd wrapper) and passed in as
// mutable arguments — keeps memory lifetime explicit on the Python side so it
// can be reused by torch's caching allocator across training steps.
//
// Segment metadata crosses the op boundary flattened into plain int lists:
// per-segment exclusive point ends, one packed class-ID list, and per-segment
// exclusive ends into that packed list (node segments additionally carry a
// 0/1 select_matches flag per segment). This file re-inflates them into the
// per-segment structures the launchers consume; each segment's class-id list
// is a non-owning at::IntArrayRef slice of the packed argument (valid for the
// whole op call, consumed synchronously on the host — see the alias notes in
// BgTrackNodeSemanticLosses.h). All tensor validation lives with the launch
// code in BgTrackNodeSemanticLossesCUDA.cu.

namespace
{
    // Non-owning view into the op's packed class-id argument (kept alive by
    // the dispatcher for the duration of the call). Safe because the launch
    // consumes every id synchronously on the host — packed into uint32 class
    // masks before any kernel launch returns — and nothing retains the view
    // past the op call (see the alias notes in BgTrackNodeSemanticLosses.h).
    at::IntArrayRef slice_class_ids(const at::IntArrayRef class_ids, const int64_t begin, const int64_t end)
    {
        return class_ids.slice(begin, end - begin);
    }

    std::vector<BackgroundInTrackSemanticSegment> unpack_background_segments(
        const at::IntArrayRef segment_ends,
        const at::IntArrayRef segment_class_ids,
        const at::IntArrayRef segment_class_id_ends
    )
    {
        TORCH_CHECK(
            segment_ends.size() == segment_class_id_ends.size(),
            "background segment ends and class-id ends must have the same length"
        );
        std::vector<BackgroundInTrackSemanticSegment> segments;
        segments.reserve(segment_ends.size());
        int64_t previous_id_end = 0;
        for(size_t segment_idx = 0; segment_idx < segment_ends.size(); ++segment_idx)
        {
            const int64_t id_end = segment_class_id_ends[segment_idx];
            TORCH_CHECK(
                id_end >= previous_id_end && id_end <= static_cast<int64_t>(segment_class_ids.size()),
                "background class-id ends must be nondecreasing and bounded by the packed class-id list"
            );
            segments.emplace_back(
                segment_ends[segment_idx], slice_class_ids(segment_class_ids, previous_id_end, id_end)
            );
            previous_id_end = id_end;
        }
        TORCH_CHECK(
            previous_id_end == static_cast<int64_t>(segment_class_ids.size()),
            "background segments must consume the entire packed class-id list"
        );
        return segments;
    }

    std::vector<NodeSemanticSegment> unpack_node_segments(
        const at::IntArrayRef segment_ends,
        const at::IntArrayRef segment_class_ids,
        const at::IntArrayRef segment_class_id_ends,
        const at::IntArrayRef segment_select_matches
    )
    {
        TORCH_CHECK(
            segment_ends.size() == segment_class_id_ends.size() && segment_ends.size() == segment_select_matches.size(),
            "node segment ends, class-id ends, and select flags must have the same length"
        );
        std::vector<NodeSemanticSegment> segments;
        segments.reserve(segment_ends.size());
        int64_t previous_id_end = 0;
        for(size_t segment_idx = 0; segment_idx < segment_ends.size(); ++segment_idx)
        {
            const int64_t id_end = segment_class_id_ends[segment_idx];
            TORCH_CHECK(
                id_end >= previous_id_end && id_end <= static_cast<int64_t>(segment_class_ids.size()),
                "node class-id ends must be nondecreasing and bounded by the packed class-id list"
            );
            const int64_t select = segment_select_matches[segment_idx];
            TORCH_CHECK(select == 0 || select == 1, "node segment select flags must be 0 or 1");
            segments.emplace_back(
                segment_ends[segment_idx], slice_class_ids(segment_class_ids, previous_id_end, id_end), select != 0
            );
            previous_id_end = id_end;
        }
        TORCH_CHECK(
            previous_id_end == static_cast<int64_t>(segment_class_ids.size()),
            "node segments must consume the entire packed class-id list"
        );
        return segments;
    }
} // namespace

void bg_track_node_semantic_losses_fwd(
    const at::Tensor &positions,
    const at::Tensor &density_logits,
    const at::Tensor &semantic_logits,
    std::optional<int64_t> n_semantic_points,
    at::TensorList background_semantic_logits,
    at::IntArrayRef background_segment_ends,
    at::IntArrayRef background_segment_class_ids,
    at::IntArrayRef background_segment_class_id_ends,
    const at::Tensor &other_density_logits,
    at::TensorList other_semantic_logits,
    at::IntArrayRef other_segment_ends,
    at::IntArrayRef other_segment_class_ids,
    at::IntArrayRef other_segment_class_id_ends,
    at::IntArrayRef other_segment_select_matches,
    const at::Tensor &camera_timestamps_startend_us,
    const at::Tensor &tracks_packinfo,
    const at::Tensor &tracks_poses,
    const at::Tensor &tracks_timestamps_us,
    const at::Tensor &cuboids_dims,
    at::IntArrayRef background_allowed_class_ids,
    at::OptionalIntArrayRef node_primary_class_ids,
    bool node_primary_select_matches,
    double density_logits_min,
    double background_lambda,
    double node_lambda,
    at::Tensor track_boxes,
    at::Tensor selection,
    at::Tensor workspace,
    at::Tensor background_unweighted_loss,
    at::Tensor background_weighted_loss,
    at::Tensor node_unweighted_loss,
    at::Tensor node_weighted_loss
)
{
    const std::vector<BackgroundInTrackSemanticSegment> semantic_segments = unpack_background_segments(
        background_segment_ends, background_segment_class_ids, background_segment_class_id_ends
    );
    const std::vector<NodeSemanticSegment> other_segments = unpack_node_segments(
        other_segment_ends, other_segment_class_ids, other_segment_class_id_ends, other_segment_select_matches
    );
    std::optional<NodeSemanticPredicate> node_primary_predicate;
    if(node_primary_class_ids.has_value())
    {
        node_primary_predicate.emplace(*node_primary_class_ids, node_primary_select_matches);
    }

    launch_bg_track_node_semantic_losses_fwd_kernel(
        density_logits_min,
        positions,
        density_logits,
        semantic_logits,
        n_semantic_points,
        background_semantic_logits.vec(),
        semantic_segments,
        other_density_logits,
        other_semantic_logits.vec(),
        other_segments,
        camera_timestamps_startend_us,
        tracks_packinfo,
        tracks_poses,
        tracks_timestamps_us,
        cuboids_dims,
        background_allowed_class_ids,
        node_primary_predicate,
        background_lambda,
        node_lambda,
        track_boxes,
        selection,
        workspace,
        background_unweighted_loss,
        background_weighted_loss,
        node_unweighted_loss,
        node_weighted_loss
    );
}

void bg_track_node_semantic_losses_bwd(
    const at::Tensor &background_density_logits,
    const at::Tensor &other_density_logits,
    const at::Tensor &selection,
    const at::Tensor &workspace,
    std::optional<at::Tensor> grad_background_unweighted,
    std::optional<at::Tensor> grad_background_weighted,
    std::optional<at::Tensor> grad_node_unweighted,
    std::optional<at::Tensor> grad_node_weighted,
    double background_lambda,
    double node_lambda,
    std::optional<at::Tensor> grad_background_density_logits,
    std::optional<at::Tensor> grad_other_density_logits
)
{
    launch_bg_track_node_semantic_losses_bwd_kernel(
        background_density_logits,
        other_density_logits,
        selection,
        workspace,
        grad_background_unweighted,
        grad_background_weighted,
        grad_node_unweighted,
        grad_node_weighted,
        background_lambda,
        node_lambda,
        grad_background_density_logits,
        grad_other_density_logits
    );
}

void register_bg_track_node_semantic_losses_cuda_impl(torch::Library &m)
{
    m.impl("bg_track_node_semantic_losses_fwd", &bg_track_node_semantic_losses_fwd);
    m.impl("bg_track_node_semantic_losses_bwd", &bg_track_node_semantic_losses_bwd);
}
} // namespace gsplat
