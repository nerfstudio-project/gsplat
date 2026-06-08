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

#include "DistributedCollectives.h"
#include "TorchUtils.h"

#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/SymInt.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <numeric>

namespace gsplat {
namespace {

// --- The only irreducible job: reach the dispatcher-registered functional ---
// collective and synchronize the async result. We always call the `_autograd`
// variant: torch records a backward (the reverse collective) only when grad is
// enabled and an input requires grad; under no_grad / for non-floating inputs
// (the integer count tensors) it degrades to a plain forward. So there is no
// autograd-vs-plain choice to make.
at::Tensor all_gather_into_tensor(
    const at::Tensor &input, int64_t world_size, const std::string &pg
) {
    using Sig = at::Tensor(const at::Tensor &, int64_t, const std::string &);

    at::Tensor gathered = call_torch_op<Sig>(
        "_c10d_functional_autograd::all_gather_into_tensor", input, world_size, pg);

    return c10d::wait_tensor(gathered);
}

at::Tensor all_to_all_single(
    const at::Tensor &input,
    at::SymIntArrayRef output_splits,
    at::SymIntArrayRef input_splits,
    const std::string &pg
) {
    using Sig = at::Tensor(
        const at::Tensor &, at::SymIntArrayRef, at::SymIntArrayRef, const std::string &);

    at::Tensor scattered = call_torch_op<Sig>(
        "_c10d_functional_autograd::all_to_all_single",
        input, output_splits, input_splits, pg);

    return c10d::wait_tensor(scattered);
}

// --- Pure host-side bookkeeping (no torch-distributed interaction) -----------

std::vector<int64_t> to_int64_vector(const at::Tensor &tensor) {
    at::Tensor cpu = tensor.to(at::kCPU, at::kLong).contiguous();
    const int64_t *data = cpu.const_data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + cpu.numel());
}

std::vector<c10::SymInt> to_symints(const std::vector<int64_t> &values) {
    std::vector<c10::SymInt> out;
    out.reserve(values.size());
    for (int64_t v : values) {
        out.emplace_back(v);
    }
    return out;
}

int64_t trailing_numel(const at::Tensor &tensor) {
    int64_t size = 1;
    for (int64_t dim = 1; dim < tensor.dim(); ++dim) {
        size *= tensor.size(dim);
    }
    return size;
}

std::vector<int64_t> cumulative_offsets(const std::vector<int64_t> &sizes) {
    std::vector<int64_t> offsets;
    offsets.reserve(sizes.size());
    int64_t running = 0;
    for (int64_t size : sizes) {
        offsets.push_back(running);
        running += size;
    }
    return offsets;
}

// Repeat offsets[i] counts[i] times into a [sum(counts)] index tensor.
at::Tensor repeat_rank_offsets(
    const std::vector<int64_t> &offsets,
    const std::vector<int64_t> &counts,
    const at::TensorOptions &options
) {
    TORCH_CHECK(
        offsets.size() == counts.size(),
        "repeat_rank_offsets requires offsets and counts of equal length"
    );
    std::vector<int64_t> values;
    int64_t total = std::accumulate(counts.begin(), counts.end(), int64_t{0});
    values.reserve(total);
    for (size_t idx = 0; idx < offsets.size(); ++idx) {
        values.insert(values.end(), counts[idx], offsets[idx]);
    }
    return at::tensor(values, options.dtype(at::kLong));
}

std::vector<int64_t> dense_scatter_input_splits(
    const std::vector<int64_t> &C_world, int64_t local_N
) {
    std::vector<int64_t> splits;
    splits.reserve(C_world.size());
    for (int64_t C_i : C_world) {
        splits.push_back(C_i * local_N);
    }
    return splits;
}

std::vector<int64_t> dense_scatter_output_splits(
    int64_t local_C, const std::vector<int64_t> &N_world
) {
    std::vector<int64_t> splits;
    splits.reserve(N_world.size());
    for (int64_t N_i : N_world) {
        splits.push_back(local_C * N_i);
    }
    return splits;
}

// Dense all-to-all output arrives grouped by sending rank. Rebuild the local
// camera-major layout [local_C, sum(N_world), ...] the rasterizer expects.
at::Tensor reshape_dense_distributed_view(
    int64_t local_C, const at::Tensor &world_view,
    const std::vector<int64_t> &N_world
) {
    std::vector<at::Tensor> per_camera;
    per_camera.reserve(local_C);
    for (int64_t camera = 0; camera < local_C; ++camera) {
        std::vector<at::Tensor> chunks;
        chunks.reserve(N_world.size());
        int64_t rank_start = 0;
        for (int64_t N_i : N_world) {
            chunks.push_back(world_view.narrow(0, rank_start + camera * N_i, N_i));
            rank_start += local_C * N_i;
        }
        per_camera.push_back(at::cat(chunks, 0));
    }
    return at::stack(per_camera, 0);
}

// --- Payload batching: the primitives are single-tensor, so a list is packed --
// into one collective (reshape -> cat -> collective -> split -> restore shapes),
// avoiding one launch per tensor. This is caller-side payload prep, not a
// representation tweak.
std::vector<at::Tensor> gather_batched(
    const std::vector<at::Tensor> &tensors,
    int64_t world_size, const std::string &pg
) {
    TORCH_CHECK(!tensors.empty(), "gather_batched requires a non-empty list");
    const int64_t N = tensors[0].size(0);
    std::vector<at::Tensor> flat;
    std::vector<int64_t> feature_sizes;
    flat.reserve(tensors.size());
    feature_sizes.reserve(tensors.size());
    for (const at::Tensor &t : tensors) {
        TORCH_CHECK(t.size(0) == N, "gathered tensors must share dim 0");
        // Explicit trailing size (not -1) so an N == 0 shard reshapes cleanly;
        // see the all_to_all_batched note. Equivalent to -1 for N >= 1.
        flat.push_back(t.reshape({N, trailing_numel(t)}));
        feature_sizes.push_back(trailing_numel(t));
    }
    at::Tensor gathered =
        all_gather_into_tensor(at::cat(flat, -1).contiguous(), world_size, pg);
    std::vector<at::Tensor> chunks = gathered.split(feature_sizes, -1);
    std::vector<at::Tensor> out;
    out.reserve(tensors.size());
    for (size_t idx = 0; idx < tensors.size(); ++idx) {
        std::vector<int64_t> shape = tensors[idx].sizes().vec();
        shape[0] = -1;
        out.push_back(chunks[idx].reshape(shape));
    }
    return out;
}

std::vector<at::Tensor> all_to_all_batched(
    const std::vector<at::Tensor> &tensors,
    const std::vector<int64_t> &input_splits,
    const std::vector<int64_t> &output_splits,
    int64_t world_size, const std::string &pg
) {
    TORCH_CHECK(!tensors.empty(), "all_to_all_batched requires a non-empty list");
    const int64_t N = tensors[0].size(0);
    std::vector<at::Tensor> flat;
    std::vector<int64_t> feature_sizes;
    flat.reserve(tensors.size());
    feature_sizes.reserve(tensors.size());
    for (const at::Tensor &t : tensors) {
        TORCH_CHECK(t.size(0) == N, "all-to-all tensors must share dim 0");
        // Use an explicit trailing size, not -1: when a rank has zero local
        // Gaussians (N == 0, e.g. fully culled), reshape({0, -1}) cannot infer
        // -1 from a zero known dim and raises mid-collective, which can hang
        // the other ranks already inside the matching all-to-all.
        flat.push_back(t.reshape({N, trailing_numel(t)}));
        feature_sizes.push_back(trailing_numel(t));
    }
    std::vector<c10::SymInt> out_sym = to_symints(output_splits);
    std::vector<c10::SymInt> in_sym = to_symints(input_splits);
    at::Tensor gathered = all_to_all_single(
        at::cat(flat, -1).contiguous(),
        at::SymIntArrayRef(out_sym),
        at::SymIntArrayRef(in_sym),
        pg
    );
    std::vector<at::Tensor> chunks = gathered.split(feature_sizes, -1);
    std::vector<at::Tensor> out;
    out.reserve(tensors.size());
    for (size_t idx = 0; idx < tensors.size(); ++idx) {
        std::vector<int64_t> shape = tensors[idx].sizes().vec();
        shape[0] = -1;
        out.push_back(chunks[idx].reshape(shape));
    }
    return out;
}

// Gather one int per rank into a host vector (counts).
std::vector<int64_t> all_gather_int(
    int64_t value, const at::Tensor &like,
    int64_t world_size, const std::string &pg
) {
    at::Tensor input = at::full({1}, value, like.options().dtype(at::kInt));
    at::Tensor gathered = all_gather_into_tensor(input, world_size, pg);
    std::vector<int64_t> values = to_int64_vector(gathered.view({-1}));
    TORCH_CHECK(
        static_cast<int64_t>(values.size()) == world_size,
        "all_gather_int expected ", world_size, " values, got ", values.size()
    );
    return values;
}

// Exchange one int per rank (rank i sends values[j] to rank j).
std::vector<int64_t> all_to_all_int(
    const std::vector<int64_t> &values, const at::Tensor &like,
    int64_t world_size, const std::string &pg
) {
    at::Tensor input = at::tensor(values, like.options().dtype(at::kInt));
    std::vector<int64_t> unit(world_size, 1);
    std::vector<c10::SymInt> unit_sym = to_symints(unit);
    at::Tensor output = all_to_all_single(
        input, at::SymIntArrayRef(unit_sym), at::SymIntArrayRef(unit_sym), pg
    );
    return to_int64_vector(output);
}

} // namespace

DistributedCameraGather gather_cameras_for_distributed(
    const at::Tensor &viewmats,
    const at::Tensor &Ks,
    int64_t local_N,
    int64_t local_C,
    int64_t world_size,
    const std::string &process_group_name
) {
    std::vector<int64_t> N_world =
        all_gather_int(local_N, viewmats, world_size, process_group_name);
    std::vector<int64_t> C_world =
        all_gather_int(local_C, viewmats, world_size, process_group_name);
    for (int64_t C_i : C_world) {
        TORCH_CHECK(
            C_i == local_C,
            "Distributed rasterization requires each rank to render the same "
            "number of cameras"
        );
    }
    std::vector<at::Tensor> cameras =
        gather_batched({viewmats, Ks}, world_size, process_group_name);
    DistributedCameraGather result;
    result.viewmats = cameras[0];
    result.Ks = cameras[1];
    result.N_world = std::move(N_world);
    result.C_world = std::move(C_world);
    result.global_C = result.viewmats.size(0);
    return result;
}

DistributedProjection scatter_projection_for_distributed(
    bool packed,
    const DistributedProjection &projection,
    const std::vector<int64_t> &C_world,
    const std::vector<int64_t> &N_world,
    int64_t local_C,
    int64_t local_N,
    int64_t global_C,
    int64_t world_size,
    const std::string &process_group_name
) {
    DistributedProjection out;
    const bool has_features = projection.features.defined();

    if (packed) {
        // Count visible Gaussians per destination rank from the per-camera
        // histogram, then exchange the counts so each rank knows its inbox.
        at::Tensor per_camera_counts =
            at::bincount(projection.camera_ids, c10::nullopt, global_C);
        std::vector<int64_t> per_camera_counts_host =
            to_int64_vector(per_camera_counts);
        std::vector<int64_t> send_splits(world_size, 0);
        int64_t camera_offset = 0;
        for (int64_t rank = 0; rank < world_size; ++rank) {
            for (int64_t camera = 0; camera < C_world[rank]; ++camera) {
                send_splits[rank] += per_camera_counts_host[camera_offset + camera];
            }
            camera_offset += C_world[rank];
        }
        std::vector<int64_t> output_splits = all_to_all_int(
            send_splits, projection.camera_ids, world_size, process_group_name
        );

        out.radii = all_to_all_batched(
            {projection.radii}, send_splits, output_splits,
            world_size, process_group_name
        )[0];
        std::vector<at::Tensor> payload =
            has_features
                ? all_to_all_batched(
                      {projection.means2d, projection.depths, projection.conics,
                       projection.opacities, projection.features},
                      send_splits, output_splits, world_size, process_group_name
                  )
                : all_to_all_batched(
                      {projection.means2d, projection.depths, projection.conics,
                       projection.opacities},
                      send_splits, output_splits, world_size, process_group_name
                  );
        out.means2d = payload[0];
        out.depths = payload[1];
        out.conics = payload[2];
        out.opacities = payload[3];
        if (has_features) {
            out.features = payload[4];
        }

        // camera_ids are global pre-scatter (turn local for this rank);
        // gaussian_ids are rank-local pre-scatter (turn global). Both ride the
        // same all-to-all routing as the payload.
        at::Tensor camera_offsets = repeat_rank_offsets(
            cumulative_offsets(C_world), send_splits, projection.camera_ids.options()
        );
        at::Tensor gaussian_offsets = repeat_rank_offsets(
            cumulative_offsets(N_world), send_splits, projection.gaussian_ids.options()
        );
        std::vector<at::Tensor> ids = all_to_all_batched(
            {projection.camera_ids - camera_offsets,
             projection.gaussian_ids + gaussian_offsets},
            send_splits, output_splits, world_size, process_group_name
        );
        out.camera_ids = ids[0];
        out.gaussian_ids = ids[1];
        out.batch_ids = at::zeros_like(out.camera_ids);
    } else {
        std::vector<int64_t> input_splits =
            dense_scatter_input_splits(C_world, local_N);
        std::vector<int64_t> output_splits =
            dense_scatter_output_splits(local_C, N_world);

        out.radii = reshape_dense_distributed_view(
            local_C,
            all_to_all_batched(
                {projection.radii.flatten(0, 1)}, input_splits, output_splits,
                world_size, process_group_name
            )[0],
            N_world
        );
        std::vector<at::Tensor> payload =
            has_features
                ? all_to_all_batched(
                      {projection.means2d.flatten(0, 1),
                       projection.depths.flatten(0, 1),
                       projection.conics.flatten(0, 1),
                       projection.opacities.flatten(0, 1),
                       projection.features.flatten(0, 1)},
                      input_splits, output_splits, world_size, process_group_name
                  )
                : all_to_all_batched(
                      {projection.means2d.flatten(0, 1),
                       projection.depths.flatten(0, 1),
                       projection.conics.flatten(0, 1),
                       projection.opacities.flatten(0, 1)},
                      input_splits, output_splits, world_size, process_group_name
                  );
        out.means2d = reshape_dense_distributed_view(local_C, payload[0], N_world);
        out.depths = reshape_dense_distributed_view(local_C, payload[1], N_world);
        out.conics = reshape_dense_distributed_view(local_C, payload[2], N_world);
        out.opacities = reshape_dense_distributed_view(local_C, payload[3], N_world);
        if (has_features) {
            out.features = reshape_dense_distributed_view(local_C, payload[4], N_world);
        }
    }
    return out;
}

} // namespace gsplat
