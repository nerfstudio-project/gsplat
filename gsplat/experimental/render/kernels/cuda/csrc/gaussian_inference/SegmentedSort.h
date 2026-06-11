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

#include <cuda_runtime.h>
#include <cstdint>

namespace higs {

// internal API for the segmented sort pipeline.
//
// the sort handles variable-length segments via a cascade of block-sort
// kernels followed by a merge phase for the largest segments.
//
// block-sort cascade:
//   four tiers handle progressively larger segments. each tier sorts
//   segments up to its capacity and skips the rest. the highest tier
//   also acts as an epilog: segments exceeding its capacity are split
//   into fixed-size blocks and queued for the overflow + merge phases.
//
// overflow block sort:
//   sorts each overflow block independently. the persistent variant
//   claims work from a GPU-side atomic counter.
//
// merge phase:
//   iterative 2-way merge-path passes double the sorted run length each
//   pass, ping-ponging between the caller's two buffer pairs (A and B)
//   until all overflow segments are fully merged. parity flags ensure the
//   final pass always writes to buffer A (index 0).
namespace seg_sort {

// one node in the binary merge tree. describes a 2-way merge of two
// adjacent sorted runs within an overflow segment. the split point
// (a_end) and partition count are derived at use:
//   a_end        = min(begin + (stride_and_flag & 0x7FFFFFFF), end)
//   n_partitions = (end - begin + MERGE_PARTITION_SIZE - 1) / MERGE_PARTITION_SIZE
//   flag         = stride_and_flag & 0x80000000
struct __align__(16) MergeTreeNode
{
    int begin;           // start of merge range
    int end;             // end of merge range
    int stride_and_flag; // sign bit = ping-pong direction, lower 31 bits = merge stride
    int parent_idx;      // index in tree_nodes array, -1 for root
};

// one work item in the FIFO merge queue. aligned to 8 bytes so the
// compiler uses a single 64-bit load/store for the pair.
struct __align__(8) MergeWorkItem
{
    int32_t node_idx;
    int32_t partition_idx;
};

// scalar counters for the overflow block-sort phase. zero-init before each
// sort run. each field is padded to a 32-byte L2 sector boundary so that
// concurrent atomics from different kernels never serialize on the same
// L2 atomic unit.
struct __align__(32) OverflowState
{
    alignas(32) int32_t block_count;   // number of overflow blocks (T3 epilog atomicAdd)
    alignas(32) int32_t claim_counter; // persistent overflow kernel work-claim counter (atomicAdd)
};

// scalar counters shared between the T3 epilog (producer) and the
// persistent merge kernel (consumer). zero-init before each sort run.
// each field is padded to a 32-byte L2 sector boundary so that concurrent
// atomics to different counters never serialize on the same L2 atomic unit.
//
// queue_tail is the commit counter: consumers may read slots with
// index < queue_tail. the T3 epilog uses queue_tail as both reservation
// and commit (safe via stream ordering). the merge kernel reserves via
// queue_reserved and commits to queue_tail with an ordered CAS.
struct __align__(32) MergeQueueState
{
    alignas(32) int32_t total_tree_nodes;  // tree node allocator (T3 epilog atomicAdd)
    alignas(32) int32_t total_queue_items; // total partitions across all tree levels (termination bound)
    alignas(32) int32_t queue_head;        // consumer cursor (merge kernel atomicAdd)
    alignas(32) int32_t queue_tail;        // commit counter: items at [0, tail) are readable
    alignas(32) int32_t queue_reserved;    // reservation counter for dynamic enqueue during merge
};

struct PersistentMergeArgs
{
    MergeQueueState *queue_state;
    MergeTreeNode *tree_nodes;
    int32_t *node_done_counts;
    MergeWorkItem *work_items;
    int32_t *keys_a;
    int32_t *vals_a;
    int32_t *keys_b;
    int32_t *vals_b;
};

struct MergeArgs
{
    const int4 *partitions;
    int32_t *keys_a;
    int32_t *vals_a;
    int32_t *keys_b;
    int32_t *vals_b;
};

struct BlockSortArgs
{
    int32_t n_segments;
    int32_t n_items;
    const int32_t *offsets;
    const int32_t *keys_in;
    const int32_t *values_in;

    int32_t *values_out;
    int32_t *keys_out;

    int2 *overflow_blocks;
    OverflowState *overflow_state;

    MergeQueueState *merge_queue_state;
    MergeTreeNode *tree_nodes;
    int32_t *node_done_counts;
    MergeWorkItem *work_items;

    int32_t *values_out_b;
    int32_t *keys_out_b;
};

struct PipelineState
{
    // host-side layout parameters, set by SegmentedSortSetup
    int32_t n_segments     = 0;
    int32_t n_items        = 0;
    int n_max_merge_passes = 0;
    int sm_count           = 0;

    // overflow bookkeeping (device); written by T3 epilog, consumed by
    // the persistent overflow kernel
    struct OverflowState *overflow_state = nullptr;
    int2 *overflow_blocks                = nullptr;

    // merge tree + FIFO queue (device); the T3 epilog builds a binary merge
    // tree and seeds the queue with leaf-level work. the persistent merge
    // kernel consumes items, tracks per-node completion, and dynamically
    // enqueues parent nodes as their children finish.
    struct MergeQueueState *merge_queue_state = nullptr;
    struct MergeTreeNode *tree_nodes          = nullptr;
    int32_t *node_done_counts                 = nullptr;
    struct MergeWorkItem *work_items          = nullptr;

    // range within scratch that must be zeroed before each sort run
    void *run_init_ptr   = nullptr;
    size_t run_init_size = 0;
};

} // namespace seg_sort

using SegmentedSortState = seg_sort::PipelineState;

// two-pass setup for the sort pipeline scratch memory.
//   pass 1: scratch = nullptr  -> computes and returns required byte count
//   pass 2: scratch != nullptr -> lays out sub-regions within the provided buffer
// returns the total scratch size in bytes (256-aligned).
size_t SegmentedSortSetup(SegmentedSortState &s, int32_t n_segments, int32_t n_items, void *scratch);

// sort values by key within each segment defined by d_offsets[0..n_segments].
// segment i spans items [d_offsets[i], d_offsets[i+1]).
// n_segments and n_items are read from the state set by SegmentedSortSetup.
//
// d_keys and d_values are host arrays of 2 device pointers each (double
// buffers). input is expected in d_keys[0] / d_values[0]. both buffer
// pairs are used as working space and may be modified. only sorted values
// are guaranteed; key buffers contain intermediate merge data.
//
// returns the buffer index (0 or 1) where the sorted values reside.
// the result is deterministic and requires no device synchronization.
int SegmentedSortAsync(const SegmentedSortState &st, const int32_t *d_offsets, int32_t *const *d_keys,
                       int32_t *const *d_values, cudaStream_t s);

} // namespace higs
