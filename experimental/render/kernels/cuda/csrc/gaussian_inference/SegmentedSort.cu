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

#include <cmath>
#include <cub/block/block_radix_rank.cuh>
#include "SegmentedSort.h"

// terminology used throughout this file:
//
//   segment    — a top-level contiguous range of items in the input array,
//                defined by offsets[i] to offsets[i+1]. each segment is
//                sorted independently.
//
//   block      — a fixed-size (CAPACITY = CTA_SIZE * ITEMS_PER_THREAD) chunk
//                of an overflow segment. when a segment exceeds the largest
//                cascade tier's capacity, it is split into blocks that are
//                independently sorted before being merged back together.
//
//   partition  — a fixed-size (MERGE_PARTITION_SIZE) chunk of work within a
//                merge operation. each CTA processes one partition of a merge
//                tree node, reading from one ping-pong buffer and writing to
//                the other.

namespace higs {

namespace seg_sort {

// block sort (cascade) — shared by all tiers
static constexpr int BLOCK_SORT_ITEMS_PER_THREAD = 16;
static constexpr int BLOCK_SORT_RADIX_BITS       = 8;

template<int CTA_SIZE>
using BlockSortRankT = cub::BlockRadixRankMatch<CTA_SIZE, BLOCK_SORT_RADIX_BITS, false, cub::BLOCK_SCAN_WARP_SCANS>;

template<int CTA_SIZE>
union BlockSortSmem
{
    typename BlockSortRankT<CTA_SIZE>::TempStorage rank;
    int32_t raw[CTA_SIZE * BLOCK_SORT_ITEMS_PER_THREAD];
};

// cascade tier configuration: CTA size and min segment size
static constexpr int BLOCK_SORT_T0_CTA_SIZE = 64;
static constexpr int BLOCK_SORT_T0_MIN_SIZE = 0;
static constexpr int BLOCK_SORT_T1_CTA_SIZE = 128;
static constexpr int BLOCK_SORT_T1_MIN_SIZE = BLOCK_SORT_T0_CTA_SIZE * BLOCK_SORT_ITEMS_PER_THREAD;
static constexpr int BLOCK_SORT_T2_CTA_SIZE = 256;
static constexpr int BLOCK_SORT_T2_MIN_SIZE = BLOCK_SORT_T1_CTA_SIZE * BLOCK_SORT_ITEMS_PER_THREAD;
static constexpr int BLOCK_SORT_T3_CTA_SIZE = 512;
static constexpr int BLOCK_SORT_T3_MIN_SIZE = BLOCK_SORT_T2_CTA_SIZE * BLOCK_SORT_ITEMS_PER_THREAD;
static constexpr int BLOCK_SORT_T3_CAPACITY = BLOCK_SORT_T3_CTA_SIZE * BLOCK_SORT_ITEMS_PER_THREAD;

// merge kernel configuration
static constexpr int MERGE_CTA_SIZE         = 256;
static constexpr int MERGE_ITEMS_PER_THREAD = 15;
static constexpr int MERGE_PARTITION_SIZE   = MERGE_CTA_SIZE * MERGE_ITEMS_PER_THREAD;
static constexpr int MERGE_MIN_BLOCKS       = 3;

// pairs a typed pointer member with an element count and alignment.
// the template constructor deduces sizeof(T) so callers specify counts,
// not bytes. alignment controls the start offset of this entry within
// the scratch buffer: 32 for structs containing L2-sector-padded atomic
// counters, 256 for bulk arrays (cache-line alignment).
struct ScratchEntry
{
    void **ptr;
    size_t elem_size;
    size_t alignment;
    int count;

    template<typename T>
    ScratchEntry(T *&p, int c, size_t align)
        : ptr(reinterpret_cast<void **>(&p))
        , elem_size(sizeof(T))
        , alignment(align)
        , count(c)
    {
    }
};

// device-scope relaxed load. compiles to ld.relaxed.gpu.b32 which goes
// through the regular L2 cache path — unlike atomicAdd(&v, 0) which uses
// the L2 atomic RMW pipeline. use for spin-wait polling where only
// visibility (not ordering) is needed.
__device__ __forceinline__ int LoadRelaxedGpu(const int *addr)
{
    int val;
    asm volatile("ld.relaxed.gpu.b32 %0, [%1];" : "=r"(val) : "l"(addr));
    return val;
}

// tiled merge-path for one MERGE_PARTITION_SIZE chunk of a 2-way merge.
// the int4 descriptor encodes {a_begin, a_end, b_end, diagonal | direction_flag}.
// sign bit of desc.w selects the ping-pong direction.
//
// two-level merge-path: CTA-level partitions global data into
// smem, then thread-level partitions smem into per-thread work.
// output is scattered blocked->striped for coalesced writes.
// smem_pairs is caller-provided so the allocation can be aliased (e.g. with
// the persistent merge kernel's queue-position broadcast slot).
__device__ void MergePartition(int4 desc, int32_t *keys_a, int32_t *vals_a, int32_t *keys_b, int32_t *vals_b,
                               int2 *smem_pairs)
{
    const int a_begin        = desc.x;
    const int a_end          = desc.y;
    const int b_end          = desc.z;
    const bool swap_buffers  = (desc.w < 0);
    const int partition_diag = desc.w & 0x7FFFFFFF;

    // uniform branch: all threads in the block read from the same buffer
    const int32_t *keys_in = swap_buffers ? keys_b : keys_a;
    const int32_t *vals_in = swap_buffers ? vals_b : vals_a;
    int32_t *keys_out      = swap_buffers ? keys_a : keys_b;
    int32_t *vals_out      = swap_buffers ? vals_a : vals_b;

    const int a_len       = a_end - a_begin;
    const int b_len       = b_end - a_end;
    const int merge_total = a_len + b_len;
    const int out_end     = min(partition_diag + MERGE_PARTITION_SIZE, merge_total);
    const int out_count   = out_end - partition_diag;

    if (out_count <= 0)
    {
        return;
    }

    // degenerate copy node (odd leftover, empty B run): skip merge-path
    // and just copy A → output with coalesced accesses
    if (b_len == 0)
    {
#pragma unroll 1
        for (int i = threadIdx.x; i < out_count; i += MERGE_CTA_SIZE)
        {
            keys_out[a_begin + partition_diag + i] = keys_in[a_begin + partition_diag + i];
            vals_out[a_begin + partition_diag + i] = vals_in[a_begin + partition_diag + i];
        }
        return;
    }

    const int32_t *a_keys = keys_in + a_begin;
    const int32_t *b_keys = keys_in + a_end;
    const int32_t *a_vals = vals_in + a_begin;
    const int32_t *b_vals = vals_in + a_end;

    constexpr int2 SENTINEL = {INT_MAX, 0};

    // CTA-level merge-path: warp-cooperative binary search on DRAM.
    // warp 0 finds mp_start (at partition_diag), warp 1 finds mp_stop (at out_end)
    // in parallel. 32 probes per round divide [lo,hi) into 33 sub-intervals,
    // __popc on the ballot locates the crossover, narrowing by ~33x per round.
    // completes in ~3 rounds for 8K-element runs vs ~13 scalar iterations.
    constexpr int MP_BROADCAST_IDX = MERGE_PARTITION_SIZE + 2;
    if (threadIdx.x < 64)
    {
        const int lane    = threadIdx.x & 31;
        const int warp_id = threadIdx.x >> 5;
        const int diag    = warp_id == 0 ? partition_diag : out_end;

        // exact a * b / 33 without 64-bit math; b must be in [1, 32].
        // the /33 literals are compile-time constants so the compiler emits
        // multiply-high + shift, not actual division.
        auto div33 = [](int a, int b) {
            const int q = a / 33;
            const int r = a - q * 33;
            return q * b + r * b / 33;
        };

        int lo = max(0, diag - b_len);
        int hi = min(diag, a_len);
        while (lo < hi)
        {
            const int range        = hi - lo;
            const int probe        = lo + div33(range, lane + 1);
            const bool take_more_a = (a_keys[probe] <= b_keys[diag - probe - 1]);
            const uint32_t ballot  = __ballot_sync(0xFFFFFFFF, take_more_a);
            const int n_true       = __popc(ballot);

            // 32 probes create 33 sub-intervals. n_true tells us which one
            // contains the crossover. the general formula narrows to
            // [probe[n_true-1]+1, probe[n_true]], but the edge cases need
            // special handling: n_true=0 has no "last true probe" so lo must
            // stay put, and n_true=32 has no "first false probe" so hi stays.
            if (n_true == 32)
            {
                lo = lo + div33(range, 32) + 1;
            }
            else if (n_true == 0)
            {
                hi = lo + div33(range, 1);
            }
            else
            {
                hi = lo + div33(range, n_true + 1);
                lo = lo + div33(range, n_true) + 1;
            }
        }

        if (lane == 0)
        {
            reinterpret_cast<int *>(&smem_pairs[MP_BROADCAST_IDX])[warp_id] = lo;
        }
    }

    __syncthreads();
    const int a_start = smem_pairs[MP_BROADCAST_IDX].x;
    const int a_count = smem_pairs[MP_BROADCAST_IDX].y - a_start;
    const int b_start = partition_diag - a_start;
    const int b_count = out_count - a_count;
    const int b_base  = a_count + 1;

    // fused coalesced loads: key+val packed into smem with sentinel gap
#pragma unroll 1
    for (int i = threadIdx.x; i < a_count; i += MERGE_CTA_SIZE)
    {
        smem_pairs[i] = make_int2(a_keys[a_start + i], a_vals[a_start + i]);
    }
#pragma unroll 1
    for (int i = threadIdx.x; i < b_count; i += MERGE_CTA_SIZE)
    {
        smem_pairs[b_base + i] = make_int2(b_keys[b_start + i], b_vals[b_start + i]);
    }
    if (threadIdx.x == 0)
    {
        smem_pairs[a_count]          = SENTINEL;
        smem_pairs[b_base + b_count] = SENTINEL;
    }
    __syncthreads();

    // thread-level merge-path on smem: binary search for the split between
    // A and B at this thread's diagonal. uses <= for stability (equal keys prefer A).
    const int t_diag = min((int)threadIdx.x * MERGE_ITEMS_PER_THREAD, out_count);
    int ta_lo        = max(0, t_diag - b_count);
    int ta_hi        = min(t_diag, a_count);
    while (ta_lo < ta_hi)
    {
        const int mid = (ta_lo + ta_hi) / 2;
        if (smem_pairs[mid].x <= smem_pairs[b_base + t_diag - mid - 1].x)
        {
            ta_lo = mid + 1;
        }
        else
        {
            ta_hi = mid;
        }
    }

    // sequential merge into packed registers. prefetch both pairs;
    // unconditional reload — smem has IPT padding slots past the
    // sentinels so reads are safe even for partial-tile threads.
    const int2 *a_pair_ptr = &smem_pairs[ta_lo];
    const int2 *b_pair_ptr = &smem_pairs[b_base + t_diag - ta_lo];
    int2 a_pair            = *a_pair_ptr;
    int2 b_pair            = *b_pair_ptr;
    int2 r_kv[MERGE_ITEMS_PER_THREAD];

#pragma unroll
    for (int i = 0; i < MERGE_ITEMS_PER_THREAD; i++)
    {
        if (a_pair.x <= b_pair.x)
        {
            r_kv[i] = a_pair;
            a_pair  = *(++a_pair_ptr);
        }
        else
        {
            r_kv[i] = b_pair;
            b_pair  = *(++b_pair_ptr);
        }
    }
    __syncthreads();

    // blocked scatter then striped read+store through smem.
    // IPT=15 gives stride 15 int2 between lanes; GCD(15, 16 wide banks) = 1,
    // so all lanes hit different bank pairs (optimal 2-way conflict).
#pragma unroll
    for (int i = 0; i < MERGE_ITEMS_PER_THREAD; i++)
    {
        const int blocked = threadIdx.x * MERGE_ITEMS_PER_THREAD + i;
        if (blocked < out_count)
        {
            smem_pairs[blocked] = r_kv[i];
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < MERGE_ITEMS_PER_THREAD; i++)
    {
        const int striped = threadIdx.x + i * MERGE_CTA_SIZE;
        if (striped < out_count)
        {
            const int2 pair                              = smem_pairs[striped];
            keys_out[a_begin + partition_diag + striped] = pair.x;
            vals_out[a_begin + partition_diag + striped] = pair.y;
        }
    }
}

// strictly out-of-place: each partition reads from one buffer pair and writes
// to the other (selected by the per-partition direction flag). the input and
// output buffer pairs must not alias.
__global__ void __launch_bounds__(MERGE_CTA_SIZE, MERGE_MIN_BLOCKS) MergeKernel(const MergeArgs args)
{
    // smem layout: [A(a_count) | sentinel | B(b_count) | sentinel | padding]
    // IPT extra slots past data guarantee unconditional merge reloads are
    // safe even for partial-tile threads (no bounds checks in the loop).
    __shared__ int2 smem_pairs[MERGE_PARTITION_SIZE + MERGE_ITEMS_PER_THREAD];
    MergePartition(args.partitions[blockIdx.x], args.keys_a, args.vals_a, args.keys_b, args.vals_b, smem_pairs);
}

// signal partition completion and enqueue the parent node when all children
// are done. warp-cooperative: must be called by warp 0 (threadIdx.x < 32).
// lane 0 handles atomics and conditionals, all lanes cooperate on writing
// queue entries via a strided loop.
__device__ void UpdateMergeQueue(const PersistentMergeArgs &args, const MergeWorkItem &work_item,
                                 const MergeTreeNode &node)
{
    const int lane         = threadIdx.x & 31;
    const int n_partitions = (node.end - node.begin + MERGE_PARTITION_SIZE - 1) / MERGE_PARTITION_SIZE;
    const int parent       = node.parent_idx;

    // speculative read of the parent node — one L2 transaction (all lanes
    // read the same 16B address). skipped for root nodes (parent < 0).
    int parent_n_partitions = 0;
    if (parent >= 0)
    {
        const MergeTreeNode parent_node = args.tree_nodes[parent];
        parent_n_partitions = (parent_node.end - parent_node.begin + MERGE_PARTITION_SIZE - 1) / MERGE_PARTITION_SIZE;
    }

    // lane 0 handles the atomic completion chain and reserves queue slots.
    // q_base < 0 signals "nothing to enqueue" to the rest of the warp.
    int q_base = -1;
    if (lane == 0)
    {
        const int done = atomicAdd(&args.node_done_counts[work_item.node_idx], 1) + 1;
        if (done == n_partitions && parent >= 0)
        {
            const int prev = atomicAdd(&args.node_done_counts[parent], 1);
            if (prev == -1)
            {
                q_base = atomicAdd(&args.queue_state->queue_reserved, parent_n_partitions);
            }
        }
    }

    q_base = __shfl_sync(0xFFFFFFFF, q_base, 0);
    if (q_base < 0)
    {
        return;
    }

#pragma unroll 1
    for (int t = lane; t < parent_n_partitions; t += 32)
    {
        args.work_items[q_base + t] = {parent, t};
    }

    __syncwarp();
    if (lane == 0)
    {
        __threadfence(); // release: entry writes visible before queue_tail commit
        while (atomicCAS(&args.queue_state->queue_tail, q_base, q_base + parent_n_partitions) != q_base)
        {
            __nanosleep(32);
        }
    }
}

// persistent-CTA merge kernel driven by a device-side FIFO queue.
// each CTA loops: claim a queue slot, spin until committed, merge
// the partition, signal completion, and enqueue parent nodes when
// both children are done. terminates when all queue items are consumed.
//
// strictly out-of-place per partition (same semantics as MergeKernel):
// the two buffer pairs (a and b) must not alias.
__global__ void __launch_bounds__(MERGE_CTA_SIZE, MERGE_MIN_BLOCKS)
    MergePersistentKernel(const PersistentMergeArgs args)
{
    const int total = args.queue_state->total_queue_items;
    if (total == 0)
    {
        return;
    }

    // smem layout: [A(a_count) | sentinel | B(b_count) | sentinel | padding]
    // IPT extra slots past data guarantee unconditional merge reloads are
    // safe even for partial-tile threads (no bounds checks in the loop).
    // the first int is aliased as the queue-position broadcast slot, safe
    // because it is consumed into registers before MergePartition overwrites smem.
    __shared__ int2 smem_pairs[MERGE_PARTITION_SIZE + MERGE_ITEMS_PER_THREAD];

    while (true)
    {
        if (threadIdx.x == 0)
        {
            reinterpret_cast<int *>(smem_pairs)[0] = atomicAdd(&args.queue_state->queue_head, 1);
        }
        __syncthreads(); // broadcast pos to all threads
        const int pos = reinterpret_cast<int *>(smem_pairs)[0];
        if (pos >= total)
        {
            return;
        }

        // spin until queue_tail has advanced past our slot (producer committed)
        if (threadIdx.x == 0)
        {
            while (pos >= LoadRelaxedGpu(&args.queue_state->queue_tail))
            {
                __nanosleep(32);
            }
        }
        __syncthreads(); // broadcast spin completion to all threads
        __threadfence(); // acquire: see producer's entry + data writes

        const MergeWorkItem work_item = args.work_items[pos];
        const MergeTreeNode node      = args.tree_nodes[work_item.node_idx];
        const int merge_stride        = node.stride_and_flag & 0x7FFFFFFF;
        const int flag                = node.stride_and_flag & 0x80000000;
        const int a_end               = min(node.begin + merge_stride, node.end);
        const int4 desc =
            make_int4(node.begin, a_end, node.end, (work_item.partition_idx * MERGE_PARTITION_SIZE) | flag);

        MergePartition(desc, args.keys_a, args.vals_a, args.keys_b, args.vals_b, smem_pairs);

        __threadfence(); // release: per-thread, flush this thread's output writes
        __syncthreads(); // ensure all threads have fenced before warp 0 signals

        if (threadIdx.x < 32)
        {
            UpdateMergeQueue(args, work_item, node);
        }
        // no __syncthreads() needed: next iteration's s_pos broadcast synchronizes
    }
}

// emit CAPACITY-chunked overflow blocks for T3 overflow re-sort.
// sign bit of the block descriptor's begin field tells the overflow kernel
// which ping-pong buffer to write to: k-odd segments start from B, k-even
// from A, so that after all merge passes every segment's result lands in
// buffer A.
// CTA-cooperative: must be called by all CTA_SIZE threads.
template<int CTA_SIZE>
__device__ void WriteOverflowBlocks(const BlockSortArgs &args, int seg_start, int seg_end, int seg_size,
                                    int n_merge_passes, int32_t *smem)
{
    constexpr int CAPACITY = CTA_SIZE * BLOCK_SORT_ITEMS_PER_THREAD;
    const int flag         = (n_merge_passes & 1) << 31;
    const int n_blocks     = (seg_size + CAPACITY - 1) / CAPACITY;
    if (threadIdx.x == 0)
    {
        smem[0] = atomicAdd(&args.overflow_state->block_count, n_blocks);
    }
    __syncthreads();
    const int base_idx = smem[0];
#pragma unroll 1
    for (int c = threadIdx.x; c < n_blocks; c += CTA_SIZE)
    {
        args.overflow_blocks[base_idx + c] =
            make_int2((seg_start + c * CAPACITY) | flag, min(seg_start + (c + 1) * CAPACITY, seg_end));
    }
}

// build a binary merge tree for one overflow segment and enqueue leaf-level
// partitions into the FIFO queue.
//
// the segment was pre-sorted into sorted_run_count = ceil(seg_size / CAPACITY)
// fixed-size runs by the block-sort phase. the merge tree groups adjacent runs
// at each level, doubling the merged run size until one sorted run remains:
//
//   level 0 (leaves): nodes merging 1x-capacity runs  → 2x-capacity merged runs
//   level 1:          nodes merging 2x-capacity runs  → 4x-capacity merged runs
//   ...
//   level n_merge_passes-1 (root): final merge covering the entire segment
//
// every level includes copy nodes for odd leftovers (degenerate node with
// empty B run) so all runs land in the correct ping-pong buffer.
//
// the tree is laid out contiguously in the global tree_nodes array starting
// at tree_base_idx (reserved via atomicAdd). leaf-level partitions are
// enqueued into the FIFO queue with a single batched atomicAdd on queue_tail.
//
// CTA-cooperative: must be called by all CTA_SIZE threads.
template<int CTA_SIZE>
__device__ void WriteMergeTree(const BlockSortArgs &args, int seg_start, int seg_end, int seg_size, int n_merge_passes,
                               int32_t *smem)
{
    constexpr int CAPACITY             = CTA_SIZE * BLOCK_SORT_ITEMS_PER_THREAD;
    constexpr int FULL_LEAF_NODE_SIZE  = 2 * CAPACITY;
    constexpr int FULL_LEAF_PARTITIONS = (FULL_LEAF_NODE_SIZE + MERGE_PARTITION_SIZE - 1) / MERGE_PARTITION_SIZE;

    const int sorted_run_count = (seg_size + CAPACITY - 1) / CAPACITY;
    const int leaf_node_count  = (sorted_run_count + 1) / 2;

    // ping-pong direction flag for leaf level: ensures the final merge pass
    // writes to buffer A regardless of how many passes there are
    const int leaf_ping_pong_flag = (n_merge_passes & 1) << 31;

    // total tree nodes via Legendre's formula (p=2):
    // sum_{k=1}^{P} ceil(n/2^k) = (n-1) - popcount(n-1) + P
    const int total_node_count = (sorted_run_count - 1) - __popc(sorted_run_count - 1) + n_merge_passes;

    // every leaf node except the last covers exactly FULL_LEAF_NODE_SIZE items
    // and produces exactly FULL_LEAF_PARTITIONS merge partitions. the last
    // node may be smaller (partial second run or single-run copy node).
    const int last_leaf_begin = seg_start + (leaf_node_count - 1) * FULL_LEAF_NODE_SIZE;
    const int last_leaf_end   = min(min(last_leaf_begin + CAPACITY, seg_end) + CAPACITY, seg_end);
    const int last_leaf_partitions =
        (last_leaf_end - last_leaf_begin + MERGE_PARTITION_SIZE - 1) / MERGE_PARTITION_SIZE;
    const int leaf_partition_count = (leaf_node_count - 1) * FULL_LEAF_PARTITIONS + last_leaf_partitions;

    // batch all global reservations in one thread-0 block, broadcast via smem.
    //   smem[0]: queue_base  — starting index in the FIFO queue for leaf entries
    //   smem[1]: tree_base   — starting index in the tree_nodes array for this tree
    //   smem[2]: accumulator — per-thread internal partition counts (smem atomic)
    int *smem_queue_base     = &smem[0];
    int *smem_tree_base      = &smem[1];
    int *smem_internal_parts = &smem[2];
    if (threadIdx.x == 0)
    {
        atomicAdd(&args.merge_queue_state->queue_reserved, leaf_partition_count);
        *smem_queue_base     = atomicAdd(&args.merge_queue_state->queue_tail, leaf_partition_count);
        *smem_tree_base      = atomicAdd(&args.merge_queue_state->total_tree_nodes, total_node_count);
        *smem_internal_parts = 0;
    }
    __syncthreads();

    const int tree_base_idx = *smem_tree_base;

    // -- leaf level (d=0) --
    // write node metadata and done counters. leaf nodes at indices
    // [tree_base_idx, tree_base_idx + leaf_node_count). each leaf's parent
    // is at the next level: leaf p's parent is the (p/2)-th node at level 1.
#pragma unroll 1
    for (int p = threadIdx.x; p < leaf_node_count; p += CTA_SIZE)
    {
        const int node_begin = seg_start + p * FULL_LEAF_NODE_SIZE;
        const int node_end   = (p < leaf_node_count - 1) ? node_begin + FULL_LEAF_NODE_SIZE : last_leaf_end;
        const int node_idx   = tree_base_idx + p;
        const int parent     = (n_merge_passes > 1) ? tree_base_idx + leaf_node_count + p / 2 : -1;

        args.tree_nodes[node_idx]       = {.begin           = node_begin,
                                           .end             = node_end,
                                           .stride_and_flag = leaf_ping_pong_flag | CAPACITY,
                                           .parent_idx      = parent};
        args.node_done_counts[node_idx] = 0;
    }

    // enqueue leaf partitions into the FIFO queue. flat CTA-strided loop over
    // all leaf_partition_count entries. node index and partition index within
    // the node are derived from the flat index via constexpr division by
    // FULL_LEAF_PARTITIONS (compiler emits multiply-high, no actual division).
    // the last node may have fewer partitions, handled by the else branch.
#pragma unroll 1
    for (int i = threadIdx.x; i < leaf_partition_count; i += CTA_SIZE)
    {
        int leaf_node_idx;
        int partition_idx;
        if (i < (leaf_node_count - 1) * FULL_LEAF_PARTITIONS)
        {
            leaf_node_idx = i / FULL_LEAF_PARTITIONS;
            partition_idx = i - leaf_node_idx * FULL_LEAF_PARTITIONS;
        }
        else
        {
            leaf_node_idx = leaf_node_count - 1;
            partition_idx = i - (leaf_node_count - 1) * FULL_LEAF_PARTITIONS;
        }
        args.work_items[*smem_queue_base + i] = {.node_idx      = tree_base_idx + leaf_node_idx,
                                                 .partition_idx = partition_idx};
    }

    // -- internal levels (d=1 .. n_merge_passes-1) --
    // internal nodes are numbered 0..internal_node_count-1 across all levels:
    //   [0, count_at_level_1)                                  → level 1
    //   [count_at_level_1, count_at_level_1 + count_at_level_2) → level 2
    //   ...
    // each thread locates its level by scanning cumulative counts in registers.
    // node count at level k: ((sorted_run_count - 1) >> (k + 1)) + 1
    const int internal_node_count = total_node_count - leaf_node_count;
    int local_internal_parts      = 0;

#pragma unroll 1
    for (int flat_idx = threadIdx.x; flat_idx < internal_node_count; flat_idx += CTA_SIZE)
    {
        // scan cumulative node counts to find which level this node belongs to
        int tree_level  = 1;
        int level_start = 0;
        int cumulative  = 0;
#pragma unroll 1
        for (int k = 1; k < n_merge_passes - 1; k++)
        {
            cumulative += ((sorted_run_count - 1) >> (k + 1)) + 1;
            if (flat_idx >= cumulative)
            {
                tree_level  = k + 1;
                level_start = cumulative;
            }
        }
        const int pos_in_level = flat_idx - level_start;

        // merge node geometry: at level d, each node merges two runs of
        // (CAPACITY << d) items, starting at seg_start + pos * 2 * stride
        const int merge_stride = CAPACITY << tree_level;
        const int node_begin   = seg_start + pos_in_level * 2 * merge_stride;
        const int node_end     = min(min(node_begin + merge_stride, seg_end) + merge_stride, seg_end);
        const int n_partitions = (node_end - node_begin + MERGE_PARTITION_SIZE - 1) / MERGE_PARTITION_SIZE;
        local_internal_parts += n_partitions;

        // position of this node within the segment's tree
        const int tree_offset = leaf_node_count + level_start + pos_in_level;
        const int node_idx    = tree_base_idx + tree_offset;

        // parent is at the next level, position pos_in_level / 2
        int parent = -1;
        if (tree_level + 1 < n_merge_passes)
        {
            const int nodes_at_this_level = ((sorted_run_count - 1) >> (tree_level + 1)) + 1;
            const int next_level_start    = tree_offset - pos_in_level + nodes_at_this_level;
            parent                        = tree_base_idx + next_level_start + pos_in_level / 2;
        }

        const int ping_pong_flag  = ((tree_level & 1) ^ (n_merge_passes & 1)) << 31;
        args.tree_nodes[node_idx] = {.begin           = node_begin,
                                     .end             = node_end,
                                     .stride_and_flag = ping_pong_flag | merge_stride,
                                     .parent_idx      = parent};

        // child count: 2 if both child nodes exist at the level below,
        // 1 if this is a copy node (odd leftover at the end of that level)
        const int child_node_count      = ((sorted_run_count - 1) >> tree_level) + 1;
        const int n_children            = (2 * pos_in_level + 1 < child_node_count) ? 2 : 1;
        args.node_done_counts[node_idx] = -n_children;
    }

    // reduce per-thread internal partition counts via smem atomic,
    // then thread 0 commits the total to the global queue state
    if (local_internal_parts > 0)
    {
        atomicAdd(smem_internal_parts, local_internal_parts);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicAdd(&args.merge_queue_state->total_queue_items, leaf_partition_count + *smem_internal_parts);
    }
}

// radix sort one segment in-CTA: load warp-striped, rank, scatter, write coalesced.
// warp-striped arrangement is required by BlockRadixRankMatch for stability:
// the ranking loop processes items in ITEM-major order (all lanes' ITEM=0,
// then ITEM=1, ...). for equal-digit items to preserve original order, ITEM=0
// across lanes must map to logically earlier positions than ITEM=1. warp-striped
// satisfies this (lane l's ITEM=i → position w*WS*IPT + l + i*WS, monotone in i).
template<int CTA_SIZE>
__device__ void BlockRadixSortSegment(BlockSortSmem<CTA_SIZE> &smem, const int32_t *keys_in, const int32_t *values_in,
                                      int32_t *out_keys, int32_t *out_vals, int seg_start, int seg_size)
{
    constexpr int WARP_THREADS = 32;
    constexpr int END_BIT      = 32;

    const int warp_id     = threadIdx.x / WARP_THREADS;
    const int lane        = threadIdx.x % WARP_THREADS;
    const int warp_offset = warp_id * WARP_THREADS * BLOCK_SORT_ITEMS_PER_THREAD;

    // load in warp-striped order: thread (warp w, lane l) owns logical positions
    //   [w*WS*IPT + l, w*WS*IPT + l+WS, w*WS*IPT + l+2*WS, ...].
    // blocked arrangement would break stability because lane l's ITEM=i maps to
    // l*IPT + i, so lane 1's ITEM=0 (= IPT) can be logically after lane 0's
    // ITEM=1 (= 1), yet it would be ranked first.
    // the layout is also naturally coalesced: consecutive lanes access consecutive
    // addresses within each warp.
    uint32_t keys[BLOCK_SORT_ITEMS_PER_THREAD];
    int32_t vals[BLOCK_SORT_ITEMS_PER_THREAD];

#pragma unroll
    for (int i = 0; i < BLOCK_SORT_ITEMS_PER_THREAD; i++)
    {
        const int idx = warp_offset + lane + i * WARP_THREADS;
        if (idx < seg_size)
        {
            keys[i] = static_cast<uint32_t>(keys_in[seg_start + idx]) ^ 0x80000000u;
            vals[i] = values_in[seg_start + idx];
        }
        else
        {
            keys[i] = 0xFFFFFFFFu;
            vals[i] = 0;
        }
    }

    int begin_bit = 0;
    while (true)
    {
        const int pass_bits = min(BLOCK_SORT_RADIX_BITS, END_BIT - begin_bit);
        const cub::BFEDigitExtractor<uint32_t> extractor(begin_bit, pass_bits);

        int ranks[BLOCK_SORT_ITEMS_PER_THREAD];
        BlockSortRankT<CTA_SIZE>(smem.rank).RankKeys(keys, ranks, extractor);
        begin_bit += BLOCK_SORT_RADIX_BITS;
        __syncthreads();

        if (begin_bit >= END_BIT)
        {
            // last pass: no further ranking, so no need to maintain warp-striped.
            // scatter vals to striped (thread t gets positions [t, t+BT, t+2*BT, ...])
            // for coalesced output writes. if keys_out is requested, scatter keys too.
#pragma unroll
            for (int i = 0; i < BLOCK_SORT_ITEMS_PER_THREAD; i++)
            {
                smem.raw[ranks[i]] = vals[i];
            }
            __syncthreads();
#pragma unroll
            for (int i = 0; i < BLOCK_SORT_ITEMS_PER_THREAD; i++)
            {
                vals[i] = smem.raw[threadIdx.x + i * CTA_SIZE];
            }

            if (out_keys)
            {
                __syncthreads();
#pragma unroll
                for (int i = 0; i < BLOCK_SORT_ITEMS_PER_THREAD; i++)
                {
                    smem.raw[ranks[i]] = static_cast<int32_t>(keys[i] ^ 0x80000000u);
                }
                __syncthreads();
#pragma unroll
                for (int i = 0; i < BLOCK_SORT_ITEMS_PER_THREAD; i++)
                {
                    keys[i] = static_cast<uint32_t>(smem.raw[threadIdx.x + i * CTA_SIZE]);
                }
            }
            break;
        }

        // scatter items to their ranked positions, then read back in warp-striped
        // order to re-establish the arrangement for the next ranking pass.
        auto scatterWarpStriped = [&](auto &items, const int(&ranks)[BLOCK_SORT_ITEMS_PER_THREAD]) {
#pragma unroll
            for (int i = 0; i < BLOCK_SORT_ITEMS_PER_THREAD; i++)
            {
                smem.raw[ranks[i]] = items[i];
            }
            __syncthreads();
#pragma unroll
            for (int i = 0; i < BLOCK_SORT_ITEMS_PER_THREAD; i++)
            {
                items[i] = smem.raw[warp_offset + lane + i * WARP_THREADS];
            }
            __syncthreads();
        };

        scatterWarpStriped(keys, ranks);
        scatterWarpStriped(vals, ranks);
    }

    // write output in striped arrangement (coalesced: consecutive threads write
    // consecutive addresses).
#pragma unroll
    for (int i = 0; i < BLOCK_SORT_ITEMS_PER_THREAD; i++)
    {
        const int idx = threadIdx.x + i * CTA_SIZE;
        if (idx < seg_size)
        {
            out_vals[seg_start + idx] = vals[i];
            if (out_keys)
            {
                out_keys[seg_start + idx] = static_cast<int32_t>(keys[i]);
            }
        }
    }
}

// Block-level segmented sort kernel.
// All tiers target 512 threads/SM occupancy: min_blocks = 512 / CTA_SIZE.
// This constrains the compiler's register allocation uniformly across tiers.
//
// supports in-place operation (keys_in == keys_out and/or values_in == values_out
// for the same segment range) because each CTA loads its entire segment into
// registers before writing any output. different segments occupy disjoint
// index ranges, so no cross-CTA aliasing hazard exists.
//
// IS_PERSISTENT: persistent-CTA mode for overflow block sorting.
// a fixed grid of CTAs loops, claiming block indices from an atomic
// counter until all work is consumed. avoids host sync to read the count.
template<int CTA_SIZE, int MIN_SIZE = 0, bool IS_EPILOG = false, bool IS_PERSISTENT = false>
__global__ void __launch_bounds__(CTA_SIZE, 512 / CTA_SIZE) BlockSortKernel(const BlockSortArgs args)
{
#if __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    constexpr int CAPACITY = CTA_SIZE * BLOCK_SORT_ITEMS_PER_THREAD;
    __shared__ BlockSortSmem<CTA_SIZE> smem;

    if constexpr (IS_PERSISTENT)
    {
        const int n_blocks = args.overflow_state->block_count;

        while (true)
        {
            if (threadIdx.x == 0)
            {
                smem.raw[0] = atomicAdd(&args.overflow_state->claim_counter, 1);
            }
            __syncthreads();
            const int block_idx = smem.raw[0];
            if (block_idx >= n_blocks)
            {
                return;
            }

            const int2 block_desc = args.overflow_blocks[block_idx];
            const bool to_b       = (block_desc.x < 0);
            const int block_begin = block_desc.x & 0x7FFFFFFF;
            const int block_size  = block_desc.y - block_begin;

            int32_t *out_vals = to_b ? args.values_out_b : args.values_out;
            int32_t *out_keys = to_b ? args.keys_out_b : args.keys_out;

            BlockRadixSortSegment<CTA_SIZE>(smem, args.keys_in, args.values_in, out_keys, out_vals, block_begin,
                                            block_size);
            // no __syncthreads() needed here: the barrier at the top of the
            // next iteration (broadcasting smem.raw[0]) synchronizes all threads.
            // BlockRadixSortSegment's last smem accesses are reads into registers;
            // the subsequent DRAM output writes don't touch smem.
        }
    }
    else
    {
        const int seg_idx = blockIdx.x;
        int seg_start, seg_end;

        // when offsets == nullptr this is the standalone overflow block-sort
        // path; the sign bit of the block descriptor's begin field selects
        // which ping-pong buffer to write to (set by the epilog based on
        // n_merge_passes parity so that all segments converge to buffer A
        // after the merge passes).
        int32_t *out_vals = args.values_out;
        int32_t *out_keys = args.keys_out;
        if (args.offsets)
        {
            seg_start = args.offsets[seg_idx];
            seg_end   = args.offsets[seg_idx + 1];
        }
        else
        {
            const int2 block_desc = args.overflow_blocks[seg_idx];
            const bool to_b       = (block_desc.x < 0);
            seg_start             = block_desc.x & 0x7FFFFFFF;
            seg_end               = block_desc.y;
            if (to_b)
            {
                out_vals = args.values_out_b;
                out_keys = args.keys_out_b;
            }
        }
        const int seg_size = seg_end - seg_start;

        if (seg_size <= MIN_SIZE || seg_size > CAPACITY)
        {
            if constexpr (IS_EPILOG)
            {
                if (seg_size > CAPACITY)
                {
                    const int n_merge_passes = 32 - __clz((seg_size - 1) / CAPACITY);

                    WriteOverflowBlocks<CTA_SIZE>(args, seg_start, seg_end, seg_size, n_merge_passes, smem.raw);

                    if (args.merge_queue_state)
                    {
                        WriteMergeTree<CTA_SIZE>(args, seg_start, seg_end, seg_size, n_merge_passes, smem.raw);
                    }
                }
            }
            return;
        }

        BlockRadixSortSegment<CTA_SIZE>(smem, args.keys_in, args.values_in, out_keys, out_vals, seg_start, seg_size);
    }
}

} // namespace seg_sort

// two-pass setup for the segmented sort scratch buffer.
// pass 1 (scratch == nullptr): computes and returns the required byte count.
// pass 2 (scratch != nullptr): lays out sub-regions within the provided buffer
//   and populates the state struct. always returns the total size.
//
// capacities are worst-case upper bounds assuming any single overflow segment
// could span all n_items. each entry has an explicit alignment: 32 bytes for
// structs containing L2-sector-padded atomic counters, 256 bytes for bulk
// arrays.
size_t SegmentedSortSetup(SegmentedSortState &s, int32_t n_segments, int32_t n_items, void *scratch)
{
    using namespace seg_sort;

    // max merge passes for one segment
    const int n_max_merge_passes = (int)std::ceil(std::log2(std::fmax((double)n_items / BLOCK_SORT_T3_CAPACITY, 1.0)));
    // upper bound on overflow block descriptors across all segments
    const int overflow_blocks_cap = n_items / BLOCK_SORT_T3_CAPACITY + n_segments;
    // upper bound on tree nodes across all overflow segments.
    // per-segment node count via Legendre: nodes(r) = (r-1) - popcount(r-1) + P.
    // the ratio nodes(r)/r peaks at ~1.25 for small r and approaches 1.0 for
    // large r. 3/2 is a proven upper bound: nodes(r) ≤ r + ceil(log2(r)) - 1 ≤ 3r/2.
    const int max_tree_nodes = overflow_blocks_cap + overflow_blocks_cap / 2;
    // upper bound on total merge queue items across all tree levels
    const int merge_pass_stride = n_items / MERGE_PARTITION_SIZE + 1;
    const int max_merge_queue   = n_max_merge_passes * merge_pass_stride;

    // zero-init fields MUST come first: they are zeroed with a single memset
    // from the scratch base before each sort run. changing their order or
    // moving non-zero-init fields before them will break the memset.
    //
    // alignment = 32 for structs whose members are individually padded to
    // 32-byte L2 sector boundaries (OverflowState, MergeQueueState), so
    // concurrent atomics to different counters never share an L2 sector.
    // alignment = 256 for bulk arrays (cache-line alignment).
    constexpr int N_ZERO_INIT       = 2;
    constexpr int N_ENTRIES         = 6;
    ScratchEntry entries[N_ENTRIES] = {
        {s.overflow_state, 1, 32},
        {s.merge_queue_state, 1, 32},
        {s.node_done_counts, max_tree_nodes, 256},
        {s.work_items, max_merge_queue, 256},
        {s.overflow_blocks, overflow_blocks_cap, 256},
        {s.tree_nodes, max_tree_nodes, 256},
    };

    auto alignUp = [](size_t x, size_t a) -> size_t { return (x + a - 1) & ~(a - 1); };

    size_t offsets[N_ENTRIES];
    size_t total = 0;
    for (int i = 0; i < N_ENTRIES; i++)
    {
        total      = alignUp(total, entries[i].alignment);
        offsets[i] = total;
        total += entries[i].elem_size * entries[i].count;
    }
    total = alignUp(total, 256);

    if (!scratch)
    {
        return total;
    }

    s.n_segments         = n_segments;
    s.n_items            = n_items;
    s.n_max_merge_passes = n_max_merge_passes;

    char *base = static_cast<char *>(scratch);
    for (int i = 0; i < N_ENTRIES; i++)
    {
        *entries[i].ptr = entries[i].count > 0 ? base + offsets[i] : nullptr;
    }

    s.run_init_ptr  = scratch;
    s.run_init_size = offsets[N_ZERO_INIT];

    if (!s.sm_count)
    {
        cudaDeviceGetAttribute(&s.sm_count, cudaDevAttrMultiProcessorCount, 0);
    }

    return total;
}

int SegmentedSortAsync(const SegmentedSortState &st, const int32_t *d_offsets, int32_t *const *d_keys,
                       int32_t *const *d_values, cudaStream_t s)
{
    using namespace seg_sort;

    if (st.n_segments <= 0 || st.n_items <= 0)
    {
        return 0;
    }

    cudaMemsetAsync(st.run_init_ptr, 0, st.run_init_size, s);

    // buffer A = caller's index 0 (input + final result)
    // buffer B = caller's index 1 (working space)

    // cascade tiers T0-T2: sort segments that fit within their capacity.
    // values are written in-place back to buffer A (safe: single CTA per
    // segment loads all items into registers before writing).
    const BlockSortArgs cascade_args = {.n_segments = st.n_segments,
                                        .n_items    = st.n_items,
                                        .offsets    = d_offsets,
                                        .keys_in    = d_keys[0],
                                        .values_in  = d_values[0],
                                        .values_out = d_values[0]};

    // cascade tier T3 (epilog): sorts T3-sized segments and emits overflow
    // descriptors (blocks, merge tree) for segments that exceed T3 capacity
    const BlockSortArgs cascade_epilog_args = {.n_segments        = st.n_segments,
                                               .n_items           = st.n_items,
                                               .offsets           = d_offsets,
                                               .keys_in           = d_keys[0],
                                               .values_in         = d_values[0],
                                               .values_out        = d_values[0],
                                               .overflow_blocks   = st.overflow_blocks,
                                               .overflow_state    = st.overflow_state,
                                               .merge_queue_state = st.merge_queue_state,
                                               .tree_nodes        = st.tree_nodes,
                                               .node_done_counts  = st.node_done_counts,
                                               .work_items        = st.work_items};

    // persistent overflow: re-sorts oversized segments in T3-capacity blocks.
    // fixed grid reads block_count on-device; each CTA claims overflow blocks
    // via atomic counter until all are processed.
    // if there are no overflow segments, all CTAs exit immediately.
    const BlockSortArgs overflow_args = {.n_items         = st.n_items,
                                         .keys_in         = d_keys[0],
                                         .values_in       = d_values[0],
                                         .values_out      = d_values[0],
                                         .keys_out        = d_keys[0],
                                         .overflow_blocks = st.overflow_blocks,
                                         .overflow_state  = st.overflow_state,
                                         .values_out_b    = d_values[1],
                                         .keys_out_b      = d_keys[1]};

    // PDL attribute: allows the driver to overlap consecutive tier launches on
    // SM 9.0+. on older hardware the attribute is silently ignored and the
    // kernels execute with normal stream serialization.
    cudaLaunchAttribute pdl_attr;
    pdl_attr.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
    pdl_attr.val.programmaticStreamSerializationAllowed = 1;

    auto launchPdl = [&](auto kernel_fn, int block_threads, const BlockSortArgs &args) {
        cudaLaunchConfig_t config = {};
        config.gridDim            = dim3(st.n_segments);
        config.blockDim           = dim3(block_threads);
        config.stream             = s;
        config.attrs              = &pdl_attr;
        config.numAttrs           = 1;
        cudaLaunchKernelEx(&config, kernel_fn, args);
    };

    // T0: first in chain, normal launch
    BlockSortKernel<BLOCK_SORT_T0_CTA_SIZE, BLOCK_SORT_T0_MIN_SIZE>
        <<<st.n_segments, BLOCK_SORT_T0_CTA_SIZE, 0, s>>>(cascade_args);
    // T1-T3: PDL secondaries — can overlap with the preceding tier
    launchPdl(BlockSortKernel<BLOCK_SORT_T1_CTA_SIZE, BLOCK_SORT_T1_MIN_SIZE>, BLOCK_SORT_T1_CTA_SIZE, cascade_args);
    launchPdl(BlockSortKernel<BLOCK_SORT_T2_CTA_SIZE, BLOCK_SORT_T2_MIN_SIZE>, BLOCK_SORT_T2_CTA_SIZE, cascade_args);
    launchPdl(BlockSortKernel<BLOCK_SORT_T3_CTA_SIZE, BLOCK_SORT_T3_MIN_SIZE, true>, BLOCK_SORT_T3_CTA_SIZE,
              cascade_epilog_args);

    BlockSortKernel<BLOCK_SORT_T3_CTA_SIZE, BLOCK_SORT_T0_MIN_SIZE, false, true>
        <<<st.sm_count, BLOCK_SORT_T3_CTA_SIZE, 0, s>>>(overflow_args);

    // persistent merge: a single kernel launch processes the entire merge tree.
    // the T3 epilog has already built the tree and seeded the FIFO queue with
    // leaf-level partitions. the kernel dynamically enqueues parent nodes as
    // children complete. exits when total_queue_items are consumed.
    const PersistentMergeArgs merge_args = {
        st.merge_queue_state, st.tree_nodes, st.node_done_counts, st.work_items, d_keys[0],
        d_values[0],          d_keys[1],     d_values[1]};
    MergePersistentKernel<<<MERGE_MIN_BLOCKS * st.sm_count, MERGE_CTA_SIZE, 0, s>>>(merge_args);

    return 0;
}

} // namespace higs
