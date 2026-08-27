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

// Wave-width-safe 32-lane "macro-tile" primitives for the gaussian_inference
// kernels.
//
// BACKGROUND: this subsystem is built around the invariant "one macro-tile /
// one cooperative sub-group == exactly 32 lanes" (MACRO_TILE_SIZE == 32,
// enforced by static_assert in MacroTileRasterize.hip and
// IntersectMTConfig.h). That invariant matches NVIDIA's fixed 32-wide warp
// and AMD RDNA's 32-wide wavefront (wave32) exactly, because on both of
// those architectures one hardware warp/wavefront IS one macro-tile.
//
// AMD CDNA (gfx900/906/908/90a/942/950 -- MI100/MI200/MI300/MI350 series)
// has NO wave32 hardware mode: its physical wavefront is always 64 lanes.
// Several kernels in this subsystem pack TWO logical 32-lane macro-tiles
// into one 64-lane physical CDNA wavefront (e.g.
// `warp_id = threadIdx.x >> 5` in MacroTileRasterize.hip, with warp_id 0 and
// 1 sharing physical wavefront 0; or `macro_tile_post_blend_kernel`'s "1
// warp per render tile" with 8 such warps per CTA). When that happens, raw
// wavefront-width-implicit intrinsics silently operate across BOTH logical
// tiles at once instead of scoping to the caller's own 32 lanes:
//
//   - `__shfl(var, srcLane)` / `__shfl_xor(var, mask)` / `__shfl_up(...)`
//     default their `width` argument to `warpSize` (64 on CDNA), so e.g. a
//     broadcast intended to mean "read lane 0 of MY OWN 32-lane tile"
//     instead reads physical lane 0 of the WHOLE 64-lane wavefront -- i.e.
//     the OTHER logical tile's lane 0, for any thread whose tile occupies
//     the upper half.
//   - `__ballot(pred)` always returns the full (up to 64-bit) vote mask for
//     the entire physical wavefront -- there is no `width` parameter for
//     ballot on HIP/AMD. Truncating that raw mask to `uint32_t` does NOT
//     scope it to "my own 32 lanes"; it just keeps the lower 32 bits, which
//     belong to whichever logical tile happens to occupy the lower half of
//     the physical wavefront, regardless of which tile is asking.
//
// FIX, part 1 (`__shfl`/`__shfl_xor`/`__shfl_up`): HIP's raw shuffle
// intrinsics accept an explicit `width` argument -- the AMD-native,
// documented equivalent of CUDA's `__shfl_sync(mask, var, srcLane, width)`.
// Passing `width=32` at call sites makes the shuffle treat the physical
// wavefront as independent 32-lane segments, identical to what
// `cooperative_groups::thread_block_tile<32>::shfl*()` does internally
// (`return __shfl(var, srcRank, numThreads);` in
// amd_hip_cooperative_groups.h) -- this is exactly the tile-scoping
// mechanism already established as the correct AMD pattern in this same
// codebase (see gsplat/hip/include/cooperative_groups/reduce.h). Passing
// width=32 explicitly is a strict no-op on wave32 hardware (NVIDIA warp /
// RDNA wavefront), where width=32 already equals the physical wavefront
// size, so it cannot regress RDNA behavior.
//
// FIX, part 2 (`__ballot`): since raw `__ballot` has no width parameter,
// `Ballot32()` below manually re-derives a tile-scoped 32-bit mask from the
// raw (up to 64-bit) wavefront ballot by shifting down to the CALLING
// lane's own 32-lane segment (`__lane_id() & ~31u`) before truncating. This
// mirrors exactly what `cooperative_groups::thread_block_tile<32>::ballot()`
// computes via its `build_mask()` helper, just without that API's more
// general (and here unnecessary) divergent-mask waterfall machinery, which
// matters in these latency-sensitive per-tile/per-batch hot loops. The
// shift is a no-op on wave32 hardware (the segment base is always 0 because
// `__lane_id()` never exceeds 31 there), and correctly isolates each
// 32-lane macro-tile's own votes on wave64 hardware.
//
// NONE of MACRO_TILE_SIZE / FUSED_MACRO_TILE_WIDTH*HEIGHT / the 32-wide
// shared-memory and register layouts in this subsystem change because of
// this fix: only the shuffle/ballot primitives above are made
// wavefront-width-safe, exactly per the codebase's own recommended
// approach. Verified on both real RDNA (wave32, gfx1100) and real CDNA
// (wave64, gfx950) hardware; covered by tests/experimental/render/*.

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace higs
{

// Tile-scoped 32-lane ballot: returns a mask where bit i is set iff lane i
// of THIS thread's own 32-lane macro-tile satisfied `predicate`. Correct
// regardless of whether the underlying physical wavefront is 32 lanes
// (NVIDIA warp / RDNA wave32) or 64 lanes (AMD CDNA wave64), and regardless
// of how many independent 32-lane macro-tiles are concurrently active and
// packed into that one physical wavefront.
__device__ __forceinline__ uint32_t Ballot32(bool predicate)
{
    const unsigned long long full     = __ballot(predicate);
    const uint32_t macro_tile_seg_base = __lane_id() & ~31u;
    return static_cast<uint32_t>((full >> macro_tile_seg_base) & 0xFFFFFFFFull);
}

} // namespace higs
