/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <bit>
#include <cstdint>

namespace gsplat {

/** Bits needed to index `count` items as 0..count-1 (0 when count <= 1).
 *
 * This is the field width used to pack the image id and tile id into the high
 * 32 bits of an intersection sort key. Every pack and unpack site MUST use this
 * helper so the (image, tile) id round-trips identically -- `floor(log2)+1`
 * over-counts power-of-two counts (8 -> 4 bits instead of 3) and yields 1 for a
 * single item, so a packer and an unpacker that disagree corrupt the decode.
 *
 * Requires C++20 (`std::bit_width`); the extension is built as C++20.
 */
inline uint32_t bits_for_count(int64_t count) {
    if (count <= 1) {
        return 0u;
    } else {
        return static_cast<uint32_t>(
            std::bit_width(static_cast<uint64_t>(count) - 1u));
    }
}

} // namespace gsplat
