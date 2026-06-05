/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <ostream>

#include "MathUtils.h"

// bits_for_count() is the field width used to pack the (image, tile) id into
// the high 32 bits of an intersection sort key. Every pack and unpack site
// must agree on it, so this pins the contract -- especially the power-of-two
// boundary, where the old floor(log2)+1 formula disagreed by one bit and a
// packer/unpacker mismatch corrupts the decoded ids.
//
// Pytest owns discovery/execution via tests/test_cpp.py; this calls the gsplat
// C++ helper directly.

namespace {

struct SmallCountCase {
    int64_t count;
    uint32_t expected_bits;

    // Let GoogleTest print the case (e.g. in --gtest_list_tests and failure
    // messages) instead of dumping the raw bytes of the struct. Defined in
    // class as a hidden friend so it travels with the struct and is found via
    // ADL.
    friend std::ostream &operator<<(std::ostream &os, const SmallCountCase &c) {
        return os << "{count=" << c.count
                  << ", expected_bits=" << c.expected_bits << "}";
    }
};

class BitsForCountSmall : public ::testing::TestWithParam<SmallCountCase> {};

TEST_P(BitsForCountSmall, MatchesExpectedWidth) {
    const SmallCountCase param = GetParam();
    EXPECT_EQ(gsplat::bits_for_count(param.count), param.expected_bits)
        << "count = " << param.count;
}

INSTANTIATE_TEST_SUITE_P(
    SmallCounts,
    BitsForCountSmall,
    ::testing::Values(
        SmallCountCase{0, 0u},
        SmallCountCase{1, 0u},
        SmallCountCase{2, 1u},
        SmallCountCase{3, 2u},
        SmallCountCase{4, 2u},
        SmallCountCase{5, 3u},
        SmallCountCase{7, 3u},
        SmallCountCase{8, 3u},
        SmallCountCase{9, 4u}));

TEST(BitsForCount, PowersOfTwoNeedKBits) {
    // 2^k items use indices 0..2^k-1, which need exactly k bits. The old
    // floor(log2)+1 formula returned k+1 here -- the regression this guards.
    for (uint32_t k = 1; k <= 31; ++k) {
        const int64_t count = int64_t{1} << k;
        EXPECT_EQ(gsplat::bits_for_count(count), k) << "count = 2^" << k;
        EXPECT_EQ(gsplat::bits_for_count(count + 1), k + 1)
            << "count = 2^" << k << " + 1";
    }
}

TEST(BitsForCount, WidthMinimallyIndexesMaxValue) {
    // The width must hold the largest index (count - 1) and be minimal: one
    // fewer bit must NOT suffice.
    for (int64_t count = 2; count <= 4096; ++count) {
        const uint32_t bits = gsplat::bits_for_count(count);
        const int64_t max_index = count - 1;
        EXPECT_LE(max_index, (int64_t{1} << bits) - 1) << "count = " << count;
        EXPECT_GT(max_index, (int64_t{1} << (bits - 1)) - 1) << "count = " << count;
    }
}

} // namespace
