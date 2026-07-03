/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <type_traits>

#include <c10/util/Exception.h>

#include "ChannelChunks.h"

struct ChannelChunkValidationErrorCase
{
    int channels;
    std::vector<int> candidates;
    const char *expected_message;
};

const ChannelChunkValidationErrorCase kInvalidCompiledWidthCase{
    .channels         = 3,
    .candidates       = {0, 1, 3},
    .expected_message = "GSPLAT_NUM_CHANNELS entries must be > 0, got 0",
};

const ChannelChunkValidationErrorCase kNoUsableCompiledWidthCase{
    .channels         = 3,
    .candidates       = {4, 6},
    .expected_message = "No compiled channel width is usable for 3 channels",
};

class ChannelChunkValidationErrorTest : public testing::TestWithParam<ChannelChunkValidationErrorCase>
{
};

TEST_P(ChannelChunkValidationErrorTest, ReportsActionableError)
{
    const ChannelChunkValidationErrorCase &param = GetParam();
    EXPECT_THAT(
        [&param] { gsplat::plan_channel_chunks(param.channels, param.channels, param.candidates); },
        testing::ThrowsMessage<c10::Error>(testing::HasSubstr(param.expected_message))
    );
}

INSTANTIATE_TEST_SUITE_P(
    InvalidCompiledWidths, ChannelChunkValidationErrorTest, testing::Values(kInvalidCompiledWidthCase)
);

INSTANTIATE_TEST_SUITE_P(
    NoUsableCompiledWidths, ChannelChunkValidationErrorTest, testing::Values(kNoUsableCompiledWidthCase)
);

using testing::ElementsAre;

using PlanChannelChunks = std::vector<int> (*)(int, int, std::vector<int>);
static_assert(std::is_same_v<decltype(&gsplat::plan_channel_chunks), PlanChannelChunks>);

TEST(ChannelChunks, MinimizesLaunchCountInsteadOfGreedilyTakingTheLimit)
{
    EXPECT_THAT(gsplat::plan_channel_chunks(46, 32, {1, 3, 5, 9, 23, 32}), ElementsAre(23, 23));
}

TEST(ChannelChunks, PrefersLargerWidthsWhenLaunchCountsTie)
{
    EXPECT_THAT(gsplat::plan_channel_chunks(48, 32, {1, 16, 24, 32}), ElementsAre(32, 16));
}

TEST(ChannelChunks, AllowsRepeatedCompiledWidths)
{
    EXPECT_THAT(gsplat::plan_channel_chunks(64, 32, {1, 3, 32}), ElementsAre(32, 32));
}

TEST(ChannelChunks, LimitNeedNotBeACompiledWidth)
{
    EXPECT_THAT(gsplat::plan_channel_chunks(46, 31, {1, 23, 32}), ElementsAre(23, 23));
}

TEST(ChannelChunks, NormalizesUnsortedDuplicateWidths)
{
    EXPECT_THAT(gsplat::plan_channel_chunks(12, 8, {3, 8, 3, 6}), ElementsAre(6, 6));
}

TEST(ChannelChunks, RejectsAnImpossibleExactDecomposition)
{
    EXPECT_THAT(
        [] { gsplat::plan_channel_chunks(7, 6, {4, 6, 8}); },
        testing::ThrowsMessage<c10::Error>(testing::HasSubstr("Cannot exactly decompose 7 channels"))
    );
}

TEST(ChannelChunks, RejectsANonPositiveLimit)
{
    EXPECT_THAT(
        [] { gsplat::plan_channel_chunks(3, 0, {1, 3}); },
        testing::ThrowsMessage<c10::Error>(testing::HasSubstr("channel_chunk must be > 0"))
    );
}
