/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <c10/util/Exception.h>

namespace gsplat
{
/**
 * Decompose a feature width into the fewest available kernel channel widths.
 *
 * The largest compiled width sets the per-launch limit. Among plans with the
 * same number of launches, larger chunks are preferred so the result is
 * deterministic and normally starts with the largest usable compiled width.
 */
inline std::vector<int> plan_channel_chunks(int channels, std::vector<int> candidates)
{
    TORCH_CHECK(channels > 0, "channels must be > 0, got ", channels);
    TORCH_CHECK(!candidates.empty(), "GSPLAT_NUM_CHANNELS must contain at least one entry");

    const auto invalid_width = std::ranges::find_if(candidates, [](int width) { return width <= 0; });
    if(invalid_width != candidates.end())
    {
        TORCH_CHECK(false, "GSPLAT_NUM_CHANNELS entries must be > 0, got ", *invalid_width);
    }

    // The candidate list is owned by this function, so normalize it in place
    // instead of allocating a second container for the usable widths.
    const int max_channels_per_launch = *std::ranges::max_element(candidates);
    const int max_width               = std::min(channels, max_channels_per_launch);
    std::erase_if(candidates, [max_width](int width) { return width > max_width; });
    std::ranges::sort(candidates, std::greater<>());
    const auto duplicates = std::ranges::unique(candidates);
    candidates.erase(duplicates.begin(), duplicates.end());

    TORCH_CHECK(
        !candidates.empty(),
        "No compiled channel width is usable for ",
        channels,
        " channels with max channels per launch=",
        max_channels_per_launch
    );

    // Exact unbounded coin change. Only the minimum launch count is retained;
    // reconstruction scans the descending candidate list again to recover the
    // deterministic larger-width tie break described above.
    constexpr int unreachable = std::numeric_limits<int>::max();
    std::vector<int> launch_count(static_cast<size_t>(channels) + 1, unreachable);
    launch_count[0] = 0;

    for(int total = 1; total <= channels; ++total)
    {
        for(int width: candidates)
        {
            if(width > total || launch_count[total - width] == unreachable)
            {
                continue;
            }
            const int candidate_count = launch_count[total - width] + 1;
            if(candidate_count < launch_count[total])
            {
                launch_count[total] = candidate_count;
            }
        }
    }

    if(launch_count[channels] == unreachable)
    {
        std::ostringstream widths;
        for(size_t i = 0; i < candidates.size(); ++i)
        {
            if(i > 0)
            {
                widths << ",";
            }
            widths << candidates[i];
        }
        TORCH_CHECK(
            false,
            "Cannot exactly decompose ",
            channels,
            " channels with max channels per launch=",
            max_channels_per_launch,
            " and compiled widths {",
            widths.str(),
            "}"
        );
    }

    std::vector<int> result;
    result.reserve(static_cast<size_t>(launch_count[channels]));
    for(int remaining = channels; remaining > 0;)
    {
        const auto next = std::ranges::find_if(
            candidates,
            [&](int width)
            {
                return width <= remaining
                    && launch_count[remaining - width] != unreachable
                    && launch_count[remaining - width] + 1 == launch_count[remaining];
            }
        );
        TORCH_INTERNAL_ASSERT(next != candidates.end());
        result.push_back(*next);
        remaining -= *next;
    }
    return result;
}
} // namespace gsplat
