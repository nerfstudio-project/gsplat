/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/core/Tensor.h>

#include <tuple>

namespace gsplat
{
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> launch_priming_decode_for_batch(
    const at::Tensor &packed, const at::Tensor &batch_ids
);
} // namespace gsplat
