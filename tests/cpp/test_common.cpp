/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/torch.h>

#include "Common.h"

// CHECK_DENSE asserts a strided (dense) layout: kernels index raw dense
// storage, so a sparse-layout tensor must be rejected attributably rather
// than silently read as garbage. The macro names the offending variable.
namespace
{
void require_dense(const at::Tensor &x)
{
    CHECK_DENSE(x);
}
} // namespace

TEST(Common, CheckDenseAcceptsStridedTensor)
{
    EXPECT_NO_THROW(require_dense(at::ones({2, 3}))); // strided by default
}

TEST(Common, CheckDenseThrowsNamedErrorOnSparseTensor)
{
    at::Tensor sparse = at::ones({2, 3}).to_sparse();
    EXPECT_THAT(
        [&] { require_dense(sparse); },
        testing::ThrowsMessage<c10::Error>(testing::HasSubstr("x must be a dense tensor"))
    );
}
