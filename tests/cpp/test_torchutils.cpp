/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>

#include "TorchUtils.h"

// Unit coverage for the gsplat::TorchUtils helpers:
//   - as_optional_tensor / as_tensor: inverse round-trip between
//     at::optional<at::Tensor> and at::Tensor, with an undefined Tensor as the
//     nullopt sentinel.
//   - checked_deref: returns *ptr when set, else raises a TORCH_CHECK error
//     that names the field (turning a null deref into an attributable failure).
//
// Pytest owns discovery/execution via tests/test_cpp.py; this calls the helpers
// directly.

namespace {

struct Holder : c10::intrusive_ptr_target {
    int value;
    explicit Holder(int v) : value(v) {}
};

} // namespace

TEST(TorchUtils, OptionalTensorRoundTripDefined) {
    at::Tensor t = at::ones({2, 3});
    at::optional<at::Tensor> opt = gsplat::as_optional_tensor(t);
    ASSERT_TRUE(opt.has_value());
    at::Tensor back = gsplat::as_tensor(opt);
    EXPECT_TRUE(back.defined());
    EXPECT_TRUE(back.equal(t));
}

TEST(TorchUtils, OptionalTensorRoundTripUndefined) {
    at::Tensor undefined; // .defined() == false -> the nullopt sentinel
    at::optional<at::Tensor> opt = gsplat::as_optional_tensor(undefined);
    EXPECT_FALSE(opt.has_value());
    EXPECT_FALSE(gsplat::as_tensor(opt).defined());
}

TEST(TorchUtils, AsTensorOfNulloptIsUndefined) {
    EXPECT_FALSE(gsplat::as_tensor(c10::nullopt).defined());
}

TEST(TorchUtils, CheckedDerefReturnsReferenceWhenSet) {
    auto p = c10::make_intrusive<Holder>(42);
    const Holder &r = gsplat::checked_deref(p, "holder");
    EXPECT_EQ(r.value, 42);
    EXPECT_EQ(&r, p.get()); // a reference to the held object, not a copy
}

TEST(TorchUtils, CheckedDerefThrowsNamedErrorWhenNull) {
    c10::intrusive_ptr<Holder> p; // null
    EXPECT_THAT(
        [&] { gsplat::checked_deref(p, "fov_horiz_rad"); },
        testing::ThrowsMessage<c10::Error>(
            testing::HasSubstr("fov_horiz_rad must be set")));
}
