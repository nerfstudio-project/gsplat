/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/Functions.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/torch.h>

#include "TorchUtils.h"

// Unit coverage for the gsplat::TorchUtils helpers:
//   - as_optional_tensor / as_tensor: inverse round-trip between
//     at::optional<at::Tensor> and at::Tensor, with an undefined Tensor as the
//     nullopt sentinel.
//   - checked_deref: returns *ptr when set, else raises a TORCH_CHECK error
//     that names the field (turning a null deref into an attributable failure).
//   - gsplat::detail traits: classify the type shapes that drive dispatcher
//     marshalling and saved-state routing.
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

// Named classification flags for a TraitCase. The case table encodes expected
// trait combinations with explicit flag names.
enum class Trait : unsigned {
    None = 0,
    Optional = 1u << 0,
    IntrusivePtr = 1u << 1,
    TensorList = 1u << 2,
    OptionalTensor = 1u << 3,
    OptionalIntrusive = 1u << 4,
};

constexpr Trait operator|(Trait a, Trait b) {
    return static_cast<Trait>(static_cast<unsigned>(a) | static_cast<unsigned>(b));
}

constexpr bool has_trait(Trait flags, Trait t) {
    return (static_cast<unsigned>(flags) & static_cast<unsigned>(t)) != 0;
}

// Print a flag set by name (e.g. "Optional|OptionalTensor"), so a case is
// identifiable from its traits rather than an opaque value.
inline std::ostream &operator<<(std::ostream &os, Trait flags) {
    if (flags == Trait::None) {
        return os << "None";
    }
    const char *sep = "";
    const auto emit = [&](Trait t, const char *name) {
        if (has_trait(flags, t)) {
            os << sep << name;
            sep = "|";
        }
    };
    emit(Trait::Optional, "Optional");
    emit(Trait::IntrusivePtr, "IntrusivePtr");
    emit(Trait::TensorList, "TensorList");
    emit(Trait::OptionalTensor, "OptionalTensor");
    emit(Trait::OptionalIntrusive, "OptionalIntrusive");
    return os;
}

template <class Type, Trait Flags, class OptionalValue, class IntrusiveValue>
struct TraitCase {
    using type = Type;
    using optional_value = OptionalValue;
    using intrusive_value = IntrusiveValue;

    static constexpr Trait flags = Flags;
    static constexpr bool is_optional = has_trait(Flags, Trait::Optional);
    static constexpr bool is_intrusive_ptr = has_trait(Flags, Trait::IntrusivePtr);
    static constexpr bool is_tensor_list = has_trait(Flags, Trait::TensorList);
    static constexpr bool is_optional_tensor = has_trait(Flags, Trait::OptionalTensor);
    static constexpr bool is_optional_intrusive =
        has_trait(Flags, Trait::OptionalIntrusive);
};

template <class Case> class TorchTraitTest : public ::testing::Test {};

using TorchTraitCases = ::testing::Types<
    TraitCase<at::Tensor, Trait::None, void, void>,
    TraitCase<at::optional<at::Tensor>, Trait::Optional | Trait::OptionalTensor, at::Tensor, void>,
    TraitCase<Holder, Trait::None, void, void>,
    TraitCase<c10::intrusive_ptr<Holder>, Trait::IntrusivePtr, void, Holder>,
    TraitCase<
        at::optional<c10::intrusive_ptr<Holder>>,
        Trait::Optional | Trait::OptionalIntrusive,
        c10::intrusive_ptr<Holder>,
        void>,
    TraitCase<std::vector<at::Tensor>, Trait::TensorList, void, void>,
    TraitCase<c10::List<at::Tensor>, Trait::TensorList, void, void>,
    TraitCase<at::TensorList, Trait::TensorList, void, void>>;
TYPED_TEST_SUITE(TorchTraitTest, TorchTraitCases);

TYPED_TEST(TorchTraitTest, ClassifiesOptionalTypesAtCompileTime) {
    using T = typename TypeParam::type;
    static_assert(gsplat::detail::is_optional<T>::value == TypeParam::is_optional);
    SUCCEED();
}

TYPED_TEST(TorchTraitTest, ClassifiesIntrusivePtrTypesAtCompileTime) {
    using T = typename TypeParam::type;
    static_assert(
        gsplat::detail::is_intrusive_ptr<T>::value == TypeParam::is_intrusive_ptr);
    SUCCEED();
}

TYPED_TEST(TorchTraitTest, ExtractsClassifierValueTypesAtCompileTime) {
    using T = typename TypeParam::type;
    static_assert(std::is_same_v<
                  gsplat::detail::value_type_t<gsplat::detail::is_optional<T>>,
                  typename TypeParam::optional_value>);
    static_assert(std::is_same_v<
                  gsplat::detail::value_type_t<gsplat::detail::is_intrusive_ptr<T>>,
                  typename TypeParam::intrusive_value>);
    SUCCEED();
}

TYPED_TEST(TorchTraitTest, ClassifiesTensorListTypesAtCompileTime) {
    using T = typename TypeParam::type;
    static_assert(gsplat::detail::is_tensor_list<T>::value == TypeParam::is_tensor_list);
    static_assert(gsplat::detail::is_tensor_list_v<T> == TypeParam::is_tensor_list);
    SUCCEED();
}

TYPED_TEST(TorchTraitTest, ClassifiesOptionalTensorTypesAtCompileTime) {
    using T = typename TypeParam::type;
    static_assert(
        gsplat::detail::is_optional_tensor_v<T> == TypeParam::is_optional_tensor);
    SUCCEED();
}

TYPED_TEST(TorchTraitTest, ClassifiesOptionalIntrusiveTypesAtCompileTime) {
    using T = typename TypeParam::type;
    static_assert(
        gsplat::detail::is_optional_intrusive_v<T> == TypeParam::is_optional_intrusive);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// tensor_requires_grad: true only for a defined tensor that requires grad;
// false for non-grad / undefined / nullopt (both Tensor and optional overloads).
// ---------------------------------------------------------------------------

TEST(TensorRequiresGradTest, tensor_overload) {
    at::Tensor grad_t = at::ones({2}).set_requires_grad(true);
    EXPECT_TRUE(gsplat::tensor_requires_grad(grad_t));

    at::Tensor plain = at::ones({2});
    EXPECT_FALSE(gsplat::tensor_requires_grad(plain));

    at::Tensor undefined;
    EXPECT_FALSE(gsplat::tensor_requires_grad(undefined));
}

TEST(TensorRequiresGradTest, optional_overload) {
    at::Tensor grad_t = at::ones({2}).set_requires_grad(true);
    EXPECT_TRUE(
        gsplat::tensor_requires_grad(at::optional<at::Tensor>(grad_t))
    );

    at::Tensor plain = at::ones({2});
    EXPECT_FALSE(
        gsplat::tensor_requires_grad(at::optional<at::Tensor>(plain))
    );

    EXPECT_FALSE(gsplat::tensor_requires_grad(at::optional<at::Tensor>{}));
}

// ---------------------------------------------------------------------------
// needs_custom_autograd: true only when GradMode is enabled AND some input
// requires grad. Off either way otherwise.
// ---------------------------------------------------------------------------

TEST(NeedsCustomAutogradTest, requires_grad_mode_and_a_grad_input) {
    at::Tensor grad_t = at::ones({2}).set_requires_grad(true);
    at::Tensor plain = at::ones({2});

    {
        // GradMode enabled + a grad-requiring input -> true.
        at::AutoGradMode guard(true);
        EXPECT_TRUE(gsplat::needs_custom_autograd(grad_t, plain));
        // GradMode enabled but no grad-requiring input -> false.
        EXPECT_FALSE(gsplat::needs_custom_autograd(plain, plain));
        // Mixes optional + tensor inputs.
        EXPECT_TRUE(gsplat::needs_custom_autograd(
            at::optional<at::Tensor>(grad_t), plain
        ));
    }
    {
        // GradMode disabled -> always false, even with a grad-requiring input.
        at::AutoGradMode guard(false);
        EXPECT_FALSE(gsplat::needs_custom_autograd(grad_t, plain));
    }
}
