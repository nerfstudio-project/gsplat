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
#if defined(__GNUC__) && !defined(__clang__)
// GCC PR110498 can diagnose std::vector<bool>::reserve inside
// torch::autograd::Function::apply as either of these warnings.
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Warray-bounds"
#    pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
#include <torch/csrc/autograd/custom_function.h>
#if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic pop
#endif
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

namespace
{
struct Holder : c10::intrusive_ptr_target
{
    int value;

    explicit Holder(int v)
        : value(v)
    {
    }
};
} // namespace

TEST(TorchUtils, OptionalTensorRoundTripDefined)
{
    at::Tensor t                 = at::ones({2, 3});
    at::optional<at::Tensor> opt = gsplat::as_optional_tensor(t);
    ASSERT_TRUE(opt.has_value());
    at::Tensor back = gsplat::as_tensor(opt);
    EXPECT_TRUE(back.defined());
    EXPECT_TRUE(back.equal(t));
}

TEST(TorchUtils, OptionalTensorRoundTripUndefined)
{
    at::Tensor undefined; // .defined() == false -> the nullopt sentinel
    at::optional<at::Tensor> opt = gsplat::as_optional_tensor(undefined);
    EXPECT_FALSE(opt.has_value());
    EXPECT_FALSE(gsplat::as_tensor(opt).defined());
}

TEST(TorchUtils, AsTensorOfNulloptIsUndefined)
{
    EXPECT_FALSE(gsplat::as_tensor(c10::nullopt).defined());
}

TEST(TorchUtils, CheckedDerefReturnsReferenceWhenSet)
{
    auto p          = c10::make_intrusive<Holder>(42);
    const Holder &r = gsplat::checked_deref(p, "holder");
    EXPECT_EQ(r.value, 42);
    EXPECT_EQ(&r, p.get()); // a reference to the held object, not a copy
}

TEST(TorchUtils, CheckedDerefThrowsNamedErrorWhenNull)
{
    c10::intrusive_ptr<Holder> p; // null
    EXPECT_THAT(
        [&] { gsplat::checked_deref(p, "fov_horiz_rad"); },
        testing::ThrowsMessage<c10::Error>(testing::HasSubstr("fov_horiz_rad must be set"))
    );
}

// Named classification flags for a TraitCase. The case table encodes expected
// trait combinations with explicit flag names.
enum class Trait : unsigned
{
    None              = 0,
    Optional          = 1u << 0,
    IntrusivePtr      = 1u << 1,
    TensorList        = 1u << 2,
    OptionalTensor    = 1u << 3,
    OptionalIntrusive = 1u << 4,
};

constexpr Trait operator|(Trait a, Trait b)
{
    return static_cast<Trait>(static_cast<unsigned>(a) | static_cast<unsigned>(b));
}

constexpr bool has_trait(Trait flags, Trait t)
{
    return (static_cast<unsigned>(flags) & static_cast<unsigned>(t)) != 0;
}

// Print a flag set by name (e.g. "Optional|OptionalTensor"), so a case is
// identifiable from its traits rather than an opaque value.
inline std::ostream &operator<<(std::ostream &os, Trait flags)
{
    if(flags == Trait::None)
    {
        return os << "None";
    }
    const char *sep = "";
    const auto emit = [&](Trait t, const char *name)
    {
        if(has_trait(flags, t))
        {
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

template<class Type, Trait Flags, class OptionalValue, class IntrusiveValue>
struct TraitCase
{
    using type            = Type;
    using optional_value  = OptionalValue;
    using intrusive_value = IntrusiveValue;

    static constexpr Trait flags                = Flags;
    static constexpr bool is_optional           = has_trait(Flags, Trait::Optional);
    static constexpr bool is_intrusive_ptr      = has_trait(Flags, Trait::IntrusivePtr);
    static constexpr bool is_tensor_list        = has_trait(Flags, Trait::TensorList);
    static constexpr bool is_optional_tensor    = has_trait(Flags, Trait::OptionalTensor);
    static constexpr bool is_optional_intrusive = has_trait(Flags, Trait::OptionalIntrusive);
};

template<class Case>
class TorchTraitTest : public ::testing::Test
{
};

using TorchTraitCases = ::testing::Types<
    TraitCase<at::Tensor, Trait::None, void, void>,
    TraitCase<at::optional<at::Tensor>, Trait::Optional | Trait::OptionalTensor, at::Tensor, void>,
    TraitCase<Holder, Trait::None, void, void>,
    TraitCase<c10::intrusive_ptr<Holder>, Trait::IntrusivePtr, void, Holder>,
    TraitCase<
        at::optional<c10::intrusive_ptr<Holder>>,
        Trait::Optional | Trait::OptionalIntrusive,
        c10::intrusive_ptr<Holder>,
        void
    >,
    TraitCase<std::vector<at::Tensor>, Trait::TensorList, void, void>,
    TraitCase<c10::List<at::Tensor>, Trait::TensorList, void, void>,
    TraitCase<at::TensorList, Trait::TensorList, void, void>
>;
TYPED_TEST_SUITE(TorchTraitTest, TorchTraitCases);

TYPED_TEST(TorchTraitTest, ClassifiesOptionalTypesAtCompileTime)
{
    using T = typename TypeParam::type;
    static_assert(gsplat::detail::is_optional<T>::value == TypeParam::is_optional);
    SUCCEED();
}

TYPED_TEST(TorchTraitTest, ClassifiesIntrusivePtrTypesAtCompileTime)
{
    using T = typename TypeParam::type;
    static_assert(gsplat::detail::is_intrusive_ptr<T>::value == TypeParam::is_intrusive_ptr);
    SUCCEED();
}

TYPED_TEST(TorchTraitTest, ExtractsClassifierValueTypesAtCompileTime)
{
    using T = typename TypeParam::type;
    static_assert(
        std::is_same_v<gsplat::detail::value_type_t<gsplat::detail::is_optional<T>>, typename TypeParam::optional_value>
    );
    static_assert(std::is_same_v<
                  gsplat::detail::value_type_t<gsplat::detail::is_intrusive_ptr<T>>,
                  typename TypeParam::intrusive_value
    >);
    SUCCEED();
}

TYPED_TEST(TorchTraitTest, ClassifiesTensorListTypesAtCompileTime)
{
    using T = typename TypeParam::type;
    static_assert(gsplat::detail::is_tensor_list<T>::value == TypeParam::is_tensor_list);
    static_assert(gsplat::detail::is_tensor_list_v<T> == TypeParam::is_tensor_list);
    SUCCEED();
}

TYPED_TEST(TorchTraitTest, ClassifiesOptionalTensorTypesAtCompileTime)
{
    using T = typename TypeParam::type;
    static_assert(gsplat::detail::is_optional_tensor_v<T> == TypeParam::is_optional_tensor);
    SUCCEED();
}

TYPED_TEST(TorchTraitTest, ClassifiesOptionalIntrusiveTypesAtCompileTime)
{
    using T = typename TypeParam::type;
    static_assert(gsplat::detail::is_optional_intrusive_v<T> == TypeParam::is_optional_intrusive);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// tensor_or_zeros / tensor_or_zeros_like: return the gradient if present, else
// a zero tensor of the given shape / like a reference. Used to materialize an
// absent output grad (set_materialize_grads(false) -> nullopt) as an explicit
// zero for kernels that read dense gradients.
// ---------------------------------------------------------------------------

TEST(TensorOrZerosTest, returns_grad_when_present)
{
    // The shape / reference is ignored when the optional holds a value.
    at::Tensor g = at::tensor({1.0f, 2.0f, 3.0f});
    EXPECT_TRUE(at::equal(gsplat::tensor_or_zeros(at::optional<at::Tensor>(g), {9, 9}, g.options()), g));
    EXPECT_TRUE(at::equal(gsplat::tensor_or_zeros_like(at::optional<at::Tensor>(g), at::empty({9})), g));
}

TEST(TensorOrZerosTest, zeros_when_absent)
{
    auto options = at::TensorOptions().dtype(at::kFloat);
    at::Tensor z = gsplat::tensor_or_zeros(c10::nullopt, {2, 3}, options);
    EXPECT_EQ(z.sizes(), (at::IntArrayRef{2, 3}));
    EXPECT_TRUE(at::equal(z, at::zeros({2, 3}, options)));

    at::Tensor reference = at::ones({4, 5});
    at::Tensor zl        = gsplat::tensor_or_zeros_like(c10::nullopt, reference);
    EXPECT_EQ(zl.sizes(), reference.sizes());
    EXPECT_TRUE(at::equal(zl, at::zeros_like(reference)));
}

// ---------------------------------------------------------------------------
// tensor_requires_grad: true only for a defined tensor that requires grad;
// false for non-grad / undefined / nullopt (both Tensor and optional overloads).
// ---------------------------------------------------------------------------

TEST(TensorRequiresGradTest, tensor_overload)
{
    at::Tensor grad_t = at::ones({2}).set_requires_grad(true);
    EXPECT_TRUE(gsplat::tensor_requires_grad(grad_t));

    at::Tensor plain = at::ones({2});
    EXPECT_FALSE(gsplat::tensor_requires_grad(plain));

    at::Tensor undefined;
    EXPECT_FALSE(gsplat::tensor_requires_grad(undefined));
}

TEST(TensorRequiresGradTest, optional_overload)
{
    at::Tensor grad_t = at::ones({2}).set_requires_grad(true);
    EXPECT_TRUE(gsplat::tensor_requires_grad(at::optional<at::Tensor>(grad_t)));

    at::Tensor plain = at::ones({2});
    EXPECT_FALSE(gsplat::tensor_requires_grad(at::optional<at::Tensor>(plain)));

    EXPECT_FALSE(gsplat::tensor_requires_grad(at::optional<at::Tensor>{}));
}

// ---------------------------------------------------------------------------
// needs_custom_autograd: true only when GradMode is enabled AND some input
// requires grad. Off either way otherwise.
// ---------------------------------------------------------------------------

TEST(NeedsCustomAutogradTest, requires_grad_mode_and_a_grad_input)
{
    at::Tensor grad_t = at::ones({2}).set_requires_grad(true);
    at::Tensor plain  = at::ones({2});

    {
        // GradMode enabled + a grad-requiring input -> true.
        at::AutoGradMode guard(true);
        EXPECT_TRUE(gsplat::needs_custom_autograd(grad_t, plain));
        // GradMode enabled but no grad-requiring input -> false.
        EXPECT_FALSE(gsplat::needs_custom_autograd(plain, plain));
        // Mixes optional + tensor inputs.
        EXPECT_TRUE(gsplat::needs_custom_autograd(at::optional<at::Tensor>(grad_t), plain));
    }
    {
        // GradMode disabled -> always false, even with a grad-requiring input.
        at::AutoGradMode guard(false);
        EXPECT_FALSE(gsplat::needs_custom_autograd(grad_t, plain));
    }
}

// ---------------------------------------------------------------------------
// contiguous_optional: a non-contiguous optional becomes contiguous; an
// already-contiguous one is returned untouched; nullopt stays nullopt.
// ---------------------------------------------------------------------------

TEST(ContiguousOptionalTest, makes_noncontiguous_contiguous)
{
    // A transpose produces a non-contiguous view; contiguous_optional copies it.
    at::Tensor base       = at::arange(6).reshape({2, 3});
    at::Tensor non_contig = base.t(); // [3, 2], non-contiguous
    ASSERT_FALSE(non_contig.is_contiguous());

    at::optional<at::Tensor> out = gsplat::contiguous_optional(at::optional<at::Tensor>(non_contig));
    ASSERT_TRUE(out.has_value());
    EXPECT_TRUE(out.value().is_contiguous());
    // Same values, just laid out contiguously.
    EXPECT_TRUE(at::equal(out.value(), non_contig));
}

TEST(ContiguousOptionalTest, already_contiguous_returned_as_is)
{
    at::Tensor contig = at::ones({2, 3});
    ASSERT_TRUE(contig.is_contiguous());

    at::optional<at::Tensor> out = gsplat::contiguous_optional(at::optional<at::Tensor>(contig));
    ASSERT_TRUE(out.has_value());
    EXPECT_TRUE(out.value().is_contiguous());
    // contiguous() on an already-contiguous tensor returns the same storage.
    EXPECT_TRUE(out.value().is_same(contig));
}

TEST(ContiguousOptionalTest, nullopt_stays_nullopt)
{
    EXPECT_FALSE(gsplat::contiguous_optional(at::optional<at::Tensor>{}).has_value());
}

// ---------------------------------------------------------------------------
// make_sparse_coo_grad: builds a sparse-COO tensor carrying the supplied
// indices/values at the requested dense shape.
// ---------------------------------------------------------------------------

TEST(MakeSparseCooGradTest, builds_expected_sparse_tensor)
{
    // Two non-zero rows of a [4, 2] dense gradient, addressed by row index.
    // (Flat initializer lists + reshape; at::tensor has no nested-brace overload.)
    at::Tensor indices = at::tensor({0, 3}, at::kLong).reshape({1, 2});        // [1, nnz]
    at::Tensor values  = at::tensor({1.0f, 2.0f, 3.0f, 4.0f}).reshape({2, 2}); // [nnz, 2]

    at::Tensor sparse = gsplat::make_sparse_coo_grad(indices, values, {4, 2}, /*is_coalesced=*/true);

    EXPECT_TRUE(sparse.is_sparse());
    EXPECT_EQ(sparse.sizes(), (std::vector<int64_t>{4, 2}));
    EXPECT_EQ(sparse._nnz(), 2);
    EXPECT_TRUE(at::equal(sparse._indices(), indices));
    EXPECT_TRUE(at::equal(sparse._values(), values));

    // Densifying reproduces the addressed rows and leaves the rest zero.
    at::Tensor dense    = sparse.to_dense();
    at::Tensor expected = at::zeros({4, 2});
    expected[0].copy_(values[0]);
    expected[3].copy_(values[1]);
    EXPECT_TRUE(at::equal(dense, expected));
}

namespace
{
// ---------------------------------------------------------------------------
// Test fixtures: instrumented payloads and struct-returning ops.
// ---------------------------------------------------------------------------

// Counts copy/move construction so a test can assert the identity path neither
// copies nor moves its argument.
struct Tracked
{
    int value                = 0;
    static inline int copies = 0;
    static inline int moves  = 0;

    Tracked() = default;

    explicit Tracked(int v)
        : value(v)
    {
    }

    Tracked(const Tracked &o)
        : value(o.value)
    {
        ++copies;
    }

    Tracked(Tracked &&o) noexcept
        : value(o.value)
    {
        ++moves;
    }

    Tracked &operator=(const Tracked &) = default;
    Tracked &operator=(Tracked &&)      = default;

    static void reset()
    {
        copies = 0;
        moves  = 0;
    }
};

// A C++ argument that decomposes into two dispatcher arguments (the shape a
// future tensor-carrying object would take).
struct Pair
{
    int a    = 0;
    double b = 0.0;
};

// A 1<->N struct whose columns are copy-instrumented, so a typed copy test can
// assert the marshalling round-trip moves rather than copies the payload.
struct TrackedPair
{
    Tracked a;
    Tracked b;
};

// A copy-instrumented element that marshals 1<->1 to a native int64 (its spec is
// below). Unlike Tracked, it reduces to a dispatch type, so it can ride the tuple
// spec's recursive (to_torch_args) path -- letting a test assert that path neither
// copies nor moves the element.
struct CountedScalar
{
    int64_t value            = 0;
    static inline int copies = 0;
    static inline int moves  = 0;

    CountedScalar() = default;

    explicit CountedScalar(int64_t v)
        : value(v)
    {
    }

    CountedScalar(const CountedScalar &o)
        : value(o.value)
    {
        ++copies;
    }

    CountedScalar(CountedScalar &&o) noexcept
        : value(o.value)
    {
        ++moves;
    }

    static void reset()
    {
        copies = 0;
        moves  = 0;
    }
};

enum UnscopedEnum : int
{
    UNSCOPED_X = 0,
    UNSCOPED_Y = 7
};
enum class ScopedEnum : int64_t
{
    A = 0,
    B = 5
};

// Result structs cross native->dispatch via a co-located TorchArgDef spec
// (below), like the gsplat ops.
struct IntResult
{
    int64_t value;
};

struct DoubleResult
{
    double value;
};

// Reads a native (Tensor) argument by const&, returning the refcount it observes.
// at::Tensor is torch-native, so it crosses via the identity TorchArgDef; a
// wrapper that copied the argument along the way would bump its use_count.
IntResult read_tensor_use_count(const at::Tensor &t)
{
    return IntResult{static_cast<int64_t>(t.use_count())};
}

// Mixed natural arguments, including a 1->2 decomposition followed by a scalar.
DoubleResult pair_then_scalar(const Pair &p, int64_t k)
{
    return DoubleResult{p.a + p.b + static_cast<double>(k)};
}

// A scalar-converting op: float argument crosses as double.
DoubleResult scale(float x)
{
    return DoubleResult{static_cast<double>(x) * 2.0};
}
} // namespace

namespace gsplat
{
// Identity marshalling for the copy-instrumented payload (mirrors the native
// primary template, which Tracked cannot use since it is not torch-native). A
// 1<->1 mapping: a bare value, no tuple.
template<>
struct TorchArgDef<Tracked>
{
    template<class U>
    static auto to(U &&v)
    {
        return std::forward<U>(v);
    }

    template<class U>
    static decltype(auto) from(U &&v)
    {
        return std::forward<U>(v);
    }
};

// 1<->N marshalling: one Pair decomposes into (int64_t, double).
template<>
struct TorchArgDef<Pair>
{
    static std::tuple<int64_t, double> to(const Pair &p)
    {
        return {p.a, p.b};
    }

    template<class TT>
    static Pair from(TT &&t)
    {
        static_assert(std::tuple_size_v<std::decay_t<TT>> == 2);
        return Pair{static_cast<int>(std::get<0>(t)), std::get<1>(t)};
    }
};

// 1<->N marshalling, move-correct: TrackedPair decomposes into (Tracked,
// Tracked), moving each column in and out rather than copying it.
template<>
struct TorchArgDef<TrackedPair>
{
    static std::tuple<Tracked, Tracked> to(TrackedPair p)
    {
        return {std::move(p.a), std::move(p.b)};
    }

    template<class TT>
    static TrackedPair from(TT &&t)
    {
        return TrackedPair{std::move(std::get<0>(t)), std::move(std::get<1>(t))};
    }
};

// Result structs cross native->dispatch via to_torch_args; to()-only specs (a
// result is never un-marshalled inbound, so no from()).
template<>
struct TorchArgDef<IntResult>
{
    static auto to(const IntResult &r)
    {
        return to_torch_args(r.value);
    }
};

template<>
struct TorchArgDef<DoubleResult>
{
    static auto to(const DoubleResult &r)
    {
        return to_torch_args(r.value);
    }
};

// 1<->1 to a native int64. to() reads the value through a forwarding reference
// (no copy/move of the element), so a tuple round-trip that copies an element
// would show up in CountedScalar's counters.
template<>
struct TorchArgDef<CountedScalar>
{
    template<class U>
    static int64_t to(U &&v)
    {
        return v.value;
    }

    static CountedScalar from(int64_t v)
    {
        return CountedScalar{v};
    }
};
} // namespace gsplat

// ---------------------------------------------------------------------------
// Type mappings: torch_arg_t<T> is the dispatcher type(s) T crosses as -- a bare
// type for a 1<->1 mapping, a tuple for a 1<->N mapping. Asserted at run time
// (EXPECT, not static_assert) so a mismatch surfaces as a test failure rather
// than a build break, and parameterized over the type pairs so each mapping is
// an independent case.
// ---------------------------------------------------------------------------

template<class Natural, class ExpectedTorch>
struct MapCase
{
    using natural  = Natural;
    using expected = ExpectedTorch;
};

template<class Case>
class TorchArgMappingTest : public ::testing::Test
{
};

using MappingCases = ::testing::Types<
    MapCase<float, double>,
    MapCase<const float &, double>,
    MapCase<uint32_t, int64_t>,
    MapCase<int64_t, int64_t>,
    MapCase<UnscopedEnum, int64_t>,
    MapCase<ScopedEnum, int64_t>,
    MapCase<bool, bool>,
    MapCase<at::Tensor, at::Tensor>,
    MapCase<const at::Tensor &, at::Tensor>,
    MapCase<at::optional<at::Tensor>, at::optional<at::Tensor>>,
    MapCase<Pair, std::tuple<int64_t, double>>,
    // optional<torch-native> stays identity; optional<marshalled> marshals per
    // column (a 1<->N inner -> a tuple of optionals, not an optional of tuple).
    MapCase<std::optional<int64_t>, std::optional<int64_t>>,
    MapCase<std::optional<uint32_t>, std::optional<int64_t>>,
    MapCase<std::optional<Pair>, std::tuple<std::optional<int64_t>, std::optional<double>>>
>;
TYPED_TEST_SUITE(TorchArgMappingTest, MappingCases);

TYPED_TEST(TorchArgMappingTest, crosses_as_expected_torch_types)
{
    EXPECT_TRUE((std::is_same_v<gsplat::torch_arg_t<typename TypeParam::natural>, typename TypeParam::expected>));
}

// ---------------------------------------------------------------------------
// to() / from() round-trips.
// ---------------------------------------------------------------------------

TEST(TorchArgTest, scalar_round_trip)
{
    // 1<->1: to() yields the bare dispatcher value, from() takes it directly.
    double f = gsplat::TorchArgDef<float>::to(1.5f);
    EXPECT_DOUBLE_EQ(f, 1.5);
    EXPECT_FLOAT_EQ(gsplat::TorchArgDef<float>::from(f), 1.5f);

    int64_t u = gsplat::TorchArgDef<uint32_t>::to(42u);
    EXPECT_EQ(u, 42);
    EXPECT_EQ(gsplat::TorchArgDef<uint32_t>::from(u), 42u);

    int64_t e = gsplat::TorchArgDef<ScopedEnum>::to(ScopedEnum::B);
    EXPECT_EQ(e, 5);
    EXPECT_EQ(gsplat::TorchArgDef<ScopedEnum>::from(e), ScopedEnum::B);
}

TEST(TorchArgTest, tuple_round_trip)
{
    // 1<->N: to() yields a tuple, from() rebuilds the type from it.
    std::tuple<int64_t, double> t = gsplat::TorchArgDef<Pair>::to(Pair{3, 1.5});
    EXPECT_EQ(std::get<0>(t), 3);
    EXPECT_DOUBLE_EQ(std::get<1>(t), 1.5);

    Pair back = gsplat::TorchArgDef<Pair>::from(t);
    EXPECT_EQ(back.a, 3);
    EXPECT_DOUBLE_EQ(back.b, 1.5);
}

TEST(TorchArgTest, tuple_marshals_non_native_elements)
{
    // A tuple of non-native elements is not passed through: each element is
    // marshalled (1<->1 replace) -- float -> double, ScopedEnum -> int64.
    using Def = gsplat::TorchArgDef<std::tuple<float, ScopedEnum>>;
    auto d    = Def::to(std::tuple<float, ScopedEnum>{1.5f, ScopedEnum::B});
    static_assert(std::is_same_v<decltype(d), std::tuple<double, int64_t>>);
    EXPECT_DOUBLE_EQ(std::get<0>(d), 1.5);
    EXPECT_EQ(std::get<1>(d), 5);

    auto back = Def::from(d);
    static_assert(std::is_same_v<decltype(back), std::tuple<float, ScopedEnum>>);
    EXPECT_FLOAT_EQ(std::get<0>(back), 1.5f);
    EXPECT_EQ(std::get<1>(back), ScopedEnum::B);
}

TEST(TorchArgTest, tuple_expands_one_to_many_element)
{
    // A 1<->N element (Pair -> (int64, double)) is expanded into its columns and
    // spliced; a scalar element is replaced. tuple<Pair, int> crosses as
    // (int64, double, int64), and from() regroups it column-for-column.
    using Def = gsplat::TorchArgDef<std::tuple<Pair, int>>;
    auto d    = Def::to(
        std::tuple<Pair, int>{
            Pair{3, 1.5},
            7
    }
    );
    static_assert(std::is_same_v<decltype(d), std::tuple<int64_t, double, int64_t>>);
    EXPECT_EQ(std::get<0>(d), 3);
    EXPECT_DOUBLE_EQ(std::get<1>(d), 1.5);
    EXPECT_EQ(std::get<2>(d), 7);

    auto back = Def::from(d);
    EXPECT_EQ(std::get<0>(back).a, 3);
    EXPECT_DOUBLE_EQ(std::get<0>(back).b, 1.5);
    EXPECT_EQ(std::get<1>(back), 7);
}

TEST(TorchArgTest, tuple_all_native_is_identity)
{
    // When every element is already a dispatch type the tuple crosses unchanged.
    using Def = gsplat::TorchArgDef<std::tuple<int64_t, double>>;
    auto d    = Def::to(std::tuple<int64_t, double>{3, 1.5});
    static_assert(std::is_same_v<decltype(d), std::tuple<int64_t, double>>);
    EXPECT_EQ(std::get<0>(d), 3);
    EXPECT_DOUBLE_EQ(std::get<1>(d), 1.5);

    auto back = Def::from(d);
    static_assert(std::is_same_v<decltype(back), std::tuple<int64_t, double>>);
    EXPECT_EQ(std::get<0>(back), 3);
}

TEST(TorchArgTest, tuple_marshals_without_spurious_copy)
{
    // The recursive tuple path forwards each element to its TorchArgDef and moves
    // it through to_torch_args / the regroup -- it never copies an element.
    using Def = gsplat::TorchArgDef<std::tuple<CountedScalar, CountedScalar>>;

    std::tuple<CountedScalar, CountedScalar> t{CountedScalar{1}, CountedScalar{2}};
    CountedScalar::reset();
    auto d = Def::to(std::move(t));
    EXPECT_EQ(CountedScalar::copies, 0) << "tuple to() copied an element";
    static_assert(std::is_same_v<decltype(d), std::tuple<int64_t, int64_t>>);
    EXPECT_EQ(std::get<0>(d), 1);
    EXPECT_EQ(std::get<1>(d), 2);

    CountedScalar::reset();
    auto back = Def::from(std::move(d));
    EXPECT_EQ(CountedScalar::copies, 0) << "tuple from() copied an element";
    EXPECT_EQ(std::get<0>(back).value, 1);
    EXPECT_EQ(std::get<1>(back).value, 2);
}

TEST(TorchArgTest, optional_scalar_round_trip)
{
    // 1<->1 marshalled inner: optional<uint32> crosses as optional<int64>.
    using Def = gsplat::TorchArgDef<std::optional<uint32_t>>;

    // present: the inner uint32 marshals to int64 inside the optional.
    std::optional<int64_t> d = Def::to(std::optional<uint32_t>(42u));
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(*d, 42);

    std::optional<uint32_t> back = Def::from(d);
    ASSERT_TRUE(back.has_value());
    EXPECT_EQ(*back, 42u);

    // absent round-trips to absent.
    EXPECT_FALSE(Def::to(std::optional<uint32_t>{}).has_value());
    EXPECT_FALSE(Def::from(std::optional<int64_t>{}).has_value());
}

TEST(TorchArgTest, optional_tuple_round_trip)
{
    // 1<->N inner: optional<Pair> crosses as a tuple of one optional per column
    // (optional<int64>, optional<double>) -- never an optional of a tuple.
    using Def = gsplat::TorchArgDef<std::optional<Pair>>;

    // present: one optional per column, each carrying the marshalled value.
    auto cols = Def::to(std::optional<Pair>(Pair{3, 1.5}));
    static_assert(std::tuple_size_v<decltype(cols)> == 2);

    ASSERT_TRUE(std::get<0>(cols).has_value());
    ASSERT_TRUE(std::get<1>(cols).has_value());
    EXPECT_EQ(*std::get<0>(cols), 3);
    EXPECT_DOUBLE_EQ(*std::get<1>(cols), 1.5);

    std::optional<Pair> back = Def::from(cols);
    ASSERT_TRUE(back.has_value());
    EXPECT_EQ(back->a, 3);
    EXPECT_DOUBLE_EQ(back->b, 1.5);

    // absent: every column nullopt, and from() reads absence from the first.
    auto none = Def::to(std::optional<Pair>{});
    EXPECT_FALSE(std::get<0>(none).has_value());
    EXPECT_FALSE(std::get<1>(none).has_value());
    EXPECT_FALSE(Def::from(none).has_value());
}

// ---------------------------------------------------------------------------
// No-spurious-copy guard: a TorchArgDef path must move the wrapped payload
// through to()/from(), never copy it. Parametrized over the marshalling shapes
// so each is its own case; the body counts Tracked copies.
// ---------------------------------------------------------------------------

// Build a payload of the case type carrying instrumented Tracked values.
template<class T>
T make_tracked_payload();

template<>
Tracked make_tracked_payload<Tracked>()
{
    return Tracked{1};
}

template<>
std::optional<Tracked> make_tracked_payload<std::optional<Tracked>>()
{
    return Tracked{1};
}

template<>
TrackedPair make_tracked_payload<TrackedPair>()
{
    return TrackedPair{Tracked{1}, Tracked{2}};
}

template<>
std::optional<TrackedPair> make_tracked_payload<std::optional<TrackedPair>>()
{
    return TrackedPair{Tracked{1}, Tracked{2}};
}

template<class T>
class NoSpuriousCopyTest : public ::testing::Test
{
};

using CopyCases = ::testing::Types<
    Tracked,                // identity (1<->1)
    std::optional<Tracked>, // 1<->1 optional
    TrackedPair,            // 1<->N struct
    std::optional<TrackedPair>
>; // 1<->N optional
TYPED_TEST_SUITE(NoSpuriousCopyTest, CopyCases);

TYPED_TEST(NoSpuriousCopyTest, marshals_without_spurious_copy)
{
    using T   = TypeParam;
    using Def = gsplat::TorchArgDef<T>;

    // to(): crossing an owned (rvalue) value moves the payload out, never copies.
    T value = make_tracked_payload<T>();
    Tracked::reset();
    auto crossed = Def::to(std::move(value));
    EXPECT_EQ(Tracked::copies, 0) << "to() introduced a spurious copy";

    // from(): rebuilding from an owned dispatcher value moves it back, no copy.
    Tracked::reset();
    T back = Def::from(std::move(crossed));
    EXPECT_EQ(Tracked::copies, 0) << "from() introduced a spurious copy";
    (void)back;
}

// ---------------------------------------------------------------------------
// to_torch_op wiring.
// ---------------------------------------------------------------------------

TEST(ToTorchOpTest, flattens_one_to_many_and_regroups)
{
    // pair_then_scalar(Pair, int64_t): Pair -> (int64_t, double), so the
    // registered wrapper takes (int64_t, double, int64_t) and rebuilds the Pair
    // from the first two before calling the op.
    auto result = gsplat::to_torch_op<&pair_then_scalar>(2, 3.5, 10);
    // DoubleResult has one field, so the boxed result comes back bare (1<->1).
    EXPECT_DOUBLE_EQ(result, 15.5);
}

TEST(ToTorchOpTest, converts_scalar_argument)
{
    // scale(float): the wrapper takes a double and narrows it back to float.
    auto result = gsplat::to_torch_op<&scale>(2.5);
    EXPECT_DOUBLE_EQ(result, 5.0);
}

TEST(ToTorchOpTest, identity_path_does_not_copy)
{
    // The wrapper forwards a native argument by const& all the way to the op, so
    // no intermediate copy bumps its refcount (and a const& cannot be moved from).
    at::Tensor t   = at::ones({2});
    int64_t before = static_cast<int64_t>(t.use_count());
    auto inside    = gsplat::to_torch_op<&read_tensor_use_count>(t);
    EXPECT_EQ(inside, before);
}

// ---------------------------------------------------------------------------
// to_torch_args: recursive native->dispatch flattening (the result-boxing core).
// ---------------------------------------------------------------------------

TEST(ToTorchArgsTest, flattens_and_recurses_to_dispatch_types)
{
    // enum -> int64, float -> double, Pair -> (int64, double); a non-dispatch
    // value (the Pair) is reduced by its TorchArgDef::to and spliced in flat, so
    // every element of the result is a dispatch type, in order.
    auto flat = gsplat::to_torch_args(ScopedEnum::B, 1.5f, Pair{3, 2.5});
    EXPECT_TRUE((std::is_same_v<decltype(flat), std::tuple<int64_t, double, int64_t, double>>));
    EXPECT_EQ(std::get<0>(flat), 5);
    EXPECT_DOUBLE_EQ(std::get<1>(flat), 1.5);
    EXPECT_EQ(std::get<2>(flat), 3);
    EXPECT_DOUBLE_EQ(std::get<3>(flat), 2.5);
}

TEST(ToTorchArgsTest, unwraps_single_dispatch_value)
{
    // A single arg mapping to one dispatch value comes back bare (the 1<->1 case),
    // not wrapped in a 1-tuple, so it matches a single-output op's m.impl return.
    auto one = gsplat::to_torch_args(int64_t{7});
    EXPECT_TRUE((std::is_same_v<decltype(one), int64_t>));
    EXPECT_EQ(one, 7);

    // A single arg mapping to several dispatch values stays a tuple (the 1<->N case).
    auto pair = gsplat::to_torch_args(Pair{3, 1.5});
    EXPECT_TRUE((std::is_same_v<decltype(pair), std::tuple<int64_t, double>>));
    EXPECT_EQ(std::get<0>(pair), 3);
    EXPECT_DOUBLE_EQ(std::get<1>(pair), 1.5);

    // Zero args -> empty tuple.
    auto none = gsplat::to_torch_args();
    EXPECT_TRUE((std::is_same_v<decltype(none), std::tuple<>>));
}

// ---------------------------------------------------------------------------
// call_torch_op: the dispatcher-CALL mirror of to_torch_op. A native call site
// passes native args and receives the native result struct, while the op is
// reached through the dispatcher (so an Autograd-key registration would fire).
// Exercised end-to-end against a self-contained op registered here -- the gtest
// binary does not link ext.cpp, so no gsplat:: ops exist in this process.
// ---------------------------------------------------------------------------

namespace gsplat
{
// A multi-output result with an optional field: the same shape (Tensor, Tensor,
// Tensor?) the real projection/rasterize forwards produce, so the test covers
// both the multi-value and the optional reconstruction paths of from().
struct CallTorchOpResult
{
    at::Tensor scaled;
    at::Tensor shifted;
    at::optional<at::Tensor> extra;
};

template<>
struct TorchArgDef<CallTorchOpResult>
{
    static auto to(const CallTorchOpResult &r)
    {
        return to_torch_args(r.scaled, r.shifted, r.extra);
    }

    template<class TT>
    static CallTorchOpResult from(TT &&t)
    {
        return CallTorchOpResult{
            std::get<0>(std::forward<TT>(t)),
            std::get<1>(t),
            std::get<2>(t),
        };
    }
};

// Native-typed forward: scalar args (double/int64/bool) exercise arg flattening;
// the optional output is present iff `with_extra`.
CallTorchOpResult call_torch_op_probe_fwd(const at::Tensor &a, double scale, int64_t shift, bool with_extra)
{
    return CallTorchOpResult{
        a * scale,
        a + static_cast<double>(shift),
        with_extra ? at::optional<at::Tensor>(a.clone()) : at::nullopt,
    };
}
} // namespace gsplat

TORCH_LIBRARY(gsplat_call_torch_op_test, m)
{
    m.def(
        "probe(Tensor a, float scale, int shift, bool with_extra) -> (Tensor, Tensor, Tensor?)",
        gsplat::to_torch_op<&gsplat::call_torch_op_probe_fwd>
    );
}

TEST(CallTorchOpTest, struct_round_trips_through_dispatcher)
{
    at::Tensor a                = at::tensor({1.0f, 2.0f, 3.0f});
    gsplat::CallTorchOpResult r = gsplat::call_torch_op<&gsplat::call_torch_op_probe_fwd>(
        "gsplat_call_torch_op_test::probe", a, /*scale=*/2.0, /*shift=*/10, /*with_extra=*/true
    );
    EXPECT_TRUE(at::allclose(r.scaled, a * 2.0));
    EXPECT_TRUE(at::allclose(r.shifted, a + 10.0));
    ASSERT_TRUE(r.extra.has_value());
    EXPECT_TRUE(at::allclose(r.extra.value(), a));
}

TEST(CallTorchOpTest, signature_form_round_trips_through_dispatcher)
{
    at::Tensor a                = at::tensor({1.0f, 2.0f, 3.0f});
    // The signature-explicit overload: the op's native signature is supplied as
    // a type instead of a forward-function pointer. Marshalling and result
    // reconstruction otherwise match the forward-pointer form.
    using Sig                   = gsplat::CallTorchOpResult(const at::Tensor &, double, int64_t, bool);
    gsplat::CallTorchOpResult r = gsplat::call_torch_op<Sig>(
        "gsplat_call_torch_op_test::probe", a, /*scale=*/2.0, /*shift=*/10, /*with_extra=*/true
    );
    EXPECT_TRUE(at::allclose(r.scaled, a * 2.0));
    EXPECT_TRUE(at::allclose(r.shifted, a + 10.0));
    ASSERT_TRUE(r.extra.has_value());
    EXPECT_TRUE(at::allclose(r.extra.value(), a));
}

TEST(CallTorchOpTest, optional_output_absent_round_trips_as_nullopt)
{
    at::Tensor a                = at::tensor({4.0f, 5.0f});
    gsplat::CallTorchOpResult r = gsplat::call_torch_op<&gsplat::call_torch_op_probe_fwd>(
        "gsplat_call_torch_op_test::probe", a, /*scale=*/1.0, /*shift=*/0, /*with_extra=*/false
    );
    EXPECT_TRUE(at::allclose(r.scaled, a));
    EXPECT_FALSE(r.extra.has_value());
}

// ---------------------------------------------------------------------------
// ctx_save / ctx_load / apply_bwd end-to-end through a throwaway
// torch::autograd::Function. The backward free-function's signature is the
// saved-state source of truth: leading saved params, trailing grad bundle.
// Modeled on the autograd Functions in Projection.cpp (forward uses ctx_save,
// backward uses apply_bwd; backward free-fn takes saved params then a trailing
// grad-bundle reference). Saves a tensor pair plus a non-tensor scalar to
// exercise both the save_for_backward and saved_data IValue routes.
// ---------------------------------------------------------------------------

namespace
{
// Grad bundle: the engine-supplied trailing parameter, opted in via the
// is_grad_bundle tag (mirrors the *Grad structs in Projection.cpp).
struct ScaleMulGrad
{
    static constexpr bool is_grad_bundle = true;
    at::Tensor grad_out;
};

struct ScaleMulBwdResult
{
    at::Tensor v_a;
    at::Tensor v_b;
};

// Backward free-function: saved state (a, b, scale) then the grad bundle.
// y = a * b * scale  =>  dy/da = b * scale, dy/db = a * scale.
ScaleMulBwdResult scale_mul_bwd(const at::Tensor &a, const at::Tensor &b, double scale, const ScaleMulGrad &grad)
{
    return ScaleMulBwdResult{
        .v_a = grad.grad_out * b * scale,
        .v_b = grad.grad_out * a * scale,
    };
}

// Records the saved state ctx_load reconstructed, so a test can assert the
// round-trip recovers exactly what forward saved.
struct CtxLoadProbe
{
    static inline bool ran = false;
    static inline at::Tensor a;
    static inline at::Tensor b;
    static inline double scale = 0.0;
};

class ScaleMulAutograd : public torch::autograd::Function<ScaleMulAutograd>
{
public:
    struct FwdInput
    {
        // SCALE is a non-differentiable double input; its grad slot stays an
        // undefined tensor. COUNT must equal forward()'s input arity (3) so the
        // returned grad list is sized correctly.
        enum
        {
            A,
            B,
            SCALE,
            COUNT
        };
    };

    struct FwdOutput
    {
        enum
        {
            Y,
            COUNT
        };
    };

    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx, const at::Tensor &a, const at::Tensor &b, double scale
    )
    {
        static_assert(
            FwdInput::COUNT == gsplat::fwd_input_count<&forward>(),
            "FwdInput must have one enumerator per forward input"
        );

        at::Tensor y = a * b * scale;

        // Save the values backward needs, type-checked against scale_mul_bwd's
        // parameters (a, b, scale) in order.
        gsplat::ctx_save<&scale_mul_bwd>(ctx, a, b, scale);

        torch::autograd::variable_list out(FwdOutput::COUNT);
        out[FwdOutput::Y] = y;
        return out;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs
    )
    {
        // Round-trip the saved state out, to exercise ctx_load explicitly (the
        // saved variables are only valid to unpack during backward). apply_bwd
        // below re-loads them itself to invoke scale_mul_bwd.
        auto [la, lb, lscale] = gsplat::ctx_load<&scale_mul_bwd>(ctx);
        CtxLoadProbe::ran     = true;
        CtxLoadProbe::a       = la;
        CtxLoadProbe::b       = lb;
        CtxLoadProbe::scale   = lscale;

        ScaleMulGrad grad{.grad_out = grad_outputs[FwdOutput::Y]};
        ScaleMulBwdResult g = gsplat::apply_bwd<&scale_mul_bwd>(ctx, grad);

        torch::autograd::variable_list grads(FwdInput::COUNT);
        grads[FwdInput::A] = g.v_a;
        grads[FwdInput::B] = g.v_b;
        return grads;
    }
};
} // namespace

TEST(CtxSaveLoadApplyBwdTest, round_trips_saved_state_and_computes_gradients)
{
    constexpr double kScale = 2.0;
    at::Tensor a            = at::tensor({1.0, 2.0, 3.0}).set_requires_grad(true);
    at::Tensor b            = at::tensor({4.0, 5.0, 6.0}).set_requires_grad(true);

    CtxLoadProbe::ran = false;

    // Forward: y = a * b * scale. ctx_save stores (a, b, scale) for backward.
    torch::autograd::variable_list out = ScaleMulAutograd::apply(a, b, kScale);
    at::Tensor y                       = out[ScaleMulAutograd::FwdOutput::Y];
    EXPECT_TRUE(at::allclose(y, a.detach() * b.detach() * kScale));

    // Backward runs the saved-state plumbing: an explicit ctx_load round-trip
    // (recorded into the probe) plus apply_bwd, which re-loads the state from
    // ctx and invokes scale_mul_bwd with the grad bundle appended.
    y.sum().backward();

    // ctx_load round-trip: the saved tensors and scalar come back intact, in
    // order (tensors via save_for_backward, scalar via saved_data IValue).
    ASSERT_TRUE(CtxLoadProbe::ran);
    EXPECT_TRUE(at::equal(CtxLoadProbe::a, a.detach()));
    EXPECT_TRUE(at::equal(CtxLoadProbe::b, b.detach()));
    EXPECT_DOUBLE_EQ(CtxLoadProbe::scale, kScale);

    // dy/da = b * scale, dy/db = a * scale (grad_out is ones from .sum()).
    ASSERT_TRUE(a.grad().defined());
    ASSERT_TRUE(b.grad().defined());
    EXPECT_TRUE(at::allclose(a.grad(), b.detach() * kScale));
    EXPECT_TRUE(at::allclose(b.grad(), a.detach() * kScale));
}

namespace
{
// A second forward signature, with a different arity than ScaleMulAutograd's.
// The test exercises fwd_input_count with a different parameter count to verify
// it tracks the arity encoded in the function type. Never invoked; only its
// type is inspected.
torch::autograd::variable_list two_input_forward(
    torch::autograd::AutogradContext * /*ctx*/, const at::Tensor & /*x*/, const at::Tensor & /*y*/
)
{
    return {};
}
} // namespace

// fwd_input_count<&Fwd>() == forward's arity minus the leading AutogradContext*.
// It is a compile-time constant, so assert it via static_assert as well.
TEST(FwdInputCountTest, counts_inputs_excluding_context)
{
    static_assert(gsplat::fwd_input_count<&ScaleMulAutograd::forward>() == 3);
    static_assert(gsplat::fwd_input_count<&two_input_forward>() == 2);
    EXPECT_EQ(gsplat::fwd_input_count<&ScaleMulAutograd::forward>(), 3u);
    EXPECT_EQ(gsplat::fwd_input_count<&two_input_forward>(), 2u);
}
