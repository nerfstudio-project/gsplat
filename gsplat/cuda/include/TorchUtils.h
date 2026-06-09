/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/core/List.h> // c10::List<at::Tensor> for is_tensor_list
#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h> // c10::Dispatcher for call_torch_op
#include <ATen/core/grad_mode.h>
#include <ATen/core/ivalue.h> // c10::IValue for saved_data
#include <ATen/ops/sparse_coo_tensor.h> // at::sparse_coo_tensor

#include <concepts>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <c10/util/intrusive_ptr.h>

#include <torch/csrc/autograd/custom_function.h> // torch::autograd::AutogradContext / variable_list

namespace gsplat {

namespace detail {

// Classification traits for the dispatcher marshalling and the saved-state
// plumbing. Each exposes an `inner` alias defined for ALL types (void when no
// match) so dependent member access is always well-formed -- an `if constexpr`
// `&&` parses both operands, so a guarded `typename V::value_type` would be
// ill-formed for non-optional V otherwise.

// is_optional<T>: value_type is the optional's element type, else void.
template <class T>
struct is_optional : std::false_type {
    using value_type = void;
};

template <class T>
struct is_optional<at::optional<T>> : std::true_type {
    using value_type = T;
};

// is_intrusive_ptr<T>: value_type is the held class, else void.
template <class T>
struct is_intrusive_ptr : std::false_type {
    using value_type = void;
};

template <class T>
struct is_intrusive_ptr<c10::intrusive_ptr<T>> : std::true_type {
    using value_type = T;
};

// Extracts a classifier's value_type (its element/held type, or void).
template <class T>
using value_type_t = typename T::value_type;

// is_tensor_list<T>: a sequence-of-tensors type. A saved input of such a type
// must round-trip through save_for_backward, not saved_data -- the scalar/else
// routing static_asserts against it, since storing tensors as a saved_data
// IValue bypasses the saved-variable version/aliasing/double-backward checks.
template <class T>
struct is_tensor_list : std::false_type {};

template <>
struct is_tensor_list<std::vector<at::Tensor>> : std::true_type {};

template <>
struct is_tensor_list<c10::List<at::Tensor>> : std::true_type {};

template <>
struct is_tensor_list<at::TensorList> : std::true_type {};

template <class T>
inline constexpr bool is_tensor_list_v = is_tensor_list<T>::value;

template <class V>
inline constexpr bool is_optional_tensor_v =
    std::is_same_v<value_type_t<is_optional<V>>, at::Tensor>;

template <class V>
inline constexpr bool is_optional_intrusive_v =
    is_intrusive_ptr<value_type_t<is_optional<V>>>::value;

} // namespace detail

// Round-trip helpers between `at::optional<at::Tensor>` and `at::Tensor`.
// An undefined Tensor (default-constructed, `.defined() == false`) is the
// sentinel for the `nullopt` side; the two helpers are inverses, so an
// optional converted to a Tensor and back yields the original value.
inline at::optional<at::Tensor> as_optional_tensor(const at::Tensor &tensor) {
    if (tensor.defined()) {
        return tensor;
    }
    return c10::nullopt;
}

inline at::Tensor as_tensor(const at::optional<at::Tensor> &tensor) {
    return tensor.value_or(at::Tensor{});
}

inline bool tensor_requires_grad(const at::Tensor &tensor) {
    return tensor.defined() && tensor.requires_grad();
}

inline bool tensor_requires_grad(const at::optional<at::Tensor> &tensor) {
    return tensor.has_value() && tensor.value().requires_grad();
}

// Mirrors torch::autograd::Function::apply's node-creation predicate for
// wrappers whose custom forward does non-trivial saved-state setup.
template <typename... Tensors>
inline bool needs_custom_autograd(const Tensors &...tensors) {
    return at::GradMode::is_enabled() &&
           (tensor_requires_grad(tensors) || ...);
}

namespace detail {

template <class T>
inline T *tensor_data_ptr(const at::Tensor &tensor)
{
    using ValueT = std::remove_const_t<T>;
    if constexpr (std::is_const_v<T>) {
        return tensor.const_data_ptr<ValueT>();
    } else {
        return tensor.mutable_data_ptr<ValueT>();
    }
}

} // namespace detail

/** Return a data pointer with constness inferred from `T`.
 *
 * `T = float` returns a mutable `float *`; `T = const float` returns a
 * read-only `const float *`.
 */
template <class T>
inline T *data_ptr(const at::Tensor &tensor)
{
    return detail::tensor_data_ptr<T>(tensor);
}

/** Return a data pointer when the tensor is enabled, or nullptr.
 *
 * Pointer constness is inferred from `T`.
 * The tensor is not inspected when `enabled == false`, which lets callers pass
 * placeholder tensors for code paths that do not allocate a given buffer.
 */
template <class T>
inline T *data_ptr_or_null(bool enabled, const at::Tensor &tensor)
{
    return enabled ? data_ptr<T>(tensor) : nullptr;
}

/** Return a data pointer for an optional tensor, or nullptr.
 *
 * Pointer constness is inferred from `T`. Empty optionals map to nullptr.
 */
template <class T>
inline T *data_ptr_or_null(const at::optional<at::Tensor> &tensor)
{
    return tensor.has_value() ? data_ptr<T>(tensor.value()) : nullptr;
}

/** Return an optional tensor pointer only when both gates are true. */
template <class T>
inline T *data_ptr_or_null(
    bool enabled,
    const at::optional<at::Tensor> &tensor)
{
    return enabled ? data_ptr_or_null<T>(tensor) : nullptr;
}

/** Return a tensor data pointer reinterpreted as `PointerT *`.
 *
 * Pointer constness is inferred from `PointerT`. `ScalarT` names the tensor
 * element type before reinterpretation.
 */
template <class PointerT, class ScalarT>
inline PointerT *data_ptr_as(const at::Tensor &tensor)
{
    static_assert(
        !std::is_const_v<ScalarT>,
        "Put const on PointerT, not ScalarT");
    using TensorPointerT = std::conditional_t<
        std::is_const_v<PointerT>,
        const ScalarT,
        ScalarT>;
    return reinterpret_cast<PointerT *>(data_ptr<TensorPointerT>(tensor));
}

/** Return a reinterpreted data pointer when enabled, or nullptr.
 *
 * Pointer constness is inferred from `PointerT`.
 */
template <class PointerT, class ScalarT>
inline PointerT *data_ptr_as_or_null(bool enabled, const at::Tensor &tensor)
{
    return enabled ? data_ptr_as<PointerT, ScalarT>(tensor) : nullptr;
}

/** Dereference a required intrusive_ptr, or raise a clear error if unset.
 *
 * `name` is woven into the error message so a missing field reports itself
 * by name instead of crashing on a null dereference.
 */
template <class T>
inline const T &checked_deref(const c10::intrusive_ptr<T> &ptr, const char *name)
{
    TORCH_CHECK(ptr, name, " must be set");
    return *ptr;
}

/** Dereference a required `at::optional<intrusive_ptr>`, or raise a clear error
 * if the optional is empty or the pointer it holds is null. Both failure modes
 * report through the same `name`, so the deref site needs no separate presence
 * check on the optional.
 */
template <class T>
inline const T &checked_deref(const at::optional<c10::intrusive_ptr<T>> &opt, const char *name)
{
    TORCH_CHECK(opt.has_value(), name, " must be set");
    return checked_deref(*opt, name);
}

inline at::optional<at::Tensor> contiguous_optional(const at::optional<at::Tensor> &tensor) {
    if (tensor.has_value()) {
        return tensor.value().contiguous();
    }
    return c10::nullopt;
}

inline at::Tensor make_sparse_coo_grad(const at::Tensor &indices, const at::Tensor &values,
                                       at::IntArrayRef size, bool is_coalesced) {
    return at::sparse_coo_tensor(indices, values, size,
                                 c10::nullopt, // dtype, inferred from values
                                 c10::nullopt, // layout
                                 c10::nullopt, // device, inferred from indices/values
                                 c10::nullopt, // pin_memory
                                 is_coalesced);
}

// ---------------------------------------------------------------------------
// TorchArgDef: how a native (C++) parameter type crosses the torch dispatcher.
//
// torch's schema speaks only Tensor / int64 / double / bool / str / custom
// classes. A kernel-facing type -- an enum, a uint32_t, a float -- must be
// marshalled to one or more of those. `to()` and `from()` are inverses:
//   - a 1<->1 mapping uses a bare torch value (no tuple);
//   - a 1<->N mapping uses a std::tuple of the several torch arguments.
//
//   - Primary template: torch types pass through unchanged (const&
//     preserved by `to_torch_op`). A type that is neither a torch type nor
//     matched by a partial specialization fails the static_assert -- add a
//     specialization rather than letting an unmarshallable type reach the
//     dispatcher.
//   - Partial specializations marshal enums and small scalars to int64/double.
//
// `torch_arg_t<T>` exposes the torch type(s) for outside consumers.
// `to_torch_op` (below) is the wiring: it flattens every argument's torch
// types into the registered `m.impl` signature and regroups the incoming
// torch arguments back through `from()` before calling the op.
//
// Extending it: to let a new kernel-facing type cross the dispatcher, add a
// TorchArgDef<T> specialization -- a full specialization for one concrete type,
// or a partial one constrained by a concept/requires-clause (as the enum,
// integral, and floating-point specializations below do) for a whole family.
// Each specialization provides the inverse pair: to() maps the native value to
// its torch carrier(s) -- a bare torch value for a 1<->1 mapping, a std::tuple
// for 1<->N -- and from() reconstructs the native value from those carrier(s).
// ---------------------------------------------------------------------------

namespace detail {

// A type the torch dispatcher carries directly: it crosses through the identity
// TorchArgDef unchanged, with no marshalling.
template <class T>
struct is_torch
    : std::bool_constant<std::same_as<T, at::Tensor> || std::same_as<T, int64_t> ||
                         std::same_as<T, double> || std::same_as<T, bool> ||
                         std::same_as<T, std::string> ||
                         std::same_as<T, at::SymIntArrayRef> ||
                         is_intrusive_ptr<T>::value> {};

// optional<T> is a torch type exactly when T is: inherit T's trait directly.
template <class X>
struct is_torch<std::optional<X>> : is_torch<X> {};

// A tuple is a torch type exactly when every element is (then it is the flat
// pack of torch values that crosses unchanged -- the 1<->N representation).
template <class... Ts>
struct is_torch<std::tuple<Ts...>> : std::conjunction<is_torch<Ts>...> {};

template <class T>
concept IsTorch = is_torch<T>::value;

// std::integral / std::floating_point are standard concepts; enums have no
// standard concept, so define one for symmetry in the specializations below.
template <class T>
concept Enumeration = std::is_enum_v<T>;

} // namespace detail

// Primary template = the identity TorchArgDef: a torch type crosses
// unchanged. A type that reaches here without being a torch type has no
// marshalling specialization and cannot cross the dispatcher; add a TorchArgDef
// specialization or keep it C++-resident.
template <class T>
struct TorchArgDef {
    static_assert(detail::IsTorch<T>,
                  "TorchArgDef: type is not a torch type and has no marshalling "
                  "specialization; it cannot cross the dispatcher. Add a TorchArgDef<T> "
                  "specialization or keep it C++-resident.");
    // 1<->1 identity: a torch type crosses unchanged. `to` yields a
    // decayed value; `from` forwards the torch reference through, so an
    // identity argument reaches the op with no copy.
    template <class U>
    static auto to(U &&v) {
        return std::forward<U>(v);
    }
    template <class U>
    static decltype(auto) from(U &&v) {
        return std::forward<U>(v);
    }
};

// Strip cv/ref qualifiers so callers can instantiate TorchArgDef with an op's
// declared parameter type (e.g. const at::Tensor&) directly; the work is
// delegated to the decayed specialization.
template <class T>
    requires(!std::same_as<T, std::decay_t<T>>)
struct TorchArgDef<T> : TorchArgDef<std::decay_t<T>> {};

// Enums (scoped or unscoped) cross as int64.
template <detail::Enumeration T>
struct TorchArgDef<T> {
    static int64_t to(T v) { return static_cast<int64_t>(v); }
    static T from(int64_t v) { return static_cast<T>(v); }
};

// Integral types other than bool cross as int64.
template <class T>
    requires(std::integral<T> && !std::same_as<T, bool>)
struct TorchArgDef<T> {
    static int64_t to(T v) { return static_cast<int64_t>(v); }
    static T from(int64_t v) { return static_cast<T>(v); }
};

// Floating-point types cross as double.
template <std::floating_point T>
struct TorchArgDef<T> {
    static double to(T v) { return static_cast<double>(v); }
    static T from(double v) { return static_cast<T>(v); }
};

// Both defined below; forward-declared here so the tuple TorchArgDef (just below)
// can marshal each element through to_torch_args (in its to()) and regroup the
// torch values via ToNativeArgs (in its from()).
template <class... Args>
auto to_torch_args(Args &&...args);

namespace detail {
template <class ArgsTuple>
struct ToNativeArgs;
} // namespace detail

// A std::tuple with a non-torch element crosses element-wise: each element is
// marshalled through its own TorchArgDef -- a 1<->1 element replaced by its
// torch value, a 1<->N element expanded into its values -- and the results
// concatenated (the same flattening to_torch_args performs). An all-torch tuple
// is itself a pack of torch values, so it is not marshalled here: it falls
// to the primary (identity) spec. (Result structs own their TorchArgDef specs.)
template <class... Ts>
    requires(!detail::IsTorch<std::tuple<Ts...>>)
struct TorchArgDef<std::tuple<Ts...>> {
    static auto to(std::tuple<Ts...> t) {
        return std::apply(
            [](auto &&...es) { return to_torch_args(std::forward<decltype(es)>(es)...); },
            std::move(t));
    }
    // `d` is the flat value pack to()/torch_arg_t yields (bare when it is a
    // single value); to_tuple normalizes a bare value into a 1-tuple.
    template <class D>
    static std::tuple<Ts...> from(D &&d) {
        return detail::ToNativeArgs<std::tuple<Ts...>>::to_tuple(std::forward<D>(d));
    }
};

// The torch type(s) T crosses as: a bare type for a 1<->1 mapping, a tuple
// for a 1<->N mapping -- exactly what TorchArgDef<T>::to() yields.
template <class T>
using torch_arg_t = decltype(TorchArgDef<T>::to(std::declval<T>()));

namespace detail {

// Decomposes a free-function pointer R(*)(A...): its return type, its parameter
// tuple (as declared), and the decayed parameter tuple. Shared by to_torch_op
// (reads `args`) and the saved-state plumbing (reads `decayed_args`).
template <class F>
struct fn_traits;

template <class R, class... A>
struct fn_traits<R (*)(A...)> {
    using return_type = R;
    using args = std::tuple<A...>;
    using decayed_args = std::tuple<std::decay_t<A>...>;
};

// Parameter form of a flat torch type: by const& for class types, by value
// for scalars.
template <class T>
using as_param_t = std::conditional_t<std::is_scalar_v<T>, T, const T &>;

// IsTuple<T>: whether T is a std::tuple (the form a 1<->N marshalling yields).
// Detected the way the STL detects tuple-like types -- a class-template partial
// specialization -- fronted by a concept.
template <class T>
struct is_tuple : std::false_type {};

template <class... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};

template <class T>
concept IsTuple = is_tuple<T>::value;

// flatten_one: one value -> a tuple of its torch values. A single torch
// scalar / Tensor becomes a 1-tuple; anything else (a non-torch scalar/struct,
// or any tuple) is reduced by TorchArgDef::to -- a 1<->1 value to a bare value, a
// 1<->N value or a tuple to its value tuple -- and that result is spliced in.
// (A torch tuple is itself a value pack, so it splices rather than wrapping.)
template <class A>
auto flatten_one(A &&a) {
    using T = std::decay_t<A>;
    if constexpr (IsTorch<T> && !IsTuple<T>) {
        return std::make_tuple(std::forward<A>(a));
    } else {
        auto reduced = TorchArgDef<T>::to(std::forward<A>(a));
        if constexpr (IsTuple<std::decay_t<decltype(reduced)>>) {
            return reduced;
        } else {
            return std::make_tuple(std::move(reduced));
        }
    }
}

// as_tuple<T>: normalize a torch type to a tuple -- a bare 1<->1 type is
// wrapped in a 1-tuple, a 1<->N tuple passes through unchanged.
template <class R>
struct as_tuple {
    using type = std::tuple<R>;
};

template <class... Ts>
struct as_tuple<std::tuple<Ts...>> {
    using type = std::tuple<Ts...>;
};

template <class T>
using as_tuple_t = typename as_tuple<T>::type;

// The flat m.impl parameter pack for an op: each argument's torch tuple
// (as_tuple_t<torch_arg_t<Arg>>) concatenated. Takes the op's argument tuple so
// ToTorchOp can bind it straight from fn_traits, with no intermediate template.
template <class ArgsTuple>
struct flat_torch_types;

template <class... Args>
struct flat_torch_types<std::tuple<Args...>> {
    using type = decltype(std::tuple_cat(std::declval<as_tuple_t<torch_arg_t<Args>>>()...));
};

template <class ArgsTuple>
using flat_torch_types_t = typename flat_torch_types<ArgsTuple>::type;

// Regroups a flat torch tuple back into the native values it was flattened
// from: each type T_i consumes its value count (1 for a 1<->1 mapping, N for a
// 1<->N one) and is rebuilt via TorchArgDef<T_i>::from (a 1<->1 value bare, a
// 1<->N value as its value slice). Shared by the to_torch_op kernel (which
// spreads the rebuilt values into the op) and by TorchArgDef<std::tuple<...>>
// (which assembles them into the tuple). A torch value forwards with no copy.
template <class ArgsTuple>
struct ToNativeArgs;

template <class... Args>
struct ToNativeArgs<std::tuple<Args...>> {
    // Start of argument `arg` in the flat pack: prefix sum of preceding arities.
    static constexpr std::size_t offset(std::size_t arg) {
        constexpr std::size_t arity[sizeof...(Args)] = {
            std::tuple_size_v<as_tuple_t<torch_arg_t<Args>>>...};
        std::size_t off = 0;
        for (std::size_t i = 0; i < arg; ++i) {
            off += arity[i];
        }
        return off;
    }

    // Rebuild argument `I` (C++ type `A`) from its slice of `packed`, beginning
    // at offset(I): a 1<->1 value forwards its single torch value to from();
    // a 1<->N value hands from() its slice as a reference tuple. `packed`'s value
    // category is forwarded, so an rvalue pack moves each element out (each is
    // read once, at a distinct offset) and a torch value forwards with no copy.
    template <std::size_t I, class A, class Packed>
    static decltype(auto) rebuild(Packed &&packed) {
        using TA = TorchArgDef<A>;
        constexpr std::size_t off = offset(I);
        if constexpr (IsTuple<torch_arg_t<A>>) {
            return [&]<std::size_t... J>(std::index_sequence<J...>) -> decltype(auto) {
                return TA::from(
                    std::forward_as_tuple(std::get<off + J>(std::forward<Packed>(packed))...));
            }(std::make_index_sequence<std::tuple_size_v<torch_arg_t<A>>>{});
        } else {
            return TA::from(std::get<off>(std::forward<Packed>(packed)));
        }
    }

    // Assemble std::tuple<Args...> (owned) from the flat pack, moving each
    // rebuilt value in when `packed` is an rvalue. A single flat value
    // arrives bare (to_torch_args unwraps a 1-tuple), so normalize it first.
    template <class Packed>
    static std::tuple<Args...> to_tuple(Packed &&packed) {
        if constexpr (IsTuple<std::decay_t<Packed>>) {
            return [&]<std::size_t... I>(std::index_sequence<I...>) {
                return std::tuple<Args...>(rebuild<I, Args>(std::forward<Packed>(packed))...);
            }(std::index_sequence_for<Args...>{});
        } else {
            return to_tuple(std::forward_as_tuple(std::forward<Packed>(packed)));
        }
    }
};

template <auto Fn, class ArgsTuple, class FlatTuple>
struct ToTorchOpImpl;

// Synthesizes the dispatcher kernel for a struct-returning op `Fn`.
//   - Args... are the op's C++ parameter types.
//   - Flat... are the torch argument types: each argument's torch
//     tuple (as_tuple_t<torch_arg_t<Arg>>), concatenated (flat_torch_types_t).
// `call` is the kernel registered with m.impl. It receives the flat torch
// arguments, regroups them into one slice per op argument, rebuilds each C++
// argument through TorchArgDef::from(), invokes the op, and boxes the result
// through TorchArgDef::to (a result struct becomes its field tuple, a
// torch result passes through). An argument whose type is unchanged
// forwards through by reference, so the common case copies nothing.
template <auto Fn, class... Args, class... Flat>
struct ToTorchOpImpl<Fn, std::tuple<Args...>, std::tuple<Flat...>> {
    // Every flattened argument value must be a torch type (a 1<->N argument's
    // values and a tuple argument's elements included). A non-torch value means
    // a TorchArgDef failed to reduce it.
    static_assert((IsTorch<Flat> && ...),
                  "to_torch_op: an argument did not reduce to torch types; add "
                  "a TorchArgDef specialization (or marshal its non-torch members).");

    // The registered kernel. Its parameter list is the flat dispatcher pack; it
    // captures the arguments by reference into a tuple and hands them to invoke.
    static auto call(as_param_t<Flat>... flat) {
        return invoke(std::forward_as_tuple(flat...), std::index_sequence_for<Args...>{});
    }

  private:
    // Rebuild every op argument from its slice of `packed` (via ToNativeArgs),
    // call the op, and box the result through TorchArgDef::to (struct -> tuple,
    // a torch type -> itself). `Arg...` indexes the op arguments.
    template <class Packed, std::size_t... Arg>
    static auto invoke(Packed &&packed, std::index_sequence<Arg...>) {
        decltype(auto) result =
            Fn(ToNativeArgs<std::tuple<Args...>>::template rebuild<Arg, Args>(packed)...);
        return TorchArgDef<std::decay_t<decltype(result)>>::to(
            std::forward<decltype(result)>(result));
    }
};

// Recover the op's argument tuple from its pointer via fn_traits, then bind the
// kernel to those arguments and their flattened torch values.
template <auto Fn>
using ToTorchOp = ToTorchOpImpl<Fn, typename fn_traits<decltype(Fn)>::args,
                                flat_torch_types_t<typename fn_traits<decltype(Fn)>::args>>;

} // namespace detail

// std::optional<T> with a non-torch inner T crosses as one std::optional
// per torch value of T. A torch optional needs no specialization --
// it crosses unchanged via the identity primary template.

// 1<->1 inner: a single std::optional<D>.
template <class T>
    requires(!detail::IsTorch<std::optional<T>> && !detail::IsTuple<torch_arg_t<T>>)
struct TorchArgDef<std::optional<T>> {
    static std::optional<torch_arg_t<T>> to(std::optional<T> v) {
        if (v.has_value()) {
            return TorchArgDef<T>::to(std::move(*v));
        }
        return std::nullopt;
    }
    static std::optional<T> from(std::optional<torch_arg_t<T>> d) {
        if (d.has_value()) {
            return TorchArgDef<T>::from(std::move(*d));
        }
        return std::nullopt;
    }
};

// 1<->N inner: a std::tuple of N std::optionals, one per torch value --
// never a single std::optional of a tuple. All values are present or absent
// together; presence is read from the first.
template <class T>
    requires(!detail::IsTorch<std::optional<T>> && detail::IsTuple<torch_arg_t<T>>)
struct TorchArgDef<std::optional<T>> {
  private:
    using inner_t = torch_arg_t<T>; // std::tuple<D0, ..., DN-1>

    template <std::size_t... Is>
    static std::tuple<std::optional<std::tuple_element_t<Is, inner_t>>...>
    to_values(std::optional<T> v, std::index_sequence<Is...>) {
        if (!v.has_value()) {
            return {}; // every value nullopt
        }
        inner_t reduced = TorchArgDef<T>::to(std::move(*v));
        return {std::move(std::get<Is>(reduced))...};
    }

    template <class Values, std::size_t... Is>
    static std::optional<T> from_values(Values &&vals, std::index_sequence<Is...>) {
        if (!std::get<0>(vals).has_value()) {
            return std::nullopt;
        }
        return TorchArgDef<T>::from(
            std::make_tuple(std::move(*std::get<Is>(std::forward<Values>(vals)))...));
    }

  public:
    static auto to(std::optional<T> v) {
        return to_values(std::move(v), std::make_index_sequence<std::tuple_size_v<inner_t>>{});
    }
    template <class Values>
    static std::optional<T> from(Values &&vals) {
        return from_values(std::forward<Values>(vals),
                           std::make_index_sequence<std::tuple_size_v<inner_t>>{});
    }
};

// Adapts an op into the function the torch dispatcher's `m.impl` expects. The
// m.impl argument list is each op argument's torch type(s)
// (`torch_arg_t`) concatenated; the result is boxed by `TorchArgDef::to` (a
// result struct via its co-located TorchArgDef spec, a torch result
// as-is). A generic lambda has no single signature for `m.impl` to infer and
// could not be registered directly.
template <auto Fn>
inline constexpr auto to_torch_op = &detail::ToTorchOp<Fn>::call;

// Flatten native values into torch values: each value is reduced by
// detail::flatten_one (torch leaf -> 1-tuple; struct / 1<->N -> its tuple,
// recursing) and the per-value tuples are concatenated. A result struct's
// TorchArgDef::to forwards its fields here. The flattened result is unwrapped at
// the boundary to match an op's m.impl return arity: exactly one torch
// value comes back bare (the 1<->1 case), zero or two-plus as a tuple. The
// static_assert turns a value that failed to reduce to a torch type into
// a precise error.
template <class... Args>
auto to_torch_args(Args &&...args) {
    static_assert((detail::IsTorch<torch_arg_t<std::decay_t<Args>>> && ...),
                  "to_torch_args: a value did not reduce to a torch type; add a "
                  "TorchArgDef<T> specialization for it.");
    auto flat = std::tuple_cat(detail::flatten_one(std::forward<Args>(args))...);
    if constexpr (std::tuple_size_v<decltype(flat)> == 1) {
        return std::get<0>(std::move(flat));
    } else {
        return flat;
    }
}

namespace detail {

// Assemble the dispatcher's typed<> signature for an op whose torch result is
// `Ret` and whose flat torch argument pack is `FlatTuple`: `Ret(as_param_t<F>...)`
// -- the very signature `to_torch_op`'s kernel is registered with (the kernel is
// `auto call(as_param_t<Flat>... flat) -> torch_arg_t<Result>`).
template <class Ret, class FlatTuple>
struct typed_sig;

template <class Ret, class... Flat>
struct typed_sig<Ret, std::tuple<Flat...>> {
    using type = Ret(as_param_t<Flat>...);
};

template <class Ret, class FlatTuple>
using typed_sig_t = typename typed_sig<Ret, FlatTuple>::type;

// Shared core of both call_torch_op forms: marshal the native args to torch,
// invoke the already-typed dispatcher handle, and rebuild the native result
// `ResultT` via TorchArgDef::from. The two entry points differ only in how they
// obtain the handle (the FwdFn form caches it in a static, the signature form
// does not), so the marshalling itself lives here.
template <class ResultT, class TypedOp, class... Args>
auto invoke_torch_op(TypedOp &op, Args &&...args) {
    auto flat = to_torch_args(std::forward<Args>(args)...);
    auto ret = [&] {
        if constexpr (IsTuple<std::decay_t<decltype(flat)>>) {
            return std::apply(
                [&](auto &&...f) { return op.call(std::forward<decltype(f)>(f)...); },
                std::move(flat));
        } else {
            return op.call(std::move(flat));
        }
    }();
    return TorchArgDef<ResultT>::from(std::move(ret));
}

} // namespace detail

// Call a registered op through the dispatcher from native (C++) code -- the
// mirror image of `to_torch_op`. `FwdFn`'s signature is the single source of
// truth: its argument types flatten (via `TorchArgDef::to`, the same path
// `to_torch_args` takes) into the dispatcher's typed<> argument pack, and its
// declared return struct is reconstructed (via `TorchArgDef::from`) from the
// torch values the op returns. So the call site passes and receives native
// types -- a result struct comes back as a struct, not a flat tuple.
//
// Unlike a direct C++ call to the op's free function, this routes through the
// dispatcher, so the op's registered autograd fires and the call is recorded in
// the autograd graph.
template <auto FwdFn, class... Args>
auto call_torch_op(const char *qualified_name, Args &&...args) {
    using Traits = detail::fn_traits<decltype(FwdFn)>;
    using ResultT = std::decay_t<typename Traits::return_type>;
    using DispatchSig = detail::typed_sig_t<
        torch_arg_t<ResultT>, detail::flat_torch_types_t<typename Traits::args>>;

    // FwdFn is unique per op, so the resolved handle is cached in a static.
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow(qualified_name, "").typed<DispatchSig>();

    return detail::invoke_torch_op<ResultT>(op, std::forward<Args>(args)...);
}

// Signature-explicit form, for ops with no gsplat C++ forward function: the
// caller supplies the native signature `R(Args...)`. A signature is not unique
// to an op, so the handle is resolved each call rather than cached.
template <class Sig, class... Args>
auto call_torch_op(const char *qualified_name, Args &&...args) {
    using Traits = detail::fn_traits<std::add_pointer_t<Sig>>;
    using ResultT = std::decay_t<typename Traits::return_type>;
    using DispatchSig = detail::typed_sig_t<
        torch_arg_t<ResultT>, detail::flat_torch_types_t<typename Traits::args>>;

    auto op = c10::Dispatcher::singleton()
                  .findSchemaOrThrow(qualified_name, "").typed<DispatchSig>();

    return detail::invoke_torch_op<ResultT>(op, std::forward<Args>(args)...);
}

// ---------------------------------------------------------------------------
// Saved-state plumbing for C++ torch::autograd::Function subclasses.
//
// The backward free-function's signature is the single source of truth for
// the saved-state layout. It takes its saved state as its leading parameters
// and the engine-supplied grad bundle as one trailing
// `const torch::autograd::variable_list&`. `bwd_saved_t<&bwd>` reconstructs
// the saved-state tuple type (params before that trailing list); `ctx_save`
// and `ctx_load` route each saved value to/from the AutogradContext by type:
//   - at::Tensor                          -> save_for_backward variable_list
//   - at::optional<at::Tensor>            -> save_for_backward (boxed via
//                                            as_tensor / unboxed via
//                                            as_optional_tensor)
//   - c10::intrusive_ptr<X> (custom class)-> saved_data[index] IValue
//   - at::optional<intrusive_ptr<X>>      -> saved_data[index] IValue or None
//   - other non-tensor values (scalar / enum / uint32_t / float / ...)
//                                         -> crossed via TorchArgDef to a torch
//                                            scalar, stored as saved_data[index]
//                                            IValue
// ---------------------------------------------------------------------------

namespace detail {

// Tuple with its last element removed.
template <class Tuple>
struct drop_last;

template <class... Ts>
struct drop_last<std::tuple<Ts...>> {
    template <std::size_t... I>
    static auto pick(std::index_sequence<I...>)
        -> std::tuple<std::tuple_element_t<I, std::tuple<Ts...>>...>;

    using type = decltype(pick(std::make_index_sequence<sizeof...(Ts) - 1>{}));
};

template <class Tuple>
using drop_last_t = typename drop_last<Tuple>::type;

// The backward function's saved-state tuple: its parameter types before the
// trailing `const torch::autograd::variable_list&`.
template <auto BwdFn>
using bwd_saved_t = drop_last_t<typename fn_traits<decltype(BwdFn)>::decayed_args>;

// The backward function's trailing parameter: the engine grad bundle. It must
// be a variable_list — that is the boundary `drop_last` peels off to delimit
// the saved state (everything before it). ctx_save/ctx_load static_assert this.
template <auto BwdFn>
using bwd_grad_bundle_t =
    std::tuple_element_t<std::tuple_size_v<typename fn_traits<decltype(BwdFn)>::decayed_args> - 1,
                         typename fn_traits<decltype(BwdFn)>::decayed_args>;

} // namespace detail

// Number of logical forward inputs a torch::autograd::Function sees: real
// forward parameter count minus the leading AutogradContext*. Used by a
// `FwdInput::COUNT == fwd_input_count<&forward>()` assert to pin the position
// enum to the arity derived from the function type.
template <auto FwdFn>
inline constexpr std::size_t fwd_input_count() {
    using args = typename detail::fn_traits<decltype(FwdFn)>::decayed_args;
    static_assert(std::is_same_v<std::tuple_element_t<0, args>, torch::autograd::AutogradContext *>,
                  "forward's first parameter must be torch::autograd::AutogradContext*");
    return std::tuple_size_v<args> - 1;
}

namespace detail {

// A backward function's trailing parameter is its grad bundle: a struct that
// opts in with a `static constexpr bool is_grad_bundle = true`. ctx_save/
// ctx_load assert the boundary is such a struct, so a stray saved tensor-list
// in the last position can't masquerade as the bundle. This is a compile-time
// type check — no RTTI, no runtime cost.
template <class T>
concept GradBundle = requires { requires T::is_grad_bundle; };

// Save one value into the context at pack-index I.
template <std::size_t I, class V>
inline void save_one(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list &tensors,
                     const V &v) {
    if constexpr (std::is_same_v<V, at::Tensor>) {
        tensors.push_back(v);
    } else if constexpr (is_optional_tensor_v<V>) {
        tensors.push_back(as_tensor(v));
    } else if constexpr (is_optional_intrusive_v<V>) {
        // at::optional<c10::intrusive_ptr<X>> -> IValue(class) or None
        ctx->saved_data[std::to_string(I)] = v.has_value() ? c10::IValue(v.value()) : c10::IValue();
    } else if constexpr (is_intrusive_ptr<V>::value) {
        ctx->saved_data[std::to_string(I)] = c10::IValue(v);
    } else {
        static_assert(!is_tensor_list_v<V>,
                      "ctx_save: a tensor-list saved input (vector<Tensor> / List<Tensor> "
                      "/ TensorList) must route through save_for_backward, not saved_data; "
                      "extend the router before saving one.");
        // A non-tensor saved value (scalar, enum, ...) crosses as its torch
        // form via TorchArgDef, then stored as an IValue.
        ctx->saved_data[std::to_string(I)] = c10::IValue(TorchArgDef<V>::to(v));
    }
}

// Load one value of type V from the context at pack-index I. `tensors` is the
// saved variable_list and `t` is a running cursor into it (advanced only for
// tensor-ish slots, matching the push order in save_one).
//
// The non-tensor branches bind the saved entry to a named IValue before any
// `.to<>()` / `.toCustomClass<>()` call. nvcc's EDG frontend treats the
// `saved_data[std::to_string(I)]` subscript as dependent on the pack index I
// and refuses to parse a member-template call directly on it; a named,
// non-dependent IValue (or a `.template` disambiguator) is required.
template <std::size_t I, class V>
inline V load_one(torch::autograd::AutogradContext *ctx,
                  const torch::autograd::variable_list &tensors, std::size_t &t) {
    if constexpr (std::is_same_v<V, at::Tensor>) {
        return tensors[t++];
    } else if constexpr (is_optional_tensor_v<V>) {
        return as_optional_tensor(tensors[t++]);
    } else if constexpr (is_optional_intrusive_v<V>) {
        using X = value_type_t<is_intrusive_ptr<value_type_t<is_optional<V>>>>;
        const c10::IValue &iv = ctx->saved_data[std::to_string(I)];
        return iv.isNone() ? V{} : V(iv.toCustomClass<X>());
    } else if constexpr (is_intrusive_ptr<V>::value) {
        using X = value_type_t<is_intrusive_ptr<V>>;
        const c10::IValue &iv = ctx->saved_data[std::to_string(I)];
        return iv.toCustomClass<X>();
    } else {
        static_assert(!is_tensor_list_v<V>,
                      "ctx_load: a tensor-list saved input must round-trip via "
                      "save_for_backward, not saved_data.");
        const c10::IValue &iv = ctx->saved_data[std::to_string(I)];
        return TorchArgDef<V>::from(iv.to<torch_arg_t<V>>());
    }
}

template <auto BwdFn, class Tuple, std::size_t... I>
inline void ctx_save_impl(torch::autograd::AutogradContext *ctx, const Tuple &vals,
                          std::index_sequence<I...>) {
    torch::autograd::variable_list tensors;
    (save_one<I>(ctx, tensors, std::get<I>(vals)), ...);
    ctx->save_for_backward(std::move(tensors));
}

template <auto BwdFn, std::size_t... I>
inline bwd_saved_t<BwdFn> ctx_load_impl(torch::autograd::AutogradContext *ctx,
                                        std::index_sequence<I...>) {
    const torch::autograd::variable_list tensors = ctx->get_saved_variables();
    std::size_t t = 0;
    using Saved = bwd_saved_t<BwdFn>;
    // Brace-init enforces left-to-right evaluation so the tensor cursor `t`
    // advances in slot order.
    return Saved{load_one<I, std::tuple_element_t<I, Saved>>(ctx, tensors, t)...};
}

} // namespace detail

// Save the values that backward needs, type-checked against bwd's parameters.
// The values must be passed in bwd's parameter order; a wrong-typed or
// out-of-order (different type) save is a compile error.
template <auto BwdFn, class... Vals>
inline void ctx_save(torch::autograd::AutogradContext *ctx, const Vals &...vals) {
    static_assert(detail::GradBundle<detail::bwd_grad_bundle_t<BwdFn>>,
                  "ctx_save: backward's last parameter must be the grad bundle "
                  "(variable_list, or a struct tagged is_grad_bundle).");
    static_assert(std::is_same_v<std::tuple<std::decay_t<Vals>...>, detail::bwd_saved_t<BwdFn>>,
                  "ctx_save values must match the backward function's saved-state "
                  "parameters (types and order) exactly.");
    detail::ctx_save_impl<BwdFn>(ctx, std::forward_as_tuple(vals...),
                                 std::make_index_sequence<sizeof...(Vals)>{});
}

// Reconstruct backward's saved-state tuple for structured-binding / std::apply.
template <auto BwdFn>
inline detail::bwd_saved_t<BwdFn> ctx_load(torch::autograd::AutogradContext *ctx) {
    static_assert(detail::GradBundle<detail::bwd_grad_bundle_t<BwdFn>>,
                  "ctx_load: backward's last parameter must be the grad bundle "
                  "(variable_list, or a struct tagged is_grad_bundle).");
    constexpr std::size_t N = std::tuple_size_v<detail::bwd_saved_t<BwdFn>>;
    return detail::ctx_load_impl<BwdFn>(ctx, std::make_index_sequence<N>{});
}

// Restore bwd's saved state from `ctx` and invoke it with `bundle` (the grad
// outputs) appended as the trailing argument. The call is copy-free:
// ctx_load's tuple is moved into bwd's by-value params, while `bundle` stays
// a reference via forward_as_tuple. Both are consumed within this
// full-expression, so the reference can't dangle.
template <auto BwdFn, class Bundle>
inline auto apply_bwd(torch::autograd::AutogradContext *ctx, const Bundle &bundle) {
    return std::apply(BwdFn, std::tuple_cat(ctx_load<BwdFn>(ctx), std::forward_as_tuple(bundle)));
}

} // namespace gsplat
