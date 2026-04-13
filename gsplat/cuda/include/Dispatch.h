/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 #pragma once

#include <tuple>
#include <type_traits>
#include <variant>

// =============================================================================
// Runtime-to-Compile-Time Dispatch Library
//
// Problem: A function like project<BlockSize, Camera, Distortion, BatchSize>()
// is templated on values and types that must be known at compile time. But in
// many applications, these values are only known at runtime (e.g. from config
// files, user input, or polymorphic objects stored in std::variant).
//
// Solution: This library bridges that gap. You describe each parameter as either
// an IntParam (a runtime int that must match one of several allowed compile-time
// values) or a TypeParam/MappedTypeParam (a runtime std::variant that must match
// one of several allowed types). The library then generates the full combinatorial
// set of template instantiations and dispatches to the right one at runtime.
//
// Usage example:
//
//   using namespace dispatch;
//   bool ok = dispatch(
//       IntParam<1, 2, 4>{block_size},             // runtime int -> compile-time int
//       MappedTypeParam<MyMap, A, B>{variant_ab},   // runtime variant -> compile-time type
//       [&]<typename BlockSizeT, typename Resolved>() {
//           constexpr int BS = BlockSizeT::value;
//           // ... use BS and Resolved as compile-time constants ...
//       }
//   );
//
// Returns true if a matching instantiation was found and the callable was
// invoked, false if a runtime integer value did not match any allowed
// compile-time value. (TypeParam/MappedTypeParam always match because
// std::variant can only hold one of its listed alternatives.)
//
// The callable (lambda) is always the last argument. It receives one template
// parameter per dispatched dimension, in order. Integer dimensions become
// std::integral_constant<int, V>, and type dimensions become the resolved type.
// =============================================================================


namespace dispatch {

// =============================================================================
// Parameter Spec Types
//
// These are the types the user constructs at the call site to describe each
// dispatch dimension. They pair a runtime value with the set of allowed
// compile-time alternatives.
// =============================================================================

// IntParam: wraps a runtime int and lists the allowed compile-time values.
//
// Example: dispatch::IntParam<1, 2, 4>{block_size}
//   - block_size is the runtime value (e.g. 2)
//   - <1, 2, 4> are the allowed compile-time values
//   - At dispatch time, if block_size == 2, the callable receives
//     std::integral_constant<int, 2> as its corresponding template argument.
//   - If block_size doesn't match any allowed value, dispatch returns false
//     and the callable is not invoked.
template <int... Values>
struct IntParam {
    int value;

    // True when v matches one of the allowed compile-time values.
    static constexpr bool contains(int v) { return ((v == Values) || ...); }
};

// TypeParam: wraps a std::variant and resolves to the active alternative's own
// type (identity mapping). Use this when the variant alternative types are
// exactly the types you want as template parameters.
//
// Example: dispatch::TypeParam<float, double>{my_variant}
//   - my_variant is std::variant<float, double>
//   - If the variant currently holds a double, the callable receives `double`
//     as its corresponding template argument.
//
// The Resolve alias template defines how a variant alternative type maps to the
// type that gets passed to the callable. For TypeParam, Resolve<T> = T (identity).
//
// TypeParam dispatch always succeeds because std::variant can only hold one of
// its listed alternative types — there is no "invalid" case.
template <typename... Types>
struct TypeParam {
    std::variant<Types...> value;

    template <typename T>
    using Resolve = T;
};

// MappedTypeParam: like TypeParam, but applies a user-defined type mapping.
// Use this when the variant holds "core" types but your template function
// expects different "trait" or "container" types.
//
// The first template parameter (Map) is a template that maps each variant
// alternative type to the desired resolved type:  Map<T>::type
//
// Example:
//   // User defines the mapping (e.g. in main.cpp):
//   template <typename K> struct KernelToTrait;
//   template <> struct KernelToTrait<FisheyeKernel> { using type = Fisheye; };
//
//   dispatch::MappedTypeParam<KernelToTrait, FisheyeKernel, PinholeKernel>{camera_variant}
//   // If the variant holds a FisheyeKernel, the callable receives `Fisheye`.
//
// Like TypeParam, MappedTypeParam dispatch always succeeds.
template <template<typename> class Map, typename... Types>
struct MappedTypeParam {
    std::variant<Types...> value;

    template <typename T>
    using Resolve = typename Map<T>::type;
};



namespace detail {

// =============================================================================
// Type Detection Traits
//
// These traits let the dispatch machinery detect which kind of parameter spec
// it is looking at (IntParam vs TypeParam/MappedTypeParam vs the callable).
//
// They work via partial specialization: the general template inherits from
// std::false_type, but specialized versions for specific patterns (like
// IntParam<Vs...>) inherit from std::true_type. This lets us write
// `if constexpr (is_int_param<T>::value)` to branch at compile time.
// =============================================================================

// is_int_param<T>: true when T is IntParam<...>.
template <typename T>
struct is_int_param : std::false_type {};
template <int... Vs>
struct is_int_param<IntParam<Vs...>> : std::true_type {};

// is_type_param<T>: true when T is TypeParam<...> or MappedTypeParam<...>.
// Both are handled by the same dispatch_type_param function because they share
// the same Resolve<T> interface.
template <typename T>
struct is_type_param : std::false_type {};
template <typename... Ts>
struct is_type_param<TypeParam<Ts...>> : std::true_type {};
template <template<typename> class Map, typename... Ts>
struct is_type_param<MappedTypeParam<Map, Ts...>> : std::true_type {};


// =============================================================================
// Resolved Type List
//
// As the dispatch recurses through each parameter, it accumulates the resolved
// compile-time types into a Resolved<...> type list. This is a "type-level list"
// — it carries no data, only type information via its template parameters.
//
// For example, after resolving IntParam<1,2,4>{2} and TypeParam<A,B>{variant_a},
// the list would be Resolved<std::integral_constant<int,2>, A>.
//
// When all parameters are resolved, this list is unpacked and its types are
// passed as template arguments to the user's callable.
// =============================================================================
template <typename... Ts>
struct Resolved {};

// Append<List, T>: appends a type T to a Resolved<...> type list.
//
// The general template is intentionally left undefined — only the partial
// specialization below is valid. This ensures Append can only be used with
// Resolved<...> as its first argument.
//
// Example: Append<Resolved<int, float>, double>::type  ==  Resolved<int, float, double>
//
// This uses partial specialization to "unpack" the types inside Resolved<Rs...>:
// the compiler matches Resolved<Rs...> against the first argument, binding Rs...
// to whatever types are already in the list, then constructs a new Resolved with
// T appended.
template <typename List, typename T>
struct Append;

template <typename... Rs, typename T>
struct Append<Resolved<Rs...>, T> {
    using type = Resolved<Rs..., T>;
};


// =============================================================================
// Core Dispatch Recursion
//
// The dispatch works by processing parameters one at a time from left to right.
// Each step:
//   1. Looks at the first parameter in the remaining list
//   2. Determines its kind (IntParam, TypeParam/MappedTypeParam, or callable)
//   3. Resolves the runtime value to a compile-time type
//   4. Appends that type to the Resolved list
//   5. Recurses with the remaining parameters
//
// When no parameters remain (only the callable is left), it invokes the callable
// with the full Resolved type list as template arguments.
//
// All functions in the chain return bool: true if dispatch succeeded and the
// callable was invoked, false if an IntParam's runtime value didn't match any
// of its allowed compile-time values. The return value propagates back through
// the entire recursion so the caller can check whether dispatch succeeded.
// =============================================================================

// Forward declarations — needed because dispatch_impl, dispatch_int_param, and
// dispatch_type_param are mutually recursive (each may call dispatch_impl to
// continue processing the next parameter).
template <typename ResolvedList, typename First, typename... Rest>
bool dispatch_impl(First&& first, Rest&&... rest);

template <typename ResolvedList, int... Values, typename... Remaining>
bool dispatch_int_param(IntParam<Values...>& param, Remaining&&... remaining);

template <typename ResolvedList, typename ParamT, typename... Remaining>
bool dispatch_type_param(ParamT& param, Remaining&&... remaining);

// Base case: all parameters have been resolved. Invoke the callable.
// The Resolved<ResolvedTypes...> parameter is an empty object whose template
// arguments carry the resolved types. We unpack them and pass them as explicit
// template arguments to the callable's operator().
// Returns true because reaching this point means all parameters matched
// successfully and the callable was invoked.
template <typename... ResolvedTypes, typename Callable>
bool dispatch_invoke(Resolved<ResolvedTypes...>, Callable&& callable) {
    // This calls: callable.operator()<ResolvedTypes...>()
    // For example: callable.operator()<integral_constant<int,2>, PerfectPinhole, ...>()
    callable.template operator()<ResolvedTypes...>();
    return true;
}

// Main recursive entry point. Examines the first parameter and delegates to the
// appropriate handler based on its type.
//
// Template parameters:
//   ResolvedList — Resolved<Types...> accumulated so far (starts as Resolved<>)
//   First        — the next parameter to process (IntParam, TypeParam, or callable)
//   Rest...      — remaining parameters after First
//
// The `if constexpr` branches are evaluated at compile time, so only the matching
// branch is compiled for each instantiation. This avoids any runtime overhead
// from the type checking.
//
// Returns true if dispatch succeeded (callable invoked), false if an IntParam
// value didn't match any of its allowed compile-time values.
template <typename ResolvedList, typename First, typename... Rest>
bool dispatch_impl(First&& first, Rest&&... rest) {
    // std::decay_t strips references and const/volatile qualifiers so we can
    // match the underlying type against our detection traits.
    using FirstDecayed = std::decay_t<First>;
    if constexpr (is_int_param<FirstDecayed>::value) {
        return dispatch_int_param<ResolvedList>(first, std::forward<Rest>(rest)...);
    } else if constexpr (is_type_param<FirstDecayed>::value) {
        return dispatch_type_param<ResolvedList>(first, std::forward<Rest>(rest)...);
    } else {
        // Not an IntParam or TypeParam — must be the callable (last argument).
        static_assert(sizeof...(Rest) == 0, "Callable must be the last argument");
        return dispatch_invoke(ResolvedList{}, std::forward<First>(first));
    }
}


// =============================================================================
// Integer Parameter Dispatch
//
// Resolves a runtime int to a compile-time std::integral_constant<int, V> by
// trying each allowed value in sequence. This is a recursive if-chain that
// peels off one allowed value at a time from the template parameter list.
//
// Example: for IntParam<1, 2, 4>{block_size} with block_size == 2:
//   1. dispatch_int_helper<Resolved<>, 1, 2, 4> — checks 1, no match
//   2. dispatch_int_helper<Resolved<>, 2, 4>    — checks 2, match!
//   3. Appends integral_constant<int, 2> to Resolved and recurses to next param
//
// If no value matches, returns false (the callable is not invoked).
// =============================================================================

// dispatch_int_helper: recursive template that tries one value at a time.
//
// Template parameters:
//   ResolvedList — the type list accumulated so far
//   First        — the compile-time int value to test against the runtime value
//   Rest...      — remaining compile-time int values to try if First doesn't match
//   IntParamT    — the IntParam<...> type (deduced from the function argument)
//   Remaining... — the rest of the dispatch parameters after this IntParam
//
// Returns true if this value (or a subsequent one) matched and the callable
// was eventually invoked, false if no allowed value matched the runtime value.
template <typename ResolvedList, int First, int... Rest, typename IntParamT, typename... Remaining>
bool dispatch_int_helper(IntParamT& param, Remaining&&... remaining) {
    if (param.value == First) {
        // Match found. Append std::integral_constant<int, First> to the Resolved list.
        using NewResolved = typename Append<ResolvedList, std::integral_constant<int, First>>::type;
        if constexpr (sizeof...(Remaining) == 0) {
            static_assert(sizeof(IntParamT) == 0, "No callable provided");
        } else {
            // Continue dispatching the next parameter with the extended Resolved list.
            return dispatch_impl<NewResolved>(std::forward<Remaining>(remaining)...);
        }
    } else if constexpr (sizeof...(Rest) > 0) {
        // No match yet — try the next allowed value. This peels off the first
        // value from Rest... by recursing with <ResolvedList, Rest...>.
        return dispatch_int_helper<ResolvedList, Rest...>(param, std::forward<Remaining>(remaining)...);
    } else {
        // No more values to try — runtime value doesn't match any allowed value.
        // Return false to signal that dispatch failed. The callable is not invoked.
        return false;
    }
}

// dispatch_int_param: entry point that extracts the allowed values from the
// IntParam type and forwards them to dispatch_int_helper.
//
// This function exists because dispatch_impl can't directly deduce the int...
// values from IntParam — it only sees the IntParam as a single type. This
// intermediary deduces IntParam<Values...> and passes Values... explicitly.
//
// Returns true if the runtime value matched and dispatch succeeded, false otherwise.
template <typename ResolvedList, int... Values, typename... Remaining>
bool dispatch_int_param(IntParam<Values...>& param, Remaining&&... remaining) {
    return dispatch_int_helper<ResolvedList, Values...>(param, std::forward<Remaining>(remaining)...);
}


// =============================================================================
// Type Parameter Dispatch
//
// Resolves a runtime std::variant to a compile-time type using std::visit.
// std::visit calls our lambda with the actual alternative held by the variant,
// giving us access to the concrete type at compile time (via `auto&`).
//
// The resolved type is determined by the parameter's Resolve<T> alias:
//   - For TypeParam:       Resolve<T> = T              (identity)
//   - For MappedTypeParam: Resolve<T> = Map<T>::type   (custom mapping)
//
// This single function handles both TypeParam and MappedTypeParam because they
// share the same Resolve interface.
//
// Type dispatch itself always resolves successfully (a variant always holds
// exactly one of its alternatives), but a subsequent IntParam in the chain
// might still fail, so the return value is propagated from the recursive call.
// =============================================================================
template <typename ResolvedList, typename ParamT, typename... Remaining>
bool dispatch_type_param(ParamT& param, Remaining&&... remaining) {
    // std::visit calls our lambda with a reference to whichever type the variant
    // currently holds. The `auto&` parameter means the compiler generates one
    // instantiation of this lambda per variant alternative, each with the
    // concrete type known at compile time.
    //
    // The `-> bool` trailing return type is needed because each lambda
    // instantiation calls a different dispatch_impl specialization — without
    // it the compiler cannot deduce a common return type across alternatives.
    return std::visit([&](auto& alternative) -> bool {
        // Determine the concrete type of the active variant alternative.
        using AltType = std::decay_t<decltype(alternative)>;

        // Apply the Resolve mapping. For TypeParam this is identity (AltType itself).
        // For MappedTypeParam this applies the user's Map template.
        using ResolvedType = typename std::decay_t<ParamT>::template Resolve<AltType>;

        // Append ResolvedType to the Resolved list.
        using NewResolved = typename Append<ResolvedList, ResolvedType>::type;

        // Continue dispatching the next parameter.
        return dispatch_impl<NewResolved>(std::forward<Remaining>(remaining)...);
    }, param.value);
}

} // namespace detail


// =============================================================================
// Public API
// =============================================================================

// dispatch(param1, param2, ..., paramN, callable)
//
// Resolves each parameter from runtime to compile-time, then invokes the callable
// with the resolved types as template arguments.
//
// Parameters must be IntParam, TypeParam, or MappedTypeParam instances, except
// for the last argument which must be the callable (typically a generic lambda).
//
// Returns true if all parameters were resolved successfully and the callable was
// invoked. Returns false if an IntParam's runtime value did not match any of its
// allowed compile-time values — in this case the callable is NOT invoked.
//
// Starts the recursion with an empty Resolved<> type list.
template <typename... Args>
bool dispatch(Args&&... args) {
    return detail::dispatch_impl<detail::Resolved<>>(std::forward<Args>(args)...);
}

} // namespace dispatch
