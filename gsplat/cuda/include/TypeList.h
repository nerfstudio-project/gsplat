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

// Compile-time type list and generic metafunctions.
//
// This header provides generic metaprogramming utilities for working with
// compile-time type lists:
//   - Represent lists of types                        (TypeList)
//   - Convert a type list to a std::variant           (TypeListToVariant)
//   - Concatenate type lists                          (TypeListCat)
//   - Apply a template to each type in a list         (TypeListApply)
//   - Compute cartesian products of templates × args  (CartesianProduct)
//
// The gsplat namespace at the bottom of this file adds domain-specific
// utilities that depend on gsplat conventions (e.g. KernelParameters).

#pragma once

#include <type_traits>
#include <variant>

// ============================================================================
// Generic type list utilities
// ============================================================================

// A compile-time list of types.
template <typename... Ts>
struct TypeList {};

// Convert a TypeList<A, B, C> to std::variant<A, B, C>.
template <typename TL>
struct TypeListToVariantImpl;

template <typename... Ts>
struct TypeListToVariantImpl<TypeList<Ts...>> {
    using type = std::variant<Ts...>;
};

template <typename TL>
using TypeListToVariant = typename TypeListToVariantImpl<TL>::type;

// Concatenate two TypeLists.
template <typename TL1, typename TL2>
struct TypeListCatImpl;

template <typename... Ts, typename... Us>
struct TypeListCatImpl<TypeList<Ts...>, TypeList<Us...>> {
    using type = TypeList<Ts..., Us...>;
};

template <typename TL1, typename TL2>
using TypeListCat = typename TypeListCatImpl<TL1, TL2>::type;

// Apply a template to each type in a TypeList, producing a new TypeList.
// TypeListApply<TypeList<A, B>, Tmpl> -> TypeList<Tmpl<A>, Tmpl<B>>
template <typename TL, template <typename> class Tmpl>
struct TypeListApplyImpl;

template <typename... Ts, template <typename> class Tmpl>
struct TypeListApplyImpl<TypeList<Ts...>, Tmpl> {
    using type = TypeList<Tmpl<Ts>...>;
};

template <typename TL, template <typename> class Tmpl>
using TypeListApply = typename TypeListApplyImpl<TL, Tmpl>::type;

// Cartesian product of template wrappers and argument types.
//
// Given a list of template wrappers and a list of argument types, produces
// a TypeList containing every wrapper instantiated with every argument:
//
//   CartesianProduct<
//       TypeList<Wrapper<Tmpl1>, Wrapper<Tmpl2>>,
//       TypeList<Arg1, Arg2>
//   > = TypeList<Tmpl1<Arg1>, Tmpl1<Arg2>, Tmpl2<Arg1>, Tmpl2<Arg2>>
//
// Each wrapper must expose: template <typename A> using Apply = TmplN<A>;

// Helper: instantiate one template wrapper with every argument type.
template <typename ArgTypes, template <typename...> class Template>
struct InstantiateOneWrapper;

template <typename... Args, template <typename...> class Template>
struct InstantiateOneWrapper<TypeList<Args...>, Template> {
    using type = TypeList<Template<Args>...>;
};

// CartesianProduct takes a TypeList of template wrappers and a TypeList of
// argument types.
// Each wrapper must expose a nested `template <typename A> using Apply = ...;`
template <typename Wrappers, typename ArgTypes>
struct CartesianProductImpl;

template <typename ArgTypes>
struct CartesianProductImpl<TypeList<>, ArgTypes> {
    using type = TypeList<>;
};

template <typename First, typename... Rest, typename ArgTypes>
struct CartesianProductImpl<TypeList<First, Rest...>, ArgTypes> {
    // Instantiate First with all argument types
    using head = typename InstantiateOneWrapper<ArgTypes, First::template Apply>::type;
    using tail = typename CartesianProductImpl<TypeList<Rest...>, ArgTypes>::type;
    using type = TypeListCat<head, tail>;
};

template <typename Wrappers, typename ArgTypes>
using CartesianProduct = typename CartesianProductImpl<Wrappers, ArgTypes>::type;

// ============================================================================
// gsplat-specific type list utilities
// ============================================================================
//
// These utilities depend on gsplat conventions: each sensor model type must
// have a nested KernelParameters type that holds the runtime parameters for
// CUDA kernel dispatch.
//
// See Cameras.cuh, ExternalDistortion.cuh, Lidars.cuh, and Sensors.cuh for
// the concrete type lists built with these utilities.

namespace gsplat {

// Map TypeList<A, B, C> to std::variant<A::KernelParameters, B::KernelParameters, ...>.
// Each type in the list must have a nested KernelParameters type.
template <typename TL>
struct TypeListToKernelParamsVariantImpl;

template <typename... Ts>
struct TypeListToKernelParamsVariantImpl<TypeList<Ts...>> {
    using type = std::variant<typename Ts::KernelParameters...>;
};

template <typename TL>
using TypeListToKernelParamsVariant = typename TypeListToKernelParamsVariantImpl<TL>::type;

// Find the type T in a TypeList such that T::KernelParameters is KP.
// Produces a compile error if no match is found.
// Used to reverse-map from a KernelParameters variant alternative back to the
// sensor model type, enabling host-side variant dispatch to recover the full
// model type for kernel template instantiation.
template <typename KP, typename TL>
struct FindByKernelParamsImpl;

template <typename KP>
struct FindByKernelParamsImpl<KP, TypeList<>> {
    static_assert(sizeof(KP) == 0, "KernelParameters type not found in TypeList");
};

template <typename KP, typename Head, typename... Tail>
struct FindByKernelParamsImpl<KP, TypeList<Head, Tail...>>
    : std::conditional_t<
          std::is_same_v<KP, typename Head::KernelParameters>,
          std::type_identity<Head>,
          FindByKernelParamsImpl<KP, TypeList<Tail...>>
      > {};

template <typename KP, typename TL>
using FindByKernelParams = typename FindByKernelParamsImpl<KP, TL>::type;

} // namespace gsplat
