/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/core/List.h> // c10::List<at::Tensor> for is_tensor_list
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>

#include <type_traits>
#include <vector>

#include <c10/util/intrusive_ptr.h>

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

} // namespace gsplat
