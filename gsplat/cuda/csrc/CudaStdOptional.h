/*
 * SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Minimal cuda::std::optional shim for HIP/ROCm.
//
// <cuda/std/optional> via ROCm's libcu++ reroute pulls in NVIDIA-only
// fp8 types (__nv_fp8_e4m3 / __nv_fp8_e5m2) that don't compile under
// hipcc. Falling back to std::optional fails too: value() throws
// std::bad_optional_access, which hipcc rejects in __host__ __device__
// instantiations even when guarded by has_value() at the call site.
//
// Under USE_ROCM, provide the API gsplat uses (default/value ctor,
// has_value, value, operator*/->/bool) — all noexcept.
// Under CUDA, <cuda/std/optional> is included as before.

#ifdef USE_ROCM
#include <utility>

namespace cuda { namespace std {

struct nullopt_t { explicit constexpr nullopt_t(int) {} };
inline constexpr nullopt_t nullopt{0};

template <typename T>
class optional {
public:
    constexpr optional() noexcept : has_value_(false) {}
    constexpr optional(nullopt_t) noexcept : has_value_(false) {}

    __host__ __device__ optional(const T& v) : has_value_(true) {
        new (&storage_) T(v);
    }
    __host__ __device__ optional(T&& v) : has_value_(true) {
        new (&storage_) T(::std::move(v));
    }

    __host__ __device__ optional(const optional& o) : has_value_(o.has_value_) {
        if (has_value_) new (&storage_) T(*o.ptr());
    }
    __host__ __device__ optional(optional&& o) noexcept : has_value_(o.has_value_) {
        if (has_value_) new (&storage_) T(::std::move(*o.ptr()));
    }

    __host__ __device__ ~optional() { if (has_value_) ptr()->~T(); }

    __host__ __device__ optional& operator=(nullopt_t) noexcept {
        reset();
        return *this;
    }
    __host__ __device__ optional& operator=(const optional& o) {
        if (this == &o) return *this;
        reset();
        if (o.has_value_) {
            new (&storage_) T(*o.ptr());
            has_value_ = true;
        }
        return *this;
    }
    __host__ __device__ optional& operator=(optional&& o) noexcept {
        if (this == &o) return *this;
        reset();
        if (o.has_value_) {
            new (&storage_) T(::std::move(*o.ptr()));
            has_value_ = true;
        }
        return *this;
    }
    template <typename U = T>
    __host__ __device__ optional& operator=(U&& v) {
        reset();
        new (&storage_) T(::std::forward<U>(v));
        has_value_ = true;
        return *this;
    }

    __host__ __device__ constexpr bool has_value() const noexcept { return has_value_; }
    __host__ __device__ constexpr explicit operator bool() const noexcept { return has_value_; }

    // Non-throwing value() — caller is responsible for checking has_value().
    // Matches the existing gsplat call pattern `opt.has_value() ? &opt.value() : nullptr`.
    __host__ __device__ T& value() & noexcept { return *ptr(); }
    __host__ __device__ const T& value() const & noexcept { return *ptr(); }
    __host__ __device__ T&& value() && noexcept { return ::std::move(*ptr()); }

    __host__ __device__ T& operator*() & noexcept { return *ptr(); }
    __host__ __device__ const T& operator*() const & noexcept { return *ptr(); }
    __host__ __device__ T* operator->() noexcept { return ptr(); }
    __host__ __device__ const T* operator->() const noexcept { return ptr(); }

    __host__ __device__ void reset() noexcept {
        if (has_value_) { ptr()->~T(); has_value_ = false; }
    }

private:
    __host__ __device__ T* ptr() noexcept { return reinterpret_cast<T*>(&storage_); }
    __host__ __device__ const T* ptr() const noexcept { return reinterpret_cast<const T*>(&storage_); }

    alignas(T) unsigned char storage_[sizeof(T)];
    bool has_value_;
};

} } // namespace cuda::std
#else
#include <cuda/std/optional>
#endif
