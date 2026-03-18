/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

/**
 * @file TensorView.h
 * @brief Type-safe strided tensor view with compile-time shape validation
 *
 * TensorView provides zero-overhead strided access with:
 * - Compile-time shape specification (integer literals or typed dimension tags)
 * - Partial indexing returning sub-views
 * - Automatic batch dimension flattening
 * - Negative index support and bounds checking
 *
 * Example:
 * @code
 *   TensorView<float, CAMERA, RAY, 3> rays;
 *   auto ray = rays(cam_idx, ray_idx);    // Returns TensorView<float, 3>
 *   float x = ray(0);                     // Returns float&
 * @endcode
 */

#pragma once
#include <torch/extension.h>
#include <array>
#include <concepts>
#include <string_view>

namespace gsplat {

/**
 * @brief Type-safe multi-dimensional tensor view
 *
 * @tparam T Element type
 * @tparam SHAPE Variadic shape (typed tags like CAMERA or integer literals)
 */
template<typename T, auto... SHAPE>
class TensorView
{
public:
    static constexpr int ndims = sizeof...(SHAPE);
    static_assert(ndims > 0, "TensorView must have at least one dimension");

    /**
     * @brief Default constructor (creates null view)
     */
    __host__ __device__
    TensorView() : m_data(nullptr), m_sizes{}, m_strides{} {}

    /**
     * @brief Implicit cast constructor
     */
    template <typename U>
        requires std::convertible_to<U*,T*>
    __host__ __device__
    TensorView(const TensorView <U, SHAPE...> &that)
        : m_data(that.m_data)
        , m_sizes{that.m_sizes}
        , m_strides{that.m_strides}
    {
    }

    /**
     * @brief Implicit assignment operator
     */
    template <typename U>
        requires std::convertible_to<U*,T*>
    __host__ __device__
    TensorView &operator=(const TensorView <U, SHAPE...> &that)
    {
        if(this != &that)
        {
            m_data = that.m_data;
            m_sizes = that.m_sizes;
            m_strides = that.m_strides;
        }
        return *this;
    }

    /**
     * @brief Create TensorView from raw pointer with dimension validation
     *
     * @param data Data pointer
     * @param sizes Size array
     * @param strides Stride array
     * @return Validated TensorView
     */
    __host__ __device__
    TensorView(T* data,
               const std::array<int64_t, ndims>& sizes,
               const std::array<int64_t, ndims>& strides)
        : m_data(data)
        , m_sizes(sizes)
        , m_strides(strides)
    {
        // Get expected sizes from SHAPE pack (integral values or -1 for dynamic dimensions)
        constexpr auto expected = std::array{
            ([]() constexpr {
                using DimType = decltype(SHAPE);
                if constexpr (std::is_integral_v<DimType>)
                    return static_cast<int64_t>(SHAPE);
                else
                    // Dynamic size (e.g., CAMERA tag)
                    return static_cast<int64_t>(-1);
            }())...
        };

        for (int i = 0; i < ndims; ++i)
        {
            if (expected[i] != -1)
            {
#ifdef __CUDA_ARCH__
                assert(sizes[i] == expected[i]);
#else
                TORCH_CHECK_INDEX(sizes[i] == expected[i],
                    "TensorView dimension ", i, " size mismatch: expected ", expected[i], ", got ", sizes[i]);
#endif
            }
        }
    }

    /**
     * @brief Multi-dimensional indexing operator
     *
     * Full indexing (sizeof...(Indices) == ndims): returns T&
     * Partial indexing (sizeof...(Indices) < ndims): returns sub-TensorView
     *
     * @tparam Indices Integral index types
     * @param indices Index values (supports negative indexing)
     * @return Element reference or sub-view
     */
    template<std::integral... Indices>
    __host__ __device__
    decltype(auto) operator()(Indices... indices) const
    {
        // Compute offset with negative index support
        int64_t offset = [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            int64_t idx_array[] = {(static_cast<int64_t>(indices) < 0
                                    ? static_cast<int64_t>(indices) + m_sizes[Is]
                                    : static_cast<int64_t>(indices))...};

#ifndef NDEBUG
            // Bounds checking (this doesn't compile if NDEBUG is defined...)
            ((assert(idx_array[Is] >= 0 && idx_array[Is] < m_sizes[Is])), ...);
#endif

            return ((idx_array[Is] * m_strides[Is]) + ...);
        }(std::make_index_sequence<sizeof...(Indices)>{});

        if constexpr (sizeof...(Indices) == ndims)
        {
            // Full indexing - return element reference
            return m_data[offset];
        }
        else
        {
            static_assert(sizeof...(Indices) < ndims, "Too many indices");

            // Partial indexing - return sub-view with remaining dimensions
            constexpr int K = sizeof...(Indices);

            return [&]<std::size_t... Is>(std::index_sequence<Is...>)
            {
                // Extract tail of SHAPE using tuple manipulation
                using ShapeTuple = std::tuple<std::integral_constant<decltype(SHAPE), SHAPE>...>;

                return TensorView<T, std::tuple_element_t<K + Is, ShapeTuple>::value...>(
                    m_data + offset,
                    {m_sizes[K + Is]...},
                    {m_strides[K + Is]...}
                );
            }(std::make_index_sequence<ndims-K>{});
        }
    }

    /**
     * @brief Get size of dimension (supports negative indexing)
     * @param dim Dimension index (negative counts from end)
     */
    __host__ __device__
    int64_t shape(int dim) const
    {
        return m_sizes[dim < 0 ? dim + ndims : dim];
    }

    /**
     * @brief Get stride of dimension (supports negative indexing)
     * @param dim Dimension index (negative counts from end)
     */
    __host__ __device__
    int64_t stride(int dim) const
    {
        return m_strides[dim < 0 ? dim + ndims : dim];
    }

    /**
     * @brief Check if view is valid (non-null data pointer)
     */
    __host__ __device__
    explicit operator bool() const
    {
        return m_data != nullptr;
    }

    __host__ __device__
    T *data() const
    {
        return m_data;
    }

private:
    T* m_data;
    const std::array<int64_t, ndims> m_sizes;
    const std::array<int64_t, ndims> m_strides;

};

/**
 * @brief Create TensorView from raw pointer with dimension validation
 *
 * @param data Data pointer
 * @param sizes Size array
 * @param strides Stride array
 * @return Validated TensorView
 */
template<auto... SHAPE, typename T>
TensorView<T, SHAPE...> make_tensor_view(T* data,
                                         const std::array<int64_t, sizeof...(SHAPE)>& sizes,
                                         const std::array<int64_t, sizeof...(SHAPE)>& strides,
                                         std::string_view tensor_name = "Tensor")
{
    // Validate dimensions
    // Get expected sizes from SHAPE pack (integral values or -1 for dynamic dimensions)
    constexpr auto expected = std::array{
        ([]() constexpr {
            using DimType = decltype(SHAPE);
            if constexpr (std::is_integral_v<DimType>)
                return static_cast<int>(SHAPE);
            else
                return -1;  // Dynamic size (e.g., CAMERA tag)
        }())...
    };

    for (int i = expected.size()-1; i>=0; --i)
    {
        if (expected[i] != -1)
        {
            TORCH_CHECK_INDEX(sizes[i] == expected[i],
                tensor_name, " (shape: ", torch::IntArrayRef(sizes), "): dimension ", i, " size mismatch: expected ", expected[i], ", got ", sizes[i]);
        }
    }

    return TensorView<T, SHAPE...>(data, sizes, strides);
}

} // namespace gsplat

