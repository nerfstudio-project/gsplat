/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/core/Tensor.h>

namespace gsplat {

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

} // namespace gsplat
