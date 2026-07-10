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

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>

#include "DeformLosses.h"

namespace gsplat
{
// Host-side validation helpers for the deform-smoothness loss family. The
// grouped gaussian dispatch entry (GaussianLosses.cpp) is the loss's only
// entry point in this MR; the helpers live here, next to the kernels they
// guard, so a later standalone deform op can reuse them unchanged.

// CHECK_INPUT verifies CUDA/privateuseone + contiguity but not cross-tensor
// device index or dtype. The kernels dispatch on deformation.scalar_type()
// and read every pointer as scalar_t on deformation's device, so every
// other tensor argument must share deformation's device and dtype.
// Used by the grouped gaussian dispatch entry (GaussianLosses.cpp), declared
// in DeformLosses.h.
void deform_losses_check_matches_deformation(const at::Tensor &deformation, const at::Tensor &x, const char *name)
{
    TORCH_CHECK(
        x.device() == deformation.device(),
        name,
        " must be on the same device as deformation (",
        deformation.device(),
        "), got ",
        x.device()
    );
    TORCH_CHECK(
        x.scalar_type() == deformation.scalar_type(),
        name,
        " must have the same dtype as deformation (",
        deformation.scalar_type(),
        "), got ",
        x.scalar_type()
    );
}

// Right-align the mask shape against deformation's [N, D] and derive the
// virtual element strides (0 on broadcast dimensions) the kernels use to
// map a deformation element to its mask source element. One-way broadcast
// only: the mask may never force deformation to expand (that would change
// the numerator element count).
// Used by the grouped gaussian dispatch entry (GaussianLosses.cpp), declared
// in DeformLosses.h.
void deform_losses_resolve_mask_broadcast(
    const at::Tensor &deformation, const at::Tensor &mask, int64_t &stride_row, int64_t &stride_col
)
{
    const int64_t N = deformation.size(0);
    const int64_t D = deformation.size(1);
    TORCH_CHECK(mask.dim() <= 2, "mask must have at most 2 dims to broadcast against [N, D], got ", mask.sizes());
    const int64_t m_rows = mask.dim() == 2 ? mask.size(0) : 1;
    const int64_t m_cols = mask.dim() == 0 ? 1 : mask.size(-1);
    TORCH_CHECK(
        (m_rows == 1 || m_rows == N) && (m_cols == 1 || m_cols == D),
        "mask (",
        mask.sizes(),
        ") must broadcast to deformation's shape [",
        N,
        ", ",
        D,
        "] without expanding deformation"
    );
    stride_row = (m_rows == 1) ? 0 : m_cols;
    stride_col = (m_cols == 1) ? 0 : 1;
}
} // namespace gsplat
