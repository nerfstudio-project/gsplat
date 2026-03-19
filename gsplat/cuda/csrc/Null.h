/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#include <cstdint>

// Forward declaration: Postpone `#include <ATen/core/Tensor.h>` to .cu file,
// in case .cu file does not need it.
namespace at {
class Tensor;
}

namespace gsplat {

// This .h file only declares function for launching CUDA kernels. It should be
// included by both a .cpp and .cu file, to server as the bridge between them.
//
// The .cu file will implement the actual CUDA kernel with this launch function.
//
// The .cpp file will define the complete operator, in which this launch
// function will be called.
void launch_null_kernel(const at::Tensor input, at::Tensor output);

} // namespace gsplat