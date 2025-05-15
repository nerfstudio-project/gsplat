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