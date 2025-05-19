#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD

// TODO: replacing the following with per-operation kernels might make compile
// faster.
// https://github.com/pytorch/pytorch/blob/740ce0fa5f8c7e9e51422b614f8187ab93a60b8b/aten/src/ATen/native/cuda/ScanKernels.cpp#L8-L17
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h" // where all the macros are defined
#include "Null.h"   // where the launch function is declared
#include "Ops.h"    // a collection of all gsplat operators

namespace gsplat {

at::Tensor null(const at::Tensor input) {
    DEVICE_GUARD(input);
    CHECK_INPUT(input);
    at::Tensor output = at::empty_like(input);
    launch_null_kernel(input, output);
    return output;
}

} // namespace gsplat
