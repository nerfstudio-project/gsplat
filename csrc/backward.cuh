#include <cuda.h>
#include <cuda_runtime.h>

#include "helpers.cuh"

// for f : R(n) -> R(m), J in R(m, n),
// v is cotangent in R(m), e.g. dL/df in R(m),
// compute vjp i.e. vT J -> R(n)
void project_gaussians_backward_impl(
);


// compute jacobians of output image wrt binned and sorted gaussians
void rasterize_backward_impl(
);



