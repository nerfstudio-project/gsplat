#include "bindings.h"
#include "rasterize.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_forward", &rasterize_forward_tensor);
    m.def("compute_cov2d_bounds_forward", &compute_cov2d_bounds_forward_tensor);
}
