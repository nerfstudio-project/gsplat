#include <torch/extension.h>
#include "rasterize.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_forward", &rasterize_forward_tensor);
}
