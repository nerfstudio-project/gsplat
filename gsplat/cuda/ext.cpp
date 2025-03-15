#include <torch/extension.h>

#include "Ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("null", &gsplat::null);

}