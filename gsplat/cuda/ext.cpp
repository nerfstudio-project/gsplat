#include <torch/extension.h>

#include "Ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("null", &gsplat::null);

    m.def("projection_3dgs_fwd", &gsplat::projection_3dgs_fwd);
    m.def("projection_3dgs_bwd", &gsplat::projection_3dgs_bwd);

    m.def("projection_3dgs_fused_fwd", &gsplat::projection_3dgs_fused_fwd);
    m.def("projection_3dgs_fused_bwd", &gsplat::projection_3dgs_fused_bwd);

}