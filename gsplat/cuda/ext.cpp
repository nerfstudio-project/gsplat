#include <torch/extension.h>

#include "Ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("null", &gsplat::null);

    m.def("projection_ewa_3d_fwd", &gsplat::projection_ewa_3d_fwd);
    m.def("projection_ewa_3d_bwd", &gsplat::projection_ewa_3d_bwd);

    m.def("projection_ewa_3d_fused_fwd", &gsplat::projection_ewa_3d_fused_fwd);
    m.def("projection_ewa_3d_fused_bwd", &gsplat::projection_ewa_3d_fused_bwd);

    m.def("projection_ewa_3d_packed_fwd", &gsplat::projection_ewa_3d_packed_fwd);
    m.def("projection_ewa_3d_packed_bwd", &gsplat::projection_ewa_3d_packed_bwd);

}