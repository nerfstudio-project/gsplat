#include <torch/extension.h>

#include "Ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("null", &gsplat::null);

    m.def("projection_ewa_3dgs_fwd", &gsplat::projection_ewa_3dgs_fwd);
    m.def("projection_ewa_3dgs_bwd", &gsplat::projection_ewa_3dgs_bwd);

    m.def("projection_ewa_3dgs_fused_fwd", &gsplat::projection_ewa_3dgs_fused_fwd);
    m.def("projection_ewa_3dgs_fused_bwd", &gsplat::projection_ewa_3dgs_fused_bwd);

    m.def("projection_ewa_3dgs_packed_fwd", &gsplat::projection_ewa_3dgs_packed_fwd);
    m.def("projection_ewa_3dgs_packed_bwd", &gsplat::projection_ewa_3dgs_packed_bwd);

    m.def("spherical_harmonics_fwd", &gsplat::spherical_harmonics_fwd);
    m.def("spherical_harmonics_bwd", &gsplat::spherical_harmonics_bwd);

    m.def("adam", &gsplat::adam);

    m.def("intersect_tile", &gsplat::intersect_tile);
    m.def("intersect_offset", &gsplat::intersect_offset);
}