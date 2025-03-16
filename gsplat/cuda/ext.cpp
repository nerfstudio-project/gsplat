#include <torch/extension.h>

#include "Ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    py::enum_<gsplat::CameraModelType>(m, "CameraModelType")
        .value("PINHOLE", gsplat::CameraModelType::PINHOLE)
        .value("ORTHO", gsplat::CameraModelType::ORTHO)
        .value("FISHEYE", gsplat::CameraModelType::FISHEYE)
        .export_values();

    m.def("null", &gsplat::null);

    m.def("quat_scale_to_covar_preci_fwd", &gsplat::quat_scale_to_covar_preci_fwd);
    m.def("quat_scale_to_covar_preci_bwd", &gsplat::quat_scale_to_covar_preci_bwd);
    
    m.def("spherical_harmonics_fwd", &gsplat::spherical_harmonics_fwd);
    m.def("spherical_harmonics_bwd", &gsplat::spherical_harmonics_bwd);

    m.def("adam", &gsplat::adam);
    m.def("relocation", &gsplat::relocation);

    m.def("intersect_tile", &gsplat::intersect_tile);
    m.def("intersect_offset", &gsplat::intersect_offset);

    m.def("projection_ewa_simple_fwd", &gsplat::projection_ewa_simple_fwd);
    m.def("projection_ewa_simple_bwd", &gsplat::projection_ewa_simple_bwd);
    m.def("projection_ewa_3dgs_fused_fwd", &gsplat::projection_ewa_3dgs_fused_fwd);
    m.def("projection_ewa_3dgs_fused_bwd", &gsplat::projection_ewa_3dgs_fused_bwd);
    m.def("projection_ewa_3dgs_packed_fwd", &gsplat::projection_ewa_3dgs_packed_fwd);
    m.def("projection_ewa_3dgs_packed_bwd", &gsplat::projection_ewa_3dgs_packed_bwd);

    m.def("rasterize_to_pixels_3dgs_fwd", &gsplat::rasterize_to_pixels_3dgs_fwd);
    m.def("rasterize_to_pixels_3dgs_bwd", &gsplat::rasterize_to_pixels_3dgs_bwd);
    m.def("rasterize_to_indices_3dgs", &gsplat::rasterize_to_indices_3dgs);

    m.def("projection_2dgs_fused_fwd", &gsplat::projection_2dgs_fused_fwd);
    m.def("projection_2dgs_fused_bwd", &gsplat::projection_2dgs_fused_bwd);
    m.def("projection_2dgs_packed_fwd", &gsplat::projection_2dgs_packed_fwd);
    m.def("projection_2dgs_packed_bwd", &gsplat::projection_2dgs_packed_bwd);

    m.def("rasterize_to_pixels_2dgs_fwd", &gsplat::rasterize_to_pixels_2dgs_fwd);
    m.def("rasterize_to_pixels_2dgs_bwd", &gsplat::rasterize_to_pixels_2dgs_bwd);
    m.def("rasterize_to_indices_2dgs", &gsplat::rasterize_to_indices_2dgs);
}