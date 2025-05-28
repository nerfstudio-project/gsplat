#include <torch/extension.h>

#include "Ops.h"
#include "Cameras.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    py::enum_<gsplat::CameraModelType>(m, "CameraModelType")
        .value("PINHOLE", gsplat::CameraModelType::PINHOLE)
        .value("ORTHO", gsplat::CameraModelType::ORTHO)
        .value("FISHEYE", gsplat::CameraModelType::FISHEYE)
        .export_values();

    m.def("spherical_harmonics_fwd", &gsplat::spherical_harmonics_fwd);
    m.def("spherical_harmonics_bwd", &gsplat::spherical_harmonics_bwd);

    m.def("intersect_tile", &gsplat::intersect_tile);
    m.def("intersect_offset", &gsplat::intersect_offset);

    m.def(
        "projection_ewa_3dgs_fused_fwd", &gsplat::projection_ewa_3dgs_fused_fwd
    );
    m.def(
        "projection_ewa_3dgs_fused_bwd", &gsplat::projection_ewa_3dgs_fused_bwd
    );

    m.def(
        "rasterize_to_pixels_3dgs_fwd", &gsplat::rasterize_to_pixels_3dgs_fwd
    );
    m.def(
        "rasterize_to_pixels_3dgs_bwd", &gsplat::rasterize_to_pixels_3dgs_bwd
    );

}