#include "bindings.h"
#include "cameras.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::enum_<gsplat::CameraModelType>(m, "CameraModelType")
        .value("PINHOLE", gsplat::CameraModelType::PINHOLE)
        .value("ORTHO", gsplat::CameraModelType::ORTHO)
        .value("FISHEYE", gsplat::CameraModelType::FISHEYE)
        .export_values();

    m.def("compute_sh_fwd", &gsplat::compute_sh_fwd_tensor);
    m.def("compute_sh_bwd", &gsplat::compute_sh_bwd_tensor);

    m.def(
        "quat_scale_to_covar_preci_fwd",
        &gsplat::quat_scale_to_covar_preci_fwd_tensor
    );
    m.def(
        "quat_scale_to_covar_preci_bwd",
        &gsplat::quat_scale_to_covar_preci_bwd_tensor
    );

    m.def("proj_fwd", &gsplat::proj_fwd_tensor);
    m.def("proj_bwd", &gsplat::proj_bwd_tensor);

    m.def(
        "fully_fused_projection_fwd", &gsplat::fully_fused_projection_fwd_tensor
    );
    m.def(
        "fully_fused_projection_bwd", &gsplat::fully_fused_projection_bwd_tensor
    );

    m.def("isect_tiles", &gsplat::isect_tiles_tensor);
    m.def("isect_offset_encode", &gsplat::isect_offset_encode_tensor);

    m.def("rasterize_to_pixels_fwd", &gsplat::rasterize_to_pixels_fwd_tensor);
    m.def("rasterize_to_pixels_bwd", &gsplat::rasterize_to_pixels_bwd_tensor);

    m.def(
        "rasterize_to_indices_in_range",
        &gsplat::rasterize_to_indices_in_range_tensor
    );

    // packed version
    m.def(
        "fully_fused_projection_packed_fwd",
        &gsplat::fully_fused_projection_packed_fwd_tensor
    );
    m.def(
        "fully_fused_projection_packed_bwd",
        &gsplat::fully_fused_projection_packed_bwd_tensor
    );

    m.def("compute_relocation", &gsplat::compute_relocation_tensor);

    // 2DGS
    m.def(
        "fully_fused_projection_fwd_2dgs",
        &gsplat::fully_fused_projection_fwd_2dgs_tensor
    );
    m.def(
        "fully_fused_projection_bwd_2dgs",
        &gsplat::fully_fused_projection_bwd_2dgs_tensor
    );

    m.def(
        "fully_fused_projection_packed_fwd_2dgs",
        &gsplat::fully_fused_projection_packed_fwd_2dgs_tensor
    );
    m.def(
        "fully_fused_projection_packed_bwd_2dgs",
        &gsplat::fully_fused_projection_packed_bwd_2dgs_tensor
    );

    m.def(
        "rasterize_to_pixels_fwd_2dgs",
        &gsplat::rasterize_to_pixels_fwd_2dgs_tensor
    );
    m.def(
        "rasterize_to_pixels_bwd_2dgs",
        &gsplat::rasterize_to_pixels_bwd_2dgs_tensor
    );

    m.def(
        "rasterize_to_indices_in_range_2dgs",
        &gsplat::rasterize_to_indices_in_range_2dgs_tensor
    );

    m.def("selective_adam_update", &gsplat::selective_adam_update);

    m.def("fully_fused_projection_3dgut_fwd", &gsplat::fully_fused_projection_3dgut_fwd_tensor);

    // Cameras from 3DGUT
    py::enum_<ShutterType>(m, "ShutterType")
        .value("ROLLING_TOP_TO_BOTTOM", ShutterType::ROLLING_TOP_TO_BOTTOM)
        .value("ROLLING_LEFT_TO_RIGHT", ShutterType::ROLLING_LEFT_TO_RIGHT)
        .value("ROLLING_BOTTOM_TO_TOP", ShutterType::ROLLING_BOTTOM_TO_TOP)
        .value("ROLLING_RIGHT_TO_LEFT", ShutterType::ROLLING_RIGHT_TO_LEFT)
        .value("GLOBAL", ShutterType::GLOBAL);

    py::class_<CameraModelParameters>(m, "CameraModelParameters")
        .def(py::init<>())
        .def_readwrite("resolution", &CameraModelParameters::resolution)
        .def_readwrite("shutter_type", &CameraModelParameters::shutter_type);

    py::class_<OpenCVPinholeCameraModelParameters, CameraModelParameters>(m, "OpenCVPinholeCameraModelParameters")
        .def(py::init<>())
        .def_readwrite("principal_point", &OpenCVPinholeCameraModelParameters::principal_point)
        .def_readwrite("focal_length", &OpenCVPinholeCameraModelParameters::focal_length)
        .def_readwrite("radial_coeffs", &OpenCVPinholeCameraModelParameters::radial_coeffs)
        .def_readwrite("tangential_coeffs", &OpenCVPinholeCameraModelParameters::tangential_coeffs)
        .def_readwrite("thin_prism_coeffs", &OpenCVPinholeCameraModelParameters::thin_prism_coeffs);

    py::class_<OpenCVFisheyeCameraModelParameters, CameraModelParameters>(m, "OpenCVFisheyeCameraModelParameters")
        .def(py::init<>())
        .def_readwrite("principal_point", &OpenCVFisheyeCameraModelParameters::principal_point)
        .def_readwrite("focal_length", &OpenCVFisheyeCameraModelParameters::focal_length)
        .def_readwrite("radial_coeffs", &OpenCVFisheyeCameraModelParameters::radial_coeffs)
        .def_readwrite("max_angle", &OpenCVFisheyeCameraModelParameters::max_angle);

    py::class_<FThetaCameraModelParameters, CameraModelParameters>(m, "FThetaCameraModelParameters")
        .def(py::init<>())
        .def_readwrite("principal_point", &FThetaCameraModelParameters::principal_point)
        .def_readwrite("reference_poly", &FThetaCameraModelParameters::reference_poly)
        .def_readwrite("pixeldist_to_angle_poly", &FThetaCameraModelParameters::pixeldist_to_angle_poly)
        .def_readwrite("angle_to_pixeldist_poly", &FThetaCameraModelParameters::angle_to_pixeldist_poly)
        .def_readwrite("max_angle", &FThetaCameraModelParameters::max_angle);

    py::enum_<FThetaCameraModelParameters::PolynomialType>(m, "PolynomialType")
        .value("PIXELDIST_TO_ANGLE", FThetaCameraModelParameters::PolynomialType::PIXELDIST_TO_ANGLE)
        .value("ANGLE_TO_PIXELDIST", FThetaCameraModelParameters::PolynomialType::ANGLE_TO_PIXELDIST);

    py::class_<RollingShutterParameters>(m, "RollingShutterParameters")
        .def(py::init<>())
        .def_readwrite("T_world_sensors", &RollingShutterParameters::T_world_sensors)
        .def_readwrite("timestamps_us", &RollingShutterParameters::timestamps_us);

    py::class_<CameraNRE>(m, "CameraNRE")
        .def(py::init<>())
        .def_readwrite("camera_model_parameters", &CameraNRE::camera_model_parameters)
        .def_readwrite("rolling_shutter_parameters", &CameraNRE::rolling_shutter_parameters);

    py::class_<CameraGSplat>(m, "CameraGSplat")
        .def(py::init<>())
        .def_readwrite("resolution", &CameraGSplat::resolution)
        .def_readwrite("cam_pos", &CameraGSplat::_position)
        .def_readwrite("viewmatrix", &CameraGSplat::_viewmatrix)
        .def_readwrite("projmatrix", &CameraGSplat::_projmatrix)
        .def_readwrite("inv_viewprojmatrix", &CameraGSplat::_inv_viewprojmatrix)
        .def_readwrite("tan_fovx", &CameraGSplat::tan_fovx)
        .def_readwrite("tan_fovy", &CameraGSplat::tan_fovy);
}
