/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>

#include "Ops.h"
#include "Cameras.h"
#include "csrc/Config.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    // We define exported types as being "module_local"
    // so that they don't clash with other modules that export
    // types with the same name as ours.
    // Ref: https://pybind11.readthedocs.io/en/stable/advanced/classes.html#module-local-class-bindings

    py::enum_<gsplat::CameraModelType>(m, "CameraModelType", py::module_local())
        .value("PINHOLE", gsplat::CameraModelType::PINHOLE)
        .value("ORTHO", gsplat::CameraModelType::ORTHO)
        .value("FISHEYE", gsplat::CameraModelType::FISHEYE)
        .value("FTHETA", gsplat::CameraModelType::FTHETA)
        .export_values();

    m.def("null", &gsplat::null);

#if GSPLAT_BUILD_3DGS
    m.def(
        "quat_scale_to_covar_preci_fwd", &gsplat::quat_scale_to_covar_preci_fwd
    );
    m.def(
        "quat_scale_to_covar_preci_bwd", &gsplat::quat_scale_to_covar_preci_bwd
    );
#endif

    m.def("spherical_harmonics_fwd", &gsplat::spherical_harmonics_fwd);
    m.def("spherical_harmonics_bwd", &gsplat::spherical_harmonics_bwd);

#if GSPLAT_BUILD_ADAM
    m.def("adam", &gsplat::adam);
#endif

#if GSPLAT_BUILD_RELOC
    m.def("relocation", &gsplat::relocation);
#endif

    m.def("intersect_tile", &gsplat::intersect_tile);
    m.def("intersect_offset", &gsplat::intersect_offset);

#if GSPLAT_BUILD_3DGS
    m.def("projection_ewa_simple_fwd", &gsplat::projection_ewa_simple_fwd);
    m.def("projection_ewa_simple_bwd", &gsplat::projection_ewa_simple_bwd);
    m.def(
        "projection_ewa_3dgs_fused_fwd", &gsplat::projection_ewa_3dgs_fused_fwd
    );
    m.def(
        "projection_ewa_3dgs_fused_bwd", &gsplat::projection_ewa_3dgs_fused_bwd
    );
    m.def(
        "projection_ewa_3dgs_packed_fwd",
        &gsplat::projection_ewa_3dgs_packed_fwd
    );
    m.def(
        "projection_ewa_3dgs_packed_bwd",
        &gsplat::projection_ewa_3dgs_packed_bwd
    );

    m.def(
        "rasterize_to_pixels_3dgs_fwd", &gsplat::rasterize_to_pixels_3dgs_fwd
    );
    m.def(
        "rasterize_to_pixels_3dgs_bwd", &gsplat::rasterize_to_pixels_3dgs_bwd
    );
    m.def("rasterize_to_indices_3dgs", &gsplat::rasterize_to_indices_3dgs);
#endif

#if GSPLAT_BUILD_2DGS
    m.def("projection_2dgs_fused_fwd", &gsplat::projection_2dgs_fused_fwd);
    m.def("projection_2dgs_fused_bwd", &gsplat::projection_2dgs_fused_bwd);
    m.def("projection_2dgs_packed_fwd", &gsplat::projection_2dgs_packed_fwd);
    m.def("projection_2dgs_packed_bwd", &gsplat::projection_2dgs_packed_bwd);

    m.def(
        "rasterize_to_pixels_2dgs_fwd", &gsplat::rasterize_to_pixels_2dgs_fwd
    );
    m.def(
        "rasterize_to_pixels_2dgs_bwd", &gsplat::rasterize_to_pixels_2dgs_bwd
    );
    m.def("rasterize_to_indices_2dgs", &gsplat::rasterize_to_indices_2dgs);
#endif

#if GSPLAT_BUILD_3DGUT
    m.def("projection_ut_3dgs_fused", &gsplat::projection_ut_3dgs_fused);
    m.def("rasterize_to_pixels_from_world_3dgs_fwd", &gsplat::rasterize_to_pixels_from_world_3dgs_fwd);
    m.def("rasterize_to_pixels_from_world_3dgs_bwd", &gsplat::rasterize_to_pixels_from_world_3dgs_bwd);
#endif

    // Cameras from 3DGUT
    py::enum_<ShutterType>(m, "ShutterType", py::module_local())
        .value("ROLLING_TOP_TO_BOTTOM", ShutterType::ROLLING_TOP_TO_BOTTOM)
        .value("ROLLING_LEFT_TO_RIGHT", ShutterType::ROLLING_LEFT_TO_RIGHT)
        .value("ROLLING_BOTTOM_TO_TOP", ShutterType::ROLLING_BOTTOM_TO_TOP)
        .value("ROLLING_RIGHT_TO_LEFT", ShutterType::ROLLING_RIGHT_TO_LEFT)
        .value("GLOBAL", ShutterType::GLOBAL)
        .export_values();

    py::class_<UnscentedTransformParameters>(m, "UnscentedTransformParameters", py::module_local())
        .def(py::init<>())
        .def_readwrite("alpha", &UnscentedTransformParameters::alpha)
        .def_readwrite("beta", &UnscentedTransformParameters::beta)
        .def_readwrite("kappa", &UnscentedTransformParameters::kappa)
        .def_readwrite("in_image_margin_factor", &UnscentedTransformParameters::in_image_margin_factor)
        .def_readwrite("require_all_sigma_points_valid", &UnscentedTransformParameters::require_all_sigma_points_valid);

    // FTheta Camera support
    py::enum_<FThetaCameraDistortionParameters::PolynomialType>(m, "FThetaPolynomialType", py::module_local())
        .value("PIXELDIST_TO_ANGLE", FThetaCameraDistortionParameters::PolynomialType::PIXELDIST_TO_ANGLE)
        .value("ANGLE_TO_PIXELDIST", FThetaCameraDistortionParameters::PolynomialType::ANGLE_TO_PIXELDIST)
        .export_values();
    py::class_<FThetaCameraDistortionParameters>(m, "FThetaCameraDistortionParameters", py::module_local())
        .def(py::init<>())
        .def_readwrite("reference_poly", &FThetaCameraDistortionParameters::reference_poly)
        .def_readwrite("pixeldist_to_angle_poly", &FThetaCameraDistortionParameters::pixeldist_to_angle_poly)
        .def_readwrite("angle_to_pixeldist_poly", &FThetaCameraDistortionParameters::angle_to_pixeldist_poly)
        .def_readwrite("max_angle", &FThetaCameraDistortionParameters::max_angle)
        .def_readwrite("linear_cde", &FThetaCameraDistortionParameters::linear_cde);
}
