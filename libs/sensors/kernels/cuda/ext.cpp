/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// ext.cpp — TORCH_LIBRARY and PYBIND11_MODULE initialisation point.
//
// This is the sole translation unit that registers the gsplat_sensors torch
// library.  It has two top-level blocks:
//
//   TORCH_LIBRARY(gsplat_sensors, m)
//     Registers TorchScript custom classes (OpenCVPinholeProjection,
//     NoExternalDistortion, BivariateWindshieldDistortion) with their
//     constructors, readable properties, transform helpers
//     (OpenCVPinholeProjection only), and pickle hooks.
//     Then registers every C++ wrapper function from camera_torch.h as a
//     named op in the gsplat_sensors namespace so they are reachable from
//     Python as torch.ops.gsplat_sensors.<name>().
//
//   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
//     Exposes ShutterType as a pybind11 enum so it is importable directly
//     from the .so (re-exported as an IntEnum from cameras/types.py).

#include "csrc/camera_torch.h"
#include "csrc/external_distortion_torch.h"
#include "csrc/shutter_type.h"

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <array>
#include <tuple>

TORCH_LIBRARY(gsplat_sensors, m) {
    using gsplat_sensors::BivariateWindshieldDistortion;
    using gsplat_sensors::NoExternalDistortion;
    using gsplat_sensors::OpenCVPinholeProjection;

    m.class_<OpenCVPinholeProjection>("OpenCVPinholeProjection")
        .def(
            torch::init([](
                at::Tensor focal_length,
                at::Tensor principal_point,
                at::Tensor radial_coeffs,
                at::Tensor tangential_coeffs,
                at::Tensor thin_prism_coeffs,
                std::array<int64_t, 2> resolution) {
                auto ptr = c10::make_intrusive<OpenCVPinholeProjection>(
                    std::move(focal_length),
                    std::move(principal_point),
                    std::move(radial_coeffs),
                    std::move(tangential_coeffs),
                    std::move(thin_prism_coeffs),
                    resolution);
                gsplat_sensors::check_projection(ptr);
                return ptr;
            }),
            "Construct from per-component tensors",
            {torch::arg("focal_length"),
             torch::arg("principal_point"),
             torch::arg("radial_coeffs"),
             torch::arg("tangential_coeffs"),
             torch::arg("thin_prism_coeffs"),
             torch::arg("resolution")}
        )
        .def_readwrite("focal_length", &OpenCVPinholeProjection::focal_length)
        .def_readwrite("principal_point", &OpenCVPinholeProjection::principal_point)
        .def_readwrite("radial_coeffs", &OpenCVPinholeProjection::radial_coeffs)
        .def_readwrite("tangential_coeffs", &OpenCVPinholeProjection::tangential_coeffs)
        .def_readwrite("thin_prism_coeffs", &OpenCVPinholeProjection::thin_prism_coeffs)
        .def_property(
            "resolution",
            [](const c10::intrusive_ptr<OpenCVPinholeProjection>& self) {
                return std::make_tuple(self->resolution[0], self->resolution[1]);
            },
            [](const c10::intrusive_ptr<OpenCVPinholeProjection>& self, std::array<int64_t, 2> resolution) {
                self->resolution = resolution;
            }
        )
        .def(
            "transform",
            [](const c10::intrusive_ptr<OpenCVPinholeProjection>& self,
               std::tuple<double, double> scale,
               std::tuple<double, double> offset,
               std::tuple<int64_t, int64_t> new_resolution) {
                auto opts = self->focal_length.options();
                auto scale_t = at::tensor({std::get<0>(scale), std::get<1>(scale)}, at::kDouble).to(opts);
                auto offset_t = at::tensor({std::get<0>(offset), std::get<1>(offset)}, at::kDouble).to(opts);
                return c10::make_intrusive<OpenCVPinholeProjection>(
                    self->focal_length * scale_t,
                    self->principal_point * scale_t - offset_t,
                    self->radial_coeffs.clone(),
                    self->tangential_coeffs.clone(),
                    self->thin_prism_coeffs.clone(),
                    std::array<int64_t, 2>{std::get<0>(new_resolution), std::get<1>(new_resolution)});
            },
            "Transform image-domain intrinsics",
            {torch::arg("scale"), torch::arg("offset"), torch::arg("new_resolution")}
        )
        .def_pickle(
            [](const c10::intrusive_ptr<OpenCVPinholeProjection>& self) -> c10::IValue {
                return c10::IValue(c10::ivalue::Tuple::create({
                    c10::IValue(static_cast<int64_t>(1)),
                    c10::IValue(self->focal_length),
                    c10::IValue(self->principal_point),
                    c10::IValue(self->radial_coeffs),
                    c10::IValue(self->tangential_coeffs),
                    c10::IValue(self->thin_prism_coeffs),
                    c10::IValue(self->resolution[0]),
                    c10::IValue(self->resolution[1]),
                }));
            },
            [](c10::IValue state) -> c10::intrusive_ptr<OpenCVPinholeProjection> {
                auto elems = state.toTuple()->elements();
                TORCH_CHECK(elems.size() == 8 && elems[0].toInt() == 1, "unsupported OpenCVPinholeProjection pickle state");
                return c10::make_intrusive<OpenCVPinholeProjection>(
                    elems[1].toTensor(),
                    elems[2].toTensor(),
                    elems[3].toTensor(),
                    elems[4].toTensor(),
                    elems[5].toTensor(),
                    std::array<int64_t, 2>{elems[6].toInt(), elems[7].toInt()});
            }
        );

    m.class_<NoExternalDistortion>("NoExternalDistortion")
        .def(torch::init([]() { return c10::make_intrusive<NoExternalDistortion>(); }))
        .def_pickle(
            [](const c10::intrusive_ptr<NoExternalDistortion>&) -> c10::IValue {
                return c10::IValue(c10::ivalue::Tuple::create({c10::IValue(static_cast<int64_t>(1))}));
            },
            [](c10::IValue state) -> c10::intrusive_ptr<NoExternalDistortion> {
                auto elems = state.toTuple()->elements();
                TORCH_CHECK(elems.size() == 1 && elems[0].toInt() == 1, "unsupported NoExternalDistortion pickle state");
                return c10::make_intrusive<NoExternalDistortion>();
            }
        );

    m.class_<BivariateWindshieldDistortion>("BivariateWindshieldDistortion")
        .def(
            torch::init([](
                at::Tensor distortion_coeffs,
                int64_t reference_polynomial,
                int64_t h_poly_degree,
                int64_t v_poly_degree) {
                auto ptr = c10::make_intrusive<BivariateWindshieldDistortion>(
                    std::move(distortion_coeffs),
                    reference_polynomial,
                    h_poly_degree,
                    v_poly_degree);
                gsplat_sensors::check_bivariate_windshield_distortion(ptr);
                return ptr;
            }),
            "Construct from packed windshield distortion coefficients",
            {torch::arg("distortion_coeffs"),
             torch::arg("reference_polynomial"),
             torch::arg("h_poly_degree"),
             torch::arg("v_poly_degree")}
        )
        .def_readonly("distortion_coeffs", &BivariateWindshieldDistortion::distortion_coeffs)
        .def_readonly("reference_polynomial", &BivariateWindshieldDistortion::reference_polynomial)
        .def_readonly("h_poly_degree", &BivariateWindshieldDistortion::h_poly_degree)
        .def_readonly("v_poly_degree", &BivariateWindshieldDistortion::v_poly_degree)
        .def_pickle(
            [](const c10::intrusive_ptr<BivariateWindshieldDistortion>& self) -> c10::IValue {
                return c10::IValue(c10::ivalue::Tuple::create({
                    c10::IValue(static_cast<int64_t>(1)),
                    c10::IValue(self->distortion_coeffs),
                    c10::IValue(self->reference_polynomial),
                    c10::IValue(self->h_poly_degree),
                    c10::IValue(self->v_poly_degree),
                }));
            },
            [](c10::IValue state) -> c10::intrusive_ptr<BivariateWindshieldDistortion> {
                auto elems = state.toTuple()->elements();
                TORCH_CHECK(
                    elems.size() == 5 && elems[0].isInt() && elems[0].toInt() <= 1,
                    "unsupported BivariateWindshieldDistortion pickle state");
                auto ptr = c10::make_intrusive<BivariateWindshieldDistortion>(
                    elems[1].toTensor(),
                    elems[2].toInt(),
                    elems[3].toInt(),
                    elems[4].toInt());
                gsplat_sensors::check_bivariate_windshield_distortion(ptr);
                return ptr;
            }
        );

    m.def("generate_image_points", &gsplat_sensors::generate_image_points);
    m.def(
        "camera_rays_to_image_points_opencv_pinhole_no_external",
        &gsplat_sensors::camera_rays_to_image_points_opencv_pinhole_no_external);
    m.def(
        "image_points_to_camera_rays_opencv_pinhole_no_external",
        &gsplat_sensors::image_points_to_camera_rays_opencv_pinhole_no_external);
    m.def(
        "project_world_points_mean_pose_opencv_pinhole_no_external",
        &gsplat_sensors::project_world_points_mean_pose_opencv_pinhole_no_external);
    m.def(
        "project_world_points_shutter_pose_opencv_pinhole_no_external",
        &gsplat_sensors::project_world_points_shutter_pose_opencv_pinhole_no_external);
    m.def(
        "image_points_to_world_rays_static_pose_opencv_pinhole_no_external",
        &gsplat_sensors::image_points_to_world_rays_static_pose_opencv_pinhole_no_external);
    m.def(
        "image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external",
        &gsplat_sensors::image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external);
    m.def(
        "camera_rays_to_image_points_opencv_pinhole_bivariate_windshield",
        &gsplat_sensors::camera_rays_to_image_points_opencv_pinhole_bivariate_windshield);
    m.def(
        "image_points_to_camera_rays_opencv_pinhole_bivariate_windshield",
        &gsplat_sensors::image_points_to_camera_rays_opencv_pinhole_bivariate_windshield);
    m.def(
        "project_world_points_mean_pose_opencv_pinhole_bivariate_windshield",
        &gsplat_sensors::project_world_points_mean_pose_opencv_pinhole_bivariate_windshield);
    m.def(
        "project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield",
        &gsplat_sensors::project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield);
    m.def(
        "image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield",
        &gsplat_sensors::image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield);
    m.def(
        "image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield",
        &gsplat_sensors::image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield);
    m.def(
        "camera_rays_to_image_points_opencv_pinhole_no_external_backward",
        &gsplat_sensors::camera_rays_to_image_points_opencv_pinhole_no_external_backward);
    m.def(
        "image_points_to_camera_rays_opencv_pinhole_no_external_backward",
        &gsplat_sensors::image_points_to_camera_rays_opencv_pinhole_no_external_backward);
    m.def(
        "project_world_points_mean_pose_opencv_pinhole_no_external_backward",
        &gsplat_sensors::project_world_points_mean_pose_opencv_pinhole_no_external_backward);
    m.def(
        "project_world_points_shutter_pose_opencv_pinhole_no_external_backward",
        &gsplat_sensors::project_world_points_shutter_pose_opencv_pinhole_no_external_backward);
    m.def(
        "image_points_to_world_rays_static_pose_opencv_pinhole_no_external_backward",
        &gsplat_sensors::image_points_to_world_rays_static_pose_opencv_pinhole_no_external_backward);
    m.def(
        "image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external_backward",
        &gsplat_sensors::image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external_backward);
    m.def(
        "camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_backward",
        &gsplat_sensors::camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_backward);
    m.def(
        "image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_backward",
        &gsplat_sensors::image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_backward);
    m.def(
        "project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_backward",
        &gsplat_sensors::project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_backward);
    m.def(
        "project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_backward",
        &gsplat_sensors::project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_backward);
    m.def(
        "image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_backward",
        &gsplat_sensors::image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_backward);
    m.def(
        "image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_backward",
        &gsplat_sensors::image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_backward);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using gsplat_sensors::ShutterType;
    namespace py = pybind11;

    py::enum_<ShutterType>(
        m,
        "ShutterType",
        "Source of truth for OpenCV pinhole shutter modes. Bound from C++ "
        "(see csrc/shutter_type.h) and re-exported as an IntEnum from "
        "libs/sensors/kernels/cameras/types.py.")
        .value("ROLLING_TOP_TO_BOTTOM", ShutterType::ROLLING_TOP_TO_BOTTOM)
        .value("ROLLING_LEFT_TO_RIGHT", ShutterType::ROLLING_LEFT_TO_RIGHT)
        .value("ROLLING_BOTTOM_TO_TOP", ShutterType::ROLLING_BOTTOM_TO_TOP)
        .value("ROLLING_RIGHT_TO_LEFT", ShutterType::ROLLING_RIGHT_TO_LEFT)
        .value("GLOBAL", ShutterType::GLOBAL);
}
