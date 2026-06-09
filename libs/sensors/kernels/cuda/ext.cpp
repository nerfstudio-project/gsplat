/*
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

// ext.cpp — TORCH_LIBRARY and PYBIND11_MODULE initialisation point.
//
// This is the sole translation unit that registers the gsplat_sensors torch
// library.

#include "csrc/camera_torch.h"
#include "csrc/external_distortion_torch.h"
#include "csrc/lidar_torch.h"
#include "csrc/shutter_type.h"

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <array>
#include <tuple>

TORCH_LIBRARY(gsplat_sensors, m) {
    using gsplat_sensors::BivariateWindshieldDistortion;
    using gsplat_sensors::FThetaProjection;
    using gsplat_sensors::NoExternalDistortion;
    using gsplat_sensors::OpenCVPinholeProjection;
    using gsplat_sensors::RowOffsetStructuredSpinningLidarProjection;

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

    m.class_<FThetaProjection>("FThetaProjection")
        .def(
            torch::init([](
                at::Tensor principal_point,
                at::Tensor fw_poly,
                at::Tensor bw_poly,
                at::Tensor A,
                std::array<int64_t, 2> resolution,
                int64_t reference_polynomial,
                int64_t fw_poly_degree,
                int64_t bw_poly_degree,
                int64_t newton_iterations,
                double max_angle,
                double min_2d_norm) {
                auto ptr = c10::make_intrusive<FThetaProjection>(
                    std::move(principal_point),
                    std::move(fw_poly),
                    std::move(bw_poly),
                    std::move(A),
                    resolution,
                    reference_polynomial,
                    fw_poly_degree,
                    bw_poly_degree,
                    newton_iterations,
                    max_angle,
                    min_2d_norm);
                gsplat_sensors::check_ftheta_projection(ptr);
                return ptr;
            }),
            "Construct from per-component tensors and scalar config",
            {torch::arg("principal_point"),
             torch::arg("fw_poly"),
             torch::arg("bw_poly"),
             torch::arg("A"),
             torch::arg("resolution"),
             torch::arg("reference_polynomial"),
             torch::arg("fw_poly_degree"),
             torch::arg("bw_poly_degree"),
             torch::arg("newton_iterations"),
             torch::arg("max_angle"),
             torch::arg("min_2d_norm")}
        )
        .def_readonly("principal_point", &FThetaProjection::principal_point)
        .def_readonly("fw_poly", &FThetaProjection::fw_poly)
        .def_readonly("bw_poly", &FThetaProjection::bw_poly)
        .def_readonly("A", &FThetaProjection::A)
        // Ainv is exposed as a read-only property; torch::class_<T> uses
        // def_property with a getter only (no def_property_readonly overload).
        .def_property(
            "Ainv",
            [](const c10::intrusive_ptr<FThetaProjection>& self) -> at::Tensor {
                return self->compute_ainv();
            }
        )
        .def_readonly("reference_polynomial", &FThetaProjection::reference_polynomial)
        .def_readonly("fw_poly_degree", &FThetaProjection::fw_poly_degree)
        .def_readonly("bw_poly_degree", &FThetaProjection::bw_poly_degree)
        .def_readonly("newton_iterations", &FThetaProjection::newton_iterations)
        .def_readonly("max_angle", &FThetaProjection::max_angle)
        .def_readonly("min_2d_norm", &FThetaProjection::min_2d_norm)
        // torch::Library is designed for operators, not standalone constants.
        // Expose the compile-time term cap via a static getter so Python can
        // query the authoritative C++ value.
        .def_static(
            "get_max_polynomial_terms",
            []() -> int64_t {
                return static_cast<int64_t>(kFThetaMaxPolynomialTerms);
            }
        )
        .def_property(
            "resolution",
            [](const c10::intrusive_ptr<FThetaProjection>& self) {
                return std::make_tuple(self->resolution[0], self->resolution[1]);
            }
        )
        .def(
            "transform",
            [](const c10::intrusive_ptr<FThetaProjection>& self,
               std::tuple<double, double> scale,
               std::tuple<double, double> offset,
               std::tuple<int64_t, int64_t> new_resolution) {
                auto opts = self->principal_point.options();
                double scale_u = std::get<0>(scale);
                double scale_v = std::get<1>(scale);

                auto scale_t =
                    at::tensor({scale_u, scale_v}, at::kDouble).to(opts);
                auto offset_t = at::tensor(
                    {std::get<0>(offset), std::get<1>(offset)}, at::kDouble)
                    .to(opts);
                auto half_t = at::tensor({0.5, 0.5}, at::kDouble).to(opts);
                auto new_principal_point =
                    (self->principal_point + half_t) * scale_t - half_t - offset_t;

                auto new_fw_poly = self->fw_poly * scale_v;

                auto bw_size = self->bw_poly.size(0);
                auto powers = at::arange(bw_size, opts);
                auto inv_scale_v_t = at::full({}, 1.0 / scale_v, opts);
                auto bw_factors = at::pow(inv_scale_v_t, powers);
                auto new_bw_poly = self->bw_poly * bw_factors;

                double ratio = scale_u / scale_v;
                auto ratio_vec =
                    at::tensor({ratio, ratio, 1.0, 1.0}, at::kDouble).to(opts);
                auto new_A = self->A * ratio_vec;

                auto ptr = c10::make_intrusive<FThetaProjection>(
                    new_principal_point,
                    new_fw_poly,
                    new_bw_poly,
                    new_A,
                    std::array<int64_t, 2>{
                        std::get<0>(new_resolution),
                        std::get<1>(new_resolution)},
                    self->reference_polynomial,
                    self->fw_poly_degree,
                    self->bw_poly_degree,
                    self->newton_iterations,
                    self->max_angle,
                    self->min_2d_norm);
                gsplat_sensors::check_ftheta_projection(ptr);
                return ptr;
            },
            "Transform image-domain intrinsics",
            {torch::arg("scale"), torch::arg("offset"), torch::arg("new_resolution")}
        )
        .def_pickle(
            [](const c10::intrusive_ptr<FThetaProjection>& self) -> c10::IValue {
                // Version 2 omits Ainv (computed on demand from A).
                return c10::IValue(c10::ivalue::Tuple::create({
                    c10::IValue(static_cast<int64_t>(2)),
                    c10::IValue(self->principal_point),
                    c10::IValue(self->fw_poly),
                    c10::IValue(self->bw_poly),
                    c10::IValue(self->A),
                    c10::IValue(self->resolution[0]),
                    c10::IValue(self->resolution[1]),
                    c10::IValue(self->reference_polynomial),
                    c10::IValue(self->fw_poly_degree),
                    c10::IValue(self->bw_poly_degree),
                    c10::IValue(self->newton_iterations),
                    c10::IValue(self->max_angle),
                    c10::IValue(self->min_2d_norm),
                }));
            },
            [](c10::IValue state) -> c10::intrusive_ptr<FThetaProjection> {
                auto elems = state.toTuple()->elements();
                TORCH_CHECK(
                    !elems.empty() && elems[0].isInt(),
                    "unsupported FThetaProjection pickle state");
                int64_t version = elems[0].toInt();
                c10::intrusive_ptr<FThetaProjection> ptr;
                if (version == 2) {
                    TORCH_CHECK(
                        elems.size() == 13,
                        "unsupported FThetaProjection pickle state");
                    ptr = c10::make_intrusive<FThetaProjection>(
                        elems[1].toTensor(),
                        elems[2].toTensor(),
                        elems[3].toTensor(),
                        elems[4].toTensor(),
                        std::array<int64_t, 2>{elems[5].toInt(), elems[6].toInt()},
                        elems[7].toInt(),
                        elems[8].toInt(),
                        elems[9].toInt(),
                        elems[10].toInt(),
                        elems[11].toDouble(),
                        elems[12].toDouble());
                } else if (version == 1) {
                    // Version 1 payload carries an Ainv tensor at index 5;
                    // ignore it (the inverse is recomputed from A).
                    TORCH_CHECK(
                        elems.size() == 14,
                        "unsupported FThetaProjection pickle state");
                    ptr = c10::make_intrusive<FThetaProjection>(
                        elems[1].toTensor(),
                        elems[2].toTensor(),
                        elems[3].toTensor(),
                        elems[4].toTensor(),
                        std::array<int64_t, 2>{elems[6].toInt(), elems[7].toInt()},
                        elems[8].toInt(),
                        elems[9].toInt(),
                        elems[10].toInt(),
                        elems[11].toInt(),
                        elems[12].toDouble(),
                        elems[13].toDouble());
                } else {
                    TORCH_CHECK(
                        false, "unsupported FThetaProjection pickle state");
                }
                gsplat_sensors::check_ftheta_projection(ptr);
                return ptr;
            }
        );

    m.class_<RowOffsetStructuredSpinningLidarProjection>("RowOffsetStructuredSpinningLidarProjection")
        .def(
            torch::init([](
                at::Tensor row_elevations_rad,
                at::Tensor column_azimuths_rad,
                at::Tensor row_azimuth_offsets_rad,
                double fov_vert_start_rad,
                double fov_vert_span_rad,
                double fov_horiz_start_rad,
                double fov_horiz_span_rad,
                int64_t spinning_direction,
                bool has_row_offsets) {
                auto ptr = c10::make_intrusive<RowOffsetStructuredSpinningLidarProjection>(
                    std::move(row_elevations_rad),
                    std::move(column_azimuths_rad),
                    std::move(row_azimuth_offsets_rad),
                    fov_vert_start_rad,
                    fov_vert_span_rad,
                    fov_horiz_start_rad,
                    fov_horiz_span_rad,
                    spinning_direction,
                    has_row_offsets);
                gsplat_sensors::check_lidar_projection(ptr);
                return ptr;
            }),
            "Construct from per-component angle tables and scalar FOV / direction fields",
            {torch::arg("row_elevations_rad"),
             torch::arg("column_azimuths_rad"),
             torch::arg("row_azimuth_offsets_rad"),
             torch::arg("fov_vert_start_rad"),
             torch::arg("fov_vert_span_rad"),
             torch::arg("fov_horiz_start_rad"),
             torch::arg("fov_horiz_span_rad"),
             torch::arg("spinning_direction"),
             torch::arg("has_row_offsets")}
        )
        .def_readonly("row_elevations_rad", &RowOffsetStructuredSpinningLidarProjection::row_elevations_rad)
        .def_readonly("column_azimuths_rad", &RowOffsetStructuredSpinningLidarProjection::column_azimuths_rad)
        .def_readonly("row_azimuth_offsets_rad", &RowOffsetStructuredSpinningLidarProjection::row_azimuth_offsets_rad)
        .def_readonly("fov_vert_start_rad", &RowOffsetStructuredSpinningLidarProjection::fov_vert_start_rad)
        .def_readonly("fov_vert_span_rad", &RowOffsetStructuredSpinningLidarProjection::fov_vert_span_rad)
        .def_readonly("fov_horiz_start_rad", &RowOffsetStructuredSpinningLidarProjection::fov_horiz_start_rad)
        .def_readonly("fov_horiz_span_rad", &RowOffsetStructuredSpinningLidarProjection::fov_horiz_span_rad)
        .def_readonly("spinning_direction", &RowOffsetStructuredSpinningLidarProjection::spinning_direction)
        .def_readonly("has_row_offsets", &RowOffsetStructuredSpinningLidarProjection::has_row_offsets)
        .def_pickle(
            [](const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection>& self) -> c10::IValue {
                return c10::IValue(c10::ivalue::Tuple::create({
                    c10::IValue(static_cast<int64_t>(1)),
                    c10::IValue(self->row_elevations_rad),
                    c10::IValue(self->column_azimuths_rad),
                    c10::IValue(self->row_azimuth_offsets_rad),
                    c10::IValue(self->fov_vert_start_rad),
                    c10::IValue(self->fov_vert_span_rad),
                    c10::IValue(self->fov_horiz_start_rad),
                    c10::IValue(self->fov_horiz_span_rad),
                    c10::IValue(self->spinning_direction),
                    c10::IValue(self->has_row_offsets),
                }));
            },
            [](c10::IValue state) -> c10::intrusive_ptr<RowOffsetStructuredSpinningLidarProjection> {
                auto elems = state.toTuple()->elements();
                TORCH_CHECK(
                    elems.size() == 10 && elems[0].toInt() == 1,
                    "unsupported RowOffsetStructuredSpinningLidarProjection pickle state");
                auto ptr = c10::make_intrusive<RowOffsetStructuredSpinningLidarProjection>(
                    elems[1].toTensor(),
                    elems[2].toTensor(),
                    elems[3].toTensor(),
                    elems[4].toDouble(),
                    elems[5].toDouble(),
                    elems[6].toDouble(),
                    elems[7].toDouble(),
                    elems[8].toInt(),
                    elems[9].toBool());
                gsplat_sensors::check_lidar_projection(ptr);
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

    m.def(
        "camera_rays_to_image_points_ftheta_no_external",
        &gsplat_sensors::camera_rays_to_image_points_ftheta_no_external);
    m.def(
        "image_points_to_camera_rays_ftheta_no_external",
        &gsplat_sensors::image_points_to_camera_rays_ftheta_no_external);
    m.def(
        "camera_rays_to_image_points_ftheta_bivariate_windshield",
        &gsplat_sensors::camera_rays_to_image_points_ftheta_bivariate_windshield);
    m.def(
        "image_points_to_camera_rays_ftheta_bivariate_windshield",
        &gsplat_sensors::image_points_to_camera_rays_ftheta_bivariate_windshield);
    m.def(
        "project_world_points_mean_pose_ftheta_no_external",
        &gsplat_sensors::project_world_points_mean_pose_ftheta_no_external);
    m.def(
        "project_world_points_mean_pose_ftheta_bivariate_windshield",
        &gsplat_sensors::project_world_points_mean_pose_ftheta_bivariate_windshield);
    m.def(
        "image_points_to_world_rays_static_pose_ftheta_no_external",
        &gsplat_sensors::image_points_to_world_rays_static_pose_ftheta_no_external);
    m.def(
        "image_points_to_world_rays_static_pose_ftheta_bivariate_windshield",
        &gsplat_sensors::image_points_to_world_rays_static_pose_ftheta_bivariate_windshield);
    m.def(
        "project_world_points_shutter_pose_ftheta_no_external",
        &gsplat_sensors::project_world_points_shutter_pose_ftheta_no_external);
    m.def(
        "project_world_points_shutter_pose_ftheta_bivariate_windshield",
        &gsplat_sensors::project_world_points_shutter_pose_ftheta_bivariate_windshield);
    m.def(
        "image_points_to_world_rays_shutter_pose_ftheta_no_external",
        &gsplat_sensors::image_points_to_world_rays_shutter_pose_ftheta_no_external);
    m.def(
        "image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield",
        &gsplat_sensors::image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield);

    m.def(
        "camera_rays_to_image_points_ftheta_no_external_backward",
        &gsplat_sensors::camera_rays_to_image_points_ftheta_no_external_backward);
    m.def(
        "image_points_to_camera_rays_ftheta_no_external_backward",
        &gsplat_sensors::image_points_to_camera_rays_ftheta_no_external_backward);
    m.def(
        "camera_rays_to_image_points_ftheta_bivariate_windshield_backward",
        &gsplat_sensors::camera_rays_to_image_points_ftheta_bivariate_windshield_backward);
    m.def(
        "image_points_to_camera_rays_ftheta_bivariate_windshield_backward",
        &gsplat_sensors::image_points_to_camera_rays_ftheta_bivariate_windshield_backward);
    m.def(
        "project_world_points_mean_pose_ftheta_no_external_backward",
        &gsplat_sensors::project_world_points_mean_pose_ftheta_no_external_backward);
    m.def(
        "project_world_points_mean_pose_ftheta_bivariate_windshield_backward",
        &gsplat_sensors::project_world_points_mean_pose_ftheta_bivariate_windshield_backward);
    m.def(
        "image_points_to_world_rays_static_pose_ftheta_no_external_backward",
        &gsplat_sensors::image_points_to_world_rays_static_pose_ftheta_no_external_backward);
    m.def(
        "image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_backward",
        &gsplat_sensors::image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_backward);
    m.def(
        "project_world_points_shutter_pose_ftheta_no_external_backward",
        &gsplat_sensors::project_world_points_shutter_pose_ftheta_no_external_backward);
    m.def(
        "project_world_points_shutter_pose_ftheta_bivariate_windshield_backward",
        &gsplat_sensors::project_world_points_shutter_pose_ftheta_bivariate_windshield_backward);
    m.def(
        "image_points_to_world_rays_shutter_pose_ftheta_no_external_backward",
        &gsplat_sensors::image_points_to_world_rays_shutter_pose_ftheta_no_external_backward);
    m.def(
        "image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_backward",
        &gsplat_sensors::image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_backward);
    m.def("sensor_rays_to_sensor_angles", &gsplat_sensors::sensor_rays_to_sensor_angles);
    m.def(
        "sensor_rays_to_sensor_angles_backward",
        &gsplat_sensors::sensor_rays_to_sensor_angles_backward);
    m.def("sensor_angles_to_sensor_rays", &gsplat_sensors::sensor_angles_to_sensor_rays);
    m.def(
        "sensor_angles_to_sensor_rays_backward",
        &gsplat_sensors::sensor_angles_to_sensor_rays_backward);
    m.def("elements_to_sensor_angles", &gsplat_sensors::elements_to_sensor_angles);
    m.def(
        "elements_to_sensor_angles_backward",
        &gsplat_sensors::elements_to_sensor_angles_backward);
    m.def("generate_spinning_lidar_rays", &gsplat_sensors::generate_spinning_lidar_rays);
    m.def(
        "generate_spinning_lidar_rays_backward",
        &gsplat_sensors::generate_spinning_lidar_rays_backward);
    m.def(
        "inverse_project_spinning_lidar",
        &gsplat_sensors::inverse_project_spinning_lidar);
    m.def(
        "inverse_project_spinning_lidar_backward",
        &gsplat_sensors::inverse_project_spinning_lidar_backward);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using gsplat_sensors::ShutterType;
    using gsplat_sensors::SpinningDirection;
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

    py::enum_<SpinningDirection>(
        m,
        "SpinningDirection",
        "Source of truth for spinning-LiDAR azimuth sweep direction. Bound from "
        "C++ (see csrc/lidar_params.h) and re-exported as an IntEnum from "
        "libs/sensors/kernels/lidars/types.py.")
        .value("CLOCKWISE", SpinningDirection::CLOCKWISE)
        .value("COUNTERCLOCKWISE", SpinningDirection::COUNTERCLOCKWISE);
}
