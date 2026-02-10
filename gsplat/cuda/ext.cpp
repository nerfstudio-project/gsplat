/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
#include "ExternalDistortion.h"
#include "csrc/Config.h"

#if BUILD_CAMERA_WRAPPERS
#include "CameraWrappers.h"
#endif

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
        .value("LIDAR", gsplat::CameraModelType::LIDAR)
        .export_values();

    py::enum_<gsplat::extdist::ModelType>(m, "ExternalDistortionModelType", py::module_local())
        .value("BIVARIATE_WINDSHIELD", gsplat::extdist::ModelType::BIVARIATE_WINDSHIELD);

    m.def("null", &gsplat::null);
}

TORCH_LIBRARY(gsplat, m) {
    m.class_<UnscentedTransformParameters>("UnscentedTransformParameters")
        .def(
            torch::init([](double alpha,
                           double beta,
                           double kappa,
                           double in_image_margin_factor,
                           bool require_all_sigma_points_valid) {
                return c10::make_intrusive<UnscentedTransformParameters>(
                    static_cast<float>(alpha),
                    static_cast<float>(beta),
                    static_cast<float>(kappa),
                    static_cast<float>(in_image_margin_factor),
                    require_all_sigma_points_valid
                );
            }),
            "Dataclass constructor",
            {torch::arg("alpha") = 0.1,
             torch::arg("beta") = 2.,
             torch::arg("kappa") = 0.,
             torch::arg("in_image_margin_factor") = 0.1,
             torch::arg("require_all_sigma_points_valid") = false}
        )
        .def_property(
            "alpha",
            [](const c10::intrusive_ptr<UnscentedTransformParameters> &self) {
                return static_cast<double>(self->alpha);
            },
            [](const c10::intrusive_ptr<UnscentedTransformParameters> &self,
               double alpha) { self->alpha = static_cast<float>(alpha); }
        )
        .def_property(
            "beta",
            [](const c10::intrusive_ptr<UnscentedTransformParameters> &self) {
                return static_cast<double>(self->beta);
            },
            [](const c10::intrusive_ptr<UnscentedTransformParameters> &self,
               double beta) { self->beta = static_cast<float>(beta); }
        )
        .def_property(
            "kappa",
            [](const c10::intrusive_ptr<UnscentedTransformParameters> &self) {
                return static_cast<double>(self->kappa);
            },
            [](const c10::intrusive_ptr<UnscentedTransformParameters> &self,
               double kappa) { self->kappa = static_cast<float>(kappa); }
        )
        .def_property(
            "in_image_margin_factor",
            [](const c10::intrusive_ptr<UnscentedTransformParameters> &self) {
                return static_cast<double>(self->in_image_margin_factor);
            },
            [](const c10::intrusive_ptr<UnscentedTransformParameters> &self,
               double in_image_margin_factor) {
                self->in_image_margin_factor =
                    static_cast<float>(in_image_margin_factor);
            }
        )
        .def_readwrite(
            "require_all_sigma_points_valid",
            &UnscentedTransformParameters::require_all_sigma_points_valid
        );

    using FThetaPolynomialType = FThetaCameraDistortionParameters::PolynomialType;
    constexpr auto FThetaPolynomialDegree = FThetaCameraDistortionParameters::PolynomialDegree;
    m.class_<FThetaCameraDistortionParameters>("FThetaCameraDistortionParameters")
        .def(
            torch::init([](int64_t reference_poly,
                           std::array<double, FThetaPolynomialDegree> pixeldist_to_angle_poly,
                           std::array<double, FThetaPolynomialDegree> angle_to_pixeldist_poly,
                           double max_angle,
                           std::array<double, 3> linear_cde) {
                std::array<float, FThetaPolynomialDegree> pixeldist_to_angle_poly_f;
                for (auto i = 0; i < FThetaPolynomialDegree; ++i)
                    pixeldist_to_angle_poly_f[i] =
                        static_cast<float>(pixeldist_to_angle_poly[i]);

                std::array<float, FThetaPolynomialDegree> angle_to_pixeldist_poly_f;
                for (auto i = 0; i < FThetaPolynomialDegree; ++i)
                    angle_to_pixeldist_poly_f[i] =
                        static_cast<float>(angle_to_pixeldist_poly[i]);

                std::array<float, 3> linear_cde_f = {
                    static_cast<float>(linear_cde[0]),
                    static_cast<float>(linear_cde[1]),
                    static_cast<float>(linear_cde[2])
                };

                return c10::make_intrusive<FThetaCameraDistortionParameters>(
                    static_cast<FThetaPolynomialType>(reference_poly),
                    pixeldist_to_angle_poly_f,
                    angle_to_pixeldist_poly_f,
                    static_cast<float>(max_angle),
                    linear_cde_f
                );
            }),
            "Dataclass constructor",
            {torch::arg("reference_poly") = 0,
             torch::arg("pixeldist_to_angle_poly") = std::array<double, FThetaPolynomialDegree>{},
             torch::arg("angle_to_pixeldist_poly") = std::array<double, FThetaPolynomialDegree>{},
             torch::arg("max_angle") = 0.,
             torch::arg("linear_cde") = std::array<double, 3>{}}
        )
        .def_property(
            "reference_poly",
            [](const c10::intrusive_ptr<FThetaCameraDistortionParameters> &self
            ) { return static_cast<int64_t>(self->reference_poly); },
            [](const c10::intrusive_ptr<FThetaCameraDistortionParameters> &self,
               int64_t reference_poly) {
                self->reference_poly =
                    static_cast<FThetaPolynomialType>(reference_poly);
            }
        )
        .def_property(
            "pixeldist_to_angle_poly",
            [](const c10::intrusive_ptr<FThetaCameraDistortionParameters> &self
            ) {
                std::array<double, FThetaPolynomialDegree>
                    pixeldist_to_angle_poly;
                for (int i = 0; i < FThetaPolynomialDegree; ++i) {
                    pixeldist_to_angle_poly[i] =
                        static_cast<double>(self->pixeldist_to_angle_poly[i]);
                }
                return pixeldist_to_angle_poly;
            },
            [](const c10::intrusive_ptr<FThetaCameraDistortionParameters> &self,
               const std::array<double, FThetaPolynomialDegree>
                   &pixeldist_to_angle_poly) {
                for (int i = 0; i < FThetaPolynomialDegree; ++i) {
                    self->pixeldist_to_angle_poly[i] =
                        static_cast<float>(pixeldist_to_angle_poly[i]);
                }
            }
        )
        .def_property(
            "angle_to_pixeldist_poly",
            [](const c10::intrusive_ptr<FThetaCameraDistortionParameters> &self
            ) {
                std::array<double, FThetaPolynomialDegree>
                    angle_to_pixeldist_poly;
                for (int i = 0; i < FThetaPolynomialDegree; ++i) {
                    angle_to_pixeldist_poly[i] =
                        static_cast<double>(self->angle_to_pixeldist_poly[i]);
                }
                return angle_to_pixeldist_poly;
            },
            [](const c10::intrusive_ptr<FThetaCameraDistortionParameters> &self,
               const std::array<double, FThetaPolynomialDegree>
                   &angle_to_pixeldist_poly) {
                for (int i = 0; i < FThetaPolynomialDegree; ++i) {
                    self->angle_to_pixeldist_poly[i] =
                        static_cast<float>(angle_to_pixeldist_poly[i]);
                }
            }
        )
        .def_property(
            "max_angle",
            [](const c10::intrusive_ptr<FThetaCameraDistortionParameters> &self
            ) { return static_cast<double>(self->max_angle); },
            [](const c10::intrusive_ptr<FThetaCameraDistortionParameters> &self,
               double max_angle) {
                self->max_angle = static_cast<float>(max_angle);
            }
        )
        .def_property(
            "linear_cde",
            [](const c10::intrusive_ptr<FThetaCameraDistortionParameters> &self
            ) {
                std::array<double, 3> linear_cde;
                for (int i = 0; i < 3; ++i) {
                    linear_cde[i] = static_cast<double>(self->linear_cde[i]);
                }
                return linear_cde;
            },
            [](const c10::intrusive_ptr<FThetaCameraDistortionParameters> &self,
               const std::array<double, 3> &linear_cde) {
                for (int i = 0; i < 3; ++i) {
                    self->linear_cde[i] = static_cast<float>(linear_cde[i]);
                }
            }
        );

    m.class_<gsplat::extdist::BivariateWindshieldModelParameters>("BivariateWindshieldModelParameters")
        .def(torch::init<>())
        .def_readwrite("horizontal_poly", &gsplat::extdist::BivariateWindshieldModelParameters::horizontal_poly)
        .def_readwrite("vertical_poly", &gsplat::extdist::BivariateWindshieldModelParameters::vertical_poly)
        .def_readwrite("horizontal_poly_inverse", &gsplat::extdist::BivariateWindshieldModelParameters::horizontal_poly_inverse)
        .def_readwrite("vertical_poly_inverse", &gsplat::extdist::BivariateWindshieldModelParameters::vertical_poly_inverse);

    // Lidar sensor support
    m.class_<gsplat::FOV>("FOV")
        .def(
            torch::init([](double start, double span) {
                return c10::make_intrusive<gsplat::FOV>(
                    static_cast<float>(start), static_cast<float>(span)
                );
            }),
            "Constructor",
            {torch::arg("start") = 0., torch::arg("span") = 0.}
        )
        .def_property(
            "start",
            [](const c10::intrusive_ptr<gsplat::FOV> &self) { return static_cast<double>(self->start); },
            [](const c10::intrusive_ptr<gsplat::FOV> &self, double start) { self->start = static_cast<float>(start); }
        )
        .def_property(
            "span",
            [](const c10::intrusive_ptr<gsplat::FOV> &self) { return static_cast<double>(self->span); },
            [](const c10::intrusive_ptr<gsplat::FOV> &self, double span) { self->span = static_cast<float>(span); }
        );

    m.class_<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt>("RowOffsetStructuredSpinningLidarModelParametersExt")
        .def(
            torch::init([](at::Tensor row_elevations_rad,
                           at::Tensor column_azimuths_rad,
                           at::Tensor row_azimuth_offsets_rad,
                           int64_t spinning_direction,
                           double spinning_frequency_hz,
                           c10::intrusive_ptr<gsplat::FOV> fov_vert_rad,
                           c10::intrusive_ptr<gsplat::FOV> fov_horiz_rad,
                           double fov_eps_rad,
                           at::Tensor angles_to_columns_map,
                           int64_t n_bins_azimuth,
                           int64_t n_bins_elevation,
                           at::Tensor cdf_elevation,
                           at::Tensor cdf_dense_ray_mask,
                           at::Tensor tiles_pack_info,
                           at::Tensor tiles_to_elements_map) {
                return c10::make_intrusive<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt>(
                    std::move(row_elevations_rad),
                    std::move(column_azimuths_rad),
                    std::move(row_azimuth_offsets_rad),
                    static_cast<gsplat::SpinningDirection>(spinning_direction),
                    static_cast<float>(spinning_frequency_hz),
                    std::move(fov_vert_rad),
                    std::move(fov_horiz_rad),
                    static_cast<float>(fov_eps_rad),
                    std::move(angles_to_columns_map),
                    static_cast<int>(n_bins_azimuth),
                    static_cast<int>(n_bins_elevation),
                    std::move(cdf_elevation),
                    std::move(cdf_dense_ray_mask),
                    std::move(tiles_pack_info),
                    std::move(tiles_to_elements_map));
            }),
            "Constructor",
            {torch::arg("row_elevations_rad"),
             torch::arg("column_azimuths_rad"),
             torch::arg("row_azimuth_offsets_rad"),
             torch::arg("spinning_direction"),
             torch::arg("spinning_frequency_hz"),
             torch::arg("fov_vert_rad"),
             torch::arg("fov_horiz_rad"),
             torch::arg("fov_eps_rad"),
             torch::arg("angles_to_columns_map"),
             torch::arg("n_bins_azimuth"),
             torch::arg("n_bins_elevation"),
             torch::arg("cdf_elevation"),
             torch::arg("cdf_dense_ray_mask"),
             torch::arg("tiles_pack_info"),
             torch::arg("tiles_to_elements_map")}
        )
        .def_readwrite("row_elevations_rad", &gsplat::RowOffsetStructuredSpinningLidarModelParametersExt::row_elevations_rad)
        .def_readwrite("column_azimuths_rad", &gsplat::RowOffsetStructuredSpinningLidarModelParametersExt::column_azimuths_rad)
        .def_readwrite("row_azimuth_offsets_rad", &gsplat::RowOffsetStructuredSpinningLidarModelParametersExt::row_azimuth_offsets_rad)
        .def_property(
            "spinning_direction",
            [](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &self) {
                return static_cast<int64_t>(self->spinning_direction);
            },
            [](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &self, int64_t v) {
                self->spinning_direction = static_cast<gsplat::SpinningDirection>(v);
            }
        )
        .def_property(
            "spinning_frequency_hz",
            [](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &self) {
                return static_cast<double>(self->spinning_frequency_hz);
            },
            [](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &self, double v) {
                self->spinning_frequency_hz = static_cast<float>(v);
            }
        )
        .def_readwrite("fov_vert_rad", &gsplat::RowOffsetStructuredSpinningLidarModelParametersExt::fov_vert_rad)
        .def_readwrite("fov_horiz_rad", &gsplat::RowOffsetStructuredSpinningLidarModelParametersExt::fov_horiz_rad)
        .def_property(
            "fov_eps_rad",
            [](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &self) {
                return static_cast<double>(self->fov_eps_rad);
            },
            [](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &self, double v) {
                self->fov_eps_rad = static_cast<float>(v);
            }
        )
        .def_readwrite("angles_to_columns_map", &gsplat::RowOffsetStructuredSpinningLidarModelParametersExt::angles_to_columns_map)
        .def_property(
            "n_rows",
            [](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &self) {
                return static_cast<int64_t>(self->n_rows());
            }
        )
        .def_property(
            "n_columns",
            [](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &self) {
                return static_cast<int64_t>(self->n_columns());
            }
        )
        .def_property(
            "n_bins_azimuth",
            [](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &self) {
                return static_cast<int64_t>(self->n_bins_azimuth);
            },
            [](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &self, int64_t v) {
                self->n_bins_azimuth = static_cast<int>(v);
            }
        )
        .def_property(
            "n_bins_elevation",
            [](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &self) {
                return static_cast<int64_t>(self->n_bins_elevation);
            },
            [](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &self, int64_t v) {
                self->n_bins_elevation = static_cast<int>(v);
            }
        )
        .def_readwrite("cdf_elevation", &gsplat::RowOffsetStructuredSpinningLidarModelParametersExt::cdf_elevation)
        .def_readwrite("cdf_dense_ray_mask", &gsplat::RowOffsetStructuredSpinningLidarModelParametersExt::cdf_dense_ray_mask)
        .def_readwrite("tiles_pack_info", &gsplat::RowOffsetStructuredSpinningLidarModelParametersExt::tiles_pack_info)
        .def_readwrite("tiles_to_elements_map", &gsplat::RowOffsetStructuredSpinningLidarModelParametersExt::tiles_to_elements_map)
        .def_property(
            "cdf_resolution_elevation",
            [](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &self) {
                return static_cast<int64_t>(self->cdf_resolution_elevation());
            }
        )
        .def_property(
            "cdf_resolution_azimuth",
            [](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &self) {
                return static_cast<int64_t>(self->cdf_resolution_azimuth());
            }
        );

    // ==================== Camera Model Bindings ====================

#if BUILD_CAMERA_WRAPPERS
    m.class_<gsplat::PyBaseCameraModel<>>("BaseCameraModel")
        .def(
            "camera_ray_to_image_point",
            [](const c10::intrusive_ptr<gsplat::PyBaseCameraModel<>> &self,
               const torch::Tensor &camera_ray,
               double margin_factor) {
                return self->camera_ray_to_image_point(camera_ray, static_cast<float>(margin_factor));
            },
            "Project camera rays to image points",
            {torch::arg("camera_ray"), torch::arg("margin_factor") = 0.0}
        )
        .def(
            "image_point_to_camera_ray",
            &gsplat::PyBaseCameraModel<>::image_point_to_camera_ray,
            "Unproject image points to camera rays",
            {torch::arg("image_points")}
        )
        .def(
            "shutter_relative_frame_time",
            &gsplat::PyBaseCameraModel<>::shutter_relative_frame_time,
            "Compute relative frame time for rolling shutter",
            {torch::arg("image_points")}
        )
        .def(
            "image_point_to_world_ray_shutter_pose",
            &gsplat::PyBaseCameraModel<>::image_point_to_world_ray_shutter_pose,
            "Unproject image point to world ray with rolling shutter",
            {torch::arg("image_points"), torch::arg("pose_start"), torch::arg("pose_end")}
        )
        .def(
            "world_point_to_image_point_shutter_pose",
            [](const c10::intrusive_ptr<gsplat::PyBaseCameraModel<>> &self,
               const torch::Tensor &world_points,
               const torch::Tensor &pose_start,
               const torch::Tensor &pose_end,
               double margin_factor) {
                return self->world_point_to_image_point_shutter_pose(
                    world_points, pose_start, pose_end, static_cast<float>(margin_factor)
                );
            },
            "Project world points to image with rolling shutter",
            {torch::arg("world_points"), torch::arg("pose_start"), torch::arg("pose_end"), torch::arg("margin_factor") = 0.0
            }
        )
        .def_property(
            "width",
            [](const c10::intrusive_ptr<gsplat::PyBaseCameraModel<>> &self) { return static_cast<int64_t>(self->width()); }
        )
        .def_property(
            "height",
            [](const c10::intrusive_ptr<gsplat::PyBaseCameraModel<>> &self) { return static_cast<int64_t>(self->height()); }
        )
        .def_property(
            "rs_type",
            [](const c10::intrusive_ptr<gsplat::PyBaseCameraModel<>> &self) {
                return static_cast<int64_t>(self->rs_type());
            }
        )
        .def_property("principal_points", &gsplat::PyBaseCameraModel<>::principal_points)
        .def_property("focal_lengths", &gsplat::PyBaseCameraModel<>::focal_lengths)
        .def_static(
            "create",
            [](int64_t width,
               int64_t height,
               const std::string &camera_model,
               const torch::Tensor &principal_points,
               const std::optional<torch::Tensor> &focal_lengths,
               const std::optional<torch::Tensor> &radial_coeffs,
               const std::optional<torch::Tensor> &tangential_coeffs,
               const std::optional<torch::Tensor> &thin_prism_coeffs,
               const std::optional<c10::intrusive_ptr<FThetaCameraDistortionParameters>> &ftheta_coeffs,
               int64_t rs_type) {
                return gsplat::PyBaseCameraModel<>::create(
                    width,
                    height,
                    camera_model,
                    principal_points,
                    focal_lengths,
                    radial_coeffs,
                    tangential_coeffs,
                    thin_prism_coeffs,
                    ftheta_coeffs,
                    static_cast<ShutterType>(rs_type)
                );
            },
            "Factory method to create camera from model type"
        );

    m.class_<gsplat::PyPerfectPinholeCameraModel>("PerfectPinholeCameraModel")
        .def(
            torch::init([](int64_t width,
                           int64_t height,
                           const torch::Tensor &focal_lengths,
                           const torch::Tensor &principal_points,
                           int64_t rs_type) {
                return c10::make_intrusive<gsplat::PyPerfectPinholeCameraModel>(
                    width, height, focal_lengths, principal_points, static_cast<ShutterType>(rs_type)
                );
            }),
            "Constructor",
            {torch::arg("width"),
             torch::arg("height"),
             torch::arg("focal_lengths"),
             torch::arg("principal_points"),
             torch::arg("rs_type")}
        );

    m.class_<gsplat::PyOpenCVPinholeCameraModel>("OpenCVPinholeCameraModel")
        .def(
            torch::init([](int64_t width,
                           int64_t height,
                           const torch::Tensor &focal_lengths,
                           const torch::Tensor &principal_points,
                           const std::optional<torch::Tensor> &radial_coeffs,
                           const std::optional<torch::Tensor> &tangential_coeffs,
                           const std::optional<torch::Tensor> &thin_prism_coeffs,
                           int64_t rs_type) {
                return c10::make_intrusive<gsplat::PyOpenCVPinholeCameraModel>(
                    width,
                    height,
                    focal_lengths,
                    principal_points,
                    radial_coeffs,
                    tangential_coeffs,
                    thin_prism_coeffs,
                    static_cast<ShutterType>(rs_type)
                );
            }),
            "Constructor",
            {torch::arg("width"),
             torch::arg("height"),
             torch::arg("focal_lengths"),
             torch::arg("principal_points"),
             torch::arg("radial_coeffs"),
             torch::arg("tangential_coeffs"),
             torch::arg("thin_prism_coeffs"),
             torch::arg("rs_type")}
        );

    m.class_<gsplat::PyOpenCVFisheyeCameraModel>("OpenCVFisheyeCameraModel")
        .def(
            torch::init([](

                            int64_t width,
                            int64_t height,
                            const torch::Tensor &focal_lengths,
                            const torch::Tensor &principal_points,
                            const std::optional<torch::Tensor> &radial_coeffs, // [..., 4]
                            int64_t rs_type

                        ) {
                return c10::make_intrusive<gsplat::PyOpenCVFisheyeCameraModel>(
                    width, height, focal_lengths, principal_points, radial_coeffs, static_cast<ShutterType>(rs_type)
                );
            }),
            "Constructor",
            {torch::arg("width"),
             torch::arg("height"),
             torch::arg("focal_lengths"),
             torch::arg("principal_points"),
             torch::arg("radial_coeffs"),
             torch::arg("rs_type")}
        );

    m.class_<gsplat::PyFThetaCameraModel>("FThetaCameraModel")
        .def(
            torch::init([](int64_t width,
                           int64_t height,
                           const torch::Tensor &principal_points,
                           const torch::Tensor &pixeldist_to_angle_poly,
                           const torch::Tensor &angle_to_pixeldist_poly,
                           const torch::Tensor &linear_cde,
                           int64_t reference_poly,
                           const torch::Tensor &max_angle,
                           int64_t rs_type) {
                return c10::make_intrusive<gsplat::PyFThetaCameraModel>(
                    width,
                    height,
                    principal_points,
                    pixeldist_to_angle_poly,
                    angle_to_pixeldist_poly,
                    linear_cde,
                    static_cast<FThetaCameraDistortionParameters::PolynomialType>(reference_poly),
                    max_angle,
                    static_cast<ShutterType>(rs_type)
                );
            }),
            "Constructor",
            {torch::arg("width"),
             torch::arg("height"),
             torch::arg("principal_points"),
             torch::arg("pixeldist_to_angle_poly"),
             torch::arg("angle_to_pixeldist_poly"),
             torch::arg("linear_cde"),
             torch::arg("reference_poly"),
             torch::arg("max_angle"),
             torch::arg("rs_type")}
        );

    m.class_<gsplat::PyRowOffsetStructuredSpinningLidarModel>("RowOffsetStructuredSpinningLidarModel")
        .def(
            torch::init([](const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &params) {
                return c10::make_intrusive<gsplat::PyRowOffsetStructuredSpinningLidarModel>(*params);
            }),
            "Constructor",
            {torch::arg("params")}
        )
        .def(
            "camera_ray_to_image_point",
            [](const c10::intrusive_ptr<gsplat::PyRowOffsetStructuredSpinningLidarModel> &self,
               const torch::Tensor &camera_ray,
               double margin_factor) {
                return self->camera_ray_to_image_point(camera_ray, static_cast<float>(margin_factor));
            },
            "Project camera rays to image points",
            {torch::arg("camera_ray"), torch::arg("margin_factor") = 0.0}
        )
        .def(
            "image_point_to_camera_ray",
            [](const c10::intrusive_ptr<gsplat::PyRowOffsetStructuredSpinningLidarModel> &self,
               const torch::Tensor &image_points) {
                return self->image_point_to_camera_ray(image_points);
            },
            "Unproject image points to camera rays",
            {torch::arg("image_points")}
        )
        .def(
            "shutter_relative_frame_time",
            [](const c10::intrusive_ptr<gsplat::PyRowOffsetStructuredSpinningLidarModel> &self,
               const torch::Tensor &image_points) {
                return self->shutter_relative_frame_time(image_points);
            },
            "Compute relative frame time for rolling shutter",
            {torch::arg("image_points")}
        )
        .def(
            "image_point_to_world_ray_shutter_pose",
            [](const c10::intrusive_ptr<gsplat::PyRowOffsetStructuredSpinningLidarModel> &self,
               const torch::Tensor &image_points,
               const torch::Tensor &pose_start,
               const torch::Tensor &pose_end) {
                return self->image_point_to_world_ray_shutter_pose(image_points, pose_start, pose_end);
            },
            "Unproject image point to world ray with rolling shutter",
            {torch::arg("image_points"), torch::arg("pose_start"), torch::arg("pose_end")}
        )
        .def(
            "world_point_to_image_point_shutter_pose",
            [](const c10::intrusive_ptr<gsplat::PyRowOffsetStructuredSpinningLidarModel> &self,
               const torch::Tensor &world_points,
               const torch::Tensor &pose_start,
               const torch::Tensor &pose_end,
               double margin_factor) {
                return self->world_point_to_image_point_shutter_pose(
                    world_points, pose_start, pose_end, static_cast<float>(margin_factor)
                );
            },
            "Project world points to image with rolling shutter",
            {torch::arg("world_points"), torch::arg("pose_start"), torch::arg("pose_end"), torch::arg("margin_factor") = 0.0}
        )
        .def_property(
            "width",
            [](const c10::intrusive_ptr<gsplat::PyRowOffsetStructuredSpinningLidarModel> &self) { return static_cast<int64_t>(self->width()); }
        )
        .def_property(
            "height",
            [](const c10::intrusive_ptr<gsplat::PyRowOffsetStructuredSpinningLidarModel> &self) { return static_cast<int64_t>(self->height()); }
        )
        .def_property(
            "rs_type",
            [](const c10::intrusive_ptr<gsplat::PyRowOffsetStructuredSpinningLidarModel> &self) {
                return static_cast<int64_t>(self->rs_type());
            }
        )
        .def_property(
            "principal_points",
            [](const c10::intrusive_ptr<gsplat::PyRowOffsetStructuredSpinningLidarModel> &self) { return self->principal_points(); }
        )
        .def_property(
            "focal_lengths",
            [](const c10::intrusive_ptr<gsplat::PyRowOffsetStructuredSpinningLidarModel> &self) { return self->focal_lengths(); }
        );
#endif


#if GSPLAT_BUILD_3DGS
    m.def("quat_scale_to_covar_preci_fwd(Tensor quats, Tensor scales, bool compute_covar, bool compute_preci, bool triu) -> (Tensor, Tensor)");
    m.def("quat_scale_to_covar_preci_bwd(Tensor quats, Tensor scales, bool triu, Tensor? v_covars, Tensor? v_precis) -> (Tensor, Tensor)");
#endif

    m.def("spherical_harmonics_fwd(int degrees_to_use, Tensor dirs, Tensor coeffs, Tensor? masks) -> Tensor");
    m.def("spherical_harmonics_bwd(int K, int degrees_to_use, Tensor dirs, Tensor coeffs, Tensor? masks, Tensor v_colors, bool compute_v_dirs) -> (Tensor, Tensor)");

    m.def("intersect_tile(Tensor means2d, Tensor radii, Tensor depths, Tensor? image_ids, Tensor? gaussian_ids, int I, int tile_size, int tile_width, int tile_height, bool sort, bool segmented) -> (Tensor, Tensor, Tensor)");
    m.def("intersect_offset(Tensor isect_ids, int I, int tile_width, int tile_height) -> Tensor");
#if GSPLAT_BUILD_3DGUT
    m.def("intersect_tile_lidar(__torch__.torch.classes.gsplat.RowOffsetStructuredSpinningLidarModelParametersExt lidar, Tensor means2d, Tensor radii, Tensor depths, Tensor? image_ids, Tensor? gaussian_ids, int I, bool sort, bool segmented) -> (Tensor, Tensor, Tensor)");
#endif

#if GSPLAT_BUILD_3DGS
    m.def("projection_ewa_simple_fwd(Tensor means, Tensor covars, Tensor Ks, int width, int height, int camera_model) -> (Tensor, Tensor)");
    m.def("projection_ewa_simple_bwd(Tensor means, Tensor covars, Tensor Ks, int width, int height, int camera_model, Tensor v_means2d, Tensor v_covars2d) -> (Tensor, Tensor)");

    m.def("projection_ewa_3dgs_fused_fwd(Tensor means, Tensor? covars, Tensor? quats, Tensor? scales, Tensor? opacities, Tensor viewmats, Tensor Ks, int image_width, int image_height, float eps2d, float near_plane, float far_plane, float radius_clip, bool calc_compensations, int camera_model) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("projection_ewa_3dgs_fused_bwd(Tensor means, Tensor? covars, Tensor? quats, Tensor? scales, Tensor viewmats, Tensor Ks, int image_width, int image_height, float eps2d, int camera_model, Tensor radii, Tensor conics, Tensor? compensations, Tensor v_means2d, Tensor v_depths, Tensor v_conics, Tensor? v_compensations, bool viewmats_requires_grad) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

    m.def("projection_ewa_3dgs_packed_fwd(Tensor means, Tensor? covars, Tensor? quats, Tensor? scales, Tensor? opacities, Tensor viewmats, Tensor Ks, int image_width, int image_height, float eps2d, float near_plane, float far_plane, float radius_clip, bool calc_compensations, int camera_model) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("projection_ewa_3dgs_packed_bwd(Tensor means, Tensor? covars, Tensor? quats, Tensor? scales, Tensor viewmats, Tensor Ks, int image_width, int image_height, float eps2d, int camera_model, Tensor batch_ids, Tensor camera_ids, Tensor gaussian_ids, Tensor conics, Tensor? compensations, Tensor v_means2d, Tensor v_depths, Tensor v_conics, Tensor? v_compensations, bool viewmats_requires_grad, bool sparse_grad) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

    m.def("rasterize_to_pixels_3dgs_fwd(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor? backgrounds, Tensor? masks, int image_width, int image_height, int tile_size, Tensor tile_offsets, Tensor flatten_ids) -> (Tensor, Tensor, Tensor)");
    m.def("rasterize_to_pixels_3dgs_bwd(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor? backgrounds, Tensor? masks, int image_width, int image_height, int tile_size, Tensor tile_offsets, Tensor flatten_ids, Tensor render_alphas, Tensor last_ids, Tensor v_render_colors, Tensor v_render_alphas, bool absgrad) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("rasterize_to_indices_3dgs(int range_start, int range_end, Tensor transmittances, Tensor means2d, Tensor conics, Tensor opacities, int image_width, int image_height, int tile_size, Tensor tile_offsets, Tensor flatten_ids) -> (Tensor, Tensor)");
#endif

#if GSPLAT_BUILD_2DGS
    m.def("projection_2dgs_fused_fwd(Tensor means, Tensor quats, Tensor scales, Tensor viewmats, Tensor Ks, int image_width, int image_height, float eps2d, float near_plane, float far_plane, float radius_clip) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("projection_2dgs_fused_bwd(Tensor means, Tensor quats, Tensor scales, Tensor viewmats, Tensor Ks, int image_width, int image_height, Tensor radii, Tensor ray_transforms, Tensor v_means2d, Tensor v_depths, Tensor v_normals, Tensor v_ray_transforms, bool viewmats_requires_grad) -> (Tensor, Tensor, Tensor, Tensor)");

    m.def("projection_2dgs_packed_fwd(Tensor means, Tensor quats, Tensor scales, Tensor viewmats, Tensor Ks, int image_width, int image_height, float near_plane, float far_plane, float radius_clip) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("projection_2dgs_packed_bwd(Tensor means, Tensor quats, Tensor scales, Tensor viewmats, Tensor Ks, int image_width, int image_height, Tensor batch_ids, Tensor camera_ids, Tensor gaussian_ids, Tensor ray_transforms, Tensor v_means2d, Tensor v_depths, Tensor v_ray_transforms, Tensor v_normals, bool viewmats_requires_grad, bool sparse_grad) -> (Tensor, Tensor, Tensor, Tensor)");

    m.def("rasterize_to_pixels_2dgs_fwd(Tensor means2d, Tensor ray_transforms, Tensor colors, Tensor opacities, Tensor normals, Tensor? backgrounds, Tensor? masks, int image_width, int image_height, int tile_size, Tensor tile_offsets, Tensor flatten_ids) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("rasterize_to_pixels_2dgs_bwd(Tensor means2d, Tensor ray_transforms, Tensor colors, Tensor opacities, Tensor normals, Tensor densify, Tensor? backgrounds, Tensor? masks, int image_width, int image_height, int tile_size, Tensor tile_offsets, Tensor flatten_ids, Tensor render_colors, Tensor render_alphas, Tensor last_ids, Tensor median_ids, Tensor v_render_colors, Tensor v_render_alphas, Tensor v_render_normals, Tensor v_render_distort, Tensor v_render_median, bool absgrad) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("rasterize_to_indices_2dgs(int range_start, int range_end, Tensor transmittances, Tensor means2d, Tensor ray_transforms, Tensor opacities, int image_width, int image_height, int tile_size, Tensor tile_offsets, Tensor flatten_ids) -> (Tensor, Tensor)");
#endif

#if GSPLAT_BUILD_ADAM
    m.def("adam(Tensor(a!) param, Tensor param_grad, Tensor(b!) exp_avg, Tensor(c!) exp_avg_sq, Tensor? valid, float lr, float b1, float b2, float eps) -> ()");
#endif

#if GSPLAT_BUILD_RELOC
    m.def("relocation(Tensor opacities, Tensor scales, Tensor ratios, Tensor binoms, int n_max) -> (Tensor, Tensor)");
#endif

#if GSPLAT_BUILD_3DGUT
    m.def("projection_ut_3dgs_fused(Tensor means, Tensor quats, Tensor scales, Tensor? opacities, Tensor viewmats0, Tensor? viewmats1, Tensor Ks, int image_width, int image_height, float eps2d, float near_plane, float far_plane, float radius_clip, bool calc_compensations, int camera_model, bool global_z_order, __torch__.torch.classes.gsplat.UnscentedTransformParameters ut_params, int rs_type, Tensor? radial_coeffs, Tensor? tangential_coeffs, Tensor? thin_prism_coeffs, __torch__.torch.classes.gsplat.FThetaCameraDistortionParameters ftheta_coeffs, __torch__.torch.classes.gsplat.RowOffsetStructuredSpinningLidarModelParametersExt? lidar_coeffs, __torch__.torch.classes.gsplat.BivariateWindshieldModelParameters? external_distortion_params) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("rasterize_to_pixels_from_world_3dgs_fwd(Tensor means, Tensor quats, Tensor scales, Tensor colors, Tensor opacities, Tensor? backgrounds, Tensor? masks, int image_width, int image_height, int tile_size, Tensor viewmats0, Tensor? viewmats1, Tensor Ks, int camera_model, __torch__.torch.classes.gsplat.UnscentedTransformParameters ut_params, int rs_type, Tensor? rays, Tensor? radial_coeffs, Tensor? tangential_coeffs, Tensor? thin_prism_coeffs, __torch__.torch.classes.gsplat.FThetaCameraDistortionParameters ftheta_coeffs, __torch__.torch.classes.gsplat.RowOffsetStructuredSpinningLidarModelParametersExt? lidar_coeffs, __torch__.torch.classes.gsplat.BivariateWindshieldModelParameters? external_distortion_params, Tensor tile_offsets, Tensor flatten_ids, bool use_hit_distance, Tensor? sample_counts, Tensor? normals) -> (Tensor, Tensor, Tensor)");
    m.def("rasterize_to_pixels_from_world_3dgs_bwd(Tensor means, Tensor quats, Tensor scales, Tensor colors, Tensor opacities, Tensor? backgrounds, Tensor? masks, int image_width, int image_height, int tile_size, Tensor viewmats0, Tensor? viewmats1, Tensor Ks, int camera_model, __torch__.torch.classes.gsplat.UnscentedTransformParameters ut_params, int rs_type, Tensor? rays, Tensor? radial_coeffs, Tensor? tangential_coeffs, Tensor? thin_prism_coeffs, __torch__.torch.classes.gsplat.FThetaCameraDistortionParameters ftheta_coeffs, __torch__.torch.classes.gsplat.RowOffsetStructuredSpinningLidarModelParametersExt? lidar_coeffs, __torch__.torch.classes.gsplat.BivariateWindshieldModelParameters? external_distortion_params, Tensor tile_offsets, Tensor flatten_ids, bool use_hit_distance, Tensor render_alphas, Tensor last_ids, Tensor v_render_colors, Tensor v_render_alphas, Tensor? v_render_normals) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor?)");
#endif
}

TORCH_LIBRARY_IMPL(gsplat, CUDA, m) {
#if GSPLAT_BUILD_3DGS
    m.impl("quat_scale_to_covar_preci_fwd", &gsplat::quat_scale_to_covar_preci_fwd);
    m.impl("quat_scale_to_covar_preci_bwd", &gsplat::quat_scale_to_covar_preci_bwd);
#endif

    m.impl("intersect_tile", &gsplat::intersect_tile);
    m.impl("intersect_offset", &gsplat::intersect_offset);

    m.impl("spherical_harmonics_fwd", &gsplat::spherical_harmonics_fwd);
    m.impl("spherical_harmonics_bwd", &gsplat::spherical_harmonics_bwd);

#if GSPLAT_BUILD_3DGS
    m.impl("projection_ewa_simple_fwd", &gsplat::projection_ewa_simple_fwd);
    m.impl("projection_ewa_simple_bwd", &gsplat::projection_ewa_simple_bwd);
    m.impl("projection_ewa_3dgs_fused_fwd", &gsplat::projection_ewa_3dgs_fused_fwd);
    m.impl("projection_ewa_3dgs_fused_bwd", &gsplat::projection_ewa_3dgs_fused_bwd);
    m.impl("projection_ewa_3dgs_packed_fwd", &gsplat::projection_ewa_3dgs_packed_fwd);
    m.impl("projection_ewa_3dgs_packed_bwd", &gsplat::projection_ewa_3dgs_packed_bwd);
    m.impl("rasterize_to_pixels_3dgs_fwd", &gsplat::rasterize_to_pixels_3dgs_fwd);
    m.impl("rasterize_to_pixels_3dgs_bwd", &gsplat::rasterize_to_pixels_3dgs_bwd);
    m.impl("rasterize_to_indices_3dgs", &gsplat::rasterize_to_indices_3dgs);
#endif

#if GSPLAT_BUILD_2DGS
    m.impl("projection_2dgs_fused_fwd", &gsplat::projection_2dgs_fused_fwd);
    m.impl("projection_2dgs_fused_bwd", &gsplat::projection_2dgs_fused_bwd);
    m.impl("projection_2dgs_packed_fwd", &gsplat::projection_2dgs_packed_fwd);
    m.impl("projection_2dgs_packed_bwd", &gsplat::projection_2dgs_packed_bwd);
    m.impl("rasterize_to_pixels_2dgs_fwd", &gsplat::rasterize_to_pixels_2dgs_fwd);
    m.impl("rasterize_to_pixels_2dgs_bwd", &gsplat::rasterize_to_pixels_2dgs_bwd);
    m.impl("rasterize_to_indices_2dgs", &gsplat::rasterize_to_indices_2dgs);
#endif

#if GSPLAT_BUILD_ADAM
    m.impl("adam", &gsplat::adam);
#endif

#if GSPLAT_BUILD_RELOC
    m.impl("relocation", &gsplat::relocation);
#endif

#if GSPLAT_BUILD_3DGUT
    m.impl("projection_ut_3dgs_fused", &gsplat::projection_ut_3dgs_fused);
    m.impl("rasterize_to_pixels_from_world_3dgs_fwd", &gsplat::rasterize_to_pixels_from_world_3dgs_fwd);
    m.impl("rasterize_to_pixels_from_world_3dgs_bwd", &gsplat::rasterize_to_pixels_from_world_3dgs_bwd);
    m.impl("intersect_tile_lidar", &gsplat::intersect_tile_lidar);
#endif
}
