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
        .export_values();

    py::enum_<gsplat::extdist::ModelType>(m, "ExternalDistortionModelType", py::module_local())
        .value("BIVARIATE_WINDSHIELD", gsplat::extdist::ModelType::BIVARIATE_WINDSHIELD);

    m.def("null", &gsplat::null);

    // Cameras from 3DGUT
    py::enum_<ShutterType>(m, "ShutterType", py::module_local())
        .value("ROLLING_TOP_TO_BOTTOM", ShutterType::ROLLING_TOP_TO_BOTTOM)
        .value("ROLLING_LEFT_TO_RIGHT", ShutterType::ROLLING_LEFT_TO_RIGHT)
        .value("ROLLING_BOTTOM_TO_TOP", ShutterType::ROLLING_BOTTOM_TO_TOP)
        .value("ROLLING_RIGHT_TO_LEFT", ShutterType::ROLLING_RIGHT_TO_LEFT)
        .value("GLOBAL", ShutterType::GLOBAL);

    py::class_<UnscentedTransformParameters>(m, "UnscentedTransformParameters", py::module_local())
        .def(py::init<>())
        .def(py::init([](
            float alpha, float beta, float kappa, float in_image_margin_factor, bool require_all_sigma_points_valid) {
                return UnscentedTransformParameters {alpha, beta, kappa, in_image_margin_factor, require_all_sigma_points_valid};
            }),
            "Dataclass constructor for UnscentedTransformParameters",
            py::arg("alpha") = 0.1,
            py::arg("beta") = 2.0,
            py::arg("kappa") = 0.0,
            py::arg("in_image_margin_factor") = 0.1,
            py::arg("require_all_sigma_points_valid") = true)
        .def_readwrite("alpha", &UnscentedTransformParameters::alpha)
        .def_readwrite("beta", &UnscentedTransformParameters::beta)
        .def_readwrite("kappa", &UnscentedTransformParameters::kappa)
        .def_readwrite("in_image_margin_factor", &UnscentedTransformParameters::in_image_margin_factor)
        .def_readwrite("require_all_sigma_points_valid", &UnscentedTransformParameters::require_all_sigma_points_valid);

    // FTheta Camera support
    py::enum_<FThetaCameraDistortionParameters::PolynomialType>(m, "FThetaPolynomialType", py::module_local())
        .value("PIXELDIST_TO_ANGLE", FThetaCameraDistortionParameters::PolynomialType::PIXELDIST_TO_ANGLE)
        .value("ANGLE_TO_PIXELDIST", FThetaCameraDistortionParameters::PolynomialType::ANGLE_TO_PIXELDIST);

    py::class_<FThetaCameraDistortionParameters>(m, "FThetaCameraDistortionParameters", py::module_local())
        .def(py::init<>())
        .def(py::init([](
            const FThetaCameraDistortionParameters::PolynomialType& reference_poly,
            const std::array<float, FThetaCameraDistortionParameters::PolynomialDegree>& pixeldist_to_angle_poly,
            const std::array<float, FThetaCameraDistortionParameters::PolynomialDegree>& angle_to_pixeldist_poly,
            float max_angle,
            const std::array<float, 3>& linear_cde) {
                return FThetaCameraDistortionParameters {reference_poly, pixeldist_to_angle_poly, angle_to_pixeldist_poly, max_angle, linear_cde};
            }),
            "Dataclass constructor for FThetaCameraDistortionParameters",
            py::arg("reference_poly"),
            py::arg("pixeldist_to_angle_poly"),
            py::arg("angle_to_pixeldist_poly"),
            py::arg("max_angle"),
            py::arg("linear_cde"))
        .def_readwrite("reference_poly", &FThetaCameraDistortionParameters::reference_poly)
        .def_readwrite("pixeldist_to_angle_poly", &FThetaCameraDistortionParameters::pixeldist_to_angle_poly)
        .def_readwrite("angle_to_pixeldist_poly", &FThetaCameraDistortionParameters::angle_to_pixeldist_poly)
        .def_readwrite("max_angle", &FThetaCameraDistortionParameters::max_angle)
        .def_readwrite("linear_cde", &FThetaCameraDistortionParameters::linear_cde);

    // External Distortion support
    py::enum_<gsplat::extdist::ReferencePolynomialType>(m, "ExternalDistortionReferencePolynomial", py::module_local())
        .value("FORWARD", gsplat::extdist::ReferencePolynomialType::FORWARD)
        .value("BACKWARD", gsplat::extdist::ReferencePolynomialType::BACKWARD);
    
    py::class_<gsplat::extdist::BivariateWindshieldModelParameters>(m, "BivariateWindshieldModelParameters", py::module_local())
        .def(py::init<>())
        .def_readonly_static("MAX_ORDER", &gsplat::extdist::BivariateWindshieldModelParameters::MAX_ORDER)
        .def_readonly_static("MAX_COEFFS", &gsplat::extdist::BivariateWindshieldModelParameters::MAX_COEFFS)
        .def_readwrite("reference_poly", &gsplat::extdist::BivariateWindshieldModelParameters::reference_poly)
        .def_readwrite("horizontal_poly", &gsplat::extdist::BivariateWindshieldModelParameters::horizontal_poly)
        .def_readwrite("vertical_poly", &gsplat::extdist::BivariateWindshieldModelParameters::vertical_poly)
        .def_readwrite("horizontal_poly_inverse", &gsplat::extdist::BivariateWindshieldModelParameters::horizontal_poly_inverse)
        .def_readwrite("vertical_poly_inverse", &gsplat::extdist::BivariateWindshieldModelParameters::vertical_poly_inverse);

    // ==================== Camera Model Bindings ====================

#if BUILD_CAMERA_WRAPPERS
    py::class_<gsplat::PyBaseCameraModel<>, std::shared_ptr<gsplat::PyBaseCameraModel<>>>(m, "BaseCameraModel", py::module_local())
        .def("camera_ray_to_image_point", &gsplat::PyBaseCameraModel<>::camera_ray_to_image_point,
             py::arg("camera_ray"), py::arg("margin_factor") = 0.0f)
        .def("image_point_to_camera_ray", &gsplat::PyBaseCameraModel<>::image_point_to_camera_ray,
             py::arg("image_points"))
        .def("shutter_relative_frame_time", &gsplat::PyBaseCameraModel<>::shutter_relative_frame_time,
             py::arg("image_points"))
        .def("image_point_to_world_ray_shutter_pose",
             &gsplat::PyBaseCameraModel<>::image_point_to_world_ray_shutter_pose,
             py::arg("image_points"),
             py::arg("pose_start"), py::arg("pose_end"))
        .def("world_point_to_image_point_shutter_pose",
             &gsplat::PyBaseCameraModel<>::world_point_to_image_point_shutter_pose,
             py::arg("world_points"),
             py::arg("pose_start"), py::arg("pose_end"),
             py::arg("margin_factor") = 0.0f)
        .def_property_readonly("width", &gsplat::PyBaseCameraModel<>::width)
        .def_property_readonly("height", &gsplat::PyBaseCameraModel<>::height)
        .def_property_readonly("rs_type", &gsplat::PyBaseCameraModel<>::rs_type)
        .def_property_readonly("principal_points", &gsplat::PyBaseCameraModel<>::principal_points)
        .def_property_readonly("focal_lengths", &gsplat::PyBaseCameraModel<>::focal_lengths)
        .def_static("create", &gsplat::PyBaseCameraModel<>::create,
             py::arg("width"),
             py::arg("height"),
             py::arg("camera_model"),
             py::arg("principal_points"),
             py::arg("focal_lengths") = std::nullopt,
             py::arg("radial_coeffs") = std::nullopt,
             py::arg("tangential_coeffs") = std::nullopt,
             py::arg("thin_prism_coeffs") = std::nullopt,
             py::arg("ftheta_coeffs") = std::nullopt,
             py::arg("rs_type") = ShutterType::GLOBAL);

    py::class_<gsplat::PyPerfectPinholeCameraModel, gsplat::PyBaseCameraModel<>,
               std::shared_ptr<gsplat::PyPerfectPinholeCameraModel>>(m, "PerfectPinholeCameraModel", py::module_local())
        .def(py::init<int, int, const torch::Tensor&, const torch::Tensor&, ShutterType>(),
             py::arg("width"), py::arg("height"),
             py::arg("focal_lengths"), py::arg("principal_points"),
             py::arg("rs_type"));

    py::class_<gsplat::PyOpenCVPinholeCameraModel, gsplat::PyBaseCameraModel<>,
               std::shared_ptr<gsplat::PyOpenCVPinholeCameraModel>>(m, "OpenCVPinholeCameraModel", py::module_local())
        .def(py::init<int, int, const torch::Tensor&, const torch::Tensor&,
                      std::optional<torch::Tensor>, std::optional<torch::Tensor>,
                      std::optional<torch::Tensor>, ShutterType>(),
             py::arg("width"), py::arg("height"),
             py::arg("focal_lengths"), py::arg("principal_points"),
             py::arg("radial_coeffs") = std::nullopt,
             py::arg("tangential_coeffs") = std::nullopt,
             py::arg("thin_prism_coeffs") = std::nullopt,
             py::arg("rs_type"));

    py::class_<gsplat::PyOpenCVFisheyeCameraModel, gsplat::PyBaseCameraModel<>,
               std::shared_ptr<gsplat::PyOpenCVFisheyeCameraModel>>(m, "OpenCVFisheyeCameraModel", py::module_local())
        .def(py::init<int, int, const torch::Tensor&, const torch::Tensor&,
                      std::optional<torch::Tensor>, ShutterType>(),
             py::arg("width"), py::arg("height"),
             py::arg("focal_lengths"), py::arg("principal_points"),
             py::arg("radial_coeffs") = std::nullopt,
             py::arg("rs_type"));

    py::class_<gsplat::PyFThetaCameraModel, gsplat::PyBaseCameraModel<>,
               std::shared_ptr<gsplat::PyFThetaCameraModel>>(m, "FThetaCameraModel", py::module_local())
        .def(py::init<int, int, const torch::Tensor&, const torch::Tensor&,
                      const torch::Tensor&, const torch::Tensor&, FThetaCameraDistortionParameters::PolynomialType, const torch::Tensor &, ShutterType>(),
             py::arg("width"), py::arg("height"),
             py::arg("principal_points"),
             py::arg("pixeldist_to_angle_poly"),
             py::arg("angle_to_pixeldist_poly"),
             py::arg("linear_cde"),
             py::arg("reference_poly"),
             py::arg("max_angle"),
             py::arg("rs_type"));
#endif
}

TORCH_LIBRARY(gsplat, m) {
#if GSPLAT_BUILD_3DGS
    m.def("quat_scale_to_covar_preci_fwd(Tensor quats, Tensor scales, bool compute_covar, bool compute_preci, bool triu) -> (Tensor, Tensor)");
    m.def("quat_scale_to_covar_preci_bwd(Tensor quats, Tensor scales, bool triu, Tensor? v_covars, Tensor? v_precis) -> (Tensor, Tensor)");
#endif

    m.def("spherical_harmonics_fwd(int degrees_to_use, Tensor dirs, Tensor coeffs, Tensor? masks) -> Tensor");
    m.def("spherical_harmonics_bwd(int K, int degrees_to_use, Tensor dirs, Tensor coeffs, Tensor? masks, Tensor v_colors, bool compute_v_dirs) -> (Tensor, Tensor)");

    m.def("intersect_tile(Tensor means2d, Tensor radii, Tensor depths, Tensor? image_ids, Tensor? gaussian_ids, int I, int tile_size, int tile_width, int tile_height, bool sort, bool segmented) -> (Tensor, Tensor, Tensor)");
    m.def("intersect_offset(Tensor isect_ids, int I, int tile_width, int tile_height) -> Tensor");

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
    m.def("projection_ut_3dgs_fused(Tensor means, Tensor quats, Tensor scales, Tensor? opacities, Tensor viewmats0, Tensor? viewmats1, Tensor Ks, int image_width, int image_height, float eps2d, float near_plane, float far_plane, float radius_clip, bool calc_compensations, int camera_model, bool global_z_order, float alpha, float beta, float kappa, float in_image_margin_factor, bool require_all_sigma_points_valid, int rs_type, Tensor? radial_coeffs, Tensor? tangential_coeffs, Tensor? thin_prism_coeffs, int reference_poly, float[6] pixeldist_to_angle_poly, float[6] angle_to_pixeldist_poly, float max_angle, float[3] linear_cde, int external_reference_poly, Tensor[] external_distortion_params) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("rasterize_to_pixels_from_world_3dgs_fwd(Tensor means, Tensor quats, Tensor scales, Tensor colors, Tensor opacities, Tensor? backgrounds, Tensor? masks, int image_width, int image_height, int tile_size, Tensor viewmats0, Tensor? viewmats1, Tensor Ks, int camera_model, float alpha, float beta, float kappa, float in_image_margin_factor, bool require_all_sigma_points_valid, int rs_type, Tensor? rays, Tensor? radial_coeffs, Tensor? tangential_coeffs, Tensor? thin_prism_coeffs, int ftheta_reference_poly, float[6] pixeldist_to_angle_poly, float[6] angle_to_pixeldist_poly, float max_angle, float[3] linear_cde, int external_reference_poly, Tensor[] external_distortion_params, Tensor tile_offsets, Tensor flatten_ids, bool use_hit_distance, Tensor? sample_counts, Tensor? normals) -> (Tensor, Tensor, Tensor)");
    m.def("rasterize_to_pixels_from_world_3dgs_bwd(Tensor means, Tensor quats, Tensor scales, Tensor colors, Tensor opacities, Tensor? backgrounds, Tensor? masks, int image_width, int image_height, int tile_size, Tensor viewmats0, Tensor? viewmats1, Tensor Ks, int camera_model, float alpha, float beta, float kappa, float in_image_margin_factor, bool require_all_sigma_points_valid, int rs_type, Tensor? rays, Tensor? radial_coeffs, Tensor? tangential_coeffs, Tensor? thin_prism_coeffs, int ftheta_reference_poly, float[6] pixeldist_to_angle_poly, float[6] angle_to_pixeldist_poly, float max_angle, float[3] linear_cde, int external_reference_poly, Tensor[] external_distortion_params, Tensor tile_offsets, Tensor flatten_ids, bool use_hit_distance, Tensor render_alphas, Tensor last_ids, Tensor v_render_colors, Tensor v_render_alphas, Tensor? v_render_normals) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor?)");
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
#endif
}
