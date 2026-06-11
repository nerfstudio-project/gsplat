# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from gsplat.sensors.kernels._backend import _SENSORS_CUDA as _C

_TENSOR = "Tensor"
_INT = "int"
_FLOAT = "float"
_BOOL = "bool"
_FTHETA_PROJECTION = "FThetaProjection"
_FISHEYE_PROJECTION = "OpenCVFisheyeProjection"
_NO_EXTERNAL = "NoExternalDistortion"
_BIVARIATE = "BivariateWindshieldDistortion"
_CUSTOM_CLASSES = {
    _FTHETA_PROJECTION,
    _FISHEYE_PROJECTION,
    _NO_EXTERNAL,
    _BIVARIATE,
}
_PRIMITIVE_TYPES = {_TENSOR, _INT, _FLOAT, _BOOL}


def _normalize_type(type_):
    type_name = str(type_)
    if type_name in _PRIMITIVE_TYPES:
        return type_name
    for class_name in _CUSTOM_CLASSES:
        if type_name.endswith(f".{class_name}"):
            return class_name
    return type_name


def _flatten_return_types(return_type):
    type_name = str(return_type)
    if type_name.startswith("(") and type_name.endswith(")"):
        return [
            _normalize_type(part.strip())
            for part in type_name[1:-1].split(",")
            if part.strip()
        ]
    return [_normalize_type(return_type)]


def _schema_signature(schema):
    arg_names = [arg.name for arg in schema.arguments]
    arg_types = [_normalize_type(arg.type) for arg in schema.arguments]
    returns = [
        type_
        for return_value in schema.returns
        for type_ in _flatten_return_types(return_value.type)
    ]
    return arg_names, arg_types, returns


def _has_stable_arg_names(arg_names):
    return all(arg_names) and not all(
        name.startswith("arg") or name.startswith("_") for name in arg_names
    )


def _forward_arg_names(name):
    if name.startswith("camera_rays_to_image_points_"):
        return ["projection", "external_distortion", "camera_rays"]
    if name.startswith("image_points_to_camera_rays_"):
        return ["projection", "external_distortion", "image_points"]
    if name.startswith("project_world_points_mean_pose_"):
        return [
            "projection",
            "external_distortion",
            "world_points",
            "start_translation",
            "start_rotation",
            "end_translation",
            "end_rotation",
            "start_timestamp_us",
            "end_timestamp_us",
        ]
    if name.startswith("project_world_points_shutter_pose_"):
        return [
            "projection",
            "external_distortion",
            "world_points",
            "start_translation",
            "start_rotation",
            "end_translation",
            "end_rotation",
            "width",
            "height",
            "shutter_type",
            "start_timestamp_us",
            "end_timestamp_us",
            "max_iterations",
            "stop_mean_error_px",
            "stop_delta_mean_error_px",
            "initial_relative_time",
        ]
    if name.startswith("image_points_to_world_rays_static_pose_"):
        return [
            "projection",
            "external_distortion",
            "image_points",
            "translations",
            "rotations",
            "timestamp_us",
        ]
    if name.startswith("image_points_to_world_rays_shutter_pose_"):
        return [
            "projection",
            "external_distortion",
            "image_points",
            "start_translation",
            "start_rotation",
            "end_translation",
            "end_rotation",
            "width",
            "height",
            "shutter_type",
            "start_timestamp_us",
            "end_timestamp_us",
        ]
    raise AssertionError(f"no forward arg-name rule for {name}")


def _intrinsic_backward_names(name):
    if "_ftheta_" in name:
        return [
            "need_principal_point_grad",
            "need_fw_poly_grad",
            "need_bw_poly_grad",
            "need_A_grad",
            "need_Ainv_grad",
        ]
    return [
        "need_principal_point_grad",
        "need_focal_length_grad",
        "need_forward_poly_grad",
    ]


def _backward_arg_names(name):
    include_distortion = "_bivariate_windshield_backward" in name
    distortion_names = ["need_distortion_coeffs_grad"] if include_distortion else []
    intrinsic_names = _intrinsic_backward_names(name)

    if name.startswith("camera_rays_to_image_points_"):
        return [
            "projection",
            "external_distortion",
            "camera_rays",
            "grad_image_points",
            "scratch",
            "need_camera_ray_grad",
            *intrinsic_names,
            *distortion_names,
        ]
    if name.startswith("image_points_to_camera_rays_"):
        return [
            "projection",
            "external_distortion",
            "image_points",
            "grad_camera_rays",
            "scratch",
            "need_image_point_grad",
            *intrinsic_names,
            *distortion_names,
        ]
    if name.startswith("project_world_points_mean_pose_"):
        return [
            "projection",
            "external_distortion",
            "world_points",
            "start_rotation",
            "end_rotation",
            "grad_image_points",
            "scratch",
            "need_world_point_grad",
            "need_start_translation_grad",
            "need_end_translation_grad",
            "need_start_rotation_grad",
            "need_end_rotation_grad",
            *intrinsic_names,
            *distortion_names,
        ]
    if name.startswith("image_points_to_world_rays_static_pose_"):
        return [
            "projection",
            "external_distortion",
            "image_points",
            "translations",
            "rotations",
            "grad_world_rays",
            "scratch",
            "need_image_point_grad",
            "need_translation_grad",
            "need_rotation_grad",
            *intrinsic_names,
            *distortion_names,
        ]
    if name.startswith("project_world_points_shutter_pose_ftheta_"):
        return [
            "projection",
            "external_distortion",
            "world_points",
            "start_rotation",
            "end_rotation",
            "shutter_type",
            "max_iterations",
            "initial_relative_time",
            "valid_flags",
            "grad_image_points",
            "scratch",
            "need_world_point_grad",
            "need_start_translation_grad",
            "need_end_translation_grad",
            "need_start_rotation_grad",
            "need_end_rotation_grad",
            *intrinsic_names,
            *distortion_names,
        ]
    if name.startswith("project_world_points_shutter_pose_opencv_fisheye_"):
        return [
            "projection",
            "external_distortion",
            "start_rotation",
            "end_rotation",
            "grad_image_points",
            "scratch",
            "need_world_point_grad",
            "need_start_translation_grad",
            "need_end_translation_grad",
            "need_start_rotation_grad",
            "need_end_rotation_grad",
            *intrinsic_names,
            *distortion_names,
        ]
    if name.startswith("image_points_to_world_rays_shutter_pose_"):
        return [
            "projection",
            "external_distortion",
            "image_points",
            "start_rotation",
            "end_rotation",
            "shutter_type",
            "grad_world_rays",
            "scratch",
            "need_image_point_grad",
            "need_start_translation_grad",
            "need_end_translation_grad",
            "need_start_rotation_grad",
            "need_end_rotation_grad",
            *intrinsic_names,
            *distortion_names,
        ]
    raise AssertionError(f"no backward arg-name rule for {name}")


def _expected_arg_names(name):
    if name.endswith("_backward"):
        return _backward_arg_names(name)
    return _forward_arg_names(name)


def _assert_schema(name, expected_args, expected_returns):
    schema = getattr(torch.ops.gsplat_sensors, name).default._schema
    assert schema.name == f"gsplat_sensors::{name}"
    actual_names, actual_types, actual_returns = _schema_signature(schema)
    expected_return_types = [_TENSOR] * expected_returns
    # Some builds expose placeholder argument names; only enforce names when the
    # schema preserves them.
    if _has_stable_arg_names(actual_names):
        assert actual_names == _expected_arg_names(name)
    assert actual_types == expected_args
    assert actual_returns == expected_return_types


def test_extension_loads():
    """The native extension must load so Torch ops and classes can register."""
    assert _C is not None


def test_required_torch_ops_registered():
    """Keep the native op registration surface stable for Python callers."""
    required = [
        "generate_image_points",
        "camera_rays_to_image_points_opencv_pinhole_no_external",
        "camera_rays_to_image_points_opencv_pinhole_bivariate_windshield",
        "image_points_to_camera_rays_opencv_pinhole_no_external",
        "image_points_to_camera_rays_opencv_pinhole_bivariate_windshield",
        "project_world_points_mean_pose_opencv_pinhole_no_external",
        "project_world_points_mean_pose_opencv_pinhole_bivariate_windshield",
        "project_world_points_shutter_pose_opencv_pinhole_no_external",
        "project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield",
        "image_points_to_world_rays_static_pose_opencv_pinhole_no_external",
        "image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield",
        "image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external",
        "image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield",
        "camera_rays_to_image_points_ftheta_no_external",
        "camera_rays_to_image_points_ftheta_bivariate_windshield",
        "image_points_to_camera_rays_ftheta_no_external",
        "image_points_to_camera_rays_ftheta_bivariate_windshield",
        "project_world_points_mean_pose_ftheta_no_external",
        "project_world_points_mean_pose_ftheta_bivariate_windshield",
        "project_world_points_shutter_pose_ftheta_no_external",
        "project_world_points_shutter_pose_ftheta_bivariate_windshield",
        "image_points_to_world_rays_static_pose_ftheta_no_external",
        "image_points_to_world_rays_static_pose_ftheta_bivariate_windshield",
        "image_points_to_world_rays_shutter_pose_ftheta_no_external",
        "image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield",
        "sensor_rays_to_sensor_angles",
        "sensor_rays_to_sensor_angles_backward",
        "sensor_angles_to_sensor_rays",
        "sensor_angles_to_sensor_rays_backward",
        "elements_to_sensor_angles",
        "elements_to_sensor_angles_backward",
        "generate_spinning_lidar_rays",
        "generate_spinning_lidar_rays_backward",
        "inverse_project_spinning_lidar",
        "inverse_project_spinning_lidar_backward",
        "camera_rays_to_image_points_opencv_fisheye_no_external",
        "camera_rays_to_image_points_opencv_fisheye_bivariate_windshield",
        "image_points_to_camera_rays_opencv_fisheye_no_external",
        "image_points_to_camera_rays_opencv_fisheye_bivariate_windshield",
        "project_world_points_mean_pose_opencv_fisheye_no_external",
        "project_world_points_mean_pose_opencv_fisheye_bivariate_windshield",
        "project_world_points_shutter_pose_opencv_fisheye_no_external",
        "project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield",
        "image_points_to_world_rays_static_pose_opencv_fisheye_no_external",
        "image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield",
        "image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external",
        "image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield",
    ]
    for name in required:
        assert hasattr(torch.ops.gsplat_sensors, name)


def _forward_schema_cases(projection, distortion, sensor_suffix):
    return [
        (
            f"camera_rays_to_image_points_{sensor_suffix}",
            [projection, distortion, _TENSOR],
            3,
        ),
        (
            f"image_points_to_camera_rays_{sensor_suffix}",
            [projection, distortion, _TENSOR],
            2,
        ),
        (
            f"project_world_points_mean_pose_{sensor_suffix}",
            [
                projection,
                distortion,
                _TENSOR,
                _TENSOR,
                _TENSOR,
                _TENSOR,
                _TENSOR,
                _INT,
                _INT,
            ],
            6,
        ),
        (
            f"project_world_points_shutter_pose_{sensor_suffix}",
            [
                projection,
                distortion,
                _TENSOR,
                _TENSOR,
                _TENSOR,
                _TENSOR,
                _TENSOR,
                _INT,
                _INT,
                _INT,
                _INT,
                _INT,
                _INT,
                _FLOAT,
                _FLOAT,
                _FLOAT,
            ],
            6,
        ),
        (
            f"image_points_to_world_rays_static_pose_{sensor_suffix}",
            [projection, distortion, _TENSOR, _TENSOR, _TENSOR, _INT],
            5,
        ),
        (
            f"image_points_to_world_rays_shutter_pose_{sensor_suffix}",
            [
                projection,
                distortion,
                _TENSOR,
                _TENSOR,
                _TENSOR,
                _TENSOR,
                _TENSOR,
                _INT,
                _INT,
                _INT,
                _INT,
                _INT,
            ],
            5,
        ),
    ]


_FTHETA_NO_EXTERNAL_BACKWARD = [
    (
        "camera_rays_to_image_points_ftheta_no_external_backward",
        [_FTHETA_PROJECTION, _NO_EXTERNAL, _TENSOR, _TENSOR, _TENSOR] + [_BOOL] * 6,
        6,
    ),
    (
        "image_points_to_camera_rays_ftheta_no_external_backward",
        [_FTHETA_PROJECTION, _NO_EXTERNAL, _TENSOR, _TENSOR, _TENSOR] + [_BOOL] * 6,
        6,
    ),
    (
        "project_world_points_mean_pose_ftheta_no_external_backward",
        [_FTHETA_PROJECTION, _NO_EXTERNAL, _TENSOR, _TENSOR, _TENSOR, _TENSOR, _TENSOR]
        + [_BOOL] * 10,
        10,
    ),
    (
        "image_points_to_world_rays_static_pose_ftheta_no_external_backward",
        [_FTHETA_PROJECTION, _NO_EXTERNAL, _TENSOR, _TENSOR, _TENSOR, _TENSOR, _TENSOR]
        + [_BOOL] * 8,
        8,
    ),
    (
        "project_world_points_shutter_pose_ftheta_no_external_backward",
        [
            _FTHETA_PROJECTION,
            _NO_EXTERNAL,
            _TENSOR,
            _TENSOR,
            _TENSOR,
            _INT,
            _INT,
            _FLOAT,
            _TENSOR,
            _TENSOR,
            _TENSOR,
        ]
        + [_BOOL] * 10,
        10,
    ),
    (
        "image_points_to_world_rays_shutter_pose_ftheta_no_external_backward",
        [
            _FTHETA_PROJECTION,
            _NO_EXTERNAL,
            _TENSOR,
            _TENSOR,
            _TENSOR,
            _INT,
            _TENSOR,
            _TENSOR,
        ]
        + [_BOOL] * 10,
        10,
    ),
]


_FTHETA_BIVARIATE_BACKWARD = [
    (
        "camera_rays_to_image_points_ftheta_bivariate_windshield_backward",
        [_FTHETA_PROJECTION, _BIVARIATE, _TENSOR, _TENSOR, _TENSOR] + [_BOOL] * 7,
        7,
    ),
    (
        "image_points_to_camera_rays_ftheta_bivariate_windshield_backward",
        [_FTHETA_PROJECTION, _BIVARIATE, _TENSOR, _TENSOR, _TENSOR] + [_BOOL] * 7,
        7,
    ),
    (
        "project_world_points_mean_pose_ftheta_bivariate_windshield_backward",
        [_FTHETA_PROJECTION, _BIVARIATE, _TENSOR, _TENSOR, _TENSOR, _TENSOR, _TENSOR]
        + [_BOOL] * 11,
        11,
    ),
    (
        "image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_backward",
        [_FTHETA_PROJECTION, _BIVARIATE, _TENSOR, _TENSOR, _TENSOR, _TENSOR, _TENSOR]
        + [_BOOL] * 9,
        9,
    ),
    (
        "project_world_points_shutter_pose_ftheta_bivariate_windshield_backward",
        [
            _FTHETA_PROJECTION,
            _BIVARIATE,
            _TENSOR,
            _TENSOR,
            _TENSOR,
            _INT,
            _INT,
            _FLOAT,
            _TENSOR,
            _TENSOR,
            _TENSOR,
        ]
        + [_BOOL] * 11,
        11,
    ),
    (
        "image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_backward",
        [
            _FTHETA_PROJECTION,
            _BIVARIATE,
            _TENSOR,
            _TENSOR,
            _TENSOR,
            _INT,
            _TENSOR,
            _TENSOR,
        ]
        + [_BOOL] * 11,
        11,
    ),
]


_FISHEYE_NO_EXTERNAL_BACKWARD = [
    (
        "camera_rays_to_image_points_opencv_fisheye_no_external_backward",
        [_FISHEYE_PROJECTION, _NO_EXTERNAL, _TENSOR, _TENSOR, _TENSOR] + [_BOOL] * 4,
        4,
    ),
    (
        "image_points_to_camera_rays_opencv_fisheye_no_external_backward",
        [_FISHEYE_PROJECTION, _NO_EXTERNAL, _TENSOR, _TENSOR, _TENSOR] + [_BOOL] * 4,
        4,
    ),
    (
        "project_world_points_mean_pose_opencv_fisheye_no_external_backward",
        [_FISHEYE_PROJECTION, _NO_EXTERNAL, _TENSOR, _TENSOR, _TENSOR, _TENSOR, _TENSOR]
        + [_BOOL] * 8,
        8,
    ),
    (
        "image_points_to_world_rays_static_pose_opencv_fisheye_no_external_backward",
        [_FISHEYE_PROJECTION, _NO_EXTERNAL, _TENSOR, _TENSOR, _TENSOR, _TENSOR, _TENSOR]
        + [_BOOL] * 6,
        6,
    ),
    (
        "project_world_points_shutter_pose_opencv_fisheye_no_external_backward",
        [_FISHEYE_PROJECTION, _NO_EXTERNAL, _TENSOR, _TENSOR, _TENSOR, _TENSOR]
        + [_BOOL] * 8,
        8,
    ),
    (
        "image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external_backward",
        [_FISHEYE_PROJECTION, _NO_EXTERNAL, _TENSOR, _TENSOR, _TENSOR, _TENSOR, _TENSOR]
        + [_BOOL] * 8,
        8,
    ),
]


_FISHEYE_BIVARIATE_BACKWARD = [
    (
        "camera_rays_to_image_points_opencv_fisheye_bivariate_windshield_backward",
        [_FISHEYE_PROJECTION, _BIVARIATE, _TENSOR, _TENSOR, _TENSOR] + [_BOOL] * 5,
        5,
    ),
    (
        "image_points_to_camera_rays_opencv_fisheye_bivariate_windshield_backward",
        [_FISHEYE_PROJECTION, _BIVARIATE, _TENSOR, _TENSOR, _TENSOR] + [_BOOL] * 5,
        5,
    ),
    (
        "project_world_points_mean_pose_opencv_fisheye_bivariate_windshield_backward",
        [_FISHEYE_PROJECTION, _BIVARIATE, _TENSOR, _TENSOR, _TENSOR, _TENSOR, _TENSOR]
        + [_BOOL] * 9,
        9,
    ),
    (
        "image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield_backward",
        [_FISHEYE_PROJECTION, _BIVARIATE, _TENSOR, _TENSOR, _TENSOR, _TENSOR, _TENSOR]
        + [_BOOL] * 7,
        7,
    ),
    (
        "project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield_backward",
        [_FISHEYE_PROJECTION, _BIVARIATE, _TENSOR, _TENSOR, _TENSOR, _TENSOR]
        + [_BOOL] * 9,
        9,
    ),
    (
        "image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield_backward",
        [_FISHEYE_PROJECTION, _BIVARIATE, _TENSOR, _TENSOR, _TENSOR, _TENSOR, _TENSOR]
        + [_BOOL] * 9,
        9,
    ),
]


_SCHEMA_SNAPSHOT_CASES = (
    _forward_schema_cases(_FTHETA_PROJECTION, _NO_EXTERNAL, "ftheta_no_external")
    + _forward_schema_cases(
        _FTHETA_PROJECTION, _BIVARIATE, "ftheta_bivariate_windshield"
    )
    + _FTHETA_NO_EXTERNAL_BACKWARD
    + _FTHETA_BIVARIATE_BACKWARD
    + _forward_schema_cases(
        _FISHEYE_PROJECTION, _NO_EXTERNAL, "opencv_fisheye_no_external"
    )
    + _forward_schema_cases(
        _FISHEYE_PROJECTION, _BIVARIATE, "opencv_fisheye_bivariate_windshield"
    )
    + _FISHEYE_NO_EXTERNAL_BACKWARD
    + _FISHEYE_BIVARIATE_BACKWARD
)


@pytest.mark.parametrize(
    "name, expected_args, expected_returns", _SCHEMA_SNAPSHOT_CASES
)
def test_ftheta_and_fisheye_torch_op_schemas(name, expected_args, expected_returns):
    """Snapshot FTheta/fisheye op schemas so ABI changes are intentional."""
    _assert_schema(name, expected_args, expected_returns)


def test_bivariate_windshield_class_registered():
    """Bivariate distortion must remain constructible from TorchScript/Python bindings."""
    assert hasattr(torch.classes.gsplat_sensors, "BivariateWindshieldDistortion")


def test_ftheta_projection_class_registered():
    """FTheta projection must remain available as a TorchScript custom class."""
    assert hasattr(torch.classes.gsplat_sensors, "FThetaProjection")


def test_row_offset_spinning_lidar_projection_class_registered():
    """Row-offset lidar projection must remain available through TorchScript bindings."""
    assert hasattr(
        torch.classes.gsplat_sensors, "RowOffsetStructuredSpinningLidarProjection"
    )


def test_spinning_direction_enum_registered():
    """SpinningDirection integer values are part of the native/Python contract."""
    assert hasattr(_C, "SpinningDirection")
    assert int(_C.SpinningDirection.CLOCKWISE) == 0
    assert int(_C.SpinningDirection.COUNTERCLOCKWISE) == 1


def test_opencv_fisheye_projection_class_registered():
    """OpenCV fisheye projection must remain available as a TorchScript custom class."""
    assert hasattr(torch.classes.gsplat_sensors, "OpenCVFisheyeProjection")
