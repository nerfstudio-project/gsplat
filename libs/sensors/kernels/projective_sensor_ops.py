# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-family projective sensor dispatch tables.

This module imports from `kernels.cameras.ops`; the reverse import is forbidden
so per-family wrappers can stay independent of shared dispatch.
"""

from __future__ import annotations

from collections.abc import Callable

from .cameras import ops as camera_ops
from .cameras.types import (
    BivariateWindshieldDistortion,
    NoExternalDistortion,
    OpenCVPinholeProjection,
    script_class_name,
)

DispatchKey = tuple[object, object]

_CAMERA_RAYS_TO_IMAGE_POINTS_BACKENDS: dict[DispatchKey, Callable] = {
    (
        OpenCVPinholeProjection,
        NoExternalDistortion,
    ): camera_ops.camera_rays_to_image_points,
    (
        OpenCVPinholeProjection,
        BivariateWindshieldDistortion,
    ): camera_ops.camera_rays_to_image_points,
}
_IMAGE_POINTS_TO_CAMERA_RAYS_BACKENDS: dict[DispatchKey, Callable] = {
    (
        OpenCVPinholeProjection,
        NoExternalDistortion,
    ): camera_ops.image_points_to_camera_rays,
    (
        OpenCVPinholeProjection,
        BivariateWindshieldDistortion,
    ): camera_ops.image_points_to_camera_rays,
}
_PROJECT_WORLD_POINTS_MEAN_POSE_BACKENDS: dict[DispatchKey, Callable] = {
    (
        OpenCVPinholeProjection,
        NoExternalDistortion,
    ): camera_ops.project_world_points_mean_pose,
    (
        OpenCVPinholeProjection,
        BivariateWindshieldDistortion,
    ): camera_ops.project_world_points_mean_pose,
}
_PROJECT_WORLD_POINTS_SHUTTER_POSE_BACKENDS: dict[DispatchKey, Callable] = {
    (
        OpenCVPinholeProjection,
        NoExternalDistortion,
    ): camera_ops.project_world_points_shutter_pose,
    (
        OpenCVPinholeProjection,
        BivariateWindshieldDistortion,
    ): camera_ops.project_world_points_shutter_pose,
}
_IMAGE_POINTS_TO_WORLD_RAYS_STATIC_POSE_BACKENDS: dict[DispatchKey, Callable] = {
    (
        OpenCVPinholeProjection,
        NoExternalDistortion,
    ): camera_ops.image_points_to_world_rays_static_pose,
    (
        OpenCVPinholeProjection,
        BivariateWindshieldDistortion,
    ): camera_ops.image_points_to_world_rays_static_pose,
}
_IMAGE_POINTS_TO_WORLD_RAYS_SHUTTER_POSE_BACKENDS: dict[DispatchKey, Callable] = {
    (
        OpenCVPinholeProjection,
        NoExternalDistortion,
    ): camera_ops.image_points_to_world_rays_shutter_pose,
    (
        OpenCVPinholeProjection,
        BivariateWindshieldDistortion,
    ): camera_ops.image_points_to_world_rays_shutter_pose,
}
_PIXEL_GRID_TO_WORLD_RAYS_SHUTTER_POSE_BACKENDS: dict[DispatchKey, Callable] = {
    (
        OpenCVPinholeProjection,
        NoExternalDistortion,
    ): camera_ops.pixel_grid_to_world_rays_shutter_pose,
    (
        OpenCVPinholeProjection,
        BivariateWindshieldDistortion,
    ): camera_ops.pixel_grid_to_world_rays_shutter_pose,
}

_DISPATCH_TABLES = {
    "camera_rays_to_image_points": _CAMERA_RAYS_TO_IMAGE_POINTS_BACKENDS,
    "image_points_to_camera_rays": _IMAGE_POINTS_TO_CAMERA_RAYS_BACKENDS,
    "project_world_points_mean_pose": _PROJECT_WORLD_POINTS_MEAN_POSE_BACKENDS,
    "project_world_points_shutter_pose": _PROJECT_WORLD_POINTS_SHUTTER_POSE_BACKENDS,
    "image_points_to_world_rays_static_pose": _IMAGE_POINTS_TO_WORLD_RAYS_STATIC_POSE_BACKENDS,
    "image_points_to_world_rays_shutter_pose": _IMAGE_POINTS_TO_WORLD_RAYS_SHUTTER_POSE_BACKENDS,
    "pixel_grid_to_world_rays_shutter_pose": _PIXEL_GRID_TO_WORLD_RAYS_SHUTTER_POSE_BACKENDS,
}

_REGISTERED_CLASS_NAMES = {
    OpenCVPinholeProjection: "OpenCVPinholeProjection",
    NoExternalDistortion: "NoExternalDistortion",
    BivariateWindshieldDistortion: "BivariateWindshieldDistortion",
}


def _registered_name(cls: object) -> str:
    """Return the registered string name for a projection or distortion class."""
    return _REGISTERED_CLASS_NAMES[cls]


def _lookup(
    table: dict[DispatchKey, Callable], projection: object, external_distortion: object
) -> Callable:
    """Return the backend callable for a (projection, external_distortion) pair.

    Args:
        table: One of the module-level ``_*_BACKENDS`` dispatch dicts.
        projection: Camera projection instance (e.g. ``OpenCVPinholeProjection``).
        external_distortion: External distortion instance; must not be ``None``.

    Returns:
        Callable that implements the operation for the matched pair.

    Raises:
        TypeError: If ``external_distortion`` is ``None`` or the pair is not registered.
    """
    if external_distortion is None:
        raise TypeError(
            "external_distortion=None is not supported; pass NoExternalDistortion() explicitly"
        )
    projection_name = script_class_name(projection)
    distortion_name = script_class_name(external_distortion)
    for (projection_cls, distortion_cls), backend in table.items():
        if projection_name == _registered_name(
            projection_cls
        ) and distortion_name == _registered_name(distortion_cls):
            return backend
    raise TypeError(
        "Unsupported camera projection/distortion pair: "
        f"({projection_name}, {distortion_name})"
    )


def camera_rays_to_image_points(camera_rays, projection, external_distortion, **kwargs):
    """Project camera-space rays to 2-D image points.

    Args:
        camera_rays: ``(..., 3)`` unit-direction tensor in camera space.
        projection: Camera projection parameters.
        external_distortion: External distortion parameters.
        **kwargs: Forwarded verbatim to the family-specific backend.

    Returns:
        Tuple of ``(image_points, valid_mask)`` as returned by the backend.
    """
    return _lookup(
        _CAMERA_RAYS_TO_IMAGE_POINTS_BACKENDS, projection, external_distortion
    )(camera_rays, projection, external_distortion, **kwargs)


def image_points_to_camera_rays(
    image_points, projection, external_distortion, **kwargs
):
    """Unproject 2-D image points to unit-direction rays in camera space.

    Args:
        image_points: ``(..., 2)`` pixel-coordinate tensor.
        projection: Camera projection parameters.
        external_distortion: External distortion parameters.
        **kwargs: Forwarded verbatim to the family-specific backend.

    Returns:
        ``(..., 3)`` unit-direction tensor in camera space.
    """
    return _lookup(
        _IMAGE_POINTS_TO_CAMERA_RAYS_BACKENDS, projection, external_distortion
    )(image_points, projection, external_distortion, **kwargs)


def project_world_points_mean_pose(
    world_points, projection, external_distortion, *args, **kwargs
):
    """Project world-space 3-D points using the midpoint of a dynamic pose.

    Args:
        world_points: ``(N, 3)`` tensor of 3-D world-space positions.
        projection: Camera projection parameters.
        external_distortion: External distortion parameters.
        *args: Additional positional args forwarded to the backend (e.g. dynamic
            pose, resolution).
        **kwargs: Additional keyword args forwarded to the backend.

    Returns:
        Tuple ``(image_points, valid_flags, timestamps_us, pose_t, pose_r)`` as
        returned by the backend; the trailing four entries are ``None`` unless
        the corresponding ``return_*`` keyword is set.
    """
    return _lookup(
        _PROJECT_WORLD_POINTS_MEAN_POSE_BACKENDS, projection, external_distortion
    )(world_points, projection, external_distortion, *args, **kwargs)


def project_world_points_shutter_pose(
    world_points, projection, external_distortion, *args, **kwargs
):
    """Project world-space 3-D points with per-point rolling-shutter poses.

    Args:
        world_points: ``(N, 3)`` tensor of 3-D world-space positions.
        projection: Camera projection parameters.
        external_distortion: External distortion parameters.
        *args: Additional positional args forwarded to the backend (e.g. dynamic pose).
        **kwargs: Additional keyword args forwarded to the backend.

    Returns:
        Tuple ``(image_points, valid_flags, timestamps_us, pose_t, pose_r)`` as
        returned by the backend; the trailing four entries are ``None`` unless
        the corresponding ``return_*`` keyword is set.
    """
    return _lookup(
        _PROJECT_WORLD_POINTS_SHUTTER_POSE_BACKENDS, projection, external_distortion
    )(world_points, projection, external_distortion, *args, **kwargs)


def image_points_to_world_rays_static_pose(
    image_points, projection, external_distortion, *args, **kwargs
):
    """Unproject image points to world-space rays using a static (single) pose.

    Args:
        image_points: ``(N, 2)`` pixel-coordinate tensor.
        projection: Camera projection parameters.
        external_distortion: External distortion parameters.
        *args: Additional positional args forwarded to the backend (e.g. pose).
        **kwargs: Additional keyword args forwarded to the backend.

    Returns:
        Tuple ``(world_rays, timestamps_us, pose_t, pose_r)`` as returned by the
        backend; ``world_rays`` is ``(N, 6)`` packed
        ``[origin_xyz, direction_xyz]`` and the trailing three entries are
        ``None`` unless the corresponding ``return_*`` keyword is set.
    """
    return _lookup(
        _IMAGE_POINTS_TO_WORLD_RAYS_STATIC_POSE_BACKENDS,
        projection,
        external_distortion,
    )(image_points, projection, external_distortion, *args, **kwargs)


def image_points_to_world_rays_shutter_pose(
    image_points, projection, external_distortion, *args, **kwargs
):
    """Unproject image points to world-space rays using per-row rolling-shutter poses.

    Args:
        image_points: ``(N, 2)`` pixel-coordinate tensor.
        projection: Camera projection parameters.
        external_distortion: External distortion parameters.
        *args: Additional positional args forwarded to the backend (e.g. dynamic pose).
        **kwargs: Additional keyword args forwarded to the backend.

    Returns:
        Tuple ``(world_rays, timestamps_us, pose_t, pose_r)`` as returned by the
        backend; ``world_rays`` is ``(N, 6)`` packed
        ``[origin_xyz, direction_xyz]`` and the trailing three entries are
        ``None`` unless the corresponding ``return_*`` keyword is set.
    """
    return _lookup(
        _IMAGE_POINTS_TO_WORLD_RAYS_SHUTTER_POSE_BACKENDS,
        projection,
        external_distortion,
    )(image_points, projection, external_distortion, *args, **kwargs)


def pixel_grid_to_world_rays_shutter_pose(
    projection, external_distortion, *args, **kwargs
):
    """Generate world-space rays for a full pixel grid with rolling-shutter poses.

    Args:
        projection: Camera projection parameters.
        external_distortion: External distortion parameters.
        *args: Additional positional args forwarded to the backend
            (e.g. ``resolution``, ``shutter_type``, ``dynamic_pose``).
        **kwargs: Additional keyword args forwarded to the backend.

    Returns:
        Tuple ``(world_rays, timestamps_us, pose_t, pose_r)`` as returned by the
        backend; ``world_rays`` is ``(H * W, 6)`` packed
        ``[origin_xyz, direction_xyz]`` in row-major pixel order, and the
        trailing three entries are ``None`` unless the corresponding
        ``return_*`` keyword is set.
    """
    return _lookup(
        _PIXEL_GRID_TO_WORLD_RAYS_SHUTTER_POSE_BACKENDS,
        projection,
        external_distortion,
    )(projection, external_distortion, *args, **kwargs)


__all__ = [
    "_DISPATCH_TABLES",
    "camera_rays_to_image_points",
    "image_points_to_camera_rays",
    "image_points_to_world_rays_static_pose",
    "image_points_to_world_rays_shutter_pose",
    "pixel_grid_to_world_rays_shutter_pose",
    "project_world_points_mean_pose",
    "project_world_points_shutter_pose",
]
