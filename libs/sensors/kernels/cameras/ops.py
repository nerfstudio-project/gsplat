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

"""Layer 0 Python bindings for camera kernel operations.

Layer 0 GPU operations for camera projection. Provides differentiable PyTorch
entry points around the hand-written CUDA camera kernels; each public function
is backed by a ``torch.autograd.Function`` so forward and backward CUDA kernels
are invoked transparently. All functions are differentiable and support PyTorch
autograd.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .._backend import _C  # noqa: F401  # ensures torch.ops registrations exist
from ..common.pose import DynamicPose, Pose, Trajectory
from .types import (
    BivariateWindshieldDistortion,
    CameraProjection,
    ExternalDistortion,
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    REGISTERED_CAMERA_PROJECTION_NAMES,
    REGISTERED_DISTORTION_NAMES,
    ShutterType,
    script_class_name,
)


def _raise_or_target_device(
    tensor: Tensor, allow_device_transfer: bool
) -> torch.device:
    """Return a CUDA device for the given tensor, raising if transfer is disallowed.

    Args:
        tensor: Input tensor whose device is inspected.
        allow_device_transfer: If False, raises when the tensor is not already on CUDA.
    """
    if tensor.device.type == "cuda":
        return tensor.device
    if not allow_device_transfer:
        raise RuntimeError(
            "CPU inputs require allow_device_transfer=True before gsplat_sensors camera ops launch."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("gsplat_sensors camera ops require CUDA")
    return torch.device("cuda")


def _to_dev(
    tensor: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool,
) -> Tensor:
    """Move a tensor to the target device and dtype, ensuring contiguity.

    Args:
        tensor: Source tensor.
        device: Target CUDA device.
        dtype: Target floating-point dtype.
        allow_device_transfer: If False, raises when the tensor needs a device or
            dtype transfer before the kernel can launch.
    """
    target_device = device
    if device.type == "cuda" and device.index is None and tensor.device.type == "cuda":
        target_device = tensor.device
    needs_transfer = tensor.device != target_device or tensor.dtype != dtype
    if needs_transfer and not allow_device_transfer:
        raise RuntimeError(
            f"Tensor on {tensor.device} (dtype={tensor.dtype}) requires transfer "
            f"to {target_device} (dtype={dtype}). Set allow_device_transfer=True to allow "
            "implicit conversion before gsplat_sensors camera ops launch."
        )
    if not needs_transfer and tensor.is_contiguous():
        return tensor
    converted = (
        tensor.to(device=target_device, dtype=dtype) if needs_transfer else tensor
    )
    return converted if converted.is_contiguous() else converted.contiguous()


_SUPPORTED_PROJECTIONS = frozenset(REGISTERED_CAMERA_PROJECTION_NAMES)
_SUPPORTED_DISTORTIONS = frozenset(REGISTERED_DISTORTION_NAMES)


def _check_pair(projection: object, external_distortion: object) -> None:
    """Validate that the projection/distortion pair is a supported combination.

    Args:
        projection: Supported camera projection parameters.
        external_distortion: External distortion parameters (NoExternalDistortion or
            BivariateWindshieldDistortion). Passing None raises TypeError.
    """
    if external_distortion is None:
        raise TypeError(
            "external_distortion=None is not supported; pass NoExternalDistortion() explicitly"
        )

    if (
        script_class_name(projection) not in _SUPPORTED_PROJECTIONS
        or script_class_name(external_distortion) not in _SUPPORTED_DISTORTIONS
    ):
        raise TypeError(
            "Unsupported camera projection/distortion pair: "
            f"({type(projection).__name__}, {type(external_distortion).__name__})"
        )


def _projection_tensors(
    projection: OpenCVPinholeProjection,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Extract and transfer the five OpenCV pinhole intrinsic tensors to the target device.

    Args:
        projection: OpenCV pinhole projection parameters.
        device: Target CUDA device.
        dtype: Target floating-point dtype.
        allow_device_transfer: Passed through to ``_to_dev``; see its docstring.

    Returns:
        Five-tuple of (focal_length, principal_point, radial_coeffs,
        tangential_coeffs, thin_prism_coeffs) on the target device.
    """
    return (
        _to_dev(projection.focal_length, device, dtype, allow_device_transfer),
        _to_dev(projection.principal_point, device, dtype, allow_device_transfer),
        _to_dev(projection.radial_coeffs, device, dtype, allow_device_transfer),
        _to_dev(projection.tangential_coeffs, device, dtype, allow_device_transfer),
        _to_dev(projection.thin_prism_coeffs, device, dtype, allow_device_transfer),
    )


def _projection_on_device(
    projection: OpenCVPinholeProjection,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool,
    resolution: tuple[int, int] | None = None,
) -> tuple[OpenCVPinholeProjection, tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """Build a device-local OpenCVPinholeProjection and return its component tensors.

    Args:
        projection: OpenCV pinhole projection parameters whose tensors may be on CPU.
        device: Target CUDA device.
        dtype: Target floating-point dtype.
        allow_device_transfer: Passed through to ``_to_dev``; see its docstring.
        resolution: If provided, overrides ``projection.resolution`` on the returned
            struct (useful when resolution is passed as a kernel argument separately).

    Returns:
        Tuple of (projection_on_device, (focal_length, principal_point,
        radial_coeffs, tangential_coeffs, thin_prism_coeffs)).
    """
    tensors = _projection_tensors(projection, device, dtype, allow_device_transfer)
    projected = OpenCVPinholeProjection(
        focal_length=tensors[0],
        principal_point=tensors[1],
        radial_coeffs=tensors[2],
        tangential_coeffs=tensors[3],
        thin_prism_coeffs=tensors[4],
        resolution=projection.resolution if resolution is None else resolution,
    )
    return projected, tensors


def _external_distortion_on_device(
    external_distortion: BivariateWindshieldDistortion,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool,
) -> tuple[BivariateWindshieldDistortion, Tensor]:
    """Build a device-local BivariateWindshieldDistortion and return its coefficients tensor.

    Args:
        external_distortion: BivariateWindshieldDistortion parameters; a value of
            any other registered type raises TypeError.
        device: Target CUDA device.
        dtype: Target floating-point dtype.
        allow_device_transfer: Passed through to ``_to_dev``; see its docstring.

    Returns:
        Tuple of (distortion_on_device, distortion_coeffs) where distortion_coeffs
        is the raw coefficients tensor on the target device.
    """
    if script_class_name(external_distortion) != "BivariateWindshieldDistortion":
        raise TypeError(
            "Expected BivariateWindshieldDistortion, "
            f"got {script_class_name(external_distortion)}"
        )
    distortion_coeffs = _to_dev(
        external_distortion.distortion_coeffs,
        device,
        dtype,
        allow_device_transfer,
    )
    return (
        BivariateWindshieldDistortion(
            distortion_coeffs,
            int(external_distortion.reference_polynomial),
            int(external_distortion.h_poly_degree),
            int(external_distortion.v_poly_degree),
        ),
        distortion_coeffs,
    )


def _projection_tensors_ftheta(
    projection: FThetaProjection,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Extract and transfer the four FTheta intrinsic tensors to the target device.

    Args:
        projection: FTheta projection parameters.
        device: Target CUDA device.
        dtype: Target floating-point dtype.
        allow_device_transfer: Passed through to ``_to_dev``; see its docstring.

    Returns:
        Four-tuple of (principal_point, fw_poly, bw_poly, A) on the target device.
        ``Ainv`` is derived on demand from ``A`` inside the CUDA struct's
        ``to_kernel_params()``.
    """
    return (
        _to_dev(projection.principal_point, device, dtype, allow_device_transfer),
        _to_dev(projection.fw_poly, device, dtype, allow_device_transfer),
        _to_dev(projection.bw_poly, device, dtype, allow_device_transfer),
        _to_dev(projection.A, device, dtype, allow_device_transfer),
    )


def _projection_on_device_ftheta(
    projection: FThetaProjection,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool,
    resolution: tuple[int, int] | None = None,
) -> tuple[FThetaProjection, tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Build a device-local FThetaProjection and return its component tensors.

    Args:
        projection: FTheta projection parameters whose tensors may be on CPU.
        device: Target CUDA device.
        dtype: Target floating-point dtype.
        allow_device_transfer: Passed through to ``_to_dev``; see its docstring.
        resolution: If provided, overrides ``projection.resolution`` on the returned
            struct (useful when resolution is passed as a kernel argument separately).

    Returns:
        Tuple of (projection_on_device, (principal_point, fw_poly, bw_poly, A)).
    """
    tensors = _projection_tensors_ftheta(
        projection, device, dtype, allow_device_transfer
    )
    projected = FThetaProjection(
        principal_point=tensors[0],
        fw_poly=tensors[1],
        bw_poly=tensors[2],
        A=tensors[3],
        resolution=projection.resolution if resolution is None else resolution,
        reference_polynomial=int(projection.reference_polynomial),
        fw_poly_degree=int(projection.fw_poly_degree),
        bw_poly_degree=int(projection.bw_poly_degree),
        newton_iterations=int(projection.newton_iterations),
        max_angle=float(projection.max_angle),
        min_2d_norm=float(projection.min_2d_norm),
    )
    return projected, tensors


def _projection_tensors_fisheye(
    projection: OpenCVFisheyeProjection,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Extract and transfer the four fisheye intrinsic tensors to the target device.

    Args:
        projection: OpenCV fisheye projection parameters.
        device: Target CUDA device.
        dtype: Target floating-point dtype.
        allow_device_transfer: Passed through to ``_to_dev``; see its docstring.

    Returns:
        Four-tuple of (principal_point, focal_length, forward_poly,
        approx_backward_factor) on the target device.
    """
    return (
        _to_dev(projection.principal_point, device, dtype, allow_device_transfer),
        _to_dev(projection.focal_length, device, dtype, allow_device_transfer),
        _to_dev(projection.forward_poly, device, dtype, allow_device_transfer),
        _to_dev(
            projection.approx_backward_factor, device, dtype, allow_device_transfer
        ),
    )


def _projection_on_device_fisheye(
    projection: OpenCVFisheyeProjection,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool,
    resolution: tuple[int, int] | None = None,
) -> tuple[OpenCVFisheyeProjection, tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Build a device-local OpenCVFisheyeProjection and return its component tensors.

    Args:
        projection: OpenCV fisheye projection parameters whose tensors may be on CPU.
        device: Target CUDA device.
        dtype: Target floating-point dtype.
        allow_device_transfer: Passed through to ``_to_dev``; see its docstring.
        resolution: If provided, overrides ``projection.resolution`` on the returned
            struct (useful when resolution is passed as a kernel argument separately).

    Returns:
        Tuple of (projection_on_device, (principal_point, focal_length,
        forward_poly, approx_backward_factor)).
    """
    tensors = _projection_tensors_fisheye(
        projection, device, dtype, allow_device_transfer
    )
    projected = OpenCVFisheyeProjection(
        principal_point=tensors[0],
        focal_length=tensors[1],
        forward_poly=tensors[2],
        approx_backward_factor=tensors[3],
        resolution=projection.resolution if resolution is None else resolution,
        newton_iterations=int(projection.newton_iterations),
        max_angle=float(projection.max_angle),
        min_2d_norm=float(projection.min_2d_norm),
    )
    return projected, tensors


def _projection_on_device_any(
    projection: CameraProjection,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool,
    resolution: tuple[int, int] | None = None,
) -> tuple[CameraProjection, tuple]:
    """Dispatch to the projection-specific transfer helper.

    Returns ``(device_projection, component_tensors)`` where the second slot is
    a tuple of component tensors in the order the autograd Function forwards
    expect (pinhole: 5-tuple ``focal_length, principal_point, radial_coeffs,
    tangential_coeffs, thin_prism_coeffs``; FTheta: 4-tuple
    ``principal_point, fw_poly, bw_poly, A``; fisheye: 4-tuple
    ``principal_point, focal_length, forward_poly, approx_backward_factor``).
    """
    dispatch = {
        "OpenCVPinholeProjection": _projection_on_device,
        "FThetaProjection": _projection_on_device_ftheta,
        "OpenCVFisheyeProjection": _projection_on_device_fisheye,
    }
    class_name = script_class_name(projection)
    try:
        transfer = dispatch[class_name]
    except KeyError as exc:
        raise TypeError(f"Unknown camera projection class: {class_name}") from exc
    return transfer(projection, device, dtype, allow_device_transfer, resolution)


def _select_camera_op(
    external_distortion: ExternalDistortion,
    fn_no_external: type,
    fn_bivariate: type,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool,
) -> tuple[object, object, tuple]:
    """Select the autograd op + dispatch args for the given external distortion.

    Returns ``(apply, distortion_arg, extra_distortion_args)`` where the caller
    splices ``distortion_arg`` plus ``*extra_distortion_args`` into the apply
    arg tuple right after ``projection_cuda``. This collapses the BivariateWindshield
    vs. NoExternalDistortion branching that otherwise repeats across every public op.
    """
    if script_class_name(external_distortion) == "BivariateWindshieldDistortion":
        distortion_cuda, distortion_coeffs = _external_distortion_on_device(
            external_distortion, device, dtype, allow_device_transfer
        )
        return fn_bivariate.apply, distortion_cuda, (distortion_coeffs,)
    return fn_no_external.apply, external_distortion, ()


def _timestamp_bounds(
    start_timestamp_us: int | None, end_timestamp_us: int | None
) -> tuple[int, int]:
    """Normalize optional timestamp arguments to a (start, end) int pair.

    Args:
        start_timestamp_us: Start timestamp in microseconds, or None.
        end_timestamp_us: End timestamp in microseconds, or None.

    Returns:
        ``(start_timestamp_us, end_timestamp_us)`` as ints, or ``(0, 0)`` when
        both are None. Raises ValueError if exactly one is None.
    """
    if start_timestamp_us is None and end_timestamp_us is None:
        return 0, 0
    if start_timestamp_us is None or end_timestamp_us is None:
        raise ValueError(
            "start_timestamp_us and end_timestamp_us must be provided together"
        )
    return int(start_timestamp_us), int(end_timestamp_us)


def _zero_like(shape: tuple[int, ...], reference: Tensor) -> Tensor:
    """Allocate a zero tensor with the same device and dtype as ``reference``."""
    return torch.zeros(shape, device=reference.device, dtype=reference.dtype)


class _CameraRaysToImagePoints(torch.autograd.Function):
    """Autograd Function wrapping ``camera_rays_to_image_points_opencv_pinhole_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        camera_rays: Tensor,
        projection: OpenCVPinholeProjection,
        external_distortion: NoExternalDistortion,
        focal_length: Tensor,
        principal_point: Tensor,
        radial_coeffs: Tensor,
        tangential_coeffs: Tensor,
        thin_prism_coeffs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del (
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        )
        (
            image_points,
            valid_flags,
            scratch,
        ) = torch.ops.gsplat_sensors.camera_rays_to_image_points_opencv_pinhole_no_external(
            projection, external_distortion, camera_rays
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(camera_rays, scratch)
        ctx.mark_non_differentiable(valid_flags)
        return image_points, valid_flags

    @staticmethod
    def backward(
        ctx, grad_image_points: Tensor | None, grad_valid_flags: Tensor | None
    ):
        del grad_valid_flags
        camera_rays, scratch = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((camera_rays.shape[0], 2), camera_rays)
        grads = torch.ops.gsplat_sensors.camera_rays_to_image_points_opencv_pinhole_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            camera_rays,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[5],
            ctx.needs_input_grad[6],
            ctx.needs_input_grad[7],
        )
        (
            grad_camera_rays,
            grad_focal,
            grad_pp,
            grad_radial,
            grad_tangential,
            grad_thin,
        ) = grads
        return (
            grad_camera_rays if ctx.needs_input_grad[0] else None,
            None,
            None,
            grad_focal if ctx.needs_input_grad[3] else None,
            grad_pp if ctx.needs_input_grad[4] else None,
            grad_radial if ctx.needs_input_grad[5] else None,
            grad_tangential if ctx.needs_input_grad[6] else None,
            grad_thin if ctx.needs_input_grad[7] else None,
        )


class _ImagePointsToCameraRays(torch.autograd.Function):
    """Autograd Function wrapping ``image_points_to_camera_rays_opencv_pinhole_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        projection: OpenCVPinholeProjection,
        external_distortion: NoExternalDistortion,
        focal_length: Tensor,
        principal_point: Tensor,
        radial_coeffs: Tensor,
        tangential_coeffs: Tensor,
        thin_prism_coeffs: Tensor,
    ) -> Tensor:
        del (
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        )
        (
            camera_rays,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_camera_rays_opencv_pinhole_no_external(
            projection, external_distortion, image_points
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(image_points, scratch)
        return camera_rays

    @staticmethod
    def backward(ctx, grad_camera_rays: Tensor | None):
        image_points, scratch = ctx.saved_tensors
        if grad_camera_rays is None:
            grad_camera_rays = _zero_like((image_points.shape[0], 3), image_points)
        grads = torch.ops.gsplat_sensors.image_points_to_camera_rays_opencv_pinhole_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            grad_camera_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[5],
            ctx.needs_input_grad[6],
            ctx.needs_input_grad[7],
        )
        (
            grad_image_points,
            grad_focal,
            grad_pp,
            grad_radial,
            grad_tangential,
            grad_thin,
        ) = grads
        return (
            grad_image_points if ctx.needs_input_grad[0] else None,
            None,
            None,
            grad_focal if ctx.needs_input_grad[3] else None,
            grad_pp if ctx.needs_input_grad[4] else None,
            grad_radial if ctx.needs_input_grad[5] else None,
            grad_tangential if ctx.needs_input_grad[6] else None,
            grad_thin if ctx.needs_input_grad[7] else None,
        )


class _ProjectWorldPointsMeanPose(torch.autograd.Function):
    """Autograd Function wrapping ``project_world_points_mean_pose_opencv_pinhole_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: OpenCVPinholeProjection,
        external_distortion: NoExternalDistortion,
        focal_length: Tensor,
        principal_point: Tensor,
        radial_coeffs: Tensor,
        tangential_coeffs: Tensor,
        thin_prism_coeffs: Tensor,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        del (
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        )
        (
            image_points,
            valid_flags,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.project_world_points_mean_pose_opencv_pinhole_no_external(
            projection,
            external_distortion,
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(start_timestamp_us),
            int(end_timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(world_points, start_rotation, end_rotation, scratch)
        ctx.mark_non_differentiable(valid_flags, timestamps_us)
        return image_points, valid_flags, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_image_points: Tensor | None,
        grad_valid_flags: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_valid_flags, grad_timestamps_us, grad_pose_t, grad_pose_r
        world_points, start_rotation, end_rotation, scratch = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((world_points.shape[0], 2), world_points)
        grads = torch.ops.gsplat_sensors.project_world_points_mean_pose_opencv_pinhole_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            world_points,
            start_rotation,
            end_rotation,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            ctx.needs_input_grad[10],
            ctx.needs_input_grad[11],
        )
        (
            grad_world,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_focal,
            grad_pp,
            grad_radial,
            grad_tangential,
            grad_thin,
        ) = grads
        return (
            grad_world if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_focal if ctx.needs_input_grad[7] else None,
            grad_pp if ctx.needs_input_grad[8] else None,
            grad_radial if ctx.needs_input_grad[9] else None,
            grad_tangential if ctx.needs_input_grad[10] else None,
            grad_thin if ctx.needs_input_grad[11] else None,
            None,
            None,
        )


class _ProjectWorldPointsShutterPose(torch.autograd.Function):
    """Autograd Function wrapping ``project_world_points_shutter_pose_opencv_pinhole_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: OpenCVPinholeProjection,
        external_distortion: NoExternalDistortion,
        focal_length: Tensor,
        principal_point: Tensor,
        radial_coeffs: Tensor,
        tangential_coeffs: Tensor,
        thin_prism_coeffs: Tensor,
        width: int,
        height: int,
        shutter_type: int,
        start_timestamp_us: int,
        end_timestamp_us: int,
        max_iterations: int,
        stop_mean_error_px: float,
        stop_delta_mean_error_px: float,
        initial_relative_time: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        del (
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        )
        (
            image_points,
            valid_flags,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.project_world_points_shutter_pose_opencv_pinhole_no_external(
            projection,
            external_distortion,
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(width),
            int(height),
            int(shutter_type),
            int(start_timestamp_us),
            int(end_timestamp_us),
            int(max_iterations),
            float(stop_mean_error_px),
            float(stop_delta_mean_error_px),
            float(initial_relative_time),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.shutter_type = int(shutter_type)
        ctx.max_iterations = int(max_iterations)
        ctx.initial_relative_time = float(initial_relative_time)
        ctx.save_for_backward(
            world_points, start_rotation, end_rotation, valid_flags, scratch
        )
        ctx.mark_non_differentiable(valid_flags, timestamps_us)
        return image_points, valid_flags, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_image_points: Tensor | None,
        grad_valid_flags: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_valid_flags, grad_timestamps_us, grad_pose_t, grad_pose_r
        (
            world_points,
            start_rotation,
            end_rotation,
            valid_flags,
            scratch,
        ) = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((world_points.shape[0], 2), world_points)
        grads = torch.ops.gsplat_sensors.project_world_points_shutter_pose_opencv_pinhole_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            world_points,
            start_rotation,
            end_rotation,
            ctx.shutter_type,
            ctx.max_iterations,
            ctx.initial_relative_time,
            valid_flags,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            ctx.needs_input_grad[10],
            ctx.needs_input_grad[11],
        )
        (
            grad_world,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_focal,
            grad_pp,
            grad_radial,
            grad_tangential,
            grad_thin,
        ) = grads
        return (
            grad_world if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_focal if ctx.needs_input_grad[7] else None,
            grad_pp if ctx.needs_input_grad[8] else None,
            grad_radial if ctx.needs_input_grad[9] else None,
            grad_tangential if ctx.needs_input_grad[10] else None,
            grad_thin if ctx.needs_input_grad[11] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _ImagePointsToWorldRaysStaticPose(torch.autograd.Function):
    """Autograd Function wrapping ``image_points_to_world_rays_static_pose_opencv_pinhole_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        translations: Tensor,
        rotations: Tensor,
        projection: OpenCVPinholeProjection,
        external_distortion: NoExternalDistortion,
        focal_length: Tensor,
        principal_point: Tensor,
        radial_coeffs: Tensor,
        tangential_coeffs: Tensor,
        thin_prism_coeffs: Tensor,
        timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del (
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        )
        (
            world_rays,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_world_rays_static_pose_opencv_pinhole_no_external(
            projection,
            external_distortion,
            image_points,
            translations,
            rotations,
            int(timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(image_points, translations, rotations, scratch)
        ctx.mark_non_differentiable(timestamps_us)
        return world_rays, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_world_rays: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_timestamps_us, grad_pose_t, grad_pose_r
        image_points, translations, rotations, scratch = ctx.saved_tensors
        if grad_world_rays is None:
            grad_world_rays = _zero_like((image_points.shape[0], 6), image_points)
        grads = torch.ops.gsplat_sensors.image_points_to_world_rays_static_pose_opencv_pinhole_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            translations,
            rotations,
            grad_world_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[5],
            ctx.needs_input_grad[6],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
        )
        (
            grad_points,
            grad_trans,
            grad_rots,
            grad_focal,
            grad_pp,
            grad_radial,
            grad_tangential,
            grad_thin,
        ) = grads
        return (
            grad_points if ctx.needs_input_grad[0] else None,
            grad_trans if ctx.needs_input_grad[1] else None,
            grad_rots if ctx.needs_input_grad[2] else None,
            None,
            None,
            grad_focal if ctx.needs_input_grad[5] else None,
            grad_pp if ctx.needs_input_grad[6] else None,
            grad_radial if ctx.needs_input_grad[7] else None,
            grad_tangential if ctx.needs_input_grad[8] else None,
            grad_thin if ctx.needs_input_grad[9] else None,
            None,
        )


class _ImagePointsToWorldRaysShutterPose(torch.autograd.Function):
    """Autograd Function wrapping ``image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: OpenCVPinholeProjection,
        external_distortion: NoExternalDistortion,
        focal_length: Tensor,
        principal_point: Tensor,
        radial_coeffs: Tensor,
        tangential_coeffs: Tensor,
        thin_prism_coeffs: Tensor,
        width: int,
        height: int,
        shutter_type: int,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del (
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        )
        (
            world_rays,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external(
            projection,
            external_distortion,
            image_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(width),
            int(height),
            int(shutter_type),
            int(start_timestamp_us),
            int(end_timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.shutter_type = int(shutter_type)
        ctx.save_for_backward(image_points, start_rotation, end_rotation, scratch)
        ctx.mark_non_differentiable(timestamps_us)
        return world_rays, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_world_rays: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_timestamps_us, grad_pose_t, grad_pose_r
        image_points, start_rotation, end_rotation, scratch = ctx.saved_tensors
        if grad_world_rays is None:
            grad_world_rays = _zero_like((image_points.shape[0], 6), image_points)
        grads = torch.ops.gsplat_sensors.image_points_to_world_rays_shutter_pose_opencv_pinhole_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            start_rotation,
            end_rotation,
            ctx.shutter_type,
            grad_world_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            ctx.needs_input_grad[10],
            ctx.needs_input_grad[11],
        )
        (
            grad_points,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_focal,
            grad_pp,
            grad_radial,
            grad_tangential,
            grad_thin,
        ) = grads
        return (
            grad_points if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_focal if ctx.needs_input_grad[7] else None,
            grad_pp if ctx.needs_input_grad[8] else None,
            grad_radial if ctx.needs_input_grad[9] else None,
            grad_tangential if ctx.needs_input_grad[10] else None,
            grad_thin if ctx.needs_input_grad[11] else None,
            None,
            None,
            None,
            None,
            None,
        )


class _CameraRaysToImagePointsBivariateWindshield(torch.autograd.Function):
    """Autograd Function wrapping ``camera_rays_to_image_points_opencv_pinhole_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        camera_rays: Tensor,
        projection: OpenCVPinholeProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        focal_length: Tensor,
        principal_point: Tensor,
        radial_coeffs: Tensor,
        tangential_coeffs: Tensor,
        thin_prism_coeffs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del (
            distortion_coeffs,
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        )
        (
            image_points,
            valid_flags,
            scratch,
        ) = torch.ops.gsplat_sensors.camera_rays_to_image_points_opencv_pinhole_bivariate_windshield(
            projection, external_distortion, camera_rays
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(camera_rays, scratch)
        ctx.mark_non_differentiable(valid_flags)
        return image_points, valid_flags

    @staticmethod
    def backward(
        ctx, grad_image_points: Tensor | None, grad_valid_flags: Tensor | None
    ):
        del grad_valid_flags
        camera_rays, scratch = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((camera_rays.shape[0], 2), camera_rays)
        grads = torch.ops.gsplat_sensors.camera_rays_to_image_points_opencv_pinhole_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            camera_rays,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[5],
            ctx.needs_input_grad[6],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[3],
        )
        (
            grad_camera_rays,
            grad_focal,
            grad_pp,
            grad_radial,
            grad_tangential,
            grad_thin,
            grad_distortion,
        ) = grads
        return (
            grad_camera_rays if ctx.needs_input_grad[0] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[3] else None,
            grad_focal if ctx.needs_input_grad[4] else None,
            grad_pp if ctx.needs_input_grad[5] else None,
            grad_radial if ctx.needs_input_grad[6] else None,
            grad_tangential if ctx.needs_input_grad[7] else None,
            grad_thin if ctx.needs_input_grad[8] else None,
        )


class _ImagePointsToCameraRaysBivariateWindshield(torch.autograd.Function):
    """Autograd Function wrapping ``image_points_to_camera_rays_opencv_pinhole_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        projection: OpenCVPinholeProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        focal_length: Tensor,
        principal_point: Tensor,
        radial_coeffs: Tensor,
        tangential_coeffs: Tensor,
        thin_prism_coeffs: Tensor,
    ) -> Tensor:
        del (
            distortion_coeffs,
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        )
        (
            camera_rays,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_camera_rays_opencv_pinhole_bivariate_windshield(
            projection, external_distortion, image_points
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(image_points, scratch)
        return camera_rays

    @staticmethod
    def backward(ctx, grad_camera_rays: Tensor | None):
        image_points, scratch = ctx.saved_tensors
        if grad_camera_rays is None:
            grad_camera_rays = _zero_like((image_points.shape[0], 3), image_points)
        grads = torch.ops.gsplat_sensors.image_points_to_camera_rays_opencv_pinhole_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            grad_camera_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[5],
            ctx.needs_input_grad[6],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[3],
        )
        (
            grad_image_points,
            grad_focal,
            grad_pp,
            grad_radial,
            grad_tangential,
            grad_thin,
            grad_distortion,
        ) = grads
        return (
            grad_image_points if ctx.needs_input_grad[0] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[3] else None,
            grad_focal if ctx.needs_input_grad[4] else None,
            grad_pp if ctx.needs_input_grad[5] else None,
            grad_radial if ctx.needs_input_grad[6] else None,
            grad_tangential if ctx.needs_input_grad[7] else None,
            grad_thin if ctx.needs_input_grad[8] else None,
        )


class _ProjectWorldPointsMeanPoseBivariateWindshield(torch.autograd.Function):
    """Autograd Function wrapping ``project_world_points_mean_pose_opencv_pinhole_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: OpenCVPinholeProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        focal_length: Tensor,
        principal_point: Tensor,
        radial_coeffs: Tensor,
        tangential_coeffs: Tensor,
        thin_prism_coeffs: Tensor,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        del (
            distortion_coeffs,
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        )
        (
            image_points,
            valid_flags,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.project_world_points_mean_pose_opencv_pinhole_bivariate_windshield(
            projection,
            external_distortion,
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(start_timestamp_us),
            int(end_timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(world_points, start_rotation, end_rotation, scratch)
        ctx.mark_non_differentiable(valid_flags, timestamps_us)
        return image_points, valid_flags, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_image_points: Tensor | None,
        grad_valid_flags: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_valid_flags, grad_timestamps_us, grad_pose_t, grad_pose_r
        world_points, start_rotation, end_rotation, scratch = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((world_points.shape[0], 2), world_points)
        grads = torch.ops.gsplat_sensors.project_world_points_mean_pose_opencv_pinhole_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            world_points,
            start_rotation,
            end_rotation,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            ctx.needs_input_grad[10],
            ctx.needs_input_grad[11],
            ctx.needs_input_grad[12],
            ctx.needs_input_grad[7],
        )
        (
            grad_world,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_focal,
            grad_pp,
            grad_radial,
            grad_tangential,
            grad_thin,
            grad_distortion,
        ) = grads
        return (
            grad_world if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[7] else None,
            grad_focal if ctx.needs_input_grad[8] else None,
            grad_pp if ctx.needs_input_grad[9] else None,
            grad_radial if ctx.needs_input_grad[10] else None,
            grad_tangential if ctx.needs_input_grad[11] else None,
            grad_thin if ctx.needs_input_grad[12] else None,
            None,
            None,
        )


class _ProjectWorldPointsShutterPoseBivariateWindshield(torch.autograd.Function):
    """Autograd Function wrapping ``project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: OpenCVPinholeProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        focal_length: Tensor,
        principal_point: Tensor,
        radial_coeffs: Tensor,
        tangential_coeffs: Tensor,
        thin_prism_coeffs: Tensor,
        width: int,
        height: int,
        shutter_type: int,
        start_timestamp_us: int,
        end_timestamp_us: int,
        max_iterations: int,
        stop_mean_error_px: float,
        stop_delta_mean_error_px: float,
        initial_relative_time: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        del (
            distortion_coeffs,
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        )
        (
            image_points,
            valid_flags,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield(
            projection,
            external_distortion,
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(width),
            int(height),
            int(shutter_type),
            int(start_timestamp_us),
            int(end_timestamp_us),
            int(max_iterations),
            float(stop_mean_error_px),
            float(stop_delta_mean_error_px),
            float(initial_relative_time),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.shutter_type = int(shutter_type)
        ctx.max_iterations = int(max_iterations)
        ctx.initial_relative_time = float(initial_relative_time)
        ctx.save_for_backward(
            world_points, start_rotation, end_rotation, valid_flags, scratch
        )
        ctx.mark_non_differentiable(valid_flags, timestamps_us)
        return image_points, valid_flags, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_image_points: Tensor | None,
        grad_valid_flags: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_valid_flags, grad_timestamps_us, grad_pose_t, grad_pose_r
        (
            world_points,
            start_rotation,
            end_rotation,
            valid_flags,
            scratch,
        ) = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((world_points.shape[0], 2), world_points)
        grads = torch.ops.gsplat_sensors.project_world_points_shutter_pose_opencv_pinhole_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            world_points,
            start_rotation,
            end_rotation,
            ctx.shutter_type,
            ctx.max_iterations,
            ctx.initial_relative_time,
            valid_flags,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            ctx.needs_input_grad[10],
            ctx.needs_input_grad[11],
            ctx.needs_input_grad[12],
            ctx.needs_input_grad[7],
        )
        (
            grad_points,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_focal,
            grad_pp,
            grad_radial,
            grad_tangential,
            grad_thin,
            grad_distortion,
        ) = grads
        return (
            grad_points if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[7] else None,
            grad_focal if ctx.needs_input_grad[8] else None,
            grad_pp if ctx.needs_input_grad[9] else None,
            grad_radial if ctx.needs_input_grad[10] else None,
            grad_tangential if ctx.needs_input_grad[11] else None,
            grad_thin if ctx.needs_input_grad[12] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _ImagePointsToWorldRaysStaticPoseBivariateWindshield(torch.autograd.Function):
    """Autograd Function wrapping ``image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        translations: Tensor,
        rotations: Tensor,
        projection: OpenCVPinholeProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        focal_length: Tensor,
        principal_point: Tensor,
        radial_coeffs: Tensor,
        tangential_coeffs: Tensor,
        thin_prism_coeffs: Tensor,
        timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del (
            distortion_coeffs,
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        )
        (
            world_rays,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield(
            projection,
            external_distortion,
            image_points,
            translations,
            rotations,
            int(timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(image_points, translations, rotations, scratch)
        ctx.mark_non_differentiable(timestamps_us)
        return world_rays, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_world_rays: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_timestamps_us, grad_pose_t, grad_pose_r
        image_points, translations, rotations, scratch = ctx.saved_tensors
        if grad_world_rays is None:
            grad_world_rays = _zero_like((image_points.shape[0], 6), image_points)
        grads = torch.ops.gsplat_sensors.image_points_to_world_rays_static_pose_opencv_pinhole_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            translations,
            rotations,
            grad_world_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[6],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            ctx.needs_input_grad[10],
            ctx.needs_input_grad[5],
        )
        (
            grad_points,
            grad_trans,
            grad_rots,
            grad_focal,
            grad_pp,
            grad_radial,
            grad_tangential,
            grad_thin,
            grad_distortion,
        ) = grads
        return (
            grad_points if ctx.needs_input_grad[0] else None,
            grad_trans if ctx.needs_input_grad[1] else None,
            grad_rots if ctx.needs_input_grad[2] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[5] else None,
            grad_focal if ctx.needs_input_grad[6] else None,
            grad_pp if ctx.needs_input_grad[7] else None,
            grad_radial if ctx.needs_input_grad[8] else None,
            grad_tangential if ctx.needs_input_grad[9] else None,
            grad_thin if ctx.needs_input_grad[10] else None,
            None,
        )


class _ImagePointsToWorldRaysShutterPoseBivariateWindshield(torch.autograd.Function):
    """Autograd Function wrapping ``image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: OpenCVPinholeProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        focal_length: Tensor,
        principal_point: Tensor,
        radial_coeffs: Tensor,
        tangential_coeffs: Tensor,
        thin_prism_coeffs: Tensor,
        width: int,
        height: int,
        shutter_type: int,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del (
            distortion_coeffs,
            focal_length,
            principal_point,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        )
        (
            world_rays,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield(
            projection,
            external_distortion,
            image_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(width),
            int(height),
            int(shutter_type),
            int(start_timestamp_us),
            int(end_timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.shutter_type = int(shutter_type)
        ctx.save_for_backward(image_points, start_rotation, end_rotation, scratch)
        ctx.mark_non_differentiable(timestamps_us)
        return world_rays, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_world_rays: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_timestamps_us, grad_pose_t, grad_pose_r
        image_points, start_rotation, end_rotation, scratch = ctx.saved_tensors
        if grad_world_rays is None:
            grad_world_rays = _zero_like((image_points.shape[0], 6), image_points)
        grads = torch.ops.gsplat_sensors.image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            start_rotation,
            end_rotation,
            ctx.shutter_type,
            grad_world_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            ctx.needs_input_grad[10],
            ctx.needs_input_grad[11],
            ctx.needs_input_grad[12],
            ctx.needs_input_grad[7],
        )
        (
            grad_points,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_focal,
            grad_pp,
            grad_radial,
            grad_tangential,
            grad_thin,
            grad_distortion,
        ) = grads
        return (
            grad_points if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[7] else None,
            grad_focal if ctx.needs_input_grad[8] else None,
            grad_pp if ctx.needs_input_grad[9] else None,
            grad_radial if ctx.needs_input_grad[10] else None,
            grad_tangential if ctx.needs_input_grad[11] else None,
            grad_thin if ctx.needs_input_grad[12] else None,
            None,
            None,
            None,
            None,
            None,
        )


# =============================================================================
# FTheta autograd Function classes (6 ops x 2 distortion variants).
# CUDA backward kernels emit (grad_A, grad_Ainv); these Functions consolidate
# grad_Ainv into grad_A via the inverse-matrix VJP so the user-facing
# FThetaProjection exposes a single trainable A.
# =============================================================================


def _consolidate_ftheta_grad_a(
    grad_A: Tensor, grad_Ainv: Tensor, projection: FThetaProjection
) -> Tensor:
    """Fold grad_Ainv into grad_A using the matrix-inverse VJP.

    With ``B = inv(A)``, ``dB = -B (dA) B`` gives the cotangent pullback
    ``grad_A_contrib = -B.T @ grad_B @ B.T``. ``A`` is stored as a flat
    (4,) row-major 2x2 tensor; reshape to 2x2 for the matmul, then
    flatten back to (4,) before adding to ``grad_A``.
    """
    Ainv_2x2 = projection.Ainv.to(grad_A.dtype).view(2, 2)
    grad_Ainv_2x2 = grad_Ainv.view(2, 2)
    grad_A_contrib = -Ainv_2x2.T @ grad_Ainv_2x2 @ Ainv_2x2.T
    return grad_A + grad_A_contrib.reshape(4)


class _CameraRaysToImagePointsFTheta(torch.autograd.Function):
    """Autograd Function wrapping ``camera_rays_to_image_points_ftheta_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        camera_rays: Tensor,
        projection: FThetaProjection,
        external_distortion: NoExternalDistortion,
        principal_point: Tensor,
        fw_poly: Tensor,
        bw_poly: Tensor,
        A: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del principal_point, fw_poly, bw_poly, A
        (
            image_points,
            valid_flags,
            scratch,
        ) = torch.ops.gsplat_sensors.camera_rays_to_image_points_ftheta_no_external(
            projection, external_distortion, camera_rays
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(camera_rays, scratch)
        ctx.mark_non_differentiable(valid_flags)
        return image_points, valid_flags

    @staticmethod
    def backward(
        ctx, grad_image_points: Tensor | None, grad_valid_flags: Tensor | None
    ):
        del grad_valid_flags
        camera_rays, scratch = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((camera_rays.shape[0], 2), camera_rays)
        need_A = ctx.needs_input_grad[6]
        grads = torch.ops.gsplat_sensors.camera_rays_to_image_points_ftheta_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            camera_rays,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[5],
            need_A,
            need_A,
        )
        grad_camera_rays, grad_pp, grad_fw, grad_bw, grad_A, grad_Ainv = grads
        if need_A:
            grad_A = _consolidate_ftheta_grad_a(grad_A, grad_Ainv, ctx.projection)
        return (
            grad_camera_rays if ctx.needs_input_grad[0] else None,
            None,
            None,
            grad_pp if ctx.needs_input_grad[3] else None,
            grad_fw if ctx.needs_input_grad[4] else None,
            grad_bw if ctx.needs_input_grad[5] else None,
            grad_A if need_A else None,
        )


class _ImagePointsToCameraRaysFTheta(torch.autograd.Function):
    """Autograd Function wrapping ``image_points_to_camera_rays_ftheta_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        projection: FThetaProjection,
        external_distortion: NoExternalDistortion,
        principal_point: Tensor,
        fw_poly: Tensor,
        bw_poly: Tensor,
        A: Tensor,
    ) -> Tensor:
        del principal_point, fw_poly, bw_poly, A
        (
            camera_rays,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_camera_rays_ftheta_no_external(
            projection, external_distortion, image_points
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(image_points, scratch)
        return camera_rays

    @staticmethod
    def backward(ctx, grad_camera_rays: Tensor | None):
        image_points, scratch = ctx.saved_tensors
        if grad_camera_rays is None:
            grad_camera_rays = _zero_like((image_points.shape[0], 3), image_points)
        need_A = ctx.needs_input_grad[6]
        grads = torch.ops.gsplat_sensors.image_points_to_camera_rays_ftheta_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            grad_camera_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[5],
            need_A,
            need_A,
        )
        grad_image_points, grad_pp, grad_fw, grad_bw, grad_A, grad_Ainv = grads
        if need_A:
            grad_A = _consolidate_ftheta_grad_a(grad_A, grad_Ainv, ctx.projection)
        return (
            grad_image_points if ctx.needs_input_grad[0] else None,
            None,
            None,
            grad_pp if ctx.needs_input_grad[3] else None,
            grad_fw if ctx.needs_input_grad[4] else None,
            grad_bw if ctx.needs_input_grad[5] else None,
            grad_A if need_A else None,
        )


class _CameraRaysToImagePointsFThetaBivariateWindshield(torch.autograd.Function):
    """Autograd Function wrapping ``camera_rays_to_image_points_ftheta_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        camera_rays: Tensor,
        projection: FThetaProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        principal_point: Tensor,
        fw_poly: Tensor,
        bw_poly: Tensor,
        A: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del distortion_coeffs, principal_point, fw_poly, bw_poly, A
        (
            image_points,
            valid_flags,
            scratch,
        ) = torch.ops.gsplat_sensors.camera_rays_to_image_points_ftheta_bivariate_windshield(
            projection, external_distortion, camera_rays
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(camera_rays, scratch)
        ctx.mark_non_differentiable(valid_flags)
        return image_points, valid_flags

    @staticmethod
    def backward(
        ctx, grad_image_points: Tensor | None, grad_valid_flags: Tensor | None
    ):
        del grad_valid_flags
        camera_rays, scratch = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((camera_rays.shape[0], 2), camera_rays)
        need_A = ctx.needs_input_grad[7]
        grads = torch.ops.gsplat_sensors.camera_rays_to_image_points_ftheta_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            camera_rays,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[5],
            ctx.needs_input_grad[6],
            need_A,
            need_A,
            ctx.needs_input_grad[3],
        )
        (
            grad_camera_rays,
            grad_pp,
            grad_fw,
            grad_bw,
            grad_A,
            grad_Ainv,
            grad_distortion,
        ) = grads
        if need_A:
            grad_A = _consolidate_ftheta_grad_a(grad_A, grad_Ainv, ctx.projection)
        return (
            grad_camera_rays if ctx.needs_input_grad[0] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[3] else None,
            grad_pp if ctx.needs_input_grad[4] else None,
            grad_fw if ctx.needs_input_grad[5] else None,
            grad_bw if ctx.needs_input_grad[6] else None,
            grad_A if need_A else None,
        )


class _ImagePointsToCameraRaysFThetaBivariateWindshield(torch.autograd.Function):
    """Autograd Function wrapping ``image_points_to_camera_rays_ftheta_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        projection: FThetaProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        principal_point: Tensor,
        fw_poly: Tensor,
        bw_poly: Tensor,
        A: Tensor,
    ) -> Tensor:
        del distortion_coeffs, principal_point, fw_poly, bw_poly, A
        (
            camera_rays,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_camera_rays_ftheta_bivariate_windshield(
            projection, external_distortion, image_points
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(image_points, scratch)
        return camera_rays

    @staticmethod
    def backward(ctx, grad_camera_rays: Tensor | None):
        image_points, scratch = ctx.saved_tensors
        if grad_camera_rays is None:
            grad_camera_rays = _zero_like((image_points.shape[0], 3), image_points)
        need_A = ctx.needs_input_grad[7]
        grads = torch.ops.gsplat_sensors.image_points_to_camera_rays_ftheta_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            grad_camera_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[5],
            ctx.needs_input_grad[6],
            need_A,
            need_A,
            ctx.needs_input_grad[3],
        )
        (
            grad_image_points,
            grad_pp,
            grad_fw,
            grad_bw,
            grad_A,
            grad_Ainv,
            grad_distortion,
        ) = grads
        if need_A:
            grad_A = _consolidate_ftheta_grad_a(grad_A, grad_Ainv, ctx.projection)
        return (
            grad_image_points if ctx.needs_input_grad[0] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[3] else None,
            grad_pp if ctx.needs_input_grad[4] else None,
            grad_fw if ctx.needs_input_grad[5] else None,
            grad_bw if ctx.needs_input_grad[6] else None,
            grad_A if need_A else None,
        )


class _ProjectWorldPointsMeanPoseFTheta(torch.autograd.Function):
    """Autograd Function wrapping ``project_world_points_mean_pose_ftheta_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: FThetaProjection,
        external_distortion: NoExternalDistortion,
        principal_point: Tensor,
        fw_poly: Tensor,
        bw_poly: Tensor,
        A: Tensor,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        del principal_point, fw_poly, bw_poly, A
        (
            image_points,
            valid_flags,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.project_world_points_mean_pose_ftheta_no_external(
            projection,
            external_distortion,
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(start_timestamp_us),
            int(end_timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(world_points, start_rotation, end_rotation, scratch)
        ctx.mark_non_differentiable(valid_flags, timestamps_us)
        return image_points, valid_flags, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_image_points: Tensor | None,
        grad_valid_flags: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_valid_flags, grad_timestamps_us, grad_pose_t, grad_pose_r
        world_points, start_rotation, end_rotation, scratch = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((world_points.shape[0], 2), world_points)
        need_A = ctx.needs_input_grad[10]
        grads = torch.ops.gsplat_sensors.project_world_points_mean_pose_ftheta_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            world_points,
            start_rotation,
            end_rotation,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            need_A,
            need_A,
        )
        (
            grad_world,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_pp,
            grad_fw,
            grad_bw,
            grad_A,
            grad_Ainv,
        ) = grads
        if need_A:
            grad_A = _consolidate_ftheta_grad_a(grad_A, grad_Ainv, ctx.projection)
        return (
            grad_world if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_pp if ctx.needs_input_grad[7] else None,
            grad_fw if ctx.needs_input_grad[8] else None,
            grad_bw if ctx.needs_input_grad[9] else None,
            grad_A if need_A else None,
            None,
            None,
        )


class _ProjectWorldPointsMeanPoseFThetaBivariateWindshield(torch.autograd.Function):
    """Autograd Function wrapping ``project_world_points_mean_pose_ftheta_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: FThetaProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        principal_point: Tensor,
        fw_poly: Tensor,
        bw_poly: Tensor,
        A: Tensor,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        del distortion_coeffs, principal_point, fw_poly, bw_poly, A
        (
            image_points,
            valid_flags,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.project_world_points_mean_pose_ftheta_bivariate_windshield(
            projection,
            external_distortion,
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(start_timestamp_us),
            int(end_timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(world_points, start_rotation, end_rotation, scratch)
        ctx.mark_non_differentiable(valid_flags, timestamps_us)
        return image_points, valid_flags, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_image_points: Tensor | None,
        grad_valid_flags: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_valid_flags, grad_timestamps_us, grad_pose_t, grad_pose_r
        world_points, start_rotation, end_rotation, scratch = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((world_points.shape[0], 2), world_points)
        need_A = ctx.needs_input_grad[11]
        grads = torch.ops.gsplat_sensors.project_world_points_mean_pose_ftheta_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            world_points,
            start_rotation,
            end_rotation,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            ctx.needs_input_grad[10],
            need_A,
            need_A,
            ctx.needs_input_grad[7],
        )
        (
            grad_world,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_pp,
            grad_fw,
            grad_bw,
            grad_A,
            grad_Ainv,
            grad_distortion,
        ) = grads
        if need_A:
            grad_A = _consolidate_ftheta_grad_a(grad_A, grad_Ainv, ctx.projection)
        return (
            grad_world if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[7] else None,
            grad_pp if ctx.needs_input_grad[8] else None,
            grad_fw if ctx.needs_input_grad[9] else None,
            grad_bw if ctx.needs_input_grad[10] else None,
            grad_A if need_A else None,
            None,
            None,
        )


class _ImagePointsToWorldRaysStaticPoseFTheta(torch.autograd.Function):
    """Autograd Function wrapping ``image_points_to_world_rays_static_pose_ftheta_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        translations: Tensor,
        rotations: Tensor,
        projection: FThetaProjection,
        external_distortion: NoExternalDistortion,
        principal_point: Tensor,
        fw_poly: Tensor,
        bw_poly: Tensor,
        A: Tensor,
        timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del principal_point, fw_poly, bw_poly, A
        (
            world_rays,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_world_rays_static_pose_ftheta_no_external(
            projection,
            external_distortion,
            image_points,
            translations,
            rotations,
            int(timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(image_points, translations, rotations, scratch)
        ctx.mark_non_differentiable(timestamps_us)
        return world_rays, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_world_rays: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_timestamps_us, grad_pose_t, grad_pose_r
        image_points, translations, rotations, scratch = ctx.saved_tensors
        if grad_world_rays is None:
            grad_world_rays = _zero_like((image_points.shape[0], 6), image_points)
        need_A = ctx.needs_input_grad[8]
        grads = torch.ops.gsplat_sensors.image_points_to_world_rays_static_pose_ftheta_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            translations,
            rotations,
            grad_world_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[5],
            ctx.needs_input_grad[6],
            ctx.needs_input_grad[7],
            need_A,
            need_A,
        )
        grad_pts, grad_t, grad_r, grad_pp, grad_fw, grad_bw, grad_A, grad_Ainv = grads
        if need_A:
            grad_A = _consolidate_ftheta_grad_a(grad_A, grad_Ainv, ctx.projection)
        return (
            grad_pts if ctx.needs_input_grad[0] else None,
            grad_t if ctx.needs_input_grad[1] else None,
            grad_r if ctx.needs_input_grad[2] else None,
            None,
            None,
            grad_pp if ctx.needs_input_grad[5] else None,
            grad_fw if ctx.needs_input_grad[6] else None,
            grad_bw if ctx.needs_input_grad[7] else None,
            grad_A if need_A else None,
            None,
        )


class _ImagePointsToWorldRaysStaticPoseFThetaBivariateWindshield(
    torch.autograd.Function
):
    """Autograd Function wrapping ``image_points_to_world_rays_static_pose_ftheta_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        translations: Tensor,
        rotations: Tensor,
        projection: FThetaProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        principal_point: Tensor,
        fw_poly: Tensor,
        bw_poly: Tensor,
        A: Tensor,
        timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del distortion_coeffs, principal_point, fw_poly, bw_poly, A
        (
            world_rays,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_world_rays_static_pose_ftheta_bivariate_windshield(
            projection,
            external_distortion,
            image_points,
            translations,
            rotations,
            int(timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(image_points, translations, rotations, scratch)
        ctx.mark_non_differentiable(timestamps_us)
        return world_rays, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_world_rays: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_timestamps_us, grad_pose_t, grad_pose_r
        image_points, translations, rotations, scratch = ctx.saved_tensors
        if grad_world_rays is None:
            grad_world_rays = _zero_like((image_points.shape[0], 6), image_points)
        need_A = ctx.needs_input_grad[9]
        grads = torch.ops.gsplat_sensors.image_points_to_world_rays_static_pose_ftheta_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            translations,
            rotations,
            grad_world_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[6],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            need_A,
            need_A,
            ctx.needs_input_grad[5],
        )
        (
            grad_pts,
            grad_t,
            grad_r,
            grad_pp,
            grad_fw,
            grad_bw,
            grad_A,
            grad_Ainv,
            grad_distortion,
        ) = grads
        if need_A:
            grad_A = _consolidate_ftheta_grad_a(grad_A, grad_Ainv, ctx.projection)
        return (
            grad_pts if ctx.needs_input_grad[0] else None,
            grad_t if ctx.needs_input_grad[1] else None,
            grad_r if ctx.needs_input_grad[2] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[5] else None,
            grad_pp if ctx.needs_input_grad[6] else None,
            grad_fw if ctx.needs_input_grad[7] else None,
            grad_bw if ctx.needs_input_grad[8] else None,
            grad_A if need_A else None,
            None,
        )


class _ProjectWorldPointsShutterPoseFTheta(torch.autograd.Function):
    """Autograd Function wrapping ``project_world_points_shutter_pose_ftheta_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: FThetaProjection,
        external_distortion: NoExternalDistortion,
        principal_point: Tensor,
        fw_poly: Tensor,
        bw_poly: Tensor,
        A: Tensor,
        width: int,
        height: int,
        shutter_type: int,
        start_timestamp_us: int,
        end_timestamp_us: int,
        max_iterations: int,
        stop_mean_error_px: float,
        stop_delta_mean_error_px: float,
        initial_relative_time: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        del principal_point, fw_poly, bw_poly, A
        (
            image_points,
            valid_flags,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.project_world_points_shutter_pose_ftheta_no_external(
            projection,
            external_distortion,
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(width),
            int(height),
            int(shutter_type),
            int(start_timestamp_us),
            int(end_timestamp_us),
            int(max_iterations),
            float(stop_mean_error_px),
            float(stop_delta_mean_error_px),
            float(initial_relative_time),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.shutter_type = int(shutter_type)
        ctx.max_iterations = int(max_iterations)
        ctx.initial_relative_time = float(initial_relative_time)
        ctx.save_for_backward(
            world_points, start_rotation, end_rotation, valid_flags, scratch
        )
        ctx.mark_non_differentiable(valid_flags, timestamps_us)
        return image_points, valid_flags, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_image_points: Tensor | None,
        grad_valid_flags: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_valid_flags, grad_timestamps_us, grad_pose_t, grad_pose_r
        (
            world_points,
            start_rotation,
            end_rotation,
            valid_flags,
            scratch,
        ) = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((world_points.shape[0], 2), world_points)
        need_A = ctx.needs_input_grad[10]
        grads = torch.ops.gsplat_sensors.project_world_points_shutter_pose_ftheta_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            world_points,
            start_rotation,
            end_rotation,
            ctx.shutter_type,
            ctx.max_iterations,
            ctx.initial_relative_time,
            valid_flags,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            need_A,
            need_A,
        )
        (
            grad_world,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_pp,
            grad_fw,
            grad_bw,
            grad_A,
            grad_Ainv,
        ) = grads
        if need_A:
            grad_A = _consolidate_ftheta_grad_a(grad_A, grad_Ainv, ctx.projection)
        return (
            grad_world if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_pp if ctx.needs_input_grad[7] else None,
            grad_fw if ctx.needs_input_grad[8] else None,
            grad_bw if ctx.needs_input_grad[9] else None,
            grad_A if need_A else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _ProjectWorldPointsShutterPoseFThetaBivariateWindshield(torch.autograd.Function):
    """Autograd Function wrapping ``project_world_points_shutter_pose_ftheta_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: FThetaProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        principal_point: Tensor,
        fw_poly: Tensor,
        bw_poly: Tensor,
        A: Tensor,
        width: int,
        height: int,
        shutter_type: int,
        start_timestamp_us: int,
        end_timestamp_us: int,
        max_iterations: int,
        stop_mean_error_px: float,
        stop_delta_mean_error_px: float,
        initial_relative_time: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        del distortion_coeffs, principal_point, fw_poly, bw_poly, A
        (
            image_points,
            valid_flags,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.project_world_points_shutter_pose_ftheta_bivariate_windshield(
            projection,
            external_distortion,
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(width),
            int(height),
            int(shutter_type),
            int(start_timestamp_us),
            int(end_timestamp_us),
            int(max_iterations),
            float(stop_mean_error_px),
            float(stop_delta_mean_error_px),
            float(initial_relative_time),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.shutter_type = int(shutter_type)
        ctx.max_iterations = int(max_iterations)
        ctx.initial_relative_time = float(initial_relative_time)
        ctx.save_for_backward(
            world_points, start_rotation, end_rotation, valid_flags, scratch
        )
        ctx.mark_non_differentiable(valid_flags, timestamps_us)
        return image_points, valid_flags, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_image_points: Tensor | None,
        grad_valid_flags: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_valid_flags, grad_timestamps_us, grad_pose_t, grad_pose_r
        (
            world_points,
            start_rotation,
            end_rotation,
            valid_flags,
            scratch,
        ) = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((world_points.shape[0], 2), world_points)
        need_A = ctx.needs_input_grad[11]
        grads = torch.ops.gsplat_sensors.project_world_points_shutter_pose_ftheta_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            world_points,
            start_rotation,
            end_rotation,
            ctx.shutter_type,
            ctx.max_iterations,
            ctx.initial_relative_time,
            valid_flags,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            ctx.needs_input_grad[10],
            need_A,
            need_A,
            ctx.needs_input_grad[7],
        )
        (
            grad_world,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_pp,
            grad_fw,
            grad_bw,
            grad_A,
            grad_Ainv,
            grad_distortion,
        ) = grads
        if need_A:
            grad_A = _consolidate_ftheta_grad_a(grad_A, grad_Ainv, ctx.projection)
        return (
            grad_world if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[7] else None,
            grad_pp if ctx.needs_input_grad[8] else None,
            grad_fw if ctx.needs_input_grad[9] else None,
            grad_bw if ctx.needs_input_grad[10] else None,
            grad_A if need_A else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _ImagePointsToWorldRaysShutterPoseFTheta(torch.autograd.Function):
    """Autograd Function wrapping ``image_points_to_world_rays_shutter_pose_ftheta_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: FThetaProjection,
        external_distortion: NoExternalDistortion,
        principal_point: Tensor,
        fw_poly: Tensor,
        bw_poly: Tensor,
        A: Tensor,
        width: int,
        height: int,
        shutter_type: int,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del principal_point, fw_poly, bw_poly, A
        (
            world_rays,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_world_rays_shutter_pose_ftheta_no_external(
            projection,
            external_distortion,
            image_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(width),
            int(height),
            int(shutter_type),
            int(start_timestamp_us),
            int(end_timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.shutter_type = int(shutter_type)
        ctx.save_for_backward(image_points, start_rotation, end_rotation, scratch)
        ctx.mark_non_differentiable(timestamps_us)
        return world_rays, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_world_rays: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_timestamps_us, grad_pose_t, grad_pose_r
        image_points, start_rotation, end_rotation, scratch = ctx.saved_tensors
        if grad_world_rays is None:
            grad_world_rays = _zero_like((image_points.shape[0], 6), image_points)
        need_A = ctx.needs_input_grad[10]
        grads = torch.ops.gsplat_sensors.image_points_to_world_rays_shutter_pose_ftheta_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            start_rotation,
            end_rotation,
            ctx.shutter_type,
            grad_world_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            need_A,
            need_A,
        )
        (
            grad_points,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_pp,
            grad_fw,
            grad_bw,
            grad_A,
            grad_Ainv,
        ) = grads
        if need_A:
            grad_A = _consolidate_ftheta_grad_a(grad_A, grad_Ainv, ctx.projection)
        return (
            grad_points if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_pp if ctx.needs_input_grad[7] else None,
            grad_fw if ctx.needs_input_grad[8] else None,
            grad_bw if ctx.needs_input_grad[9] else None,
            grad_A if need_A else None,
            None,
            None,
            None,
            None,
            None,
        )


class _ImagePointsToWorldRaysShutterPoseFThetaBivariateWindshield(
    torch.autograd.Function
):
    """Autograd Function wrapping ``image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: FThetaProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        principal_point: Tensor,
        fw_poly: Tensor,
        bw_poly: Tensor,
        A: Tensor,
        width: int,
        height: int,
        shutter_type: int,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del distortion_coeffs, principal_point, fw_poly, bw_poly, A
        (
            world_rays,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield(
            projection,
            external_distortion,
            image_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(width),
            int(height),
            int(shutter_type),
            int(start_timestamp_us),
            int(end_timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.shutter_type = int(shutter_type)
        ctx.save_for_backward(image_points, start_rotation, end_rotation, scratch)
        ctx.mark_non_differentiable(timestamps_us)
        return world_rays, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_world_rays: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_timestamps_us, grad_pose_t, grad_pose_r
        image_points, start_rotation, end_rotation, scratch = ctx.saved_tensors
        if grad_world_rays is None:
            grad_world_rays = _zero_like((image_points.shape[0], 6), image_points)
        need_A = ctx.needs_input_grad[11]
        grads = torch.ops.gsplat_sensors.image_points_to_world_rays_shutter_pose_ftheta_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            start_rotation,
            end_rotation,
            ctx.shutter_type,
            grad_world_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            ctx.needs_input_grad[10],
            need_A,
            need_A,
            ctx.needs_input_grad[7],
        )
        (
            grad_points,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_pp,
            grad_fw,
            grad_bw,
            grad_A,
            grad_Ainv,
            grad_distortion,
        ) = grads
        if need_A:
            grad_A = _consolidate_ftheta_grad_a(grad_A, grad_Ainv, ctx.projection)
        return (
            grad_points if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[7] else None,
            grad_pp if ctx.needs_input_grad[8] else None,
            grad_fw if ctx.needs_input_grad[9] else None,
            grad_bw if ctx.needs_input_grad[10] else None,
            grad_A if need_A else None,
            None,
            None,
            None,
            None,
            None,
        )


# =============================================================================
# OpenCV-fisheye autograd Function classes (6 ops x 2 distortion variants).
# Intrinsics are the 4-tuple (principal_point, focal_length, forward_poly,
# approx_backward_factor). approx_backward_factor receives no gradient: it is
# threaded as a positional arg, del'd in forward, and always returns None.
# There is no A/Ainv matrix, so no grad-A consolidation.
# =============================================================================


class _CameraRaysToImagePointsFisheye(torch.autograd.Function):
    """Autograd Function wrapping ``camera_rays_to_image_points_opencv_fisheye_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        camera_rays: Tensor,
        projection: OpenCVFisheyeProjection,
        external_distortion: NoExternalDistortion,
        principal_point: Tensor,
        focal_length: Tensor,
        forward_poly: Tensor,
        approx_backward_factor: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del principal_point, focal_length, forward_poly, approx_backward_factor
        (
            image_points,
            valid_flags,
            scratch,
        ) = torch.ops.gsplat_sensors.camera_rays_to_image_points_opencv_fisheye_no_external(
            projection, external_distortion, camera_rays
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(camera_rays, scratch)
        ctx.mark_non_differentiable(valid_flags)
        return image_points, valid_flags

    @staticmethod
    def backward(
        ctx, grad_image_points: Tensor | None, grad_valid_flags: Tensor | None
    ):
        del grad_valid_flags
        camera_rays, scratch = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((camera_rays.shape[0], 2), camera_rays)
        grads = torch.ops.gsplat_sensors.camera_rays_to_image_points_opencv_fisheye_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            camera_rays,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[5],
        )
        grad_camera_rays, grad_pp, grad_focal, grad_fw = grads
        return (
            grad_camera_rays if ctx.needs_input_grad[0] else None,
            None,
            None,
            grad_pp if ctx.needs_input_grad[3] else None,
            grad_focal if ctx.needs_input_grad[4] else None,
            grad_fw if ctx.needs_input_grad[5] else None,
            None,
        )


class _ImagePointsToCameraRaysFisheye(torch.autograd.Function):
    """Autograd Function wrapping ``image_points_to_camera_rays_opencv_fisheye_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        projection: OpenCVFisheyeProjection,
        external_distortion: NoExternalDistortion,
        principal_point: Tensor,
        focal_length: Tensor,
        forward_poly: Tensor,
        approx_backward_factor: Tensor,
    ) -> Tensor:
        del principal_point, focal_length, forward_poly, approx_backward_factor
        (
            camera_rays,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_camera_rays_opencv_fisheye_no_external(
            projection, external_distortion, image_points
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(image_points, scratch)
        return camera_rays

    @staticmethod
    def backward(ctx, grad_camera_rays: Tensor | None):
        image_points, scratch = ctx.saved_tensors
        if grad_camera_rays is None:
            grad_camera_rays = _zero_like((image_points.shape[0], 3), image_points)
        grads = torch.ops.gsplat_sensors.image_points_to_camera_rays_opencv_fisheye_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            grad_camera_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[5],
        )
        grad_image_points, grad_pp, grad_focal, grad_fw = grads
        return (
            grad_image_points if ctx.needs_input_grad[0] else None,
            None,
            None,
            grad_pp if ctx.needs_input_grad[3] else None,
            grad_focal if ctx.needs_input_grad[4] else None,
            grad_fw if ctx.needs_input_grad[5] else None,
            None,
        )


class _CameraRaysToImagePointsFisheyeBivariateWindshield(torch.autograd.Function):
    """Autograd Function wrapping ``camera_rays_to_image_points_opencv_fisheye_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        camera_rays: Tensor,
        projection: OpenCVFisheyeProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        principal_point: Tensor,
        focal_length: Tensor,
        forward_poly: Tensor,
        approx_backward_factor: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del distortion_coeffs, principal_point, focal_length
        del forward_poly, approx_backward_factor
        (
            image_points,
            valid_flags,
            scratch,
        ) = torch.ops.gsplat_sensors.camera_rays_to_image_points_opencv_fisheye_bivariate_windshield(
            projection, external_distortion, camera_rays
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(camera_rays, scratch)
        ctx.mark_non_differentiable(valid_flags)
        return image_points, valid_flags

    @staticmethod
    def backward(
        ctx, grad_image_points: Tensor | None, grad_valid_flags: Tensor | None
    ):
        del grad_valid_flags
        camera_rays, scratch = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((camera_rays.shape[0], 2), camera_rays)
        grads = torch.ops.gsplat_sensors.camera_rays_to_image_points_opencv_fisheye_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            camera_rays,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[5],
            ctx.needs_input_grad[6],
            ctx.needs_input_grad[3],
        )
        grad_camera_rays, grad_pp, grad_focal, grad_fw, grad_distortion = grads
        return (
            grad_camera_rays if ctx.needs_input_grad[0] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[3] else None,
            grad_pp if ctx.needs_input_grad[4] else None,
            grad_focal if ctx.needs_input_grad[5] else None,
            grad_fw if ctx.needs_input_grad[6] else None,
            None,
        )


class _ImagePointsToCameraRaysFisheyeBivariateWindshield(torch.autograd.Function):
    """Autograd Function wrapping ``image_points_to_camera_rays_opencv_fisheye_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        projection: OpenCVFisheyeProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        principal_point: Tensor,
        focal_length: Tensor,
        forward_poly: Tensor,
        approx_backward_factor: Tensor,
    ) -> Tensor:
        del distortion_coeffs, principal_point, focal_length
        del forward_poly, approx_backward_factor
        (
            camera_rays,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_camera_rays_opencv_fisheye_bivariate_windshield(
            projection, external_distortion, image_points
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(image_points, scratch)
        return camera_rays

    @staticmethod
    def backward(ctx, grad_camera_rays: Tensor | None):
        image_points, scratch = ctx.saved_tensors
        if grad_camera_rays is None:
            grad_camera_rays = _zero_like((image_points.shape[0], 3), image_points)
        grads = torch.ops.gsplat_sensors.image_points_to_camera_rays_opencv_fisheye_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            grad_camera_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[5],
            ctx.needs_input_grad[6],
            ctx.needs_input_grad[3],
        )
        grad_image_points, grad_pp, grad_focal, grad_fw, grad_distortion = grads
        return (
            grad_image_points if ctx.needs_input_grad[0] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[3] else None,
            grad_pp if ctx.needs_input_grad[4] else None,
            grad_focal if ctx.needs_input_grad[5] else None,
            grad_fw if ctx.needs_input_grad[6] else None,
            None,
        )


class _ProjectWorldPointsMeanPoseFisheye(torch.autograd.Function):
    """Autograd Function wrapping ``project_world_points_mean_pose_opencv_fisheye_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: OpenCVFisheyeProjection,
        external_distortion: NoExternalDistortion,
        principal_point: Tensor,
        focal_length: Tensor,
        forward_poly: Tensor,
        approx_backward_factor: Tensor,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        del principal_point, focal_length, forward_poly, approx_backward_factor
        (
            image_points,
            valid_flags,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.project_world_points_mean_pose_opencv_fisheye_no_external(
            projection,
            external_distortion,
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(start_timestamp_us),
            int(end_timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(world_points, start_rotation, end_rotation, scratch)
        ctx.mark_non_differentiable(valid_flags, timestamps_us)
        return image_points, valid_flags, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_image_points: Tensor | None,
        grad_valid_flags: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_valid_flags, grad_timestamps_us, grad_pose_t, grad_pose_r
        world_points, start_rotation, end_rotation, scratch = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((world_points.shape[0], 2), world_points)
        grads = torch.ops.gsplat_sensors.project_world_points_mean_pose_opencv_fisheye_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            world_points,
            start_rotation,
            end_rotation,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
        )
        (
            grad_world,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_pp,
            grad_focal,
            grad_fw,
        ) = grads
        return (
            grad_world if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_pp if ctx.needs_input_grad[7] else None,
            grad_focal if ctx.needs_input_grad[8] else None,
            grad_fw if ctx.needs_input_grad[9] else None,
            None,
            None,
            None,
        )


class _ProjectWorldPointsMeanPoseFisheyeBivariateWindshield(torch.autograd.Function):
    """Autograd Function wrapping ``project_world_points_mean_pose_opencv_fisheye_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: OpenCVFisheyeProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        principal_point: Tensor,
        focal_length: Tensor,
        forward_poly: Tensor,
        approx_backward_factor: Tensor,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        del distortion_coeffs, principal_point, focal_length
        del forward_poly, approx_backward_factor
        (
            image_points,
            valid_flags,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.project_world_points_mean_pose_opencv_fisheye_bivariate_windshield(
            projection,
            external_distortion,
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(start_timestamp_us),
            int(end_timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(world_points, start_rotation, end_rotation, scratch)
        ctx.mark_non_differentiable(valid_flags, timestamps_us)
        return image_points, valid_flags, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_image_points: Tensor | None,
        grad_valid_flags: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_valid_flags, grad_timestamps_us, grad_pose_t, grad_pose_r
        world_points, start_rotation, end_rotation, scratch = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((world_points.shape[0], 2), world_points)
        grads = torch.ops.gsplat_sensors.project_world_points_mean_pose_opencv_fisheye_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            world_points,
            start_rotation,
            end_rotation,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            ctx.needs_input_grad[10],
            ctx.needs_input_grad[7],
        )
        (
            grad_world,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_pp,
            grad_focal,
            grad_fw,
            grad_distortion,
        ) = grads
        return (
            grad_world if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[7] else None,
            grad_pp if ctx.needs_input_grad[8] else None,
            grad_focal if ctx.needs_input_grad[9] else None,
            grad_fw if ctx.needs_input_grad[10] else None,
            None,
            None,
            None,
        )


class _ImagePointsToWorldRaysStaticPoseFisheye(torch.autograd.Function):
    """Autograd Function wrapping ``image_points_to_world_rays_static_pose_opencv_fisheye_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        translations: Tensor,
        rotations: Tensor,
        projection: OpenCVFisheyeProjection,
        external_distortion: NoExternalDistortion,
        principal_point: Tensor,
        focal_length: Tensor,
        forward_poly: Tensor,
        approx_backward_factor: Tensor,
        timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del principal_point, focal_length, forward_poly, approx_backward_factor
        (
            world_rays,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_world_rays_static_pose_opencv_fisheye_no_external(
            projection,
            external_distortion,
            image_points,
            translations,
            rotations,
            int(timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(image_points, translations, rotations, scratch)
        ctx.mark_non_differentiable(timestamps_us)
        return world_rays, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_world_rays: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_timestamps_us, grad_pose_t, grad_pose_r
        image_points, translations, rotations, scratch = ctx.saved_tensors
        if grad_world_rays is None:
            grad_world_rays = _zero_like((image_points.shape[0], 6), image_points)
        grads = torch.ops.gsplat_sensors.image_points_to_world_rays_static_pose_opencv_fisheye_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            translations,
            rotations,
            grad_world_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[5],
            ctx.needs_input_grad[6],
            ctx.needs_input_grad[7],
        )
        grad_pts, grad_t, grad_r, grad_pp, grad_focal, grad_fw = grads
        return (
            grad_pts if ctx.needs_input_grad[0] else None,
            grad_t if ctx.needs_input_grad[1] else None,
            grad_r if ctx.needs_input_grad[2] else None,
            None,
            None,
            grad_pp if ctx.needs_input_grad[5] else None,
            grad_focal if ctx.needs_input_grad[6] else None,
            grad_fw if ctx.needs_input_grad[7] else None,
            None,
            None,
        )


class _ImagePointsToWorldRaysStaticPoseFisheyeBivariateWindshield(
    torch.autograd.Function
):
    """Autograd Function wrapping ``image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        translations: Tensor,
        rotations: Tensor,
        projection: OpenCVFisheyeProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        principal_point: Tensor,
        focal_length: Tensor,
        forward_poly: Tensor,
        approx_backward_factor: Tensor,
        timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del distortion_coeffs, principal_point, focal_length
        del forward_poly, approx_backward_factor
        (
            world_rays,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield(
            projection,
            external_distortion,
            image_points,
            translations,
            rotations,
            int(timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(image_points, translations, rotations, scratch)
        ctx.mark_non_differentiable(timestamps_us)
        return world_rays, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_world_rays: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_timestamps_us, grad_pose_t, grad_pose_r
        image_points, translations, rotations, scratch = ctx.saved_tensors
        if grad_world_rays is None:
            grad_world_rays = _zero_like((image_points.shape[0], 6), image_points)
        grads = torch.ops.gsplat_sensors.image_points_to_world_rays_static_pose_opencv_fisheye_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            translations,
            rotations,
            grad_world_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[6],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[5],
        )
        (
            grad_pts,
            grad_t,
            grad_r,
            grad_pp,
            grad_focal,
            grad_fw,
            grad_distortion,
        ) = grads
        return (
            grad_pts if ctx.needs_input_grad[0] else None,
            grad_t if ctx.needs_input_grad[1] else None,
            grad_r if ctx.needs_input_grad[2] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[5] else None,
            grad_pp if ctx.needs_input_grad[6] else None,
            grad_focal if ctx.needs_input_grad[7] else None,
            grad_fw if ctx.needs_input_grad[8] else None,
            None,
            None,
        )


class _ProjectWorldPointsShutterPoseFisheye(torch.autograd.Function):
    """Autograd Function wrapping ``project_world_points_shutter_pose_opencv_fisheye_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: OpenCVFisheyeProjection,
        external_distortion: NoExternalDistortion,
        principal_point: Tensor,
        focal_length: Tensor,
        forward_poly: Tensor,
        approx_backward_factor: Tensor,
        width: int,
        height: int,
        shutter_type: int,
        start_timestamp_us: int,
        end_timestamp_us: int,
        max_iterations: int,
        stop_mean_error_px: float,
        stop_delta_mean_error_px: float,
        initial_relative_time: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        del principal_point, focal_length, forward_poly, approx_backward_factor
        (
            image_points,
            valid_flags,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.project_world_points_shutter_pose_opencv_fisheye_no_external(
            projection,
            external_distortion,
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(width),
            int(height),
            int(shutter_type),
            int(start_timestamp_us),
            int(end_timestamp_us),
            int(max_iterations),
            float(stop_mean_error_px),
            float(stop_delta_mean_error_px),
            float(initial_relative_time),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(start_rotation, end_rotation, scratch)
        ctx.mark_non_differentiable(valid_flags, timestamps_us)
        return image_points, valid_flags, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_image_points: Tensor | None,
        grad_valid_flags: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_valid_flags, grad_timestamps_us, grad_pose_t, grad_pose_r
        start_rotation, end_rotation, scratch = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((scratch.shape[0], 2), scratch)
        grads = torch.ops.gsplat_sensors.project_world_points_shutter_pose_opencv_fisheye_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            start_rotation,
            end_rotation,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
        )
        (
            grad_world,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_pp,
            grad_focal,
            grad_fw,
        ) = grads
        return (
            grad_world if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_pp if ctx.needs_input_grad[7] else None,
            grad_focal if ctx.needs_input_grad[8] else None,
            grad_fw if ctx.needs_input_grad[9] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _ProjectWorldPointsShutterPoseFisheyeBivariateWindshield(torch.autograd.Function):
    """Autograd Function wrapping ``project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: OpenCVFisheyeProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        principal_point: Tensor,
        focal_length: Tensor,
        forward_poly: Tensor,
        approx_backward_factor: Tensor,
        width: int,
        height: int,
        shutter_type: int,
        start_timestamp_us: int,
        end_timestamp_us: int,
        max_iterations: int,
        stop_mean_error_px: float,
        stop_delta_mean_error_px: float,
        initial_relative_time: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        del distortion_coeffs, principal_point, focal_length
        del forward_poly, approx_backward_factor
        (
            image_points,
            valid_flags,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield(
            projection,
            external_distortion,
            world_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(width),
            int(height),
            int(shutter_type),
            int(start_timestamp_us),
            int(end_timestamp_us),
            int(max_iterations),
            float(stop_mean_error_px),
            float(stop_delta_mean_error_px),
            float(initial_relative_time),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(start_rotation, end_rotation, scratch)
        ctx.mark_non_differentiable(valid_flags, timestamps_us)
        return image_points, valid_flags, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_image_points: Tensor | None,
        grad_valid_flags: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_valid_flags, grad_timestamps_us, grad_pose_t, grad_pose_r
        start_rotation, end_rotation, scratch = ctx.saved_tensors
        if grad_image_points is None:
            grad_image_points = _zero_like((scratch.shape[0], 2), scratch)
        grads = torch.ops.gsplat_sensors.project_world_points_shutter_pose_opencv_fisheye_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            start_rotation,
            end_rotation,
            grad_image_points.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            ctx.needs_input_grad[10],
            ctx.needs_input_grad[7],
        )
        (
            grad_world,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_pp,
            grad_focal,
            grad_fw,
            grad_distortion,
        ) = grads
        return (
            grad_world if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[7] else None,
            grad_pp if ctx.needs_input_grad[8] else None,
            grad_focal if ctx.needs_input_grad[9] else None,
            grad_fw if ctx.needs_input_grad[10] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _ImagePointsToWorldRaysShutterPoseFisheye(torch.autograd.Function):
    """Autograd Function wrapping ``image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: OpenCVFisheyeProjection,
        external_distortion: NoExternalDistortion,
        principal_point: Tensor,
        focal_length: Tensor,
        forward_poly: Tensor,
        approx_backward_factor: Tensor,
        width: int,
        height: int,
        shutter_type: int,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del principal_point, focal_length, forward_poly, approx_backward_factor
        (
            world_rays,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external(
            projection,
            external_distortion,
            image_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(width),
            int(height),
            int(shutter_type),
            int(start_timestamp_us),
            int(end_timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(image_points, start_rotation, end_rotation, scratch)
        ctx.mark_non_differentiable(timestamps_us)
        return world_rays, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_world_rays: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_timestamps_us, grad_pose_t, grad_pose_r
        image_points, start_rotation, end_rotation, scratch = ctx.saved_tensors
        if grad_world_rays is None:
            grad_world_rays = _zero_like((image_points.shape[0], 6), image_points)
        grads = torch.ops.gsplat_sensors.image_points_to_world_rays_shutter_pose_opencv_fisheye_no_external_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            start_rotation,
            end_rotation,
            grad_world_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[7],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
        )
        (
            grad_points,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_pp,
            grad_focal,
            grad_fw,
        ) = grads
        return (
            grad_points if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_pp if ctx.needs_input_grad[7] else None,
            grad_focal if ctx.needs_input_grad[8] else None,
            grad_fw if ctx.needs_input_grad[9] else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _ImagePointsToWorldRaysShutterPoseFisheyeBivariateWindshield(
    torch.autograd.Function
):
    """Autograd Function wrapping ``image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor,
        start_translation: Tensor,
        start_rotation: Tensor,
        end_translation: Tensor,
        end_rotation: Tensor,
        projection: OpenCVFisheyeProjection,
        external_distortion: BivariateWindshieldDistortion,
        distortion_coeffs: Tensor,
        principal_point: Tensor,
        focal_length: Tensor,
        forward_poly: Tensor,
        approx_backward_factor: Tensor,
        width: int,
        height: int,
        shutter_type: int,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del distortion_coeffs, principal_point, focal_length
        del forward_poly, approx_backward_factor
        (
            world_rays,
            timestamps_us,
            pose_t,
            pose_r,
            scratch,
        ) = torch.ops.gsplat_sensors.image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield(
            projection,
            external_distortion,
            image_points,
            start_translation,
            start_rotation,
            end_translation,
            end_rotation,
            int(width),
            int(height),
            int(shutter_type),
            int(start_timestamp_us),
            int(end_timestamp_us),
        )
        ctx.projection = projection
        ctx.external_distortion = external_distortion
        ctx.save_for_backward(image_points, start_rotation, end_rotation, scratch)
        ctx.mark_non_differentiable(timestamps_us)
        return world_rays, timestamps_us, pose_t, pose_r

    @staticmethod
    def backward(
        ctx,
        grad_world_rays: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_pose_t: Tensor | None,
        grad_pose_r: Tensor | None,
    ):
        del grad_timestamps_us, grad_pose_t, grad_pose_r
        image_points, start_rotation, end_rotation, scratch = ctx.saved_tensors
        if grad_world_rays is None:
            grad_world_rays = _zero_like((image_points.shape[0], 6), image_points)
        grads = torch.ops.gsplat_sensors.image_points_to_world_rays_shutter_pose_opencv_fisheye_bivariate_windshield_backward(
            ctx.projection,
            ctx.external_distortion,
            image_points,
            start_rotation,
            end_rotation,
            grad_world_rays.contiguous(),
            scratch,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[3],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[4],
            ctx.needs_input_grad[8],
            ctx.needs_input_grad[9],
            ctx.needs_input_grad[10],
            ctx.needs_input_grad[7],
        )
        (
            grad_points,
            grad_start_t,
            grad_end_t,
            grad_start_r,
            grad_end_r,
            grad_pp,
            grad_focal,
            grad_fw,
            grad_distortion,
        ) = grads
        return (
            grad_points if ctx.needs_input_grad[0] else None,
            grad_start_t if ctx.needs_input_grad[1] else None,
            grad_start_r if ctx.needs_input_grad[2] else None,
            grad_end_t if ctx.needs_input_grad[3] else None,
            grad_end_r if ctx.needs_input_grad[4] else None,
            None,
            None,
            grad_distortion if ctx.needs_input_grad[7] else None,
            grad_pp if ctx.needs_input_grad[8] else None,
            grad_focal if ctx.needs_input_grad[9] else None,
            grad_fw if ctx.needs_input_grad[10] else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# op_name -> (no_external_fn, bivariate_fn)
_FISHEYE_OP_TABLE = {
    "camera_rays_to_image_points": (
        _CameraRaysToImagePointsFisheye,
        _CameraRaysToImagePointsFisheyeBivariateWindshield,
    ),
    "image_points_to_camera_rays": (
        _ImagePointsToCameraRaysFisheye,
        _ImagePointsToCameraRaysFisheyeBivariateWindshield,
    ),
    "project_world_points_mean_pose": (
        _ProjectWorldPointsMeanPoseFisheye,
        _ProjectWorldPointsMeanPoseFisheyeBivariateWindshield,
    ),
    "image_points_to_world_rays_static_pose": (
        _ImagePointsToWorldRaysStaticPoseFisheye,
        _ImagePointsToWorldRaysStaticPoseFisheyeBivariateWindshield,
    ),
    "project_world_points_shutter_pose": (
        _ProjectWorldPointsShutterPoseFisheye,
        _ProjectWorldPointsShutterPoseFisheyeBivariateWindshield,
    ),
    "image_points_to_world_rays_shutter_pose": (
        _ImagePointsToWorldRaysShutterPoseFisheye,
        _ImagePointsToWorldRaysShutterPoseFisheyeBivariateWindshield,
    ),
}


# op_name -> (no_external_fn, bivariate_fn)
_FTHETA_OP_TABLE = {
    "camera_rays_to_image_points": (
        _CameraRaysToImagePointsFTheta,
        _CameraRaysToImagePointsFThetaBivariateWindshield,
    ),
    "image_points_to_camera_rays": (
        _ImagePointsToCameraRaysFTheta,
        _ImagePointsToCameraRaysFThetaBivariateWindshield,
    ),
    "project_world_points_mean_pose": (
        _ProjectWorldPointsMeanPoseFTheta,
        _ProjectWorldPointsMeanPoseFThetaBivariateWindshield,
    ),
    "image_points_to_world_rays_static_pose": (
        _ImagePointsToWorldRaysStaticPoseFTheta,
        _ImagePointsToWorldRaysStaticPoseFThetaBivariateWindshield,
    ),
    "project_world_points_shutter_pose": (
        _ProjectWorldPointsShutterPoseFTheta,
        _ProjectWorldPointsShutterPoseFThetaBivariateWindshield,
    ),
    "image_points_to_world_rays_shutter_pose": (
        _ImagePointsToWorldRaysShutterPoseFTheta,
        _ImagePointsToWorldRaysShutterPoseFThetaBivariateWindshield,
    ),
}


def _select_op(
    projection: CameraProjection,
    op_name: str,
    pinhole_no_external: type,
    pinhole_bivariate: type,
    external_distortion: ExternalDistortion,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool,
) -> tuple[object, object, tuple]:
    """Dispatch on (projection family, distortion family) -> autograd Function.

    Mirrors :func:`_select_camera_op` for pinhole; for FTheta projections the
    classes come from ``_FTHETA_OP_TABLE``.
    """
    projection_class = script_class_name(projection)
    if projection_class == "OpenCVPinholeProjection":
        no_external, bivariate = pinhole_no_external, pinhole_bivariate
    elif projection_class == "FThetaProjection":
        no_external, bivariate = _FTHETA_OP_TABLE[op_name]
    elif projection_class == "OpenCVFisheyeProjection":
        no_external, bivariate = _FISHEYE_OP_TABLE[op_name]
    else:
        raise TypeError(f"Unknown camera projection class: {projection_class}")
    return _select_camera_op(
        external_distortion,
        no_external,
        bivariate,
        device,
        dtype,
        allow_device_transfer,
    )


def camera_rays_to_image_points(
    camera_rays: Tensor,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    *,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor]:
    """Project camera-frame rays to image-plane pixel coordinates.

    Supports both NoExternalDistortion and BivariateWindshieldDistortion variants.

    Args:
        camera_rays: (N, 3) normalized ray directions in camera frame.
        projection: Registered camera projection parameters.
        external_distortion: External distortion parameters (NoExternalDistortion or
            BivariateWindshieldDistortion).
        allow_device_transfer: If False (default), raises if any tensor needs a
            device or dtype transfer before the kernel launches.

    Returns:
        image_points: (N, 2) pixel coordinates.
        valid_flags: (N,) bool mask indicating which projections are within the image.
    """
    _check_pair(projection, external_distortion)
    device = _raise_or_target_device(camera_rays, allow_device_transfer)
    rays = _to_dev(camera_rays, device, torch.float32, allow_device_transfer)
    projection_cuda, tensors = _projection_on_device_any(
        projection, device, torch.float32, allow_device_transfer
    )
    apply_fn, distortion_arg, extra_distortion_args = _select_op(
        projection,
        "camera_rays_to_image_points",
        _CameraRaysToImagePoints,
        _CameraRaysToImagePointsBivariateWindshield,
        external_distortion,
        device,
        torch.float32,
        allow_device_transfer,
    )
    return apply_fn(
        rays,
        projection_cuda,
        distortion_arg,
        *extra_distortion_args,
        *tensors,
    )


def image_points_to_camera_rays(
    image_points: Tensor,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    *,
    allow_device_transfer: bool = False,
) -> Tensor:
    """Back-project image-plane pixel coordinates to normalized camera-frame rays.

    Supports both NoExternalDistortion and BivariateWindshieldDistortion variants.

    Args:
        image_points: (N, 2) pixel coordinates.
        projection: Registered camera projection parameters.
        external_distortion: External distortion parameters (NoExternalDistortion or
            BivariateWindshieldDistortion).
        allow_device_transfer: If False (default), raises if any tensor needs a
            device or dtype transfer before the kernel launches.

    Returns:
        camera_rays: (N, 3) normalized ray directions in camera frame.
    """
    _check_pair(projection, external_distortion)
    device = _raise_or_target_device(image_points, allow_device_transfer)
    pts = _to_dev(image_points, device, torch.float32, allow_device_transfer)
    projection_cuda, tensors = _projection_on_device_any(
        projection, device, torch.float32, allow_device_transfer
    )
    apply_fn, distortion_arg, extra_distortion_args = _select_op(
        projection,
        "image_points_to_camera_rays",
        _ImagePointsToCameraRays,
        _ImagePointsToCameraRaysBivariateWindshield,
        external_distortion,
        device,
        torch.float32,
        allow_device_transfer,
    )
    return apply_fn(
        pts,
        projection_cuda,
        distortion_arg,
        *extra_distortion_args,
        *tensors,
    )


def generate_image_points(
    resolution: tuple[int, int],
    *,
    device: torch.device | str = "cuda",
    allow_device_transfer: bool = False,
) -> Tensor:
    """Generate a pixel-center grid covering the full sensor resolution.

    Args:
        resolution: (width, height) in pixels.
        device: Target CUDA device. Defaults to ``"cuda"``; a CPU device is
            accepted only when ``allow_device_transfer=True``.
        allow_device_transfer: If False (default), raises when ``device`` is not CUDA.

    Returns:
        image_points: ``(height, width, 2)`` float32 tensor of pixel-center
        coordinates indexed ``[y, x]`` with ``out[y, x] == (x + 0.5, y + 0.5)``.
        Reshape to ``(height * width, 2)`` to feed the backprojection functions.
    """
    target = torch.device(device)
    if target.type != "cuda":
        if not allow_device_transfer:
            raise RuntimeError(
                "CPU inputs require allow_device_transfer=True before gsplat_sensors camera ops launch."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("gsplat_sensors camera ops require CUDA")
        target = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("gsplat_sensors camera ops require CUDA")
    width, height = resolution
    return torch.ops.gsplat_sensors.generate_image_points(
        int(width), int(height), target
    )


def _quat_slerp_wxyz(q0: Tensor, q1: Tensor, alpha: Tensor) -> Tensor:
    """Differentiable quaternion SLERP in wxyz convention.

    Falls back to normalized linear interpolation (NLERP) when the two
    quaternions are nearly identical (dot > 0.9995) to avoid acos singularities.

    Args:
        q0: (*, 4) start quaternions in wxyz order.
        q1: (*, 4) end quaternions in wxyz order.
        alpha: (*,) interpolation coefficient in [0, 1].
    """
    while alpha.ndim < q0.ndim:
        alpha = alpha.unsqueeze(-1)
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    q1 = torch.where((q0 * q1).sum(dim=-1, keepdim=True) < 0, -q1, q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    close = dot > 0.9995
    # torch.where computes both branches before selecting, so torch.acos must
    # have a finite derivative on the slerp branch even when q0 == q1
    # (otherwise the unselected branch's NaN gradient pollutes q0 / q1.grad).
    # Clamping below 1 keeps acos' = -1/sqrt(1 - dot^2) bounded.
    dot_safe = dot.clamp(max=1.0 - 1e-6)
    nlerp = torch.nn.functional.normalize(q0 * (1 - alpha) + q1 * alpha, dim=-1)
    theta = torch.acos(dot_safe)
    sin_theta = torch.sin(theta)
    weight0 = torch.sin((1 - alpha) * theta) / sin_theta
    weight1 = torch.sin(alpha * theta) / sin_theta
    return torch.where(close, nlerp, weight0 * q0 + weight1 * q1)


def _unpack_static_pose(
    pose: Pose,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Extract a static Pose into (translation, rotation, times) tensors for kernel dispatch.

    Args:
        pose: Static camera pose.
        device: Target CUDA device.
        dtype: Target floating-point dtype.
        allow_device_transfer: Passed through to ``_to_dev``; see its docstring.

    Returns:
        Three-tuple of (translations (1, 3), rotations (1, 4), control_times (1,))
        each on the target device.
    """
    return (
        _to_dev(pose.translation.reshape(1, 3), device, dtype, allow_device_transfer),
        _to_dev(pose.rotation.reshape(1, 4), device, dtype, allow_device_transfer),
        torch.zeros((1,), device=device, dtype=dtype),
    )


def unpack_dynamic_pose_components(
    dynamic_pose: DynamicPose,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Extract a DynamicPose into its four component tensors for kernel dispatch.

    Args:
        dynamic_pose: Time-varying dynamic pose holding start and end Pose objects.
        device: Target CUDA device.
        dtype: Target floating-point dtype.
        allow_device_transfer: Passed through to ``_to_dev``; see its docstring.

    Returns:
        Four-tuple of (start_translation (3,), start_rotation (4,) wxyz,
        end_translation (3,), end_rotation (4,) wxyz) each on the target device.
    """
    return (
        _to_dev(
            dynamic_pose.start_pose.translation, device, dtype, allow_device_transfer
        ),
        _to_dev(dynamic_pose.start_pose.rotation, device, dtype, allow_device_transfer),
        _to_dev(
            dynamic_pose.end_pose.translation, device, dtype, allow_device_transfer
        ),
        _to_dev(dynamic_pose.end_pose.rotation, device, dtype, allow_device_transfer),
    )


def _unpack_trajectory(
    trajectory: Trajectory,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Stack a Trajectory's control poses into (translations, rotations, times) tensors.

    Args:
        trajectory: Trajectory whose control poses are stacked in order.
        device: Target CUDA device.
        dtype: Target floating-point dtype.
        allow_device_transfer: Passed through to ``_to_dev``; see its docstring.

    Returns:
        Three-tuple of (control_translations (K, 3), control_rotations (K, 4) wxyz,
        control_times (K,)) each on the target device.
    """
    translations = torch.stack(
        [
            _to_dev(p.translation, device, dtype, allow_device_transfer)
            for p in trajectory.control_poses
        ]
    )
    rotations = torch.stack(
        [
            _to_dev(p.rotation, device, dtype, allow_device_transfer)
            for p in trajectory.control_poses
        ]
    )
    return (
        translations,
        rotations,
        _to_dev(trajectory.control_times, device, dtype, allow_device_transfer),
    )


def interpolate_dynamic_pose(
    start_translation: Tensor,
    start_rotation: Tensor,
    end_translation: Tensor,
    end_rotation: Tensor,
    relative_time: Tensor,
) -> tuple[Tensor, Tensor]:
    """Linearly interpolate translation and SLERP-interpolate rotation at given relative times.

    Args:
        start_translation: (3,) translation at time 0.
        start_rotation: (4,) quaternion (wxyz) at time 0.
        end_translation: (3,) translation at time 1.
        end_rotation: (4,) quaternion (wxyz) at time 1.
        relative_time: (N,) interpolation coefficients in [0, 1].

    Returns:
        trans: (N, 3) interpolated translations.
        rot: (N, 4) interpolated quaternions in wxyz order.
    """
    trans = start_translation * (1 - relative_time).unsqueeze(
        -1
    ) + end_translation * relative_time.unsqueeze(-1)
    rot = _quat_slerp_wxyz(
        start_rotation.expand(relative_time.numel(), 4),
        end_rotation.expand(relative_time.numel(), 4),
        relative_time,
    )
    return trans, rot


def mean_pose_to_static_pose(
    dynamic_pose: DynamicPose,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Pose:
    """Return the static pose at the midpoint of ``dynamic_pose``.

    Uses the same lerp + slerp the kernel-side mean-pose ops use internally,
    so callers can hand the result to a static-pose op as a drop-in
    replacement for the dynamic-pose path.
    """
    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose, device, dtype
    )
    rel = torch.full((1,), 0.5, device=device, dtype=dtype)
    pose_t, pose_r = interpolate_dynamic_pose(start_t, start_r, end_t, end_r, rel)
    return Pose(translation=pose_t[0], rotation=pose_r[0])


def relative_frame_times(
    image_points: Tensor, resolution: tuple[int, int], shutter_type: ShutterType
) -> Tensor:
    """Compute per-pixel relative exposure times in [0, 1] for a given shutter model.

    For global shutter all values are 0. For rolling shutter the value encodes how
    far along the scan the pixel's row (or column) falls.

    Args:
        image_points: (N, 2) pixel coordinates.
        resolution: (width, height) in pixels.
        shutter_type: Rolling or global shutter behavior (ShutterType enum).

    Returns:
        relative_times: (N,) float tensor in [0, 1].
    """
    if image_points.ndim != 2 or image_points.shape[-1] != 2:
        raise ValueError(
            f"image_points must have shape (N, 2), got {tuple(image_points.shape)}"
        )
    width, height = resolution
    safe_w = max(width, 2) - 1
    safe_h = max(height, 2) - 1
    if shutter_type == ShutterType.ROLLING_TOP_TO_BOTTOM:
        return torch.floor(image_points[:, 1]).clamp(0, safe_h) / safe_h
    if shutter_type == ShutterType.ROLLING_BOTTOM_TO_TOP:
        return (height - torch.ceil(image_points[:, 1])).clamp(0, safe_h) / safe_h
    if shutter_type == ShutterType.ROLLING_LEFT_TO_RIGHT:
        return torch.floor(image_points[:, 0]).clamp(0, safe_w) / safe_w
    if shutter_type == ShutterType.ROLLING_RIGHT_TO_LEFT:
        return (width - torch.ceil(image_points[:, 0])).clamp(0, safe_w) / safe_w
    if shutter_type == ShutterType.GLOBAL:
        return torch.zeros(
            (image_points.shape[0],),
            device=image_points.device,
            dtype=image_points.dtype,
        )
    raise ValueError(f"Unsupported ShutterType: {shutter_type!r} (allowed: 1..5)")


def project_world_points_mean_pose(
    world_points: Tensor,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    dynamic_pose: DynamicPose,
    resolution: tuple[int, int],
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_valid_flags: bool = False,
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None,]:
    """Project world points to image coordinates using the mean of the start/end pose.

    Supports both NoExternalDistortion and BivariateWindshieldDistortion variants.
    All outputs are fully differentiable except ``valid_flags`` and ``timestamps_us``.

    Args:
        world_points: (N, 3) world coordinates.
        projection: Registered camera projection parameters.
        external_distortion: External distortion parameters (NoExternalDistortion or
            BivariateWindshieldDistortion).
        dynamic_pose: Time-varying dynamic pose whose midpoint is used for projection.
        resolution: (width, height) in pixels.
        start_timestamp_us: Start timestamp in microseconds (int64). Must be
            supplied together with ``end_timestamp_us`` or both omitted;
            ``_timestamp_bounds`` raises ``ValueError`` if exactly one is
            provided, regardless of ``return_timestamps``.
        end_timestamp_us: End timestamp in microseconds (int64). Same
            both-or-neither contract as ``start_timestamp_us``.
        return_valid_flags: If True, return validity mask.
        return_timestamps: If True, return per-point timestamps.
        return_poses: If True, return per-point poses (translation + rotation).
        allow_device_transfer: If False (default), raises if any tensor needs a
            device or dtype transfer before the kernel launches.

    Returns:
        image_points: (N, 2) projected pixel coordinates.
        valid_flags: (N,) bool mask, or None if ``return_valid_flags=False``.
        timestamps_us: (N,) int64 microsecond timestamps (midpoint), or None if
            ``return_timestamps=False``.
        pose_t: (N, 3) per-point pose translations, or None if ``return_poses=False``.
        pose_r: (N, 4) per-point pose rotations (wxyz), or None if ``return_poses=False``.
    """
    _check_pair(projection, external_distortion)
    device = _raise_or_target_device(world_points, allow_device_transfer)
    pts = _to_dev(world_points, device, torch.float32, allow_device_transfer)
    projection_cuda, tensors = _projection_on_device_any(
        projection, device, torch.float32, allow_device_transfer, resolution=resolution
    )
    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose, device, torch.float32, allow_device_transfer
    )
    start, end = _timestamp_bounds(start_timestamp_us, end_timestamp_us)
    apply_fn, distortion_arg, extra_distortion_args = _select_op(
        projection,
        "project_world_points_mean_pose",
        _ProjectWorldPointsMeanPose,
        _ProjectWorldPointsMeanPoseBivariateWindshield,
        external_distortion,
        device,
        torch.float32,
        allow_device_transfer,
    )
    image_points, valid_flags, timestamps_us, pose_t, pose_r = apply_fn(
        pts,
        start_t,
        start_r,
        end_t,
        end_r,
        projection_cuda,
        distortion_arg,
        *extra_distortion_args,
        *tensors,
        start,
        end,
    )
    return (
        image_points,
        valid_flags if return_valid_flags else None,
        timestamps_us if return_timestamps else None,
        pose_t if return_poses else None,
        pose_r if return_poses else None,
    )


def project_world_points_shutter_pose(
    world_points: Tensor,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    max_iterations: int = 10,
    stop_mean_error_px: float = 0.001,
    stop_delta_mean_error_px: float = 0.00001,
    initial_relative_time: float = 0.5,
    return_valid_flags: bool = False,
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    """Project world points to image coordinates accounting for rolling shutter.

    Uses an iterative solver to find the per-point shutter time consistent with the
    projected pixel row (or column). Supports both NoExternalDistortion and
    BivariateWindshieldDistortion variants. All outputs are fully differentiable
    except ``valid_flags`` and ``timestamps_us``.

    Args:
        world_points: (N, 3) world coordinates.
        projection: Registered camera projection parameters.
        external_distortion: External distortion parameters (NoExternalDistortion or
            BivariateWindshieldDistortion).
        resolution: (width, height) in pixels.
        shutter_type: Rolling or global shutter behavior.
        dynamic_pose: Time-varying dynamic pose interpolated at each point's shutter time.
        start_timestamp_us: Start timestamp in microseconds (int64). Must be
            supplied together with ``end_timestamp_us`` or both omitted;
            ``_timestamp_bounds`` raises ``ValueError`` if exactly one is
            provided, regardless of ``return_timestamps``.
        end_timestamp_us: End timestamp in microseconds (int64). Same
            both-or-neither contract as ``start_timestamp_us``.
        max_iterations: Maximum number of solver iterations. Defaults to 10.
        stop_mean_error_px: Convergence threshold on mean projection error in pixels.
        stop_delta_mean_error_px: Convergence threshold on change in mean error.
        initial_relative_time: Initial relative-time guess for the solver in [0, 1].
            Defaults to 0.5 (frame midpoint).
        return_valid_flags: If True, return validity mask.
        return_timestamps: If True, return per-point timestamps.
        return_poses: If True, return per-point poses (translation + rotation).
        allow_device_transfer: If False (default), raises if any tensor needs a
            device or dtype transfer before the kernel launches.

    Returns:
        image_points: (N, 2) projected pixel coordinates.
        valid_flags: (N,) bool mask, or None if ``return_valid_flags=False``.
        timestamps_us: (N,) int64 microsecond timestamps, or None if
            ``return_timestamps=False``.
        pose_t: (N, 3) per-point pose translations, or None if ``return_poses=False``.
        pose_r: (N, 4) per-point pose rotations (wxyz), or None if ``return_poses=False``.
    """
    _check_pair(projection, external_distortion)
    device = _raise_or_target_device(world_points, allow_device_transfer)
    pts = _to_dev(world_points, device, torch.float32, allow_device_transfer)
    projection_cuda, tensors = _projection_on_device_any(
        projection, device, torch.float32, allow_device_transfer, resolution=resolution
    )
    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose, device, torch.float32, allow_device_transfer
    )
    start, end = _timestamp_bounds(start_timestamp_us, end_timestamp_us)
    width, height = resolution
    apply_fn, distortion_arg, extra_distortion_args = _select_op(
        projection,
        "project_world_points_shutter_pose",
        _ProjectWorldPointsShutterPose,
        _ProjectWorldPointsShutterPoseBivariateWindshield,
        external_distortion,
        device,
        torch.float32,
        allow_device_transfer,
    )
    apply_args = (
        pts,
        start_t,
        start_r,
        end_t,
        end_r,
        projection_cuda,
        distortion_arg,
        *extra_distortion_args,
        *tensors,
        int(width),
        int(height),
        int(ShutterType(shutter_type)),
        start,
        end,
        int(max_iterations),
        float(stop_mean_error_px),
        float(stop_delta_mean_error_px),
        float(initial_relative_time),
    )
    (
        image_points,
        valid_flags,
        timestamps_us,
        pose_t,
        pose_r,
    ) = apply_fn(*apply_args)
    return (
        image_points,
        valid_flags if return_valid_flags else None,
        timestamps_us if return_timestamps else None,
        pose_t if return_poses else None,
        pose_r if return_poses else None,
    )


def image_points_to_world_rays_static_pose(
    image_points: Tensor,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    pose: Pose,
    *,
    timestamp_us: int | None = None,
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Back-project image pixels to world-frame rays using a single static camera pose.

    Supports both NoExternalDistortion and BivariateWindshieldDistortion variants.
    All outputs are fully differentiable except ``timestamps_us``.

    Args:
        image_points: (N, 2) pixel coordinates.
        projection: Registered camera projection parameters.
        external_distortion: External distortion parameters (NoExternalDistortion or
            BivariateWindshieldDistortion).
        pose: Static camera pose (translation + rotation).
        timestamp_us: Timestamp assigned to every returned ray (microseconds, int64).
            Defaults to 0 when None.
        return_timestamps: If True, return per-point timestamps.
        return_poses: If True, return per-point poses (translation + rotation).
        allow_device_transfer: If False (default), raises if any tensor needs a
            device or dtype transfer before the kernel launches.

    Returns:
        world_rays: (N, 6) packed [origin (3), direction (3)] in world frame.
        timestamps_us: (N,) int64 microsecond timestamps (constant), or None if
            ``return_timestamps=False``.
        pose_t: (N, 3) per-point pose translations (same for all points), or None if
            ``return_poses=False``.
        pose_r: (N, 4) per-point pose rotations (wxyz, same for all points), or None if
            ``return_poses=False``.
    """
    _check_pair(projection, external_distortion)
    device = _raise_or_target_device(image_points, allow_device_transfer)
    pts = _to_dev(image_points, device, torch.float32, allow_device_transfer)
    projection_cuda, tensors = _projection_on_device_any(
        projection, device, torch.float32, allow_device_transfer
    )
    translations, rotations, _ = _unpack_static_pose(
        pose, device, torch.float32, allow_device_transfer
    )
    timestamp = 0 if timestamp_us is None else int(timestamp_us)
    apply_fn, distortion_arg, extra_distortion_args = _select_op(
        projection,
        "image_points_to_world_rays_static_pose",
        _ImagePointsToWorldRaysStaticPose,
        _ImagePointsToWorldRaysStaticPoseBivariateWindshield,
        external_distortion,
        device,
        torch.float32,
        allow_device_transfer,
    )
    (world_rays, timestamps_us, pose_t, pose_r,) = apply_fn(
        pts,
        translations,
        rotations,
        projection_cuda,
        distortion_arg,
        *extra_distortion_args,
        *tensors,
        timestamp,
    )
    return (
        world_rays,
        timestamps_us if return_timestamps else None,
        pose_t if return_poses else None,
        pose_r if return_poses else None,
    )


def _image_points_to_world_rays_shutter_pose_impl(
    pts: Tensor,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Shared implementation for shutter-pose ray generation (explicit points or pixel grid).

    Called by both ``image_points_to_world_rays_shutter_pose`` and
    ``pixel_grid_to_world_rays_shutter_pose``; assumes ``pts`` is already on the
    correct device and dtype.

    Args:
        pts: (N, 2) pixel coordinates already on the target device.
        projection: Registered camera projection parameters.
        external_distortion: External distortion parameters (NoExternalDistortion or
            BivariateWindshieldDistortion).
        resolution: (width, height) in pixels.
        shutter_type: Rolling or global shutter behavior.
        dynamic_pose: Time-varying dynamic pose interpolated at each point's shutter time.
        start_timestamp_us: Start timestamp for timestamp computation (microseconds, int64).
        end_timestamp_us: End timestamp for timestamp computation (microseconds, int64).
        return_timestamps: If True, return per-point timestamps.
        return_poses: If True, return per-point poses (translation + rotation).
        allow_device_transfer: Passed through to projection/distortion helpers.

    Returns:
        world_rays: (N, 6) packed [origin (3), direction (3)] in world frame.
        timestamps_us: (N,) int64, or None if ``return_timestamps=False``.
        pose_t: (N, 3), or None if ``return_poses=False``.
        pose_r: (N, 4) wxyz, or None if ``return_poses=False``.
    """
    projection_cuda, tensors = _projection_on_device_any(
        projection,
        pts.device,
        torch.float32,
        allow_device_transfer,
        resolution=resolution,
    )
    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose, pts.device, torch.float32, allow_device_transfer
    )
    start, end = _timestamp_bounds(start_timestamp_us, end_timestamp_us)
    width, height = resolution
    apply_fn, distortion_arg, extra_distortion_args = _select_op(
        projection,
        "image_points_to_world_rays_shutter_pose",
        _ImagePointsToWorldRaysShutterPose,
        _ImagePointsToWorldRaysShutterPoseBivariateWindshield,
        external_distortion,
        pts.device,
        torch.float32,
        allow_device_transfer,
    )
    apply_args = (
        pts,
        start_t,
        start_r,
        end_t,
        end_r,
        projection_cuda,
        distortion_arg,
        *extra_distortion_args,
        *tensors,
        int(width),
        int(height),
        int(ShutterType(shutter_type)),
        start,
        end,
    )
    (
        world_rays,
        timestamps_us,
        pose_t,
        pose_r,
    ) = apply_fn(*apply_args)
    return (
        world_rays,
        timestamps_us if return_timestamps else None,
        pose_t if return_poses else None,
        pose_r if return_poses else None,
    )


def image_points_to_world_rays_shutter_pose(
    image_points: Tensor,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Back-project image pixels to world-frame rays accounting for rolling shutter.

    Supports both NoExternalDistortion and BivariateWindshieldDistortion variants.
    All outputs are fully differentiable except ``timestamps_us``.

    Args:
        image_points: (N, 2) pixel coordinates.
        projection: Registered camera projection parameters.
        external_distortion: External distortion parameters (NoExternalDistortion or
            BivariateWindshieldDistortion).
        resolution: (width, height) in pixels.
        shutter_type: Rolling or global shutter behavior.
        dynamic_pose: Time-varying dynamic pose interpolated at each point's shutter time.
        start_timestamp_us: Start timestamp for timestamp computation (microseconds, int64).
        end_timestamp_us: End timestamp for timestamp computation (microseconds, int64).
        return_timestamps: If True, return per-point timestamps.
        return_poses: If True, return per-point poses (translation + rotation).
        allow_device_transfer: If False (default), raises if any tensor needs a
            device or dtype transfer before the kernel launches.

    Returns:
        world_rays: (N, 6) packed [origin (3), direction (3)] in world frame.
        timestamps_us: (N,) int64 microsecond timestamps, or None if
            ``return_timestamps=False``.
        pose_t: (N, 3) per-point pose translations, or None if ``return_poses=False``.
        pose_r: (N, 4) per-point pose rotations (wxyz), or None if ``return_poses=False``.

    Note:
        Rolling-shutter relative time is treated as non-differentiable, matching
        the Slang reference implementation's ``no_diff`` time computation. The
        direct image-point gradient through projection and external distortion
        is still preserved.
    """
    _check_pair(projection, external_distortion)
    device = _raise_or_target_device(image_points, allow_device_transfer)
    pts = _to_dev(image_points, device, torch.float32, allow_device_transfer)
    return _image_points_to_world_rays_shutter_pose_impl(
        pts,
        projection,
        external_distortion,
        resolution,
        shutter_type,
        dynamic_pose,
        start_timestamp_us=start_timestamp_us,
        end_timestamp_us=end_timestamp_us,
        return_timestamps=return_timestamps,
        return_poses=return_poses,
        allow_device_transfer=allow_device_transfer,
    )


def pixel_grid_to_world_rays_shutter_pose(
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Generate world-frame rays for every pixel in the full sensor grid with rolling shutter.

    Equivalent to calling ``generate_image_points`` followed by
    ``image_points_to_world_rays_shutter_pose``. This spares the caller from
    creating the grid manually, but still allocates the internal pixel-coordinate
    tensor before dispatching to the ray kernel. Supports both NoExternalDistortion
    and BivariateWindshieldDistortion variants.

    Args:
        projection: Registered camera projection parameters.
        external_distortion: External distortion parameters (NoExternalDistortion or
            BivariateWindshieldDistortion).
        resolution: (width, height) in pixels.
        shutter_type: Rolling or global shutter behavior.
        dynamic_pose: Time-varying dynamic pose interpolated at each pixel's shutter time.
        start_timestamp_us: Start timestamp for timestamp computation (microseconds, int64).
        end_timestamp_us: End timestamp for timestamp computation (microseconds, int64).
        return_timestamps: If True, return per-pixel timestamps.
        return_poses: If True, return per-pixel poses (translation + rotation).
        allow_device_transfer: If False (default), raises if any tensor needs a
            device or dtype transfer before the kernel launches.

    Returns:
        world_rays: (height * width, 6) packed [origin (3), direction (3)] in
            world frame, in row-major pixel order.
        timestamps_us: (height * width,) int64 microsecond timestamps, or None if
            ``return_timestamps=False``.
        pose_t: (height * width, 3) per-pixel pose translations, or None if
            ``return_poses=False``.
        pose_r: (height * width, 4) per-pixel pose rotations (wxyz), or None if
            ``return_poses=False``.
    """
    _check_pair(projection, external_distortion)
    device = _raise_or_target_device(
        dynamic_pose.start_pose.translation, allow_device_transfer
    )
    pts = generate_image_points(
        resolution, device=device, allow_device_transfer=True
    ).reshape(-1, 2)
    return _image_points_to_world_rays_shutter_pose_impl(
        pts,
        projection,
        external_distortion,
        resolution,
        shutter_type,
        dynamic_pose,
        start_timestamp_us=start_timestamp_us,
        end_timestamp_us=end_timestamp_us,
        return_timestamps=return_timestamps,
        return_poses=return_poses,
        allow_device_transfer=allow_device_transfer,
    )


__all__ = [
    "camera_rays_to_image_points",
    "generate_image_points",
    "image_points_to_camera_rays",
    "image_points_to_world_rays_static_pose",
    "image_points_to_world_rays_shutter_pose",
    "interpolate_dynamic_pose",
    "mean_pose_to_static_pose",
    "pixel_grid_to_world_rays_shutter_pose",
    "project_world_points_mean_pose",
    "project_world_points_shutter_pose",
    "relative_frame_times",
    "unpack_dynamic_pose_components",
]
