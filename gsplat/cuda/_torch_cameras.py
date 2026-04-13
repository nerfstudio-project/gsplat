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

"""Camera models and rolling shutter utilities for GSplat.

This module contains camera model implementations including perfect pinhole, OpenCV pinhole,
OpenCV fisheye, and FTheta models, along with rolling shutter support.
"""

import math
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from torch import Tensor
from gsplat._helper import assert_shape

import numpy as np

from ._math import (
    _numerically_stable_norm2,
    PolynomialProxy,
    FullPolynomialProxy,
    OddPolynomialProxy,
    EvenPolynomialProxy,
    _eval_poly_inverse_horner_newton,
    _rotmat_to_quat,
    _quat_inverse,
    _quat_to_rotmat,
    _quat_rotate,
    _quat_slerp,
    _safe_normalize,
)
from ._wrapper import (
    RollingShutterType,
    FThetaPolynomialType,
    FThetaCameraDistortionParameters,
    SpinningDirection,
)


def _project_to_image(
    cam_ray: Tensor,  # [B, 2] normalized camera coordinates
    focal_lengths: Tensor,  # [B, 2] focal lengths (fx, fy)
    principal_points: Tensor,  # [B, 2] principal points (cx, cy)
) -> Tensor:
    """Project normalized camera coordinates to image pixel coordinates.

    Args:
        cam_ray: [B, 2] normalized camera coordinates (x/z, y/z)
        focal_lengths: [B, 2] focal lengths (fx, fy)
        principal_points: [B, 2] principal points (cx, cy)

    Returns:
        image_point: [B, 2] pixel coordinates
    """

    # Preconditions
    B = cam_ray.shape[:-1]
    assert_shape("cam_ray", cam_ray, B + (2,))
    assert_shape("focal_lengths", focal_lengths, B + (2,))
    assert_shape("principal_points", principal_points, B + (2,))

    result = cam_ray * focal_lengths + principal_points

    # Postconditions
    assert_shape("result", result, B + (2,))

    return result


def _unproject_from_image(
    image_point: Tensor,  # [B, 2]
    focal_lengths: Tensor,  # [B, 2]
    principal_points: Tensor,  # [B, 2]
) -> Tensor:
    """Unproject image pixel coordinates to normalized camera coordinates.

    Args:
        image_point: [B, 2] pixel coordinates
        focal_lengths: [B, 2] focal lengths (fx, fy)
        principal_points: [B, 2] principal points (cx, cy)

    Returns:
        uv: [B, M, 2] normalized camera coordinates (x/z, y/z)
    """
    # Preconditions
    B = image_point.shape[:-1]
    assert_shape("image_point", image_point, B + (2,))
    assert_shape("focal_lengths", focal_lengths, B + (2,))
    assert_shape("principal_points", principal_points, B + (2,))

    result = (image_point - principal_points) / focal_lengths

    # Postconditions
    assert_shape("result", result, B + (2,))

    return result


def _viewmat_to_pose(viewmat: Tensor) -> Tensor:
    """
    Convert 4x4 view matrix to 7D pose tensor [t_x, t_y, t_z, q_w, q_x, q_y, q_z].

    Args:
        viewmat: View matrix [B, 4, 4] (world-to-camera transform)

    Returns:
        Pose tensor [B, 7] with translation [0:3] and quaternion [3:7]
    """
    # Preconditions
    B = viewmat.shape[:-2]
    assert_shape("viewmat", viewmat, B + (4, 4))

    R = viewmat[..., :3, :3]  # [B, 3, 3]
    t = viewmat[..., :3, 3]  # [B, 3]
    q = _rotmat_to_quat(R)  # [B, 4] in format (w, x, y, z)
    # Concatenate: [t_x, t_y, t_z, q_w, q_x, q_y, q_z]
    result = torch.cat([t, q], dim=-1)  # [B, 7]

    # Postconditions
    assert_shape("result", result, B + (7,))

    return result


def _pose_camera_world_position(pose: Tensor) -> Tensor:
    """
    Get camera world position from pose tensor.

    Computes: c = R^{-1} * (-t) where R is rotation from q, t is translation

    Args:
        pose: Pose tensor [B, 7] with format [t_x, t_y, t_z, q_w, q_x, q_y, q_z]

    Returns:
        Camera position in world coordinates [B, 3]
    """
    # Preconditions
    B = pose.shape[:-1]
    assert_shape("pose", pose, B + (7,))

    t = pose[..., :3]  # [B, 3] translation
    q = pose[..., 3:]  # [B, 4] quaternion
    result = torch.matmul(_quat_to_rotmat(_quat_inverse(q)), -t[..., None]).squeeze(
        -1
    )  # [B, 3]

    # Postconditions
    assert_shape("result", result, B + (3,))

    return result


def _pose_camera_ray_to_world_ray(
    pose: Tensor, camera_ray: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Transform camera ray to world ray using pose.

    Computes:
        - origin: o_world = -R^{-1} * t (camera position in world coordinates)
        - direction: d_world = R^{-1} * d_camera
    where R is rotation matrix from quaternion q, t is translation

    Args:
        pose: Pose tensor [B, 7] with format [t_x, t_y, t_z, q_w, q_x, q_y, q_z]
        camera_ray: Camera ray direction d_camera [B, 3]

    Returns:
        Tuple of (origin, direction, valid) tensors
    """
    # Preconditions
    B = pose.shape[:-1]
    assert_shape("pose", pose, B + (7,))
    assert_shape("camera_ray", camera_ray, B + (3,))

    t = pose[..., :3]  # [B, 3] translation
    q = pose[..., 3:]  # [B, 4] quaternion
    q_inv = _quat_inverse(q)

    # Avoid extra fp32 drift from rebuilding the inverse rotation matrix and
    # applying a separate matmul for the translated camera origin.
    origin = _quat_rotate(q_inv, -t)  # [B, 3]

    # Use quaternion rotation for the direction as well instead of rebuilding
    # R_inv and applying a separate matmul in fp32.
    direction = _quat_rotate(q_inv, camera_ray)  # [B, 3]

    # Postconditions
    assert_shape("origin", origin, B + (3,))
    assert_shape("direction", direction, B + (3,))

    return origin, direction  # [B, 3]


def _pose_world_points_to_camera_ray(pose: Tensor, world_points: Tensor) -> Tensor:
    """
    Transform world points to camera space using pose.

    Applies the world-to-camera transformation:
        p_camera = R * p_world + t
    where R is rotation from quaternion q, t is translation

    Args:
        pose: Pose tensor [B, 7]  with format [t_x, t_y, t_z, q_w, q_x, q_y, q_z]
        world_points: Points in world space p_world [B, 3]

    Returns:
        Points in camera space p_camera
    """
    # Preconditions
    B = pose.shape[:-1]
    assert_shape("pose", pose, B + (7,))
    assert_shape("world_points", world_points, B + (3,))

    t = pose[..., :3]  # [B, 3] translation
    q = pose[..., 3:]  # [B, 4] quaternion

    # Apply world-to-camera transformation: R * p_world + t
    camera_points = _quat_rotate(q, world_points) + t

    # Postconditions
    assert_shape("camera_points", camera_points, B + (3,))

    return camera_points  # [B, 3]


class _BaseCameraModel(ABC):
    def __init__(
        self,
        width: int,
        height: int,
        shutter_type: RollingShutterType = RollingShutterType.GLOBAL,
    ):
        self.width = width
        self.height = height
        self.shutter_type = shutter_type

    @staticmethod
    def create(
        width: Optional[int] = None,
        height: Optional[int] = None,
        camera_model: str = "pinhole",
        principal_points: Optional[Tensor] = None,  # [B, 2]
        focal_lengths: Optional[Tensor] = None,  # [B, 2]
        radial_coeffs: Optional[Tensor] = None,
        tangential_coeffs: Optional[Tensor] = None,
        thin_prism_coeffs: Optional[Tensor] = None,
        ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
        rs_type: RollingShutterType = RollingShutterType.GLOBAL,
        # Optional[RowOffsetStructuredSpinningLidarModelParameters]
        # Can't type it here to avoid circular import with _torch_lidars
        lidar_coeffs=None,
    ) -> "_BaseCameraModel":
        """
        Factory method to create appropriate camera model.
            Args:
            width: Image width (required for non-lidar models)
            height: Image height (required for non-lidar models)
            camera_model: "pinhole", "fisheye", "ftheta", or "lidar"
            principal_points: Principal points [B, 2] (cx, cy) - required for non-lidar models
            focal_lengths: Focal lengths [B, 2] (fx, fy) - required for pinhole and fisheye
            radial_coeffs: [B, 6] or [B, 4] radial distortion coefficients (pinhole/fisheye)
            tangential_coeffs: [B, 2] tangential distortion coefficients (pinhole only)
            thin_prism_coeffs: [B, 4] thin prism distortion coefficients (pinhole only)
            ftheta_coeffs: F-theta parameters (ftheta only)
            rs_type: Rolling shutter type (default: GLOBAL)
            lidar_coeffs: Lidar camera parameters (lidar only)

        Returns:
            Camera model instance

        Note:
            For ftheta model, focal_lengths parameter is not used as focal length
            is embedded in the polynomial distortion model.
        """
        if camera_model == "lidar":
            assert (
                lidar_coeffs is not None
            ), "lidar_coeffs is required for lidar camera model"
            from ._torch_lidars import _RowOffsetStructuredSpinningLidarModel

            return _RowOffsetStructuredSpinningLidarModel(lidar_coeffs)

        # Preconditions for non-lidar models
        assert width is not None, "width is required for non-lidar camera models"
        assert height is not None, "height is required for non-lidar camera models"
        assert (
            principal_points is not None
        ), "principal_points is required for non-lidar camera models"
        B = principal_points.shape[:-1]
        assert_shape("principal_points", principal_points, B + (2,))
        focal_lengths is None or assert_shape("focal_lengths", focal_lengths, B + (2,))
        radial_coeffs is None or assert_shape("radial_coeffs", radial_coeffs, B + (1,))
        tangential_coeffs is None or assert_shape(
            "tangential_coeffs", tangential_coeffs, B + (2,)
        )
        thin_prism_coeffs is None or assert_shape(
            "thin_prism_coeffs", thin_prism_coeffs, B + (4,)
        )

        if camera_model == "pinhole":
            if ftheta_coeffs is not None:
                raise ValueError(
                    "pinhole camera model does not support ftheta_coeffs parameter"
                )
            if focal_lengths is None:
                raise ValueError("focal_lengths is required for pinhole camera model")

            if (
                radial_coeffs is not None
                or tangential_coeffs is not None
                or thin_prism_coeffs is not None
            ):
                return _OpenCVPinholeCameraModel(
                    focal_lengths=focal_lengths,
                    principal_points=principal_points,
                    width=width,
                    height=height,
                    rs_type=rs_type,
                    radial_coeffs=radial_coeffs,
                    tangential_coeffs=tangential_coeffs,
                    thin_prism_coeffs=thin_prism_coeffs,
                )
            else:
                return _PerfectPinholeCameraModel(
                    focal_lengths=focal_lengths,
                    principal_points=principal_points,
                    width=width,
                    height=height,
                    rs_type=rs_type,
                )

        elif camera_model == "fisheye":
            if ftheta_coeffs is not None:
                raise ValueError(
                    "fisheye camera model does not support ftheta_coeffs parameter"
                )
            if tangential_coeffs is not None or thin_prism_coeffs is not None:
                raise ValueError(
                    "fisheye camera model does not support tangential_coeffs or thin_prism_coeffs parameters"
                )
            if focal_lengths is None:
                raise ValueError("focal_lengths is required for fisheye camera model")

            return _OpenCVFisheyeCameraModel(
                focal_lengths=focal_lengths,
                principal_points=principal_points,
                width=width,
                height=height,
                rs_type=rs_type,
                radial_coeffs=radial_coeffs,
            )

        elif camera_model == "ftheta":
            if ftheta_coeffs is None:
                raise ValueError("ftheta requires ftheta_coeffs parameter")
            if (
                radial_coeffs is not None
                or tangential_coeffs is not None
                or thin_prism_coeffs is not None
            ):
                raise ValueError(
                    "ftheta camera model does not support radial_coeffs, tangential_coeffs, or thin_prism_coeffs parameters"
                )
            if focal_lengths is not None:
                raise ValueError(
                    "ftheta camera model does not support focal_lengths parameter"
                )

            # Use ftheta_coeffs directly (no conversion needed)
            return _FThetaCameraModel(
                principal_points=principal_points,
                width=width,
                height=height,
                rs_type=rs_type,
                dist_params=ftheta_coeffs,
            )

        else:
            raise ValueError(
                f"Unsupported camera model: {camera_model}. "
                f"Supported: pinhole, fisheye, ftheta, lidar"
            )

    def shutter_relative_frame_time(
        self,
        pixel_coords: Tensor,  # [M, 2] (x, y) coordinates
    ) -> Tensor:
        """Compute relative frame time for rolling shutter based on pixel position.

        Matches the shutter_relative_frame_time function in CUDA's BaseCameraModel.

        Args:
            pixel_coords: Pixel coordinates [M, 2] as (x, y)

        Returns:
            Relative frame time [M] in range [0, 1]
        """
        # Preconditions
        M = pixel_coords.shape[:-1]
        assert_shape("pixel_coords", pixel_coords, M + (2,))

        device = pixel_coords.device

        if self.shutter_type == RollingShutterType.GLOBAL:
            # Global shutter - all pixels at time 0.0 (matches CUDA default)
            # Note: This value is typically unused since viewmats_rs is None for GLOBAL
            t = torch.full(M, 0.0, device=device, dtype=pixel_coords.dtype)
        else:
            px = pixel_coords[..., 0]  # [M]
            py = pixel_coords[..., 1]  # [M]

            if self.shutter_type == RollingShutterType.ROLLING_TOP_TO_BOTTOM:
                t = (
                    (torch.floor(py) / float(self.height - 1))
                    if self.height > 1
                    else 0.5
                )
            elif self.shutter_type == RollingShutterType.ROLLING_LEFT_TO_RIGHT:
                t = (torch.floor(px) / float(self.width - 1)) if self.width > 1 else 0.5
            elif self.shutter_type == RollingShutterType.ROLLING_BOTTOM_TO_TOP:
                # TODO: this returns > 1 for the topmost row, but should be 1
                t = (
                    ((self.height - torch.ceil(py)) / float(self.height - 1))
                    if self.height > 1
                    else 0.5
                )
            else:
                # TODO: this returns > 1 for the rightmost column, but should be 1
                assert self.shutter_type == RollingShutterType.ROLLING_RIGHT_TO_LEFT
                t = (
                    ((self.width - torch.ceil(px)) / float(self.width - 1))
                    if self.width > 1
                    else 0.5
                )

        # Postconditions
        assert_shape("t", t, M)
        return t

    @abstractmethod
    def camera_ray_to_image_point(
        self,
        camera_ray: Tensor,
        margin_factor: float,
    ) -> Tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def image_point_to_camera_ray(
        self,
        image_point: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        ...

    @property
    @abstractmethod
    def focal_lengths(self) -> Tensor:
        ...

    @property
    @abstractmethod
    def principal_points(self) -> Tensor:
        ...

    def image_point_to_world_ray_shutter_pose(
        self,
        image_point: Tensor,  # [B, M, 2]
        shutter_pose_start: Tensor,  # [B, 7]
        shutter_pose_end: Tensor,  # [B, 7]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Convert an image point to a world ray using a shutter pose.

        Args:
            image_point: Image point [B, M, 2]
            shutter_pose_start: Start shutter pose [B, 7] in format [tx, ty, tz, qw, qx, qy, qz]
            shutter_pose_end: End shutter pose [B, 7] in format [tx, ty, tz, qw, qx, qy, qz]
        Returns:
            Tuple of (origin, direction, valid) tensors
        """
        # Preconditions
        B = shutter_pose_start.shape[:-1]
        M = (
            (image_point.shape[-2],)
            if image_point.ndim == shutter_pose_start.ndim + 1
            else ()
        )
        assert_shape("image_point", image_point, B + M + (2,))
        assert_shape("shutter_pose_start", shutter_pose_start, B + (7,))
        assert_shape("shutter_pose_end", shutter_pose_end, B + (7,))

        camera_ray, valid = self.image_point_to_camera_ray(image_point)

        relative_time = self.shutter_relative_frame_time(image_point)
        interpolated_pose = _interpolate_shutter_pose(
            shutter_pose_start[..., None, :],
            shutter_pose_end[..., None, :],
            relative_time,
        )
        world_ray_org, world_ray_dir = _pose_camera_ray_to_world_ray(
            interpolated_pose, camera_ray
        )

        # Invalid world rays are set to 0
        world_ray_org = world_ray_org * valid[..., None]
        world_ray_dir = world_ray_dir * valid[..., None]

        # Postconditions
        assert_shape("world_ray_org", world_ray_org, B + M + (3,))
        assert_shape("world_ray_dir", world_ray_dir, B + M + (3,))
        assert_shape("valid", valid, B + M)
        return world_ray_org, world_ray_dir, valid

    def world_point_to_image_point_shutter_pose(
        self,
        world_points: Tensor,  # [B, M, 3] - points in world space
        shutter_pose_start: Tensor,  # [B, 7]
        shutter_pose_end: Tensor,  # [B, 7]
        margin_factor: float,
        rolling_shutter_iterations=10,
    ) -> Tuple[Tensor, Tensor]:
        """Project world points to image coordinates with rolling shutter correction.

        1. Project with start pose to get initial image point
        2. For rolling shutter, iteratively refine:
        - Compute relative frame time based on image point position
        - Interpolate camera pose at that time
        - Transform world point with interpolated pose
        - Project to get new image point
        3. Return converged image point after rolling_shutter_iterations

        Args:
            world_points: Points in world space [B, M, 3]
            shutter_pose_start: Start shutter pose [B, 7]
            shutter_pose_end: End shutter pose [B, 7]
            margin_factor: Tolerance for out-of-image-bounds

        Returns:
            points_2d: [B, M, 2] - projected 2D points
            valid: [B, M] - validity mask
        """
        # Preconditions
        B = shutter_pose_start.shape[:-1]
        M = (
            (world_points.shape[-2],)
            if world_points.ndim == shutter_pose_start.ndim + 1
            else ()
        )
        assert_shape("world_points", world_points, B + M + (3,))
        assert_shape("shutter_pose_start", shutter_pose_start, B + (7,))
        assert_shape("shutter_pose_end", shutter_pose_end, B + (7,))

        image_points_start, valid_start = self.camera_ray_to_image_point(
            _pose_world_points_to_camera_ray(
                shutter_pose_start[..., None, :], world_points
            ),
            margin_factor,
        )

        if self.shutter_type == RollingShutterType.GLOBAL:
            assert_shape("image_points_start", image_points_start, B + M + (2,))
            assert_shape("valid_start", valid_start, B + M)
            return image_points_start, valid_start

        image_points_end, valid_end = self.camera_ray_to_image_point(
            _pose_world_points_to_camera_ray(
                shutter_pose_end[..., None, :], world_points
            ),
            margin_factor,
        )

        # Select initial image point: prefer start if valid, otherwise use end
        init_image_points = torch.where(
            valid_start[..., None], image_points_start, image_points_end
        )  # [B, M, 2]

        # Iterative refinement for rolling shutter
        # Only iterate if at least one pose had valid projection (valid_start OR valid_end)
        valid = valid_start | valid_end

        image_points_rs_prev = init_image_points

        for _ in range(rolling_shutter_iterations):
            # Only iterate on points that are still valid
            if not valid.any():
                break

            # Compute relative frame time based on current image point position
            relative_time = self.shutter_relative_frame_time(
                image_points_rs_prev
            )  # [B, M]

            pose_rs = _interpolate_shutter_pose(
                shutter_pose_start[..., None, :],
                shutter_pose_end[..., None, :],
                relative_time,
            )

            image_points_rs, valid_rs = self.camera_ray_to_image_point(
                _pose_world_points_to_camera_ray(pose_rs, world_points), margin_factor
            )

            image_points_rs_prev = image_points_rs

        # Points that went through iterative refinement are returned as is.
        # Only the onles that were not valid before the loop are returned as init_image_points.
        final_image_points = torch.where(
            valid[..., None], image_points_rs_prev, init_image_points
        )

        # If the point in the last iteration was invalid, the point is marked as invalid.
        # It'll be returned as is.
        valid = valid & valid_rs

        # Postconditions
        assert_shape("final_image_points", final_image_points, B + M + (2,))
        assert_shape("valid", valid, B + M)
        return final_image_points, valid

    def check_image_bounds(
        self,
        points_2d: Tensor,  # [M, 2] - 2D image points
        margin_factor: float,
    ) -> Tensor:
        """Check if 2D image points are within image bounds with margin.

        Matches CUDA image_point_in_image_bounds_margin function.

        Args:
            points_2d: 2D image coordinates [M, 2]
            margin_factor: Tolerance for out-of-bounds (0.1 = 10% margin)

        Returns:
            valid_bounds: [M] - True if point is within bounds
        """
        # Preconditions
        M = points_2d.shape[:-1]
        assert_shape("points_2d", points_2d, M + (2,))

        u = points_2d[..., 0]  # [M]
        v = points_2d[..., 1]  # [M]

        margin_x = self.width * margin_factor
        margin_y = self.height * margin_factor

        in_bounds_x = (u >= -margin_x) & (u < self.width + margin_x)
        in_bounds_y = (v >= -margin_y) & (v < self.height + margin_y)
        valid_bounds = in_bounds_x & in_bounds_y

        # Postconditions
        assert_shape("valid_bounds", valid_bounds, M)

        return valid_bounds


class _PerfectPinholeCameraModel(_BaseCameraModel):
    def __init__(
        self,
        focal_lengths: Tensor,  # [B, 2]
        principal_points: Tensor,  # [B, 2]
        width: int,
        height: int,
        rs_type: RollingShutterType,
    ):
        # Preconditions
        B = focal_lengths.shape[:-1]
        assert_shape("focal_lengths", focal_lengths, B + (2,))
        assert_shape("principal_points", principal_points, B + (2,))

        super().__init__(width, height, rs_type)
        self._focal_lengths = focal_lengths  # [B, 2]
        self._principal_points = principal_points  # [B, 2]

    @property
    def focal_lengths(self) -> Tensor:
        return self._focal_lengths

    @property
    def principal_points(self) -> Tensor:
        return self._principal_points

    def camera_ray_to_image_point(
        self,
        cam_ray: Tensor,
        margin_factor: float,
    ) -> Tuple[Tensor, Tensor]:
        # Preconditions
        M = cam_ray.shape[:-1]
        assert_shape("cam_ray", cam_ray, M + (3,))

        # Points behind camera are invalid
        valid_depth = cam_ray[..., 2] > 0.0

        # Perspective projection: [x/z, y/z]
        uv = cam_ray[..., :2] / cam_ray[..., 2:3]

        image_point = _project_to_image(
            uv, self.focal_lengths[..., None, :], self.principal_points[..., None, :]
        )

        # Zero out points behind camera
        # (CUDA behavior: only depth invalidity zeros coordinates)
        image_point = torch.where(
            valid_depth[..., None], image_point, torch.zeros_like(image_point)
        )

        # Check if points are within image bounds
        valid_bounds = self.check_image_bounds(image_point, margin_factor)

        valid = valid_depth & valid_bounds

        # Postconditions
        assert_shape("image_point", image_point, M + (2,))
        assert_shape("valid", valid, M)

        return image_point, valid

    def image_point_to_camera_ray(
        self,
        image_point: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Unproject image points to camera rays using perfect pinhole model.
        Args:
            image_point: [M, 2] image points
        Returns:
            camera_ray: [M, 3] camera rays
            valid: [M] validity mask
        """
        # Preconditions
        M = image_point.shape[:-1]
        assert_shape("image_point", image_point, M + (2,))

        # Unproject the image point to camera ray
        camera_ray = _unproject_from_image(
            image_point,
            self.focal_lengths[..., None, :],
            self.principal_points[..., None, :],
        )
        camera_ray = torch.cat(
            [camera_ray, torch.ones_like(camera_ray[..., :1])], dim=-1
        )

        result = _safe_normalize(camera_ray)
        valid = torch.full_like(camera_ray[..., 0], True, dtype=torch.bool)

        # Postconditions
        assert_shape("result", result, M + (3,))
        assert_shape("valid", valid, M)
        return result, valid


class _OpenCVPinholeCameraModel(_BaseCameraModel):
    def __init__(
        self,
        focal_lengths: Tensor,  # [B, 2]
        principal_points: Tensor,  # [B, 2]
        width: int,
        height: int,
        rs_type: RollingShutterType,
        radial_coeffs: Optional[Tensor] = None,  # [B, 4] or [B, 6]
        tangential_coeffs: Optional[Tensor] = None,  # [B, 2]
        thin_prism_coeffs: Optional[Tensor] = None,  # [B, 4]
        max_undistortion_iterations: int = 5,
        min_2d_norm: float = 1e-12,
    ):
        # Preconditions
        B = focal_lengths.shape[:-1]
        assert_shape("focal_lengths", focal_lengths, B + (2,))
        assert_shape("principal_points", principal_points, B + (2,))
        radial_coeffs is None or assert_shape(
            "radial_coeffs", radial_coeffs[:-1], B + (1,)
        )
        tangential_coeffs is None or assert_shape(
            "tangential_coeffs", tangential_coeffs, B + (2,)
        )
        thin_prism_coeffs is None or assert_shape(
            "thin_prism_coeffs", thin_prism_coeffs, B + (4,)
        )

        super().__init__(width, height, rs_type)
        self._focal_lengths = focal_lengths  # [B, 2]
        self._principal_points = principal_points  # [B, 2]
        self.max_undistortion_iterations = max_undistortion_iterations
        self.min_2d_norm = min_2d_norm

        device = focal_lengths.device
        dtype = focal_lengths.dtype

        # Radial coefficients: [B, 4] or [B, 6]
        if radial_coeffs is not None:
            # Pad to 6 components
            radial_coeffs = F.pad(
                radial_coeffs, (0, 6 - radial_coeffs.shape[-1]), value=0.0
            )
            self.radial_coeffs = radial_coeffs
        else:
            self.radial_coeffs = torch.zeros(B + (6,), device=device, dtype=dtype)

        # Tangential coefficients: [B, 2]
        if tangential_coeffs is not None:
            self.tangential_coeffs = tangential_coeffs
        else:
            self.tangential_coeffs = torch.zeros(B + (2,), device=device, dtype=dtype)

        # Thin prism coefficients: [B, 4]
        if thin_prism_coeffs is not None:
            self.thin_prism_coeffs = thin_prism_coeffs
        else:
            self.thin_prism_coeffs = torch.zeros(B + (4,), device=device, dtype=dtype)

        # Postconditions
        assert_shape("radial_coeffs", self.radial_coeffs, B + (6,))
        assert_shape("tangential_coeffs", self.tangential_coeffs, B + (2,))
        assert_shape("thin_prism_coeffs", self.thin_prism_coeffs, B + (4,))

    @property
    def focal_lengths(self) -> Tensor:
        return self._focal_lengths

    @property
    def principal_points(self) -> Tensor:
        return self._principal_points

    def _compute_distortion(
        self,
        uv: Tensor,  # [M, 2] - normalized image coords
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Preconditions
        M = uv.shape[:-1]
        assert_shape("uv", uv, M + (2,))

        # Extract individual coefficients
        k1 = self.radial_coeffs[..., 0:1]  # [B, 1]
        k2 = self.radial_coeffs[..., 1:2]  # [B, 1]
        k3 = self.radial_coeffs[..., 2:3]  # [B, 1]
        k4 = self.radial_coeffs[..., 3:4]  # [B, 1]
        k5 = self.radial_coeffs[..., 4:5]  # [B, 1]
        k6 = self.radial_coeffs[..., 5:6]  # [B, 1]

        p1 = self.tangential_coeffs[..., 0:1]  # [B, 1]
        p2 = self.tangential_coeffs[..., 1:2]  # [B, 1]

        s1 = self.thin_prism_coeffs[..., 0:1]  # [B, 1]
        s2 = self.thin_prism_coeffs[..., 1:2]  # [B, 1]
        s3 = self.thin_prism_coeffs[..., 2:3]  # [B, 1]
        s4 = self.thin_prism_coeffs[..., 3:4]  # [B, 1]

        u = uv[..., 0]  # [B, M]
        v = uv[..., 1]  # [B, M]

        # Compute auxiliary terms (always computed in CUDA)
        uv_squared = torch.stack([u * u, v * v], dim=-1)  # [B, M, 2]
        r2 = uv_squared[..., 0] + uv_squared[..., 1]  # [B, M]
        a1 = 2.0 * u * v  # [B, M]
        a2 = r2 + 2.0 * uv_squared[..., 0]  # [B, M]
        a3 = r2 + 2.0 * uv_squared[..., 1]  # [B, M]

        # Compute icD (radial distortion factor)
        icD_numerator = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        icD_denominator = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))

        # No clamping - let icD be as extreme as needed to match CUDA behavior
        icD = icD_numerator / icD_denominator  # [B, M]

        # Compute delta (tangential + thin prism)
        delta_x = p1 * a1 + p2 * a2 + r2 * (s1 + r2 * s2)  # [B, M]
        delta_y = p1 * a3 + p2 * a1 + r2 * (s3 + r2 * s4)  # [B, M]
        delta = torch.stack([delta_x, delta_y], dim=-1)  # [B, M, 2]

        # Postconditions
        assert_shape("icD", icD, M)
        assert_shape("delta", delta, M + (2,))
        assert_shape("r2", r2, M)

        return icD, delta, r2

    def camera_ray_to_image_point(
        self,
        cam_ray: Tensor,
        margin_factor: float,
    ) -> Tuple[Tensor, Tensor]:
        # Preconditions
        M = cam_ray.shape[:-1]
        assert_shape("cam_ray", cam_ray, M + (3,))

        # Points behind camera are invalid
        valid_depth = cam_ray[..., 2] > 0.0

        # Perspective projection: [x/z, y/z]
        uv = cam_ray[..., :2] / cam_ray[..., 2:3]

        icD, delta, r2 = self._compute_distortion(uv)
        valid_distortion = icD > 0.8

        uvND = icD.unsqueeze(-1) * uv + delta

        # Apply intrinsics: [u, v] * [fx, fy] + [cx, cy] -> [M, 2]
        image_point = _project_to_image(
            uvND, self.focal_lengths[..., None, :], self.principal_points[..., None, :]
        )

        valid_bounds = self.check_image_bounds(image_point, margin_factor)

        valid = valid_depth & valid_distortion & valid_bounds

        # Postconditions
        assert_shape("image_point", image_point, M + (2,))
        assert_shape("valid", valid, M)

        return image_point, valid

    def _compute_undistortion_iterative(
        self,
        image_point: torch.Tensor,  # [M, 2] - distorted image points
    ) -> torch.Tensor:
        """
        Undistort image points using an iterative method.
        Args:
            image_point: [M, 2] distorted image points (in normalized camera space)
        Returns:
            uv_undistorted: [M, 2] undistorted camera normalized coordinates
        """
        # Preconditions
        M = image_point.shape[:-1]
        assert_shape("image_point", image_point, M + (2,))

        # Initial guess for the undistorted point
        cam_point_0 = _unproject_from_image(
            image_point,
            self.focal_lengths[..., None, :],
            self.principal_points[..., None, :],
        )

        # Start from distorted points as initial guess
        uv_hat = cam_point_0
        for i in range(self.max_undistortion_iterations):
            # Compute current distortion for uv_hat
            icD, delta, _ = self.compute_distortion(uv_hat)

            uv_next = (uv_0 - delta) / icD

            residual = uv_hat - uv_next

            converged = torch.dot(residual, residual) < self.min_2d_norm

            if torch.all(converged, dim=-1):
                break

            uv_hat = uv_next

        # Postconditions
        assert_shape("uv_hat", uv_hat, M + (2,))

        return uv_hat

    def _compute_residual_and_jacobian(
        self, uv_hat: torch.Tensor, uv: torch.Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes the residual and Jacobian for the Newton method of undistortion.
        This matches the CUDA implementation exactly.

        Args:
            uv_hat: [M, 2], current estimate of undistorted coordinates (x, y)
            uv: [M, 2], target distorted coordinates (xd, yd)
        Returns:
            residual: [M, 2], the residual [fx, fy] where fx=0, fy=0 is the solution
            jacobian: [M, 2, 2], the Jacobian matrix [[fx_x, fx_y], [fy_x, fy_y]]
            valid: [M], the validity mask
        """
        # Preconditions
        M = uv_hat.shape[:-1]
        assert_shape("uv_hat", uv_hat, M + (2,))
        assert_shape("uv", uv, M + (2,))

        x = uv_hat[..., 0]
        y = uv_hat[..., 1]
        xd = uv[..., 0]
        yd = uv[..., 1]

        k1 = self.radial_coeffs[..., 0:1]  # [B, 1]
        k2 = self.radial_coeffs[..., 1:2]  # [B, 1]
        k3 = self.radial_coeffs[..., 2:3]  # [B, 1]
        k4 = self.radial_coeffs[..., 3:4]  # [B, 1]
        k5 = self.radial_coeffs[..., 4:5]  # [B, 1]
        k6 = self.radial_coeffs[..., 5:6]  # [B, 1]

        p1 = self.tangential_coeffs[..., 0:1]  # [B, 1]
        p2 = self.tangential_coeffs[..., 1:2]  # [B, 1]

        s1 = self.thin_prism_coeffs[..., 0:1]  # [B, 1]
        s2 = self.thin_prism_coeffs[..., 1:2]  # [B, 1]
        s3 = self.thin_prism_coeffs[..., 2:3]  # [B, 1]
        s4 = self.thin_prism_coeffs[..., 3:4]  # [B, 1]

        # Compute r = x² + y²
        r = x * x + y * y
        r2 = r * r

        # Compute α = 1 + k1·r + k2·r² + k3·r³
        # Compute β = 1 + k4·r + k5·r² + k6·r³
        # Compute d = α/β (inverse radial distortion coefficient)
        alpha = 1.0 + r * (k1 + r * (k2 + r * k3))
        beta = 1.0 + r * (k4 + r * (k5 + r * k6))
        d = alpha / beta  # iCD

        # Negative iCD means the distortion makes point flipped across
        # the image center. This cannot be produced by real lenses.
        valid = d > 0.0

        # The perfect projection is:
        # xd = x·d + 2·p1·x·y + p2·(r + 2·x²) + s1·r + s2·r²
        # yd = y·d + 2·p2·x·y + p1·(r + 2·y²) + s3·r + s4·r²
        #
        # We define the residual as:
        # fx = x·d + 2·p1·x·y + p2·(r + 2·x²) + s1·r + s2·r² - xd
        # fy = y·d + 2·p2·x·y + p1·(r + 2·y²) + s3·r + s4·r² - yd
        #
        # The solution satisfies fx = 0 and fy = 0

        fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) + s1 * r + s2 * r2 - xd
        fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) + s3 * r + s4 * r2 - yd

        # Compute derivatives for the Jacobian
        # First, compute derivatives of α and β w.r.t. r
        alpha_r = k1 + r * (2.0 * k2 + r * (3.0 * k3))
        beta_r = k4 + r * (2.0 * k5 + r * (3.0 * k6))

        # Compute derivative of d w.r.t. r (using quotient rule)
        d_r = (alpha_r * beta - alpha * beta_r) / (beta * beta)

        # Compute derivative of d w.r.t. x and y (using chain rule: ∂d/∂x = d_r·∂r/∂x)
        d_x = 2.0 * x * d_r
        d_y = 2.0 * y * d_r

        # Compute Jacobian components: ∂fx/∂x, ∂fx/∂y, ∂fy/∂x, ∂fy/∂y
        fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
        fx_x = fx_x + 2.0 * x * (s1 + 2.0 * s2 * r)

        fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y
        fx_y = fx_y + 2.0 * y * (s1 + 2.0 * s2 * r)

        fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
        fy_x = fy_x + 2.0 * x * (s3 + 2.0 * s4 * r)

        fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y
        fy_y = fy_y + 2.0 * y * (s3 + 2.0 * s4 * r)

        # Stack residuals and jacobian
        residual = torch.stack([fx, fy], dim=-1)  # [M, 2]
        jacobian = torch.stack(
            [torch.stack([fx_x, fx_y], dim=-1), torch.stack([fy_x, fy_y], dim=-1)],
            dim=-2,
        )  # [M, 2, 2]

        # Expand valid mask for broadcasting: [M] -> [M, 1] for residual, [M, 1, 1] for jacobian
        residual = torch.where(valid[..., None], residual, torch.zeros_like(residual))
        jacobian = torch.where(
            valid[..., None, None], jacobian, torch.zeros_like(jacobian)
        )

        # Postconditions
        assert_shape("residual", residual, M + (2,))
        assert_shape("jacobian", jacobian, M + (2, 2))
        assert_shape("valid", valid, M)

        return residual, jacobian, valid

    def _compute_undistortion_newton(
        self,
        image_point: torch.Tensor,  # [M, 2] - distorted image points (normalized camera coords)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Undistort image points using the Newton-Raphson method.
        Args:
            image_point: [M, 2] distorted image points (in normalized camera space)
        Returns:
            uv_undistorted: [M, 2] undistorted camera normalized coordinates
            converged: [M], the convergence mask
        """
        # Preconditions
        M = image_point.shape[:-1]
        assert_shape("image_point", image_point, M + (2,))

        # Initial guess: undistorted normalized cam coordinates
        uv_0 = _unproject_from_image(
            image_point,
            self.focal_lengths[..., None, :],
            self.principal_points[..., None, :],
        )

        # No points have converged initially
        converged = torch.zeros_like(image_point[..., 0], dtype=torch.bool)

        # All points are valid initially
        valid_points = torch.ones_like(image_point[..., 0], dtype=torch.bool)

        eps: float = 1e-6
        uv_hat = uv_0
        for i in range(self.max_undistortion_iterations):
            res, J, valid_jac = self._compute_residual_and_jacobian(
                uv_hat, uv_0
            )  # [M, 2], [M, 2, 2]
            valid_points = valid_points & valid_jac

            det = torch.linalg.det(J)
            valid_points = valid_points & (torch.abs(det) >= eps)

            fx = res[..., 0]
            fy = res[..., 1]
            fx_x = J[..., 0, 0]
            fx_y = J[..., 0, 1]
            fy_x = J[..., 1, 0]
            fy_y = J[..., 1, 1]

            # Compute delta = -J⁻¹·res
            delta = -torch.stack(
                [(fx * fy_y - fy * fx_y) / det, (fy * fx_x - fx * fy_x) / det], dim=-1
            )

            # Do not update points that already converged or aren't valid
            uv_hat = torch.where(
                (converged | ~valid_points)[..., None], uv_hat, uv_hat + delta
            )

            # Check if both delta.x and delta.y are less than eps
            delta_converged = (torch.abs(delta[..., 0]) < eps) & (
                torch.abs(delta[..., 1]) < eps
            )
            converged = converged | (valid_points & delta_converged)

        # Postconditions
        assert_shape("uv_hat", uv_hat, M + (2,))
        assert_shape("converged", converged, M)

        return uv_hat, converged

    def image_point_to_camera_ray(
        self,
        image_point: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Preconditions
        M = image_point.shape[:-1]
        assert_shape("image_point", image_point, M + (2,))

        uv, converged = self._compute_undistortion_newton(image_point)

        camera_ray = torch.stack(
            [uv[..., 0], uv[..., 1], torch.ones_like(uv[..., 0])], dim=-1
        )

        camera_ray = _safe_normalize(camera_ray)

        # Postconditions
        assert_shape("camera_ray", camera_ray, M + (3,))
        assert_shape("converged", converged, M)
        return camera_ray, converged


class _OpenCVFisheyeCameraModel(_BaseCameraModel):
    """
    OpenCV-compatible fisheye camera model.

    Implements the OpenCV fisheye distortion model with up to 4 radial coefficients.
    Uses a 9th-degree odd polynomial for forward projection and Newton iteration
    for inverse (unprojection).
    """

    def __init__(
        self,
        focal_lengths: Tensor,  # [B, 2]
        principal_points: Tensor,  # [B, 2]
        width: int,
        height: int,
        rs_type: RollingShutterType,
        radial_coeffs: Optional[Tensor] = None,  # [B, 4]
        min_2d_norm: float = 1e-6,
        newton_iterations: int = 20,
    ):
        """
        Initialize OpenCV fisheye camera model.

        Args:
            focal_lengths: [B, 2] focal lengths (fx, fy)
            principal_points: [B, 2] principal points (cx, cy)
            width: Image width in pixels
            height: Image height in pixels
            rs_type: Rolling shutter type
            radial_coeffs: [B, 4] radial distortion coefficients [k1, k2, k3, k4]
            min_2d_norm: Minimum 2D norm to avoid singularities at image center
            newton_iterations: Number of Newton iterations for polynomial inversion
        """
        # Preconditions
        B = focal_lengths.shape[:-1]
        assert_shape("focal_lengths", focal_lengths, B + (2,))
        assert_shape("principal_points", principal_points, B + (2,))
        radial_coeffs is None or assert_shape("radial_coeffs", radial_coeffs, B + (4,))

        super().__init__(width, height, rs_type)

        device = focal_lengths.device
        dtype = focal_lengths.dtype

        self.min_2d_norm = min_2d_norm
        self._focal_lengths = focal_lengths  # [B, 2]
        self._principal_points = principal_points  # [B, 2]
        self.newton_iterations = newton_iterations

        # Radial coefficients: [B, 4]
        if radial_coeffs is not None:
            self.radial_coeffs = radial_coeffs
        else:
            self.radial_coeffs = torch.zeros(B + (4,), device=device, dtype=dtype)

        # Extract coefficients
        k1 = self.radial_coeffs[..., 0]  # [B]
        k2 = self.radial_coeffs[..., 1]  # [B]
        k3 = self.radial_coeffs[..., 2]  # [B]
        k4 = self.radial_coeffs[..., 3]  # [B]

        # Initialize 9th-degree odd-only forward polynomial
        # Maps angles to normalized distances: θ¹ + k₁·θ³ + k₂·θ⁵ + k₃·θ⁷ + k₄·θ⁹
        self.forward_poly_odd = OddPolynomialProxy(
            torch.stack(
                [
                    torch.ones_like(k1),  # coefficient for θ¹
                    k1,  # coefficient for θ³
                    k2,  # coefficient for θ⁵
                    k3,  # coefficient for θ⁷
                    k4,  # coefficient for θ⁹
                ],
                dim=-1,
            )
        )  # [B, 5]

        # 8th-degree even polynomial (derivative of forward polynomial)
        # 1 + 3·k₁·θ² + 5·k₂·θ⁴ + 7·k₃·θ⁶ + 9·k₄·θ⁸
        self.dforward_poly_even = EvenPolynomialProxy(
            torch.stack(
                [
                    torch.ones_like(k1),  # constant term
                    3.0 * k1,  # coefficient for θ²
                    5.0 * k2,  # coefficient for θ⁴
                    7.0 * k3,  # coefficient for θ⁶
                    9.0 * k4,  # coefficient for θ⁸
                ],
                dim=-1,
            )
        )  # [B, 5]

        # Compute maximum diagonal distances from principal point to image corners
        fx = focal_lengths[..., 0]  # [B]
        fy = focal_lengths[..., 1]  # [B]
        cx = principal_points[..., 0]  # [B]
        cy = principal_points[..., 1]  # [B]

        max_diag_x = torch.maximum(width - cx, cx)  # [B]
        max_diag_y = torch.maximum(height - cy, cy)  # [B]
        max_radius_pixels = torch.sqrt(
            max_diag_x * max_diag_x + max_diag_y * max_diag_y
        )  # [B]

        # Compute max_angle (maximum valid angle for the fisheye model)
        k4_is_zero = torch.abs(k4) < 1e-10  # [B]

        # Case 1: k4 == 0 (simpler case)
        # Solve for where derivative polynomial equals zero
        max_angle_k4_zero = torch.sqrt(
            self._compute_max_angle(
                3.0 * k1,  # a
                5.0 * k2,  # b
                7.0 * k3,  # c
            )
        )  # [B, 3]

        # Case 2: k₄ ≠ 0 (full cubic case)
        # Use Newton iteration to find where derivative equals zero
        ddforward_poly_odd = OddPolynomialProxy(
            torch.stack(
                [
                    6.0 * k1,  # coefficient for θ¹
                    20.0 * k2,  # coefficient for θ³
                    42.0 * k3,  # coefficient for θ⁵
                    72.0 * k4,  # coefficient for θ⁷
                ],
                dim=-1,
            )
        )  # [B, 4]

        # Initial approximation: θ ≈ π/2
        approx_poly_even = EvenPolynomialProxy(
            torch.ones((*B, 1), device=device, dtype=dtype) * 1.57
        )  # [B,1]

        # Calculate the maximum angle where the derivative polynomial equals zero.
        max_angle_k4_nonzero, converged = _eval_poly_inverse_horner_newton(
            self.dforward_poly_even,
            ddforward_poly_odd,
            approx_poly_even,
            torch.zeros_like(k1[..., None]),  # [B,1] target value (zero)
            n_iterations=newton_iterations,
        )  # [B,1]
        max_angle_k4_nonzero = max_angle_k4_nonzero.squeeze(-1)  # [B]
        converged = converged.squeeze(-1)  # [B]

        # Mark invalid solutions
        max_angle_k4_nonzero = torch.where(
            converged & (max_angle_k4_nonzero > 0.0),
            max_angle_k4_nonzero,
            torch.tensor(float("inf"), device=device, dtype=dtype),
        )  # [B]

        # Choose between the two cases
        max_angle = torch.where(
            k4_is_zero, max_angle_k4_zero, max_angle_k4_nonzero
        )  # [B]

        # Clamp max_angle to image bounds
        max_angle = torch.minimum(
            max_angle, torch.maximum(max_radius_pixels / fx, max_radius_pixels / fy)
        )  # [B]

        self.max_angle = max_angle  # [B]

        # Approximate backward polynomial (linear approximation for initial guess)
        # Maps normalized distances to angles (very crude approximation)
        max_normalized_dist = torch.maximum(width / 2.0 / fx, height / 2.0 / fy)  # [B]

        self.approx_backward_poly = FullPolynomialProxy(
            torch.stack(
                [
                    torch.zeros_like(max_angle),  # constant term (0)
                    max_angle / max_normalized_dist,  # linear term
                ],
                dim=-1,
            )
        )  # [B, 2]

        # Postconditions
        assert (
            self.max_angle.shape == B
        ), f"max_angle must have shape {B}, got {self.max_angle.shape}"
        assert_shape("approx_backward_poly", self.approx_backward_poly.coeffs, B + (2,))
        assert_shape("forward_poly_odd", self.forward_poly_odd.coeffs, B + (5,))
        assert_shape("dforward_poly_even", self.dforward_poly_even.coeffs, B + (5,))
        assert_shape("radial_coeffs", self.radial_coeffs, B + (4,))

    @property
    def focal_lengths(self) -> Tensor:
        return self._focal_lengths

    @property
    def principal_points(self) -> Tensor:
        return self._principal_points

    def _compute_max_angle(self, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        """
        Solve 1 + a·x + b·x² + c·x³ = 0 for the smallest positive root.

        This finds the maximum angle for the fisheye distortion model by solving
        for where the derivative polynomial equals zero.

        Args:
            a: [B] coefficient of x term
            b: [B] coefficient of x² term
            c: [B] coefficient of x³ term

        Returns:
            max_angle: [B] smallest positive root, or inf if none exists
        """
        # Preconditions
        B = a.shape
        assert_shape("a", a, B)
        assert_shape("b", b, B)
        assert_shape("c", c, B)

        INF = torch.tensor(float("inf"), device=a.device, dtype=a.dtype)
        PI = torch.tensor(math.pi, device=a.device, dtype=a.dtype)

        # Case 1: c == 0 (quadratic or linear)
        is_c_zero = torch.abs(c) < 1e-10  # [B]

        # Case 1a: c == 0 and b == 0 (linear)
        is_linear = is_c_zero & (torch.abs(b) < 1e-10)
        linear_result = torch.where(a >= 0.0, INF, -1.0 / a)

        # Case 1b: c == 0 and b != 0 (quadratic)
        delta_quad = a * a - 4.0 * b
        has_quad_solution = is_c_zero & ~is_linear & (delta_quad >= 0.0)
        delta_sqrt = torch.sqrt(delta_quad)
        delta_term = delta_sqrt - a
        quad_result = torch.where(delta_term > 0.0, 2.0 / delta_term, INF)

        # Case 2: c != 0 (cubic)
        boc = b / c
        boc2 = boc * boc

        t1 = (9.0 * a * boc - 2.0 * b * boc2 - 27.0) / c
        t2 = 3.0 * a / c - boc2
        delta_cubic = t1 * t1 + 4.0 * t2 * t2 * t2

        # Case 2a: delta >= 0 (one real root)
        has_real_root = ~is_c_zero & (delta_cubic >= 0.0)
        d2 = torch.sqrt(delta_cubic)
        cube_root = torch.sign((d2 + t1) / 2.0) * torch.pow(
            torch.abs((d2 + t1) / 2.0), 1.0 / 3.0
        )
        real_root_result = torch.where(
            cube_root != 0, (cube_root - (t2 / cube_root) - boc) / 3.0, INF
        )
        real_root_result = torch.where(real_root_result > 0.0, real_root_result, INF)

        # Case 2b: delta < 0 (three real roots)
        has_three_roots = ~is_c_zero & (delta_cubic < 0.0)
        theta = torch.atan2(torch.sqrt(-delta_cubic), t1) / 3.0
        two_third_pi = 2.0 * PI / 3.0

        t3 = 2.0 * torch.sqrt(-t2)

        # Try three roots: theta - 2π/3, theta, theta + 2π/3
        soln = INF
        for i in [-1, 0, 1]:
            angle = theta + i * two_third_pi
            s = (t3 * torch.cos(angle) - boc) / 3.0
            s = torch.where(s > 0.0, s, INF)
            soln = torch.minimum(soln, s)

        # Combine all cases
        result = torch.where(
            is_linear,
            linear_result,
            torch.where(
                has_quad_solution,
                quad_result,
                torch.where(
                    has_real_root,
                    real_root_result,
                    torch.where(has_three_roots, soln, INF),
                ),
            ),
        )

        # Postconditions
        assert_shape("result", result, B)

        return result

    def camera_ray_to_image_point(
        self,
        cam_ray: Tensor,
        margin_factor: float,
    ) -> Tuple[Tensor, Tensor]:
        """
        Project camera rays to image points using fisheye distortion.

        Implements forward projection: 3D camera ray → 2D image point

        Args:
            cam_ray: Camera rays with direction vectors [B, M, 3]
            margin_factor: Margin factor for boundary checking

        Returns:
            image_point: [B, M, 2] projected image coordinates
            valid: [B, M] validity mask
        """
        # Preconditions
        B = self.focal_lengths.shape[:-1]
        M = (
            (math.prod(cam_ray.shape[:-1]) // math.prod(B),)
            if cam_ray.ndim != self.focal_lengths.ndim
            else ()
        )
        assert_shape("cam_ray", cam_ray, B + M + (3,))

        # Points behind camera are invalid
        valid = cam_ray[..., 2] > 0.0  # [B, M]

        # Compute norm of xy components using numerically stable method
        cam_ray_xy_norm = _numerically_stable_norm2(
            cam_ray[..., 0], cam_ray[..., 1]
        )  # [B, M]
        # If norm is zero (point along principal axis), set to epsilon
        cam_ray_xy_norm = torch.where(
            cam_ray_xy_norm <= 0.0,
            torch.full_like(cam_ray_xy_norm, torch.finfo(cam_ray.dtype).eps),
            cam_ray_xy_norm,
        )

        # Compute angle θ = atan2(‖xy‖, z)
        theta_full = torch.atan2(cam_ray_xy_norm, cam_ray[..., 2])  # [B, M]

        # Clamp θ to max_angle to prevent projections outside valid FOV
        theta = torch.minimum(theta_full, self.max_angle[..., None])  # [B, M]

        # Evaluate forward polynomial to get normalized radial distance
        # δ = (θ + k₁·θ³ + k₂·θ⁵ + ...) / ‖xy‖
        poly_value = self.forward_poly_odd.eval_horner(theta)  # [B, M]
        delta = poly_value / cam_ray_xy_norm  # [B, M]

        # Negative delta means flipped across center (physically impossible)
        valid = valid & (delta > 0.0)  # [B, M]

        # Perspective projection
        uv = delta[..., None] * cam_ray[..., :2]

        image_point = _project_to_image(
            uv, self.focal_lengths[..., None, :], self.principal_points[..., None, :]
        )

        # Check image bounds
        valid_bounds = self.check_image_bounds(image_point, margin_factor)  # [B,M]

        # Mark FOV-clamped points as invalid
        valid = valid & (theta <= self.max_angle[..., None]) & valid_bounds  # [B,M]

        # Postconditions
        assert_shape("image_point", image_point, B + M + (2,))
        assert_shape("valid", valid, B + M)

        return image_point, valid

    def image_point_to_camera_ray(
        self,
        image_point: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Unproject image points to camera rays using fisheye undistortion.

        Implements inverse projection: 2D image point → 3D camera ray

        Args:
            image_point: [M, 2] image coordinates

        Returns:
            Camera ray [M, 3]
            Valid mask [M]
        """
        # Preconditions
        M = image_point.shape[:-1]
        assert_shape("image_point", image_point, M + (2,))

        # Normalize image coordinates to get uv (unproject from image)
        uv = _unproject_from_image(
            image_point,
            self.focal_lengths[..., None, :],
            self.principal_points[..., None, :],
        )  # [M, 2]

        # Compute radial distance (normalized)
        delta = torch.linalg.norm(uv, dim=-1)  # [..., M]

        # Evaluate inverse polynomial using Newton's method to find θ
        # Solve: forward_poly(θ) = δ
        # PyTorch broadcasting handles shape [..., N] with [..., M] automatically

        theta, converged = _eval_poly_inverse_horner_newton(
            self.forward_poly_odd,
            self.dforward_poly_even,
            self.approx_backward_poly,
            delta,  # [..., M]
            n_iterations=self.newton_iterations,
        )  # [..., M], [..., M]

        # Mark invalid: negative θ, θ ≥ max_angle, or not converged
        valid = (
            (theta >= 0.0) & (theta < self.max_angle[..., None]) & converged
        )  # [..., M]

        # Compute camera ray direction
        # At image center (δ ~ 0), return straight ahead [0, 0, 1]
        # Otherwise: [sin(θ) · uv / δ, cos(θ)]
        is_center = delta < self.min_2d_norm  # [..., M]

        # Non-center case: scale by sin(θ)/δ
        scale_factor = torch.sin(theta) / delta  # [..., M]
        camera_ray = torch.stack(
            [
                scale_factor * uv[..., 0],
                scale_factor * uv[..., 1],
                torch.cos(theta),
            ],
            dim=-1,
        )  # [..., M, 3]

        # Center case: straight ahead
        camera_ray_center = torch.stack(
            [
                torch.zeros_like(theta),
                torch.zeros_like(theta),
                torch.ones_like(theta),
            ],
            dim=-1,
        )  # [..., M, 3]

        camera_ray = torch.where(is_center[..., None], camera_ray_center, camera_ray)

        # Postconditions
        assert_shape("camera_ray", camera_ray, M + (3,))
        assert_shape("valid", valid, M)

        return camera_ray, valid


# ============================================================================
# FTheta Camera Model
# ============================================================================


class _FThetaCameraModel(_BaseCameraModel):
    """
    FTheta camera model with polynomial distortion.

    Matches CUDA FThetaCameraModel implementation.

    This model supports two modes:
    - PIXELDIST_TO_ANGLE: backward poly is reference (forward via inverse)
    - ANGLE_TO_PIXELDIST: forward poly is reference (backward via inverse)

    The model applies a linear transform [c,d;e,1] to the distorted coordinates.
    """

    def __init__(
        self,
        principal_points: Tensor,  # [B, 2]
        width: int,
        height: int,
        rs_type: RollingShutterType,
        dist_params: FThetaCameraDistortionParameters,
        min_2d_norm: float = 1e-6,
        newton_iterations: int = 3,
    ):
        """
        Initialize FTheta camera model.

        Matches CUDA FThetaCameraModel constructor.

        Args:
            principal_point: [B, 2] principal point (cx, cy)
            width: Image width in pixels
            height: Image height in pixels
            rs_type: Rolling shutter type
            dist_params: FTheta distortion parameters
            min_2d_norm: Minimum 2D norm for valid points
            newton_iterations: Number of Newton iterations for polynomial inversion

        Note:
            The dist_params corresponds to FThetaCameraDistortionParameters in the
            CUDA code (parameters.dist). The focal length is embedded in the
            polynomial distortion model, not a separate parameter.
        """

        # Preconditions
        B = principal_points.shape[:-1]
        assert_shape("principal_points", principal_points, B + (2,))
        assert (
            len(dist_params.pixeldist_to_angle_poly) == 6
        ), "pixeldist_to_angle_poly must have 6 coefficients"
        assert (
            len(dist_params.angle_to_pixeldist_poly) == 6
        ), "angle_to_pixeldist_poly must have 6 coefficients"
        assert len(dist_params.linear_cde) == 3, "linear_cde must have 3 coefficients"

        super().__init__(width, height, rs_type)

        device = principal_points.device
        dtype = principal_points.dtype
        self.reference_poly_type = dist_params.reference_poly

        # FTheta model convention: image origin = center of first pixel
        # Therefore offset principal point by +0.5 (matches CUDA lines 1075-1076)
        self._principal_points = principal_points + 0.5  # [B, 2]
        if len(B) == 0:
            self.max_angle = torch.tensor(
                dist_params.max_angle, device=device, dtype=dtype
            )
        else:
            self.max_angle = torch.tensor(
                [dist_params.max_angle], device=device, dtype=dtype
            ).expand(
                *B
            )  # [B]
        self.linear_cde = (
            torch.tensor(dist_params.linear_cde)
            .clone()
            .to(device=device, dtype=dtype)
            .expand(*B, -1)
            .squeeze(-1)
        )  # [B,3]
        self.pixeldist_to_angle_poly = (
            torch.tensor(dist_params.pixeldist_to_angle_poly)
            .clone()
            .to(device=device, dtype=dtype)
            .expand(*B, -1)
        )  # [B,6]
        self.angle_to_pixeldist_poly = (
            torch.tensor(dist_params.angle_to_pixeldist_poly)
            .clone()
            .to(device=device, dtype=dtype)
            .expand(*B, -1)
        )  # [B,6]
        self.min_2d_norm = min_2d_norm
        self.newton_iterations = newton_iterations

        device = principal_points.device
        dtype = principal_points.dtype

        # Convert polynomial coefficients to Tensors and create PolynomialProxy objects
        self.angle_to_pixeldist_poly = FullPolynomialProxy(
            self.angle_to_pixeldist_poly
        )  # [B,6]
        self.pixeldist_to_angle_poly = FullPolynomialProxy(
            self.pixeldist_to_angle_poly
        )  # [B,6]

        # Compute derivative of reference polynomial (matches CUDA lines 1066-1071)
        # Derivative: d/dx[c₀ + c₁·x + c₂·x² + ...] = c₁ + 2·c₂·x + 3·c₃·x² + ...
        if self.reference_poly_type == FThetaPolynomialType.PIXELDIST_TO_ANGLE:
            # Backward poly is reference: derivative of pixeldist_to_angle_poly
            self.dreference_poly = FullPolynomialProxy(
                torch.stack(
                    [
                        1.0 * self.pixeldist_to_angle_poly.coeffs[..., 1],
                        2.0 * self.pixeldist_to_angle_poly.coeffs[..., 2],
                        3.0 * self.pixeldist_to_angle_poly.coeffs[..., 3],
                        4.0 * self.pixeldist_to_angle_poly.coeffs[..., 4],
                        5.0 * self.pixeldist_to_angle_poly.coeffs[..., 5],
                    ],
                    dim=-1,
                )
            )  # [B,5]
        else:
            assert self.reference_poly_type == FThetaPolynomialType.ANGLE_TO_PIXELDIST
            # Forward poly is reference: derivative of angle_to_pixeldist_poly
            self.dreference_poly = FullPolynomialProxy(
                torch.stack(
                    [
                        1.0 * self.angle_to_pixeldist_poly.coeffs[..., 1],
                        2.0 * self.angle_to_pixeldist_poly.coeffs[..., 2],
                        3.0 * self.angle_to_pixeldist_poly.coeffs[..., 3],
                        4.0 * self.angle_to_pixeldist_poly.coeffs[..., 4],
                        5.0 * self.angle_to_pixeldist_poly.coeffs[..., 5],
                    ],
                    dim=-1,
                )
            )  # [B,5]

        # Postconditions
        assert_shape(
            "angle_to_pixeldist_poly", self.angle_to_pixeldist_poly.coeffs, B + (6,)
        )
        assert_shape(
            "pixeldist_to_angle_poly", self.pixeldist_to_angle_poly.coeffs, B + (6,)
        )
        assert_shape("dreference_poly", self.dreference_poly.coeffs, B + (5,))
        assert_shape("max_angle", self.max_angle, B)
        assert_shape("linear_cde", self.linear_cde, B + (3,))
        assert_shape("principal_points", self.principal_points, B + (2,))
        assert_shape("focal_lengths", self.focal_lengths, B + (2,))

    @property
    def principal_points(self) -> Tensor:
        return self._principal_points

    @property
    def focal_lengths(self) -> Tensor:
        # Focal length is proportional to the linear term of the angle_to_pixeldist_poly

        # Try to use the reference polynomial for a better estimate.
        if self.reference_poly_type == FThetaPolynomialType.PIXELDIST_TO_ANGLE:
            flen = 1.0 / self.pixeldist_to_angle_poly.coeffs[..., [1, 1]]
        else:
            assert self.reference_poly_type == FThetaPolynomialType.ANGLE_TO_PIXELDIST
            flen = self.angle_to_pixeldist_poly.coeffs[..., [1, 1]]

        return flen.expand(self.principal_points.shape)

    def camera_ray_to_image_point(
        self,
        cam_ray: Tensor,
        margin_factor: float,
    ) -> Tuple[Tensor, Tensor]:
        """
        Project camera rays to image points using FTheta distortion.

        Implements forward projection: 3D camera ray → 2D image point

        Args:
            cam_ray: Camera rays with direction [M, 3]
            margin_factor: Margin for image bounds checking

        Returns:
            image_point: [M, 2] projected image coordinates
            valid: [M] validity mask
        """
        # Preconditions
        M = cam_ray.shape[:-1]
        assert_shape("cam_ray", cam_ray, M + (3,))

        # Points behind camera are invalid
        not_behind_camera = cam_ray[..., 2] > 0.0  # [M]

        # Compute norm of xy components using numerically stable method
        cam_ray_xy_norm = _numerically_stable_norm2(
            cam_ray[..., 0], cam_ray[..., 1]
        )  # [M]
        # If norm is zero (point along principal axis), set to epsilon
        cam_ray_xy_norm = torch.where(
            cam_ray_xy_norm <= 0.0,
            torch.full_like(cam_ray_xy_norm, torch.finfo(cam_ray.dtype).eps),
            cam_ray_xy_norm,
        )

        # Compute angle θ = atan2(‖xy‖, z)
        theta_full = torch.atan2(cam_ray_xy_norm, cam_ray[..., 2])  # [M]

        # Limit angles to max_angle to prevent projected points to leave valid cone around max_angle.
        # In particular for omnidirectional cameras, this prevents points outside the FOV to be
        # wrongly projected to in-image-domain points because of badly constrained polynomials outside
        # the effective FOV (which is different to the image boundaries).

        # These FOV-clamped projections will be marked as *invalid*
        theta = theta_full.clamp(max=self.max_angle[..., None])  # [M]

        # Evaluate forward polynomial to get delta = f(θ)
        # Choice depends on which polynomial is the reference
        if self.reference_poly_type == FThetaPolynomialType.PIXELDIST_TO_ANGLE:
            # Backward poly is reference: forward via Newton inverse
            delta, converged = _eval_poly_inverse_horner_newton(
                self.pixeldist_to_angle_poly,
                self.dreference_poly,
                self.angle_to_pixeldist_poly,
                theta,  # [M]
                n_iterations=self.newton_iterations,
            )  # [M]
        else:
            # Forward poly is reference: direct evaluation
            delta = self.angle_to_pixeldist_poly.eval_horner(theta)  # [M]
            converged = torch.ones_like(delta, dtype=torch.bool)  # Always converged

        # Apply delta to normalized xy to get f(θ)-weighted 2D vectors
        # Then apply linear transform A = [[c, d], [e, 1]]
        c = self.linear_cde[..., 0:1]  # [B,1]
        d = self.linear_cde[..., 1:2]  # [B,1]
        e = self.linear_cde[..., 2:3]  # [B,1]
        cx = self.principal_points[..., 0:1]  # [B,1]
        cy = self.principal_points[..., 1:2]  # [B,1]

        # Normalized xy vectors and apply delta scaling
        image_point_x = delta * cam_ray[..., 0] / cam_ray_xy_norm  # [M]
        image_point_y = delta * cam_ray[..., 1] / cam_ray_xy_norm  # [M]

        # Apply linear transform relative to principal point
        image_point = torch.stack(
            [
                c * image_point_x + d * image_point_y + cx,
                e * image_point_x + image_point_y + cy,
            ],
            dim=-1,
        )  # [M, 2]

        # Check image bounds (matches CUDA image_point_in_image_bounds_margin)
        valid_bounds = self.check_image_bounds(image_point, margin_factor)  # [M]

        # Mark FOV-clamped points as invalid
        # TODO: This isn't happening, we need to compare against theta_full (not clamped)!
        valid = (
            not_behind_camera
            & converged
            & (theta <= self.max_angle[..., None])
            & valid_bounds
        )

        # Set to zero the image_points behind camera or that didn't converge.
        image_point = image_point * (converged & not_behind_camera)[..., None]

        # Postconditions
        assert_shape("image_point", image_point, M + (2,))
        assert_shape("valid", valid, M)

        return image_point, valid

    def image_point_to_camera_ray(
        self,
        image_point: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Unproject image points to camera rays using FTheta undistortion.

        Implements inverse projection: 2D image point → 3D camera ray

        Matches CUDA FThetaCameraModel::image_point_to_camera_ray (lines 1140-1187)

        Args:
            image_point: [M, 2] image coordinates

        Returns:
            Camera ray [M, 3]
            Valid mask [M]
        """
        # Preconditions
        M = image_point.shape[:-1]
        assert_shape("image_point", image_point, M + (2,))

        # Invert linear transform to get f(θ)-weighted normalized 2D vectors
        # Linear transform: A = [[c, d], [e, 1]]
        # Inverse: A⁻¹ = [[1, -d], [-e, c]] / (c - e·d)
        # Matches CUDA lines 1142-1145
        c = self.linear_cde[..., 0:1]  # [B,1]
        d = self.linear_cde[..., 1:2]  # [B,1]
        e = self.linear_cde[..., 2:3]  # [B,1]
        cx = self.principal_points[..., 0:1]  # [B,1]
        cy = self.principal_points[..., 1:2]  # [B,1]

        # Subtract principal point
        px = image_point[..., 0] - cx  # [M]
        py = image_point[..., 1] - cy  # [M]

        # Apply inverse linear transform
        det_inv = 1.0 / (c - e * d)  # [B,1]
        uv = torch.stack(
            [
                (px - d * py) * det_inv,
                (-e * px + c * py) * det_inv,
            ],
            dim=-1,
        )  # [M, 2]

        # Compute radial distance (normalized)
        delta = torch.linalg.norm(uv, dim=-1)  # [M]

        # Evaluate backward polynomial to get θ = f⁻¹(δ)
        # Choice depends on which polynomial is the reference
        if self.reference_poly_type == FThetaPolynomialType.PIXELDIST_TO_ANGLE:
            # Backward poly is reference: direct evaluation
            theta = self.pixeldist_to_angle_poly.eval_horner(delta)  # [M]
            converged = torch.ones_like(theta, dtype=torch.bool)  # Always converged
        else:
            # Forward poly is reference: backward via Newton inverse
            theta, converged = _eval_poly_inverse_horner_newton(
                self.angle_to_pixeldist_poly,
                self.dreference_poly,
                self.pixeldist_to_angle_poly,
                delta,  # [M]
                n_iterations=self.newton_iterations,
            )  # [M]

        # Compute camera ray direction
        # At image center (δ ~ 0), return straight ahead [0, 0, 1]
        # Otherwise: [sin(θ) · uv / δ, cos(θ)]
        # Matches CUDA lines 1173-1186
        is_center = delta < self.min_2d_norm  # [M]

        # Non-center case: scale by sin(θ)/δ
        scale_factor = torch.sin(theta) / delta  # [M]
        camera_ray = torch.stack(
            [
                scale_factor * uv[..., 0],
                scale_factor * uv[..., 1],
                torch.cos(theta),
            ],
            dim=-1,
        )  # [M, 3]

        # Center case: straight ahead
        camera_ray_center = torch.stack(
            [
                torch.zeros_like(theta),
                torch.zeros_like(theta),
                torch.ones_like(theta),
            ],
            dim=-1,
        )  # [M, 3]

        camera_ray = torch.where(
            (is_center | ~converged)[..., None], camera_ray_center, camera_ray
        )

        camera_ray = _safe_normalize(camera_ray)

        # Postconditions
        assert_shape("camera_ray", camera_ray, M + (3,))
        assert_shape("converged", converged, M)

        return camera_ray, converged


def _interpolate_shutter_pose(
    pose_start: Tensor,  # [B, 7]
    pose_end: Tensor,  # [B, 7]
    relative_time: Tensor,  # [B]
) -> Tensor:  # [B, 7]
    """
    Interpolate a shutter pose between two poses.

    Args:
        pose_start: Start pose [B, 7] with format [t_x, t_y, t_z, q_w, q_x, q_y, q_z]
        pose_end: End pose [B, 7] with format [t_x, t_y, t_z, q_w, q_x, q_y, q_z]
        relative_time: Interpolation time [B]

    Returns:
        Interpolated pose [B, 7] with format [t_x, t_y, t_z, q_w, q_x, q_y, q_z]
    """
    # Preconditions
    B = relative_time.shape
    assert_shape("relative_time", relative_time, B)
    assert_shape("pose_start", pose_start, B + (7,))
    assert_shape("pose_end", pose_end, B + (7,))

    # Extract translation and quaternion from 7D pose tensors
    t_start = pose_start[..., :3]  # [B, 3]
    q_start = pose_start[..., 3:]  # [B, 4]
    t_end = pose_end[..., :3]  # [B, 3]
    q_end = pose_end[..., 3:]  # [B, 4]

    alpha = relative_time[..., None]

    # Single interpolation code path (alpha broadcasts correctly in both cases)
    t_rs = (1.0 - alpha) * t_start + alpha * t_end  # [B, 3]
    q_rs = _quat_slerp(q_start, q_end, alpha.squeeze(-1))  # [B, 4]
    q_rs = _safe_normalize(q_rs)

    # Concatenate back to 7D pose tensor
    result = torch.cat([t_rs, q_rs], dim=-1)  # [B, 7]

    # Postconditions
    assert_shape("result", result, B + (7,))

    return result
