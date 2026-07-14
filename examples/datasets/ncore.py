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

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data
from scipy import ndimage

import ncore.data
import ncore.data.v4
import ncore.sensors
from ncore.data import PointCloudsSourceProtocol
from ncore.impl.common.transformations import (
    bbox_pose,
    is_within_3d_bboxes,
    se3_inverse,
    transform_point_cloud,
)

from gsplat.rendering import FThetaCameraDistortionParameters, FThetaPolynomialType

from .ncore_utils import FrameConversion
from .normalize import (
    similarity_from_cameras,
    align_principal_axes,
    transform_cameras,
    transform_points,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CameraRenderData:
    """Per-camera rendering parameters for gsplat rasterization."""

    camera_model: str  # "pinhole" | "fisheye" | "ftheta"
    ftheta_coeffs: Optional[
        FThetaCameraDistortionParameters
    ]  # non-None only for ftheta
    radial_coeffs: Optional[np.ndarray]  # (4,) fisheye or (4|6,) pinhole; float32
    tangential_coeffs: Optional[np.ndarray]  # (2,) pinhole only; float32 or None
    thin_prism_coeffs: Optional[np.ndarray]  # (4,) pinhole only; float32 or None


@dataclasses.dataclass
class RigidDynamicTrack:
    """A single dynamic object reconstructed as a rigid component.

    Gaussians are initialised from lidar points expressed in the object's local
    (centroid-centred, axis-aligned) frame. The per-frame SE(3) poses map that
    local frame into the scene frame at each annotated timestamp.
    """

    track_id: str
    class_id: str
    points_local: np.ndarray  # (P, 3) float32 — init points in object-local frame
    points_rgb: np.ndarray  # (P, 3) uint8
    frame_timestamps_us: np.ndarray  # (F,) int64, sorted — pose keyframe times
    poses_local_to_scene: np.ndarray  # (F, 4, 4) float32 — local -> scene per frame


def _normalize_track_class_id(class_id: Any) -> str:
    """Normalize NCore cuboid class IDs before matching them."""
    return str(class_id).strip().lower()


def frame_midpoint_timestamp_us(start_us, end_us):
    """Frame-midpoint render time for rigid dynamics (int or tensor, µs).

    Used when per-ray timestamps are unavailable. Shared by training and
    sample_inference so both render dynamic tracks at the same time.
    """
    return start_us + (end_us - start_us) // 2


def _build_pinhole_K(
    model_params: ncore.data.OpenCVPinholeCameraModelParameters
    | ncore.data.OpenCVFisheyeCameraModelParameters,
) -> np.ndarray:
    """Return a 3x3 pinhole intrinsic matrix. Caller guarantees focal_length and principal_point exist."""
    fl = model_params.focal_length
    pp = model_params.principal_point
    fx = float(fl[0]) if hasattr(fl, "__getitem__") else float(fl)
    fy = float(fl[1]) if hasattr(fl, "__getitem__") else float(fl)
    cx, cy = float(pp[0]), float(pp[1])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _load_ego_mask(
    sensor: ncore.data.CameraSensorProtocol, n_dilation: int
) -> Optional[np.ndarray]:
    """Return a dilated boolean ego mask (True = ego vehicle) or None."""
    mask_images = sensor.get_mask_images()
    if "ego" not in mask_images:
        return None
    mask = np.asarray(mask_images["ego"].convert("L")) != 0
    return ndimage.binary_dilation(mask, iterations=n_dilation).astype(bool)


def _parse_optional_coeffs(coeffs: Optional[Any]) -> Optional[np.ndarray]:
    """Convert optional distortion coefficients to float32 array, mapping all-zero arrays to None."""
    if coeffs is None:
        return None
    coeffs_array = np.array(coeffs, dtype=np.float32)
    if (coeffs_array == 0).all():
        return None
    return coeffs_array


# ---------------------------------------------------------------------------
# NCoreParser
# ---------------------------------------------------------------------------


class NCoreParser:
    """NCore v4 data parser.

    Loads all frame metadata (poses, K matrices, frame lists) eagerly at init
    time. Images are loaded lazily by NCoreDataset.__getitem__.

    Coordinate frames
    -----------------
    NCore world frame  - raw sequence frame from the SLAM/pose graph.
                         Origin is typically the start of the drive;
                         units are real-world metres.

    world_global frame - globally-consistent reference frame obtained
                         from the pose graph edge "world" -> "world_global".
                         T_world_to_scene_world rotates/aligns into this frame.

    scene frame        - world_global translated so that the mean camera
                         position is at the origin.  This is the frame stored
                         in self.camtoworlds and consumed by the trainer.
                         Keeping poses near the origin improves numerical
                         stability during 3DGS optimisation.
                         When normalize_world_space=True, an additional
                         similarity + PCA transform is applied on top.
                         world_global_to_scene (FrameConversion) applies only
                         this translation; all rotation is already handled by
                         T_world_to_scene_world.
    """

    def __init__(
        self,
        meta_json_path: str,
        factor: float = 1.0,
        test_every: int = 8,
        camera_ids: Optional[List[str]] = None,
        lidar_ids: Optional[List[str]] = None,
        seek_offset_sec: Optional[float] = None,
        duration_sec: Optional[float] = None,
        max_lidar_points: int = 500_000,
        lidar_step_frame: int = 1,
        poses_component_group: str = "default",
        intrinsics_component_group: str = "default",
        masks_component_group: str = "default",
        open_consolidated: bool = True,
        n_camera_mask_dilation_iterations: int = 30,
        lidar_color_generic_data_name: str = "rgb",
        normalize_world_space: bool = False,
        rigid_dynamic_track_class_ids: Optional[Collection[str]] = None,
        keep_dynamic_points_in_static_scene: bool = True,
    ) -> None:
        self.test_every = test_every
        self.factor = factor
        self.normalize_world_space = normalize_world_space
        self.rigid_dynamic_track_class_ids = (
            frozenset(
                _normalize_track_class_id(class_id)
                for class_id in rigid_dynamic_track_class_ids
            )
            if rigid_dynamic_track_class_ids is not None
            else None
        )
        if (
            self.rigid_dynamic_track_class_ids is not None
            and not self.rigid_dynamic_track_class_ids
        ):
            raise ValueError(
                "rigid_dynamic_track_class_ids must be non-empty when provided"
            )
        self.poses_component_group = poses_component_group
        self.intrinsics_component_group = intrinsics_component_group
        self.masks_component_group = masks_component_group
        self.open_consolidated = open_consolidated
        self.lidar_color_generic_data_name = lidar_color_generic_data_name
        self.keep_dynamic_points_in_static_scene = keep_dynamic_points_in_static_scene

        self.sequence_meta_file_path: Path = Path(meta_json_path)
        sequence_loader = self._open_sequence_loader(self.sequence_meta_file_path)

        self.sequence_id: str = sequence_loader.sequence_id

        time_range = sequence_loader.sequence_timestamp_interval_us
        start_us = time_range.start
        stop_us = time_range.stop
        if seek_offset_sec is not None:
            start_us += int(seek_offset_sec * 1e6)
        if duration_sec is not None and duration_sec > 0:
            stop_us = min(start_us + int(duration_sec * 1e6), stop_us)
        self.time_range_us = dataclasses.replace(
            time_range, start=start_us, stop=stop_us
        )

        self._resolve_sensor_ids(sequence_loader, camera_ids, lidar_ids)
        self._compute_world_global_transform(sequence_loader)

        camera_sensors = self._load_camera_data(
            sequence_loader, factor, n_camera_mask_dilation_iterations
        )
        camera_frame_ranges = {
            cid: self._get_sensor_frame_range(camera_sensors[cid].frames_timestamps_us)
            for cid in self.camera_ids
        }
        self._compute_scene_origin(camera_sensors, camera_frame_ranges)
        self._load_poses(camera_sensors, camera_frame_ranges)

        # Stub attrs for render_traj compatibility
        self.bounds = np.array([0.01, 1.0])
        self.extconf = {"spiral_radius_scale": 1.0, "no_factor_suffix": False}

        rigid_split_enabled = (
            self.rigid_dynamic_track_class_ids is not None
            and not self.keep_dynamic_points_in_static_scene
        )
        static_max_points = None if rigid_split_enabled else max_lidar_points
        self.points, self.points_rgb = self._load_point_clouds(
            sequence_loader, static_max_points, lidar_step_frame
        )

        # Rigid dynamic objects: per-track local Gaussians + per-frame poses.
        # Loaded before normalization so the same similarity transform is applied
        # to the tracks and the static points/cameras, keeping their scene-frame
        # poses consistent.
        self.rigid_dynamic_tracks: List[RigidDynamicTrack] = (
            self._load_rigid_dynamic_tracks(sequence_loader, lidar_step_frame)
            if self.rigid_dynamic_track_class_ids is not None
            else []
        )
        if rigid_split_enabled:
            self._sample_init_points(max_lidar_points)

        # Normalize the world space (orient, centre, and rescale).
        if self.normalize_world_space:
            self._normalize_world_space()

        # Scene scale: max distance of each camera from the mean camera position.
        # This matches the COLMAP convention (colmap.py:396-400).
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = float(np.max(dists))

        logger.info(
            "Loaded sequence '%s': %d frames across %d cameras, "
            "%d lidar init points, scene_scale=%.3f",
            self.sequence_id,
            len(self.frame_list),
            self.num_cameras,
            len(self.points),
            self.scene_scale,
        )

    # ------------------------------------------------------------------
    # Private init helpers
    # ------------------------------------------------------------------

    def _open_sequence_loader(self, path: Path) -> ncore.data.SequenceLoaderProtocol:
        """Open and return a SequenceLoaderV4 for this parser's component groups."""

        assert path.is_file(), f"NCoreParser: path {path} is not a file"
        with open(path, "r") as fp:
            dataset_meta = json.load(fp)
        assert all(
            key in dataset_meta
            for key in (
                "sequence_id",
                "sequence_timestamp_interval_us",
                "version",
                "component_stores",
            )
        ), f"NCoreParser: {path} is not a NCore v4 single-sequence meta-file"

        return ncore.data.v4.SequenceLoaderV4(
            ncore.data.v4.SequenceComponentGroupsReader(
                [path], open_consolidated=self.open_consolidated
            ),
            poses_component_group_name=self.poses_component_group,
            intrinsics_component_group_name=self.intrinsics_component_group,
            masks_component_group_name=self.masks_component_group,
        )

    def _resolve_sensor_ids(
        self,
        sequence_loader: ncore.data.SequenceLoaderProtocol,
        camera_ids: Optional[List[str]],
        lidar_ids: Optional[List[str]],
    ) -> None:
        """Auto-detect sensor IDs if not provided; set camera_ids and lidar_ids."""

        # Auto-detect _single_ sensors if not specified - sensors need to be specified explicitly
        # to avoid ambiguity (e.g., in case of multiple downscaled sensors)
        if not camera_ids:
            camera_ids = sequence_loader.camera_ids

            if len(camera_ids) > 1:
                raise ValueError(
                    "NCoreParser: Multiple camera sensors in dataset, explicit"
                    f" specification of a (subset) of camera sensors required to avoid ambiguity: {camera_ids}"
                )

            logger.info("Auto-detected cameras: %s", camera_ids)
        if not lidar_ids:
            point_clouds_source_ids = list(sequence_loader.lidar_ids) + list(
                sequence_loader.point_clouds_ids
            )

            if len(point_clouds_source_ids) > 1:
                raise ValueError(
                    "NCoreParser: Multiple point cloud sources in dataset, explicit"
                    f" specification of a (subset) of sources required to avoid ambiguity: {lidar_ids}"
                )

            logger.info("Auto-detected point cloud sources: %s", lidar_ids)
        else:
            point_clouds_source_ids = lidar_ids

        assert all(
            cid in sequence_loader.camera_ids for cid in camera_ids
        ), f"NCoreParser: some specified camera_ids {camera_ids} not found in dataset cameras {sequence_loader.camera_ids}"

        all_point_cloud_ids = set(sequence_loader.lidar_ids) | set(
            sequence_loader.point_clouds_ids
        )
        assert all(
            pid in all_point_cloud_ids for pid in point_clouds_source_ids
        ), f"NCoreParser: some specified lidar_ids {lidar_ids} not found in dataset point cloud sources {all_point_cloud_ids}"

        self.camera_ids: List[str] = list(camera_ids)
        self.point_clouds_source_ids: List[str] = list(point_clouds_source_ids)
        self.num_cameras: int = len(self.camera_ids)

        logger.info("Using cameras: %s", self.camera_ids)
        logger.info("Using point cloud sources: %s", self.point_clouds_source_ids)

    def _compute_world_global_transform(
        self, sequence_loader: ncore.data.SequenceLoaderProtocol
    ) -> None:
        """Set T_world_to_scene_world: transformation from NCore world -> world_global."""
        if (
            edge := sequence_loader.pose_graph.get_edge("world", "world_global")
        ) is not None:
            self.T_world_to_scene_world: np.ndarray = np.linalg.inv(
                edge.T_source_target
            ).astype(np.float32)
        else:
            self.T_world_to_scene_world = np.eye(4, dtype=np.float32)

    def _load_camera_data(
        self,
        sequence_loader: ncore.data.SequenceLoaderProtocol,
        factor: float,
        n_dilation: int,
    ) -> Dict[str, ncore.data.CameraSensorProtocol]:
        """Load intrinsics, K matrices, and ego masks for all cameras."""
        camera_sensors = {
            cid: sequence_loader.get_camera_sensor(cid) for cid in self.camera_ids
        }
        self.Ks_dict: Dict[str, np.ndarray] = {}
        self.imsize_dict: Dict[str, Tuple[int, int]] = {}
        self.mask_dict: Dict[str, Optional[np.ndarray]] = {}
        self.camera_models: Dict[str, ncore.sensors.CameraModel] = {}
        self.camera_render_data: Dict[str, CameraRenderData] = {}

        for camera_id in self.camera_ids:
            sensor = camera_sensors[camera_id]
            model_params = sensor.model_parameters
            if factor != 1.0:
                try:
                    model_params = model_params.transform(image_domain_scale=factor)
                except (AssertionError, ValueError) as e:
                    logger.error(
                        "factor=%s produces non-integer resolution for %s; "
                        "using factor=1.0 (full resolution). "
                        "Pass --data-factor 1 to suppress this error.",
                        factor,
                        camera_id,
                    )
                    raise e

            camera_model = ncore.sensors.CameraModel.from_parameters(
                model_params, device="cpu", dtype=torch.float32
            )
            self.camera_models[camera_id] = camera_model

            width = int(camera_model.resolution[0].item())
            height = int(camera_model.resolution[1].item())
            self.imsize_dict[camera_id] = (width, height)

            if isinstance(model_params, ncore.data.FThetaCameraModelParameters):
                cx = float(model_params.principal_point[0].item())
                cy = float(model_params.principal_point[1].item())
                self.Ks_dict[camera_id] = np.array(
                    [[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], dtype=np.float32
                )
                ref_poly = FThetaPolynomialType[model_params.reference_poly.name]
                ftheta_coeffs = FThetaCameraDistortionParameters(
                    reference_poly=ref_poly,
                    pixeldist_to_angle_poly=tuple(
                        float(x) for x in model_params.pixeldist_to_angle_poly
                    ),
                    angle_to_pixeldist_poly=tuple(
                        float(x) for x in model_params.angle_to_pixeldist_poly
                    ),
                    max_angle=float(model_params.max_angle),
                    linear_cde=tuple(float(x) for x in model_params.linear_cde),
                )
                self.camera_render_data[camera_id] = CameraRenderData(
                    camera_model="ftheta",
                    ftheta_coeffs=ftheta_coeffs,
                    radial_coeffs=None,
                    tangential_coeffs=None,
                    thin_prism_coeffs=None,
                )
                logger.info("%s: %dx%d (ftheta)", camera_id, width, height)
            elif isinstance(
                model_params, ncore.data.OpenCVFisheyeCameraModelParameters
            ):
                self.Ks_dict[camera_id] = _build_pinhole_K(model_params)
                self.camera_render_data[camera_id] = CameraRenderData(
                    camera_model="fisheye",
                    ftheta_coeffs=None,
                    radial_coeffs=np.array(
                        model_params.radial_coeffs, dtype=np.float32
                    ),
                    tangential_coeffs=None,
                    thin_prism_coeffs=None,
                )
                logger.info("%s: %dx%d (opencv_fisheye)", camera_id, width, height)
            elif isinstance(
                model_params, ncore.data.OpenCVPinholeCameraModelParameters
            ):
                self.Ks_dict[camera_id] = _build_pinhole_K(model_params)
                self.camera_render_data[camera_id] = CameraRenderData(
                    camera_model="pinhole",
                    ftheta_coeffs=None,
                    radial_coeffs=_parse_optional_coeffs(
                        getattr(model_params, "radial_coeffs", None)
                    ),
                    tangential_coeffs=_parse_optional_coeffs(
                        getattr(model_params, "tangential_coeffs", None)
                    ),
                    thin_prism_coeffs=_parse_optional_coeffs(
                        getattr(model_params, "thin_prism_coeffs", None)
                    ),
                )
                logger.info("%s: %dx%d (opencv_pinhole)", camera_id, width, height)
            else:
                # Unknown camera type: synthesize K from resolution, treat as perfect pinhole.
                logger.info(
                    "%s: %dx%d (unknown, synthesizing K from resolution)",
                    camera_id,
                    width,
                    height,
                )
                self.Ks_dict[camera_id] = np.array(
                    [
                        [float(width), 0.0, width / 2.0],
                        [0.0, float(width), height / 2.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )
                self.camera_render_data[camera_id] = CameraRenderData(
                    camera_model="pinhole",
                    ftheta_coeffs=None,
                    radial_coeffs=None,
                    tangential_coeffs=None,
                    thin_prism_coeffs=None,
                )

            self.mask_dict[camera_id] = _load_ego_mask(sensor, n_dilation)

        return camera_sensors

    def _get_sensor_frame_range(self, frames_timestamps_us: np.ndarray) -> range:
        """Return the frame index range whose START and END timestamps fall in time_range_us."""
        cover = self.time_range_us.cover_range(
            frames_timestamps_us[:, ncore.data.FrameTimepoint.END]
        )
        if not len(cover):
            return cover
        start_ts = frames_timestamps_us[
            cover.start : cover.stop, ncore.data.FrameTimepoint.START
        ]
        first_valid = int(
            np.searchsorted(start_ts, self.time_range_us.start, side="left")
        )
        return range(cover.start + first_valid, cover.stop)

    def _compute_scene_origin(
        self,
        camera_sensors: Dict[str, ncore.data.CameraSensorProtocol],
        camera_frame_ranges: Dict[str, range],
    ) -> None:
        """Set world_global_to_scene: translation that centres poses at the origin."""
        positions: List[np.ndarray] = []
        for camera_id in self.camera_ids:
            frame_range = camera_frame_ranges[camera_id]
            if not len(frame_range):
                continue
            T_cam_world = camera_sensors[camera_id].get_frames_T_source_target(
                source_node=camera_id,
                target_node="world",
                frame_indices=np.arange(frame_range.start, frame_range.stop),
                frame_timepoint=ncore.data.FrameTimepoint.START,
            )  # [N, 4, 4]
            cam_positions = T_cam_world[:, :3, 3]
            positions.append(
                (
                    self.T_world_to_scene_world[:3, :3] @ cam_positions.T
                    + self.T_world_to_scene_world[:3, 3:4]
                ).T
            )

        mean_position = np.vstack(positions).mean(axis=0).astype(np.float32)
        self.world_global_to_scene = FrameConversion.from_origin_scale_axis(
            target_origin=mean_position,
            target_scale=1.0,
            target_axis=[0, 1, 2],
        )

    def _load_poses(
        self,
        camera_sensors: Dict[str, ncore.data.CameraSensorProtocol],
        camera_frame_ranges: Dict[str, range],
    ) -> None:
        """Batch-load start/end poses for all frames; set camtoworlds and scene_scale."""
        self.frame_list: List[Tuple[str, int]] = []
        self.camera_idx_per_frame: List[int] = []
        frame_timestamps: List[Tuple[int, int]] = []
        starts: List[np.ndarray] = []
        ends: List[np.ndarray] = []

        for cam_idx, camera_id in enumerate(self.camera_ids):
            frame_range = camera_frame_ranges[camera_id]
            if not len(frame_range):
                continue

            sensor = camera_sensors[camera_id]
            indices = np.arange(frame_range.start, frame_range.stop)
            T_start = self._ncore_world_to_scene_poses(
                sensor.get_frames_T_source_target(
                    source_node=camera_id,
                    target_node="world",
                    frame_indices=indices,
                    frame_timepoint=ncore.data.FrameTimepoint.START,
                ).reshape(-1, 4, 4)
            )  # [N, 4, 4]
            T_end = self._ncore_world_to_scene_poses(
                sensor.get_frames_T_source_target(
                    source_node=camera_id,
                    target_node="world",
                    frame_indices=indices,
                    frame_timepoint=ncore.data.FrameTimepoint.END,
                ).reshape(-1, 4, 4)
            )  # [N, 4, 4]

            # squeeze() in transform_poses may drop the batch dim when N=1
            if T_start.ndim == 2:
                T_start = T_start[np.newaxis]
            if T_end.ndim == 2:
                T_end = T_end[np.newaxis]

            sensor_ts = sensor.frames_timestamps_us
            for local_idx, frame_idx in enumerate(frame_range):
                self.frame_list.append((camera_id, frame_idx))
                self.camera_idx_per_frame.append(cam_idx)
                frame_timestamps.append(
                    (
                        int(sensor_ts[frame_idx, ncore.data.FrameTimepoint.START]),
                        int(sensor_ts[frame_idx, ncore.data.FrameTimepoint.END]),
                    )
                )
                starts.append(T_start[local_idx])
                ends.append(T_end[local_idx])

        self.camtoworlds = np.stack(starts, axis=0)  # (N, 4, 4)
        self.camtoworlds_end = np.stack(ends, axis=0)  # (N, 4, 4)
        # (N, 2) int64 START/END capture timestamps per frame_list entry, for
        # callers that need a render time (rigid dynamics) without reopening
        # the sequence loader.
        self.frame_timestamps_us = np.asarray(frame_timestamps, dtype=np.int64)

    def _normalize_world_space(self) -> None:
        """Normalize world-space coordinates for poses and points.

        Three successive transforms are applied:
        1. ``similarity_from_cameras`` - rotate so z+ is the up axis, recenter
           at the camera focus point, and rescale by 1/median camera distance.
        2. ``align_principal_axes`` - PCA rotation that aligns the point-cloud
           principal axes to the coordinate axes.
        3. Upside-down fix - if the point cloud is inverted (median z > mean z),
           apply a 180° rotation around the x-axis.

        Operates on ``self.camtoworlds``, ``self.camtoworlds_end``, and
        ``self.points`` in-place and stores the composed transform in
        ``self.transform``.
        """
        # Ensure float64 for numerical precision during normalization.
        camtoworlds = self.camtoworlds.astype(np.float64)
        camtoworlds_end = self.camtoworlds_end.astype(np.float64)
        points = self.points.astype(np.float64) if len(self.points) else self.points

        T1 = similarity_from_cameras(camtoworlds)
        camtoworlds = transform_cameras(T1, camtoworlds)
        camtoworlds_end = transform_cameras(T1, camtoworlds_end)
        if len(points):
            points = transform_points(T1, points)

        if len(points):
            T2 = align_principal_axes(points)
        else:
            T2 = np.eye(4)
        camtoworlds = transform_cameras(T2, camtoworlds)
        camtoworlds_end = transform_cameras(T2, camtoworlds_end)
        if len(points):
            points = transform_points(T2, points)

        transform = T2 @ T1

        # Upside-down fix: if median z > mean z, flip around x-axis.
        if len(points) and np.median(points[:, 2]) > np.mean(points[:, 2]):
            T3 = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            camtoworlds = transform_cameras(T3, camtoworlds)
            camtoworlds_end = transform_cameras(T3, camtoworlds_end)
            points = transform_points(T3, points)
            transform = T3 @ transform

        self.camtoworlds = camtoworlds
        self.camtoworlds_end = camtoworlds_end
        if len(self.points):
            self.points = points.astype(np.float32)
        self.transform = transform

        # Carry the same similarity onto rigid dynamic tracks so objects stay
        # aligned with the normalized background/cameras.
        if self.rigid_dynamic_tracks:
            self._normalize_rigid_dynamic_tracks(transform)

    def _normalize_rigid_dynamic_tracks(self, transform: np.ndarray) -> None:
        """Apply the world-space normalization similarity to rigid dynamic tracks.

        ``transform`` is a similarity ``x -> sQx + b`` (uniform scale ``s``,
        rotation ``Q``). A track renders as ``x = R_pose · p_local + t_pose``, so
        the normalized placement is ``(QR_pose)(s·p_local) + (sQ·t_pose + b)``:
        local points scale by ``s`` and each pose is left-multiplied by the
        similarity then re-orthonormalized (mirrors ``transform_cameras``).
        """
        # Uniform scale carried by the similarity (rows/cols of sQ have norm s).
        scale = float(np.linalg.norm(transform[0, :3]))
        for track in self.rigid_dynamic_tracks:
            track.points_local = (track.points_local * scale).astype(np.float32)
            poses = transform @ track.poses_local_to_scene.astype(np.float64)
            rot_scale = np.linalg.norm(poses[:, 0, :3], axis=1)  # (F,) == s
            poses[:, :3, :3] = poses[:, :3, :3] / rot_scale[:, None, None]
            track.poses_local_to_scene = poses.astype(np.float32)

    # ------------------------------------------------------------------
    # Private runtime helpers
    # ------------------------------------------------------------------

    def _ncore_world_to_scene_poses(self, T_poses_world: np.ndarray) -> np.ndarray:
        """Transform poses from NCore world frame to scene frame."""
        T_poses_common = self.T_world_to_scene_world @ T_poses_world.reshape(-1, 4, 4)
        return self.world_global_to_scene.transform_poses(T_poses_common)

    def _load_point_clouds(
        self,
        sequence_loader: ncore.data.SequenceLoaderProtocol,
        max_points: int | None,
        step_frame: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and transform point clouds to scene frame for Gaussian initialization.

        Supports any PointCloudsSourceProtocol source (lidar, radar, or native
        point clouds) via the unified ``get_point_clouds_source()`` API.
        """
        if not self.point_clouds_source_ids:
            logger.warning(
                "No point cloud sources available; using empty init point cloud"
            )
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        # Pre-compute the world-to-scene transform (identity in "sensor" space = world).
        T_world_scene = self._ncore_world_to_scene_poses(
            np.eye(4, dtype=np.float32)[np.newaxis]
        )  # (4, 4)
        scale = self.world_global_to_scene.target_scale

        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []

        for source_id in self.point_clouds_source_ids:
            source: PointCloudsSourceProtocol = sequence_loader.get_point_clouds_source(
                source_id
            )
            ts = source.pc_timestamps_us

            for pc_idx in range(source.pcs_count):
                # Time filtering
                pc_ts = int(ts[pc_idx])
                if not (self.time_range_us.start <= pc_ts < self.time_range_us.stop):
                    continue

                # Step frame
                if pc_idx % step_frame != 0:
                    continue

                try:
                    pc = source.get_pc(pc_idx)
                except Exception as exc:
                    logger.warning(
                        "Failed to load point cloud %d from %r: %s",
                        pc_idx,
                        source_id,
                        exc,
                    )
                    continue

                # Transform to world frame
                pc_world = pc.transform(
                    "world", pc.reference_frame_timestamp_us, sequence_loader.pose_graph
                )
                xyz_world = pc_world.xyz

                # Color: per-point uint8 RGB (validated), or None if unavailable.
                color = self._get_pc_color(
                    pc,
                    source,
                    pc_idx,
                    self.lidar_color_generic_data_name,
                    len(xyz_world),
                )

                # Dynamic-flag handling. By default NCore drops moving-object
                # returns (dynamic_flag == 1) so they don't smear the static
                # background. When rigid dynamic tracks are requested, callers
                # can temporarily keep those returns for old static-only
                # comparisons, or drop them for the split static/dynamic path.
                point_filter = ...
                if (
                    self.rigid_dynamic_track_class_ids is None
                    or not self.keep_dynamic_points_in_static_scene
                ) and source.has_pc_generic_data(pc_idx, "dynamic_flag"):
                    point_filter = (
                        source.get_pc_generic_data(pc_idx, "dynamic_flag") != 1
                    )
                xyz_world = xyz_world[point_filter]
                if color is not None:
                    color = color[point_filter]
                if not len(xyz_world):
                    continue

                # Apply world-to-scene transform
                xyz_scene = (
                    (scale * T_world_scene[:3, :3]) @ xyz_world.T
                    + T_world_scene[:3, 3:4]
                ).T
                all_points.append(xyz_scene.astype(np.float32))
                if color is not None:
                    all_colors.append(color)
                else:
                    all_colors.append(np.full((len(xyz_scene), 3), 128, dtype=np.uint8))

        if not all_points:
            logger.warning("No point cloud data loaded")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        points = np.vstack(all_points)
        points_rgb = np.vstack(all_colors)
        if max_points is not None and len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
            points_rgb = points_rgb[idx]

        source_names = ", ".join(f"'{s}'" for s in self.point_clouds_source_ids)
        logger.info("Loaded %d point cloud points from %s", len(points), source_names)
        return points, points_rgb

    def _sample_init_points(self, max_points: int) -> None:
        """Randomly cap static + rigid-dynamic init points to one total budget."""
        if max_points < 0:
            raise ValueError("max_lidar_points must be non-negative")

        static_count = len(self.points)
        dynamic_counts = [
            len(track.points_local) for track in self.rigid_dynamic_tracks
        ]
        total = static_count + sum(dynamic_counts)
        if total <= max_points:
            return

        keep_global = np.zeros(total, dtype=bool)
        keep_global[np.random.choice(total, max_points, replace=False)] = True

        static_keep = keep_global[:static_count]
        self.points = self.points[static_keep]
        self.points_rgb = self.points_rgb[static_keep]

        offset = static_count
        kept_tracks: List[RigidDynamicTrack] = []
        for track, count in zip(self.rigid_dynamic_tracks, dynamic_counts):
            keep = keep_global[offset : offset + count]
            offset += count
            if not np.any(keep):
                continue
            track.points_local = track.points_local[keep]
            track.points_rgb = track.points_rgb[keep]
            kept_tracks.append(track)
        self.rigid_dynamic_tracks = kept_tracks

        dynamic_count = sum(
            len(track.points_local) for track in self.rigid_dynamic_tracks
        )
        logger.info(
            "NCore init points: downsampled %d -> %d total "
            "(%d static + %d rigid dynamic)",
            total,
            len(self.points) + dynamic_count,
            len(self.points),
            dynamic_count,
        )

    @staticmethod
    def _get_pc_color(
        pc: Any,
        source: PointCloudsSourceProtocol,
        pc_idx: int,
        color_name: str,
        n_points: int,
    ) -> Optional[np.ndarray]:
        """Return per-point uint8 RGB for a point cloud, or None if unavailable."""
        color: Optional[np.ndarray] = None
        if pc.has_attribute(color_name):
            color = pc.get_attribute(color_name)
        elif source.has_pc_generic_data(pc_idx, color_name):
            color = source.get_pc_generic_data(pc_idx, color_name)
        if color is None:
            return None
        if color.shape != (n_points, 3):
            raise ValueError(
                "Color data length does not match point cloud length "
                "(expecting 3-channel RGB color per point)"
            )
        if color.dtype != np.uint8:
            raise ValueError("Expected color data in uint8 format")
        return color

    def _load_rigid_dynamic_tracks(
        self,
        sequence_loader: ncore.data.SequenceLoaderProtocol,
        step_frame: int,
    ) -> List[RigidDynamicTrack]:
        """Load moving objects as rigid tracks for dynamic-rigid training.

        Reads NCore cuboid track observations, derives a per-frame SE(3) pose for
        each track (object-local -> scene frame), associates the ``dynamic_flag``
        lidar points with their owning cuboid, and stores those points in each
        object's local frame.

        This reads the dynamic returns independently of ``_load_point_clouds``.
        When ``rigid_dynamic_track_class_ids`` is provided, those returns are
        *also* kept in the static background for now, while track-local copies
        are exposed on ``parser.rigid_dynamic_tracks``.
        """
        assert self.rigid_dynamic_track_class_ids is not None
        pose_graph = sequence_loader.pose_graph

        # 1) Group rigid cuboid observations by track and build per-frame world
        # bboxes. NCore class IDs are dataset-specific strings, so callers must
        # explicitly identify which classes they want to model as rigid.
        all_track_obs: Dict[str, List[ncore.data.CuboidTrackObservation]] = {}
        for obs in sequence_loader.get_cuboid_track_observations(self.time_range_us):
            all_track_obs.setdefault(obs.track_id, []).append(obs)

        skipped_by_class: Dict[str, int] = {}
        track_obs: Dict[str, List[ncore.data.CuboidTrackObservation]] = {}
        for track_id, obs_list in all_track_obs.items():
            class_ids = {_normalize_track_class_id(obs.class_id) for obs in obs_list}
            if class_ids <= self.rigid_dynamic_track_class_ids:
                track_obs[track_id] = obs_list
                continue
            for class_id in sorted(class_ids - self.rigid_dynamic_track_class_ids):
                skipped_by_class[class_id] = skipped_by_class.get(class_id, 0) + len(
                    obs_list
                )

        if not track_obs:
            logger.info(
                "rigid dynamic tracks: no matching cuboid track "
                "observations in time range"
            )
            return []

        if skipped_by_class:
            skipped = ", ".join(
                f"{class_id}={count}"
                for class_id, count in sorted(skipped_by_class.items())
            )
            logger.info(
                "rigid dynamic tracks: skipped cuboid observations "
                "outside configured class IDs: %s",
                skipped,
            )

        tracks_world: Dict[str, Dict[str, np.ndarray]] = {}
        for track_id, obs_list in track_obs.items():
            # Key on reference_frame_timestamp_us: cuboids are parameterised in a
            # sensor frame at the capture (lidar/render) timestamp, which aligns
            # with the point-cloud frames. obs.timestamp_us is the label time and
            # does NOT line up with frames.
            obs_list.sort(key=lambda o: o.reference_frame_timestamp_us)
            timestamps: List[int] = []
            bboxes_world: List[np.ndarray] = []
            for obs in obs_list:
                obs_world = obs.transform(
                    "world", obs.reference_frame_timestamp_us, pose_graph
                )
                timestamps.append(int(obs.reference_frame_timestamp_us))
                bboxes_world.append(
                    np.asarray(obs_world.bbox3.to_array(), dtype=np.float64)
                )
            ts = np.asarray(timestamps, dtype=np.int64)
            bbox_world = np.stack(bboxes_world, axis=0)  # (F, 9)

            # Per-frame local->world pose -> local->scene pose.
            poses_local_world = np.stack(
                [bbox_pose(b) for b in bbox_world], axis=0
            ).astype(
                np.float32
            )  # (F, 4, 4)
            poses_local_scene = self._ncore_world_to_scene_poses(poses_local_world)
            if poses_local_scene.ndim == 2:
                poses_local_scene = poses_local_scene[np.newaxis]

            tracks_world[track_id] = {
                "class_id": _normalize_track_class_id(obs_list[0].class_id),
                "ts": ts,
                "bbox_world": bbox_world,
                "pose_scene": poses_local_scene.astype(np.float32),
            }

        # Match tolerance: half a frame interval. Cuboid reference timestamps align
        # with the lidar frames, so the nearest match is normally exact; half an
        # interval guards float/int jitter without ever matching an adjacent frame
        # (which would associate points to the wrong, missing-this-frame keyframe).
        all_ts = np.unique(np.concatenate([tw["ts"] for tw in tracks_world.values()]))
        ts_tol = (
            max(1_000, int(0.5 * np.median(np.diff(all_ts))))
            if len(all_ts) > 1
            else 100_000
        )

        # 2) Associate dynamic points (per pc frame) with the nearest cuboid and
        #    accumulate them in each track's local frame.
        local_pts: Dict[str, List[np.ndarray]] = {tid: [] for tid in tracks_world}
        local_rgb: Dict[str, List[np.ndarray]] = {tid: [] for tid in tracks_world}

        for source_id in self.point_clouds_source_ids:
            source: PointCloudsSourceProtocol = sequence_loader.get_point_clouds_source(
                source_id
            )
            pc_timestamps = source.pc_timestamps_us
            for pc_idx in range(source.pcs_count):
                pc_ts = int(pc_timestamps[pc_idx])
                if not (self.time_range_us.start <= pc_ts < self.time_range_us.stop):
                    continue
                if pc_idx % step_frame != 0:
                    continue
                if not source.has_pc_generic_data(pc_idx, "dynamic_flag"):
                    continue
                try:
                    pc = source.get_pc(pc_idx)
                except Exception as exc:
                    logger.warning(
                        "rigid dynamic tracks: failed to load point cloud %d "
                        "from %r: %s",
                        pc_idx,
                        source_id,
                        exc,
                    )
                    continue

                dyn_mask = source.get_pc_generic_data(pc_idx, "dynamic_flag") == 1
                if not np.any(dyn_mask):
                    continue

                pc_world = pc.transform(
                    "world", pc.reference_frame_timestamp_us, pose_graph
                )
                xyz_world = pc_world.xyz
                color = self._get_pc_color(
                    pc,
                    source,
                    pc_idx,
                    self.lidar_color_generic_data_name,
                    len(xyz_world),
                )
                xyz_world = xyz_world[dyn_mask]
                color = color[dyn_mask] if color is not None else None

                # Assign each dynamic point to at most one track (first match wins).
                remaining = np.ones(len(xyz_world), dtype=bool)
                for track_id, tw in tracks_world.items():
                    nearest = int(np.argmin(np.abs(tw["ts"] - pc_ts)))
                    if abs(int(tw["ts"][nearest]) - pc_ts) > ts_tol:
                        continue
                    bbox = tw["bbox_world"][nearest]
                    inside = is_within_3d_bboxes(xyz_world, bbox[np.newaxis])[:, 0]
                    # Points expressed in the box-local (centroid-centred) frame.
                    local = transform_point_cloud(
                        xyz_world, se3_inverse(bbox_pose(bbox))
                    )
                    sel = inside & remaining
                    if not np.any(sel):
                        continue
                    local_pts[track_id].append(local[sel].astype(np.float32))
                    if color is not None:
                        local_rgb[track_id].append(color[sel])
                    else:
                        local_rgb[track_id].append(
                            np.full((int(sel.sum()), 3), 128, dtype=np.uint8)
                        )
                    remaining &= ~sel

        # 3) Emit tracks that picked up at least one point.
        tracks: List[RigidDynamicTrack] = []
        for track_id, tw in tracks_world.items():
            if not local_pts[track_id]:
                continue
            tracks.append(
                RigidDynamicTrack(
                    track_id=track_id,
                    class_id=str(tw["class_id"]),
                    points_local=np.vstack(local_pts[track_id]).astype(np.float32),
                    points_rgb=np.vstack(local_rgb[track_id]).astype(np.uint8),
                    frame_timestamps_us=tw["ts"],
                    poses_local_to_scene=tw["pose_scene"],
                )
            )

        total_pts = sum(len(t.points_local) for t in tracks)
        logger.info(
            "rigid dynamic tracks: %d/%d configured tracks with associated "
            "points (%d init points)",
            len(tracks),
            len(track_obs),
            total_pts,
        )
        return tracks


# ---------------------------------------------------------------------------
# NCoreDataset
# ---------------------------------------------------------------------------


class NCoreDataset(torch.utils.data.Dataset):
    """Image-based dataset for NCore v4 sequences.

    Returns batches compatible with gsplat trainers:
      {"K", "camtoworld", "image", "image_id", "camera_idx"}
    plus optional "camtoworld_end" (END pose for rolling shutter) and "mask".

    Images are loaded lazily per __getitem__. The underlying NCore sequence
    loader is (re-)opened per DataLoader worker to avoid sharing file handles.
    """

    def __init__(
        self,
        parser: NCoreParser,
        split: str = "train",
    ) -> None:
        self.parser = parser
        self.split = split

        # Build train/val split indices over the flat frame list.
        all_indices = np.arange(len(parser.frame_list))
        if split == "train":
            self.indices = all_indices[all_indices % parser.test_every != 0]
        else:
            self.indices = all_indices[all_indices % parser.test_every == 0]

        # Per-worker sequence loader (lazily initialised).
        self._sequence_loader: Optional[ncore.data.SequenceLoaderProtocol] = None
        self._camera_sensors: Optional[
            Dict[str, ncore.data.CameraSensorProtocol]
        ] = None
        self._current_worker_id: Optional[int] = None

    def _init_worker(self) -> None:
        """Open (or reopen) the NCore sequence loader for the current worker process."""
        worker_info = torch.utils.data.get_worker_info()
        current_worker_id: Optional[int] = (
            None if worker_info is None else worker_info.id
        )

        if self._sequence_loader is not None:
            if self._current_worker_id == current_worker_id:
                return  # already initialised for this worker
            # Worker ID changed: reload file handles.
            self._current_worker_id = current_worker_id
            self._sequence_loader.reload_resources()
            return

        # First-time initialisation for this process.
        self._current_worker_id = current_worker_id
        self._sequence_loader = ncore.data.v4.SequenceLoaderV4(
            ncore.data.v4.SequenceComponentGroupsReader(
                [self.parser.sequence_meta_file_path],
                open_consolidated=self.parser.open_consolidated,
            ),
            poses_component_group_name=self.parser.poses_component_group,
            intrinsics_component_group_name=self.parser.intrinsics_component_group,
            masks_component_group_name=self.parser.masks_component_group,
        )
        self._camera_sensors = {
            cid: self._sequence_loader.get_camera_sensor(cid)
            for cid in self.parser.camera_ids
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        self._init_worker()
        assert self._camera_sensors is not None

        index = self.indices[item]
        camera_id, frame_idx = self.parser.frame_list[index]
        camera_idx = self.parser.camera_idx_per_frame[index]

        sensor = self._camera_sensors[camera_id]
        width, height = self.parser.imsize_dict[camera_id]
        K = self.parser.Ks_dict[camera_id].copy()

        image = sensor.get_frame_image_array(frame_idx)  # HxWx3 uint8
        if self.parser.factor != 1.0:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        timestamp_start_us = int(
            sensor.frames_timestamps_us[frame_idx, ncore.data.FrameTimepoint.START]
        )
        timestamp_end_us = int(
            sensor.frames_timestamps_us[frame_idx, ncore.data.FrameTimepoint.END]
        )

        data: Dict[str, Any] = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(self.parser.camtoworlds[index]).float(),
            "camtoworld_end": torch.from_numpy(
                self.parser.camtoworlds_end[index]
            ).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
            "camera_idx": camera_idx,
            # Keep timestamp_us as the historical frame-start timestamp. Rigid
            # dynamic rendering uses the explicit start/end pair to choose the
            # frame midpoint when per-ray timestamps are unavailable.
            "timestamp_us": timestamp_start_us,
            "timestamp_start_us": timestamp_start_us,
            "timestamp_end_us": timestamp_end_us,
        }

        valid_mask: Optional[np.ndarray] = None

        # static ego mask, if present
        ego_mask = self.parser.mask_dict.get(camera_id)
        if ego_mask is not None:
            valid_mask = (~ego_mask).astype(bool)  # True = valid pixel
            if valid_mask.shape != (height, width):
                valid_mask = cv2.resize(
                    valid_mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

        # per-frame mask, if present
        if sensor.has_frame_generic_data(frame_idx, "mask"):
            frame_mask_raw = sensor.get_frame_generic_data(frame_idx, "mask")
            frame_mask = np.asarray(frame_mask_raw)
            if frame_mask.ndim == 3 and frame_mask.shape[-1] == 1:
                frame_mask = frame_mask[..., 0]
            # assumption: True/non-zero values in generic "mask" indicate valid pixels
            frame_mask = np.squeeze(frame_mask).astype(bool)
            if frame_mask.shape != (height, width):
                frame_mask = cv2.resize(
                    frame_mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            # merge with ego mask if present
            valid_mask = frame_mask if valid_mask is None else (valid_mask & frame_mask)

        if valid_mask is not None:
            data["mask"] = torch.from_numpy(valid_mask).bool()

        return data
