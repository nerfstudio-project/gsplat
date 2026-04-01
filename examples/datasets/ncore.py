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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.utils.data
from scipy import ndimage

import ncore.data
import ncore.data.v4
import ncore.sensors

from gsplat.rendering import FThetaCameraDistortionParameters, FThetaPolynomialType

from .ncore_utils import FrameConversion
from .normalize import (
    similarity_from_cameras,
    align_principal_axes,
    transform_cameras,
    transform_points,
)


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
    ) -> None:
        self.test_every = test_every
        self.factor = factor
        self.normalize_world_space = normalize_world_space
        self.poses_component_group = poses_component_group
        self.intrinsics_component_group = intrinsics_component_group
        self.masks_component_group = masks_component_group
        self.open_consolidated = open_consolidated
        self.lidar_color_generic_data_name = lidar_color_generic_data_name

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

        self.points, self.points_rgb = self._load_lidar_points(
            sequence_loader, max_lidar_points, lidar_step_frame
        )

        # Normalize the world space (orient, centre, and rescale).
        if self.normalize_world_space:
            self._normalize_world_space()

        # Scene scale: max distance of each camera from the mean camera position.
        # This matches the COLMAP convention (colmap.py:396-400).
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = float(np.max(dists))

        print(
            f"[NCoreParser] Loaded sequence '{self.sequence_id}': "
            f"{len(self.frame_list)} frames across {self.num_cameras} cameras, "
            f"{len(self.points)} lidar init points, scene_scale={self.scene_scale:.3f}"
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

            print(f"[NCoreParser] Auto-detected cameras: {camera_ids}")
        if not lidar_ids:
            lidar_ids = sequence_loader.lidar_ids

            if len(lidar_ids) > 1:
                raise ValueError(
                    "NCoreParser: Multiple lidar sensors in dataset, explicit"
                    f" specification of a (subset) of lidar sensors required to avoid ambiguity: {lidar_ids}"
                )

            print(f"[NCoreParser] Auto-detected lidars: {lidar_ids}")

        assert all(
            cid in sequence_loader.camera_ids for cid in camera_ids
        ), f"NCoreParser: some specified camera_ids {camera_ids} not found in dataset cameras {sequence_loader.camera_ids}"
        assert all(
            lid in sequence_loader.lidar_ids for lid in lidar_ids
        ), f"NCoreParser: some specified lidar_ids {lidar_ids} not found in dataset lidars {sequence_loader.lidar_ids}"

        self.camera_ids: List[str] = list(camera_ids)
        self.lidar_ids: List[str] = list(lidar_ids)
        self.num_cameras: int = len(self.camera_ids)

        print(f"[NCoreParser] Using cameras: {self.camera_ids}")
        print(f"[NCoreParser] Using lidars: {self.lidar_ids}")

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
                    print(
                        f"[NCoreParser] Error: factor={factor} produces non-integer "
                        f"resolution for {camera_id}; using factor=1.0 (full resolution). "
                        "Pass --data-factor 1 to suppress this error."
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
                print(f"[NCoreParser] {camera_id}: {width}x{height} (ftheta)")
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
                print(f"[NCoreParser] {camera_id}: {width}x{height} (opencv_fisheye)")
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
                print(f"[NCoreParser] {camera_id}: {width}x{height} (opencv_pinhole)")
            else:
                # Unknown camera type: synthesize K from resolution, treat as perfect pinhole.
                print(
                    f"[NCoreParser] {camera_id}: {width}x{height} (unknown, synthesizing K from resolution)"
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

            for local_idx, frame_idx in enumerate(frame_range):
                self.frame_list.append((camera_id, frame_idx))
                self.camera_idx_per_frame.append(cam_idx)
                starts.append(T_start[local_idx])
                ends.append(T_end[local_idx])

        self.camtoworlds = np.stack(starts, axis=0)  # (N, 4, 4)
        self.camtoworlds_end = np.stack(ends, axis=0)  # (N, 4, 4)

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

    # ------------------------------------------------------------------
    # Private runtime helpers
    # ------------------------------------------------------------------

    def _ncore_world_to_scene_poses(self, T_poses_world: np.ndarray) -> np.ndarray:
        """Transform poses from NCore world frame to scene frame."""
        T_poses_common = self.T_world_to_scene_world @ T_poses_world.reshape(-1, 4, 4)
        return self.world_global_to_scene.transform_poses(T_poses_common)

    def _load_lidar_points(
        self,
        sequence_loader: ncore.data.SequenceLoaderProtocol,
        max_points: int,
        step_frame: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and transform lidar points to scene frame for Gaussian initialisation."""
        if not self.lidar_ids:
            print(
                "[NCoreParser] No lidar sensors available; using empty init point cloud"
            )
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        lidar_id = self.lidar_ids[0]
        lidar_sensor = sequence_loader.get_lidar_sensor(lidar_id)
        lidar_frame_range = self.time_range_us.cover_range(
            lidar_sensor.get_frames_timestamps_us()
        )

        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        for lidar_frame_idx in lidar_frame_range[::step_frame]:
            try:
                pc = lidar_sensor.get_frame_point_cloud(
                    frame_index=lidar_frame_idx,
                    motion_compensation=True,
                    with_start_points=True,
                    return_index=0,
                )
            except Exception as exc:
                print(
                    f"[NCoreParser] Warning: failed to load lidar frame "
                    f"{lidar_frame_idx}: {exc}"
                )
                continue

            xyz = pc.xyz_m_end
            color: Optional[np.ndarray] = None
            if lidar_sensor.has_frame_generic_data(
                lidar_frame_idx, self.lidar_color_generic_data_name
            ):
                color = lidar_sensor.get_frame_generic_data(
                    lidar_frame_idx, self.lidar_color_generic_data_name
                )
                if color.shape != xyz.shape:
                    raise ValueError(
                        "Color data length does not match point cloud length "
                        "(expecting 3-channel RGB color per point)"
                    )
                if color.dtype != np.uint8:
                    raise ValueError("Expected color data in uint8 format")

            point_filter = ...
            if lidar_sensor.has_frame_generic_data(lidar_frame_idx, "dynamic_flag"):
                point_filter = (
                    lidar_sensor.get_frame_generic_data(lidar_frame_idx, "dynamic_flag")
                    != 1
                )
            xyz = xyz[point_filter]
            if color is not None:
                color = color[point_filter]
            if not len(xyz):
                continue

            T_sensor_scene = self._ncore_world_to_scene_poses(
                lidar_sensor.get_frames_T_sensor_target("world", lidar_frame_idx)
            )
            xyz_scene = (
                (self.world_global_to_scene.target_scale * T_sensor_scene[:3, :3])
                @ xyz.T
                + T_sensor_scene[:3, 3:4]
            ).T
            all_points.append(xyz_scene.astype(np.float32))
            if color is not None:
                all_colors.append(color)
            else:
                all_colors.append(np.full((len(xyz_scene), 3), 128, dtype=np.uint8))

        if not all_points:
            print("[NCoreParser] Warning: no lidar points loaded")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        points = np.vstack(all_points)
        points_rgb = np.vstack(all_colors)
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]
            points_rgb = points_rgb[idx]

        print(f"[NCoreParser] Loaded {len(points)} lidar points from '{lidar_id}'")
        return points, points_rgb


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

        data: Dict[str, Any] = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(self.parser.camtoworlds[index]).float(),
            "camtoworld_end": torch.from_numpy(
                self.parser.camtoworlds_end[index]
            ).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
            "camera_idx": camera_idx,
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
