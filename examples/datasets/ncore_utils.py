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

"""Coordinate-frame helpers and rigid-track classification / LiDAR-spin association."""

from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from typing import Any, Collection, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from ncore.impl.common.transformations import is_within_3d_bboxes


POINT_CHUNK_SIZE = 32_768
STATIC_EXCLUSION_PADDING_M = (0.5, 0.5, 0.25)

VRU_CLASS_IDS = frozenset(
    {
        "bicycle",
        "bicycle_with_rider",
        "cycle",
        "cyclist",
        "motorcycle",
        "motorcycle_with_rider",
        "pedestrian",
        "person",
        "person_group",
        "rider",
        "stroller",
    }
)
VEHICLE_CLASS_IDS = frozenset(
    {
        "automobile",
        "bus",
        "car",
        "construction_vehicle",
        "pickup",
        "trailer",
        "train",
        "truck",
        "van",
    }
)
DEFAULT_ACTOR_CLASS_IDS = tuple(sorted(VRU_CLASS_IDS | VEHICLE_CLASS_IDS))

MIN_VEHICLE_OBSERVATIONS = 3
MIN_VEHICLE_DURATION_S = 0.2
MIN_VEHICLE_PATH_LENGTH_M = 1.0
MIN_VEHICLE_DISPLACEMENT_M = 1.0
MIN_VRU_OBSERVATIONS = 2


class FrameConversion:
    """Converts poses and points between canonical 3D frames.

    Encodes a combined axis-permutation, origin-shift, and uniform scale as a
    single 4x4 matrix, where matrix[3,3] = 1/scale.
    """

    def __init__(self, matrix: npt.NDArray[np.float32]) -> None:
        assert matrix.shape == (4, 4)
        assert matrix.dtype == np.float32
        assert matrix[3, 3] > 0.0
        assert np.isclose(np.linalg.det(matrix[:3, :3]), 1.0)
        self.matrix = matrix

    @classmethod
    def from_origin_scale_axis(
        cls,
        target_origin: npt.NDArray[np.float32],
        target_scale: float,
        target_axis: List[int],
    ) -> "FrameConversion":
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = -target_origin
        matrix[3, 3] = 1.0 / target_scale
        assert len(np.unique(target_axis)) == 3
        matrix = matrix[target_axis + [3]]
        return cls(matrix=matrix)

    @property
    def target_scale(self) -> float:
        return 1.0 / float(self.matrix[3, 3])

    def get_transformation_matrices(
        self,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        T = self.matrix.copy()
        T *= self.target_scale
        inv_s = float(self.matrix[3, 3])
        S = np.zeros((4, 4), dtype=np.float32)
        np.fill_diagonal(S, [inv_s, inv_s, inv_s, 1.0])
        return T, S

    def transform_poses(self, T_poses_source: np.ndarray) -> np.ndarray:
        T_poses = T_poses_source.reshape((-1, 4, 4))
        T, S = self.get_transformation_matrices()
        T_poses = T @ T_poses @ S
        return T_poses.squeeze()


@dataclasses.dataclass(frozen=True)
class TrackCuboids:
    """World-space cuboid observations keyed by NCore's two timestamps.

    `annotation_timestamps_us` controls motion timing and LiDAR-spin membership.
    `reference_timestamps_us` selects the source-frame pose used to transform each
    cuboid into world coordinates.
    """

    track_id: str
    class_id: str
    annotation_timestamps_us: np.ndarray
    reference_timestamps_us: np.ndarray
    bboxes_world: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.annotation_timestamps_us)
        if self.annotation_timestamps_us.shape != (count,):
            raise ValueError("annotation_timestamps_us must have shape (N,)")
        if self.reference_timestamps_us.shape != (count,):
            raise ValueError("reference_timestamps_us must have shape (N,)")
        if self.bboxes_world.shape != (count, 9):
            raise ValueError("bboxes_world must have shape (N, 9)")


@dataclasses.dataclass(frozen=True)
class TrackMotion:
    observation_count: int
    duration_s: float
    robust_path_length_m: float
    start_to_end_displacement_m: float
    class_group: str
    is_dynamic: bool


@dataclasses.dataclass(frozen=True)
class SpinAssociation:
    """Point-aligned ownership and static-exclusion decisions for one spin."""

    owner_track_indices: np.ndarray
    owner_observation_indices: np.ndarray
    static_exclusion_mask: np.ndarray
    padded_only_mask: np.ndarray
    candidate_memberships: int
    overlap_point_count: int
    overlap_adjustments: int
    selected_observation_count: int

    @property
    def rigid_owned_mask(self) -> np.ndarray:
        return self.owner_track_indices >= 0


def normalize_class_id(class_id: Any) -> str:
    return str(class_id).strip().lower()


def _smoothed_centers(centers: np.ndarray) -> np.ndarray:
    """Suppress single-observation label jitter before measuring path length.

    Interior observations are replaced by a windowed median: the five-sample
    window is used wherever it fits, falling back to an adaptive three-sample
    median near the boundaries so short (3-4 observation) tracks are still
    smoothed. Endpoints are left untouched.
    """
    count = len(centers)
    if count < 3:
        return centers
    smoothed = centers.copy()
    for index in range(1, count - 1):
        if 2 <= index <= count - 3:
            window = centers[index - 2 : index + 3]
        else:
            window = centers[index - 1 : index + 2]
        smoothed[index] = np.median(window, axis=0)
    return smoothed


def classify_track(track: TrackCuboids) -> TrackMotion:
    centers = _smoothed_centers(track.bboxes_world[:, :3])
    observation_count = len(centers)
    duration_s = (
        float(track.annotation_timestamps_us[-1] - track.annotation_timestamps_us[0])
        / 1e6
        if observation_count > 1
        else 0.0
    )
    path_length_m = (
        float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
        if observation_count > 1
        else 0.0
    )
    displacement_m = (
        float(np.linalg.norm(centers[-1] - centers[0]))
        if observation_count > 1
        else 0.0
    )

    class_group = "vru" if track.class_id in VRU_CLASS_IDS else "vehicle"
    if class_group == "vru":
        is_dynamic = observation_count >= MIN_VRU_OBSERVATIONS and duration_s > 0
    else:
        enough_observations = (
            observation_count >= MIN_VEHICLE_OBSERVATIONS
            and duration_s >= MIN_VEHICLE_DURATION_S
        )
        has_motion = (
            path_length_m > MIN_VEHICLE_PATH_LENGTH_M
            or displacement_m > MIN_VEHICLE_DISPLACEMENT_M
        )
        is_dynamic = enough_observations and has_motion

    return TrackMotion(
        observation_count=observation_count,
        duration_s=duration_s,
        robust_path_length_m=path_length_m,
        start_to_end_displacement_m=displacement_m,
        class_group=class_group,
        is_dynamic=is_dynamic,
    )


def classify_tracks(
    tracks: Sequence[TrackCuboids],
) -> tuple[list[TrackCuboids], list[TrackCuboids], dict[str, TrackMotion]]:
    motions = {track.track_id: classify_track(track) for track in tracks}
    dynamic = [track for track in tracks if motions[track.track_id].is_dynamic]
    stationary = [track for track in tracks if not motions[track.track_id].is_dynamic]
    return dynamic, stationary, motions


def load_track_cuboids(
    loader: Any,
    class_ids: Collection[str],
    timestamp_interval_us: Any | None = None,
) -> tuple[list[TrackCuboids], Counter[str]]:
    """Load tracks while transforming poses at reference-frame timestamps."""
    normalized_classes = frozenset(normalize_class_id(value) for value in class_ids)
    interval = timestamp_interval_us or loader.sequence_timestamp_interval_us
    observations_by_track: dict[str, list[Any]] = defaultdict(list)
    available_classes: Counter[str] = Counter()
    for observation in loader.get_cuboid_track_observations(interval):
        class_id = normalize_class_id(observation.class_id)
        available_classes[class_id] += 1
        if class_id in normalized_classes:
            observations_by_track[str(observation.track_id)].append(observation)

    tracks: list[TrackCuboids] = []
    for track_id, observations in observations_by_track.items():
        track_classes = {normalize_class_id(item.class_id) for item in observations}
        if len(track_classes) != 1:
            raise ValueError(
                f"Track {track_id!r} has inconsistent class IDs: "
                f"{sorted(track_classes)}"
            )
        observations.sort(
            key=lambda item: (item.timestamp_us, item.reference_frame_timestamp_us)
        )
        bboxes_world = []
        for observation in observations:
            world_observation = observation.transform(
                "world",
                observation.reference_frame_timestamp_us,
                loader.pose_graph,
            )
            bboxes_world.append(
                np.asarray(world_observation.bbox3.to_array(), dtype=np.float64)
            )
        tracks.append(
            TrackCuboids(
                track_id=track_id,
                class_id=next(iter(track_classes)),
                annotation_timestamps_us=np.asarray(
                    [item.timestamp_us for item in observations], dtype=np.int64
                ),
                reference_timestamps_us=np.asarray(
                    [item.reference_frame_timestamp_us for item in observations],
                    dtype=np.int64,
                ),
                bboxes_world=np.stack(bboxes_world),
            )
        )
    tracks.sort(key=lambda track: track.track_id)
    return tracks, available_classes


def _annotation_spin_index(timestamp_us: int, intervals: np.ndarray) -> int:
    candidates = np.flatnonzero(
        (intervals[:, 0] <= timestamp_us) & (timestamp_us <= intervals[:, 1])
    )
    if not len(candidates):
        return -1
    return int(candidates[np.argmax(intervals[candidates, 0])])


def observations_in_spin(
    tracks: Sequence[TrackCuboids],
    start_timestamp_us: int,
    end_timestamp_us: int,
    spin_intervals_us: np.ndarray | None = None,
    spin_index: int | None = None,
) -> list[tuple[int, int]]:
    """Return deterministic ``(track, observation)`` indices for one spin."""
    if spin_intervals_us is not None:
        intervals = np.asarray(spin_intervals_us, dtype=np.int64)
        if spin_index is None:
            matches = np.flatnonzero(
                (intervals[:, 0] == start_timestamp_us)
                & (intervals[:, 1] == end_timestamp_us)
            )
            if len(matches) != 1:
                raise ValueError(
                    "Could not resolve spin_index from "
                    f"[{start_timestamp_us}, {end_timestamp_us}]"
                )
            spin_index = int(matches[0])
        assignments, _ = assign_observations_to_spins(tracks, intervals)
        return assignments[spin_index]

    selected = []
    for track_index, track in enumerate(tracks):
        timestamps = track.annotation_timestamps_us
        if start_timestamp_us > 0 and start_timestamp_us == end_timestamp_us:
            in_spin = (timestamps >= start_timestamp_us) & (
                timestamps <= end_timestamp_us
            )
        elif start_timestamp_us > 0:
            in_spin = (timestamps > start_timestamp_us) & (
                timestamps <= end_timestamp_us
            )
        else:
            in_spin = (timestamps >= start_timestamp_us) & (
                timestamps <= end_timestamp_us
            )
        for index in np.flatnonzero(in_spin):
            selected.append((track_index, int(index)))
    return sorted(
        selected,
        key=lambda item: (
            tracks[item[0]].track_id,
            int(tracks[item[0]].annotation_timestamps_us[item[1]]),
            item[1],
        ),
    )


def associate_spin_points(
    points_world: np.ndarray,
    tracks: Sequence[TrackCuboids],
    start_timestamp_us: int,
    end_timestamp_us: int,
    static_exclusion_padding_m: Sequence[float] = STATIC_EXCLUSION_PADDING_M,
    selected_observations: Sequence[tuple[int, int]] | None = None,
    spin_intervals_us: np.ndarray | None = None,
    spin_index: int | None = None,
) -> SpinAssociation:
    """Assign unpadded rigid owners and padded static exclusion for one spin."""
    points_world = np.asarray(points_world)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    padding = np.asarray(static_exclusion_padding_m, dtype=np.float64)
    if padding.shape != (3,) or np.any(padding < 0):
        raise ValueError(
            "static_exclusion_padding_m must contain three non-negative values"
        )

    selected = (
        observations_in_spin(
            tracks,
            start_timestamp_us,
            end_timestamp_us,
            spin_intervals_us=spin_intervals_us,
            spin_index=spin_index,
        )
        if selected_observations is None
        else list(selected_observations)
    )
    point_count = len(points_world)
    owner_track = np.full(point_count, -1, dtype=np.int64)
    owner_observation = np.full(point_count, -1, dtype=np.int64)
    static_exclusion = np.zeros(point_count, dtype=bool)
    membership_counts = np.zeros(point_count, dtype=np.int32)

    if selected and point_count:
        bboxes = np.stack(
            [
                tracks[track_index].bboxes_world[observation_index]
                for track_index, observation_index in selected
            ]
        )
        padded_bboxes = bboxes.copy()
        padded_bboxes[:, 3:6] += padding

        for start in range(0, point_count, POINT_CHUNK_SIZE):
            stop = min(start + POINT_CHUNK_SIZE, point_count)
            inside = is_within_3d_bboxes(points_world[start:stop], bboxes)
            inside_padded = is_within_3d_bboxes(points_world[start:stop], padded_bboxes)
            membership_counts[start:stop] = inside.sum(axis=1, dtype=np.int32)
            static_exclusion[start:stop] = inside_padded.any(axis=1)

            has_owner = inside.any(axis=1)
            if np.any(has_owner):
                first_match = inside.argmax(axis=1)
                point_indices = np.flatnonzero(has_owner) + start
                selected_array = np.asarray(selected, dtype=np.int64)
                owners = selected_array[first_match[has_owner]]
                owner_track[point_indices] = owners[:, 0]
                owner_observation[point_indices] = owners[:, 1]

    rigid_owned = owner_track >= 0
    padded_only = static_exclusion & ~rigid_owned
    overlap = membership_counts > 1
    overlap_adjustments = int(np.maximum(membership_counts - 1, 0).sum())
    return SpinAssociation(
        owner_track_indices=owner_track,
        owner_observation_indices=owner_observation,
        static_exclusion_mask=static_exclusion,
        padded_only_mask=padded_only,
        candidate_memberships=int(membership_counts.sum()),
        overlap_point_count=int(overlap.sum()),
        overlap_adjustments=overlap_adjustments,
        selected_observation_count=len(selected),
    )


def assign_observations_to_spins(
    tracks: Sequence[TrackCuboids], spin_intervals_us: np.ndarray
) -> tuple[list[list[tuple[int, int]]], dict[str, int]]:
    """Assign each in-range annotation to one spin, resolving shared boundaries forward."""
    intervals = np.asarray(spin_intervals_us, dtype=np.int64)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("spin_intervals_us must have shape (N, 2)")
    assignments: list[list[tuple[int, int]]] = [[] for _ in range(len(intervals))]
    selected = 0
    unmatched = 0
    boundary_ties = 0
    for track_index, track in enumerate(tracks):
        for observation_index, timestamp_us in enumerate(
            track.annotation_timestamps_us
        ):
            candidates = np.flatnonzero(
                (intervals[:, 0] <= timestamp_us) & (timestamp_us <= intervals[:, 1])
            )
            selected_spin = _annotation_spin_index(int(timestamp_us), intervals)
            if selected_spin < 0:
                unmatched += 1
                continue
            boundary_ties += len(candidates) > 1
            assignments[selected_spin].append((track_index, observation_index))
            selected += 1
    for spin_index, spin_assignments in enumerate(assignments):
        assignments[spin_index] = sorted(
            spin_assignments,
            key=lambda item: (
                tracks[item[0]].track_id,
                int(tracks[item[0]].annotation_timestamps_us[item[1]]),
                item[1],
            ),
        )
    return assignments, {
        "selected_annotations": selected,
        "unmatched_annotations": unmatched,
        "shared_boundary_annotations": boundary_ties,
    }
