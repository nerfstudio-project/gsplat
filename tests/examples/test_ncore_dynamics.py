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

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ncore")

from examples.datasets.ncore_utils import (  # noqa: E402
    STATIC_EXCLUSION_PADDING_M,
    TrackCuboids,
    assign_observations_to_spins,
    associate_spin_points,
    classify_track,
    load_track_cuboids,
)


def _track(
    track_id: str,
    class_id: str,
    annotations_us: list[int],
    references_us: list[int],
    centers: list[tuple[float, float, float]],
) -> TrackCuboids:
    bboxes = np.zeros((len(centers), 9), dtype=np.float64)
    bboxes[:, :3] = centers
    bboxes[:, 3:6] = 2.0
    return TrackCuboids(
        track_id=track_id,
        class_id=class_id,
        annotation_timestamps_us=np.asarray(annotations_us, dtype=np.int64),
        reference_timestamps_us=np.asarray(references_us, dtype=np.int64),
        bboxes_world=bboxes,
    )


def test_classification_filters_parked_vehicle_and_keeps_paused_vehicle_and_vru():
    parked = _track(
        "parked",
        "automobile",
        [0, 100_000, 200_000, 300_000, 400_000],
        [0, 100_000, 200_000, 300_000, 400_000],
        [(0.0, 0.0, 0.0), (0.04, 0.0, 0.0), (0.0, 0.0, 0.0)]
        + [(0.03, 0.0, 0.0), (0.02, 0.0, 0.0)],
    )
    paused = _track(
        "paused",
        "trailer",
        [0, 100_000, 200_000, 300_000],
        [0, 100_000, 200_000, 300_000],
        [(0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (1.2, 0.0, 0.0), (1.2, 0.0, 0.0)],
    )
    vru = _track(
        "vru",
        "person",
        [0, 100_000],
        [0, 100_000],
        [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0)],
    )

    assert not classify_track(parked).is_dynamic
    assert classify_track(paused).is_dynamic
    assert classify_track(vru).is_dynamic


def test_short_track_middle_outlier_is_smoothed_out():
    outlier = _track(
        "outlier",
        "automobile",
        [0, 100_000, 200_000, 300_000],
        [0, 100_000, 200_000, 300_000],
        [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    )
    motion = classify_track(outlier)
    assert motion.robust_path_length_m == pytest.approx(0.0)
    assert motion.start_to_end_displacement_m == pytest.approx(0.0)
    assert not motion.is_dynamic


def test_short_out_and_back_track_is_stationary():
    out_and_back = _track(
        "out_and_back",
        "automobile",
        [0, 100_000, 200_000],
        [0, 100_000, 200_000],
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    )
    motion = classify_track(out_and_back)
    assert motion.robust_path_length_m == pytest.approx(0.0)
    assert motion.start_to_end_displacement_m == pytest.approx(0.0)
    assert not motion.is_dynamic


def test_short_monotonic_mover_path_is_preserved():
    mover = _track(
        "mover",
        "automobile",
        [0, 100_000, 200_000, 300_000],
        [0, 100_000, 200_000, 300_000],
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
    )
    motion = classify_track(mover)
    assert motion.robust_path_length_m == pytest.approx(3.0)
    assert motion.start_to_end_displacement_m == pytest.approx(3.0)
    assert motion.is_dynamic


def test_annotation_timestamp_selects_spin_while_reference_timestamp_sets_pose():
    track = _track(
        "vehicle",
        "automobile",
        [150],
        [80],
        [(0.0, 0.0, 0.0)],
    )
    points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)

    first_spin = associate_spin_points(points, [track], 0, 99)
    second_spin = associate_spin_points(points, [track], 100, 199)

    assert not first_spin.rigid_owned_mask.any()
    np.testing.assert_array_equal(second_spin.owner_track_indices, [0])
    np.testing.assert_array_equal(second_spin.owner_observation_indices, [0])
    assert track.reference_timestamps_us[0] == 80


def test_load_track_cuboids_transforms_at_reference_timestamp():
    transform_timestamps = []

    class Observation:
        track_id = "vehicle"
        class_id = "automobile"
        timestamp_us = 150
        reference_frame_timestamp_us = 80
        bbox3 = SimpleNamespace(to_array=lambda: np.zeros(9, dtype=np.float64))

        def transform(self, frame_id, timestamp_us, pose_graph):
            assert frame_id == "world"
            assert pose_graph == "pose-graph"
            transform_timestamps.append(timestamp_us)
            return self

    loader = SimpleNamespace(
        pose_graph="pose-graph",
        get_cuboid_track_observations=lambda _interval: [Observation()],
    )

    tracks, _ = load_track_cuboids(loader, ["automobile"], object())

    assert transform_timestamps == [80]
    np.testing.assert_array_equal(tracks[0].annotation_timestamps_us, [150])
    np.testing.assert_array_equal(tracks[0].reference_timestamps_us, [80])


def test_unpadded_ownership_and_padded_static_exclusion_are_separate():
    track = _track("vehicle", "automobile", [50], [40], [(0.0, 0.0, 0.0)])
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    result = associate_spin_points(points, [track], 0, 100, STATIC_EXCLUSION_PADDING_M)

    np.testing.assert_array_equal(result.rigid_owned_mask, [True, False, False])
    np.testing.assert_array_equal(result.static_exclusion_mask, [True, True, False])
    np.testing.assert_array_equal(result.padded_only_mask, [False, True, False])
    assert len(points) == (
        int((~result.static_exclusion_mask).sum())
        + int(result.rigid_owned_mask.sum())
        + int(result.padded_only_mask.sum())
    )


def test_overlapping_cuboids_use_track_id_tie_break_and_conserve_memberships():
    track_z = _track("z-track", "automobile", [50], [50], [(0.0, 0.0, 0.0)])
    track_a = _track("a-track", "automobile", [50], [50], [(0.0, 0.0, 0.0)])
    result = associate_spin_points(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        [track_z, track_a],
        0,
        100,
    )

    np.testing.assert_array_equal(result.owner_track_indices, [1])
    assert result.overlap_point_count == 1
    assert result.candidate_memberships == 2
    assert result.candidate_memberships == (
        int(result.rigid_owned_mask.sum()) + result.overlap_adjustments
    )


def test_shared_boundary_annotation_is_assigned_to_later_spin_once():
    track = _track("vehicle", "automobile", [100], [80], [(0.0, 0.0, 0.0)])
    intervals = np.asarray([[0, 100], [100, 200]], dtype=np.int64)
    assignments, diagnostics = assign_observations_to_spins([track], intervals)

    assert assignments == [[], [(0, 0)]]
    assert diagnostics == {
        "selected_annotations": 1,
        "unmatched_annotations": 0,
        "shared_boundary_annotations": 1,
    }


def test_zero_width_spin_interval_matches_annotation_at_timestamp():
    track = _track("vehicle", "automobile", [1000], [1000], [(0.0, 0.0, 0.0)])
    points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)

    native = associate_spin_points(points, [track], 1000, 1000)
    assert native.rigid_owned_mask.any()
    assert native.static_exclusion_mask.any()
    np.testing.assert_array_equal(native.owner_track_indices, [0])

    lidar = associate_spin_points(points, [track], 500, 1000)
    assert lidar.rigid_owned_mask.any()
    assert lidar.static_exclusion_mask.any()
    np.testing.assert_array_equal(lidar.owner_track_indices, [0])

