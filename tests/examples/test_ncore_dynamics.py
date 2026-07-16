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


# ---------------------------------------------------------------------------
# Native point-cloud snapshot intervals (NCoreParser._point_cloud_interval_us)
# ---------------------------------------------------------------------------


def test_native_snapshot_intervals_partition_timeline_without_overlap():
    """Adjacent native snapshots share a boundary but never overlap or gap."""
    from examples.datasets.ncore import NCoreParser

    source = SimpleNamespace(
        pc_timestamps_us=np.asarray([1000, 3000, 5000], dtype=np.int64)
    )
    loader = SimpleNamespace(lidar_ids=set())

    intervals = [
        NCoreParser._point_cloud_interval_us(loader, "camera_pc", source, idx)
        for idx in range(3)
    ]

    # First/last snapshots mirror their only neighbour's half-width (bounded),
    # interior midpoints split the gap to each adjacent snapshot.
    assert intervals == [(0, 2000), (2000, 4000), (4000, 6000)]
    # Contiguous, non-overlapping half-open bins: each bin's end is the next
    # bin's start, and the association layer treats the shared edge as
    # belonging to exactly one bin (start-exclusive, end-inclusive).
    for (_, end), (next_start, _) in zip(intervals, intervals[1:]):
        assert end == next_start


def test_native_snapshot_interval_associates_offset_cuboid():
    """Cuboid time differing from the snapshot reference still associates.

    A native source returns one reference timestamp per snapshot. Midpoint bins
    let a cuboid with a different annotation timestamp still be selected, so its
    returns are excluded from static init and owned by the rigid track.
    """
    from examples.datasets.ncore import NCoreParser

    source = SimpleNamespace(
        pc_timestamps_us=np.asarray([1000, 3000, 5000], dtype=np.int64)
    )
    loader = SimpleNamespace(lidar_ids=set())

    # Snapshot 1 has reference time 3000 and owns the (2000, 4000] bin.
    start, end = NCoreParser._point_cloud_interval_us(loader, "camera_pc", source, 1)
    assert (start, end) == (2000, 4000)

    # Cuboid annotated at 3200 (!= the 3000 snapshot reference time).
    track = _track("vehicle", "automobile", [3200], [3000], [(0.0, 0.0, 0.0)])
    points = np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)

    association = associate_spin_points(points, [track], start, end)

    assert association.rigid_owned_mask.any()
    assert association.static_exclusion_mask.any()
    np.testing.assert_array_equal(association.owner_track_indices, [0, -1])

    # An exact interval does not select an offset annotation.
    exact = associate_spin_points(points, [track], 3000, 3000)
    assert not exact.rigid_owned_mask.any()


def test_single_native_snapshot_falls_back_to_exact_interval():
    from examples.datasets.ncore import NCoreParser

    source = SimpleNamespace(pc_timestamps_us=np.asarray([4200], dtype=np.int64))
    loader = SimpleNamespace(lidar_ids=set())

    assert NCoreParser._point_cloud_interval_us(loader, "camera_pc", source, 0) == (
        4200,
        4200,
    )


# ---------------------------------------------------------------------------
# Init-point allocation policy (NCoreParser._sample_init_points)
# ---------------------------------------------------------------------------


def _identified_points(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points whose RGB is derived from a unique per-point id.

    Each point carries its id in all three coordinates and ``id % 256`` in every
    RGB channel, so any row-preserving selection keeps ``rgb == coord % 256``.
    This lets tests detect point/color misalignment after subsampling.
    """
    points = np.repeat(ids.astype(np.float32)[:, None], 3, axis=1)
    rgb = np.repeat((ids % 256).astype(np.uint8)[:, None], 3, axis=1)
    return points, rgb


def _points_are_aligned(points: np.ndarray, rgb: np.ndarray) -> bool:
    return np.array_equal((points.astype(np.int64) % 256).astype(np.uint8), rgb)


def _rigid_track(track_id: str, ids: np.ndarray):
    from examples.datasets.ncore import RigidDynamicTrack

    points_local, points_rgb = _identified_points(ids)
    return RigidDynamicTrack(
        track_id=track_id,
        class_id="automobile",
        points_local=points_local,
        points_rgb=points_rgb,
        frame_timestamps_us=np.asarray([0], dtype=np.int64),
        poses_local_to_scene=np.eye(4, dtype=np.float32)[None],
        dimensions_local=np.ones(3, dtype=np.float32),
    )


def _build_allocation_parser(seed: int):
    """A bare NCoreParser carrying only the state `_sample_init_points` touches."""
    from examples.datasets.ncore import NCoreParser

    parser = object.__new__(NCoreParser)
    parser.rng = np.random.default_rng(seed)
    # Static ids [10_000, 12_000) — disjoint from every track's id range so the
    # id->rgb alignment check is unambiguous across pools.
    static_points, static_rgb = _identified_points(np.arange(10_000, 12_000))
    parser.points = static_points
    parser.points_rgb = static_rgb
    parser.rigid_dynamic_tracks = [
        _rigid_track("track_a", np.arange(0, 200)),
        _rigid_track("track_b", np.arange(1_000, 1_200)),
        _rigid_track("track_c", np.arange(2_000, 2_200)),
        _rigid_track("track_empty", np.empty(0, dtype=np.int64)),
    ]
    return parser


def test_sample_init_points_allocation_policy_is_deterministic_and_capped():
    max_points = 1_000
    max_dynamic_points = 250
    per_track_cap = 100

    parser = _build_allocation_parser(seed=123)
    parser._sample_init_points(max_points, max_dynamic_points, per_track_cap)

    tracks = parser.rigid_dynamic_tracks
    dynamic_count = sum(len(t.points_local) for t in tracks)

    # Emptied tracks (zero source points) are dropped, not retained at size 0.
    assert [t.track_id for t in tracks] == ["track_a", "track_b", "track_c"]

    # Per-track cap: no surviving track exceeds its cap.
    assert all(len(t.points_local) <= per_track_cap for t in tracks)
    # Global dynamic cap: three 200-point tracks capped to 100 each = 300 > 250,
    # so the global cap binds exactly.
    assert dynamic_count == max_dynamic_points

    # Static fill: budget is whatever the dynamic pool left, up to max_points.
    assert len(parser.points) == max_points - max_dynamic_points
    assert len(parser.points) + dynamic_count == max_points

    # Point/color alignment survives both subsampling stages.
    assert _points_are_aligned(parser.points, parser.points_rgb)
    for track in tracks:
        assert len(track.points_local) == len(track.points_rgb)
        assert _points_are_aligned(track.points_local, track.points_rgb)

    # Same seed + same inputs => byte-identical selections.
    replay = _build_allocation_parser(seed=123)
    replay._sample_init_points(max_points, max_dynamic_points, per_track_cap)
    assert parser.points.tobytes() == replay.points.tobytes()
    assert parser.points_rgb.tobytes() == replay.points_rgb.tobytes()
    assert [t.track_id for t in replay.rigid_dynamic_tracks] == [
        t.track_id for t in tracks
    ]
    for got, expected in zip(replay.rigid_dynamic_tracks, tracks):
        assert got.points_local.tobytes() == expected.points_local.tobytes()
        assert got.points_rgb.tobytes() == expected.points_rgb.tobytes()
