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
"""Materialize NCore static-exclusion flags using LiDAR-spin semantics."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import resource
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Collection

import numpy as np

import ncore.data.v4

if __package__:
    from .datasets.ncore_utils import (
        DEFAULT_ACTOR_CLASS_IDS,
        STATIC_EXCLUSION_PADDING_M,
        TrackCuboids,
        assign_observations_to_spins,
        associate_spin_points,
        classify_tracks,
        load_track_cuboids,
        normalize_class_id,
    )
else:
    from datasets.ncore_utils import (
        DEFAULT_ACTOR_CLASS_IDS,
        STATIC_EXCLUSION_PADDING_M,
        TrackCuboids,
        assign_observations_to_spins,
        associate_spin_points,
        classify_tracks,
        load_track_cuboids,
        normalize_class_id,
    )


COVERAGE_REPORT_NAME = "ncore_rigid_initialization_report.json"
_NCORE_V4_REQUIRED_MANIFEST_KEYS = (
    "sequence_id",
    "sequence_timestamp_interval_us",
    "version",
    "component_stores",
)
LOGGER = logging.getLogger(__name__)


def _validate_ncore_v4_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Reject non-NCore JSON before constructing opaque ncore readers."""
    missing = [key for key in _NCORE_V4_REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ValueError(
            f"{path} is not a valid NCore v4 manifest; missing required key(s): "
            f"{', '.join(missing)}"
        )


def _copy_component_generic_data(source_reader: Any, destination_writer: Any) -> None:
    """Copy component-level arrays when the NCore SDK exposes the full API."""
    get_names = getattr(source_reader, "get_generic_data_names", None)
    get_data = getattr(source_reader, "get_generic_data", None)
    set_data = getattr(destination_writer, "set_generic_data", None)
    supported = tuple(callable(method) for method in (get_names, get_data, set_data))
    if not any(supported):
        return
    if not all(supported):
        raise RuntimeError(
            "NCore SDK exposes only part of the component-level generic-data API"
        )

    component_generic_data = {name: get_data(name) for name in get_names()}
    if component_generic_data:
        set_data(component_generic_data)


def _find_lidar_store(manifest: dict[str, Any], lidar_id: str) -> dict[str, Any]:
    matches = [
        store
        for store in manifest["component_stores"]
        if lidar_id in store.get("components", {}).get("lidars", {})
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one component store for LiDAR {lidar_id!r}, "
            f"found {len(matches)}"
        )
    lidar_store = matches[0]
    if set(lidar_store["components"]) != {"lidars"} or set(
        lidar_store["components"]["lidars"]
    ) != {lidar_id}:
        raise ValueError(
            "The selected LiDAR must have its own component store; mixed or "
            "multi-LiDAR stores are not supported"
        )
    return lidar_store


def _link_unchanged_store(source: Path, destination: Path) -> None:
    """Materialize an unchanged store without a full byte copy when possible."""
    if destination.exists():
        raise FileExistsError(f"Component store already exists: {destination}")
    try:
        os.link(source, destination)
        return
    except OSError:
        pass
    try:
        os.symlink(source, destination)
        return
    except OSError:
        shutil.copy2(source, destination)


def _copy_unchanged_stores(
    input_manifest_path: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    replaced_store: dict[str, Any],
) -> list[Path]:
    sources_and_destinations = []
    for store in manifest["component_stores"]:
        if store is replaced_store:
            continue
        source = Path(store["path"])
        if not source.is_absolute():
            source = input_manifest_path.parent / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Component store does not exist: {source}")
        destination = output_dir / source.name
        sources_and_destinations.append((source, destination))

    destination_names = [
        destination.name for _, destination in sources_and_destinations
    ]
    duplicate_names = sorted(
        name for name, count in Counter(destination_names).items() if count > 1
    )
    if duplicate_names:
        raise ValueError(
            "Component stores must have unique basenames: "
            f"{', '.join(duplicate_names)}"
        )

    linked_paths = []
    for source, destination in sources_and_destinations:
        _link_unchanged_store(source, destination)
        linked_paths.append(destination)
    return linked_paths


def _track_motion_report(
    tracks: Collection[TrackCuboids], motions: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "track_id": track.track_id,
            "class_id": track.class_id,
            **dataclasses.asdict(motions[track.track_id]),
        }
        for track in tracks
    ]


def _copy_lidar_with_dynamic_flags(
    sequence_reader: Any,
    loader: Any,
    lidar_id: str,
    output_dir: Path,
    output_base_name: str,
    component_meta_data: dict[str, Any],
    dynamic_tracks: list[TrackCuboids],
    spin_assignments: list[list[tuple[int, int]]],
    derivation_meta_data: dict[str, Any],
) -> tuple[list[Path], dict[str, Any]]:
    lidar = loader.get_lidar_sensor(lidar_id)
    point_source = loader.get_point_clouds_source(lidar_id)
    writer = ncore.data.v4.SequenceComponentGroupsWriter.from_reader(
        output_dir,
        output_base_name,
        sequence_reader,
    )
    lidar_writer = writer.register_component_writer(
        ncore.data.v4.LidarSensorComponent.Writer,
        lidar_id,
        group_name=f"{lidar_id}_dynamic_flags",
        generic_meta_data={
            **component_meta_data,
            "dynamic_flag_derivation": derivation_meta_data,
        },
    )

    source_lidar_reader = sequence_reader.open_component_readers(
        ncore.data.v4.LidarSensorComponent.Reader
    )[lidar_id]
    _copy_component_generic_data(source_lidar_reader, lidar_writer)

    counts: Counter[str] = Counter()
    per_track_points: Counter[str] = Counter()
    per_class_points: Counter[str] = Counter()
    for frame_index in range(lidar.frames_count):
        point_cloud = point_source.get_pc(frame_index)
        points_world = point_cloud.transform(
            "world",
            point_cloud.reference_frame_timestamp_us,
            loader.pose_graph,
        ).xyz
        start_us, end_us = (
            int(value) for value in lidar.frames_timestamps_us[frame_index]
        )
        association = associate_spin_points(
            points_world,
            dynamic_tracks,
            start_us,
            end_us,
            STATIC_EXCLUSION_PADDING_M,
            selected_observations=spin_assignments[frame_index],
        )
        dynamic_flag = association.static_exclusion_mask.astype(np.uint8)
        rigid_count = int(association.rigid_owned_mask.sum())
        excluded_count = int(dynamic_flag.sum())

        counts["frames"] += 1
        counts["raw_points"] += len(points_world)
        counts["frames_with_annotations"] += association.selected_observation_count > 0
        counts["frames_with_rigid_points"] += rigid_count > 0
        counts["selected_observations"] += association.selected_observation_count
        counts["candidate_memberships"] += association.candidate_memberships
        counts["rigid_owned_points"] += rigid_count
        counts["static_excluded_points"] += excluded_count
        counts["padded_only_points"] += int(association.padded_only_mask.sum())
        counts["overlap_points"] += association.overlap_point_count
        counts["overlap_adjustments"] += association.overlap_adjustments
        owners = association.owner_track_indices
        track_counts = np.bincount(owners[owners >= 0], minlength=len(dynamic_tracks))
        for track_index, track in enumerate(dynamic_tracks):
            point_count = int(track_counts[track_index])
            per_track_points[track.track_id] += point_count
            per_class_points[track.class_id] += point_count

        generic_data = {
            name: lidar.get_frame_generic_data(frame_index, name)
            for name in lidar.get_frame_generic_data_names(frame_index)
            if name != "dynamic_flag"
        }
        generic_data["dynamic_flag"] = dynamic_flag
        return_count = lidar.get_frame_ray_bundle_return_count(frame_index)
        distances = np.stack(
            [
                lidar.get_frame_ray_bundle_return_distance_m(frame_index, return_index)
                for return_index in range(return_count)
            ]
        ).astype(np.float32, copy=False)
        intensities = np.stack(
            [
                lidar.get_frame_ray_bundle_return_intensity(frame_index, return_index)
                for return_index in range(return_count)
            ]
        ).astype(np.float32, copy=False)
        lidar_writer.store_frame(
            direction=lidar.get_frame_ray_bundle_direction(frame_index),
            timestamp_us=lidar.get_frame_ray_bundle_timestamp_us(frame_index),
            model_element=lidar.get_frame_ray_bundle_model_element(frame_index),
            distance_m=distances,
            intensity=intensities,
            frame_timestamps_us=lidar.frames_timestamps_us[frame_index],
            generic_data=generic_data,
            generic_meta_data=lidar.get_frame_generic_meta_data(frame_index),
        )
        print(
            f"[{frame_index + 1}/{lidar.frames_count}] "
            f"{rigid_count} rigid-owned, {excluded_count} static-excluded"
        )

    counts["static_kept_points"] = (
        counts["raw_points"] - counts["static_excluded_points"]
    )
    if counts["raw_points"] != (
        counts["static_kept_points"]
        + counts["rigid_owned_points"]
        + counts["padded_only_points"]
    ):
        raise RuntimeError("Static/rigid/padded-only point conservation failed")
    if counts["candidate_memberships"] != (
        counts["rigid_owned_points"] + counts["overlap_adjustments"]
    ):
        raise RuntimeError("Rigid ownership/overlap point conservation failed")

    return [Path(path) for path in writer.finalize()], {
        **{key: int(value) for key, value in counts.items()},
        "per_track_rigid_points": dict(sorted(per_track_points.items())),
        "per_class_rigid_points": dict(sorted(per_class_points.items())),
    }


def _validate_output(
    output_manifest_path: Path, lidar_id: str, expected_flagged_points: int
) -> int:
    loader = ncore.data.v4.SequenceLoaderV4(
        ncore.data.v4.SequenceComponentGroupsReader(
            [output_manifest_path], open_consolidated=True
        )
    )
    lidar = loader.get_lidar_sensor(lidar_id)
    source = loader.get_point_clouds_source(lidar_id)
    flagged_points = 0
    for frame_index in range(source.pcs_count):
        if not source.has_pc_generic_data(frame_index, "dynamic_flag"):
            raise RuntimeError(
                f"Output frame {frame_index} does not contain dynamic_flag"
            )
        dynamic_flag = source.get_pc_generic_data(frame_index, "dynamic_flag")
        expected_shape = (lidar.get_frame_ray_bundle_count(frame_index),)
        if dynamic_flag.shape != expected_shape:
            raise RuntimeError(
                f"Output frame {frame_index} dynamic_flag shape "
                f"{dynamic_flag.shape} does not match {expected_shape}"
            )
        if not np.all((dynamic_flag == 0) | (dynamic_flag == 1)):
            raise RuntimeError(
                f"Output frame {frame_index} dynamic_flag contains values "
                "outside {0, 1}"
            )
        flagged_points += int(dynamic_flag.sum())
    if flagged_points != expected_flagged_points:
        raise RuntimeError(
            f"Output contains {flagged_points} flagged points, expected "
            f"{expected_flagged_points}"
        )
    return source.pcs_count


def derive_dynamic_flags(
    input_manifest_path: Path,
    output_dir: Path,
    class_ids: Collection[str] = DEFAULT_ACTOR_CLASS_IDS,
    lidar_id: str | None = None,
) -> Path:
    start_time = time.perf_counter()
    input_manifest_path = input_manifest_path.resolve()
    output_dir = output_dir.resolve()
    normalized_classes = frozenset(normalize_class_id(value) for value in class_ids)
    if not input_manifest_path.is_file():
        raise FileNotFoundError(f"Input manifest does not exist: {input_manifest_path}")
    if not normalized_classes:
        raise ValueError("At least one class ID is required")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    with input_manifest_path.open() as manifest_file:
        manifest = json.load(manifest_file)
    _validate_ncore_v4_manifest(manifest, input_manifest_path)
    sequence_reader = ncore.data.v4.SequenceComponentGroupsReader(
        [input_manifest_path], open_consolidated=True
    )
    loader = ncore.data.v4.SequenceLoaderV4(sequence_reader)
    if lidar_id is None:
        if len(loader.lidar_ids) != 1:
            raise ValueError(
                "The input must contain exactly one LiDAR unless --lidar-id is "
                f"provided; available LiDAR IDs: {loader.lidar_ids}"
            )
        lidar_id = loader.lidar_ids[0]
    if lidar_id not in loader.lidar_ids:
        raise ValueError(
            f"LiDAR {lidar_id!r} not found; available LiDAR IDs: {loader.lidar_ids}"
        )

    source = loader.get_point_clouds_source(lidar_id)
    existing_flag_frames = [
        frame_index
        for frame_index in range(source.pcs_count)
        if source.has_pc_generic_data(frame_index, "dynamic_flag")
    ]
    if existing_flag_frames:
        raise ValueError(
            f"LiDAR {lidar_id!r} already contains dynamic_flag on "
            f"{len(existing_flag_frames)}/{source.pcs_count} frames"
        )

    all_tracks, available_classes = load_track_cuboids(loader, normalized_classes)
    dynamic_tracks, stationary_tracks, motions = classify_tracks(all_tracks)
    if not dynamic_tracks:
        LOGGER.warning(
            "No requested cuboid tracks were classified as dynamic; "
            "writing all-zero dynamic_flag arrays"
        )
    missing_classes = normalized_classes - set(available_classes)
    if missing_classes:
        print(f"Class IDs not present and skipped: {sorted(missing_classes)}")

    lidar = loader.get_lidar_sensor(lidar_id)
    spin_assignments, membership = assign_observations_to_spins(
        dynamic_tracks, lidar.frames_timestamps_us
    )
    spin_frame_count = len(lidar.frames_timestamps_us)
    if source.pcs_count != spin_frame_count:
        raise ValueError(
            f"LiDAR point-cloud frame count ({source.pcs_count}) does not match "
            f"spin frame count ({spin_frame_count})"
        )
    if membership["unmatched_annotations"] > 0:
        LOGGER.warning(
            "%d cuboid annotation(s) did not fall inside any LiDAR spin interval",
            membership["unmatched_annotations"],
        )
    lidar_store = _find_lidar_store(manifest, lidar_id)
    derivation_meta_data = {
        "method": "annotation_timestamp_lidar_spin_interval",
        "source_manifest": input_manifest_path.name,
        "class_ids": sorted(normalized_classes),
        "static_exclusion_padding_m": list(STATIC_EXCLUSION_PADDING_M),
        "dynamic_track_count": len(dynamic_tracks),
        "stationary_track_count": len(stationary_tracks),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.parent / f".{output_dir.name}.tmp"
    if temporary_dir.exists():
        raise FileExistsError(
            f"Temporary output directory already exists: {temporary_dir}"
        )

    try:
        temporary_dir.mkdir()
        unchanged_paths = _copy_unchanged_stores(
            input_manifest_path, temporary_dir, manifest, lidar_store
        )
        component_meta_data = lidar_store["components"]["lidars"][lidar_id].get(
            "generic_meta_data", {}
        )
        lidar_paths, coverage = _copy_lidar_with_dynamic_flags(
            sequence_reader,
            loader,
            lidar_id,
            temporary_dir,
            input_manifest_path.stem,
            component_meta_data,
            dynamic_tracks,
            spin_assignments,
            derivation_meta_data,
        )
        if not coverage["static_excluded_points"]:
            LOGGER.warning(
                "No LiDAR points fall inside padded dynamic cuboids; "
                "writing all-zero dynamic_flag arrays"
            )

        output_reader = ncore.data.v4.SequenceComponentGroupsReader(
            unchanged_paths + lidar_paths,
            open_consolidated=True,
        )
        output_manifest_path = temporary_dir / input_manifest_path.name
        with output_manifest_path.open("w") as output_manifest:
            json.dump(
                output_reader.get_sequence_meta().to_dict(), output_manifest, indent=2
            )
            output_manifest.write("\n")
        output_frame_count = _validate_output(
            output_manifest_path,
            lidar_id,
            coverage["static_excluded_points"],
        )

        report = {
            "input_manifest": str(input_manifest_path),
            "output_manifest": str(output_dir / input_manifest_path.name),
            "association": "spin",
            "spin_membership": membership,
            "dynamic_tracks": _track_motion_report(dynamic_tracks, motions),
            "stationary_tracks": _track_motion_report(stationary_tracks, motions),
            "counts": coverage,
            "wall_time_s": time.perf_counter() - start_time,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }
        with (temporary_dir / COVERAGE_REPORT_NAME).open("w") as report_file:
            json.dump(report, report_file, indent=2)
            report_file.write("\n")
        temporary_dir.rename(output_dir)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise

    output_manifest_path = output_dir / input_manifest_path.name
    print(f"Output manifest: {output_manifest_path}")
    print(
        f"Excluded {coverage['static_excluded_points']} static points across "
        f"{output_frame_count} LiDAR frames; {coverage['rigid_owned_points']} "
        "points have rigid owners"
    )
    return output_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive point-aligned NCore dynamic_flag arrays using annotation-time "
            "LiDAR-spin association"
        )
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="NCore v4 JSON manifest"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory for the derived manifest and component stores",
    )
    parser.add_argument(
        "--class-ids",
        default=",".join(DEFAULT_ACTOR_CLASS_IDS),
        help="comma-separated cuboid class IDs",
    )
    parser.add_argument(
        "--lidar-id",
        default=None,
        help="LiDAR sensor ID; required only when the manifest has multiple LiDARs",
    )
    args = parser.parse_args()
    class_ids = [value.strip() for value in args.class_ids.split(",") if value.strip()]
    derive_dynamic_flags(
        input_manifest_path=args.input,
        output_dir=args.output_dir,
        class_ids=class_ids,
        lidar_id=args.lidar_id,
    )


if __name__ == "__main__":
    main()
