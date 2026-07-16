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

import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("ncore")

import ncore.data.v4 as v4  # noqa: E402
from ncore.impl.common.transformations import HalfClosedInterval  # noqa: E402
from ncore.impl.data.types import (  # noqa: E402
    BBox3,
    CuboidTrackObservation,
    LabelSource,
    RowOffsetStructuredSpinningLidarModelParameters,
)

from examples.prepare_ncore_dynamics import (  # noqa: E402
    COVERAGE_REPORT_NAME,
    _copy_component_generic_data,
    _copy_unchanged_stores,
    derive_dynamic_flags,
)

NCORE_TEST_SCENE_ENV = "GSPLAT_NCORE_TEST_SCENE"


def _write_manifest(root: Path, component_stores: list[dict[str, object]]) -> Path:
    manifest = root / "test_scene.json"
    payload = {
        "sequence_id": "test_scene",
        "sequence_timestamp_interval_us": {"start": 0, "stop": 300_000},
        "generic_meta_data": {},
        "version": "v4",
        "component_stores": component_stores,
    }
    with manifest.open("w") as manifest_file:
        json.dump(payload, manifest_file, indent=2)
        manifest_file.write("\n")
    return manifest


def _lidar_model_parameters() -> RowOffsetStructuredSpinningLidarModelParameters:
    return RowOffsetStructuredSpinningLidarModelParameters(
        spinning_frequency_hz=10.0,
        spinning_direction="cw",
        n_rows=1,
        n_columns=1,
        row_elevations_rad=np.linspace(-0.1, 0.1, 1, dtype=np.float32),
        column_azimuths_rad=np.zeros(1, dtype=np.float32),
        row_azimuth_offsets_rad=np.zeros(1, dtype=np.float32),
    )


def _write_synthetic_ncore_manifest(
    root: Path,
    *,
    cuboid_timestamps_us: list[int],
    spin_intervals_us: list[tuple[int, int]],
    lidar_component_generic_data: dict[str, np.ndarray] | None = None,
) -> Path:
    """Build a minimal reloadable NCore v4 manifest with one LiDAR and one VRU track."""
    interval = HalfClosedInterval(0, 300_000)
    meta: dict[str, object] = {}

    shared_writer = v4.SequenceComponentGroupsWriter(
        root,
        "shared",
        "test_scene",
        interval,
        meta,
        store_type="itar",
    )
    shared_writer.register_component_writer(
        v4.PosesComponent.Writer, "default"
    ).store_static_pose("world", "lidar", np.eye(4, dtype=np.float64))
    shared_writer.register_component_writer(
        v4.IntrinsicsComponent.Writer, "default"
    ).store_lidar_intrinsics("lidar", _lidar_model_parameters())
    shared_writer.register_component_writer(
        v4.MasksComponent.Writer, "default"
    ).store_camera_masks("camera", {"valid": Image.new("L", (1, 1), 255)})
    shared_writer.register_component_writer(
        v4.CuboidsComponent.Writer, "default"
    ).store_observations(
        [
            CuboidTrackObservation(
                track_id="ped",
                class_id="person",
                timestamp_us=timestamp_us,
                reference_frame_id="world",
                reference_frame_timestamp_us=timestamp_us,
                bbox3=BBox3((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0)),
                source=LabelSource.GT_SYNTHETIC,
            )
            for timestamp_us in cuboid_timestamps_us
        ]
    )
    shared_meta = (
        v4.SequenceComponentGroupsReader(
            shared_writer.finalize(),
            open_consolidated=True,
        )
        .get_sequence_meta()
        .to_dict()["component_stores"]
    )

    lidar_writer_root = v4.SequenceComponentGroupsWriter(
        root,
        "lidar_only",
        "test_scene",
        interval,
        meta,
        store_type="itar",
    )
    lidar_writer = lidar_writer_root.register_component_writer(
        v4.LidarSensorComponent.Writer,
        "lidar",
        group_name="lidar",
    )
    if lidar_component_generic_data:
        lidar_writer.set_generic_data(lidar_component_generic_data)
    for start_us, end_us in spin_intervals_us:
        lidar_writer.store_frame(
            direction=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            timestamp_us=np.array([start_us + 50], dtype=np.uint64),
            model_element=np.array([[0, 0]], dtype=np.uint16),
            distance_m=np.array([[0.0]], dtype=np.float32),
            intensity=np.array([[1.0]], dtype=np.float32),
            frame_timestamps_us=np.array([start_us, end_us], dtype=np.uint64),
            generic_data={},
            generic_meta_data={},
        )
    lidar_meta = (
        v4.SequenceComponentGroupsReader(
            lidar_writer_root.finalize(),
            open_consolidated=True,
        )
        .get_sequence_meta()
        .to_dict()["component_stores"]
    )

    return _write_manifest(root, shared_meta + lidar_meta)


def _write_static_only_ncore_manifest(root: Path) -> Path:
    """Single stationary cuboid observation so classify_tracks yields no dynamic tracks."""
    return _write_synthetic_ncore_manifest(
        root,
        cuboid_timestamps_us=[50_000],
        spin_intervals_us=[(0, 100_000), (100_000, 200_000)],
    )


@pytest.fixture
def synthetic_manifest(tmp_path: Path) -> Path:
    return _write_synthetic_ncore_manifest(
        tmp_path / "input",
        cuboid_timestamps_us=[50_000, 150_000],
        spin_intervals_us=[(0, 100_000), (100_000, 200_000)],
    )


def test_derive_dynamic_flags_round_trip_writes_reloadable_dynamic_flags(
    tmp_path: Path,
    synthetic_manifest: Path,
):
    output_dir = tmp_path / "derived"
    output_manifest = derive_dynamic_flags(
        input_manifest_path=synthetic_manifest,
        output_dir=output_dir,
        class_ids=["person"],
        lidar_id="lidar",
    )

    assert output_manifest == output_dir / synthetic_manifest.name
    assert output_dir.is_dir()
    assert not (output_dir.parent / f".{output_dir.name}.tmp").exists()

    report = json.loads((output_dir / COVERAGE_REPORT_NAME).read_text())
    assert report["association"] == "spin"
    assert "spin_membership" in report
    expected_flagged_points = report["counts"]["static_excluded_points"]

    loader = v4.SequenceLoaderV4(
        v4.SequenceComponentGroupsReader([output_manifest], open_consolidated=True)
    )
    lidar = loader.get_lidar_sensor("lidar")
    source = loader.get_point_clouds_source("lidar")
    flagged_points = 0
    for frame_index in range(source.pcs_count):
        assert source.has_pc_generic_data(frame_index, "dynamic_flag")
        dynamic_flag = source.get_pc_generic_data(frame_index, "dynamic_flag")
        expected_shape = (lidar.get_frame_ray_bundle_count(frame_index),)
        assert dynamic_flag.shape == expected_shape
        assert set(np.unique(dynamic_flag).tolist()).issubset({0, 1})
        flagged_points += int(dynamic_flag.sum())

    assert flagged_points == expected_flagged_points


def test_derive_dynamic_flags_preserves_component_generic_data(tmp_path: Path):
    component_api = (
        getattr(v4.LidarSensorComponent.Reader, "get_generic_data_names", None),
        getattr(v4.LidarSensorComponent.Reader, "get_generic_data", None),
        getattr(v4.LidarSensorComponent.Writer, "set_generic_data", None),
    )
    if not all(callable(method) for method in component_api):
        pytest.skip("installed NCore SDK lacks component-level generic data")

    component_generic_data = {
        "calibration_offsets": np.arange(6, dtype=np.float64).reshape(3, 2),
        "channel_ids": np.array([3, 1, 4, 1, 5], dtype=np.int16),
    }
    manifest = _write_synthetic_ncore_manifest(
        tmp_path / "input",
        cuboid_timestamps_us=[50_000, 150_000],
        spin_intervals_us=[(0, 100_000), (100_000, 200_000)],
        lidar_component_generic_data=component_generic_data,
    )

    output_dir = tmp_path / "derived"
    output_manifest = derive_dynamic_flags(
        input_manifest_path=manifest,
        output_dir=output_dir,
        class_ids=["person"],
        lidar_id="lidar",
    )

    reader = v4.SequenceComponentGroupsReader([output_manifest], open_consolidated=True)
    lidar_reader = reader.open_component_readers(v4.LidarSensorComponent.Reader)[
        "lidar"
    ]

    assert set(lidar_reader.get_generic_data_names()) == set(component_generic_data)
    for name, expected in component_generic_data.items():
        actual = lidar_reader.get_generic_data(name)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)


def test_copy_component_generic_data_preserves_value_dtype_and_shape():
    source_data = {
        "calibration_offsets": np.arange(6, dtype=np.float64).reshape(3, 2),
        "channel_ids": np.array([3, 1, 4, 1, 5], dtype=np.int16),
    }

    class SourceReader:
        def get_generic_data_names(self):
            return list(source_data)

        def get_generic_data(self, name):
            return source_data[name]

    class DestinationWriter:
        def __init__(self):
            self.data = {}

        def set_generic_data(self, data):
            self.data = data

    destination = DestinationWriter()
    _copy_component_generic_data(SourceReader(), destination)

    assert set(destination.data) == set(source_data)
    for name, expected in source_data.items():
        actual = destination.data[name]
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)


def test_copy_component_generic_data_allows_sdk_without_api():
    _copy_component_generic_data(object(), object())


def test_copy_component_generic_data_rejects_partial_api():
    class SourceReader:
        def get_generic_data_names(self):
            return []

    with pytest.raises(RuntimeError, match="only part"):
        _copy_component_generic_data(SourceReader(), object())


def test_derive_dynamic_flags_rejects_invalid_manifest(tmp_path: Path):
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"foo": 1}))

    with pytest.raises(ValueError, match="not a valid NCore v4 manifest"):
        derive_dynamic_flags(
            input_manifest_path=manifest,
            output_dir=tmp_path / "derived",
            class_ids=["person"],
        )


def test_derive_dynamic_flags_allows_static_only_classification(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    manifest = _write_static_only_ncore_manifest(tmp_path / "input")

    with caplog.at_level(logging.WARNING):
        output_manifest = derive_dynamic_flags(
            input_manifest_path=manifest,
            output_dir=tmp_path / "derived",
            class_ids=["person"],
            lidar_id="lidar",
        )

    assert output_manifest.is_file()
    assert any(
        "No requested cuboid tracks were classified as dynamic" in record.message
        for record in caplog.records
    )

    loader = v4.SequenceLoaderV4(
        v4.SequenceComponentGroupsReader([output_manifest], open_consolidated=True)
    )
    source = loader.get_point_clouds_source("lidar")
    for frame_index in range(source.pcs_count):
        dynamic_flag = source.get_pc_generic_data(frame_index, "dynamic_flag")
        assert int(dynamic_flag.sum()) == 0

    report = json.loads((output_manifest.parent / COVERAGE_REPORT_NAME).read_text())
    assert report["counts"]["static_excluded_points"] == 0
    assert report["dynamic_tracks"] == []
    assert len(report["stationary_tracks"]) == 1


def test_copy_unchanged_stores_links_without_byte_copy(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_manifest = input_dir / "sequence.json"
    source_store = input_dir / "camera.zarr.itar"
    source_store.write_bytes(b"store")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    unchanged_store = {"path": source_store.name}
    replaced_store = {"path": "lidar.zarr.itar"}

    linked_paths = _copy_unchanged_stores(
        input_manifest,
        output_dir,
        {"component_stores": [unchanged_store, replaced_store]},
        replaced_store,
    )

    assert linked_paths == [output_dir / source_store.name]
    assert linked_paths[0].read_bytes() == source_store.read_bytes()
    assert linked_paths[0].samefile(source_store)


def test_unchanged_stores_reject_duplicate_basenames(tmp_path: Path):
    input_manifest = tmp_path / "sequence.json"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    output_dir = tmp_path / "output"
    first_dir.mkdir()
    second_dir.mkdir()
    output_dir.mkdir()
    first_store = first_dir / "camera.zarr.itar"
    second_store = second_dir / "camera.zarr.itar"
    first_store.write_bytes(b"first")
    second_store.write_bytes(b"second")
    first = {"path": str(first_store)}
    second = {"path": str(second_store)}
    replaced = {"path": "lidar.zarr.itar"}

    with pytest.raises(ValueError, match="unique basenames"):
        _copy_unchanged_stores(
            input_manifest,
            output_dir,
            {"component_stores": [first, second, replaced]},
            replaced,
        )

    assert not (output_dir / "camera.zarr.itar").exists()


@pytest.mark.skipif(
    not os.environ.get(NCORE_TEST_SCENE_ENV),
    reason=f"set {NCORE_TEST_SCENE_ENV} for env-gated integration coverage",
)
def test_derive_dynamic_flags_round_trip_on_env_scene(tmp_path: Path):
    scene = Path(os.environ[NCORE_TEST_SCENE_ENV])
    if not scene.is_file():
        pytest.skip(f"{NCORE_TEST_SCENE_ENV} does not exist: {scene}")

    output_dir = tmp_path / "derived_env"
    output_manifest = derive_dynamic_flags(
        input_manifest_path=scene,
        output_dir=output_dir,
        class_ids=["person", "automobile"],
    )
    report = json.loads((output_dir / COVERAGE_REPORT_NAME).read_text())
    loader = v4.SequenceLoaderV4(
        v4.SequenceComponentGroupsReader([output_manifest], open_consolidated=True)
    )
    source = loader.get_point_clouds_source(loader.lidar_ids[0])
    flagged_points = sum(
        int(source.get_pc_generic_data(frame_index, "dynamic_flag").sum())
        for frame_index in range(source.pcs_count)
    )
    assert flagged_points == report["counts"]["static_excluded_points"]
