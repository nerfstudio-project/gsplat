# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Regression coverage for proper camera normalization; no renderer required."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


def cameras(rotation):
    out = np.tile(np.eye(4), (4, 1, 1))
    out[:, :3, :3] = rotation
    out[:, :3, 3] = [[-3, 0, 2], [-1, 1, 4], [1, -1, 2], [3, 0, 4]]
    return out


def rotation_component(transform):
    scale = np.linalg.norm(transform[:3, 0])
    return transform[:3, :3] / scale


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("center", ["focus", "poses"])
def test_antiparallel_alignment_is_proper_rotation(gsplat_normalize, strict, center):
    c2w = cameras(np.diag([1.0, -1.0, -1.0]))
    before = c2w.copy()
    transform = gsplat_normalize.similarity_from_cameras(c2w, strict, center)
    rotation = rotation_component(transform)
    assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert_allclose(rotation @ np.array([0.0, 1.0, 0.0]), [0.0, -1.0, 0.0])
    normalized = gsplat_normalize.transform_cameras(transform, c2w)
    assert_allclose(np.linalg.det(normalized[:, :3, :3]), np.ones(4), atol=1e-14)
    assert_allclose(
        normalized[:, :3, :3], np.broadcast_to(np.eye(3), (4, 3, 3)), atol=1e-14
    )
    assert_array_equal(c2w, before)


@pytest.mark.parametrize("with_points", [False, True])
def test_normalize_pipeline_preserves_handedness(gsplat_normalize, with_points):
    c2w = cameras(np.diag([1.0, -1.0, -1.0]))
    points = np.random.default_rng(37).normal(size=(50, 3)) * [3.0, 2.0, 1.0]
    result = gsplat_normalize.normalize(c2w, points if with_points else None)
    transformed_cameras = result[0]
    transform = result[-1]
    assert np.linalg.det(transform[:3, :3]) > 0
    assert_allclose(
        np.linalg.det(transformed_cameras[:, :3, :3]), np.ones(4), atol=1e-12
    )
    if with_points:
        assert_allclose(
            result[1], gsplat_normalize.transform_points(transform, points), atol=1e-12
        )


@pytest.mark.parametrize("rotation", [np.eye(3), np.diag([-1.0, -1.0, 1.0])])
def test_camera_projection_and_inverse_remain_consistent(gsplat_normalize, rotation):
    c2w = cameras(rotation)
    transform = gsplat_normalize.similarity_from_cameras(c2w)
    new_cameras = gsplat_normalize.transform_cameras(transform, c2w)
    points = np.array([[0.0, 2.0, 8.0], [1.0, 3.0, 7.0], [-2.0, -1.0, 9.0]])
    new_points = gsplat_normalize.transform_points(transform, points)
    assert_allclose(np.linalg.det(new_cameras[:, :3, :3]), np.ones(4), atol=1e-12)
    scale = np.linalg.norm(transform[:3, 0])
    for old_cam, new_cam in zip(c2w, new_cameras):
        old_local = (points - old_cam[:3, 3]) @ old_cam[:3, :3]
        new_local = (new_points - new_cam[:3, 3]) @ new_cam[:3, :3]
        assert_allclose(new_local, scale * old_local, atol=1e-12)
        assert_allclose(
            new_local[:, :2] / new_local[:, 2:], old_local[:, :2] / old_local[:, 2:]
        )
    assert_allclose(
        gsplat_normalize.transform_points(np.linalg.inv(transform), new_points),
        points,
        atol=1e-12,
    )


@pytest.mark.parametrize("angle", [0.0, 0.4, -0.7, 2.0])
def test_ordinary_alignment_controls(gsplat_normalize, angle):
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    c2w = cameras(r)
    transform = gsplat_normalize.similarity_from_cameras(c2w)
    rotation = rotation_component(transform)
    assert_allclose(rotation @ (-r[:, 1]), [0.0, -1.0, 0.0], atol=1e-12)
    assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.fixture(scope="module")
def gsplat_normalize():
    import importlib.util
    from pathlib import Path

    root = next(
        p
        for p in Path(__file__).resolve().parents
        if (p / "examples/datasets/normalize.py").is_file()
    )
    path = root / "examples/datasets/normalize.py"
    spec = importlib.util.spec_from_file_location("regression_gsplat_normalize", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
