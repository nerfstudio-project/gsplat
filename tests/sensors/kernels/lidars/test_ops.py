# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the five spinning-LiDAR kernel ops.

Each op is verified against a self-contained oracle built from the reference
sensor JSON tables -- no stored fixtures. Coverage per op:
    - ray<->angle: analytic cardinal cases, unit-norm rays, the angle->ray->angle
      round-trip, dtype polymorphism, and fp64 gradcheck;
    - elements->angle: exact match to a direct table gather (elevation = row
      table, azimuth = column table + row offset) across the production sensors,
      out-of-bounds validity, and atomicAdd grad accumulation on a shared index;
    - generate: the static-pose origin/direction oracle (origins == translation,
      directions == R(q) sensor_ray) across the production sensors, the
      elements=None grid path, rolling-shutter timestamp ordering, and fp64
      gradcheck over tables + control poses (incl. rotation);
    - inverse: the generate->world->inverse round-trip recovery (>=98% of the
      reference table angles among inverse-valid points on the production
      sensors), out-of-FOV validity, max_iterations range checking, and the fp64
      IFT VJP for world_points + both control poses.

The moving-pose generate forward is exercised transitively by the inverse
round-trip (which generates with a moving, rotating pose and recovers the table
angles). Backward correctness is gated by fp64 gradcheck / IFT rather than stored
grads, which also covers the rotation grads end to end.
"""

import math

import pytest
import torch

from gsplat.sensors.kernels.common import DynamicPose, Pose
from gsplat.sensors.kernels.lidars.ops import (
    _GenerateSpinningLidarRays,
    _InverseProjectSpinningLidar,
    elements_to_sensor_angles,
    generate_spinning_lidar_rays,
    inverse_project_spinning_lidar,
    sensor_angles_to_sensor_rays,
    sensor_rays_to_sensor_angles,
)
from gsplat.sensors.kernels.lidars.types import (
    RowOffsetStructuredSpinningLidarProjection,
)

from ...conftest import LIDAR_JSON_PATHS, LIDAR_PRODUCTION_CONFIGS

FWD_ATOL = 1e-5
GRAD_ATOL = 1e-4
ROUND_TRIP_RECOVERY = 0.98
# fp64 gradcheck: the public ops force float32, so we drive the dtype-templated
# autograd Functions directly with float64 to exercise the double precision path.
GRADCHECK_KWARGS = {"eps": 1e-6, "atol": 1e-4, "rtol": 1e-3, "nondet_tol": 1e-10}


def _control_poses(device, dtype=torch.float32):
    """Return (control_translations (2,3), control_rotations (2,4) wxyz) for a
    mild moving, rotating trajectory: 0.05 m translation and a ~0.02 rad yaw."""
    ct = torch.tensor(
        [[0.0, 0.0, 0.0], [0.05, 0.02, -0.03]], device=device, dtype=dtype
    )
    half = 0.01  # half of the ~0.02 rad yaw
    cr = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [math.cos(half), 0.0, 0.0, math.sin(half)]],
        device=device,
        dtype=dtype,
    )
    cr = cr / cr.norm(dim=1, keepdim=True)
    return ct, cr


def _dynamic_pose(device, dtype=torch.float32):
    """Build a moving, rotating DynamicPose from the shared control poses."""
    ct, cr = _control_poses(device, dtype)
    return DynamicPose(
        start_pose=Pose(translation=ct[0].clone(), rotation=cr[0].clone()),
        end_pose=Pose(translation=ct[1].clone(), rotation=cr[1].clone()),
    )


def _small_synthetic_projection(device):
    """A 4x8 no-offset projection for the elements=None grid and gradcheck paths."""
    re = torch.tensor([-0.3, -0.1, 0.1, 0.3], device=device)
    ca = torch.linspace(-2.7, 2.7, 8, device=device)
    fov_vert_start = float(re.max())
    fov_vert_span = abs(fov_vert_start - float(re.min()))
    return RowOffsetStructuredSpinningLidarProjection(
        re.float(),
        ca.float(),
        torch.zeros((0,), device=device),
        fov_vert_start,
        fov_vert_span,
        -3.14159,
        6.28318,
        0,
        False,
    )


def _json_f64_tables(config, device):
    """Load (row_elev, col_az, row_offsets) as float64 plus a matching float32
    projection from a reference sensor JSON, for the fp64 gradcheck paths."""
    import json

    with LIDAR_JSON_PATHS[config].open(encoding="utf-8") as f:
        params = json.load(f)
    dt = torch.float64
    re = torch.tensor(params["row_elevations_rad"], device=device, dtype=dt)
    ca = torch.tensor(params["column_azimuths_rad"], device=device, dtype=dt)
    offsets = params.get("row_azimuth_offsets_rad")
    has_offsets = offsets is not None and any(v != 0 for v in offsets)
    ro = (
        torch.tensor(offsets, device=device, dtype=dt)
        if has_offsets
        else torch.zeros((0,), device=device, dtype=dt)
    )
    fov_vert_start = float(re.max())
    fov_vert_span = abs(fov_vert_start - float(re.min()))
    proj = RowOffsetStructuredSpinningLidarProjection(
        re.float(),
        ca.float(),
        ro.float() if has_offsets else torch.zeros((0,), device=device),
        fov_vert_start,
        fov_vert_span,
        -3.14159,
        6.28318,
        0,
        has_offsets,
    )
    return re, ca, ro, has_offsets, proj


def _sample_elements(n_rows, n_cols, device, max_per_dim=16):
    """Return a strided (row, col) grid capped to ~max_per_dim entries per axis."""
    rstep = max(1, n_rows // max_per_dim)
    cstep = max(1, n_cols // max_per_dim)
    rows = torch.arange(0, n_rows, rstep, device=device)
    cols = torch.arange(0, n_cols, cstep, device=device)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([rr.reshape(-1), cc.reshape(-1)], dim=1).to(torch.int32)


def _quat_rotation_matrix(qwxyz, device):
    """Build the 3x3 rotation matrix for a wxyz quaternion."""
    qw, qx, qy, qz = qwxyz
    return torch.tensor(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        device=device,
    )


# ===========================================================================
# Op4 / Op5: sensor_rays <-> sensor_angles
# ===========================================================================


def test_rays_angles_round_trip(sensor_device):
    """Verify angles->rays->angles recovers the original sensor angles."""
    elev = torch.linspace(-0.4, 0.4, 16, device=sensor_device)
    azim = torch.linspace(-2.0, 2.0, 16, device=sensor_device)
    angles = torch.stack([elev, azim], dim=1)
    rays = sensor_angles_to_sensor_rays(angles, allow_device_transfer=True)
    recovered = sensor_rays_to_sensor_angles(rays, allow_device_transfer=True)
    assert torch.allclose(recovered, angles, atol=1e-5)


def test_forward_x_axis_maps_to_zero_angles(sensor_device):
    """Verify the sensor +X ray [1,0,0] maps to (elevation=0, azimuth=0)."""
    rays = torch.tensor([[1.0, 0.0, 0.0]], device=sensor_device)
    angles = sensor_rays_to_sensor_angles(rays, allow_device_transfer=True)
    assert torch.allclose(angles, torch.zeros_like(angles), atol=1e-6)


def test_angles_to_rays_are_unit_norm(sensor_device):
    """Verify rays produced from angles are unit length by construction."""
    elev = torch.linspace(-1.0, 1.0, 32, device=sensor_device)
    azim = torch.linspace(-3.0, 3.0, 32, device=sensor_device)
    angles = torch.stack([elev, azim], dim=1)
    rays = sensor_angles_to_sensor_rays(angles, allow_device_transfer=True)
    norms = rays.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_sensor_rays_to_sensor_angles_dtype_polymorphism(sensor_device):
    """Verify ray->angle output dtype follows the input dtype (float32/float64)."""
    rays32 = torch.tensor([[0.3, 0.2, 0.9]], device=sensor_device, dtype=torch.float32)
    rays64 = rays32.to(torch.float64)
    out32 = sensor_rays_to_sensor_angles(rays32, allow_device_transfer=True)
    out64 = sensor_rays_to_sensor_angles(rays64, allow_device_transfer=True)
    assert out32.dtype == torch.float32
    assert out64.dtype == torch.float64


# ===========================================================================
# Op2: elements_to_sensor_angles
# ===========================================================================


@pytest.mark.parametrize("config", LIDAR_PRODUCTION_CONFIGS)
def test_elements_to_sensor_angles_matches_table_gather(
    lidar_projection_from_json, config
):
    """Verify elements->angle equals a direct table gather across production sensors.

    The kernel maps (row, col) to (row_elevations[row], column_azimuths[col] +
    row_offset[row]); the offset branch normalizes the azimuth. Elevation is
    compared exactly; azimuth is compared wrap-invariantly (the normalize seam at
    +-pi is a branch-cut artifact, not a value difference).
    """
    projection = lidar_projection_from_json(config)
    re = projection.row_elevations_rad
    ca = projection.column_azimuths_rad
    ro = projection.row_azimuth_offsets_rad
    n_rows, n_cols = re.shape[0], ca.shape[0]
    elements = _sample_elements(n_rows, n_cols, re.device, max_per_dim=24)
    rows = elements[:, 0].long()
    cols = elements[:, 1].long()

    angles, valid = elements_to_sensor_angles(
        elements, projection, allow_device_transfer=True
    )
    assert bool(valid.all())
    assert torch.allclose(angles[:, 0], re[rows], atol=1e-6)
    expected_az = ca[cols]
    if ro.numel() > 0:
        expected_az = expected_az + ro[rows]
    d_az = angles[:, 1] - expected_az
    d_az = torch.atan2(torch.sin(d_az), torch.cos(d_az))
    assert d_az.abs().max().item() < 1e-5


def test_elements_out_of_bounds_marks_invalid(lidar_projection_from_json):
    """Verify out-of-bounds row/col yields valid=False and (0, 0) angles."""
    projection = lidar_projection_from_json("generic")
    n_rows = projection.row_elevations_rad.shape[0]
    n_cols = projection.column_azimuths_rad.shape[0]
    elements = torch.tensor(
        [[0, 0], [n_rows, 0], [0, n_cols], [-1, 0], [0, -1]],
        device="cuda",
        dtype=torch.int32,
    )
    angles, valid = elements_to_sensor_angles(
        elements, projection, allow_device_transfer=True
    )
    assert valid.tolist() == [True, False, False, False, False]
    assert torch.equal(angles[1:], torch.zeros_like(angles[1:]))


def test_elements_table_grad_accumulates_on_shared_index(lidar_projection_from_json):
    """Verify many elements reading the same row/col atomically sum their grads.

    Several elements share row 0 / column 0. With an all-ones upstream grad, the
    row-0 elevation slot must receive the count of rows==0 (the azimuth output has
    no elevation-table dependence), validating the atomicAdd accumulation.
    """
    base = lidar_projection_from_json("generic")
    re = base.row_elevations_rad.clone().requires_grad_(True)
    ca = base.column_azimuths_rad.clone().requires_grad_(True)
    projection = RowOffsetStructuredSpinningLidarProjection(
        re,
        ca,
        base.row_azimuth_offsets_rad,
        base.fov_vert_start_rad,
        base.fov_vert_span_rad,
        base.fov_horiz_start_rad,
        base.fov_horiz_span_rad,
        base.spinning_direction,
        base.has_row_offsets,
    )
    n_share = 7
    elements = torch.stack(
        [
            torch.zeros(n_share, dtype=torch.int32),
            torch.arange(n_share, dtype=torch.int32),
        ],
        dim=1,
    ).cuda()
    angles, _ = elements_to_sensor_angles(
        elements, projection, allow_device_transfer=True
    )
    angles.backward(torch.ones_like(angles))
    # elevation output is column 0 -> d(elevation)/d(row_elevations[0]) = 1 per
    # element reading row 0; n_share elements all read row 0.
    assert re.grad[0].item() == pytest.approx(float(n_share))
    assert re.grad[1:].abs().sum().item() == pytest.approx(0.0)
    assert torch.allclose(
        ca.grad[:n_share], torch.ones(n_share, device=ca.device), atol=1e-6
    )


def test_projection_bindings_readonly(lidar_projection_from_json):
    """Verify RowOffsetStructuredSpinningLidarProjection fields are read-only.

    Rebinding a field after construction (which would bypass
    check_lidar_projection, re-validated only at launch) must raise.
    """
    projection = lidar_projection_from_json("generic")
    device = projection.row_elevations_rad.device
    for attr in (
        "row_elevations_rad",
        "column_azimuths_rad",
        "row_azimuth_offsets_rad",
    ):
        with pytest.raises((AttributeError, RuntimeError)):
            setattr(projection, attr, torch.zeros(1, device=device))
    with pytest.raises((AttributeError, RuntimeError, TypeError)):
        projection.has_row_offsets = False


# ===========================================================================
# Op1: generate_spinning_lidar_rays
# ===========================================================================


@pytest.mark.parametrize("config", LIDAR_PRODUCTION_CONFIGS)
def test_generate_static_pose_origin_and_direction_oracle(
    lidar_projection_from_json, sensor_device, config
):
    """Verify origins == translation and directions == R(q) sensor_ray for a static pose.

    With a static pose the time interpolation collapses, so generate reduces to a
    pure rotate-and-offset of the per-element sensor ray -- checkable in closed
    form against an independent rotation matrix on every production sensor.
    """
    projection = lidar_projection_from_json(config)
    n_rows = projection.row_elevations_rad.shape[0]
    n_cols = projection.column_azimuths_rad.shape[0]
    q = torch.tensor([0.5, 0.5, 0.5, 0.5], device=sensor_device)  # wxyz
    t = torch.tensor([1.0, -2.0, 3.0], device=sensor_device)
    pose = DynamicPose(
        start_pose=Pose(translation=t.clone(), rotation=q.clone()),
        end_pose=Pose(translation=t.clone(), rotation=q.clone()),
    )
    elements = _sample_elements(n_rows, n_cols, sensor_device).cuda()
    world_rays, _, _, _ = generate_spinning_lidar_rays(projection, elements, pose)
    angles, _ = elements_to_sensor_angles(
        elements, projection, allow_device_transfer=True
    )
    ce = torch.cos(angles[:, 0])
    se = torch.sin(angles[:, 0])
    ray = torch.stack(
        [ce * torch.cos(angles[:, 1]), ce * torch.sin(angles[:, 1]), se], dim=1
    )
    rot = _quat_rotation_matrix(q, sensor_device)
    expected_dir = (rot @ ray.T).T
    assert torch.allclose(world_rays[:, :3], t.expand_as(world_rays[:, :3]), atol=1e-5)
    assert torch.allclose(world_rays[:, 3:], expected_dir, atol=1e-5)


def test_generate_mode_equals_explicit_grid(sensor_device):
    """Verify elements=None generates the full row-major grid identical to explicit elements."""
    projection = _small_synthetic_projection(sensor_device)
    pose = _dynamic_pose(sensor_device)
    n_rows = projection.row_elevations_rad.shape[0]
    n_cols = projection.column_azimuths_rad.shape[0]
    rays_gen, _, _, _ = generate_spinning_lidar_rays(projection, None, pose)
    rr, cc = torch.meshgrid(torch.arange(n_rows), torch.arange(n_cols), indexing="ij")
    explicit = (
        torch.stack([rr.reshape(-1), cc.reshape(-1)], dim=1).cuda().to(torch.int32)
    )
    rays_exp, _, _, _ = generate_spinning_lidar_rays(projection, explicit, pose)
    assert rays_gen.shape[0] == n_rows * n_cols
    assert torch.allclose(rays_gen, rays_exp, atol=1e-6)


def test_generate_timestamps_monotonic_with_column(lidar_projection_from_json):
    """Verify per-ray timestamps increase with column index (rolling shutter order)."""
    projection = lidar_projection_from_json("generic")
    pose = _dynamic_pose(torch.device("cuda"))
    n_cols = projection.column_azimuths_rad.shape[0]
    cols = torch.arange(0, n_cols, 100, device="cuda", dtype=torch.int32)
    elements = torch.stack([torch.zeros_like(cols), cols], dim=1)
    _, timestamps, _, _ = generate_spinning_lidar_rays(
        projection,
        elements,
        pose,
        start_timestamp_us=1000,
        end_timestamp_us=5000,
        return_timestamps=True,
    )
    diffs = timestamps[1:] - timestamps[:-1]
    assert bool((diffs >= 0).all())
    assert bool((diffs > 0).any())


# ===========================================================================
# Op3: inverse_project_spinning_lidar
# ===========================================================================


@pytest.mark.parametrize("config", LIDAR_PRODUCTION_CONFIGS)
def test_generate_world_inverse_round_trip_recovery(
    lidar_projection_from_json, sensor_device, config
):
    """Verify generate->world->inverse recovers >=98% of the table angles (production).

    A point is placed at depth 10 along every generated ray, inverse-projected,
    and the recovered (elevation, azimuth) compared to the table angles. Recovery
    is measured among inverse-valid points. This is the forward oracle for the
    moving-pose generate and inverse ops jointly.
    """
    projection = lidar_projection_from_json(config)
    pose = _dynamic_pose(sensor_device)
    re = projection.row_elevations_rad
    ca = projection.column_azimuths_rad
    ro = projection.row_azimuth_offsets_rad
    n_rows, n_cols = re.shape[0], ca.shape[0]
    world_rays, _, _, _ = generate_spinning_lidar_rays(
        projection, None, pose, start_timestamp_us=0, end_timestamp_us=100000
    )
    world_points = world_rays[:, 0:3] + 10.0 * world_rays[:, 3:6]
    rows = torch.arange(n_rows, device="cuda").repeat_interleave(n_cols)
    cols = torch.arange(n_cols, device="cuda").repeat(n_rows)
    gt_elev = re[rows]
    gt_az = ca[cols]
    if ro.numel() > 0:
        gt_az = gt_az + ro[rows]
    angles, valid, _, _, _ = inverse_project_spinning_lidar(
        projection,
        world_points.detach(),
        pose,
        max_iterations=10,
        start_timestamp_us=0,
        end_timestamp_us=100000,
    )
    d_elev = (angles[:, 0] - gt_elev).abs()
    d_az = angles[:, 1] - gt_az
    d_az = torch.atan2(torch.sin(d_az), torch.cos(d_az)).abs()
    ang_err = torch.maximum(d_elev, d_az)
    recovered = valid & (ang_err < 1e-3)
    frac = recovered.float().sum().item() / max(valid.float().sum().item(), 1.0)
    assert valid.float().mean().item() >= 0.95
    assert frac >= ROUND_TRIP_RECOVERY


def test_inverse_out_of_fov_marks_invalid(lidar_projection_from_json, sensor_device):
    """Verify an in-FOV point is valid and a straight-up point (out of vertical FOV) is invalid."""
    projection = lidar_projection_from_json("generic")
    q = torch.tensor([0.5, 0.5, 0.5, 0.5], device=sensor_device)  # wxyz
    t = torch.tensor([1.0, -2.0, 3.0], device=sensor_device)
    pose = DynamicPose(
        start_pose=Pose(translation=t.clone(), rotation=q.clone()),
        end_pose=Pose(translation=t.clone(), rotation=q.clone()),
    )
    rot = _quat_rotation_matrix(q, sensor_device)
    forward_world = rot @ torch.tensor([1.0, 0.0, 0.0], device=sensor_device)
    up_world = rot @ torch.tensor([0.0, 0.0, 1.0], device=sensor_device)
    points = torch.stack([t + 5.0 * forward_world, t + 5.0 * up_world])
    _, valid, _, _, _ = inverse_project_spinning_lidar(projection, points, pose)
    assert bool(valid[0].item())
    assert not bool(valid[1].item())


@pytest.mark.parametrize("bad_iterations", [0, 33, 64])
def test_inverse_rejects_out_of_range_max_iterations(
    lidar_projection_from_json, sensor_device, bad_iterations
):
    """Verify max_iterations outside [1, 32] is loudly rejected, not silently clamped."""
    projection = lidar_projection_from_json("generic")
    pose = _dynamic_pose(sensor_device)
    world_points = torch.tensor(
        [[5.0, 0.0, 0.0], [4.0, 1.0, -0.5]], device=sensor_device
    )
    with pytest.raises(RuntimeError, match="max_iterations"):
        inverse_project_spinning_lidar(
            projection,
            world_points,
            pose,
            max_iterations=bad_iterations,
            start_timestamp_us=0,
            end_timestamp_us=100000,
        )


# ===========================================================================
# fp64 gradcheck (opt-in via -m gradcheck)
# ===========================================================================


@pytest.mark.gradcheck
def test_sensor_rays_to_sensor_angles_fp64_gradcheck(sensor_device):
    """fp64 gradcheck for the ray->angle op."""
    rays = torch.tensor(
        [[0.3, 0.2, 0.9], [-0.1, 0.5, 0.8]],
        device=sensor_device,
        dtype=torch.float64,
        requires_grad=True,
    )

    def fn(r):
        return sensor_rays_to_sensor_angles(r, allow_device_transfer=True)

    assert torch.autograd.gradcheck(fn, (rays,), **GRADCHECK_KWARGS)


@pytest.mark.gradcheck
def test_sensor_angles_to_sensor_rays_fp64_gradcheck(sensor_device):
    """fp64 gradcheck for the angle->ray op."""
    angles = torch.tensor(
        [[0.1, 0.4], [-0.3, 1.2]],
        device=sensor_device,
        dtype=torch.float64,
        requires_grad=True,
    )

    def fn(a):
        return sensor_angles_to_sensor_rays(a, allow_device_transfer=True)

    assert torch.autograd.gradcheck(fn, (angles,), **GRADCHECK_KWARGS)


@pytest.mark.gradcheck
def test_elements_to_sensor_angles_fp64_gradcheck(sensor_device):
    """fp64 gradcheck for the three angle tables of elements_to_sensor_angles.

    Uses a small hand-built projection with elevations / azimuths / offsets kept
    well inside ``(-pi, pi)`` so the ``normalize_angle`` wrap seam (a true
    discontinuity at +-pi) does not corrupt the finite-difference jacobian.
    """
    from gsplat.sensors.kernels.lidars.ops import _ElementsToSensorAngles

    re = torch.tensor([-0.3, -0.1, 0.1, 0.3], device=sensor_device, dtype=torch.float64)
    ca = torch.tensor(
        [-1.0, -0.5, 0.0, 0.5, 1.0], device=sensor_device, dtype=torch.float64
    )
    ro = torch.tensor(
        [0.05, -0.05, 0.02, -0.02], device=sensor_device, dtype=torch.float64
    )
    proj = RowOffsetStructuredSpinningLidarProjection(
        re.float(), ca.float(), ro.float(), 0.3, 0.6, -3.14159, 6.28318, 0, True
    )
    rows, cols = torch.meshgrid(torch.arange(4), torch.arange(5), indexing="ij")
    elements = (
        torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=1)
        .to(sensor_device)
        .to(torch.int32)
    )
    re_g = re.clone().requires_grad_(True)
    ca_g = ca.clone().requires_grad_(True)
    ro_g = ro.clone().requires_grad_(True)

    def fn(a, b, c):
        return _ElementsToSensorAngles.apply(elements, a, b, c, proj)[0]

    assert torch.autograd.gradcheck(fn, (re_g, ca_g, ro_g), **GRADCHECK_KWARGS)


@pytest.mark.gradcheck
def test_generate_fp64_gradcheck(sensor_device):
    """fp64 gradcheck for generate over tables + control poses (incl. rotation grads)."""
    re = torch.tensor([-0.3, -0.1, 0.1, 0.3], device=sensor_device, dtype=torch.float64)
    ca = torch.tensor(
        [-1.0, -0.5, 0.0, 0.5, 1.0], device=sensor_device, dtype=torch.float64
    )
    ro = torch.tensor(
        [0.05, -0.05, 0.02, -0.02], device=sensor_device, dtype=torch.float64
    )
    proj = RowOffsetStructuredSpinningLidarProjection(
        re.float(), ca.float(), ro.float(), 0.3, 0.6, -3.14159, 6.28318, 0, True
    )
    rows, cols = torch.meshgrid(torch.arange(4), torch.arange(5), indexing="ij")
    elements = (
        torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=1)
        .to(sensor_device)
        .to(torch.int32)
    )
    ct, cr = _control_poses(sensor_device, torch.float64)

    re_g = re.clone().requires_grad_(True)
    ca_g = ca.clone().requires_grad_(True)
    ro_g = ro.clone().requires_grad_(True)
    ct_g = ct.clone().requires_grad_(True)
    cr_g = cr.clone().requires_grad_(True)

    def fn(a, b, c, t, r):
        return _GenerateSpinningLidarRays.apply(
            elements, a, b, c, t, r, proj, 0, 100000
        )[0]

    assert torch.autograd.gradcheck(
        fn, (re_g, ca_g, ro_g, ct_g, cr_g), **GRADCHECK_KWARGS
    )


@pytest.mark.gradcheck
def test_inverse_fp64_gradcheck_world_points_and_poses(sensor_device):
    """fp64 verification of the inverse IFT VJP for world_points + the two control poses.

    The iterative forward + discrete column lookup is non-smooth, so a naive
    gradcheck of the full op is dominated by finite-difference column flips. The
    IFT backward differentiates the FROZEN-alpha single step; we validate it by
    (1) gradcheck-ing the equivalent pure-PyTorch frozen step on the SAME
    converged alpha the CUDA forward saved to scratch, and (2) confirming the CUDA
    analytic grads (world_points + both control poses, incl. rotation) match that
    pure-PyTorch autograd to 1e-6 among jointly-valid points.
    """
    re, ca, ro, _, proj = _json_f64_tables("generic", sensor_device)
    dt = torch.float64
    n_cols = ca.shape[0]

    # robustly-in-FOV seed points: place mid-grid generated rays at depth 12.
    pose32 = _dynamic_pose(sensor_device, torch.float32)
    world_rays, _, _, _ = generate_spinning_lidar_rays(
        proj, None, pose32, start_timestamp_us=0, end_timestamp_us=100000
    )
    sel = torch.tensor(
        [30 * n_cols + c for c in (200, 900, 1800, 2700, 3400)], device=sensor_device
    )
    wp_seed = (world_rays[sel, 0:3] + 12.0 * world_rays[sel, 3:6]).detach().to(dt)

    ct, cr = _control_poses(sensor_device, dt)

    wp0 = wp_seed.clone().requires_grad_(True)
    ct0 = ct.clone().requires_grad_(True)
    cr0 = cr.clone().requires_grad_(True)
    angles, valid, _, _, _ = _InverseProjectSpinningLidar.apply(
        wp0, re, ca, ro, ct0, cr0, proj, 10, 1e-4, 1e-6, 0.5, 0, 100000
    )
    assert bool(valid.all())
    upstream = torch.randn_like(angles)
    angles.backward(upstream)
    g_wp_cuda = wp0.grad.clone()
    g_ct_cuda = ct0.grad.clone()
    g_cr_cuda = cr0.grad.clone()

    # angle tables must carry no gradient (piecewise-constant lookup).
    re_g = re.clone().requires_grad_(True)
    ca_g = ca.clone().requires_grad_(True)
    ro_g = ro.clone().requires_grad_(True)
    angles2, _, _, _, _ = _InverseProjectSpinningLidar.apply(
        wp_seed.clone().requires_grad_(True),
        re_g,
        ca_g,
        ro_g,
        ct.clone().requires_grad_(True),
        cr.clone().requires_grad_(True),
        proj,
        10,
        1e-4,
        1e-6,
        0.5,
        0,
        100000,
    )
    angles2.sum().backward()
    for table_grad in (re_g.grad, ca_g.grad, ro_g.grad):
        assert table_grad is None or table_grad.abs().max() == 0

    # retrieve the exact converged alpha (scratch) the CUDA forward saved.
    with torch.no_grad():
        (
            _,
            _,
            _,
            _,
            _,
            scratch,
        ) = torch.ops.gsplat_sensors.inverse_project_spinning_lidar(
            proj, wp_seed, re, ca, ro, ct, cr, 10, 1e-4, 1e-6, 0.5, 0, 100000
        )
        best = scratch.to(dt)

    def slerp_wxyz(q0, q1, alpha):
        x0, y0, z0, w0 = q0[1], q0[2], q0[3], q0[0]
        x1, y1, z1, w1 = q1[1], q1[2], q1[3], q1[0]
        dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1
        s = torch.where(dot < 0, -1.0, 1.0)
        sx, sy, sz, sw = s * x1, s * y1, s * z1, s * w1
        c = (x0 * sx + y0 * sy + z0 * sz + w0 * sw).clamp(-1.0, 1.0)
        if float(c) > 0.9995:
            om = 1 - alpha
            rx, ry, rz, rw = (
                om * x0 + alpha * sx,
                om * y0 + alpha * sy,
                om * z0 + alpha * sz,
                om * w0 + alpha * sw,
            )
            n = torch.sqrt(rx * rx + ry * ry + rz * rz + rw * rw)
            return rx / n, ry / n, rz / n, rw / n
        theta = torch.acos(c)
        st = torch.sin(theta)
        w1s = torch.sin((1 - alpha) * theta) / st
        w2s = torch.sin(alpha * theta) / st
        return (
            w1s * x0 + w2s * sx,
            w1s * y0 + w2s * sy,
            w1s * z0 + w2s * sz,
            w1s * w0 + w2s * sw,
        )

    def frozen_step(w, t, r, alpha):
        trans = (1 - alpha) * t[0] + alpha * t[1]
        qx, qy, qz, qw = slerp_wxyz(r[0], r[1], alpha)
        v = w - trans
        cx, cy, cz, cw = -qx, -qy, -qz, qw
        vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
        t2, t3, t4 = cw * cx, cw * cy, cw * cz
        t5, t6, t7 = -cx * cx, cx * cy, cx * cz
        t8, t9, t10 = -cy * cy, cy * cz, -cz * cz
        sx = 2 * ((t8 + t10) * vx + (t6 - t4) * vy + (t3 + t7) * vz) + vx
        sy = 2 * ((t4 + t6) * vx + (t5 + t10) * vy + (t9 - t2) * vz) + vy
        sz = 2 * ((t7 - t3) * vx + (t2 + t9) * vy + (t5 + t8) * vz) + vz
        n = torch.sqrt(sx * sx + sy * sy + sz * sz)
        rx, ry, rz = sx / n, sy / n, sz / n
        elev = torch.atan2(rz, torch.sqrt(rx * rx + ry * ry))
        az = torch.atan2(ry, rx)
        return torch.stack([elev, az], dim=1)

    def fn_ref(w, t, r):
        return torch.cat(
            [frozen_step(w[i : i + 1], t, r, best[i]) for i in range(w.shape[0])],
            dim=0,
        )

    wpc = wp_seed.clone().requires_grad_(True)
    ctc = ct.clone().requires_grad_(True)
    crc = cr.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        fn_ref, (wpc, ctc, crc), eps=1e-6, atol=1e-5, rtol=1e-4
    )

    wpr = wp_seed.clone().requires_grad_(True)
    ctr = ct.clone().requires_grad_(True)
    crr = cr.clone().requires_grad_(True)
    sa_ref = fn_ref(wpr, ctr, crr)
    sa_ref.backward(upstream)
    assert (g_wp_cuda - wpr.grad).abs().max().item() < 1e-6
    assert (g_ct_cuda - ctr.grad).abs().max().item() < 1e-6
    assert (g_cr_cuda - crr.grad).abs().max().item() < 1e-6
