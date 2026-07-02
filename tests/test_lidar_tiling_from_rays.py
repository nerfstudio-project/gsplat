# SPDX-License-Identifier: Apache-2.0
"""Data-driven lidar tiling + generalized angles map — SDCMLRND-1510.

Self-contained tests for the pure-python gsplat patch that supports
non-separable scan patterns (galvo/prism lidars, e.g. lidarsky):

1. refactor regression — ``compute_angles_to_columns_map`` is bit-identical to
   the pre-patch implementation (reference copy inlined below);
2. ``compute_angles_to_values_map`` — column-map wrapper consistency, dtype
   range guard;
3. ``clamp_element_angles_to_fov`` — arbitrary (also far-out-of-FOV) angles land
   strictly inside the FOV after the relative->absolute round-trip, in-FOV
   angles are preserved;
4. tiling with ``element_angles``: full coverage, placement by REAL angles, the
   idealized-grid misplacement (the hole cause) is demonstrated and fixed,
   idealized angles reproduce the default placement (up to the documented
   FOV-edge clamp-vs-wrap);
5. time-encoded map: shutter times looked up the same way as
   ``Lidars.cuh::shutter_relative_frame_time`` match a row-major (galvo) scan
   and beat the column-linear spinning assumption by an order of magnitude.

Only the new patch is exercised; no dependency on the other test modules.
"""

from __future__ import annotations

import math

import pytest
import torch

from gsplat.cuda._lidar import (
    RowOffsetStructuredSpinningLidarModelParameters,
    SpinningDirection,
    angles_to_tile_indices,
    clamp_element_angles_to_fov,
    compute_angles_to_columns_map,
    compute_angles_to_values_map,
    compute_tiling,
    relative_sensor_angles,
    sensor_angles_to_rays,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_ROWS = 8
N_COLS = 64
TILING_KW = dict(
    n_bins_elevation=4,
    max_pts_per_tile=16,
    resolution_elevation=200,
    densification_factor_azimuth=2,
)


def _make_params(device: str = DEVICE) -> RowOffsetStructuredSpinningLidarModelParameters:
    # Descending elevations and CW-monotonic (decreasing) azimuths, partial FOV (< 2π).
    el = torch.linspace(0.20, -0.20, N_ROWS, dtype=torch.float32, device=device)
    az = torch.linspace(1.00, -1.00, N_COLS, dtype=torch.float32, device=device)
    off = torch.zeros(N_ROWS, dtype=torch.float32, device=device)
    return RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=el,
        column_azimuths_rad=az,
        row_azimuth_offsets_rad=off,
        spinning_direction=SpinningDirection.CLOCKWISE,
        spinning_frequency_hz=10.0,
    )


def _make_params_full_circle(device: str = DEVICE) -> RowOffsetStructuredSpinningLidarModelParameters:
    # Near-2π column grid + row offsets that make the effective FOV wrap → span == 2π,
    # like a real spinning lidar (the mlrnd Hesai path).
    margin = 1e-3
    el = torch.linspace(0.20, -0.20, N_ROWS, dtype=torch.float32, device=device)
    az = torch.linspace(
        math.pi / 2 - margin,
        math.pi / 2 - 2 * math.pi + margin,
        N_COLS,
        dtype=torch.float32,
        device=device,
    )
    off = torch.linspace(-0.01, 0.01, N_ROWS, dtype=torch.float32, device=device)
    return RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=el,
        column_azimuths_rad=az,
        row_azimuth_offsets_rad=off,
        spinning_direction=SpinningDirection.CLOCKWISE,
        spinning_frequency_hz=10.0,
    )


def _ideal_element_angles(p) -> torch.Tensor:
    """[n_rows, n_columns, 2] idealized (azimuth, elevation) of each element."""
    return p.elements_to_sensor_angles(p.create_elements()).reshape(p.n_rows, p.n_columns, 2)


def _nonseparable_element_angles(p) -> torch.Tensor:
    """Idealized angles + per-row azimuth skew and per-column elevation ripple.

    Mimics a galvo/prism scan: the real per-pixel angle field does not factorize
    into row-elevation + column-azimuth, so rays cross tile boundaries (and the
    edge rows/columns stick out of the fitted FOV, exercising the clamp).
    """
    base = _ideal_element_angles(p).clone()
    span_az = p.fov_horiz_rad.span
    span_el = p.fov_vert_rad.span

    rows = torch.arange(p.n_rows, device=base.device, dtype=torch.float32)
    cols = torch.arange(p.n_columns, device=base.device, dtype=torch.float32)

    # Per-row azimuth skew (grows toward the top/bottom rings), several tile widths wide.
    row_skew = (rows / (p.n_rows - 1) - 0.5) * (0.12 * span_az)  # [n_rows]
    # Per-column elevation ripple (the galvo's within-ring elevation drift).
    col_ripple = torch.sin(2 * torch.pi * cols / p.n_columns) * (0.08 * span_el)  # [n_columns]

    base[..., 0] = base[..., 0] + row_skew[:, None]
    base[..., 1] = base[..., 1] + col_ripple[None, :]
    return base


def _tile_of_each_element(tiling, n_rows: int, n_cols: int, device) -> torch.Tensor:
    """Invert ``tiles_pack_info`` + ``tiles_to_elements_map`` to a [n_rows*n_cols]
    tile id per element (flat index = el_idx * n_cols + az_idx). -1 = unassigned."""
    tile_of = torch.full((n_rows * n_cols,), -1, dtype=torch.long, device=device)
    tte = tiling.tiles_to_elements_map.to(device)
    pack = tiling.tiles_pack_info.to(device)
    for t in range(pack.shape[0]):
        off, cnt = int(pack[t, 0]), int(pack[t, 1])
        if cnt == 0:
            continue
        elems = tte[off : off + cnt]  # (cnt, 2) = (az_idx, el_idx)
        flat = elems[:, 1].long() * n_cols + elems[:, 0].long()
        tile_of[flat] = t
    return tile_of


def _expected_tile(p, tiling, angles) -> torch.Tensor:
    """Tile id each angle maps to in ``tiling``'s grid (FOV-clamped like the patch does)."""
    clamped = clamp_element_angles_to_fov(p, angles.reshape(-1, 2))
    return angles_to_tile_indices(
        p,
        clamped,
        n_bins_azimuth=tiling.n_bins_azimuth,
        n_bins_elevation=tiling.n_bins_elevation,
        cdf_elevation=tiling.cdf_elevation,
    ).long()


# --------------------------------------------------------------------------
# 1. Refactor regression: the columns map is bit-identical to the old code.
# --------------------------------------------------------------------------


def _reference_angles_to_columns_map(lidar, resolution_factor=4, dtype=torch.int32):
    """Verbatim pre-patch implementation of compute_angles_to_columns_map."""
    grid_elevations_rad, grid_azimuths_rad = torch.meshgrid(
        torch.linspace(
            lidar.fov_vert_rad.start,
            lidar.fov_vert_rad.end,
            resolution_factor * lidar.n_rows,
            device=lidar.row_elevations_rad.device,
            dtype=lidar.row_elevations_rad.dtype,
        ),
        torch.linspace(
            lidar.fov_horiz_rad.start,
            lidar.fov_horiz_rad.end,
            resolution_factor * lidar.n_columns,
            device=lidar.column_azimuths_rad.device,
            dtype=lidar.column_azimuths_rad.dtype,
        ),
        indexing="ij",
    )
    grid_angles = torch.stack([grid_azimuths_rad, grid_elevations_rad], dim=-1)
    grid_rays = sensor_angles_to_rays(lidar, grid_angles)

    elements = lidar.create_elements()
    elem_az = elements[..., 0]
    elem_el = elements[..., 1]
    raw_elevation = lidar.row_elevations_rad[elem_el]
    raw_azimuth = lidar.column_azimuths_rad[elem_az] + lidar.row_azimuth_offsets_rad[elem_el]
    raw_azimuth = torch.where(raw_azimuth > torch.pi, raw_azimuth - 2 * torch.pi, raw_azimuth)
    raw_azimuth = torch.where(raw_azimuth <= -torch.pi, raw_azimuth + 2 * torch.pi, raw_azimuth)
    sensor_angles = torch.stack([raw_azimuth, raw_elevation], dim=-1).reshape(-1, 2)
    sensor_rays = sensor_angles_to_rays(lidar, sensor_angles)

    from scipy import spatial as scipy_spatial

    kdtree = scipy_spatial.cKDTree(sensor_rays.sensor_rays.contiguous().cpu().numpy())
    _, idxs = kdtree.query(grid_rays.sensor_rays.contiguous().cpu().numpy())
    idxs = torch.from_numpy(idxs).to(device=lidar.device, dtype=torch.int32)
    return (idxs % lidar.n_columns).to(dtype).reshape(grid_angles.shape[:-1])


@pytest.mark.parametrize("make_params", [_make_params, _make_params_full_circle])
def test_columns_map_matches_prepatch_reference(make_params):
    p = make_params()
    new = compute_angles_to_columns_map(p, resolution_factor=4)
    ref = _reference_angles_to_columns_map(p, resolution_factor=4)
    assert new.dtype == ref.dtype and new.shape == ref.shape
    assert torch.equal(new.cpu(), ref.cpu())


# --------------------------------------------------------------------------
# 2. compute_angles_to_values_map basics.
# --------------------------------------------------------------------------


def test_values_map_with_column_values_equals_columns_map():
    p = _make_params()
    ideal = _ideal_element_angles(p).reshape(-1, 2)
    rays = sensor_angles_to_rays(p, ideal).sensor_rays
    columns = torch.arange(ideal.shape[0], dtype=torch.int64) % p.n_columns
    via_values = compute_angles_to_values_map(p, rays, columns, resolution_factor=4)
    direct = compute_angles_to_columns_map(p, resolution_factor=4)
    assert torch.equal(via_values.cpu(), direct.cpu())


def test_values_map_rejects_values_outside_dtype_range():
    p = _make_params()
    ideal = _ideal_element_angles(p).reshape(-1, 2)
    rays = sensor_angles_to_rays(p, ideal).sensor_rays
    too_big = torch.full((ideal.shape[0],), 2**40, dtype=torch.int64)
    with pytest.raises(AssertionError):
        compute_angles_to_values_map(p, rays, too_big, dtype=torch.int32)


# --------------------------------------------------------------------------
# 3. clamp_element_angles_to_fov.
# --------------------------------------------------------------------------


def test_clamp_lands_strictly_inside_fov_for_arbitrary_angles():
    p = _make_params()
    gen = torch.Generator().manual_seed(0)
    az = (torch.rand(5000, generator=gen) - 0.5) * 2 * math.pi
    el = (torch.rand(5000, generator=gen) - 0.5) * math.pi
    # Include the exact FOV corners and far-out points.
    extremes = torch.tensor(
        [
            [p.fov_horiz_rad.start, p.fov_vert_rad.start],
            [p.fov_horiz_rad.end, p.fov_vert_rad.end],
            [p.fov_horiz_rad.start + 0.5, p.fov_vert_rad.start + 0.5],
            [p.fov_horiz_rad.end - 0.5, p.fov_vert_rad.end - 0.5],
        ]
    )
    angles = torch.cat([torch.stack([az, el], dim=-1), extremes]).to(DEVICE)

    out = clamp_element_angles_to_fov(p, angles)
    rel = relative_sensor_angles(p, out)
    span_az, span_el = p.fov_horiz_rad.span, p.fov_vert_rad.span
    # Strictly inside the FOV — this is exactly what the downstream tiling
    # asserts / `% n_bins` quantization require.
    assert float(rel[..., 0].min()) >= 0.0 and float(rel[..., 0].max()) <= span_az
    assert float(rel[..., 1].min()) >= 0.0 and float(rel[..., 1].max()) <= span_el


def test_clamp_preserves_in_fov_angles():
    p = _make_params()
    ideal = _ideal_element_angles(p).reshape(-1, 2)
    rel_in = relative_sensor_angles(p, ideal)
    out = clamp_element_angles_to_fov(p, ideal)
    rel_out = relative_sensor_angles(p, out)
    # Interior angles move by at most the clamp margin (1e-5 rad).
    assert torch.allclose(rel_out, rel_in, atol=3e-5)


# --------------------------------------------------------------------------
# 4. Data-driven tiling.
# --------------------------------------------------------------------------


def test_every_element_placed_once_and_in_its_real_tile():
    p = _make_params()
    real = _nonseparable_element_angles(p)
    tiling = compute_tiling(p, element_angles=real, **TILING_KW)

    # Coverage: the map is a permutation of all elements (no ray lost / duplicated).
    assert tiling.tiles_to_elements_map.shape[0] == p.n_rows * p.n_columns
    tile_of = _tile_of_each_element(tiling, p.n_rows, p.n_columns, real.device)
    assert torch.all(tile_of >= 0), "some element was not assigned to any tile"
    assert int(tiling.tiles_pack_info[:, 1].sum()) == p.n_rows * p.n_columns

    # Each element sits in exactly the tile its REAL angle maps to.
    expected = _expected_tile(p, tiling, real)
    assert torch.equal(tile_of, expected)


def test_idealized_tiling_misplaces_real_rays_but_real_tiling_fixes_it():
    p = _make_params()
    real = _nonseparable_element_angles(p)

    ideal_tiling = compute_tiling(p, **TILING_KW)  # element_angles=None (old behavior)
    real_tiling = compute_tiling(p, element_angles=real, **TILING_KW)

    # Where the idealized tiling PLACED each element (by its idealized angle) ...
    ideal_placement = _tile_of_each_element(ideal_tiling, p.n_rows, p.n_columns, real.device)
    # ... vs where each element's REAL ray angle maps in the SAME idealized grid.
    real_in_ideal_grid = _expected_tile(p, ideal_tiling, real)

    # The bug is real: some real rays fall in a different tile than their element's
    # slot in the idealized tiling → those rays never test the covering gaussians → holes.
    n_mismatch = int((ideal_placement != real_in_ideal_grid).sum())
    assert n_mismatch > 0, "perturbation too small to exercise the tiling mismatch"

    # The fix: with real-angle tiling every element sits in the tile its real angle maps to.
    real_placement = _tile_of_each_element(real_tiling, p.n_rows, p.n_columns, real.device)
    real_expected = _expected_tile(p, real_tiling, real)
    assert torch.equal(real_placement, real_expected)


def test_feeding_idealized_angles_matches_the_idealized_tiling():
    """element_angles == idealized angles reproduces the default (None) placement.

    The tile grid (n_bins, cdf_elevation) is identical, and per-element placement
    matches everywhere except possibly the last azimuth column: there the
    idealized element sits exactly on the FOV azimuth end (rel_az == span), where
    the default path wraps via ``% n_bins`` while the clamped path lands in the
    boundary tile (consistent with how gaussians are binned for a partial FOV).
    """
    p = _make_params()
    ideal_angles = _ideal_element_angles(p)

    default = compute_tiling(p, **TILING_KW)
    via_angles = compute_tiling(p, element_angles=ideal_angles, **TILING_KW)

    assert via_angles.n_bins_azimuth == default.n_bins_azimuth
    assert torch.equal(via_angles.cdf_elevation, default.cdf_elevation)

    default_place = _tile_of_each_element(default, p.n_rows, p.n_columns, ideal_angles.device)
    via_place = _tile_of_each_element(via_angles, p.n_rows, p.n_columns, ideal_angles.device)

    flat = torch.arange(p.n_rows * p.n_columns, device=ideal_angles.device)
    az_idx = flat % p.n_columns
    diff = default_place != via_place
    # Only the FOV azimuth-boundary column may differ (clamp vs modulo wrap).
    assert torch.all(az_idx[diff] == p.n_columns - 1)


def test_tiling_survives_far_out_of_fov_angles():
    p = _make_params()
    real = _nonseparable_element_angles(p).reshape(-1, 2).clone()
    # Push edge rows way outside the fitted FOV (elevation) and swing azimuths out.
    real[: p.n_columns, 1] += 0.3
    real[-p.n_columns :, 1] -= 0.3
    real[:: p.n_columns, 0] += 0.5

    tiling = compute_tiling(p, element_angles=real, **TILING_KW)
    tile_of = _tile_of_each_element(tiling, p.n_rows, p.n_columns, real.device)
    assert torch.all(tile_of >= 0)
    assert int(tiling.tiles_pack_info[:, 1].sum()) == p.n_rows * p.n_columns


# --------------------------------------------------------------------------
# 5. Time-encoded angles map (rolling shutter for row-sequential scans).
# --------------------------------------------------------------------------


def test_time_map_encodes_row_major_scan_times():
    p = _make_params()
    real = _nonseparable_element_angles(p).reshape(-1, 2)
    n = p.n_rows * p.n_columns

    # Row-major emission times, like a galvo scan: t grows along the row, then ring by ring.
    t_rel = torch.arange(n, dtype=torch.float64) / (n - 1)
    values = torch.round(t_rel * (p.n_columns - 1)).to(torch.int64)
    rays = sensor_angles_to_rays(p, real).sensor_rays
    amap = compute_angles_to_values_map(p, rays, values, resolution_factor=8)

    # Emulate the map's only consumer, Lidars.cuh::shutter_relative_frame_time:
    # nearest grid node by relative angles (map_resolution = span/(dim-1),
    # index = clamp(rel/resolution + 0.5)), then value/(n_columns-1).
    rel = relative_sensor_angles(p, clamp_element_angles_to_fov(p, real)).cpu().double()
    res_el, res_az = amap.shape
    step_az = p.fov_horiz_rad.span / (res_az - 1)
    step_el = p.fov_vert_rad.span / (res_el - 1)
    ihoriz = torch.clamp((rel[..., 0] / step_az + 0.5).long(), 0, res_az - 1)
    ivert = torch.clamp((rel[..., 1] / step_el + 0.5).long(), 0, res_el - 1)
    t_from_map = amap.cpu()[ivert, ihoriz].double() / (p.n_columns - 1)

    err = (t_from_map - t_rel).abs()
    # Bounded by nearest-ray quantization: at worst the ring-adjacent element,
    # i.e. ~1/n_rows of the frame, plus the LUT value quantization.
    assert float(err.max()) <= 1.5 / p.n_rows
    assert float(err.mean()) <= 0.05

    # The column-linear spinning model (t = column/(n_columns-1), what the map
    # used to encode) is grossly wrong for a row-sequential scan.
    t_column_model = (torch.arange(n, dtype=torch.float64) % p.n_columns) / (p.n_columns - 1)
    err_col = (t_column_model - t_rel).abs()
    assert float(err_col.mean()) >= 0.25
    assert float(err.mean()) <= 0.25 * float(err_col.mean())
