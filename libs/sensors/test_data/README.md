# Sensors Test Data

`test_pinhole_camera_params.json` contains a synthetic 5-camera OpenCV pinhole
AV rig fixture (`sequence_id: synthetic_av_rig_v1`).

Rig layout (rig frame: X=forward, Y=left, Z=up; camera frame: OpenCV
X=right, Y=down, Z=forward):

| Camera             | Resolution  | Yaw   | Position (m)         |
|--------------------|-------------|-------|----------------------|
| `camera_front`     | 1920 x 1280 |   0°  | ( 1.54,  0.00, 2.12) |
| `camera_front_left`| 1920 x 1280 | +45°  | ( 1.49, +0.09, 2.12) |
| `camera_front_right`| 1920 x 1280| -45°  | ( 1.49, -0.09, 2.12) |
| `camera_side_left` | 1920 x 886  | +90°  | ( 1.43, +0.11, 2.12) |
| `camera_side_right`| 1920 x 886  | -90°  | ( 1.43, -0.11, 2.12) |

`test_ftheta_camera_params.json` contains six F-Theta camera intrinsics from a
representative NVIDIA capture sequence (camera_cross_left/right_120fov,
camera_front_tele_30fov, camera_front_wide_120fov, camera_rear_left/right_70fov).
The wide camera carries a bivariate-windshield external distortion; the rest
are no-external. Field name mapping when consumed by gsplat F-Theta tests:
`pixeldist_to_angle_poly` → `bw_poly`, `angle_to_pixeldist_poly` → `fw_poly`,
`linear_cde` → `A` (as `[c, d, e, 1]` flat 2x2), `reference_poly: "2"` →
`ReferencePolynomial.FORWARD` (use angle_to_pixeldist directly in forward).

`test_fisheye_camera_params.json` contains OpenCV fisheye camera intrinsics
used by model-level real-camera tests. The `intrinsics.forward_poly` field maps
to `OpenCVFisheyeProjection.forward_poly`; `max_angle_rad` maps to the
projection `max_angle`.

## Spinning-LiDAR test data

LiDAR fixtures here are **sensor-parameter JSONs** — the real calibration
geometry of two production sensors. The kernel tests build their oracles
directly from these tables at runtime (no stored expected-result bundles).

### Sensor-parameter JSONs (real calibration)

`row-offset-spinning-lidar-model-parameters*.json` each hold the fixed geometry
of one real row-offset spinning LiDAR. Every file carries the same keys —
`n_rows`, `n_columns`, `row_elevations_rad`, `column_azimuths_rad`,
`row_azimuth_offsets_rad`, `spinning_direction`, and `spinning_frequency_hz`
(the frequency is present but unused by the kernels). The arrays are sized
`n_rows` (elevations, offsets) and `n_columns` (azimuths). Both sensors spin
clockwise.

| Config (file suffix)        | n_rows | n_columns | spinning_direction | row offsets | elevation range (rad) |
|-----------------------------|--------|-----------|--------------------|-------------|-----------------------|
| `generic` (none)            | 128    | 3600      | cw                 | nonzero     | -0.435652 … +0.252200 |
| `waymo`                     | 64     | 2650      | cw                 | all-zero    | -0.309257 … +0.038489 |

The `generic` config is the unsuffixed file
(`row-offset-spinning-lidar-model-parameters.json`). The two configs differ in
the `has_row_offsets` kernel branch (`generic` carries per-row azimuth offsets,
`waymo` has none) and in table size, exercising both offset paths against real
calibration geometry.

### How the tests consume them

The JSONs are loaded live by the `lidar_projection_from_json` / `lidar_model`
fixtures (`conftest.py`). The five kernel ops are verified against oracles
derived from the loaded tables at runtime:

- **ray ↔ angle**: analytic cardinal cases, unit-norm rays, the
  `angles → rays → angles` round-trip, and fp64 `gradcheck`.
- **elements → angle**: an exact match to a direct table gather (elevation =
  `row_elevations_rad[row]`, azimuth = `column_azimuths_rad[col] +
  row_azimuth_offsets_rad[row]`) across both sensors, plus out-of-bounds
  validity and atomicAdd grad accumulation.
- **generate**: the static-pose origin/direction oracle (origins ==
  translation, directions == `R(q)` · sensor ray) across both sensors, the
  `elements=None` grid path, rolling-shutter timestamp ordering, and fp64
  `gradcheck` over tables + control poses.
- **inverse**: the `generate → world → inverse` round-trip recovery (≥ 98 % of
  the table angles among inverse-valid points), out-of-FOV validity,
  `max_iterations` range checking, and the fp64 IFT VJP for world points + both
  control poses.

Control rotations are passed to `Pose` in `wxyz` order; the inverse path uses
the linear nearest-column fallback.

These oracles verify the kernels' internal analytic, round-trip, and gradient
correctness; they do not assert bit-parity against an external reference
implementation.
