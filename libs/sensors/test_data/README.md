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
