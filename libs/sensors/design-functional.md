# Sensors Functional Layer Design

## Overview

`libs/sensors/functional/` is the public stateless API for camera and
spinning-LiDAR projection. It is the import path downstream code should prefer
(`gsplat_sensors.functional` or `from gsplat_sensors import functional as F`).
The layer owns three things:

- the public function names and their argument contracts,
- the cross-family return-type dataclasses (`return_types.py`),
- the conversion of the raw `tuple[Tensor, ...]` returns from the kernel layer
  into those dataclasses.

Everything else — backend dispatch, autograd, native CUDA, parameter handles,
pose unpacking — lives under `libs/sensors/kernels/`. The model layer
(`libs/sensors/models/`) builds on top of the functional API for trainable /
stateful sensors and re-exports the same dataclasses unchanged.

The only shape transformation that happens inside `functional/` is constructing
`T_sensor_world` from per-point `(translation, rotation)` tensors via
`kernels/common/utils.poses_to_matrix`, which delegates to `libs/geometry`. The
functional layer never sees `(N, 4, 4)` view matrices on input.

## Scope

Camera operations re-exported from `cameras.py`:

- `camera_rays_to_image_points`
- `image_points_to_camera_rays`
- `project_world_points_mean_pose`
- `project_world_points_shutter_pose`
- `image_points_to_world_rays_static_pose`
- `image_points_to_world_rays_shutter_pose`
- `pixel_grid_to_world_rays_shutter_pose`
- `generate_image_points`

LiDAR operations re-exported from `lidars.py`:

- `sensor_rays_to_sensor_angles`
- `sensor_angles_to_sensor_rays`
- `elements_to_sensor_angles`
- `generate_spinning_lidar_rays`
- `inverse_project_spinning_lidar`

These are thin wrappers over `kernels/lidars/ops.py`. They accept a
`RowOffsetStructuredSpinningLidarProjection` where a projection is required and
package raw kernel tuples into `SensorAnglesReturn`, `SensorRayReturn`,
`WorldRaysReturn`, and `WorldPointsToSensorAnglesReturn`.

## Module Layout

```text
libs/sensors/functional/
  __init__.py
  cameras.py
  lidars.py
  return_types.py
  test_cameras.py
  test_lidars.py
```

The layer is intentionally flat. `cameras.py` and `lidars.py` are thin per-op
wrappers; `return_types.py` is the single home for the dataclasses;
`test_cameras.py` and `test_lidars.py` exercise the public API contract.

## Public API

Every camera op forwards directly to `kernels/cameras/ops`, then either returns
the kernel output unchanged (for single-tensor results) or wraps it in a
dataclass from `return_types.py`. None of the ops mutate inputs, none detach
gradients, and none introduce autograd. Differentiability is inherited
end-to-end from the kernel layer.

| Op | Inputs (at a glance) | Returns |
| -- | -------------------- | ------- |
| `camera_rays_to_image_points` | `(N, 3)` rays, `projection`, `external_distortion` | [`ImagePointsReturn`](#return-types) |
| `image_points_to_camera_rays` | `(N, 2)` image points, `projection`, `external_distortion` | bare `Tensor` `(N, 3)` |
| `project_world_points_mean_pose` | `(N, 3)` world points, `projection`, `external_distortion`, `resolution`, `DynamicPose` | [`WorldPointsToImagePointsReturn`](#return-types) |
| `project_world_points_shutter_pose` | `(N, 3)` world points, `projection`, `external_distortion`, `resolution`, `ShutterType`, `DynamicPose` (+ solver knobs) | [`WorldPointsToImagePointsReturn`](#return-types) |
| `image_points_to_world_rays_static_pose` | `(N, 2)` image points, `projection`, `external_distortion`, `Pose` | [`WorldRaysReturn`](#return-types) |
| `image_points_to_world_rays_shutter_pose` | `(N, 2)` image points, `projection`, `external_distortion`, `resolution`, `ShutterType`, `DynamicPose` | [`WorldRaysReturn`](#return-types) |
| `pixel_grid_to_world_rays_shutter_pose` | `projection`, `external_distortion`, `resolution`, `ShutterType`, `DynamicPose` | [`WorldRaysReturn`](#return-types) |
| `generate_image_points` | `resolution`, `device` | bare `Tensor` `(H, W, 2)` |

| LiDAR op | Inputs (at a glance) | Returns |
| -- | -------------------- | ------- |
| `sensor_rays_to_sensor_angles` | `(N, 3)` sensor-frame rays | [`SensorAnglesReturn`](#return-types) |
| `sensor_angles_to_sensor_rays` | `(N, 2)` `[elevation, azimuth]` | [`SensorRayReturn`](#return-types) |
| `elements_to_sensor_angles` | `(N, 2)` int `[row, col]`, `RowOffsetStructuredSpinningLidarProjection` | [`SensorAnglesReturn`](#return-types) |
| `generate_spinning_lidar_rays` | projection, optional elements, `DynamicPose` (+ optional timestamps) | [`WorldRaysReturn`](#return-types) |
| `inverse_project_spinning_lidar` | projection, `(N, 3)` world points, `DynamicPose` (+ solver knobs) | [`WorldPointsToSensorAnglesReturn`](#return-types) |

`pixel_grid_to_world_rays_shutter_pose` is a convenience wrapper: internally it
generates a pixel-center grid and dispatches to
`image_points_to_world_rays_shutter_pose`. `generate_image_points` is the only
op that does not require any sensor parameters and is also the only op that
permits a non-CUDA `device`; in that case it must be called with
`allow_device_transfer=True`.

`camera_rays_to_image_points` does not currently surface a `return_jacobians`
kwarg at the functional layer. The `jacobians` field on `ImagePointsReturn`
exists for the model layer, which fills it in via an autograd path in
`models/cameras/camera_model.py::CameraModel.camera_rays_to_image_points`.
The functional op always returns `jacobians=None`.

## Return Types

`return_types.py` is the single source of truth for sensor return dataclasses.
The model layer re-exports the same names; nothing redeclares them.

Two rules govern shape:

- single-output ops return a bare `Tensor` (no wrapper);
- multi-output ops return a frozen-shape dataclass whose required fields are
  `Tensor` and whose optional fields are `Tensor | None`, populated only when
  the matching `return_*` kwarg is set.

| Dataclass | Required fields | Optional fields (gated by kwargs) |
| --------- | --------------- | --------------------------------- |
| `ImagePointsReturn` | `image_points (N, 2)`, `valid_flag (N,)` | `jacobians (N, 2, 3)` |
| `WorldPointsToImagePointsReturn` | `image_points (N, 2)` | `T_sensor_world (N, 4, 4)`, `valid_flag (N,)`, `valid_indices (M,)`, `timestamps_us (N,)` |
| `WorldRaysReturn` | `world_rays (N, 6)` packed `(origin_xyz, direction_xyz)` | `T_sensor_world (N, 4, 4)`, `timestamps_us (N,)` |
| `PixelsReturn` | `pixels (N, 2) int`, `valid_flag (N,)` | — |
| `WorldPointsToPixelsReturn` | `pixels (N, 2) int` | `T_sensor_world`, `valid_flag`, `valid_indices`, `timestamps_us` (same as `WorldPointsToImagePointsReturn`) |
| `SensorAnglesReturn` | `sensor_angles (N, 2)` | `valid_flag (N,)` |
| `SensorRayReturn` | `sensor_rays (N, 3)` | `valid_flag (N,)` |
| `WorldPointsToSensorAnglesReturn` | `sensor_angles (N, 2)` | same optional set as `WorldPointsToImagePointsReturn` |

`PixelsReturn` and `WorldPointsToPixelsReturn` are owned here for
single-source-of-truth purposes but are not constructed by any functional op
today. `Pixels*` is populated by the model layer when callers want
pixel-rounded outputs. The `SensorAngles*` / `SensorRay*` types are populated
by the LiDAR functional surface.

`valid_flag`, `valid_indices`, and `timestamps_us` are non-differentiable (the
kernels emit them with no gradient). Every other field flows gradients normally.

## Pose and parameter conventions

Pose objects (`Pose`, `DynamicPose`, `Trajectory`) live in
`kernels/common/pose.py`. The functional layer accepts `Pose` for static-pose
ops and `DynamicPose` for the mean-pose and rolling-shutter ops; it forwards
the dataclass directly to the kernel Python wrapper, which is responsible for
unpacking it into the separate translation / rotation tensors expected by the
C++ launch (see `design-kernels.md`).

`projection` and `external_distortion` are not Python dataclasses. They are
TorchScript custom-class handles (`torch::class_<>`) registered by the C++
extension and surfaced through `kernels/cameras/types.py` and
`kernels/lidars/types.py` as `OpenCVPinholeProjection`, `FThetaProjection`,
`OpenCVFisheyeProjection`, `NoExternalDistortion`,
`BivariateWindshieldDistortion`, and
`RowOffsetStructuredSpinningLidarProjection`. The functional layer never
inspects their internals; it just passes them through.

`external_distortion` is always an explicit instance — pass
`NoExternalDistortion()` for the identity case. There is no `None`-permissive
overload.

`T_sensor_world`, when requested, is constructed inside the functional op via
`kernels/common/utils.poses_to_matrix`, which delegates to `libs/geometry`.
Kernels themselves never return `(N, 4, 4)` matrices.

## `allow_device_transfer` contract

Every functional op accepts `allow_device_transfer: bool = False` as a
keyword-only argument. The default is strict: any input tensor that is not
already on the kernel-required CUDA device (and dtype) causes a `RuntimeError`
before launch. Setting it to `True` permits a one-time implicit `.to(...)` at
the cost of a host/device synchronization. The flag is forwarded to the kernel
wrapper unchanged; the functional layer does not perform its own device
checking.

`generate_image_points` is the one place this flag also gates a non-CUDA
output `device`.

## Naming and API Semantics

The public op names describe sensor operations (what they do), not backend
dispatch (which type they dispatch to). There is no projection/distortion suffix
at this layer; OpenCV pinhole, FTheta, and OpenCV fisheye all use the same
public names and select their backend through `(projection,
external_distortion)` pairs in the kernel dispatch table.

The conventional verbs in this API:

- `camera_rays_to_image_points` / `image_points_to_camera_rays` — operate
  purely in camera space (no pose).
- `project_world_points_*` — world → image, with pose handling indicated by
  the suffix (`_mean_pose` for global / averaged, `_shutter_pose` for
  rolling-shutter compensation).
- `image_points_to_world_rays_*` / `pixel_grid_to_world_rays_*` — image →
  world rays, again with pose handling in the suffix.
- `generate_image_points` — no sensor parameters, just a pixel-center grid.
- `sensor_rays_to_sensor_angles` / `sensor_angles_to_sensor_rays` — operate
  purely in LiDAR sensor space.
- `elements_to_sensor_angles` — maps structured LiDAR `[row, col]` elements to
  `[elevation, azimuth]` using projection tables.
- `generate_spinning_lidar_rays` / `inverse_project_spinning_lidar` — LiDAR
  rolling-shutter world-ray generation and inverse projection with a
  `DynamicPose`.

## Testing Structure

`functional/test_cameras.py` and `functional/test_lidars.py` cover the public
API surface end-to-end:
return-type construction, optional-flag behavior (only the requested
optional fields are populated), the `allow_device_transfer=False` strict
device check, consistency with the underlying kernel op for representative
cases, and gradient flow through the dataclass tensor fields.

Per-op kernel correctness, autograd backward correctness against finite
differences, and edge-case numerical coverage live one layer down in the
kernel test files (see `design-kernels.md`). The functional tests should not
duplicate that coverage.

## Design Constraints

- The cross-family return dataclasses are owned by `return_types.py` and
  re-exported from `__init__.py`. No other layer redeclares them.
- The functional layer never registers a `torch.autograd.Function`, never
  detaches tensors, and never converts to numpy. Differentiability is
  inherited from the kernel layer.
- Pose unpacking happens in the kernel Python wrapper, not here. Functional
  ops accept `Pose` or `DynamicPose` only.
- `external_distortion` is always an explicit instance (typically
  `NoExternalDistortion()`), never `None`.
- Single-tensor outputs stay as bare `Tensor` returns. Wrappers exist only
  when there is more than one output or when an output is conditional on a
  `return_*` kwarg.
- The functional layer has no dispatch table and no per-projection branches.
  Backend dispatch is a kernel-layer concern.
- `T_sensor_world` is constructed in the functional op via
  `poses_to_matrix`; kernels return `(translation, rotation)` tensors.
- Op names describe the sensor operation, not the backend dispatch suffix.
