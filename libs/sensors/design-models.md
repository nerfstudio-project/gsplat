# Sensors Models Layer Design

## Overview

`libs/sensors/models/` is an `nn.Module`-shaped layer on top of the stateless
public API in `libs/sensors/functional/`. It is optional: callers that only
need stateless projection ops can stay on `functional/`.

The model layer packages projection, external distortion, resolution, and
shutter type into a single `CameraModel(nn.Module)`; carries observations as
`nn.Module` buffers in frame containers so `.to(...)` and `state_dict` work
as expected; composes the functional ops into a method surface keyed by
`(domain pair, pose flavour)`; and forwards `Pose | DynamicPose` through to
`functional/` without unpacking it itself.

Return dataclasses live in `functional/return_types.py` and pose dataclasses
in `kernels/common/pose.py`; both are re-exported here for ergonomics. LiDAR
models and a rendering-facing sensor-parameter aggregator are out of scope
today; `models/lidars/` is an empty placeholder, and rendering callers
convert their own `viewmats` to `Pose | DynamicPose` upstream of this layer.

## Scope

The models layer currently covers:

- `Frame` base class and the `FrameId = str` type alias.
- `ImageFrame` and `ImageFrameGroup` (`dict[FrameId, ImageFrame]`).
- A single concrete `CameraModel` wrapping an `OpenCVPinholeProjection`
  together with `NoExternalDistortion` or `BivariateWindshieldDistortion`.
- Helpers in `models/common/utils.py` for resolution scaling, validity
  filtering, SE(3) matrix construction, and quaternion-order swizzles.
- Re-exports of the return dataclasses, the kernel-level parameter handles
  needed to construct a model (`OpenCVPinholeProjection`, the distortion
  types, `ReferencePolynomial`, `ShutterType`, `from_components`,
  `CameraProjection`, `ExternalDistortion`), and the pose dataclasses
  (`Pose`, `DynamicPose`, `Trajectory`).

Not covered today: LiDAR models and any rendering-facing aggregator that
bundles `viewmats` / `viewmats_rs`.

## Module Layout

```text
libs/sensors/models/
  __init__.py
  cameras/
    __init__.py
    camera_model.py             # CameraModel (concrete nn.Module)
    image_frame.py              # ImageFrame, ImageFrameGroup
    test_camera_model.py
    test_camera_model_opencv_pinhole.py
  common/
    __init__.py
    frame.py                    # Frame base, FrameId
    utils.py                    # compute_scaled_resolution, filter_by_validity, re-exports
    test_frame.py
    test_utils.py
  lidars/
    __init__.py                 # empty placeholder; __all__ = []
```

## Frame and ImageFrame

`Frame` (`models/common/frame.py`) is an `nn.Module` base for sensor
observations. Constructor arguments: `frame_id: FrameId`,
`pose: Pose | DynamicPose`, `timestamp_start_us: int`,
`timestamp_end_us: int`, and optional `metadata: dict[str, Any]`. `forward()`
is abstract; data-container subclasses raise `NotImplementedError`. Two base
properties: `is_rolling_shutter` (`start != end`) and `frame_duration_us`
(`end - start`). `Frame._apply` propagates device / dtype transfers to the
underlying `Pose` / `DynamicPose` tensors (plain attributes, not
`nn.Parameter`s, so `nn.Module._apply` would otherwise miss them).

`ImageFrame` (`models/cameras/image_frame.py`) extends `Frame` with a
`camera_model: CameraModel` attribute and an `image: Tensor` buffer of shape
`(H, W, C)` registered via `self.register_buffer("image", ...)`. It exposes
`height`, `width`, and `channels` properties from the buffer shape.
`ImageFrameGroup` is a typed alias `dict[FrameId, ImageFrame]`.

## CameraModel

`CameraModel` (`models/cameras/camera_model.py`) is a concrete `nn.Module`. A
single class covers all currently supported projection types (OpenCV pinhole
today); new projection families are added by extending the dispatch helpers
`_move_projection` / `_move_external_distortion` used by the custom `_apply`
override, not by subclassing.

The constructor takes a non-`None` `projection` (any TorchScript projection,
`OpenCVPinholeProjection` today), a non-`None` `external_distortion`
(`NoExternalDistortion` or `BivariateWindshieldDistortion`), a `resolution`
of `(width, height)`, and a `shutter_type`. `None` for either parameter
raises `TypeError`. The projection is held privately as `_projection` and
exposed via a read-only `.projection` property. `_apply` is overridden so
`.to(...)` / `.cuda()` also move the TorchScript projection and distortion
structs (the C++ class instances are not seen by `nn.Module._apply`).

```python
camera = CameraModel(
    projection=OpenCVPinholeProjection(...),
    external_distortion=NoExternalDistortion(),
    resolution=(1920, 1080),
    shutter_type=ShutterType.GLOBAL,
)
```

Public methods are grouped by domain pair, with static-, mean-, and
shutter-pose variants where pose is relevant. The `*_shutter_pose` flavours
expose `max_iterations`, `stop_mean_error_px`, `stop_delta_mean_error_px`,
and `initial_relative_time` as part of the public signature.

| Group | Methods | Returns |
| --- | --- | --- |
| World points → image points | `world_points_to_image_points_{static,mean,shutter}_pose` | `WorldPointsToImagePointsReturn` |
| World points → pixels | `world_points_to_pixels_{static,mean,shutter}_pose` | `WorldPointsToPixelsReturn` |
| Image points → world rays | `image_points_to_world_rays_{static,mean,shutter}_pose` | `WorldRaysReturn` |
| Pixels → world rays | `pixels_to_world_rays_{static,mean,shutter}_pose` | `WorldRaysReturn` |
| Camera rays ↔ image points | `camera_rays_to_image_points`, `image_points_to_camera_rays` | `ImagePointsReturn` / `Tensor` |
| Camera rays ↔ pixels | `camera_rays_to_pixels`, `pixels_to_camera_rays` | `PixelsReturn` / `Tensor` |
| Pixel ↔ image point | `pixels_to_image_points`, `image_points_to_pixels` | `Tensor` |
| Shutter timing | `image_points_relative_frame_times` | `Tensor` |
| Image-domain transform | `transform` | new `CameraModel` |

Pixel ↔ image-point conversions are `pixel.float() + 0.5` and `floor → int32`.
The static-pose world-point flavour lifts `Pose` to `DynamicPose` via
`DynamicPose.from_static_pose` and reuses `F.project_world_points_mean_pose`
with both timestamp endpoints equal, so a single functional path covers both
cases. The `image_points_to_world_rays_mean_pose` variant interpolates the
dynamic pose at relative time `0.5` and delegates to the static
back-projection.

## Image-domain transform

`CameraModel.transform(image_domain_scale, image_domain_offset=(0.0, 0.0),
new_resolution=None)` returns a new `CameraModel` whose projection has been
rescaled and shifted. The Python method composes `compute_scaled_resolution(...)`
with `self._projection.transform(...)`; the latter is the C++-side `transform`
method registered on `OpenCVPinholeProjection`. For OpenCV pinhole the rule is
`new_focal_length = focal_length * scale`,
`new_principal_point = principal_point * scale - offset`, and the distortion
coefficients (radial, tangential, thin-prism) are cloned unchanged (they live
in normalized image coordinates). `external_distortion` and `shutter_type`
are preserved on the new model; the original is not mutated.

## `return_jacobians` on `camera_rays_to_image_points`

With `return_jacobians=False` (default) the method forwards directly to the
functional op and `image_points` is differentiable w.r.t. `camera_rays`. With
`return_jacobians=True` it runs the projection on a detached `requires_grad`
leaf and computes the `(N, 2, 3)` Jacobian via `torch.autograd.grad`; the
returned `image_points` is detached on this path so the Jacobian's autograd
graph does not leak back into the caller's `camera_rays`. Callers that need
both a differentiable value path and the Jacobian should call the method
twice.

## Validity filtering and `return_*` flags

Model methods filter image points, per-point `T_sensor_world`, and per-point
timestamps to the valid subset by default; `return_all_projections=True`
skips the filter and surfaces one row per input point (internally implemented
via `filter_by_validity(...)`, which is a no-op when the mask is `None` or
`return_all_projections=True`). The other `return_*` flags
(`return_T_sensor_world`, `return_valid_flag`, `return_valid_indices`,
`return_timestamps`) gate whether the matching optional field on the return
dataclass is populated; otherwise the field is `None`.
`return_valid_indices=True` materializes the int64 index tensor via
`valid_flags_to_indices`.

## Pose handling

The model layer takes `Pose | DynamicPose` directly and never unpacks pose
to separated tensors — that happens one layer down in `kernels/cameras/ops.py`
just before the C++ launch (see `design-kernels.md`). The quaternion
convention at the public Python API is **wxyz**; the wxyz↔xyzw reorder for
`libs/geometry`'s SE(3) helpers lives in `kernels/common/utils.poses_to_matrix`.
Gradients flow through `pose.translation` / `pose.rotation` when those
tensors are `requires_grad=True`.

## Helper utilities (`models/common/utils.py`)

- `compute_scaled_resolution(original_resolution, scale, new_resolution=None)`
  returns the new `(width, height)`; an explicit `new_resolution` overrides
  the scale, otherwise the original is multiplied by `scale` (isotropic
  `float` or anisotropic `(sx, sy)`) and rounded to `int`.
- `filter_by_validity(data, valid_flags, return_all)` boolean-masks rows of
  `(N, ...)`; returns the input unchanged when `data` or `valid_flags` is
  `None`, or when `return_all=True`.
- `poses_to_matrix`, `valid_flags_to_indices`, `wxyz_to_xyzw`,
  `xyzw_to_wxyz` are re-exported from `kernels/common/utils.py`.

## Testing Structure

Tests live next to the modules they exercise:

- `models/cameras/test_camera_model.py` — cross-projection model behaviour:
  `nn.Module` semantics, `.to(device)` and `torch.save` round-trips,
  `ImageFrame` buffer registration and device propagation, and the
  validity-filtering / `return_*` flag contract.
- `models/cameras/test_camera_model_opencv_pinhole.py` — pinhole specifics:
  `camera_rays_to_image_points` value and Jacobian (including the
  detached-value contract on the `return_jacobians=True` path), round-trip
  with `image_points_to_camera_rays`, static- and shutter-pose world-point
  projection, `image_points_relative_frame_times` across shutter types, and
  the `transform` contract.
- `models/common/test_frame.py` — `Frame` base properties.
- `models/common/test_utils.py` — `compute_scaled_resolution` and
  quaternion-order helpers.

## Design Constraints

- Pose dataclasses (`Pose`, `DynamicPose`, `Trajectory`) live in
  `kernels/common/pose.py`; return dataclasses live in
  `functional/return_types.py`. Both are re-exported here; the model layer
  never redeclares them.
- `models/` may depend on `functional/` and on backend-facing parameter
  types from `kernels/`; it does not call the C++ extension directly. The
  stable public API of the package is `functional/`; `models/` is optional.
- `CameraModel` methods take explicit `pose: Pose` or
  `dynamic_pose: DynamicPose` plus explicit timestamp kwargs; there is no
  `Frame` argument. The pose object passes through unchanged from `models/`
  to `functional/` to the kernel Python wrapper — the single layer that
  unpacks it just before the C++ launch.
- Iteration-control kwargs on `*_shutter_pose` methods are part of the
  public signature.
- Observation tensors on `Frame` subclasses (`ImageFrame.image` today) are
  registered as `nn.Module` buffers so `.to(...)` propagates.
- Quaternion convention at the public Python API is wxyz; the wxyz↔xyzw
  reorder for `libs/geometry`'s SE(3) primitives lives in
  `kernels/common/utils.py`.
- `CameraModel.transform` returns a new instance; no in-place mutation.
- LiDAR models and a rendering-facing sensor-parameter aggregator are out
  of scope today; `models/lidars/` is an empty placeholder.

See `design.md` for top-level package layout and cross-cutting constraints,
`design-functional.md` for the stateless API and return-type dataclasses, and
`design-kernels.md` for the native layout, dispatch tables, and
kernel-boundary pose contract.
