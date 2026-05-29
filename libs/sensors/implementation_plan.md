# Sensors Implementation Plan

This document complements `libs/sensors/design.md` and translates the design
into a single-PR delivery plan.

## Planning Assumptions

- `libs/sensors` becomes the canonical home for sensor-specific kernels and
  functional APIs.
- this PR is scoped entirely to `libs/`; no files outside `libs/` are modified.
- rasterization integration and legacy forwarding are deferred to follow-up
  work.

## Delivery Goals

1. Implement sensor kernels and backend bindings under `libs/sensors`.
2. Expose a stateless functional API for cameras and LiDARs.
3. Expose stateful model-layer abstractions alongside the stateless API.
4. Define `ProjectiveSensorParameters` as a rendering-facing abstraction within
   the sensor library, ready for future rasterization integration but not wired
   in yet.

## Cross-Cutting Rules

- `libs/geometry` remains the owner of pose and trajectory semantics.
- `libs/sensors` owns sensor parameter packs, dispatch, and sensor-specific CUDA
  math.
- generic pose interpolation and trajectory logic should stay in
  `libs/geometry`; sensor-side native code should only provide shutter-aware
  bridge helpers where camera or lidar kernels need fused sensor timing logic.

## Scope

Land the full `libs/sensors` library — backend kernels, functional API, and
model layer — in a single PR. This is a standalone implementation with its own
tests; integration with the rest of the codebase is deferred.

The work is organized into four phases. Each phase produces testable artifacts
and must have all its tests passing before the next phase begins.

### Boundary Rule

This work is scoped entirely to `libs/`. No files outside `libs/` should be
modified. Legacy integration (forwarding existing sensor entrypoints into
`libs/sensors`, removing duplicated sensor math) is deferred to a follow-up.

### Non-Goals

- no modifications to files outside `libs/`
- no rasterization integration or legacy forwarding
- no alternative backend path
- no duplication of functional or kernel logic in the model layer

---

## Phase 1: Camera Kernels and Functional API

### Objective

Stand up the `libs/sensors` package skeleton, build system, and camera-specific
backend, and expose camera operations through the functional API with passing
tests.

### Files

- `__init__.py`, `pyproject.toml`
- `kernels/__init__.py`
- `kernels/types.py` — camera-related parameter packs and enums
  (`ShutterType`, `ExternalDistortion`, `NoExternalDistortion`,
  `BivariateWindshieldDistortion`, `ReferencePolynomial`,
  `OpenCVPinholeProjection`, `OpenCVFisheyeProjection`, `FThetaProjection`,
  `FThetaPolynomialType`, `ImagePointsReturn`,
  `WorldPointsToImagePointsReturn`, `WorldRaysReturn`)
- `kernels/_backend.py`
- `kernels/camera_ops.py`
- `kernels/projective_sensor_ops.py` — shared projective-sensor wrappers
  (camera paths only at this stage)
- `kernels/cuda/__init__.py`
- `kernels/cuda/build.py`
- `kernels/cuda/ext.cpp` — camera bindings (lidar stubs can be added later)
- `kernels/cuda/csrc/shutter_pose.cuh`, `kernels/cuda/csrc/shutter_pose.cu`
- `kernels/cuda/csrc/camera.cuh`, `kernels/cuda/csrc/camera.cu`
- `include/projective_sensor.h`
- `include/projective_sensor_variant.h` — camera-only variant initially
- `functional/__init__.py` — camera exports only
- `functional/cameras.py`

### Acceptance Criteria

- `libs/sensors` builds and loads its CUDA extension
- camera kernels are callable through `functional/cameras.py`
- rolling-shutter and mean-pose behavior are implemented for cameras
- all camera functional tests pass

### Tests

Add `functional/test_cameras.py`.

Every camera functional entrypoint must be tested across all applicable
projection families. See `libs/sensors/design.md` "Testing Structure" for the
coverage categories.

Per-operation coverage (each tested for OpenCV pinhole, OpenCV fisheye, and
F-theta where applicable):

- `camera_rays_to_image_points` — forward correctness against known formulas
- `image_points_to_camera_rays` — forward correctness against known formulas
- round-trip: `camera_rays_to_image_points` then `image_points_to_camera_rays`
  and vice versa
- `project_world_points_mean_pose` — forward correctness
- `project_world_points_shutter_pose` — forward correctness, rolling-shutter
  regression
- `image_points_to_world_rays_static_pose` — forward correctness
- `image_points_to_world_rays_shutter_pose` — forward correctness,
  rolling-shutter regression

Cross-cutting coverage:

- autograd validation for all differentiable operations
- external distortion (bivariate windshield) forward and inverse, composed with
  each camera family
- parameter-pack validation (invalid inputs, missing fields, dtype/device
  mismatches)
- backend loading behavior (`_backend.py` load and capability checks)

### Gate

All tests in `functional/test_cameras.py` pass before starting Phase 2.

---

## Phase 2: Camera Model Layer

### Objective

Add stateful camera model wrappers and frame containers on top of the Phase 1
functional API, with passing tests.

### Files

- `models/__init__.py` — camera exports only at this stage
- `models/frame.py` — base `Frame` class
- `models/return_types.py` — `ImagePointsReturn`,
  `WorldPointsToImagePointsReturn`, `WorldRaysReturn`
- `models/cameras.py` — `CameraModel`, `OpenCVPinholeCameraModel`,
  `OpenCVFisheyeCameraModel`, `FThetaCameraModel`, `ImageFrame`
- `kernels/types.py` — add `ProjectiveSensorParameters` (camera projections
  only at this stage)

### Acceptance Criteria

- camera models own projection state cleanly
- `ImageFrame` carries pose, timing, and metadata state without embedding kernel
  logic
- model methods delegate to `functional/cameras.py` rather than reimplementing
  math
- `ImageFrame.to_projective_sensor_parameters()` materializes a valid
  `ProjectiveSensorParameters`
- all camera model tests pass
- all Phase 1 camera functional tests still pass

### Tests

Add `models/test_models.py` (camera sections).

Coverage:

- model construction and validation for each camera family
- conversion from owned model state to kernel parameter packs
- `ImageFrame` -> `ProjectiveSensorParameters` conversion
- delegation to functional operators (verify model methods produce identical
  results to direct functional calls)
- frame-oriented convenience behavior

### Gate

All tests in `models/test_models.py` (camera) and `functional/test_cameras.py`
pass before starting Phase 3.

---

## Phase 3: LiDAR Kernels and Functional API

### Objective

Extend the backend and functional API with LiDAR-specific kernels, and expose
LiDAR operations through the functional API with passing tests.

### Files

- `kernels/types.py` — add LiDAR parameter packs (`SpinningDirection`,
  `RowOffsetStructuredSpinningLidarProjection`, `SensorAnglesReturn`)
- `kernels/lidar_ops.py`
- `kernels/projective_sensor_ops.py` — extend shared wrappers with lidar paths
- `kernels/cuda/csrc/lidar.cuh`, `kernels/cuda/csrc/lidar.cu`
- `kernels/cuda/ext.cpp` — add lidar bindings
- `include/projective_sensor_variant.h` — extend variant to include lidar
- `functional/__init__.py` — add lidar exports
- `functional/lidars.py`

### Acceptance Criteria

- lidar kernels are callable through `functional/lidars.py`
- rolling-shutter behavior is implemented for lidar where required
- all lidar functional tests pass
- all Phase 1 and Phase 2 tests still pass

### Tests

Add `functional/test_lidars.py`.

Every lidar functional entrypoint must be tested. See
`libs/sensors/design.md` "Testing Structure" for the coverage categories.

Per-operation coverage:

- `elements_to_sensor_angles` — forward correctness against known formulas,
  parameter-pack validation
- `generate_spinning_lidar_rays` — forward correctness, rolling-shutter
  regression, spinning-direction variants
- `inverse_project_spinning_lidar` — forward correctness, convergence behavior
- round-trip: `generate_spinning_lidar_rays` then
  `inverse_project_spinning_lidar` and vice versa

Cross-cutting coverage:

- autograd validation for all differentiable operations
- row-azimuth-offset behavior
- parameter-pack validation (invalid inputs, missing fields, dtype/device
  mismatches)

### Gate

All tests in `functional/test_lidars.py`, `functional/test_cameras.py`, and
`models/test_models.py` (camera) pass before starting Phase 4.

---

## Phase 4: LiDAR Model Layer

### Objective

Add stateful LiDAR model wrappers and frame containers, completing the full
`libs/sensors` library.

### Files

- `models/lidars.py` — `LidarModel`,
  `RowOffsetStructuredSpinningLidarModel`, `LidarFrame`
- `models/return_types.py` — add `SensorAnglesReturn`,
  `WorldPointsToSensorAnglesReturn`
- `models/__init__.py` — add lidar exports
- `kernels/types.py` — extend `ProjectiveSensorParameters` to accept lidar
  projections

### Acceptance Criteria

- lidar models own projection state cleanly
- `LidarFrame` carries pose, timing, and metadata state without embedding kernel
  logic
- model methods delegate to `functional/lidars.py` rather than reimplementing
  math
- `LidarFrame.to_projective_sensor_parameters()` materializes a valid
  `ProjectiveSensorParameters`
- all lidar model tests pass
- all prior phase tests still pass

### Tests

Extend `models/test_models.py` (lidar sections).

Coverage:

- model construction and validation for lidar
- conversion from owned model state to kernel parameter packs
- `LidarFrame` -> `ProjectiveSensorParameters` conversion
- delegation to functional operators (verify model methods produce identical
  results to direct functional calls)
- frame-oriented convenience behavior

### Gate

All tests across all phases pass:

- `functional/test_cameras.py`
- `functional/test_lidars.py`
- `models/test_models.py`