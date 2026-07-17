<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Sensors Module Design

## Overview

`gsplat/sensors` provides GPU-accelerated camera and spinning-LiDAR projection
for gsplat using hand-written CUDA kernels and PyTorch bindings. The package
is organized into three layers, mirroring `gsplat/geometry`:

- `functional/` — Layer 1, the stateless public Python API.
- `kernels/` — Layer 0, the native CUDA extension plus its Python wrappers,
  autograd glue, and cross-family dispatch tables.
- `models/` — Layer 2, `nn.Module` wrappers and frame containers built on top
  of the functional API.

`gsplat/sensors` follows the repository-wide `gsplat/**` module model: it is internal to
this monorepo and is not independently versioned or released as a standalone
package. Detailed per-layer designs live in `design-functional.md`,
`design-kernels.md`, and `design-models.md`.

## Design Goals

1. Mirror the layered architecture of `gsplat/geometry`: a curated functional
   surface in front of an explicit native backend.
2. Implement hand-written CUDA forward and backward kernels with PyTorch
   bindings; differentiability is a first-class property of every public op.
3. Reuse `gsplat/geometry` for pose, quaternion, trajectory, and
   coordinate-conversion math; do not
   introduce a second geometry stack in `gsplat/sensors`.
4. Enforce strict CUDA / Torch isolation at the native layer so CUDA and
   Torch translation units never see each other's headers.
5. Keep dispatch fully explicit: a `NoExternalDistortion` type, well-formed
   `(projection_type, distortion_type)` keys, and Python `dict` tables — no
   `std::variant`, no runtime enum.

## Scope

The package currently implements a camera slice with three projection models
and a structured spinning-LiDAR slice:

- Camera projections: `OpenCVPinholeProjection`, `FThetaProjection`, and
  `OpenCVFisheyeProjection`.
- External distortion: `NoExternalDistortion` (explicit, never `None`) and
  `BivariateWindshieldDistortion`.
- Shutter modes: `GLOBAL`, `ROLLING_TOP_TO_BOTTOM`, `ROLLING_LEFT_TO_RIGHT`,
  `ROLLING_BOTTOM_TO_TOP`, `ROLLING_RIGHT_TO_LEFT` (defined in
  `kernels/cuda/csrc/shutter_type.h` and bound to a Python `IntEnum`).
- Camera operations re-exported by `functional/`: ray ↔ image-point
  conversion, mean-pose and rolling-shutter world-point projection,
  static-pose and rolling-shutter back-projection to world rays, and pixel
  grid generation. The exhaustive list of names and signatures lives in
  `design-functional.md`.
- Rolling-shutter pose solving via slerp-based dynamic pose interpolation
  along the row or column scan axis.
- LiDAR projection: `RowOffsetStructuredSpinningLidarProjection`, with
  per-row elevations, per-column azimuths, optional per-row azimuth offsets,
  scalar FOV fields, and `SpinningDirection`.
- LiDAR operations re-exported by `functional/`: sensor ray ↔ sensor angle
  conversion, element-index → sensor-angle lookup, spinning-LiDAR world-ray
  generation, and rolling-shutter inverse projection from world points to
  sensor angles. `LidarModel` wraps the same surface for model callers, and
  `LidarFrame` / `LidarFrameSet` carry dense or sparse observation buffers.

A rendering-facing aggregator that bundles intrinsics, extrinsics, shutter,
and timing for the rasterizer is out of scope for this package. `LidarModel`
does not implement `forward()`; callers use its explicit projection methods.

## Module Layout

```text
gsplat/sensors/
  design.md
  design-functional.md
  design-kernels.md
  design-models.md
  __init__.py
  functional/
    __init__.py
    cameras.py
    lidars.py
    return_types.py
  kernels/
    __init__.py
    _backend.py
    projective_sensor_ops.py
    common/
      __init__.py
      pose.py
      pose_interp.py
      tensor_ops.py
      utils.py
    cameras/
      __init__.py
      _projection_validate.py
      ops.py
      types.py
      windshield.py
    lidars/
      __init__.py
      _projection_validate.py
      dispatch.py
      ops.py
      types.py
    cuda/
      CMakeLists.txt
      ext.cpp
      csrc/
        camera_params.h
        camera_kernel.cuh
        camera_kernel.cu
        camera_kernel_backward.cu
        projection_forward_impl.cuh
        projection_backward_impl.cuh
        ftheta_kernel.cuh
        ftheta_kernel.cu
        ftheta_kernel_backward.cu
        fisheye_kernel.cuh
        fisheye_kernel.cu
        fisheye_kernel_backward.cu
        camera_torch.h
        camera_torch.cpp
        external_distortion_params.h
        external_distortion_kernel.cuh
        external_distortion_torch.h
        external_distortion_torch.cpp
        lidar_params.h
        lidar_kernel.cuh
        lidar_kernel.cu
        lidar_kernel_backward.cu
        lidar_torch.h
        lidar_torch.cpp
        shutter_type.h
        math.cuh
  models/
    __init__.py
    cameras/
      __init__.py
      camera_model.py
      image_frame.py
    common/
      __init__.py
      frame.py
      utils.py
    lidars/
      __init__.py
      lidar_frame.py
      lidar_model.py

tests/sensors/                  # mirrored functional, kernel, and model tests + fixtures
```

The `kernels/` Python tree is organized by sensor family (`cameras/` and
`lidars/`) so each family's wrappers, type handles, and dispatch tables live
together. The mirrored tests and fixtures live under `tests/sensors/`.
`kernels/common/` owns cross-family Python helpers — most importantly the
pose dataclasses (`Pose`, `DynamicPose`, `Trajectory`).
`kernels/projective_sensor_ops.py` owns the camera `(projection, distortion)`
dispatch tables. `kernels/lidars/dispatch.py` owns the single-key LiDAR dispatch
tables; their conformance tests mirror those paths under `tests/sensors/`.

## Package Roles

- **`functional/`** — curated stateless public Python API. Wraps the kernel
  layer's raw `tuple[Tensor, ...]` outputs in dataclasses defined in
  `functional/return_types.py`, and is the single layer where named return
  objects come into existence. Per-op signatures and contracts are covered
  in `design-functional.md`.
- **`kernels/`** — backend implementation layer. Owns native CUDA sources,
  the `torch::class_<>` parameter-type registrations, the Python wrappers
  with `torch.autograd.Function` glue, cross-family Python dispatch tables,
  and the build configuration for `gsplat_sensors_cuda`. The three-tier C++
  representation (bridge POD / CUDA-only / Torch-only), templated kernel
  structure, and dispatch-table layout are covered in `design-kernels.md`.
- **`models/`** — higher-level stateful layer. Provides a single concrete
  `CameraModel(nn.Module)` that holds a `CameraProjection` +
  `ExternalDistortion` pair as registered submodules and forwards to the
  functional API, `LidarModel(nn.Module)` that holds a LiDAR projection,
  plus the `Frame` / `ImageFrame` / `ImageFrameGroup` and `LidarFrame` /
  `LidarFrameSet` containers used by callers that work in frame batches.
  `models/` is
  optional. See `design-models.md`.

## Public API Surface

Downstream code uses two import paths:

```python
import gsplat.sensors.functional as sensors
import gsplat.sensors.models as sensor_models
```

The top-level package re-exports `functional` and `models` only:

```python
from . import functional, models

__all__ = ["functional", "models"]
```

`kernels/` is implementation detail. Its submodules remain importable for
in-repo tests and the model layer, but downstream users should not depend on
entrypoint names or per-pair binding suffixes
(`_opencv_pinhole_no_external`, `_opencv_pinhole_bivariate_windshield`, or
raw LiDAR `torch.ops.gsplat_sensors.*` names). Those are dispatch
implementation details hidden behind `functional/`.

## Dependency Boundaries

```text
functional/  ->  kernels/  ->  kernels/cuda/
models/      ->  functional/ and kernels/{cameras,lidars,common}
```

Rules:

- `functional/` may depend on `kernels/` and `functional/return_types.py`;
  it does not import CUDA or Torch C++ symbols directly.
- `models/` may depend on `functional/` and on kernel-layer parameter types
  and pose dataclasses (`kernels.cameras.types`, `kernels.cameras.windshield`,
  `kernels.lidars.types`, `kernels.common.pose`). It does not call the C++
  extension directly.
- `kernels/cameras/ops.py` and `kernels/lidars/ops.py` are the only Python
  modules that call the C++ extension; they validate inputs and dispatch to
  the matching entry functions.
- **Pose contract at the public API.** Functions in `functional/`, `models/`,
  and `kernels/cameras/ops.py` accept `Pose | DynamicPose | Trajectory`
  dataclasses (defined in `kernels/common/pose.py`). The wrappers in
  `kernels/cameras/ops.py` unpack those into separated
  `(translations (K, 3), rotations (K, 4) wxyz, control_times (K,))` tensors
  immediately before the C++ launch — the C++ boundary never sees
  `(..., 4, 4)` view matrices.

## Cross-cutting Design Constraints

The following constraints apply across all three layers; layer-specific
constraints live in each per-layer doc.

- Public stateless sensor functions live exclusively in `functional/`. Kernel
  Python wrappers below it return raw `tuple[Tensor, ...]`.
- Native sources are split per family into three tiers with strict isolation:
  a POD bridge header (`*_params.h`), CUDA-only `*_kernel.cuh`,
  `projection_*_impl.cuh`, and `.cu` files, and Torch-only `*_torch.h` / `.cpp`.
  Only the bridge POD and `shutter_type.h` are shared between the CUDA and
  Torch halves.
- Using the D1–D6 operation mapping in `design-kernels.md`, D1/D2/D3/D5/D6
  implementations are shared inside the CUDA tier through compile-time
  projection and external-distortion policies. Pinhole D4 remains
  projection-local, with one forward and one backward body selected by
  external-distortion policy. These policies do not change the concrete Torch
  registrations or Python dispatch tables.
- Sensor parameter types (`OpenCVPinholeProjection`, `FThetaProjection`,
  `OpenCVFisheyeProjection`, `BivariateWindshieldDistortion`,
  `NoExternalDistortion`, and `RowOffsetStructuredSpinningLidarProjection`)
  are defined in C++ and exposed to Python via `torch::class_<>` registration
  with named keyword constructors. Python does not own sensor parameter data
  layouts.
- Camera projection types carry `.transform(...)` methods bound on the C++
  class, so image-domain re-framing of a projection is a method on the
  parameter object rather than a free function.
- Runtime dispatch happens in Python via explicit `dict` tables. Camera ops
  use `(projection_type, distortion_type)` keys and
  `tests/sensors/kernels/test_projective_sensor_ops.py` enforces the full
  Cartesian product of registered camera projection and distortion types.
  LiDAR ops use one projection-type key in `kernels/lidars/dispatch.py`, with
  conformance covered by `tests/sensors/kernels/test_lidar_dispatch.py`.
- `gsplat/geometry` is the canonical owner of pose / quaternion / trajectory /
  coordinate-conversion math. The pose dataclasses in `kernels/common/pose.py` are thin containers
  that build on those primitives.

## See also

- `design-functional.md` — public stateless API and return-type dataclasses.
- `design-kernels.md` — native layout, dispatch tables, autograd wrappers,
  `torch::class_<>` registrations, build configuration.
- `design-models.md` — `CameraModel`, frame containers, and the pose-state
  contract at the model boundary.
