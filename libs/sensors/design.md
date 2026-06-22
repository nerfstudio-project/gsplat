# Sensors Module Design

## Overview

`libs/sensors` provides GPU-accelerated camera projection for gsplat using
hand-written CUDA kernels and PyTorch bindings. The package is organized into
three layers, mirroring `libs/geometry`:

- `functional/` — Layer 1, the stateless public Python API.
- `kernels/` — Layer 0, the native CUDA extension plus its Python wrappers,
  autograd glue, and cross-family dispatch tables.
- `models/` — Layer 2, `nn.Module` wrappers and frame containers built on top
  of the functional API.

`libs/sensors` follows the repository-wide `libs/**` model: it is internal to
this monorepo and is not independently versioned or released as a standalone
package. Detailed per-layer designs live in `design-functional.md`,
`design-kernels.md`, and `design-models.md`.

## Design Goals

1. Mirror the layered architecture of `libs/geometry`: a curated functional
   surface in front of an explicit native backend.
2. Implement hand-written CUDA forward and backward kernels with PyTorch
   bindings; differentiability is a first-class property of every public op.
3. Reuse `libs/geometry` for pose, quaternion, and trajectory math; do not
   introduce a second geometry stack in `libs/sensors`.
4. Enforce strict CUDA / Torch isolation at the native layer so CUDA and
   Torch translation units never see each other's headers.
5. Keep dispatch fully explicit: a `NoExternalDistortion` type, well-formed
   `(projection_type, distortion_type)` keys, and Python `dict` tables — no
   `std::variant`, no runtime enum.

## Scope

The package currently implements an OpenCV-pinhole camera slice:

- Camera projection: `OpenCVPinholeProjection`.
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

LiDAR support is reserved for future work: `functional/lidars.py` raises
`NotImplementedError` for every name and `models/lidars/` is intentionally
empty. A rendering-facing aggregator that bundles intrinsics, extrinsics,
shutter, and timing for the rasterizer is out of scope for this package.

## Module Layout

```text
libs/sensors/
  design.md
  design-functional.md
  design-kernels.md
  design-models.md
  __init__.py
  pyproject.toml
  conftest.py
  test_data/
  functional/
    __init__.py
    cameras.py
    lidars.py
    return_types.py
    test_cameras.py
  kernels/
    __init__.py
    _backend.py
    projective_sensor_ops.py
    test_projective_sensor_ops.py
    test_backend.py
    common/
      __init__.py
      pose.py
      utils.py
    cameras/
      __init__.py
      ops.py
      types.py
      windshield.py
      test_ops.py
    cuda/
      __init__.py
      build.py
      ext.cpp
      csrc/
        camera_params.h
        camera_kernel.cuh
        camera_kernel.cu
        camera_kernel_backward.cu
        camera_torch.h
        camera_torch.cpp
        external_distortion_params.h
        external_distortion_kernel.cuh
        external_distortion_torch.h
        external_distortion_torch.cpp
        shutter_type.h
        math.cuh
  models/
    __init__.py
    cameras/
      __init__.py
      camera_model.py
      image_frame.py
      test_camera_model.py
      test_camera_model_opencv_pinhole.py
    common/
      __init__.py
      frame.py
      utils.py
    lidars/
      __init__.py
```

The `kernels/` Python tree is organized by sensor family (only `cameras/`
today) so each family's wrappers, type handles, and tests live together.
`kernels/common/` owns cross-family Python helpers — most importantly the
pose dataclasses (`Pose`, `DynamicPose`, `Trajectory`).
`kernels/projective_sensor_ops.py` and its conformance test sit at the top of
`kernels/` because they own the cross-family dispatch tables.
`models/lidars/` is intentionally empty (`__all__ = []`) and reserved for
future work, mirroring `functional/lidars.py`.

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
  functional API, plus the `Frame` / `ImageFrame` / `ImageFrameGroup`
  containers used by callers that work in frame batches. `models/` is
  optional. See `design-models.md`.

## Public API Surface

Downstream code uses two import paths:

```python
import gsplat_sensors.functional as sensors
import gsplat_sensors.models as sensor_models
```

The top-level package re-exports `functional` and `models` only:

```python
from . import functional, models

__all__ = ["functional", "models"]
```

`kernels/` is implementation detail. Its submodules remain importable for
in-repo tests and the model layer, but downstream users should not depend on
entrypoint names or per-pair binding suffixes
(`_opencv_pinhole_no_external`, `_opencv_pinhole_bivariate_windshield`, …) —
those are dispatch implementation details hidden behind `functional/`.

## Dependency Boundaries

```text
functional/  ->  kernels/  ->  kernels/cuda/
models/      ->  functional/ and kernels/{cameras,common}
```

Rules:

- `functional/` may depend on `kernels/` and `functional/return_types.py`;
  it does not import CUDA or Torch C++ symbols directly.
- `models/` may depend on `functional/` and on kernel-layer parameter types
  and pose dataclasses (`kernels.cameras.types`, `kernels.cameras.windshield`,
  `kernels.common.pose`). It does not call the C++ extension directly.
- `kernels/cameras/ops.py` is the only Python module that calls the C++
  extension; it validates inputs and dispatches to the per-pair entry
  functions.
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
  a POD bridge header (`*_params.h`), CUDA-only `*_kernel.cuh` / `.cu`, and
  Torch-only `*_torch.h` / `.cpp`. Only the bridge POD and `shutter_type.h`
  are shared between the two halves.
- Sensor parameter types (`OpenCVPinholeProjection`,
  `BivariateWindshieldDistortion`, `NoExternalDistortion`) are defined in
  C++ and exposed to Python via `torch::class_<>` registration with named
  keyword constructors. Python does not own sensor parameter data layouts.
- `OpenCVPinholeProjection` carries a `.transform(...)` method bound on the
  C++ class, so SE(3) re-framing of a projection is a method on the
  parameter object rather than a free function.
- Runtime dispatch between sensor type combinations happens in Python via
  explicit `dict[tuple[type, type], Callable]` tables, one per shared op.
  `kernels/test_projective_sensor_ops.py` enforces that every table covers
  the full Cartesian product of registered projection and distortion types.
- `libs/geometry` is the canonical owner of pose / quaternion / trajectory
  math. The pose dataclasses in `kernels/common/pose.py` are thin containers
  that build on those primitives.

## See also

- `design-functional.md` — public stateless API and return-type dataclasses.
- `design-kernels.md` — native layout, dispatch tables, autograd wrappers,
  `torch::class_<>` registrations, build configuration.
- `design-models.md` — `CameraModel`, frame containers, and the pose-state
  contract at the model boundary.
