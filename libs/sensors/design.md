# Sensors Module Design

## Overview

`libs/sensors` provides GPU-accelerated camera and LiDAR operations for gsplat
using native CUDA kernels and PyTorch bindings.

This module intentionally follows the same top-level organizational pattern as
`libs/geometry`:

- `functional/` is the stateless public API.
- `kernels/` owns backend dispatch, autograd glue, parameter packs, and native
  CUDA integration.
- `models/` adds higher-level `nn.Module` wrappers and frame-oriented data
  structures for learnable sensor state.

This design uses native CUDA `.cu`/`.cuh` translation units and the existing
gsplat extension pattern used by `libs/geometry`.

## Design Goals

1. Mirror the repo-local architecture style of `libs/geometry`.
2. Implement explicit CUDA kernels plus PyTorch bindings.
3. Keep the public API centered on sensor operations rather than backend
   implementation details.
4. Support differentiable batch projection and back-projection for cameras and
   LiDARs.
5. Reuse gsplat geometry conventions for poses, trajectories, and tensor
   contracts instead of introducing a second geometry abstraction stack.
6. Introduce a neutral `ProjectiveSensor` abstraction for shared projection
   logic without forcing LiDARs into a camera-shaped base class.
7. Keep native CUDA code organized around `.cuh` device helpers plus sibling
   `.cu` launch and export units.
8. Leave room for stateful model wrappers without forcing model concerns into
   the kernel layer.

## Scope

Initial scope should cover these major sensor families:

- camera projections:
  - OpenCV pinhole
  - OpenCV fisheye
  - F-theta
- optional external distortion:
  - no distortion
  - bivariate windshield distortion
- LiDAR projection:
  - row-offset structured spinning LiDAR
- common operations:
  - camera rays <-> image points
  - world points -> image points
  - image points/pixels -> world rays
  - LiDAR elements -> sensor angles
  - LiDAR elements -> world rays
  - world points -> LiDAR sensor angles
  - rolling shutter and mean-pose variants where applicable

## Module Layout

```text
libs/sensors/
  design.md
  __init__.py
  pyproject.toml
  include/
    projective_sensor.h        # shared C++17 trait contract for camera/LiDAR dispatch
    projective_sensor_variant.h # runtime-parameter -> concrete-sensor dispatch helpers
  kernels/
    __init__.py
    _backend.py
    projective_sensor_ops.py
    camera_ops.py
    lidar_ops.py
    types.py                   # parameter packs, enums, return dataclasses
    cuda/
      __init__.py
      build.py
      ext.cpp
      csrc/
        shutter_pose.cu
        shutter_pose.cuh
        camera.cu
        camera.cuh
        lidar.cu
        lidar.cuh
  functional/
    __init__.py
    cameras.py
    lidars.py
    test_cameras.py
    test_lidars.py
  models/
    __init__.py
    frame.py
    return_types.py
    cameras.py
    lidars.py
    test_models.py
```

## Expected File Interfaces

This section defines the expected primary interfaces for each planned Python and
binding file. These are design targets, not a requirement that every symbol be
implemented in the first iteration.

### Native Support Headers

These headers define the shared native abstraction used by common CUDA launch
sites. They are internal implementation interfaces, not user-facing APIs.

**`include/projective_sensor.h`**

Purpose:

- define a neutral common contract for camera and LiDAR kernel dispatch,
- avoid expressing LiDAR support through camera inheritance,
- keep the abstraction compatible with the repo's current C++17 toolchain.

Design notes:

- Use `ProjectiveSensor` as the common abstraction name.
- Prefer a C++17 trait-based or detection-based contract.
- If the repo intentionally moves to C++20 later, this contract can be upgraded
  to a real `concept`, but the initial design should not require C++20-only
  language or library features.

Expected interface:

```cpp
template <typename Sensor>
struct projective_sensor_traits;

template <typename Sensor, typename = void>
struct is_projective_sensor : std::false_type {};

template <typename Sensor>
struct is_projective_sensor<
    Sensor,
    std::void_t<typename projective_sensor_traits<Sensor>::kernel_parameters_type>
> : std::true_type {};

template <typename Sensor>
using projective_sensor_kernel_parameters_t =
    typename projective_sensor_traits<Sensor>::kernel_parameters_type;

// Each ProjectiveSensor is expected to provide, either directly or through traits:
// - a KernelParameters type
// - construction from (KernelParameters, sensor_index)
// - world_point_to_sensor_point_shutter_pose(...)
// - sensor_point_to_world_ray_shutter_pose(...)
// - sensor_relative_frame_time(...)
```

Common-interface rule:

- The common `ProjectiveSensor` contract is only for operations that genuinely
  apply to both cameras and LiDARs.
- Camera-only and LiDAR-only capabilities stay in their own model families.
- LiDAR-specific element indexing and angle lookup should not be forced into the
  common abstraction.

**`include/projective_sensor_variant.h`**

Purpose:

- provide runtime-to-concrete-sensor dispatch helpers for shared kernels,
- centralize the mapping between runtime parameter packs and template-specialized
  sensor kernels.

Expected interface:

```cpp
template <typename... Ts>
struct TypeList {};

using ProjectiveSensorTypes = TypeList<
    /* all supported concrete camera and lidar projective sensors */
>;

using ProjectiveSensorKernelParamsVariant =
    /* variant of each ProjectiveSensor's KernelParameters */;

template <typename KernelParameters>
using ProjectiveSensorFromKernelParams =
    /* reverse mapping from KernelParameters to concrete ProjectiveSensor */;

auto get_projective_sensor_kernel_params(...) -> ProjectiveSensorKernelParamsVariant;
```

Design rule:

- Shared dispatch should happen once at the launch boundary.
- Once a concrete sensor type is recovered, CUDA kernels should remain fully
  specialized on that concrete type.
- Avoid duplicating the runtime branching logic independently in each kernel.

### Top-Level Package

**`__init__.py`**

Purpose:

- expose the main public subpackages,
- keep imports lightweight,
- avoid triggering CUDA extension build on package import.

Expected interface:

```python
from . import functional, models

__all__ = [
    "functional",
    "models",
]
```

### Functional API

**`functional/__init__.py`**

Purpose:

- provide the curated public stateless API,
- re-export camera and LiDAR operators from sibling modules.

Expected interface:

```python
from .cameras import (
    camera_rays_to_image_points,
    image_points_to_camera_rays,
    image_points_to_world_rays_static_pose,
    image_points_to_world_rays_shutter_pose,
    project_world_points_mean_pose,
    project_world_points_shutter_pose,
)
from .lidars import (
    elements_to_sensor_angles,
    generate_spinning_lidar_rays,
    inverse_project_spinning_lidar,
)

__all__ = [
    "camera_rays_to_image_points",
    "image_points_to_camera_rays",
    "image_points_to_world_rays_static_pose",
    "image_points_to_world_rays_shutter_pose",
    "project_world_points_mean_pose",
    "project_world_points_shutter_pose",
    "elements_to_sensor_angles",
    "generate_spinning_lidar_rays",
    "inverse_project_spinning_lidar",
]
```

**`functional/cameras.py`**

Purpose:

- define the canonical stateless camera API,
- delegate directly to `kernels.camera_ops`.

Expected interface:

```python
from ..kernels.camera_ops import (
    camera_rays_to_image_points,
    image_points_to_camera_rays,
    image_points_to_world_rays_static_pose,
    image_points_to_world_rays_shutter_pose,
    project_world_points_mean_pose,
    project_world_points_shutter_pose,
)

__all__ = [
    "camera_rays_to_image_points",
    "image_points_to_camera_rays",
    "image_points_to_world_rays_static_pose",
    "image_points_to_world_rays_shutter_pose",
    "project_world_points_mean_pose",
    "project_world_points_shutter_pose",
]
```

**`functional/lidars.py`**

Purpose:

- define the canonical stateless LiDAR API,
- delegate directly to `kernels.lidar_ops`.

Expected interface:

```python
from ..kernels.lidar_ops import (
    elements_to_sensor_angles,
    generate_spinning_lidar_rays,
    inverse_project_spinning_lidar,
)

__all__ = [
    "elements_to_sensor_angles",
    "generate_spinning_lidar_rays",
    "inverse_project_spinning_lidar",
]
```

### Kernel Python Layer

**`kernels/__init__.py`**

Purpose:

- re-export backend-facing stateless operators and parameter types,
- mirror the package pattern used in `libs/geometry.kernels`.

Expected interface:

```python
from . import camera_ops, lidar_ops, projective_sensor_ops, types
from .camera_ops import (
    camera_rays_to_image_points,
    image_points_to_camera_rays,
    image_points_to_world_rays_static_pose,
    image_points_to_world_rays_shutter_pose,
    project_world_points_mean_pose,
    project_world_points_shutter_pose,
)
from .lidar_ops import (
    elements_to_sensor_angles,
    generate_spinning_lidar_rays,
    inverse_project_spinning_lidar,
)
from .projective_sensor_ops import (
    projective_sensor_points_to_world_rays_shutter_pose,
    projective_sensor_relative_frame_times,
    world_points_to_projective_sensor_points_shutter_pose,
)
from .types import (
    BivariateWindshieldDistortion,
    ExternalDistortion,
    FThetaPolynomialType,
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ReferencePolynomial,
    RowOffsetStructuredSpinningLidarProjection,
    ShutterType,
    SpinningDirection,
)
```

**`kernels/_backend.py`**

Purpose:

- lazily build/load `gsplat_sensors_cuda`,
- validate whether fast CUDA paths are applicable,
- provide internal helpers shared by `camera_ops.py` and `lidar_ops.py`.

Expected interface:

```python
_SENSORS_CUDA = None

def load_sensors_cuda():
    ...

def use_cuda_extension_for_camera_projection(...) -> bool:
    ...

def use_cuda_extension_for_camera_backprojection(...) -> bool:
    ...

def use_cuda_extension_for_lidar_projection(...) -> bool:
    ...
```

**`kernels/projective_sensor_ops.py`**

Purpose:

- hold shared stateless wrappers whose semantics are common to both cameras and
  LiDARs,
- avoid re-implementing dispatch logic in both `camera_ops.py` and
  `lidar_ops.py`.

Expected interface:

```python
def projective_sensor_points_to_world_rays_shutter_pose(...):
    ...

def world_points_to_projective_sensor_points_shutter_pose(...):
    ...

def projective_sensor_relative_frame_times(...):
    ...
```

Design rule:

- This module should only contain truly shared projective-sensor operations.
- Family-specific APIs remain in `camera_ops.py` and `lidar_ops.py`.

**`kernels/types.py`**

Purpose:

- define enums, parameter packs, and lightweight return dataclasses used by the
  stateless kernel layer.

Expected interface:

```python
from dataclasses import dataclass
from enum import IntEnum
from torch import Tensor

class ShutterType(IntEnum): ...
class ReferencePolynomial(IntEnum): ...
class FThetaPolynomialType(IntEnum): ...
class SpinningDirection(IntEnum): ...

@dataclass
class ExternalDistortion: ...

@dataclass
class NoExternalDistortion(ExternalDistortion): ...

@dataclass
class BivariateWindshieldDistortion(ExternalDistortion):
    reference_polynomial: ReferencePolynomial
    h_poly: Tensor
    v_poly: Tensor
    h_poly_inv: Tensor
    v_poly_inv: Tensor

@dataclass
class OpenCVPinholeProjection:
    focal_length: Tensor
    principal_point: Tensor
    radial_coeffs: Tensor
    tangential_coeffs: Tensor
    thin_prism_coeffs: Tensor

@dataclass
class OpenCVFisheyeProjection:
    focal_length: Tensor
    principal_point: Tensor
    forward_poly: Tensor
    dforward_poly: Tensor
    approx_backward_poly: Tensor
    max_angle: float
    newton_iterations: int
    min_2d_norm: Tensor

@dataclass
class FThetaProjection:
    reference_poly: FThetaPolynomialType
    principal_point: Tensor
    fw_poly: Tensor
    bw_poly: Tensor
    A: Tensor
    Ainv: Tensor
    dfw_poly: Tensor
    dbw_poly: Tensor
    max_angle: float
    newton_iterations: int
    min_2d_norm: Tensor

@dataclass
class RowOffsetStructuredSpinningLidarProjection:
    n_rows: int
    n_columns: int
    row_elevations_rad: Tensor
    column_azimuths_rad: Tensor
    fov_horiz_start_rad: float
    fov_horiz_span_rad: float
    fov_vert_start_rad: float
    fov_vert_span_rad: float
    row_azimuth_offsets_rad: Tensor | None = None
    spinning_frequency_hz: float = 0.0
    spinning_direction: SpinningDirection = SpinningDirection.CLOCKWISE
    angles_to_columns_map: Tensor | None = None
    angles_to_columns_map_resolution_factor: int = 1

@dataclass
class ImagePointsReturn:
    image_points: Tensor
    valid_flag: Tensor

@dataclass
class WorldPointsToImagePointsReturn:
    image_points: Tensor
    valid_flag: Tensor | None = None
    timestamps: Tensor | None = None

@dataclass
class WorldRaysReturn:
    world_rays: Tensor
    timestamps: Tensor | None = None

@dataclass
class SensorAnglesReturn:
    sensor_angles: Tensor
    valid_flag: Tensor
```

**`kernels/camera_ops.py`**

Purpose:

- expose Python wrappers for camera CUDA entrypoints,
- own custom autograd where fused CUDA backward is needed.

Expected interface:

```python
def camera_rays_to_image_points(
    camera_rays,
    projection,
    external_distortion,
    resolution,
):
    ...

def image_points_to_camera_rays(
    image_points,
    projection,
    external_distortion,
    resolution,
):
    ...

def project_world_points_shutter_pose(
    world_points,
    projection,
    external_distortion,
    resolution,
    shutter_type,
    translation,
    rotation,
    frame_times,
):
    ...

def project_world_points_mean_pose(
    world_points,
    projection,
    external_distortion,
    resolution,
    translation,
    rotation,
):
    ...

def image_points_to_world_rays_static_pose(
    image_points,
    projection,
    external_distortion,
    resolution,
    translation,
    rotation,
):
    ...

def image_points_to_world_rays_shutter_pose(
    image_points,
    projection,
    external_distortion,
    resolution,
    shutter_type,
    translation,
    rotation,
    frame_times,
):
    ...
```

**`kernels/lidar_ops.py`**

Purpose:

- expose Python wrappers for LiDAR CUDA entrypoints,
- own custom autograd where LiDAR inverse projection or rolling shutter kernels
  need fused backward logic.

Expected interface:

```python
def elements_to_sensor_angles(
    elements,
    projection,
):
    ...

def generate_spinning_lidar_rays(
    elements,
    projection,
    translation,
    rotation,
    frame_times,
):
    ...

def inverse_project_spinning_lidar(
    world_points,
    projection,
    translation,
    rotation,
    frame_times,
    max_iterations=10,
):
    ...
```

### CUDA Build and Binding Layer

**`kernels/cuda/__init__.py`**

Purpose:

- expose the build helper without forcing immediate extension compilation.

Expected interface:

```python
from .build import build_and_load_sensors_cuda, get_build_parameters

__all__ = [
    "build_and_load_sensors_cuda",
    "get_build_parameters",
]
```

**`kernels/cuda/build.py`**

Purpose:

- JIT-build and load the native extension,
- cache build settings the same way `libs/geometry` does.

Expected interface:

```python
def get_build_parameters():
    ...

def build_and_load_sensors_cuda():
    ...
```

**`kernels/cuda/ext.cpp`**

Purpose:

- define the C++/PyBind boundary for all exported CUDA functions,
- validate tensor contracts before calling native kernels.

Expected exported bindings:

```cpp
torch::Tensor camera_rays_to_image_points(...);
torch::Tensor image_points_to_camera_rays(...);
std::tuple<torch::Tensor, torch::Tensor> project_world_points_shutter_pose(...);
std::tuple<torch::Tensor, torch::Tensor> project_world_points_mean_pose(...);
torch::Tensor image_points_to_world_rays_static_pose(...);
torch::Tensor image_points_to_world_rays_shutter_pose(...);

torch::Tensor elements_to_sensor_angles(...);
torch::Tensor generate_spinning_lidar_rays(...);
std::tuple<torch::Tensor, torch::Tensor> inverse_project_spinning_lidar(...);
```

### Model Layer

**`models/__init__.py`**

Purpose:

- expose the high-level model and frame API.

Expected interface:

```python
from .cameras import (
    CameraModel,
    FThetaCameraModel,
    ImageFrame,
    OpenCVFisheyeCameraModel,
    OpenCVPinholeCameraModel,
)
from .frame import Frame
from .lidars import (
    LidarFrame,
    LidarModel,
    RowOffsetStructuredSpinningLidarModel,
)
from .return_types import (
    ImagePointsReturn,
    SensorAnglesReturn,
    WorldPointsToImagePointsReturn,
    WorldPointsToSensorAnglesReturn,
    WorldRaysReturn,
)
```

**`models/frame.py`**

Purpose:

- define the common frame base class shared by camera and LiDAR observations.

Expected interface:

```python
class Frame(nn.Module):
    def __init__(
        self,
        id,
        translation,
        rotation,
        timestamp_start_us,
        timestamp_end_us,
        metadata=None,
    ):
        ...

    def to_projective_sensor_parameters(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError
```

**`models/return_types.py`**

Purpose:

- define higher-level result objects used by model methods.

Expected interface:

```python
@dataclass
class ImagePointsReturn: ...

@dataclass
class WorldPointsToImagePointsReturn: ...

@dataclass
class WorldRaysReturn: ...

@dataclass
class SensorAnglesReturn: ...

@dataclass
class WorldPointsToSensorAnglesReturn: ...
```

**`models/cameras.py`**

Purpose:

- define camera `nn.Module` wrappers that own projection state and delegate to
  the stateless functional API,
- define `ImageFrame`.

Expected interface:

```python
class CameraModel(nn.Module):
    def __init__(self, external_distortion, resolution, shutter_type):
        ...

    def world_points_to_image_points_shutter_pose(self, world_points, frame):
        ...

    def world_points_to_image_points_mean_pose(self, world_points, frame):
        ...

    def image_points_to_world_rays_static_pose(self, image_points, frame):
        ...

    def image_points_to_world_rays_shutter_pose(self, image_points, frame):
        ...

    def camera_rays_to_image_points(self, camera_rays):
        ...

    def image_points_to_camera_rays(self, image_points):
        ...

    def get_parameters(self):
        raise NotImplementedError

class OpenCVPinholeCameraModel(CameraModel): ...
class OpenCVFisheyeCameraModel(CameraModel): ...
class FThetaCameraModel(CameraModel): ...

class ImageFrame(Frame):
    def __init__(
        self,
        id,
        camera_model,
        translation,
        rotation,
        timestamp_start_us,
        timestamp_end_us,
        image,
        metadata=None,
    ):
        ...

    def to_projective_sensor_parameters(self):
        ...
```

**`models/lidars.py`**

Purpose:

- define LiDAR `nn.Module` wrappers that own projection state and delegate to
  the stateless functional API,
- define `LidarFrame`.

Expected interface:

```python
class LidarModel(nn.Module):
    def __init__(self, projection):
        ...

    def elements_to_sensor_angles(self, elements):
        ...

    def elements_to_world_rays_shutter_pose(self, elements, frame):
        ...

    def world_points_to_sensor_angles_shutter_pose(self, world_points, frame):
        ...

    def get_parameters(self):
        raise NotImplementedError

class RowOffsetStructuredSpinningLidarModel(LidarModel): ...

class LidarFrame(Frame):
    def __init__(
        self,
        id,
        lidar_model,
        translation,
        rotation,
        timestamp_start_us,
        timestamp_end_us,
        distance_m,
        intensity=None,
        element_indices=None,
        metadata=None,
    ):
        ...

    def to_projective_sensor_parameters(self):
        ...
```

## Package Roles

### `functional/`

`functional/` is the primary public API surface, just as it is in
`libs/geometry`.

It is responsible for:

- exporting stateless camera and LiDAR operations,
- defining API-level contracts for shapes, dtypes, and return values,
- providing the import path downstream code should prefer,
- delegating execution to `kernels/`.

Representative public operations:

- `camera_rays_to_image_points`
- `image_points_to_camera_rays`
- `project_world_points_shutter_pose`
- `project_world_points_mean_pose`
- `image_points_to_world_rays_static_pose`
- `generate_spinning_lidar_rays`
- `elements_to_sensor_angles`
- `inverse_project_spinning_lidar`

### `kernels/`

`kernels/` is the backend implementation layer.

It is responsible for:

- loading and validating the `gsplat_sensors_cuda` extension,
- defining `torch.autograd.Function` wrappers where custom backward is needed,
- owning CUDA build configuration,
- defining backend-facing parameter packs and return structures,
- containing native C++ and CUDA sources.

This layer is architecture-critical but is not the primary conceptual entry
point for downstream users.

### `models/`

`models/` is the higher-level stateful layer for trainable sensor objects and
frame containers.

It is responsible for:

- `nn.Module` wrappers around camera and LiDAR parameter packs,
- learnable or stateful sensor intrinsics/extrinsics,
- frame-level containers such as `ImageFrame` and `LidarFrame`,
- converting config or dataset parameters into the working parameter packs used
  by `kernels/`.

Unlike `functional/`, `models/` is optional for users who only need direct
projection kernels.

## Public API Surface

The public stateless API should be exposed through `gsplat_sensors.functional`.

Example import style:

```python
import gsplat_sensors.functional as sensors

image_points, valid = sensors.camera_rays_to_image_points(
    camera_rays, projection, external_distortion, resolution
)
```

The top-level package should make the main public layers easy to discover:

```python
import gsplat_sensors.functional as sensors
import gsplat_sensors.models as sensor_models
```

The public API should be defined in terms of sensor semantics, not extension
entrypoint names.

## Dependency Boundaries

The module should follow this dependency direction:

```text
functional/  ->  kernels/  ->  kernels/cuda/
models/      ->  functional/ and/or kernels/
```

Rules:

- `functional/` may depend on `kernels/`.
- `models/` may depend on `functional/` and backend-facing parameter types from
  `kernels/`.
- native sources under `kernels/cuda/` may depend on `include/`.
- public API definitions do not live under `kernels/`.
- sensor kernels may depend on geometry conventions, but `libs/sensors` should
  not duplicate the public geometry API.

## Geometry Integration

Sensor kernels need static and time-varying pose behavior. In gsplat, that
should be expressed through the tensor conventions already used by
`libs/geometry`.

Recommended rule:

- `libs/geometry` remains the canonical owner of pose and trajectory semantics.
- `libs/sensors` accepts pose state in the same tensor layouts already used by
  geometry kernels.
- rolling shutter interpolation in sensor CUDA code should use sensor-owned
  device helpers that follow geometry conventions, rather than introducing a
  second incompatible pose API.

Ownership rule:

- generic pose interpolation, quaternion math, and trajectory semantics stay in
  `libs/geometry`,
- `libs/sensors` may own only the sensor-specific bridge helpers needed to map
  shutter timing or LiDAR sampling conventions onto those geometry semantics.

Practically, that means sensor kernels should operate on tensors such as:

- translations shaped like `(N, 3)` or broadcastable equivalents,
- quaternion rotations shaped like `(N, 4)` using the repo's established
  quaternion convention,
- trajectory timestamps and query times shaped like `(N,)` or `(N, 1)` where
  rolling shutter interpolation is required.

This preserves compatibility with `libs/geometry` while keeping each extension
self-contained at compile time.

## Functional Organization

The functional layer should be organized by sensor domain:

- `functional/cameras.py` for camera projection and back-projection.
- `functional/lidars.py` for LiDAR ray generation and inverse projection.
- `functional/__init__.py` for curated exports.

These files are the canonical names and import paths for stateless sensor
operators.

## Kernel Organization

The kernel layer should be organized by implementation concern:

- `kernels/_backend.py` handles extension loading and runtime capability checks.
- `kernels/projective_sensor_ops.py` contains shared projective-sensor wrappers.
- `kernels/types.py` defines enums, parameter packs, and small return dataclasses.
- `kernels/camera_ops.py` contains camera backend wrappers and autograd glue.
- `kernels/lidar_ops.py` contains LiDAR backend wrappers and autograd glue.
- `kernels/cuda/` contains build logic and native sources.

Where backward is straightforward or already provided by PyTorch tensor
composition, the Python layer can remain thin. Where rolling shutter iteration,
custom Jacobians, or fused kernels matter for performance, explicit custom
autograd wrappers should be used, mirroring `libs/geometry`.

## ProjectiveSensor Abstraction

The native kernel layer should define a neutral shared abstraction named
`ProjectiveSensor`.

This abstraction exists to support common projection and back-projection code
paths used by rasterization and other shared sensor kernels. It should not be a
camera base class with LiDAR retrofitted into it.

Design principles:

- Use `ProjectiveSensor` as the common abstraction name.
- Keep the abstraction primarily in the native C++/CUDA layer.
- Prefer a trait-based, concept-like interface that works in C++17.
- If the toolchain is intentionally upgraded later, this may become a real C++
  concept, but the design must not assume that today.
- Do not model LiDAR as a subclass of a camera base just to share dispatch.
- Keep the common contract small and projection-focused.

The common `ProjectiveSensor` contract should cover operations such as:

- world point -> sensor point under a pose or rolling-shutter trajectory,
- sensor point -> world ray,
- relative-frame-time lookup for rolling shutter,
- validity checks tied to the sensor's projective domain.

The common contract should not absorb family-specific APIs such as:

- camera-specific distortion parameter manipulation,
- LiDAR element-grid lookup,
- LiDAR angle-map preprocessing,
- family-specific configuration helpers.

At the native dispatch layer, the preferred pattern is:

1. Build a runtime `ProjectiveSensorKernelParamsVariant`.
2. Recover the concrete sensor type with `ProjectiveSensorFromKernelParams`.
3. Launch a CUDA kernel specialized on that concrete type.

This gives shared call sites a single dispatch path while preserving compile-time
specialization inside the kernel body.

## Parameter Pack Design

The kernel layer should expose parameter packs to Python through simple
dataclasses in `kernels/types.py`.

Representative types:

- `ShutterType`
- `ReferencePolynomial`
- `FThetaPolynomialType`
- `SpinningDirection`
- `ExternalDistortion`
- `NoExternalDistortion`
- `BivariateWindshieldDistortion`
- `OpenCVPinholeProjection`
- `OpenCVFisheyeProjection`
- `FThetaProjection`
- `RowOffsetStructuredSpinningLidarProjection`

These are working parameter packs for kernel execution, not necessarily the same
objects used by upstream config systems. `models/` is the correct place to
convert external parameter sources into these tensors.

### Rendering-Facing Sensor Parameters

The lower-level projection dataclasses above are sufficient as kernel-facing
parameter packs, but they are not yet the right abstraction for rasterization
integration.

Rasterization should consume a higher-level rendering-facing abstraction named
`ProjectiveSensorParameters`.

Purpose:

- group projection parameters with the additional sensor state required by
  rasterization,
- remove the need for rasterization to accept separate sensor coefficient,
  distortion, shutter, and image-size arguments,
- provide a single structured handoff from rendering code into
  `libs/sensors`.

Expected shape:

```python
@dataclass
class ProjectiveSensorParameters:
    projection: (
        OpenCVPinholeProjection
        | OpenCVFisheyeProjection
        | FThetaProjection
        | RowOffsetStructuredSpinningLidarProjection
    )
    viewmats: Tensor
    image_width: int | None = None
    image_height: int | None = None
    external_distortion: ExternalDistortion | None = None
    shutter_type: ShutterType = ShutterType.GLOBAL
```

Design rules:

- `ProjectiveSensorParameters` is the rendering-facing abstraction used by
  shared call sites such as rasterization.
- the lower-level projection dataclasses remain the backend-facing kernel
  parameter packs.
- rasterization should accept `sensor: ProjectiveSensorParameters` rather than
  separate arguments such as `camera_model`, `radial_coeffs`,
  `tangential_coeffs`, `thin_prism_coeffs`, `ftheta_coeffs`,
  `external_distortion_coeffs`, `rolling_shutter`, `viewmats`, or
  other sensor-specific pose arguments.
- camera projections require explicit image dimensions.
- lidar projections may infer image dimensions from their projection dataclass.
- pose state should travel through a single `viewmats` field.
- for `ShutterType.GLOBAL`, `viewmats` should have shape `(..., C, 4, 4)`.
- for rolling-shutter modes, `viewmats` should have shape `(..., C, 2, 4, 4)`,
  where the extra dimension stores the shutter start/end poses.
- validation of the accepted `viewmats` shape should be driven by
  `shutter_type`, not by a second pose argument.

This is intentionally a different abstraction layer from `models/`: it is still
stateless and execution-oriented, but rich enough for rendering integration to
avoid open-coding sensor family details.

Frame conversion rule:

- `ImageFrame` and `LidarFrame` should expose
  `to_projective_sensor_parameters()` to materialize the rendering-facing
  `ProjectiveSensorParameters` object from owned model + pose + timing state,
- this gives higher-level frame objects a direct path into rendering call sites
  such as rasterization without requiring callers to manually reassemble sensor
  parameters,
- the conversion should remain a thin composition layer over model-owned
  projection parameters and frame-owned pose/shutter state, not a second place
  where sensor math is implemented.

## CUDA Native Layout (`kernels/cuda/csrc/`)

Native sensor CUDA code should follow the same principle used by
`libs/geometry`: one `.cuh` per sibling `.cu`, where the header contains
device-side implementation and the translation unit owns launches, globals, and
exported CUDA entrypoints.

Shared native metaprogramming and trait contracts that do not naturally belong
to a single `.cu` translation unit should live under `include/`, not by forcing
LiDAR or camera code into the wrong module.

### `shutter_pose.cuh` / `shutter_pose.cu`

This pair owns the sensor-specific bridge between geometry pose conventions and
camera/LiDAR kernels that need shutter-aware, per-sample pose evaluation.

- `shutter_pose.cuh`:
  - device helpers for sensor-relative frame-time lookup,
  - shutter-aware pose interpolation glue from tensor inputs,
  - small fused helpers that combine sensor timing with geometry-aligned pose
    evaluation.
- `shutter_pose.cu`:
  - any exported helpers that need standalone CUDA entrypoints,
  - shared launch utilities if a reusable fused shutter-pose kernel is useful.

Boundary rule:

- this pair must not become a second generic pose or trajectory library,
- generic quaternion, SE3, and trajectory semantics remain owned by
  `libs/geometry`,
- code here should exist only when it is genuinely sensor-shaped and would be
  awkward to express as a generic geometry primitive.

### `camera.cuh` / `camera.cu`

- `camera.cuh`:
  - device helpers for camera model projection and inverse projection,
  - external distortion device math,
  - shutter-time mapping from pixel or image coordinates,
  - per-sample fused bodies for world->image and image->world operations.
- `camera.cu`:
  - `__global__` wrappers that delegate to `camera.cuh`,
  - launch helpers,
  - exported entrypoints such as:
    - `camera_rays_to_image_points_cuda`
    - `image_points_to_camera_rays_cuda`
    - `project_world_points_shutter_pose_cuda`
    - `project_world_points_mean_pose_cuda`
    - `image_points_to_world_rays_static_pose_cuda`

### `lidar.cuh` / `lidar.cu`

- `lidar.cuh`:
  - device helpers for element->angle lookup,
  - angle<->ray conversions,
  - row-offset and spinning-time logic,
  - iterative inverse projection helpers for world->sensor-angle projection.
- `lidar.cu`:
  - `__global__` wrappers and launch helpers,
  - exported entrypoints such as:
    - `generate_spinning_lidar_rays_cuda`
    - `elements_to_sensor_angles_cuda`
    - `inverse_project_spinning_lidar_cuda`

### Pattern Summary

| Location | Role |
| -------- | ---- |
| `*.cuh` | `#pragma once`, device functions, templates, and per-sample bodies |
| `*.cu` | host launch code, `__global__` kernels, PyTorch dispatch, exports |

This keeps reusable math close to the module that owns it, while staying
consistent with the repo's established CUDA extension structure.

## C++ Binding Layer

`kernels/cuda/ext.cpp` should mirror the role of `libs/geometry/kernels/cuda/ext.cpp`.

It is responsible for:

- declaring exported CUDA entrypoints,
- validating tensor device, dtype, shape, and contiguity,
- allocating outputs,
- exposing PyBind entrypoints for Python.

The extension boundary should remain explicit and low-level. Python-facing
semantic wrappers belong in `kernels/*.py`, not in `ext.cpp`.

Where a binding entrypoint supports both cameras and LiDARs through shared
projection logic, it should dispatch through the `ProjectiveSensor` variant
layer rather than open-coding repeated camera-model conditionals at each call
site.

## Build Configuration

`kernels/cuda/build.py` should mirror the JIT build pattern already used by
`libs/geometry/kernels/cuda/build.py`.

Recommended properties:

- extension name: `gsplat_sensors_cuda`
- sources:
  - `ext.cpp`
  - `csrc/shutter_pose.cu`
  - `csrc/camera.cu`
  - `csrc/lidar.cu`
- `-std=c++17`
- optional `DEBUG`, `FAST_MATH`, and `NVCC_FLAGS` environment toggles
- cached build-parameter invalidation identical to the geometry extension

Compatibility rule:

- The initial `ProjectiveSensor` abstraction should be implementable under
  C++17.
- Avoid depending on C++20-only library features unless the repo intentionally
  upgrades its extension toolchain.

Using the same build pattern reduces maintenance cost and makes sensors behave
like a first-class sibling to geometry.

## Model Layer Design

The model layer should provide the following repo-local structure:

- `models/frame.py`
  - base frame abstraction with ids, timestamps, pose state, and metadata
- `models/cameras.py`
  - `CameraModel` base class
  - `OpenCVPinholeCameraModel`
  - `OpenCVFisheyeCameraModel`
  - `FThetaCameraModel`
  - `ImageFrame`
- `models/lidars.py`
  - `LidarModel`
  - `RowOffsetStructuredSpinningLidarModel`
  - `LidarFrame`
- `models/return_types.py`
  - structured return dataclasses for higher-level APIs

These modules compose kernel-layer parameter packs instead of embedding kernel
logic directly.

## Naming and API Semantics

The naming scheme should match repo-local conventions:

- public stateless names describe sensor operations,
- backend exported names use explicit `_cuda` suffixes,
- backend backward entrypoints use `_bwd_cuda` suffixes where needed,
- common native abstractions use domain-specific names such as
  `ProjectiveSensor`.

Examples:

- public functional names:
  - `camera_rays_to_image_points`
  - `project_world_points_shutter_pose`
  - `generate_spinning_lidar_rays`
- backend forward entrypoints:
  - `camera_rays_to_image_points_cuda`
  - `project_world_points_shutter_pose_cuda`
  - `generate_spinning_lidar_rays_cuda`
- backend backward entrypoints:
  - `camera_rays_to_image_points_bwd_cuda`
  - `project_world_points_shutter_pose_bwd_cuda`

The public API should preserve:

- stable tensor shape contracts,
- explicit camera vs LiDAR naming,
- differentiable behavior where mathematically meaningful,
- rolling shutter semantics expressed as separate, named operations rather than
  hidden flags where that improves clarity.

## Testing Structure

Tests should be colocated with the public layers they validate:

- `functional/test_cameras.py`
- `functional/test_lidars.py`
- `models/test_models.py`

Coverage should include:

- forward correctness against known formulas or reference implementations,
- round-trip projection checks where appropriate,
- autograd validation for differentiable operations,
- rolling shutter regression coverage,
- parameter-pack validation and import behavior,
- backend loading behavior.

## Implementation Guidance

The safest implementation order is:

1. define `kernels/types.py` parameter packs and return types,
2. implement `kernels/cuda/build.py` and `ext.cpp`,
3. land `shutter_pose.cuh/.cu` for sensor-specific shutter-pose helpers,
4. implement camera CUDA kernels and Python wrappers,
5. implement LiDAR CUDA kernels and Python wrappers,
6. add `functional/` exports,
7. add `models/` wrappers once the stateless kernels are stable.

This sequence mirrors the repo's current geometry-first, kernels-first style.

## Design Constraints

- Public stateless sensor functions live in `functional/`.
- Backend implementation details live in `kernels/`.
- Native CUDA code follows the `.cuh` + sibling `.cu` pairing pattern.
- `libs/geometry` remains the canonical owner of public geometry semantics.
- `libs/sensors` should use the native CUDA/PyTorch extension path exclusively.
- Shared camera/LiDAR dispatch should be expressed through a neutral
  `ProjectiveSensor` abstraction, not camera inheritance.
- The package should be understandable from the directory layout alone:
  public API first, backend second, models third.
