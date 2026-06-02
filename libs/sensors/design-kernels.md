# Sensors Kernel Layer Design

## Overview

`libs/sensors/kernels/` is the backend implementation layer for `libs/sensors`.
It owns the native CUDA sources for camera projection, the C++ `torch::class_<>`
registrations for sensor parameter types, the per-pair Torch op bindings, the
Python autograd wrappers that drive them, the cross-family dispatch tables, and
the JIT build configuration for the `gsplat_sensors_cuda` extension.

It does not own the public stateless API (that is `libs/sensors/functional/`,
see `design-functional.md`) or the `nn.Module` surface that wraps it (that is
`libs/sensors/models/`, see `design-models.md`). The kernel boundary sees only
the unpacked tensors described in "Kernel-boundary pose contract" below; named
return types live in the functional layer and kernel Python wrappers return
raw tuples.

The data flow for one kernel call is short and uniform across every pair:

```text
gsplat_sensors.functional.<op>(...)
  -> kernels.projective_sensor_ops.<op>(...)       # cross-family dispatch
  -> kernels.cameras.ops.<op>(...)                  # public Python wrapper, pose unpacking
  -> _<Op>[Bivariate]?.apply(...)                   # torch.autograd.Function
  -> torch.ops.gsplat_sensors.<op>_opencv_pinhole_<distortion>(_backward)?
  -> gsplat_sensors::<op>_...(...)                  # C++ entry in camera_torch.cpp
  -> <op>_<...>_launch(...)                         # bridge prototype in camera_params.h
  -> <op>_<...>_kernel<<<...>>>(...)                # __global__ in camera_kernel.cu
  -> device math in camera_kernel.cuh + math.cuh
```

## Scope

The kernel layer covers a single sensor family (camera) at this revision:

- One camera projection model: OpenCV pinhole (radial + tangential + thin prism).
- Two external distortion models: `NoExternalDistortion` (identity) and
  `BivariateWindshieldDistortion` (packed `(42,)` coefficients, degree ≤ 2
  horizontal and ≤ 4 vertical bivariate polynomials).
- Seven shared camera ops dispatched by `(projection, distortion)` pair —
  `camera_rays_to_image_points`, `image_points_to_camera_rays`,
  `project_world_points_mean_pose`, `project_world_points_shutter_pose`,
  `image_points_to_world_rays_static_pose`,
  `image_points_to_world_rays_shutter_pose`,
  `pixel_grid_to_world_rays_shutter_pose` — plus the non-differentiable
  helper `generate_image_points` (pixel-centre coordinates).
- A rolling-shutter Newton solver inside `project_world_points_shutter_pose`,
  capped by `kMaxRollingShutterIterations = 12` in `camera_params.h`.
- Hand-written backward CUDA kernels per `(op, projection, distortion)` pair,
  exposed through one `torch.autograd.Function` per pair.

LiDAR is reserved for future work; this revision ships no LiDAR sources, no
`kernels/lidars/` Python package, and no LiDAR Torch ops. A rendering-facing
`ProjectiveSensorParameters` aggregator is also out of scope; the per-call
API takes the projection and distortion objects directly.

## Module Layout

```text
libs/sensors/kernels/
  __init__.py
  _backend.py                       # prebuilt-import then JIT fallback
  projective_sensor_ops.py          # cross-family Python dispatch tables
  test_projective_sensor_ops.py     # dispatch-table conformance test
  test_backend.py
  cameras/
    __init__.py                     # curated re-exports for tests / models
    ops.py                          # public wrappers + torch.autograd.Function glue
    types.py                        # torch.classes.* handles + ShutterType IntEnum
    windshield.py                   # from_components(...) factory for (42,) buffer
    test_ops.py                     # per-pair forward + backward + gradcheck
  common/
    pose.py                         # Pose / DynamicPose / Trajectory dataclasses
    utils.py                        # wxyz_to_xyzw, xyzw_to_wxyz, poses_to_matrix, ...
  cuda/
    __init__.py
    build.py                        # JIT build via torch.utils.cpp_extension.load
    ext.cpp                         # TORCH_LIBRARY + PYBIND11_MODULE
    csrc/
      camera_params.h               # bridge POD + per-pair _launch prototypes
      camera_kernel.cuh             # CUDA-only device math + per-thread structs
      camera_kernel.cu              # __global__ + _forward_launch definitions
      camera_kernel_backward.cu     # __global__ + _backward_launch definitions
      camera_torch.h                # Torch-only entry declarations
      camera_torch.cpp              # Torch-only entry definitions
      external_distortion_params.h  # bridge POD for distortion
      external_distortion_kernel.cuh  # CUDA-only header-only distortion math
      external_distortion_torch.h
      external_distortion_torch.cpp
      shutter_type.h                # source of truth for shutter enum
      math.cuh                      # float3 helpers + normalize3 bwd
```

There is no `include/projective_sensor.h` trait header — the kernel layer does
not use a C++ traits contract.

## Native split: bridge / CUDA-only / Torch-only

The hardest rule of the kernel layer is that the CUDA half and the Torch half
of every translation unit are kept strictly isolated; they meet only through a
small set of bridge headers. This keeps `nvcc` compilation of the kernels
independent of the ATen include graph and lets each `.cu` build without
pulling Torch templates through the compiler.

| File extension | Compiler | Allowed includes | Role |
| --- | --- | --- | --- |
| `*_params.h` | both | `<cstdint>`, sibling `*_params.h`, the forward-declared `cudaStream_t` | Bridge: POD structs + `_launch` prototypes. No CUDA, no Torch. |
| `*_kernel.cuh` | nvcc | `*_params.h`, other `*_kernel.cuh`, `<cuda_runtime.h>`, `quaternion.cuh` | CUDA-only device math, per-thread structs, `__device__` helpers. |
| `*_kernel.cu` / `*_kernel_backward.cu` | nvcc | matching `.cuh`, `<c10/cuda/CUDAException.h>` | `__global__` definitions, per-pair `_launch` definitions. |
| `*_torch.h` / `.cpp` | host C++ | `*_params.h`, ATen / `torch::CustomClassHolder` | Host structs, validators, Torch entry functions. |
| `ext.cpp` | host C++ | all `*_torch.h` | `TORCH_LIBRARY` + `PYBIND11_MODULE`. |

The bridge seam is exactly two declarations: an opaque `CUstream_st`
forward-declaration aliased as `cudaStream_t` in `camera_params.h`, plus the
POD `KernelParameters` structs. Every other type leaks only on one side or
the other.

## Three-tier C++ representation

Each registered C++ type appears in three forms — a Torch host struct that
owns its tensors, a kernel-parameter POD that the CUDA kernels consume by
value, and (for the projection / distortion types only) a per-thread register
struct loaded inside the kernel. The names are not perfectly uniform; the
table below states what each one is actually called.

| Registered type | Torch host (`*_torch.h`/`.cpp`) | Bridge POD (`*_params.h`) | Per-thread (`*_kernel.cuh`) |
| --- | --- | --- | --- |
| OpenCV pinhole | `gsplat_sensors::OpenCVPinholeProjection` | `OpenCVPinholeProjection_KernelParameters` | `OpenCVPinholeParams` |
| No external distortion | `gsplat_sensors::NoExternalDistortion` | `NoExternalDistortion_KernelParameters` (empty) | `NoExternalDistortion_Parameters` (empty no-op tag) |
| Bivariate windshield | `gsplat_sensors::BivariateWindshieldDistortion` | `BivariateWindshieldDistortion_KernelParameters` | `BivariateWindshieldParams` |

The naming asymmetry is intentional and worth flagging: the OpenCV pinhole
per-thread struct is `OpenCVPinholeParams` (not `OpenCVPinholeProjection_Parameters`),
and the bivariate per-thread struct is `BivariateWindshieldParams`. Only the
two empty / no-op slots follow the `_Parameters` suffix convention.

The Torch host struct stores per-component `at::Tensor` members (no packed
wire format) plus any frozen scalars / arrays such as the
`std::array<int64_t, 2> resolution` on `OpenCVPinholeProjection`. Its single
non-trivial method is `to_kernel_params()`, which collects raw `const float*`
pointers and scalar metadata into the bridge POD — no allocation, no device
transfer.

## Polymorphism and Dispatch

There is no C++ trait class, no `std::variant`, no runtime enum tag at the
C++/CUDA boundary. Polymorphism is realised in three layers, all visible
from Python:

**Type definition (C++ via `torch::class_<>`).** `ext.cpp` registers three
TorchScript classes (`OpenCVPinholeProjection`, `NoExternalDistortion`,
`BivariateWindshieldDistortion`) with named-kwargs `torch::init` constructors
(so Python can call them with `OpenCVPinholeProjection(focal_length=...,
principal_point=..., ...)`), per-component `def_readwrite` / `def_readonly`
properties, pickle hooks, and — for the projection only — a `transform(scale,
offset, new_resolution)` instance method that returns a freshly-constructed
projection with rescaled intrinsics. These classes are accessed from Python
as `torch.classes.gsplat_sensors.<Name>`.

**Per-pair Torch op binding (C++ via `m.def`).** For every shared op and every
`(projection, distortion[, _backward])` triple, `ext.cpp` registers one named
op. The naming pattern is
`<op>_<projection_snake>_<distortion_snake>(_backward)?`, for example
`camera_rays_to_image_points_opencv_pinhole_no_external` and its
`_backward` sibling. There are 13 forward ops (12 differentiable + the
non-differentiable `generate_image_points`) and 12 backward ops, all reachable
from Python as `torch.ops.gsplat_sensors.<name>`.

**Runtime dispatch (Python dicts in `projective_sensor_ops.py`).** Each shared
op has a module-level `dict[(projection_cls, distortion_cls), Callable]`
keyed on class identity. The seven tables are aggregated into
`_DISPATCH_TABLES`. `_lookup(...)` finds the entry by matching the registered
TorchScript class name (resolved via `script_class_name(obj)` because every
`torch.classes.*` instance reports `CustomClassHolder` as its Python class
name). Passing `external_distortion=None` raises `TypeError` — there is no
implicit `NoExternalDistortion`; callers must construct it explicitly.

`test_projective_sensor_ops.py::test_dispatch_table_keys_match_registered_pairs`
walks `_DISPATCH_TABLES` and asserts each table covers exactly
`REGISTERED_CAMERA_PROJECTIONS × REGISTERED_DISTORTIONS` (currently 1 × 2 = 2
entries per table). Adding a registered class without populating every table
fails this conformance test.

## Parameter-pack design

Camera parameter objects are stored as per-component `at::Tensor` members on
the host (e.g. `focal_length` is a `(2,)` float32 tensor, not a Python
`Tuple[float, float]`). Frozen metadata such as resolution and polynomial
degrees are POD (`std::array<int64_t, 2>`, scalar `int64_t`). This means
the per-component tensors participate directly in autograd: when the model
layer marks `requires_grad=True` on `focal_length`, the autograd backward
sees the same tensor object that the kernel parameter pack pointed at.

`to_kernel_params()` is the only host-side step between the user-facing
object and the kernel. It walks the host struct, takes `data_ptr<float>()`
from each tensor, copies frozen scalars, and emits a stack-allocated
`*_KernelParameters` POD. No tensor allocation occurs; the host object is
guaranteed to outlive the kernel launch.

Bivariate windshield distortion is the one place where multiple logical
polynomial banks share a single tensor for efficient device transfer. The
coefficient buffer is fixed at `(42,)` floats with layout
`[h_poly(6), v_poly(15), h_poly_inv(6), v_poly_inv(15)]`, and the per-axis
degrees (`h_poly_degree` in `[0, 2]`, `v_poly_degree` in `[0, 4]`) live
alongside it as POD. `kernels/cameras/windshield.py::from_components(h_poly,
v_poly, h_poly_inv, v_poly_inv, reference_polynomial)` is the ergonomic
factory: it validates triangular degrees, right-pads each polynomial to
`MAX_H_POLYNOMIAL_TERMS = 6` / `MAX_V_POLYNOMIAL_TERMS = 15`, concatenates,
and constructs the `BivariateWindshieldDistortion` TorchScript class.

## Autograd and the scratch contract

Every Torch entry forward returns a `scratch` tensor as the final element of
its result tuple, shaped `(N, K)` with K chosen per kernel (e.g. `(N, 6)` for
`camera_rays_to_image_points` no-external, `(N, 14)` for shutter pose). The
`torch.autograd.Function` wrapper in `kernels/cameras/ops.py` saves
`(primary_input, scratch)` via `ctx.save_for_backward(...)`; the matching
backward op consumes scratch instead of re-running OpenCV distortion math or
the rolling-shutter Newton iteration.

The C++ backward entries follow a fixed shape: projection + distortion
intrusive_ptr, saved tensors, scratch, then one `need_*_grad` bool per
differentiable input. The autograd `Function.backward` reads
`ctx.needs_input_grad` for each differentiable input and forwards those
booleans into the call. The `_backward_launch` passes `nullptr` for any
gradient the caller did not request, which the kernel checks before writing;
the host wrapper substitutes empty `(0,)` tensors for skipped slots and
interleaves `None` into the gradient list for non-differentiable inputs
(host projection/distortion objects, integer timestamps, scalar tolerances)
so the returned tuple matches the forward signature.

## Kernel-boundary pose contract

The public Python wrappers in `kernels/cameras/ops.py` take pose state as
`Pose`, `DynamicPose`, or `Trajectory` (all defined in `kernels/common/pose.py`)
and unpack to separated tensors immediately before `Function.apply(...)`.

- Static pose ops receive `(translation, rotation)` after `_unpack_static_pose`.
- Mean-pose and shutter-pose ops receive `(start_translation, start_rotation,
  end_translation, end_rotation)` after `unpack_dynamic_pose_components` or
  `_unpack_trajectory`.
- Time-varying poses are passed alongside `start_timestamp_us` /
  `end_timestamp_us` (int64) and a `ShutterType` int.

C++ ops never see view matrices or batched homogeneous pose stacks — only
plain translation `(N, 3)` and quaternion `(N, 4)` tensors. Quaternions are
**wxyz** at the public API, matching `libs/geometry`'s `wxyz_format` flag.
The convention helpers `wxyz_to_xyzw` / `xyzw_to_wxyz` live in
`kernels/common/utils.py`; the same module exposes `poses_to_matrix`, used
by the functional layer for downstream tools that still want a `(N, 4, 4)`
representation.

Pose interpolation (slerp + lerp) is performed on the device by the kernels
themselves, not in Python — see `trajectory_cuda::quat_slerp_pair_fwd_f` /
`quat_slerp_pair_bwd_f` in `libs/geometry/kernels/cuda/csrc/pose.cuh`.

## File-level reference

| Path | Owns |
| --- | --- |
| `kernels/_backend.py` | Prebuilt `gsplat_sensors_cuda` import, `GSPLAT_SENSORS_FORCE_JIT=1` override, JIT fallback wiring. |
| `kernels/projective_sensor_ops.py` | Seven `(projection, distortion)`-keyed dispatch dicts + the public `_DISPATCH_TABLES` registry and `_lookup` helper. |
| `kernels/test_projective_sensor_ops.py` | Dispatch-table conformance + a smoke test that the `None`-distortion path raises and the bivariate path returns a valid point. |
| `kernels/cameras/ops.py` | Public Python op wrappers, twelve `torch.autograd.Function` classes (one per `(op, distortion)` pair), and the pose / device / contiguity helpers (`_check_pair`, `_to_dev`, `_projection_on_device`, `_external_distortion_on_device`, `_select_camera_op`, `_unpack_static_pose`, `unpack_dynamic_pose_components`, `_unpack_trajectory`, `interpolate_dynamic_pose`, `relative_frame_times`). |
| `kernels/cameras/types.py` | `torch.classes.gsplat_sensors.*` Python handles, `ShutterType` IntEnum (import-time verified against C++), `ReferencePolynomial`, `REGISTERED_CAMERA_PROJECTIONS`, `REGISTERED_DISTORTIONS`, `script_class_name`, and the `CameraProjection` / `ExternalDistortion` type aliases. |
| `kernels/cameras/windshield.py` | `from_components(...)` factory + `MAX_H_POLYNOMIAL_TERMS` / `MAX_V_POLYNOMIAL_TERMS` constants. |
| `kernels/cameras/test_ops.py` | Per-pair forward and backward correctness tests plus PyTorch `gradcheck` on small inputs. |
| `kernels/common/pose.py` | `Pose`, `DynamicPose`, `Trajectory` dataclasses and `DynamicPose.from_static_pose` / `to_trajectory` helpers. |
| `kernels/common/utils.py` | `wxyz_to_xyzw`, `xyzw_to_wxyz`, `poses_to_matrix`, `valid_flags_to_indices`. |
| `kernels/cuda/build.py` | `get_build_parameters()` + `build_and_load_sensors_cuda()` JIT entry; `DEBUG` / `NVCC_FLAGS` / `VERBOSE` env handling; stale ninja lock cleanup. |
| `kernels/cuda/ext.cpp` | TorchScript class registrations, per-pair `m.def` bindings, `PYBIND11_MODULE` re-export of `ShutterType`. |
| `csrc/camera_params.h` | `OpenCVPinholeProjection_KernelParameters` POD + 13 forward + 13 backward `_launch` prototypes + `kMaxRollingShutterIterations`. Forward-declared `cudaStream_t`. |
| `csrc/external_distortion_params.h` | `NoExternalDistortion_KernelParameters` (empty) + `BivariateWindshieldDistortion_KernelParameters`. |
| `csrc/camera_kernel.cuh` | `OpenCVPinholeParams`, `ProjectionEval`, `DistortionResult`, `DistortionParamGrads`; all OpenCV pinhole device math + rolling-shutter time helper. SLERP is delegated to `libs/geometry/kernels/cuda/csrc/pose.cuh`. |
| `csrc/external_distortion_kernel.cuh` | `BivariateWindshieldParams` + `NoExternalDistortion_Parameters` no-op tag + header-only bivariate distortion math (no companion `.cu`). |
| `csrc/math.cuh` | `safe_nonzero`, `add3 / sub3 / scale3 / dot3`, `normalize3` forward and backward. |
| `csrc/shutter_type.h` | `gsplat_sensors::ShutterType` enum class (source of truth; verified at Python import). |
| `csrc/camera_kernel.cu` | Thirteen forward `__global__` kernels + matching `_forward_launch` definitions. |
| `csrc/camera_kernel_backward.cu` | Twelve backward `__global__` kernels + matching `_backward_launch` definitions; isolated TU so forward and backward can be compiled and reviewed independently. |
| `csrc/camera_torch.h` / `.cpp` | `OpenCVPinholeProjection` host struct + per-pair Torch entries + `generate_image_points` + the `check_*` validators called from `ext.cpp`. |
| `csrc/external_distortion_torch.h` / `.cpp` | `NoExternalDistortion` and `BivariateWindshieldDistortion` host structs + `to_kernel_params()` + `check_bivariate_windshield_distortion`. |

## Build configuration

The extension is built via `torch.utils.cpp_extension.load` driven by
`build_and_load_sensors_cuda` in `kernels/cuda/build.py`. The extension name
is `gsplat_sensors_cuda`. Both host C++ and nvcc compile at `-std=c++20`
(MSVC: `/std:c++20` with `/Zc:preprocessor` mirrored into nvcc via
`-Xcompiler`); host and nvcc flags are populated independently rather than
folded together on Windows. Sources split cleanly between compilers:

- `nvcc`: `camera_kernel.cu`, `camera_kernel_backward.cu`.
- Host C++: `ext.cpp`, `camera_torch.cpp`, `external_distortion_torch.cpp`.

`extra_include_paths` pulls in `libs/geometry/kernels/cuda/csrc` so the CUDA
TUs can `#include "quaternion.cuh"` from the geometry module without
duplicating quaternion device math.

Environment toggles, honoured at import time:

- `DEBUG=1` switches `-O3 -DNDEBUG` to `-g -O0`.
- `NVCC_FLAGS` is a space-separated list forwarded to nvcc.
- `VERBOSE=1` enables verbose JIT load.
- `GSPLAT_BUILD_LOCK_AGE_S` (default 1800s) bounds how long a stale ninja
  lock is tolerated before removal.

The build emits a JSON snapshot of every flag and source path into
`build_params.json` alongside the build directory. On flag change the
directory is wiped and rebuilt cleanly so a stale `.so` cannot survive a
compiler-flag toggle.

`_backend.py` prefers importing the prebuilt `gsplat_sensors_cuda` wheel on
process start and only falls through to the JIT path if that import fails or
if `GSPLAT_SENSORS_FORCE_JIT=1` is set. The JIT path is the dev workflow;
the wheel path is the deployment workflow.

## Naming and API Semantics

- Public Python op names (`camera_rays_to_image_points`,
  `project_world_points_mean_pose`, …) are family-shaped — there is no
  per-pair suffix at the Python boundary.
- Per-pair Torch op names use the explicit
  `<op>_<projection_snake>_<distortion_snake>(_backward)?` pattern (for
  example `image_points_to_world_rays_shutter_pose_opencv_pinhole_bivariate_windshield_backward`).
  This is verbose by design: every callable string in `m.def` declares its
  pair, and grep across `ext.cpp` shows the full registration table.
- `ShutterType` is defined once in `csrc/shutter_type.h` (C++ source of truth),
  re-exported through `PYBIND11_MODULE` in `ext.cpp`, and recreated as an
  `IntEnum` in `kernels/cameras/types.py`. The Python module runs
  `_verify_shutter_type_matches_cpp` at import time and refuses to load if
  the two definitions drift apart.
- `CameraProjection` is currently a type alias for `OpenCVPinholeProjection`
  because pinhole is the only registered projection; downstream dispatch
  code should iterate `REGISTERED_CAMERA_PROJECTIONS` rather than
  `isinstance`-ing against the alias.

## Testing Structure

- `kernels/cameras/test_ops.py` is the per-pair correctness suite. It runs
  every forward op on small problems, asserts the returned shapes and dtype,
  exercises the corresponding backward through `torch.autograd.grad`, and
  runs PyTorch `gradcheck` against a finite-difference reference where the
  op is differentiable.
- `kernels/test_projective_sensor_ops.py` is the dispatch-table conformance
  test: it walks `_DISPATCH_TABLES` and asserts every table covers exactly
  `REGISTERED_CAMERA_PROJECTIONS × REGISTERED_DISTORTIONS`.
- `kernels/test_backend.py` covers `gsplat_sensors_cuda` import behaviour
  (prebuilt vs JIT, `GSPLAT_SENSORS_FORCE_JIT`).
- Public-API-shape tests (return-type dataclasses, `Pose | DynamicPose`
  handling, the `CameraModel` surface) live one layer up in
  `libs/sensors/functional/` and `libs/sensors/models/` and are not
  duplicated here.

## Design Constraints

- CUDA and Torch translation units stay strictly isolated; they meet only
  through `*_params.h` bridge headers and a forward-declared `cudaStream_t`.
- Camera intrinsics and pose state are stored as per-component
  `at::Tensor` members on the host struct, never as packed scalar arrays —
  this keeps autograd visibility on every component.
- `to_kernel_params()` is pointer-collection only; it never allocates and
  never moves data between devices.
- `NoExternalDistortion` is explicit at the public API; `None` is rejected
  with a `TypeError`.
- Cross-family polymorphism is realised through Python `dict` dispatch
  tables in `projective_sensor_ops.py`. No `std::variant`, no runtime enum
  tag, no C++ trait class at the kernel boundary.
- Each `(op, projection, distortion[, _backward])` triple has its own
  `m.def` binding, its own `torch.autograd.Function`, and its own dispatch
  entry. Adding a registered class without populating every table fails the
  conformance test.
- Every forward returns a `scratch` tensor saved for the matching backward.
  Gradient pruning is wired through `ctx.needs_input_grad` into per-input
  `need_*_grad` booleans on the C++ backward entries; unused gradients
  become `nullptr` at kernel launch.
- Quaternions are wxyz at the public API; per-component pose tensors are
  the only pose representation that crosses the C++ boundary. Pose
  dataclasses are unpacked to tensors immediately before
  `Function.apply(...)`.
- The build is JIT by default with prebuilt wheel preference; flag changes
  trigger a clean rebuild via the `build_params.json` snapshot.
- `ShutterType` has exactly one definition in C++ (`csrc/shutter_type.h`)
  and is mirror-verified at Python import.
