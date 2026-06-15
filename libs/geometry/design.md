# Geometry Module Design

## Overview

`libs/geometry` is organized around three package roles:

- `functional/` defines the public Python geometry API.
- `kernels/` implements backend dispatch, autograd glue, and native CUDA
  integration.

This separation keeps the public API easy to discover while keeping backend
details and shared low-level helpers in clearly bounded locations.

`libs/geometry` follows the repository-wide `libs/**` model: it is internal to
this monorepo and is not independently versioned or released as a standalone package.

## Design Goals

1. Present a clear public geometry API centered on functional operators.
2. Keep CUDA loading, build logic, and autograd implementation details out of
   the primary user-facing namespace.
3. Provide a dedicated home for reusable kernel-side helpers.
4. Preserve stable geometry semantics, tensor conventions, and backend behavior.

## Public API validation

Geometry operators that dispatch into native CUDA code are validated at the
**Python API layer** (in the kernel wrapper modules that implement the callable
surface used by `functional/`, e.g. `kernels/quaternion_ops.py` and
`kernels/pose_ops.py`) **before** calling into the extension. This is
intentional even when checks overlap with C++/CUDA `TORCH_CHECK` guards: users
get consistent, explicit `TypeError` / `ValueError` messages without paying for
kernel launch or opaque backend errors when the contract is violated.

Required categories of validation for those Python wrapper entrypoints (the
callable surface consumed by `functional/`):

1. **Type** — Arguments that must be tensors are `torch.Tensor` instances;
   scalar parameters keep their documented Python types (e.g. `quat_lerp` uses
   a scalar `t`, while `quat_slerp` and `quat_manifold_interp` accept a scalar or
   tensor `t` per docstring).
2. **Shape and layout** — Last-dimension contracts (e.g. quaternions `(…, 4)`,
   translations and vectors `(N, 3)`, rotations `(N, 4)`, matrices `(N, 4, 4)` or
   `(N, 16)`) and compatible batch sizes for paired inputs.
3. **Dtype** — Floating-point requirements per operator (e.g. quaternion and SE3
   ops: `float32` / `float64`; trajectory helpers bound in `ext.cpp`: `float32`
   for pose and time tensors where the native path is float32-only).
4. **Device** — CUDA-only operators require CUDA tensors; multi-input functions
   require a single shared device (and dtype) where the kernels do.

Validation stays on those **wrapper** functions (the layer `functional/` calls
into), not on every internal `torch.autograd.Function`, unless a future change
explicitly needs it.

## Module Layout

```text
libs/geometry/
  design.md
  __init__.py
  pyproject.toml
  include/                # optional; add only for shared low-level headers
    ...
  kernels/
    __init__.py
    _backend.py
    quaternion_ops.py
    pose_ops.py
    cuda/
      __init__.py
      build.py
      ext.cpp
      csrc/
        quaternion.cu
        quaternion.cuh
        pose.cu
        pose.cuh
  functional/
    __init__.py
    quaternion.py
    pose.py
    test_quaternion.py
    test_pose.py
```

## Package Roles

### `functional/`

`functional/` is the public Python surface of the geometry module.

It is responsible for:

- exporting geometry operations such as quaternion, pose, and trajectory
  functions,
- defining the API-level contract for arguments, shapes, dtypes, and return
  values,
- providing the import path downstream code should use for geometry operators,
- delegating execution to backend implementations in `kernels/`.

Code in `functional/` is written from the point of view of the geometry API,
not from the point of view of CUDA implementation details.

### `kernels/`

`kernels/` is the backend implementation layer.

It is responsible for:

- loading and validating the `gsplat_geometry_cuda` extension,
- defining `torch.autograd.Function` wrappers and direct backend dispatch,
- owning native extension build configuration,
- containing native CUDA and C++ sources,
- exposing internal implementation helpers consumed by `functional/`.

`kernels/` is part of the module architecture, but it is not the primary
conceptual entry point for downstream users.

## Public API Surface

The public geometry API is exposed through `gsplat_geometry.functional`.

The top-level package `gsplat_geometry` advertises only `functional` in
`__all__` and does not import or re-export `kernels` at package initialization.
Downstream code should treat `kernels/` as implementation detail: operators are
imported from `functional/`, while submodule paths such as
`gsplat_geometry.kernels.pose_ops` remain available for in-repo tests and
internal callers who need autograd `Function` types or other backend symbols.

The `kernels/__init__.py` module intentionally does not barrel-re-export
operators or submodules; it documents the backend role only.

Example import style:

```python
from gsplat_geometry.functional import quat_multiply, se3pose_transform_point
```

The top-level package should make the functional surface easy to discover:

```python
import gsplat_geometry.functional as geometry
```

The public API is defined in terms of geometry operations and their semantics,
not in terms of backend modules or native entrypoints.

## Dependency Boundaries

The module follows this dependency direction:

```text
functional/  ->  kernels/  ->  kernels/cuda/
```

Rules:

- `functional/` may depend on `kernels/`.
- native sources under `kernels/cuda/` may depend on `include/`.
- `include/` does not depend on `functional/`.
- public API definitions do not live under `kernels/`.

This keeps the package layered in one direction and prevents backend concerns
from leaking into the user-facing API surface.

## Functional Organization

The functional layer is organized by geometry concept.

- `functional/quaternion.py` contains quaternion operations.
- `functional/pose.py` contains pose and trajectory operations.
- `functional/__init__.py` provides a curated export surface.

Functions in this layer are the canonical names for geometry operations.
Documentation, examples, and downstream imports should point here.

## Kernel Organization

The kernel layer is organized by implementation concern.

- `kernels/__init__.py` is a namespace marker only (no curated re-exports).
- `kernels/_backend.py` loads the native extension, preferring prebuilt import and falling back to JIT.
- `kernels/quaternion_ops.py` contains quaternion backend wrappers (including
  Python-level input validation on the public wrappers), autograd
  implementations, and dispatch to CUDA.
- `kernels/pose_ops.py` contains pose and trajectory backend wrappers (including
  Python-level validation on the public wrappers), autograd implementations,
  and dispatch to CUDA.
- `kernels/cuda/` contains build logic and native source code.

This layer is where explicit forward and backward kernel entrypoints are bound
to Python.

## CUDA native layout (`kernels/cuda/csrc/`)

Native geometry CUDA code is split by **one `.cuh` per sibling `.cu`** (same
basename). Each `.cu` file includes only its matching header; the header
holds the device-side implementation that kernels need, and the translation unit
holds launch helpers, `__global__` kernels, and the extension entry points
(`*_cuda`, `*_bwd_cuda`) bound from C++/Python.

**`quaternion.cuh` / `quaternion.cu`**

- `quaternion.cuh`: template and small `__device__` helpers (e.g. `cross3`,
  `quat_multiply_impl`, `quat_rotate_vector_*_impl`, `quat_to_matrix_fwd_write`,
  `quat_normalize_safe_*_write`), per-row `*_fwd_device` / `*_bwd_device` bodies
  (which delegate to those scalars where applicable), and related constants/traits
  (`QuatNormEps`, etc.). Intended to be included from quaternion kernels and,
  where needed, from pose code; `pose.cuh` does not duplicate these primitives.
- `quaternion.cu`: `__global__` wrappers that delegate to the `*_device`
  functions, `launch_*` templates, `AT_DISPATCH_FLOATING_TYPES`, and the exported
  quaternion operators.

**`pose.cuh` / `pose.cu`**

- `pose.cuh` includes `quaternion.cuh` so pose and trajectory kernels can reuse
  quaternion device math **without** introducing a third shared-only device
  header. It adds pose-specific device helpers (e.g. Shepperd matrix→quaternion,
  SE3 transforms, `trajectory_cuda` orchestration using quaternion.cuh templates)
  and the pose `*_device` entry bodies.
- `pose.cu` mirrors the quaternion pattern: globals, launches, and exported
  pose/trajectory CUDA entry points.

**Pattern summary**

| Location | Role |
| -------- | ---- |
| `*.cuh` | `#pragma once`, device functions and templates callable from device code |
| `*.cu` | Host-side launch code, `__global__` kernels, PyTorch dispatch, exports |

This keeps reusable math colocated with the module that owns it (quaternion),
while preserving a strict one-header-per-translation-unit pairing for the top-level
CUDA sources.

## Naming and API Semantics

The geometry module uses function names that describe mathematical operations at
the Python API layer and explicit backend entrypoints at the kernel layer.

Examples:

- public functional names: `quat_multiply`, `quat_rotate_vector`,
  `se3pose_transform_point`
- backend forward entrypoints: `<op>_cuda`
- backend backward entrypoints: `<op>_bwd_cuda`

The public API preserves:

- established quaternion, pose, and trajectory function names,
- existing tensor shape conventions,
- CUDA execution expectations for native-backed operators,
- autograd semantics for differentiable operations.

## Testing Structure

CUDA geometry tests are colocated under `functional/` next to the public API
they exercise (`test_quaternion.py`, `test_pose.py`). Root `pytest` discovery
(`pytest.ini` `testpaths = tests`) does not collect them; GPU CI and local
workflows pass `libs/geometry/functional` explicitly alongside `tests/` when the
full surface should run.

The test suite covers:

- forward correctness,
- autograd behavior,
- numerical behavior for important edge cases,
- public import and API behavior,
- backend loading behavior where relevant to correctness.

Tests are organized around user-visible geometry functionality rather than the
internal location of implementation code.

## Testing Requirements

The geometry module test suite verifies both API-level behavior and backend
correctness.

### Forward Correctness and API Compatibility

The test suite verifies that public geometry operators:

- produce the expected output shapes and dtypes,
- satisfy the numerical tolerances defined for each operation,
- preserve established quaternion, pose, and trajectory semantics,
- behave correctly for important edge cases.

Important edge cases include:

- quaternion sign ambiguity,
- `wxyz_format` handling for inverse matrix conversion,
- out-of-bounds flags and similar non-differentiable status outputs,
- empty-batch and degenerate-value behavior where supported.

### Backward Validation

For differentiable operators, the test suite verifies that autograd behavior is
correct and stable.

This includes:

- non-zero gradient flow for operators that are expected to be differentiable,
- direct gradient checks against reference PyTorch implementations where those
  references are well-defined,
- explicit validation of backend backward entrypoints for operations exposed
  through the public API.

Representative gradient-reference coverage includes operators such as:

- `quat_multiply`,
- `quat_rotate_vector`,
- `quat_to_matrix`,
- `se3pose_transform_point`,
- `se3pose_to_matrix`,
- `se3pose_from_matrix`.

### Numerical Regression Coverage

Selected continuous operators should also have numerical regression coverage to
guard against backend drift.

This includes:

- finite-difference checks for representative continuous operations,
- randomized comparisons against reference implementations,
- regression seeds for any previously observed numerical mismatches.

### Import and Backend Coverage

The test suite verifies that:

- public imports from `gsplat_geometry.functional` succeed,
- backend loading and extension validation behave correctly on supported CUDA
  configurations,
- the module fails clearly when native backend requirements are not met.

## Design Constraints

- Public geometry functions are defined in `functional/`.
- Backend implementation details are defined in `kernels/`.
- Shared implementation helpers are defined in `include/`.
- The package remains conceptually organized around geometry concepts first and
  implementation details second.
- Directory structure should make it clear which code is public, which code is
  backend-specific, and which code is shared support code.