# Shared Libraries Design

## Overview

The `libs/` directory contains repo-local libraries that package reusable CUDA
and PyTorch functionality behind clear Python APIs.

Current libraries include:

- `libs/geometry`
- `libs/sensors`

These libraries are intended to follow a common architectural pattern so that:

- public APIs are easy to discover,
- backend implementation details stay contained,
- native CUDA code follows consistent organization and naming,
- tests validate both API behavior and backend numerical correctness.

This document defines the shared design rules for libraries under `libs/`. Each
library may add domain-specific details in its own `design.md`, but those
library-specific documents should refine this contract rather than replace it.

## Design Goals

1. Keep the public Python API separate from backend implementation details.
2. Standardize how native CUDA extensions are organized, built, and exposed.
3. Encourage reuse through clear layering and boundary ownership.
4. Make it obvious where new functionality, tests, and native code should live.
5. Strive to make all functions differentiable by default, with documented
   exceptions where backward implementation does not make semantic or practical
   sense.
6. Preserve compatibility with the repo's active CUDA/PyTorch toolchain.

## Shared Module Pattern

Libraries under `libs/` should generally follow this shape:

```text
libs/<name>/
  design.md
  __init__.py
  pyproject.toml
  include/                 # optional; add only for shared low-level native headers
    ...
  kernels/
    __init__.py
    _backend.py
    <domain>_ops.py
    ...
    cuda/
      __init__.py
      build.py
      ext.cpp
      csrc/
        *.cu
        *.cuh
  functional/
    __init__.py
    ...
    test_*.py
  models/                  # optional; present when the library needs stateful nn.Module wrappers
    __init__.py
    ...
  components/              # optional; stateful wrappers that do not inherit nn.Module
    __init__.py
    ...
```

This pattern has four primary roles:

- `functional/`: public stateless Python API
- `kernels/`: backend dispatch, autograd glue, and native extension bindings
- `models/`: optional `torch.nn.Module` wrappers, frame structures, or trainable modules
- `components/`: optional stateful wrappers that intentionally do not inherit
  `torch.nn.Module`

Not every library must use every layer. The required default is:

- `functional/`
- `kernels/`

`models/` should be added only when the domain benefits from `torch.nn.Module`
wrappers or higher-level trainable abstractions.

`components/` may be added when the domain benefits from stateful wrappers or helper
objects that should not participate in `torch.nn.Module` semantics such as
parameter registration, module traversal, or `state_dict` ownership.

## Layer Responsibilities

### `functional/`

`functional/` is the canonical public stateless API surface.

It is responsible for:

- exposing the names downstream code should import,
- defining argument, shape, dtype, and return-value contracts,
- expressing operations in domain terms instead of backend terms,
- delegating execution to `kernels/`.

Rules:

- Public API definitions belong here.
- User-facing examples and documentation should point here.
- Code here should not know about CUDA launch details or native pointer layout.

### `kernels/`

`kernels/` is the backend implementation layer.

It is responsible for:

- loading and validating the native extension,
- defining `torch.autograd.Function` wrappers where needed,
- owning Python-side backend dispatch,
- defining backend-facing parameter packs and return structures,
- containing native C++ and CUDA source code under `kernels/cuda/`.

Rules:

- `kernels/` may expose helpers used by `functional/`.
- `kernels/` is allowed to be lower-level and more implementation-oriented.
- Backend entrypoints should remain explicit rather than hidden behind dynamic
  magic.

### `models/`

`models/` is the optional `torch.nn.Module` layer.

It is responsible for:

- `nn.Module` wrappers around functional or kernel-level operators,
- trainable parameter ownership,
- higher-level domain structures such as frames, grouped sensor state, or model
  objects,
- conversion from external parameter/config formats into kernel-layer working
  representations.

Rules:

- `models/` may depend on `functional/` and `kernels/`.
- `models/` should not become the only way to access the library's core math.
- Core stateless operations must remain available without requiring model-layer
  objects.

### `components/`

`components/` is the optional non-`nn.Module` stateful layer.

It is responsible for:

- stateful wrappers that should not inherit `torch.nn.Module`,
- helper objects that manage runtime state, caches, indexing structures, or
  grouping logic,
- higher-level composition objects that do not own trainable parameters through
  `nn.Module`.

Rules:

- `components/` may depend on `functional/`, `kernels/`, and `models/` where needed.
- `components/` should not become a second public stateless API surface.
- Objects that need `nn.Module` semantics belong in `models/`, not `components/`.
- Objects that are stateful but should not register parameters or buffers should
  prefer `components/`.

## Dependency Boundaries

Libraries should follow this direction:

```text
functional/  ->  kernels/  ->  kernels/cuda/
models/      ->  functional/ and/or kernels/
components/  ->  functional/ and/or kernels/ and optionally models/
include/     ->  native code only
```

Rules:

- `functional/` may depend on `kernels/`.
- `models/` may depend on `functional/` and `kernels/`.
- `components/` may depend on `functional/`, `kernels/`, and optionally `models/`.
- native code may depend on `include/`.
- `include/` should not depend on Python layers.
- Public API definitions should not live under `kernels/`.

This keeps backend concerns from leaking into the public API surface.

## Native CUDA Organization

Libraries with native CUDA code should follow the same top-level extension
structure:

- `kernels/cuda/build.py`: JIT build and load logic
- `kernels/cuda/ext.cpp`: C++ binding layer and PyBind exports
- `kernels/cuda/csrc/*.cuh`: device-side helpers and per-row/per-element logic
- `kernels/cuda/csrc/*.cu`: host launch code, `__global__` kernels, dispatch,
  and exported CUDA entrypoints

### `.cuh` / `.cu` Pairing

The preferred pattern is one sibling `.cuh` per top-level `.cu`.

Rules:

- `*.cuh` contains `#pragma once`, device helpers, templates, and reusable
  native implementation bodies.
- `*.cu` contains `__global__` kernels, launch helpers, dispatch, and exported
  `*_cuda` / `*_bwd_cuda` entrypoints.
- Shared native support headers that do not naturally belong to one translation
  unit may live under `include/`.

This keeps reusable device math close to the owning module while keeping host
launch logic localized.

## Build and Toolchain Rules

Libraries with native extensions should mirror the same build model unless there
is a strong reason not to:

- JIT loading via `torch.utils.cpp_extension`
- stable extension naming per library
- cached build-parameter invalidation
- explicit compile flags in `build.py`

Compatibility rules:

- Design for the repo's active toolchain first.
- Do not assume a newer C++ language level than the extension build actually
  uses.
- Avoid introducing C++20-only library dependencies into shared abstractions
  unless the repo intentionally upgrades the extension toolchain.

## Naming and API Semantics

Shared naming rules:

- public stateless functions use domain-specific, user-facing names,
- backend exported entrypoints use explicit `_cuda` suffixes,
- backend backward entrypoints use `_bwd_cuda` suffixes when applicable,
- internal native abstractions should use precise domain language rather than
  generic catch-all names.

Public APIs should preserve:

- stable shape and dtype contracts,
- explicit semantics for differentiable vs non-differentiable outputs,
- consistent import surfaces through `functional/`,
- predictable backend behavior.

## Return Types and Status Outputs

Libraries should be explicit about which outputs are:

- differentiable primary outputs,
- non-differentiable validity or status flags,
- metadata outputs such as timestamps or indices.

Rules:

- Status outputs such as validity masks or out-of-bounds flags should be clearly
  documented as non-differentiable where applicable.
- Structured return dataclasses are encouraged when an operation returns more
  than one logically distinct output.
- Public return shapes must be stable and documented.

## Boundary Ownership Between Libraries

Each library should own its domain primitives clearly.

Examples:

- `libs/geometry` owns generic quaternion, pose, SE3, and trajectory semantics.
- `libs/sensors` may consume those semantics and add sensor-specific projection
  and shutter-aware glue.

Rule:

- Generic reusable math belongs in the library that owns that math domain.
- Domain-specific fusion or convenience helpers should stay in the higher-level
  library unless they become clearly reusable across domains.

## Testing Structure

Tests for library public APIs should be colocated near the public API layer,
typically under `functional/`.

Library-specific higher-level tests may also live under:

- `models/` when model-layer behavior needs direct validation,
- `components/` when non-`nn.Module` stateful wrappers need direct validation,
- `tests/` if there is a compelling cross-library or system-level reason.

Rules:

- Tests should be organized around user-visible functionality, not internal file
  locations.
- GPU library tests do not need to be hidden behind implementation-private
  paths.
- Test placement should make it obvious which public contract is being verified.

## Testing Requirements

Libraries under `libs/` must verify both API behavior and backend correctness.

This includes validation of both forward behavior and differentiable behavior
where the operator is documented as differentiable.

### Forward Correctness

The test suite should verify that public operators:

- return the expected shapes and dtypes,
- satisfy documented numerical tolerances,
- preserve established semantics,
- behave correctly for representative edge cases.

Important edge cases include:

- empty batches when supported,
- degenerate but valid inputs,
- out-of-range or invalid cases that should produce status flags,
- domain-specific numerical corner cases already known to be fragile.

Branch-coverage rule:

- When operator behavior depends on explicit branching, tests should cover all
  meaningful branches of the forward path.
- Branching tied to status outputs, optional parameters, special-case formulas,
  or numerically sensitive fallback paths should not be left untested.

### Backward Validation

For differentiable operators, tests must do more than verify that gradients are
finite.

Required validation:

- gradients must exist where the operator is documented as differentiable,
- gradients must be numerically close to a trusted reference implementation when
  such a reference is well-defined,
- backend backward entrypoints must be validated directly or through public API
  behavior,
- non-differentiable outputs must not be treated as if they were part of the
  differentiable contract.

Backward branch-coverage rule:

- When backward behavior depends on branching in the implementation, tests
  should exercise all meaningful backward branches, not just the dominant or
  easiest path.
- If forward branching induces distinct backward behavior, the corresponding
  test inputs should cover those cases explicitly.

Reference-gradient rule:

- When a pure PyTorch reference implementation is practical and semantically
  equivalent, gradient tests should compare against that reference rather than
  only checking finiteness.
- "Close" must mean explicit tolerance-based agreement, not qualitative
  similarity.
- A finiteness-only test is not sufficient when a meaningful PyTorch reference
  exists.

Representative acceptable references include:

- direct PyTorch tensor compositions,
- analytically equivalent decompositions built from tested lower-level ops,
- finite-difference checks when an exact PyTorch formulation is not practical.

Preferred order of evidence:

1. PyTorch reference gradient comparison
2. Direct finite-difference validation
3. Finiteness-only checks as a fallback for cases without a better reference

### Numerical Regression Coverage

Continuous operators should have regression-oriented coverage to guard against
 backend drift.

This includes:

- randomized comparisons against references,
- fixed regression seeds for previously observed mismatches,
- tolerance checks for important representative inputs,
- targeted tests for branch-heavy edge cases.

### Import and Backend Coverage

The test suite should verify that:

- public imports from the `functional/` layer succeed,
- backend loading succeeds on supported configurations,
- failures are clear when backend requirements are not met,
- optional fast paths and fallback paths behave consistently where both exist.

## Documentation Requirements

Each library should maintain its own `design.md` that documents:

- library-specific goals,
- domain boundaries,
- file layout,
- public APIs,
- testing expectations beyond the shared baseline.

The shared `libs/design.md` document defines the common contract. Individual
library design docs should reference or follow it rather than redefining the
same base rules in conflicting ways.

## Design Constraints

- `functional/` is the canonical public stateless API layer.
- `kernels/` contains backend dispatch and native extension integration.
- `models/` is optional and is reserved for `torch.nn.Module`-based stateful
  wrappers.
- `components/` is optional and is reserved for stateful wrappers that should not
  inherit `torch.nn.Module`.
- Native code organization should follow the `.cuh` + sibling `.cu` pattern.
- Testing must validate gradient correctness against PyTorch references where
  that comparison is meaningful.
- Directory structure should make public, backend, and shared-support code easy
  to distinguish.
