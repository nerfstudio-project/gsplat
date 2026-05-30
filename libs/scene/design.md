# Scene Design

## Scene Types

`GaussianScene` is the training-oriented Gaussian container. It owns the
parameter dictionary used by optimizer and strategy code (`means`, `quats`,
`scales`, `opacities`, and either RGB or SH color tensors) and keeps those
tensors in the source representation expected by training paths. Topology hooks
mutate this owned state when gaussians are duplicated, split, removed,
relocated, sampled, or permuted.

`GaussianInferenceScene` is the packed QSO inference container. It owns detached
render-time tensors in viewer/kernel layout: planar means, packed
quaternion/scale/opacity (`qso_packed`), packed colors, SH metadata, and
component indexing. It can be built from a `GaussianScene` or from already
activated tensors, then used by inference render paths that expect packed
storage. The conversion boundary is one-way: training code owns the
editable Gaussian parameters, while inference scenes own compact packed tensors
and do not participate in autograd.

### Scene Type Summary

| Scene Type                 | Representation           | Tensor Layout                         | Typical Use                  |
| -------------------------- | ------------------------ | ------------------------------------- | ---------------------------- |
| `GaussianScene`            | Raw log-space parameters | `splats` ParameterDict, row-aligned N | Training, optimization       |
| `GaussianInferenceScene`   | Activated, packed QSO    | `means_planar` [3,N], `qso_packed` [N,8] fp16, `colors_packed` varies | Inference rendering  |

Supported conversions:

- `GaussianScene` → `GaussianInferenceScene` via `from_gaussian_scene()` (one-way, single-component only)
- Raw tensors → `GaussianInferenceScene` via `from_gaussian_tensors()` (pre-activated inputs)
- `GaussianInferenceScene` → training scene: not supported (the conversion boundary is one-way)

## Module Layout

```text
libs/scene/
  design.md
  __init__.py
  pyproject.toml
  sh_compression.py
  test_package_imports.py
  kernels/
    __init__.py
    _backend.py
    gaussian_inference_ops.py
    cuda/
      __init__.py
      build.py
      ext.cpp
      csrc/
        gaussian_scene_pack.cpp
        gaussian_scene_pack.cuh
  functional/
    __init__.py
    gaussian_inference.py
    test_gaussian_inference.py
  components/
    __init__.py
    base.py
    gaussian_scene.py
    gaussian_inference_scene.py
    test_gaussian_scene.py
    test_gaussian_inference_scene.py
```

## Package Roles

### `functional/`

`functional/` is the public Python surface of the scene module.

It exports scene operators such as `pack_gaussian_inference_scene` and defines the
API-level contract for arguments, shapes, dtypes, and return values.  Downstream
code should import from `gsplat_scene.functional`.

### `kernels/`

`kernels/` is the backend implementation layer.

- `_backend.py` loads the `gsplat_scene_cuda` extension (prebuilt then JIT).
- `gaussian_inference_ops.py` owns Python-level input validation and native dispatch.
- `kernels/cuda/` contains `build.py`, `ext.cpp` (PYBIND11), and `csrc/*.cu`.

`kernels/` is not a curated user-facing API.

### `components/`

Scene class implementations (`Scene`, `GaussianScene`, `GaussianInferenceScene`) and
their tests.  Components call into `kernels/` for CUDA work.

### Dependency direction

```text
functional/  ->  kernels/  ->  kernels/cuda/
components/  ->  kernels/
```

Scene CUDA ops live in `libs/scene/kernels/`.

## Goal

Provide a minimal shared scene abstraction for the current Gaussian-based paths in:

- `examples/av_trainer.py`
- `examples/simple_trainer.py`
- `examples/simple_trainer_2dgs.py`

The design is intentionally small. The base scene defines only the common contract, and `GaussianScene` owns Gaussian-specific storage and bookkeeping.

## Current Implementation

The scene code lives under:

- `libs/scene/components/base.py`
- `libs/scene/components/gaussian_scene.py`

The package exports:

```python
from gsplat_scene import Scene, GaussianScene, GaussianInferenceScene, SHCompressionMode
from gsplat_scene.functional import pack_gaussian_inference_scene
```

## Base `Scene`

`Scene` is a minimal abstract base class, not a concrete container.

```python
class Scene(ABC):
    id: str

    @abstractmethod
    def put(self, name: str, component: object) -> None: ...

    @abstractmethod
    def get(self, component: str) -> object: ...

    # Topology hooks (no-op defaults)
    def on_duplicate(self, sel: Tensor) -> None: ...
    def on_split(self, sel: Tensor, rest: Tensor) -> None: ...
    def on_remove(self, remove_mask: Tensor) -> None: ...
    def on_relocate(self, dead_indices: Tensor, sampled_indices: Tensor) -> None: ...
    def on_sample_add(self, sampled_indices: Tensor) -> None: ...
    def on_permute(self, order: Tensor) -> None: ...
```

The base class establishes that scenes:

- have a string `id` used as a key on a Stage
- support named insertion and retrieval
- provide topology hooks (no-op by default) called by strategy ops

Subclasses are free to store components however they want.

This is deliberately narrower than a mapping-like interface:

- no shared storage in the base class
- no `names()`
- no `items()`
- no generic `Scene[T]`

## `GaussianScene`

`GaussianScene` is the current concrete implementation.

```python
class GaussianScene(Scene):
    def __init__(self, id: str) -> None: ...

    splats: torch.nn.ParameterDict
    signal: dict[str, torch.Tensor]
    component_names: list[str]
    component_index: torch.Tensor
```

`id` is a required positional argument with no default (`GaussianScene()` raises
`TypeError`); the four sidecar fields are zero-initialized in `__init__`.

It owns:

- raw trainable Gaussian tensors in `splats`
- optional per-Gaussian sidecar tensors in `signal`
- exclusive component membership via `component_names` and `component_index`
- topology hooks that keep sidecars aligned during strategy mutations

It does not own:

- dataset state
- rendering logic
- activation logic such as `exp(scales)` or `sigmoid(opacities)`
- optimizer state
- trainer-specific modules like pose adjustment, appearance, or post-processing

## Storage Model

All per-Gaussian tensors remain row-aligned on the leading dimension `N`.

```text
row i
  splats["means"][i]
  splats["scales"][i]
  splats["quats"][i]
  splats["opacities"][i]
  splats["colors"][i] or splats["sh*"][i] or splats["features"][i]
  signal[*][i]
  component_index[i]
```

This stays consistent with the existing trainers.

## Required Splat Keys

`GaussianScene.validate()` requires these geometry keys whenever `splats` is non-empty:

- `means`
- `scales`
- `quats`
- `opacities`

Additional keys are trainer-dependent and are stored without reinterpretation.

Examples:

- AV trainer:
  - `colors`
- Simple trainer SH path:
  - `sh0`, `shN`
- Simple trainer deferred appearance path:
  - `features`, `colors`

## Components

Components are exclusive. Each Gaussian row belongs to exactly one component id.

Canonical storage:

```python
scene.component_names = ["scene", "road", "car"]
scene.component_index = LongTensor[N]
```

Example:

```text
component_names = ["scene", "road", "car"]
component_index = [0, 0, 2, 1, 1, 0]
```

This means:

- rows `0`, `1`, `5` belong to `scene`
- rows `3`, `4` belong to `road`
- row `2` belongs to `car`

## Public API

The current `GaussianScene` API is:

```python
class GaussianScene(Scene):
    def put(self, name: str, splats: torch.nn.ParameterDict) -> None: ...

    @classmethod
    def from_splats(
        cls,
        splats: torch.nn.ParameterDict,
        id: str,
        signal: dict[str, torch.Tensor] | None = None,
    ) -> "GaussianScene": ...

    def validate(self) -> None: ...
    def num_gaussians(self) -> int: ...
    def get(self, component: str | int) -> dict[str, object]: ...

    def state_dict(self) -> dict[str, object]: ...

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "GaussianScene": ...

    def on_duplicate(self, sel: torch.Tensor) -> None: ...
    def on_split(self, sel: torch.Tensor, rest: torch.Tensor) -> None: ...
    def on_remove(self, remove_mask: torch.Tensor) -> None: ...
    def on_relocate(
        self,
        dead_indices: torch.Tensor,
        sampled_indices: torch.Tensor,
    ) -> None: ...
    def on_sample_add(self, sampled_indices: torch.Tensor) -> None: ...
    def on_permute(self, order: torch.Tensor) -> None: ...
```

### `put(name, splats)`

`put()` adds a named component backed by a `torch.nn.ParameterDict`.

Behavior:

- the first `put()` stores the passed `ParameterDict` directly
- later `put()` calls append rows by concatenating the existing splat tensors with the new component tensors
- appended rows get a new component id in `component_index`
- existing `signal` tensors are zero-padded for the appended rows so signal stays aligned with `splats` (this is what makes `from_splats(..., signal=...)` composable with later `put()` calls)
- `validate()` is called at the end of every `put()`

Errors raised:

- `ValueError("component name must not be empty")` if `name` is empty
- `ValueError("Component {name!r} already exists in scene")` if `name` is a duplicate
- `ValueError("component splats must not be empty")` if the `ParameterDict` is empty or missing `"means"`
- the splat-key concatenation iterates over keys of the *existing* `splats`; new keys present in `component` but not in `self.splats` are silently dropped, and missing keys raise `KeyError` from the underlying `torch.cat`

Limitations:

- `put()` only accepts splats; signal cannot be added per-component (callers seed it via `from_splats(..., signal=...)`)
- appended components must match the existing splat-key layout
- `put()` is init-only: calling it after optimizers have been created on `scene.splats` will orphan the old `Parameter` objects (the docstring on the method warns about this)

### `get(component)`

`get()` returns a selected component view as a plain dictionary:

```python
{
    "name": "road",
    "index": 1,
    "mask": mask,
    "splats": {k: v[mask] for k, v in scene.splats.items()},
    "signal": {k: v[mask] for k, v in scene.signal.items()},
}
```

Notes:

- the base `Scene` contract only requires `get(component: str)`
- `GaussianScene` extends that by also accepting integer component ids
- this is a selected subset, not a zero-copy writable scene view

## `GaussianInferenceScene`

`GaussianInferenceScene` is the packed inference container.

```python
class GaussianInferenceScene(Scene):
    def __init__(self, id: str) -> None: ...

    means_planar: Optional[Tensor]        # [3, N] float32 CUDA
    qso_packed: Optional[Tensor]          # [N, 8] float16 CUDA
    colors_packed: Optional[Tensor]       # varies by sh_degree/compression
    sh_degree: Optional[int]
    sh_compression_mode: Optional[SHCompressionMode]
    num_gaussians: int
    component_names: list[str]
    component_index: Tensor
```

### `from_gaussian_scene(scene, *, id, sh_compression="none")`

Builds an inference scene from a training `GaussianScene`. Applies activations
automatically (`F.normalize` on quaternions, `.exp()` on scales, `.sigmoid()` on
opacities). Rejects appearance-optimized scenes (those with `"features"` in
splats) and multi-component scenes. Not supported under `torch.distributed` with
`world_size > 1`.

### `from_gaussian_tensors(means, quats, scales, opacities, colors, sh_degree, sh_compression, *, id)`

Builds an inference scene from pre-activated tensors. Validates activation
contracts: quaternions must be unit-norm (wxyz order), scales positive, opacities
in `[0, 1]`, all values finite.

### `put(name, component)`

Adds a pre-packed component dict with keys `means_planar`, `qso_packed`,
`colors_packed`, `sh_degree`, `sh_compression_mode`. Validates shapes, dtypes,
devices, and consistency with existing components.

### `get(component)`

Returns a component view by name or integer index, including a boolean `mask`
for indexing into the concatenated tensors.

### `is_empty()` / `release()`

`is_empty()` returns `True` when the scene has no packed tensors or zero
gaussians. `release()` clears all packed state and resets to empty.

Out of scope for this API:

- topology mutation hooks
- scene-owned rendering helpers
- scene-owned optimizer logic

## Validation Rules

`validate()` checks:

- required geometry keys exist when `splats` is non-empty
- every splat tensor has leading dimension `N`
- every signal tensor has leading dimension `N`
- `component_index.shape == (N,)`
- `component_names` is non-empty when `splats` is non-empty
- `component_index` stays within `[0, len(component_names))`

Validation relies on assertions for most invariant checks.

## Scene Serialization

`GaussianScene.state_dict()` returns:

```python
{
    "splats": scene.splats.state_dict(),
    "splats_requires_grad": {
        key: bool(value.requires_grad) for key, value in scene.splats.items()
    },
    "signal": {
        key: value.detach().clone() for key, value in scene.signal.items()
    },
    "component_names": list(scene.component_names),
    "component_index": scene.component_index.detach().clone(),
}
```

`GaussianScene.from_state_dict()` reconstructs:

1. a fresh `torch.nn.ParameterDict`
2. the stored `requires_grad` flags
3. `signal`
4. `component_names`
5. `component_index`
6. validation

Compatibility defaults:

- missing `component_names` defaults to `["scene"]`
- missing `component_index` defaults to all zeros

## Trainer Checkpoints

The trainer checkpoint format is intentionally narrower than `GaussianScene.state_dict()`.

### AV trainer

Current AV checkpoints save:

```python
{
    "step": step,
    "scene_path": scene_path,
    "scene_id": gaussian_scene.id,
    "splats": gaussian_scene.splats.state_dict(),
}
```

Key points:

- AV does not save `scene.state_dict()` — only `splats` plus the scene id
- AV rebuilds a one-component `GaussianScene` from the saved `splats` and `scene_id`
- `gaussian_scene_from_checkpoint` raises `ValueError` if `scene_id` is missing (loud failure for pre-SDK checkpoints)
- checkpoint evaluation can fall back to the saved `scene_path`

### Simple trainer

Current simple trainer checkpoints save:

```python
{
    "step": step,
    "scene_id": self.scene.id,
    "splats": self.splats.state_dict(),
    # optional trainer-owned module state
    "pose_adjust": ...,
    "app_module": ...,
    "post_processing": ...,
}
```

Key points:

- simple trainer also does not save `scene.state_dict()`
- on checkpoint load, it rebuilds the scene as `GaussianScene.from_splats(runner.splats, id=...)`

> Trainer checkpoints store only `splats + scene_id`, **not** `scene.state_dict()`. Restored scenes therefore lose:
>
> - any custom `signal` rows the trainer added during training
> - multi-component bookkeeping (`component_names`, `component_index`)
>
> If you train with non-trivial `signal` or multiple components, swap `"splats": self.splats.state_dict()` for `"scene": self.scene.state_dict()` in the trainer's save dict and load via `GaussianScene.from_state_dict(checkpoint["scene"])`. The serializer is already implemented; it is only the trainer save sites that opt in to the lossy form.

## Strategy Mutation Hooks

`GaussianScene` provides hooks so `component_index` and `signal` stay aligned with `splats` during topology-changing operations.

Update rules:

### Duplicate

```text
component_index = cat(component_index, component_index[sel])
signal[k] = cat(signal[k], signal[k][sel])
```

### Split

```text
component_index = cat(component_index[rest], component_index[sel], component_index[sel])
signal[k] = cat(signal[k][rest], signal[k][sel], signal[k][sel])
```

### Remove

```text
keep = ~remove_mask
component_index = component_index[keep]
signal[k] = signal[k][keep]
```

### Relocate

```text
component_index[dead_indices] = component_index[sampled_indices]
signal[k][dead_indices] = signal[k][sampled_indices]
```

### Sample add

```text
component_index = cat(component_index, component_index[sampled_indices])
signal[k] = cat(signal[k], signal[k][sampled_indices])
```

### Permute

```text
component_index = component_index[order]
signal[k] = signal[k][order]
```

These hooks are exercised in `libs/scene/components/test_gaussian_scene.py`.

## Rendering Responsibility

`GaussianScene` does not assemble rasterization inputs.

The trainer and render paths remain responsible for:

- activation policy for geometry and opacity
- quaternion normalization policy
- deciding whether to read `colors`, `sh0` and `shN`, or appearance features
- passing any renderer-specific `extra_signals`

That keeps the scene object focused on storage and bookkeeping.

## Gaussian Inference SH Packing

`GaussianInferenceScene` stores packed geometry in `qso_packed`: an `[N, 8]`
float16 tensor with quaternion in columns 0-3, scale in columns 4-6, and opacity
in column 7.

`GaussianInferenceScene` stores the public `sh_compression` string as
`SHCompressionMode` metadata: `NONE` = `"none"`, `PACKED_32B` = `"32b"`, and
`PACKED_16B` = `"16b"`. Nonzero compression modes are valid only for SH degree
3. The Python wrappers require this enum; the native C++ ABI still receives the
corresponding integer value and validates it before casting to its local enum.

For SH3, scene packing uses distinct `colors_packed` layouts:

- `NONE` (`"none"`): `[N, 16, 3]` float16, preserving raw SH layout
- `PACKED_32B` (`"32b"`): `[N, 48]` float32, flattening coefficients into 32-bit lanes
- `PACKED_16B` (`"16b"`): `[N, 48]` float16, flattening coefficients into fp16 lanes

RGB remains `[N, 4]` float16 with an alignment pad, and SH0-SH2 remain
`[N, K, 3]` float32.

## Supported Surface

Standardized scene APIs:

- minimal abstract `Scene`
- concrete `GaussianScene`
- raw Gaussian tensor storage in `splats`
- optional `signal` sidecars
- exclusive components via `component_index`
- sidecar update hooks for topology changes
- AV checkpoint evaluation support from saved splats
- concrete `GaussianInferenceScene` with packed QSO layout
- `SHCompressionMode` enum (`NONE`, `PACKED_32B`, `PACKED_16B`)
- `pack_gaussian_inference_scene` functional operator
- `put`/`get` multi-component assembly for inference scenes

Out of scope:

- hierarchical scene access
- scene-owned rendering helpers
- scene-owned optimizer logic
- generalized component mutation APIs like `set_component()`
- trainer checkpoint persistence of full `GaussianScene.state_dict()`
- appearance-optimized (`features`) conversion to inference scenes
- inference-to-training scene conversion (the one-way boundary)
