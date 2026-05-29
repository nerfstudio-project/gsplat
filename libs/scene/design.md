# Scene Design

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
from gsplat_scene import Scene, GaussianScene
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

Notably absent from the current implementation:

- `set_component()`
- `names()`
- `items()`
- a separate radiance API

### `put(name, splats)`

`put()` adds a named component backed by a `torch.nn.ParameterDict`.

Current behavior:

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

Current limitations:

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

## Validation Rules

`validate()` checks:

- required geometry keys exist when `splats` is non-empty
- every splat tensor has leading dimension `N`
- every signal tensor has leading dimension `N`
- `component_index.shape == (N,)`
- `component_names` is non-empty when `splats` is non-empty
- `component_index` stays within `[0, len(component_names))`

The implementation currently relies on assertions for most invariant checks.

## Scene Serialization

`GaussianScene.state_dict()` currently returns:

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

Legacy behavior supported today:

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

> ⚠️ **Persistence trade-off**: trainer checkpoints currently store only `splats + scene_id`, **not** `scene.state_dict()`. Restored scenes therefore lose:
>
> - any custom `signal` rows the trainer added during training
> - multi-component bookkeeping (`component_names`, `component_index`)
>
> If you train with non-trivial `signal` or multiple components, swap `"splats": self.splats.state_dict()` for `"scene": self.scene.state_dict()` in the trainer's save dict and load via `GaussianScene.from_state_dict(checkpoint["scene"])`. The serializer is already implemented; it is only the trainer save sites that opt in to the lossy form.

## Strategy Mutation Hooks

`GaussianScene` provides hooks so `component_index` and `signal` stay aligned with `splats` during topology-changing operations.

Current update rules are:

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

## Current Scope

Included today:

- minimal abstract `Scene`
- concrete `GaussianScene`
- raw Gaussian tensor storage in `splats`
- optional `signal` sidecars
- exclusive components via `component_index`
- sidecar update hooks for topology changes
- AV checkpoint evaluation support from saved splats

Explicitly not standardized yet:

- hierarchical scene access
- scene-owned rendering helpers
- scene-owned optimizer logic
- generalized component mutation APIs like `set_component()`
- trainer checkpoint persistence of full `GaussianScene.state_dict()`
