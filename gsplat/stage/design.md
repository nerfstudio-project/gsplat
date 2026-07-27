# Stage Design Document

## Overview

Stage orchestrates the connection between `GaussianScene` and a render function.
It does not own loss computation, optimizers, or the training loop.

```
Caller (trainer loop)
  |
  |  stage.render("scene", viewmat=..., K=..., W=..., H=..., render_mode="RGB+ED")
  v
Stage
  |-- _scenes: {scene_id: (scene, render_fn), ...}
  |
  |  scene, fn = _scenes[scene_id]
  |  fn(splats=scene.splats, **kwargs)
  v
Returns: render_fn output directly (e.g. (renders, alphas) or (renders, alphas, info))
```

## Design Decisions

### Training vs Inference

- **Training**: Stage holds exactly 1 scene. One training run = one scene.
- **Inference**: Stage holds N scenes. Multiple trained splats loaded into
  memory and rendered independently. Stage dispatches renders per scene;
  it does not composite, depth-sort, or fuse outputs across scenes.

### Render function dispatch

Stage does not know *how* to render. It pairs each scene with a user-supplied
`render_fn` and forwards `**kwargs` from the caller.

```
stage.render(scene_id, **kwargs)
  -> scene, fn = self._scenes[scene_id]
     fn(splats=scene.splats, **kwargs)
```

Stage always passes `splats` as a **named keyword argument**, never positional.
This avoids colliding with the first positional parameter of render functions
that have a different first arg (e.g., `camtoworlds` in `rasterize_splats`).

Render functions must accept a `splats` kwarg. For `simple_trainer`, this
requires adding an optional `splats` parameter to `rasterize_splats` with a
fallback to `self.splats` (see "render_fn contract" in Resolved Decisions).

### Why a dict keyed by scene.id

Scenes are stored in a `dict[str, tuple[GaussianScene, Callable]]` keyed by
`scene.id`. This gives O(1) lookup at render time and enforces unique scene ids
(duplicate ids raise `ValueError` on `add_scene`).

### What Stage owns vs delegates

| Stage owns                          | Caller / Scene owns            |
|-------------------------------------|--------------------------------|
| Scene references                    | Scene parameters (GaussianScene)|
| Scene-to-render_fn pairing          | Loss computation                |
| `render(scene_id, **kwargs)` dispatch | Optimizer / scheduler         |
|                                     | Training loop                   |
|                                     | Data loading / camera sampling  |
|                                     | Strategy orchestration           |

### What Stage explicitly does NOT do

- Define or compute loss
- Own or step optimizers
- Load data or sample cameras
- Interpret render_fn kwargs (pass-through only)
- Orchestrate strategy hooks (pre/post-backward stays in the training loop)
- Composite multi-scene outputs (caller combines if needed)

## Interface

```python
class Stage:
    def __init__(self) -> None:
        self._scenes: dict[str, tuple[GaussianScene, Callable]] = {}

    def add_scene(self, scene: GaussianScene, render_fn: Callable) -> None:
        """Register a scene and its render function, keyed by scene.id.

        Raises ValueError if scene.id is already registered.
        """

    def scene_ids(self) -> list[str]:
        """Return the ids of all registered scenes."""

    def get_scene(self, scene_id: str) -> GaussianScene:
        """Return the registered ``GaussianScene`` for ``scene_id``.

        Raises KeyError if scene_id is not registered.
        """

    def render(self, scene_id: str, **kwargs) -> Any:
        """Render a scene by id.

        The render_fn is called as: render_fn(splats=scene.splats, **kwargs)
        Returns whatever the render function returns — caller is responsible
        for matching the return arity (e.g. 3-tuple for 3DGS, 7-tuple for 2DGS).

        Raises KeyError if scene_id is not registered.
        """
```

## Integration Points

### av_trainer.py

#### Scene construction

```python
gs_scene = GaussianScene.from_splats(init_gaussians_from_lidar(scene, device=device))
stage = Stage()
stage.add_scene(gs_scene, render_gaussians)
```

#### Training forward pass

```python
renders, alphas = stage.render("scene",
    viewmat=viewmat, K=K, W=W, H=H, render_mode="RGB+ED"
)
```

#### Evaluation render

```python
rd, _ = stage.render("scene", viewmat=vm_t, K=K_t, W=W, H=H)
```

#### Optimizer creation (unchanged, operates on gs_scene.splats)

```python
optimizer = create_optimizer(gs_scene.splats, lr)
```

### simple_trainer.py

#### Scene construction

```python
self.splats, self.optimizers = create_splats_with_optimizers(...)
self.scene = GaussianScene.from_splats(self.splats)
self.splats = self.scene.splats  # alias so optimizer refs stay valid
self.stage = Stage()
self.stage.add_scene(self.scene, self.rasterize_splats)
```

#### Training forward pass

```python
renders, alphas, info = self.stage.render("scene",
    camtoworlds=camtoworlds, Ks=Ks, width=width, height=height, ...
)
```

#### Strategy hooks (unchanged, stay in training loop)

```python
self.cfg.strategy.step_post_backward(
    params=self.splats, optimizers=self.optimizers,
    state=..., step=step, info=info, scene=self.scene,
)
```

### simple_trainer_2dgs.py

Same pattern as simple_trainer.py. `Stage.render()` returns whatever the
render_fn returns — for 2DGS this is a 7-tuple
`(colors, alphas, normals, normals_from_depth, distort, median, info)`
which passes through unchanged.

#### Required change: rasterize_splats accepts splats kwarg

```python
def rasterize_splats(
    self, camtoworlds, Ks, width, height, masks=None, ...,
    splats: Optional[ParameterDict] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    splats = splats if splats is not None else self.splats
    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
```

Existing call sites pass no `splats` kwarg and work unchanged.
Stage passes `splats=scene.splats` via the kwargs dispatch.

## Resolved Decisions

1. **Activations stay inline** -- `exp(scales)`, `sigmoid(opacities)`, `normalize(quats)`
   remain inside the render_fn, not in Stage or GaussianScene. Each render_fn is
   responsible for its own activation logic.

2. **Strategy orchestration is out of scope** -- `strategy.step_pre/post_backward()`
   stays in the training loop. Stage does not call strategy hooks.

3. **Return signature matches render_fn** -- Stage does not impose a return type.
   `render()` returns whatever the render_fn returns. `av_trainer`'s
   `render_gaussians` returns `(renders, alphas)`. `simple_trainer`'s
   `rasterize_splats` returns `(renders, alphas, info)`. `simple_trainer_2dgs`'s
   `rasterize_splats` returns a 7-tuple. All work as-is.

4. **No multi-scene composition** -- Stage dispatches per-scene renders independently.
   Compositing (alpha-blend, depth-sort) is the caller's responsibility if needed.

5. **render_fn contract: splats as named kwarg** -- Stage always calls
   `fn(splats=scene.splats, **kwargs)`. Splats is passed as a **keyword**
   argument, never positional, to avoid colliding with render functions whose
   first positional arg is something else (e.g., `camtoworlds`).
   Render functions must accept a `splats` kwarg. For `simple_trainer`, this
   means adding an optional `splats` parameter to `rasterize_splats` with
   fallback `splats = splats if splats is not None else self.splats`.
   For `av_trainer`, `render_gaussians` already accepts `splats` as its first
   arg — passing it by name works without changes.
