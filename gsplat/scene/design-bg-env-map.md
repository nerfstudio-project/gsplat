# Background Environment Map Design

This document covers the `BackgroundScene` ABC and the `EnvMapBackground` concrete
implementation. For how background components integrate with `GaussianScene` and the
training loop, also read `design.md` § "Background Scene Components".

---

## Motivation

Rays that pass through all Gaussians (or miss them entirely) must resolve to *some*
color. Today, `rasterization(..., backgrounds=...)` accepts a **constant** per-camera
RGB vector — sufficient for simple cases but unable to represent direction-dependent
radiance like sky gradients, outdoor HDR light, or studio backdrops.

The goal is a general, swappable background abstraction that:

1. Is direction-dependent — the same camera can see different radiance in different
   directions.
2. Is fully differentiable — gradients flow back through the background sampling into
   its parameters and onward to the Gaussian rasterizer through the compositing step.
3. Requires no new rasterizer changes — compositing happens in Python after
   `rasterization()` returns.
4. Introduces no new hard dependencies — runtime sampling is a fused CUDA kernel in
   gsplat's own `gsplat_scene_cuda` extension (no `nvdiffrast`), and the pure-`torch`
   reference oracle it reproduces uses only ops already available in gsplat.

---

## `BackgroundScene` — abstract base class

```python
class BackgroundScene(Scene):
    """Abstract base for direction-dependent background representations.

    Subclasses must implement ``sample()``.  The default ``composite()``
    implementation is correct for most cases but may be overridden.
    (There is deliberately no loss/regularizer logic on the scene —
    losses are free functions in ``gsplat/losses.py``; see § "Why losses
    do not live on the scene".)

    Topology hooks (on_duplicate, on_split, …) are inherited as no-ops —
    background representations have no rows to permute or remove.
    """

    @abstractmethod
    def sample(self, rays_d: Tensor) -> Tensor:
        """Return background RGB for each ray direction.

        Args:
            rays_d: World-space unit ray directions, shape [N, 3].

        Returns:
            RGB radiance, shape [N, 3].  Values are non-negative; the
            concrete implementation decides whether to clamp to [0, 1]
            or allow HDR values.
        """

    def composite(
        self,
        gaussian_rgb: Tensor,   # [N, 3]
        opacity: Tensor,        # [N]
        rays_d: Tensor,         # [N, 3]
        is_training: bool = False,
    ) -> Tensor:
        """Over-composite background behind the Gaussian render.

        Default formula (premultiplied-sRGB "over"):
            final = gaussian_rgb + sample(rays_d) * (1 - opacity)

        Args:
            gaussian_rgb: Accumulated Gaussian color from the rasterizer.
            opacity:      Accumulated Gaussian alpha; ``[N]`` or ``[N, 1]``.
            rays_d:       World-space ray directions matching the raster output.
            is_training:  Enables per-step gradient/inpaint tracking in
                          subclasses that need it (e.g. ``EnvMapBackground``);
                          the side-effect-free default path ignores it.

        Returns:
            Composited RGB, same shape as ``gaussian_rgb``.
        """
        bg = self.sample(rays_d)
        # ``reshape(-1, 1)``, NOT ``unsqueeze(-1)``: an ``[N, 1]`` opacity fed
        # through ``unsqueeze(-1)`` becomes ``[N, 1, 1]`` and broadcasts the
        # blend into an ``[N, N, 3]`` tensor. ``reshape(-1, 1)`` normalizes both
        # ``[N]`` and ``[N, 1]`` inputs. Guarded by
        # ``test_composite_accepts_opacity_column_vector``.
        return gaussian_rgb + bg * (1.0 - opacity.reshape(-1, 1))

    # -- Scene ABC contract --

    def put(
        self,
        name: str,
        component: object,
        ctx: dict[str, Tensor] | None = None,
    ) -> None:
        """Store a named tensor or parameter (e.g. ``"textures"``).

        Backgrounds have no transform context; a non-``None`` ``ctx`` is
        rejected.
        """
        if ctx is not None:
            raise ValueError("BackgroundScene does not support transform context")
        setattr(self, name, component)

    def get(self, component: str) -> object:
        """Retrieve a previously stored named value."""
        return getattr(self, component)
```

### Why losses do not live on the scene

`GaussianScene` owns no loss logic — scale regularization, opacity penalties, and
sky-opacity MSE are computed in `gsplat/losses.py` / `losses_fused.py` using the
scene's tensors directly. `BackgroundScene` follows the same rule.

- **Inference / export use the scene without any loss infrastructure.** A loss or
  regularizer method that must be present at all times is dead weight.
- **Training decisions should be explicit.** A trainer that adds a regularizer term
  explicitly is legible; a silent polymorphic no-op on other background subclasses is not.

Any regularizer therefore lives in `gsplat/losses.py` as a standalone free function that
the trainer calls explicitly while owning its weight and schedule — exactly how the
Gaussian regularizers are handled.

### Why compositing lives here and not in Stage

`Stage.render()` is intentionally pass-through — it calls
`fn(splats=scene.splats, **kwargs)` and returns whatever the render function returns.
Compositing with a background requires the rasterizer output (rgb, opacity) *and* the
background scene, neither of which Stage owns. Keeping compositing in the trainer loop
(or in a thin helper on `BackgroundScene`) preserves Stage's "no multi-scene
composition" invariant while making the compositing step explicit and easy to test.

### Compositing formula

Compositing is a simple premultiplied-sRGB "over": the rasterizer output is already
alpha-premultiplied, so the background is blended in behind it directly, in the
rasterizer's (sRGB) color space:

```python
# opacity.reshape(-1, 1) normalizes [N] and [N, 1] to [N, 1] without the
# [N, N, 3] broadcast hazard that opacity.unsqueeze(-1) introduces on [N, 1].
weight = 1.0 - opacity.reshape(-1, 1)
return gaussian_rgb + self.sample(rays_d) * weight
```

There is no linear-space compositing option: blending always happens directly in the
rasterizer's output space.

---

## `EnvMapBackground` — learnable texture

### Representation

The background is stored as a single `torch.nn.Parameter` — a dense RGB texture
initialized to a neutral mid-grey (0.5). Two projections are supported:

| `EnvMapType` | Tensor shape | Sampling |
|---|---|---|
| `EQUIRECTANGULAR` | `[1, H, W, 3]` | Spherical (φ, θ) → UV, then the fused CUDA bilinear sampler with a one-column horizontal wrap |
| `CUBEMAP` | `[1, 6, H, W, 3]` | Dominant-axis face dispatch → per-face UV, then the fused CUDA bilinear sampler |

Default: **512 × 512 cubemap** (≈1.6 M parameters for RGB).

**No `nvdiffrast` dependency.** NRE's `SkyEnvMapBackground` uses `nvdiffrast` in exactly
two places — `dr.texture(tex, uv, filter_mode="linear", boundary_mode="wrap")` for
equirectangular and `dr.texture(tex, rays_d, filter_mode="linear", boundary_mode="cube")`
for cubemap. Both are **differentiable bilinear direction→RGB lookups**, not
rasterization, and both are exactly reproducible with the standard
`torch.nn.functional.grid_sample(mode="bilinear", align_corners=False,
padding_mode="border")` math:

- **Equirect (`wrap`)** → `grid_sample` on `[1,3,H,W]` with a one-column horizontal wrap
  pad (see § "Equirectangular").
- **Cubemap (`cube`)** → dominant-axis face + UV dispatch, then per-face `grid_sample`
  (see § "Cubemap").

At runtime this `grid_sample`-equivalent math is implemented as a **fused CUDA kernel**
(`gsplat_scene_cuda`; see § "Sampling"), and the same math is preserved as a
pure-`torch` **reference oracle** (`equirect_sample_reference` /
`cubemap_sample_reference`) that unit tests use to validate the kernel. The equivalence
was verified end-to-end: the reference reproduces correct sampling, canonical axis→face
routing, and gradient flow into the texture — including the exact
`torch.autograd.grad(bg_rgb, textures, grad_outputs=(1-opacity), retain_graph=True)`
tracking op NRE relies on — and the CUDA kernel matches the reference to 0.0 max-abs-diff
on both forward and texture-gradient. `nvdiffrast` remains not a `gsplat` dependency (core
deps are only `torch>=2.7` + `numpy`) and need not become one; the sampler is the
JIT/prebuilt `gsplat_scene_cuda` CUDA extension, not `nvdiffrast`.

> **One caveat, not a blocker:** `nvdiffrast`'s `boundary_mode="cube"` does true
> cross-face bilinear filtering at cube seams/corners, whereas per-face `grid_sample`
> with `padding_mode="border"` filters within each face and can leave a sub-texel
> discontinuity exactly on a seam. This is a minor quality detail; if seam-exact
> filtering is ever required it can be added in pure `torch` by padding each face with
> its neighbours' edge rows — still no `nvdiffrast`.

### Auxiliary state

| Attribute | Type | Purpose |
|---|---|---|
| `textures` | `nn.Parameter` | Learnable RGB texture |
| `texture_grads` | `Tensor` | Running max gradient magnitude per texel; used to identify unobserved regions |
| `n_grad_updates` | `int` | Step counter for gradient tracking |

All three are serialized by the hand-written `state_dict()` (see § "State persistence"),
not by `nn.Module` machinery — `EnvMapBackground` is a plain `Scene`, mirroring
`GaussianScene`.

### Sampling — fused CUDA kernel (PyTorch reference oracle)

At runtime, sampling is **CUDA-only with no fallback**. `EnvMapBackground._sample_raw`
normalizes `rays_d` to unit length, then dispatches through the thin overridable methods
`_sample_equirect` / `_sample_cubemap`, which call `sample_env_map_equirect` /
`sample_env_map_cubemap` in `gsplat/scene/kernels/env_map_sample_ops.py`. The per-projection
methods are kept as an **inheritance seam**: subclasses can swap one projection while
inheriting the other — NRE's `SkyEnvMapBackground` overrides `_sample_equirect` (historical
texture layout) but inherits `_sample_cubemap` and `_track_gradients` directly, so removing
these methods would break that subclass. Those wrappers
validate their inputs (float32 CUDA tensors, `[1, H, W, 3]` equirect / `[1, 6, H, W, 3]`
cubemap with `H == W`, `rays_d` assumed unit-normalized) and call `torch.autograd.Function`s
(`EquirectEnvMapSampleFunction` / `CubemapEnvMapSampleFunction`) backed by the
`gsplat_scene_cuda` extension. Non-CUDA or non-float32 inputs raise a clear error — there is
**no CPU/PyTorch runtime fallback** in the component.

The native side lives under `gsplat/scene/kernels/cuda/`:
`csrc/env_map_sample.cuh` (device helpers — bilinear filtering, equirect + cubemap
projection, per-ray fwd/bwd bodies) and `csrc/env_map_sample.cu` (kernels, launchers, and
the exported `sample_env_map_{equirect,cubemap}_{fwd,bwd}_cuda`). `ext.cpp` exposes four
pybind ops (`build.py` compiles the `.cu`):

- `sample_env_map_equirect_fwd(rays_d[N,3], textures[1,H,W,3]) -> out[N,3]`
- `sample_env_map_equirect_bwd(rays_d, textures, grad_out) -> grad_textures[1,H,W,3]`
- the two cubemap analogues with `textures[1,6,H,W,3]`.

The forward is **bit-exact** with `F.grid_sample(mode="bilinear", align_corners=False,
padding_mode="border")` — including the equirect one-column wrap pad and the dominant-axis
cube-face routing below. The backward scatters into `grad_textures` via `atomicAdd`, so
`grad_textures` is **zero-initialized** and reproduces the `grid_sample`-on-padded + wrap
`torch.cat` backward exactly. The kernel differentiates w.r.t. the **texture only**;
`rays_d` is a constant input and its gradient is `None` (see § "Known limitations").

The code below is the pure-`torch` **reference oracle** that the kernel reproduces (unit
tests compare the two). These live as module-level helpers in the test module
`tests/scene/components/test_env_map_background.py` — `equirect_sample_reference` /
`cubemap_sample_reference` — taking `textures` explicitly (deriving `H, W` from
`textures.shape` rather than from `self`), **not** methods on the component and not part
of `env_map_sample_ops.py`: they are test-only helpers with no runtime callers.

#### Equirectangular

```python
def equirect_sample_reference(rays_d: Tensor, textures: Tensor) -> Tensor:  # [N,3], [1,H,W,3] -> [N,3]
    H, W = textures.shape[1], textures.shape[2]
    x, y, z = rays_d.unbind(-1)
    # At the exact poles (x = y = 0) atan2(0, 0) has a NaN gradient; nudge x by a
    # tiny epsilon there so the gradient to rays_d stays finite (the azimuth value
    # is arbitrary at the pole regardless).
    pole  = (x == 0) & (y == 0)
    phi   = torch.atan2(y, torch.where(pole, x + 1e-6, x))   # azimuth ∈ (-π, π]
    # Clamp strictly inside [-1, 1]: acos' gradient blows up at the poles.
    theta = torch.acos(z.clamp(-1.0 + 1e-6, 1.0 - 1e-6))     # polar   ∈ [0, π]

    f = (phi + torch.pi) / (2.0 * torch.pi)                  # azimuth fraction ∈ [0, 1)
    # Wrap the texture by one column on each side, then remap the azimuth so
    # sampling coordinates stay strictly inside the real columns (u never
    # reaches the padded border) — correct antimeridian wrap-around. The
    # vertical axis uses "border" at the poles.
    u = (2.0 * f * W + 2.0) / (W + 2.0) - 1.0
    v = theta / torch.pi * 2.0 - 1.0

    tex = textures.permute(0, 3, 1, 2)                     # [1, 3, H, W]
    tex = torch.cat([tex[..., -1:], tex, tex[..., :1]], -1) # [1, 3, H, W+2]
    grid = torch.stack([u, v], dim=-1)[None, None]          # [1, 1, N, 2]
    return F.grid_sample(
        tex, grid, mode="bilinear", align_corners=False, padding_mode="border",
    ).squeeze(0).squeeze(1).T                                # [N, 3]
```

> **Seam note:** correct antimeridian wrap-around comes from the one-column wrap
> pad (`torch.cat([tex[..., -1:], tex, tex[..., :1]], -1)`) combined with the
> azimuth remap `u = (2·f·W + 2)/(W + 2) − 1`, which keeps `u` strictly inside
> the real columns so the padded edge is never the nearest tap. `padding_mode`
> stays `"border"` (it only ever governs the vertical/pole axis here). In the
> reference oracle the pad is a one-time `torch.cat` copy; the CUDA kernel folds the
> same wrap into its addressing (padded width `Wp = W + 2`) without materializing it.

#### Cubemap

```python
def cubemap_sample_reference(rays_d: Tensor, textures: Tensor) -> Tensor:  # [N,3], [1,6,H,W,3] -> [N,3]
    # Determine dominant axis and map to face UV in [-1, 1].
    face, u, v = _dominant_axis_to_face_uv(rays_d)  # [N], [N], [N]

    # Sample each face independently; faces are stored in dim-1 of textures.
    # textures: [1, 6, H, W, 3]  →  per-face: [1, 3, H, W]
    rgb = torch.zeros(rays_d.shape[0], 3, device=rays_d.device, dtype=textures.dtype)
    for face_idx in range(6):
        # No mask.any() early-out: it would force a GPU→CPU sync every face.
        # grid_sample tolerates a zero-row grid and the masked assign is a no-op.
        mask = face == face_idx
        grid = torch.stack([u[mask], v[mask]], dim=-1)[None, None]  # [1,1,M,2]
        face_tex = textures[0, face_idx].permute(2, 0, 1)[None]     # [1,3,H,W]
        sampled = F.grid_sample(
            face_tex, grid, mode="bilinear", align_corners=False, padding_mode="border"
        ).squeeze(0).squeeze(1).T                                      # [M, 3]
        rgb[mask] = sampled
    return rgb
```

`_dominant_axis_to_face_uv` is a standard, differentiable pure-PyTorch helper
using the OpenGL cube convention. The per-face sign pattern is **not** uniform —
`sc`/`tc` differ per face (e.g. `+Y` uses `tc=z` but `-Y` uses `tc=-z`; `±Z` flip
`x`; `±X` flip on `z`) — so all six must be spelled out. Faces are indexed
`0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z` to match `textures[0, face]`; `u=sc/ma`,
`v=tc/ma`, `ma=|dominant axis|`:

```
face 0 +X (x major, x > 0):  sc = -z,  tc = -y,  ma = |x|
face 1 -X (x major, x ≤ 0):  sc =  z,  tc = -y,  ma = |x|
face 2 +Y (y major, y > 0):  sc =  x,  tc =  z,  ma = |y|
face 3 -Y (y major, y ≤ 0):  sc =  x,  tc = -z,  ma = |y|
face 4 +Z (z major, z > 0):  sc =  x,  tc = -y,  ma = |z|
face 5 -Z (z major, z ≤ 0):  sc = -x,  tc = -y,  ma = |z|
```

The dominant axis is chosen with priority `x > y > z` via mutually exclusive
masks. This table is the design intent; the CUDA kernel
(`env_map_cubemap_corners` in `gsplat/scene/kernels/cuda/csrc/env_map_sample.cuh`)
is the runtime implementation, and the oracle `_dominant_axis_to_face_uv` plus the
locking tests `test_dominant_axis_routes_axes_to_expected_faces` and
`test_cubemap_face_routing_maps_axes_to_expected_faces` (all in
`tests/scene/components/test_env_map_background.py`) are the source of truth.

In the reference oracle all arithmetic is differentiable and `torch.autograd`
flows gradients back into `textures` through `F.grid_sample`; at runtime the same
texture gradient is produced by the CUDA kernel's backward (`atomicAdd` scatter
into a zero-initialized `grad_textures`).

### Radiance activation

After sampling, an activation prevents the texture from producing negative values:

| `saturate_radiance` | Activation | Notes |
|---|---|---|
| `False` (default) | `F.relu` | Allows HDR values > 1 |
| `True` | `torch.clamp(0, 1)` | SDR-only; avoids blown highlights |

### Coordinate system

`EnvMapBackground` makes **no assumption** about the world-space coordinate convention.
`rays_d` must be passed in whatever coordinate system the trainer uses. If the dataset
uses a different convention than the env map was designed for (e.g. OpenCV vs OpenGL),
the caller is responsible for rotating `rays_d` before passing them in, or for
subclassing `EnvMapBackground` and overriding `sample()` to apply a fixed rotation.

This is a deliberate design choice: the background is a passive sampler, not a
coordinate-system enforcer.

---

## Gradient tracking and inpainting

To detect texels that no ray ever hits (so they can be inpainted later), the tracker
accumulates a running **max** of the per-texel mean gradient into `texture_grads`.
This needs both the sampled `bg_rgb` and the accumulated `opacity`, so it lives in
`EnvMapBackground.composite()` (which samples, then tracks, then blends) — not in the
stateless `sample()`. It runs only during training and only after warm-up. This mirrors
NRE's `SkyEnvMapBackground.forward` (`nre/nre/models/background.py`), the reference
implementation:

Gradient tracking operates on the **raw pre-activation** radiance (`raw_rgb` from
`_sample_raw`), not the activated `bg_rgb`, so bright/HDR texels still receive gradient
under `saturate_radiance`. The warm-up gate is `>=` (not `>`), and an optional
`grad_track_interval` subsamples the pass; both are read from rank-uniform counters so
every DDP rank enters the `all_reduce` on the same steps.

```python
# inside EnvMapBackground.composite(), after raw_rgb = self._sample_raw(rays_d):
if is_training and raw_rgb.requires_grad:
    warmed_up = self.n_grad_updates >= self.min_grad_updates   # default warm-up: 1000
    due = self.n_grad_updates % self.grad_track_interval == 0   # default interval: 1
    if warmed_up and due:
        grad = torch.autograd.grad(
            raw_rgb, self.textures,
            grad_outputs=(1 - opacity.reshape(-1, 1)).expand_as(raw_rgb),
            retain_graph=True,
        )[0].detach()                    # [1, (6,) H, W, 3]

        # DDP: autograd.grad only sees this rank's shard, so average across ranks
        # to keep texture_grads consistent (matches NRE).
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            grad = grad / torch.distributed.get_world_size()
            torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)

        mag = grad.squeeze(0).mean(dim=-1)             # [H,W] or [6,H,W]
        self.texture_grads.copy_(torch.maximum(mag, self.texture_grads))
    self.n_grad_updates += 1
```

Notes:

- **Cost.** The extra `torch.autograd.grad(..., retain_graph=True)` runs a second
  backward through the texture each tracked step, roughly doubling the texture's
  backward cost. It is gated behind `min_grad_updates` and the training flag, and only
  touches the (small) background parameter — negligible next to Gaussian backprop.
- **`n_grad_updates` is a plain `int`, not a tensor/buffer.** A `torch.Tensor` counter
  would force a GPU→CPU sync on every `if`/`assert`; NRE keeps it as an `int` for this
  reason. It is serialized explicitly in `state_dict()` (see § "State persistence").
- **The tracking `autograd.grad` composes through the CUDA sampler.** Because sampling
  runs through the `Equirect`/`Cubemap` `EnvMapSampleFunction`, the second backward here
  invokes that Function's `backward`, which returns `grad_textures` — so gradient tracking
  works unchanged over the fused kernel path (it differentiates w.r.t. the texture only).

At inference/export time, texels with `texture_grads < inpaint_threshold` have never
been observed and can be filled with an inpainting model (e.g. LaMa) before rendering.
This step is optional and trainer-driven; `EnvMapBackground` only exposes the
`texture_grads` tensor and an `inpaint_mask()` helper — it does not run inpainting
itself.

---

## Training integration

> **Intended usage, not code in this repo.** The optimizer param group, checkpoint
> save/load, and trainer forward-pass snippets below describe how the **external AV
> trainer** consumes these components (sister MR `nre!4012`). This repo ships the
> `EnvMapBackground` / `BackgroundScene` components, their exports, and their tests
> only — there are **no in-repo call sites** of `.composite(...)` or the param-group /
> checkpoint wiring outside those tests. Treat this section as the contract the trainer
> is expected to follow, not as existing gsplat code.

### Losses

Two signals supervise the background. Both are computed in the trainer loop:

| Loss | Weight (reference) | Where computed |
|---|---|---|
| Photometric (L1 / SSIM) on composited image | primary | Trainer, on `bg_scene.composite(...)` output |
| Sky opacity MSE via `FusedCameraLosses` + `SKY_SEMANTIC` flag | λ = 0.05 | Trainer, on rasterizer `alphas` |

The sky opacity loss requires per-ray `LossFlag.SKY_SEMANTIC` bits. In
`av_trainer.py` these are currently estimated with a brightness heuristic. Proper
semantic segmentation labels are preferable when available.

### Optimizer param group

The background texture uses its own learning rate. NRE's reference value is **0.001**
(`configs/system/gaussians.yaml`, the `background` param group) — in the same range as
the Gaussian rotation/scale LRs (0.001–0.005) and well above the position LR (~1.6e-4).

`EnvMapBackground` is a plain `Scene`, not an `nn.Module`, so it exposes its trainable
tensors through a plain `parameters()` accessor (returning `[self.textures]`) — the
analogue of registering `GaussianScene` params via `gaussian_scene.splats.values()`:

```python
optimizer = torch.optim.Adam([
    {"params": list(gaussian_scene.splats.values()), "lr": splat_lr},
    {"params": list(bg_scene.parameters()),          "lr": bg_lr},   # e.g. 0.001
])
```

### Checkpoint save/load

```python
# Save
torch.save({
    "step": step,
    "scene_id": gaussian_scene.id,
    "splats": gaussian_scene.splats.state_dict(),
    "background": bg_scene.state_dict(),           # new
}, ckpt_path)

# Load
ckpt = torch.load(ckpt_path)
gaussian_scene = GaussianScene.from_state_dict(...)
bg_scene = EnvMapBackground.from_state_dict(ckpt["background"])
```

### Trainer forward pass (reference pattern)

```python
# 1. Gaussian rasterization
renders, alphas, info = stage.render(
    "scene", viewmat=viewmat, K=K, W=W, H=H, render_mode="RGB+ED"
)

# 2. Background compositing (trainer-owned)
if bg_scene is not None:
    rays_d = camera_rays_d(viewmat, K, W, H)          # [H*W, 3]
    renders = bg_scene.composite(
        renders.reshape(-1, 3),
        alphas.reshape(-1),
        rays_d,
        is_training=True,   # activates gradient/inpaint tracking during training
    ).reshape(H, W, 3)
# Evaluation / inference should call composite(...) with is_training=False (the default).

# 3. Losses
loss  = photometric_loss(renders, gt_images)
loss += bg_opacity_weight * opacity_loss(alphas, sky_mask)

# 4. Backward, DDP gradient sync, optimizer step
loss.backward()
if isinstance(bg_scene, EnvMapBackground):
    bg_scene.sync_texture_grad()   # REQUIRED under DDP — see below
optimizer.step()
optimizer.zero_grad()
```

> **DDP gradient sync — required.** Because `EnvMapBackground` is a plain `Scene`, not an
> `nn.Module`, DDP registers no all-reduce hook on `textures`. Under distributed training
> the trainer **must** call `bg_scene.sync_texture_grad()` after `loss.backward()` and
> before `optimizer.step()` on every step; it mean-reduces `textures.grad` across ranks
> (ranks with no background contribution participate with zeros) and is a no-op when
> `torch.distributed` is unavailable/uninitialized or world size is 1. Every rank must also
> call `composite(is_training=True)` an equal number of times per step: the gradient-tracking
> all-reduce is gated on the rank-local `n_grad_updates` counter and will deadlock if the
> counts drift across ranks.

---

## State persistence

`EnvMapBackground` mirrors `GaussianScene`: it is a plain `Scene` (not an `nn.Module`),
so it does **not** rely on `nn.Module.state_dict()` / buffers / `get_extra_state`.
Instead it hand-writes `state_dict()` and a `from_state_dict()` classmethod that return
and consume a plain dict. This is important because the scalar config fields
(`envmap_type`, `width`, …) are ordinary attributes — an `nn.Module.state_dict()` would
silently drop them, keeping only parameters and buffers.

**Serialization contract** (see `state_dict()` / `from_state_dict()` in
`env_map_background.py` for the exact dict — do not treat the list below as an
enumeration to copy). `state_dict()` returns a plain dict that captures, in three
groups:

- **Every scalar config field** needed to rebuild `EnvMapBackgroundConfig` and thus
  the texture shape: `envmap_type`, `width`, `height`, `saturate_radiance`,
  plus `min_grad_updates`, `grad_track_interval`, `should_inpaint`,
  `inpaint_threshold`, and `inpaint_kernel_size`. (The five inpaint/grad fields are
  easy to forget — omitting any of them is exactly the drift this section guards
  against.) The scene `id` is stored alongside them.
- **The learnable tensor and its trainability:** `textures` (detached clone) and
  `textures_requires_grad`.
- **The gradient-tracking state:** `texture_grads` (detached clone) and the plain-int
  `n_grad_updates`.

`from_state_dict()` reconstructs `EnvMapBackgroundConfig` from the scalar fields
(with `state.get(...)` defaults for the four inpaint/grad fields, so older checkpoints
still load), rebuilds the scene, validates that the saved `textures` / `texture_grads`
shapes match the shapes implied by the config (raising `ValueError` on mismatch rather
than restoring silently), then restores `textures` as an `nn.Parameter` with the saved
`requires_grad`, plus `texture_grads` and `n_grad_updates`.

`from_state_dict()` reconstructs the full `EnvMapBackground` from this dict without
requiring any external constructor arguments — useful for checkpoint-only restoration,
exactly as `GaussianScene.from_state_dict()` does.

---

## Configuration reference

Defaults mirror NRE's deployed `configs/model/background/sky_env_map.yaml` (not the
pydantic schema defaults — e.g. NRE's schema defaults `saturate_radiance=True`, but the
shipped env-map config sets it to `False`, which is what we match here):

```python
@dataclass
class EnvMapBackgroundConfig:
    envmap_type: Literal["cubemap", "equirectangular"] = "cubemap"
    width: int = 512
    height: int = 512                  # must equal width for cubemap
    saturate_radiance: bool = False    # False = relu (HDR); True = clamp [0,1]
    min_grad_updates: int = 1000       # steps before gradient tracking starts
    grad_track_interval: int = 1       # subsample the gradient-tracking pass (1 = every step)
    should_inpaint: bool = True        # enable unobserved-texel inpainting (trainer-driven)
    inpaint_threshold: float = 5e-2    # texture_grads below this → inpaint mask
    inpaint_kernel_size: int = 10      # dilation kernel size for the inpaint mask
```

---

---

## Files to create

| File | Contents |
|---|---|
| `gsplat/scene/components/background_scene.py` | `BackgroundScene` ABC |
| `gsplat/scene/components/env_map_background.py` | `EnvMapBackground`, `EnvMapBackgroundConfig`, `EnvMapType` |
| `gsplat/scene/components/__init__.py` | Export `BackgroundScene`, `EnvMapBackground` |
| `gsplat/scene/__init__.py` | Re-export from top-level `gsplat.scene` |
| `gsplat/scene/kernels/env_map_sample_ops.py` | Python autograd wrapper (`EnvMapSampleFunction`) over the fused CUDA sampler |
| `gsplat/scene/kernels/cuda/csrc/env_map_sample.cu` | Fused equirect/cubemap sampler forward + backward CUDA kernels |
| `gsplat/scene/kernels/cuda/csrc/env_map_sample.cuh` | Kernel declarations / shared device helpers |
| `tests/scene/components/test_env_map_background.py` | Unit tests: sample shape, composite, state dict round-trip |

The MR also **modifies** existing native/build wiring to register the new op:
`gsplat/scene/kernels/cuda/csrc/ext.cpp`, `gsplat/scene/kernels/cuda/build.py`,
`gsplat/scene/kernels/_backend.py`, and `setup.py`.

---

## Known limitations / future work

- **Cubemap seam filtering.** The sampler filters *within* each cube face
  (`padding_mode="border"`), so bilinear taps do not cross face boundaries. This is
  unchanged by the CUDA migration — the fused kernel is per-face, exactly like the
  reference oracle. `nvdiffrast`'s `boundary_mode="cube"` (used by NRE) does true
  cross-face filtering, so it produces marginally smoother results exactly on
  seams/corners. Chosen deliberately to avoid an `nvdiffrast` dependency. If seam-exact
  filtering is ever required, pad each face with its neighbours' edge rows before sampling
  — no new dependency. (See § "Representation".)
- **Sampler differentiates w.r.t. the texture only.** The CUDA kernel treats `rays_d` as a
  constant input; its `EnvMapSampleFunction.backward` returns `None` for the `rays_d`
  gradient and `grad_textures` for the texture. The differentiable-w.r.t.-`rays_d`
  property (e.g. finite gradients at the poles, guarded by
  `test_pole_direction_gradient_to_rays_d_is_finite`, which now runs against
  `equirect_sample_reference`) is retained only in the pure-`torch` reference oracle.
  Future work could add a `rays_d`-gradient path to the kernel if a downstream consumer
  ever needs to backprop into ray directions.
- **Equirect antimeridian seam.** Handled by a one-column horizontal wrap pad before
  `grid_sample`; the poles use `border` behaviour. (See § "Equirectangular".)
- **Inpainting is trainer-driven.** `EnvMapBackground` exposes `texture_grads` and an
  `inpaint_mask()` helper but does not run a LaMa-style inpainter itself (NRE does, via
  `simple_lama_inpainting`). Kept out of the scene to avoid that dependency.
