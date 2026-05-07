# 0009 — TDD step 5: HexPlaneField implemented (first contrib piece)

Date: 2026-05-07
Author: vnath (with assistance from Claude Code)
Branch: `vnath_gsharp`

## What landed

Fifth TDD chunk from `0001_scaffolding_commit.md` — the first piece of the experimental contrib path. `HexPlaneField` is now implemented in `gsplat/contrib/dynamic/hexplane.py`, with concrete pytest coverage in `tests/test_contrib_hexplane.py` (the previous `pytest.mark.skip` placeholders are gone).

### `gsplat/contrib/dynamic/hexplane.py` — implementation

`HexPlaneField(bounds=1.6, planes_config=None, multires=None)`:

- 4D `(x, y, z, t)` feature field decomposed into six 2D planes per scale (one for each pair of input axes — `xy`, `xz`, `xt`, `yz`, `yt`, `zt`).
- Each plane is bilinearly sampled with `F.grid_sample(padding_mode="border")`; per-scale features are the element-wise product across all six planes; multi-scale features are concatenated (`concat_features=True`, the only supported mode for now).
- Default config: 32 features per plane, `[64, 64, 64, 25]` resolution, two scales `(1, 2)` → output dim 64.
- Spatial coordinates are normalized to `[-1, 1]` via the AABB; the time coordinate is passed through (caller responsibility to pre-normalize). Out-of-range time values are clamped by `padding_mode="border"`.
- Spatio-temporal planes (those including the time axis) are initialized to ones so deformation starts identity-like — matches G-SHARP's convention at `training/scene/hexplane.py` line 100–104.

Helpers `_normalize_aabb`, `_grid_sample_wrapper`, `_init_grid_param`, `_interpolate_ms_features` are kept private (underscore-prefixed) inside the module — they're faithful ports of the G-SHARP source but aren't part of the public API.

Signature note: the scaffold's `planes_config: Sequence[dict]` was relaxed to `Optional[dict]` to match G-SHARP's actual usage (a single config shared across scales; multires expresses the multi-scale relationship).

### `tests/test_contrib_hexplane.py` — concrete coverage

7 tests, all passing on CPU under the autouse `seed=42` fixture from `tests/conftest.py`:

| Group | Positive | Negative |
|---|---|---|
| Construction | default-feat-dim (32×2 = 64), custom-config-feat-dim (16×3 = 48) | — |
| Forward | shape, finite-for-in-bounds-input | invalid-input-shape-raises (3D instead of 4D) |
| Gradient | gradients-reach-plane-params | — |
| Robustness | out-of-bounds-time-handled (clamped to grid edges, finite) | — |

The `feat_dim` tests pin the multi-scale concat contract (`output_coordinate_dim × len(multires)`). The gradient test walks every parameter in every scale and asserts at least one received a non-zero gradient — sufficient to catch a broken backprop graph.

## Verification

- `python -m pytest tests/test_contrib_hexplane.py -x` → **7 passed** in 1.69s.
- No CUDA dependency in the module (uses `F.grid_sample` only).

## Tracker status changes (`mr_tracker.md`)

No changes — this work doesn't touch any open reviewer thread.

## What unblocks next

Next TDD chunk from `0001_scaffolding_commit.md` step 6:

- `gsplat/contrib/dynamic/deformation.py`:
  - `DeformNetwork` — MLP head that takes HexPlane features + Gaussian state and predicts deltas on `(means, quats, opacities)`.
  - `DeformationTable` — per-Gaussian boolean flag controlling whether a given Gaussian is animated by the deform-net.
- Un-skip `tests/test_contrib_deformnet.py`.

@vcauxbrisebo's MR-013 question to @shsolanki on deformation integration is still open. Per @vnath's direction (2026-05-07): proceed without waiting; refactor if @shsolanki's response changes the design.
