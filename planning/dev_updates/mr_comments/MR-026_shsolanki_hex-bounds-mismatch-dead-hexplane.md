---
id: MR-026
author: Shikhar Solanki (@shsolanki) — posted via babysit-mrs agent
date: 2026-05-20
mr: !169
thread: examples/dynamic_surgical_trainer.py:164 (Config.hex_bounds); gsplat/contrib/dynamic/hexplane.py (_grid_sample_wrapper padding_mode)
status: open
labels: [contrib, dynamic, hexplane, trainer, blocking, major, performance]
---

# MR-026 — `Config.hex_bounds=1.6` mismatch makes HexPlane spatially constant (dead deformation)

## Thread (verbatim)

**Shikhar Solanki, 2026-05-20 — reframed from original "depth-PNG units" thread after empirical analysis:**

> Initial framing was wrong (loader is fine). The actual bug is that `Config.hex_bounds = 1.6` is hardcoded and doesn't match the dataset's working scale.
>
> Empirical evidence on EndoNeRF "pulling":
> - On the synthetic 32×32 × 4-frame test fixture: `means.max() = 8.30 > hex_bounds = 1.6`, so **0/initial points** fall inside the AABB.
> - On real "pulling" data with 5k and 50k init points: **0/5k and 0/50k** points are inside the AABB.
>
> Every plane query lands outside the AABB, so `HexPlaneField._grid_sample_wrapper` (which uses `padding_mode="border"`) clamps every Gaussian to the *same border feature*. The deformation field becomes spatially constant and the dynamic component is effectively dead.
>
> This explains the held-out test PSNR plateau we saw (~27 dB instead of the headline ~38 dB) — the deformation network is doing nothing.
>
> Fix options:
> - **A — Auto-derive `hex_bounds`** from `parser.bounds` (or from the init point cloud's bbox + a small pad).
> - **B — Normalize means to `[-1, 1]^3` at init time**, store the inverse transform, apply only inside the HexPlane query.
> - **C — Surface `hex_bounds` with a startup assert**: fail fast if `means.abs().max() > hex_bounds`.
>
> Pin with a regression test asserting in-AABB fraction == 1.0 after init.

## Summary
Major bug: `Config.hex_bounds=1.6` is too small for EndoNeRF metric depths (means land at ~8 in each axis). `padding_mode="border"` in `grid_sample` then clamps every query to the same edge feature, so HexPlane learns a constant field — the entire deformation component is dead. Probably explains why our held-out PSNR didn't reach the headline train-batch peak.

## Scope / affected files
- `examples/dynamic_surgical_trainer.py` — `Config.hex_bounds`, line ~164; trainer init / build_splats_from_parser.
- `gsplat/contrib/dynamic/hexplane.py` — `_grid_sample_wrapper` (overlaps with MR-017's squeeze fix).
- `tests/test_dynamic_surgical_trainer.py` — add in-AABB regression test.
- `tests/test_contrib_hexplane.py` (if it exists) — pin the bounds contract.

## Action plan
- [ ] Pick fix approach. Recommendation: **B — normalize means to `[-1, 1]^3` at init time, store inverse transform on the strategy state, apply inside HexPlane query**. Robust across datasets; aligns with grid_sample's expected coord range; doesn't require dataset-specific config tuning.
  - Alternative: **A** if normalisation has knock-on effects on the rasterizer that I haven't seen yet.
- [ ] Implement the chosen fix.
- [ ] Add a startup assert (regardless of A/B) — fail loudly if `>1%` of init points fall outside the HexPlane AABB. Helps catch future datasets that don't fit.
- [ ] Add `test_hexplane_query_in_aabb_after_init` (asserts 100% of init means produce non-border features).
- [ ] Re-run the long demo (2500 fine steps); confirm held-out test PSNR jumps. **This is the load-bearing experiment** — if held-out PSNR doesn't improve materially, the diagnosis is incomplete.
- [ ] Update `0014_long_run_validation_and_render.md` with the new numbers and a note that the previously-reported held-out PSNR was depressed by the dead HexPlane.

## Resolution
<filled after fix lands and re-run confirms PSNR jump>

## User-side reply draft
> Confirmed — `hex_bounds=1.6` had every Gaussian falling outside the AABB on EndoNeRF (mean.max ~8.30 on the fixture, 0/50k inside on real "pulling"), so `grid_sample`'s `padding_mode="border"` made HexPlane spatially constant. Fixed by normalising means to `[-1, 1]^3` at init time and applying the inverse transform inside the HexPlane query. Held-out test PSNR jumped from 27 dB → <new number> dB. Added a startup assert + regression test to lock the contract.
