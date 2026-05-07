# 0010 — TDD step 6: DeformNetwork + DeformationTable implemented

Date: 2026-05-07
Author: vnath (with assistance from Claude Code)
Branch: `vnath_gsharp`

## What landed

Sixth TDD chunk from `0001_scaffolding_commit.md`. `DeformNetwork` and `DeformationTable` are now implemented in `gsplat/contrib/dynamic/deformation.py`, with concrete pytest coverage in `tests/test_contrib_deformnet.py` (the previous `pytest.mark.skip` placeholders are gone).

@vcauxbrisebo's MR-013 question to @shsolanki on the deformation integration approach is still open. Per @vnath's direction (2026-05-07): proceeding without waiting; refactor if @shsolanki's response changes the design.

### `gsplat/contrib/dynamic/deformation.py` — implementations

- `DeformNetwork(feature_dim, hidden_dim=64, num_layers=3)`
  - Trunk: ``num_layers`` blocks of ``Linear + ReLU`` consuming ``plane_features``.
  - Three zero-initialised heads (``pos_head`` 3-d, ``quat_head`` 4-d, ``opacity_head`` 1-d) emit the per-Gaussian deltas. Zero-init makes the at-construction forward pass an exact identity on `(means, quats, opacities)` — pinned by `test_deform_net_zero_init_is_identity`.
  - `forward(means, quats, opacities, t, plane_features)` returns `(means + Δ, quats + Δ, opacities + Δ)`. Validates batch dim, `plane_features` last dim against the constructed `feature_dim`, and dtype consistency across the four input tensors.
  - The `t` argument is reserved for future time-aware extensions; the current implementation expects time information to already be encoded into `plane_features` via :class:`HexPlaneField`.
  - **Design note (locked by tests):** zero-init heads block gradient flow to the trunk at the very first forward pass — `dL/d(trunk_out) = dL/d(out) · head_weight = 0`. Head weights still receive non-zero gradients (since `dL/d(head_weight) = dL/d(out) · trunk_out`), so after one optimiser step the trunk picks up gradient. The companion test `test_deform_net_zero_init_blocks_trunk_gradient_at_init` locks this so a future change to head init flips both tests in lock-step.

- `DeformationTable(num_gaussians, device=None)`
  - Plain ``torch.bool`` mask of shape `(N,)`; no autograd, no parameters → zero overhead in the optimiser state.
  - `__len__`, `set_indices(indices, value=True)`, `prune(keep_mask)`, `duplicate(indices)`, `split(indices, factor=2)` — explicit ops mirroring the gsplat ``DefaultStrategy`` densify path.
  - `prune` shrinks via boolean indexing; `duplicate` appends children inheriting parent flags; `split` removes parents and appends `factor` children inheriting parent flags.

The scaffold's generic `update_on_densify(*args, **kwargs)` placeholder is replaced by the granular `prune` / `duplicate` / `split` methods — matches the way `DefaultStrategy` reports each densify op separately, and is easier to test.

### `tests/test_contrib_deformnet.py` — concrete coverage

15 tests, all passing on CPU under the autouse `seed=42` fixture from `tests/conftest.py`:

| Group | Coverage |
|---|---|
| `DeformNetwork` construction | invalid-construction-raises (num_layers, feature_dim) |
| `DeformNetwork` identity | zero-init-is-identity, zero-init-blocks-trunk-gradient-at-init (companion design lock) |
| `DeformNetwork` gradient flow | gradients-flow-to-mlp-weights (heads perturbed off zero before forward) |
| `DeformNetwork` validation | dtype-mismatch-raises, batch-dim-mismatch-raises, feature-dim-mismatch-raises |
| `DeformationTable` basics | init-all-static, negative-count-raises, set-indices |
| `DeformationTable` densify | prune-resizes-correctly, prune-shape-mismatch-raises, duplicate-appends-children, split-removes-parents-appends-children, split-invalid-factor-raises |

The `prune` / `duplicate` / `split` tests use a known initial mask (`[T, F, T, F, F]`) and assert exact post-op contents — locks the flag-inheritance contract under DefaultStrategy-style densify.

## Verification

- `python -m pytest tests/test_contrib_deformnet.py -x` → **15 passed** in 2.84s.
- No CUDA dependency in the module.

## Tracker status changes (`mr_tracker.md`)

No changes — this work doesn't touch any open reviewer thread. MR-013 (Vincent's question to @shsolanki on deformation integration) stays `deferred`; the implementation here is provisional and will be revisited if @shsolanki's response calls for a different shape.

## What unblocks next

Next TDD chunk from `0001_scaffolding_commit.md` step 7:

- `gsplat/contrib/dynamic/strategy.py`:
  - `DynamicStrategy(DefaultStrategy)` — applies the deform-net to `(means, quats, opacities)` *before* `rasterization()` is called, and resizes the `DeformationTable` in lock-step with each densify op.
- Un-skip `tests/test_contrib_dynamic_strategy.py`.

This is the integration point that ties HexPlane + DeformNetwork + DeformationTable into the gsplat training loop without touching `rasterization()` itself (per the proposal's "open risks" §1).
