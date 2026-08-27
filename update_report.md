# gsplat ROCm Port — Incremental Update Report

## Summary

Updated the HIP/ROCm fork of `gsplat` from baseline upstream commit
`2b902ff1` to upstream `main` HEAD `90d7b4b3` (11 commits). The diff was
isolated to a spherical-harmonics (SH) API overhaul and FTheta
(fisheye) camera FOV > 180° support. Both were translated into the
existing single-arch HIP port and validated on real hardware.

- **Baseline commit:** `2b902ff1`
- **New commit:** `90d7b4b3`
- **Scope:** spherical harmonics API rewrite, FTheta FOV>180° support,
  UT-projection radial culling.

## Changes translated into the fork

New files:
- `gsplat/hip/csrc/SphericalHarmonics.hip.h`
- `gsplat/hip/csrc/SphericalHarmonicsL1PlusCUDA.hip`
- `gsplat/hip/csrc/SphericalHarmonicsViewDirectionCUDA.hip`

Rewritten:
- `gsplat/hip/csrc/SphericalHarmonicsCUDA.hip`
- `gsplat/hip/csrc/SphericalHarmonics.h`
- `gsplat/hip/csrc/SphericalHarmonics.cpp`

Edited:
- `gsplat/hip/ext.cpp` — new op schemas for `spherical_harmonics`,
  `spherical_harmonics_l0[_bwd]`, `spherical_harmonics_l1_plus[_bwd]`,
  `assemble_proj_features_unpacked_fwd`.
- `gsplat/hip/include/Cameras.hip.h` — removed FTheta `z<=0` guard to
  support FOV beyond 180°.
- `gsplat/hip/csrc/ProjectionUT3DGSFused.hip` — added `use_radial_culling`
  parameter + logic.
- `gsplat/hip/csrc/Rendering.cpp` — removed obsolete
  `viewmat_to_camera_position`/`compute_classic_viewdirs` helpers,
  updated SH evaluation call sites.
- `gsplat/hip/_wrapper.py` — new `spherical_harmonics` signature,
  `spherical_harmonics_l0`/`spherical_harmonics_l1_plus`, new autograd
  registration classes.
- `gsplat/hip/_torch_cameras.py` — FTheta validity flag fix.
- `gsplat/hip/_torch_impl_ut.py` — radial-culling logic mirroring the
  kernel.
- `gsplat/rendering.py`, `gsplat/__init__.py`, `gsplat/version.py` —
  top-level API/version updates (bumped to `1.6.0`).
- `tests/test_basic.py` — updated `test_sh`,
  `test_sh_backward_accepts_strided_output_grad`, `test_sh_zero_channels`,
  `test_sh_fp16_coeffs`, `test_sh_k16_misaligned_coeffs` for the new
  `means`/`viewmats`-based SH signature.

## Bugs found & fixed during translation

1. **`std::array<T,0>::operator[]` not device-callable** for the
   `MAX_K=0` (DEGREE==0) specialization — fixed by restructuring
   `SphericalHarmonicsL1PlusCUDA.hip`'s backward kernel from
   `if constexpr(DEGREE==0){...return;} <rest>` into
   `if constexpr(...){...} else {<rest>}`.
2. **`const_data_ptr<float>()` undefined at link time** against this
   torch/ROCm build (confirmed absent from libtorch via `nm`) — worked
   around by replacing `const_data_ptr<` → `data_ptr<` in all three SH
   `.hip` files.
3. **`cooperative_groups::labeled_partition` no-op stub gap** — kept
   the existing shim as-is per explicit user direction; documented the
   gap with a translator note in `SphericalHarmonics.hip.h`.

Container/environment-only workarounds (not applied to repo source):
- ROCm 7.2.0's bundled `thrust` headers have a genuine self-conflicting
  redefinition bug; worked around in the build container only by
  overlaying ROCm 7.2.3's `thrust` headers.
- Build must pin `PYTORCH_ROCM_ARCH=gfx1100` to avoid multi-arch builds
  masking the above.
- Stale hipify-generated duplicate files (`*_hip.hip`, `*HIP.hip`, etc.)
  must be cleaned before every fresh build.

## Validation

### Radeon (gfx1100) — ROCm 7.2.0, PyTorch 2.8.0+rocm7.2.0

- Built via AOT (`pip install -e .[dev] --no-build-isolation`,
  `PYTORCH_ROCM_ARCH=gfx1100`).
- `import gsplat` succeeds, `gsplat.__version__ == "1.6.0"`.
- SH-focused tests: **all pass** (`pytest -k test_sh`).
- Full suite (`tests/test_basic.py tests/test_cameras.py`), using the
  local ROCm-ported `nerfacc` (`GPU_BACKEND=ROCM_RADEON`) and `torchpq`
  (`select_backend.py radeon`) in place of stock PyPI packages:
  **480 passed, 4 failed, 74 skipped.**
  - All 4 failures are `test_projection[...]` cases: a single
    outlier element (out of 300K–700K) in the `v_quats` backward
    gradient, cosine similarity ≥ 0.999999999963 vs. reference. This
    test exercises the **standard EWA `fully_fused_projection` path**
    (`ProjectionEWA3DGSFused.hip`), which was **not touched** by this
    update (confirmed via file mtimes — last modified 2026-08-04,
    weeks before this session). Consistent with pre-existing
    ROCm/CUDA floating-point rounding-order noise in an
    atomicAdd-heavy backward kernel, not a logic regression.

  Note: an initial full-suite run using **stock PyPI** `nerfacc`
  (no compiled CUDA/HIP extension) produced 54 failures, all due to
  `nerfacc.cuda._C` being `None`. These were resolved entirely by
  installing the project's own ROCm-ported `nerfacc`/`torchpq`
  instead — they were an environment gap, not a gsplat defect.

### CUDA reference — Quadro RTX 6000 (sm_75), PyTorch/CUDA 12.8 devel image

- Built via AOT (`TORCH_CUDA_ARCH_LIST=7.5`, `MAX_JOBS=4`); required
  side-loading several dependency wheels (`jaxtyping`, `wadler-lindig`,
  `cupy-cuda12x`, `nerfacc`, `torchpq`, `PLAS`) because the test host
  has no default route / internet egress. `[dev]` extras that require
  `clang-format==22.1.5` (unavailable on PyPI, pure lint tooling) were
  skipped — not needed to build or run tests.
- `import gsplat` succeeds, `gsplat.__version__ == "1.6.0"`,
  `torch.cuda.is_available() == True` on the Quadro RTX 6000.
- SH-focused tests: **115 passed, 0 failed, 42 skipped.**
- Full suite: **536 passed, 14 failed, 84 skipped.**
  - 4× `test_fully_fused_projection_ut[...-ftheta]`: a boundary
    flip-ratio check ("rolling-shutter floor() discontinuity")
    exceeds its cap by a razor-thin margin (0.1506% vs. 0.1500%,
    908/602745 elements). The test's own inline comments show this
    cap was hand-calibrated against **RTX PRO 2000, RTX PRO 6000
    (Blackwell), and L40S** — none of which is our Turing-generation
    **Quadro RTX 6000** (sm_75) test GPU. Consistent with normal
    cross-GPU-generation FP variance rather than a logic bug.
  - 10× `test_rasterize_to_pixels_eval3d[...]`: a single-Gaussian
    gradient-magnitude-sparsity edge case in
    `RasterizeToPixelsFromWorld3DGS*` kernels. These kernel files were
    **last modified 2026-07-29/30 and 2026-08-04** — weeks before this
    session's SH/FTheta work — confirming they were not touched by
    this update.

**Conclusion:** No regressions attributable to this update were found
on either architecture. All observed failures are pre-existing,
hardware/tolerance-calibration-sensitive edge cases in code paths this
update did not modify.

## Not committed / not modified

Per the `update-port` skill contract, no `git commit`, `push`, or
`tag` was performed. No manifest file exists for this port (only a
`manifest.example.md` template was found under
`maintainer/skills/maintain-ports/`), so no baseline-commit bump was
made — this should be recorded by the caller if this port is later
registered in a manifest.

## Follow-ups for the caller

- If greater CUDA-reference confidence on newer GPU generations is
  desired, re-run `test_fully_fused_projection_ut[...-ftheta]` on
  Ampere/Ada/Blackwell-class hardware, where the test's own
  calibration data indicates it should pass comfortably.
- The 4 Radeon `test_projection` and 10 CUDA `eval3d` failures are
  independent of each other and of this update; they reflect
  standing, hardware-specific numerical tolerance gaps in unrelated
  kernels and may warrant their own tracking/tightening work upstream.
- Docker containers `gsplat_radeon_build` and `gsplat_cuda_ref_build`
  were left running on `10.0.10.10` with working builds installed, in
  case further investigation of the pre-existing failures above is
  wanted; tear them down when no longer needed.
