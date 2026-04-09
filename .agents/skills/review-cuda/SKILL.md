---
name: review-cuda
description: >
  Domain-specific CUDA code review for gsplat -- focused on kernel correctness, thread synchronization,
  numerical stability, NaN safety, shared memory races, const correctness, and matching constants between
  CUDA and Python reference implementations. Use when reviewing .cu, .cuh, .h files under gsplat/cuda/,
  or any change that touches CUDA kernels, rasterization, projection, or intersection code.
---

# CUDA Code Review (gsplat)

You are reviewing CUDA kernel code in the gsplat Gaussian splatting library. This review enforces
patterns distilled from the project's merged MR review history. Every rule below was triggered by
a real bug, regression, or performance issue caught in review.

## Step 1: Get the diff

Run these in parallel:

```bash
git diff nv/main...HEAD -- '*.cu' '*.cuh' '*.h' 'gsplat/cuda/**'
git diff --name-only nv/main...HEAD
```

## Step 2: Read changed files in full

For each changed CUDA file, read the full file. Also read `gsplat/cuda/include/Common.h` for
shared constants and `gsplat/cuda/_constants.py` for the Python-side mirror.

## Step 3: Review through the CUDA lens

### Priority 1: Thread Synchronization and Shared Memory Races

The most dangerous class of bugs in gsplat kernels. These cause silent correctness failures.

Check for:
- **All threads must reach sync points**: When using `__syncthreads_count` or cooperative group
  barriers, ALL threads in the block must reach the call. Conditional early returns that skip
  some threads cause undefined behavior (hangs, incorrect results). Restructure so all threads
  hit the barrier, then conditionally skip writes.
- **Per-pixel vs per-Gaussian data**: Any value that varies per-pixel (e.g., `hit_distance` from
  per-pixel `ray_o`/`ray_d`) must NOT be stored in per-Gaussian shared memory buffers like
  `rgbs_batch`. Store in per-pixel registers instead.
- **Cooperative groups usage**: Verify `block.thread_index()` and `block.group_index()` are used
  consistently with the grid/block launch dimensions.

### Priority 2: Numerical Stability and NaN Safety

gsplat renders gradients through long chains of floating-point ops. NaN propagation is the #1
production issue.

Check for:
- **Clamp inputs to trig functions**: `acos`/`asin` inputs must be clamped to `[-1, 1]` before
  calling. Fast-math flags can produce slightly out-of-range values from `sqrt`.
- **Floor denominators in backward passes**: Use `fmaxf(1 - alpha, MIN_ONE_MINUS_ALPHA)` to
  prevent gradient explosion when alpha approaches 1.
- **Degenerate Gaussian culling**: Assert non-zero quaternion norms and non-zero scales in forward
  kernels before computing covariance.
- **Near-zero determinant guards**: Camera unprojection and matrix inverses must check for
  near-zero determinants. Return invalid/zero values rather than dividing by zero.
- **Epsilon before inverse/sqrt**: Add epsilon to eigenvalues/diagonals before `sqrt` or matrix
  inverse operations.
- **`is_near_zero` must use abs**: `return x < epsilon()` is wrong (true for all negatives).
  Must be `return abs(x) < epsilon()`.

### Priority 3: Float Precision in Constants

A single wrong constant suffix can measurably degrade kernel performance.

Check for:
- **Always use `f` suffix on float constants**: `1.f/255.f` not `1.f/255.0` (the latter promotes
  to `double`, causing register pressure and ALU stalls). This was measured to produce visible
  perf gains in the rasterization kernel.
- **Named constants over magic numbers**: Use `ALPHA_THRESHOLD`, `MAX_ALPHA`,
  `TRANSMITTANCE_THRESHOLD` from `Common.h` instead of inline `1/255.0`, `0.99f`, `1e-4f`.
- **Constants must match between CUDA and Python**: `Common.h` constants (e.g.,
  `MIN_COMPENSATION`, `ALPHA_THRESHOLD`) must be mirrored exactly in `gsplat/cuda/_constants.py`.
  Updating one without the other causes test failures.

### Priority 4: Const Correctness

Check for:
- **Use `const_data_ptr<T>()` for read-only kernel inputs**: Not `data_ptr<T>()`. The former
  returns a const pointer matching the kernel's `const` parameter declaration.
- **Apply to both forward AND backward launches**: Easy to miss in backward where you might
  copy-paste from forward.
- **`__restrict__` on non-aliasing pointers**: All kernel pointer parameters that don't alias
  should use `__restrict__`.

### Priority 5: Redundant/Duplicate Checks

Check for:
- **Don't re-check conditions already guaranteed by upstream kernels**: If the projection kernel
  already rejects Gaussians with negative determinant and invalid visibility, don't re-check
  in the intersection kernel.
- **Single authoritative threshold**: Don't have two overlapping thresholds (e.g., both
  `MAX_KERNEL_DENSITY_CUTOFF` and `ALPHA_THRESHOLD`). Consolidate to one.
- **Shared header constants**: Projection and intersection code should use the same values
  from `Common.h` (`GAUSSIAN_EXTEND`, `ALPHA_THRESHOLD`), not bake in their own copies.

### Priority 6: Undefined Behavior

Check for:
- **Signed before unsigned cast**: When computing tile bounds from `floor()`/`ceil()` of
  floating-point values, cast to `int` first, clamp with signed `min`/`max`, THEN convert to
  `uint32_t`. Casting a negative float directly to `uint32_t` is undefined behavior.
- **`fminf` vs `std::clamp` for mathematically non-negative values**: For squared values like
  `x*x + y*y`, the result cannot be negative so clamping the lower bound to 0 is redundant.
  Use `fminf`/`std::min` to avoid unnecessary NaN-handling branches.
- **Consistent C vs C++ math**: Don't mix `sqrtf` with `std::min`. Pick one style per file.

### Priority 7: Memory Access Patterns

Check for:
- **Wider memory loads**: Prefer `float2`/`float4` loads over individual `float` loads in hot
  kernel paths (e.g., loading ray origin+direction as 3x64b instead of 6x32b).
- **Upcast for precision-sensitive intermediates**: When computing quaternion-to-matrix or
  similar transforms, temporarily upcast `float32` to `float64`, compute, then cast back.

### Priority 8: Error Handling

Check for:
- **Use PyTorch's `C10_CUDA_CHECK`**: Not custom `CUDA_CHECK` macros. PyTorch's macro is
  consistent with the framework's error handling.
- **Headers must be self-contained**: Every header must `#include` all types it uses directly.
  Don't rely on transitive includes (e.g., `<cstdint>` for `uint8_t`).

### Priority 9: Device/Host Duality

Check for:
- **Prefer `__device__ __host__` over duplicate implementations**: Camera models and math
  utilities should use dual `__device__ __host__` annotations so the same code serves both
  CUDA kernels and PyTorch reference implementations.

## Step 4: Format the review

### Blocking
- Thread synchronization issues (missed barriers, per-pixel data in shared memory)
- NaN safety violations (unclamped trig inputs, unguarded divisions, missing epsilon)
- Undefined behavior (negative-to-unsigned casts, aliasing violations)
- Constant mismatches between CUDA and Python

### Should-Fix
- Missing `f` suffix on float constants
- `data_ptr` instead of `const_data_ptr` for read-only inputs
- Redundant checks already guaranteed by upstream kernels
- Mixed C/C++ math function styles

### Nit
- Minor naming inconsistencies
- Opportunities for wider memory loads
- Style preferences in kernel structure

## Output Format

```markdown
# CUDA Code Review

Branch: `{branch_name}`
Files reviewed: {count}

---

## Blocking

### {file_path}:{line_number} -- {brief description}

{Direct explanation with concrete fix.}

```cuda
// the fix
```

---

## Should-Fix

### {file_path}:{line_number} -- {brief description}

{Explanation with alternative.}

---

## Nit

### {file_path}:{line_number} -- {brief description}

{Brief note.}

---

{Acknowledge what was done well if warranted.}
```

## Voice Calibration

- Be **precise about the failure mode**: "This will hang on SM 8.0+ because threads 16-31 skip
  the barrier" not "this might have sync issues."
- **Show the fix**: Don't just flag -- provide the corrected kernel code.
- **Reference the constant**: "Use `ALPHA_THRESHOLD` from Common.h:42" not "use a named constant."
- **Cite perf evidence when available**: "Removing the double promotion here saved ~10us/call
  in prior profiling."
