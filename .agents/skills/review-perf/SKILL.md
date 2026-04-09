---
name: review-perf
description: >
  Domain-specific performance review for gsplat -- focused on GPU-CPU synchronization avoidance,
  profiling evidence requirements, CUDA kernel efficiency, Python-side vectorization, and tracing
  infrastructure. Use when reviewing optimization MRs, kernel changes with perf claims, contract
  assertions on hot paths, or any change to rendering/rasterization throughput.
---

# Performance Code Review (gsplat)

You are reviewing code for performance implications in the gsplat Gaussian splatting library.
This review enforces patterns distilled from the project's merged MR review history. Every rule
below was triggered by a real performance regression, unsubstantiated claim, or avoidable
GPU-CPU sync caught in review.

## Step 1: Get the diff

Run these in parallel:

```bash
git diff nv/main...HEAD
git diff --name-only nv/main...HEAD
git log nv/main...HEAD --oneline
```

## Step 2: Read changed files in full

For each changed file, read it in full. Pay special attention to hot paths: rasterization kernels,
projection kernels, loss computation, and the main `rasterization()` function in `rendering.py`.

## Step 3: Review through the performance lens

### Priority 1: GPU-CPU Synchronization Avoidance

The single most impactful performance concern in this codebase. One sync can dominate total frame time.

Check for:
- **Contract assertions must not cause GPU-CPU syncs**: Checks like `(images >= 0).all()` launch
  a GPU kernel AND sync to read the boolean result. These must be gated behind an
  `ENFORCE_CONTRACTS` flag (default: False).
- **Distinguish cheap from expensive assertions**:
  - **CPU-only (cheap)**: shape checks, dtype checks, `.dim()`, `.size()` -- always OK
  - **GPU kernel + sync (expensive)**: `.all()`, `.any()`, `.min()`, `.max()` on GPU tensors --
    must be gated
- **Hidden syncs in Python**: `tensor.item()`, `bool(tensor)`, `float(tensor)`, printing a
  tensor, or comparing a GPU tensor to a scalar all trigger syncs. Flag these on hot paths.

### Priority 2: Profiling Evidence Requirements

Performance claims without evidence are rejected in this project.

Check for:
- **nsys/ncu profiling required for optimization MRs**: "We typically describe a little bit the
  context with before and after snapshots of nsys (and if possible ncu)."
- **Include dataset description and resolution**: Perf numbers without context are meaningless.
  Specify the scene, image resolution, and GPU model.
- **Don't rely on end-to-end training time for kernel perf**: MagLev/training loop timing has
  ~5% variance. Use Nsight Systems for kernel-level measurement.
- **Even trivial-looking changes need profiling**: `std::clamp` to `std::min` was validated with
  SASS diff, stall metrics, and estimated speedup. The overhead was real.
- **Scope optimizations to validated paths first**: Apply to 3DGS first, validate, then extend
  to 3DGUT/2DGS. Don't apply universally before validating on each variant.

### Priority 3: CUDA Kernel Efficiency

Check for:
- **Float constant promotion to double**: `1.f/255.0` promotes to `double` due to the
  unsuffixed `255.0`. Always use `1.f/255.f`. This was measured to produce visible perf gains.
- **Unnecessary NaN-handling branches**: For values that are mathematically non-negative (like
  `x*x + y*y`), use `fminf`/`std::min` instead of `std::clamp`. Clamp adds a lower-bound
  branch that the hardware must evaluate.
- **Wider memory loads**: Prefer `float2`/`float4` loads over individual `float` loads in hot
  paths (e.g., 3x64b instead of 6x32b for ray data).
- **Template specialization vs runtime branching**: Don't speculatively add compile-time template
  variants. Measure runtime cost first; only template-specialize if the cost is significant.
  More templates = slower builds.
- **`__noinline__` for compile time vs runtime tradeoff**: When deeply nested code in unrolled
  loops causes extreme compile times, `__noinline__` is acceptable. Document the measured
  runtime impact (e.g., "~74us increase on projection").

### Priority 4: Python-Side Performance

Check for:
- **Use `torch.lerp` over manual interpolation**: `torch.lerp` enables fusion and avoids
  intermediate tensor allocations.
- **Use `math.prod()` for shape arithmetic**: Not `torch.prod(torch.tensor([...]))`.
- **Stay on-device**: Avoid unnecessary CPU-GPU transfers. Use `torch.from_numpy(array).to(device)`
  over `torch.tensor(array, device=device)` (the latter copies).
- **Python for-loops over tensors**: Flag any Python loop that iterates over tensor elements.
  Use vectorized ops: `torch.roll`, `torch.einsum`, `einops.repeat`.

### Priority 5: Tracing and Profiling Infrastructure

Check for:
- **NVTX tracing should be opt-in**: Disabled by default. In production, traces being recorded
  by default is undesirable.
- **NVTX trace names should match function names**: "It should be easier to find the code/trace
  mapping." If the function name isn't clear enough, rename the function.
- **Use context managers over push/pop**: `trace_range` (context manager) or `trace_function`
  (decorator) instead of manual `trace_push`/`trace_pop`. If push/pop is used, wrap in
  `try/finally`.
- **nvtx/PyTorch conflicts**: The `nvtx` package can conflict with PyTorch's built-in NVTX.
  Investigate before adopting.

### Priority 6: Anti-aliasing and Culling Performance

gsplat-specific performance patterns:

Check for:
- **`MIN_COMPENSATION` floor for anti-aliasing**: The compensation factor `sqrt(det_orig /
  det_blur)` must be floored at `MIN_COMPENSATION` (0.005), not zero. Zero would cull small
  Gaussians entirely. This is independent of `ALPHA_THRESHOLD`.
- **Single authoritative culling threshold**: Don't have redundant thresholds that overlap
  (e.g., both `MAX_KERNEL_DENSITY_CUTOFF` and alpha-based culling).

## Step 4: Format the review

### Blocking
- Unguarded GPU-CPU syncs on hot paths (assertions, `.item()`, `.all()`)
- Performance claims without nsys/ncu evidence
- Float-to-double promotion in CUDA constants

### Should-Fix
- Missing `ENFORCE_CONTRACTS` guard on value-range assertions
- Python for-loops over tensor elements
- Manual interpolation where `torch.lerp` would work
- NVTX tracing enabled by default

### Nit
- Wider memory load opportunities
- Minor template vs runtime suggestions
- Trace name mismatches

## Output Format

```markdown
# Performance Code Review

Branch: `{branch_name}`
Files reviewed: {count}

---

## Blocking

### {file_path}:{line_number} -- {brief description}

{Explanation of the perf impact with concrete fix.}

**Impact**: {estimated or measured impact}

```{language}
// the fix
```

---

## Should-Fix

### {file_path}:{line_number} -- {brief description}

{Explanation with suggested improvement.}

---

## Nit

### {file_path}:{line_number} -- {brief description}

{Brief note.}

---

{Acknowledge good perf work if warranted.}
```

## Voice Calibration

- **Quantify when possible**: "This `.all()` call adds ~200us of GPU-CPU sync per frame"
  not "this might be slow."
- **Demand evidence**: "Show nsys before/after on [dataset] at [resolution] on [GPU]."
- **Distinguish measured from estimated**: "Measured 10us regression" vs "Estimated impact
  from SASS analysis."
- **Acknowledge valid tradeoffs**: "The `__noinline__` adds ~74us but saves 3 minutes of
  compile time -- reasonable tradeoff for dev workflow."
