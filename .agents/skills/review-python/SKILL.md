---
name: review-python
description: >
  Domain-specific Python code review for gsplat -- focused on API design, tensor safety, losses module
  conventions, rendering pipeline correctness, type annotations, and Python version compatibility.
  Use when reviewing .py files in gsplat/, especially rendering.py, _wrapper.py, _torch_impl*.py,
  losses.py, strategy/, optimizers/, or any Python code that interfaces with the CUDA backend.
---

# Python Code Review (gsplat)

You are reviewing Python code in the gsplat Gaussian splatting library. This review enforces
patterns distilled from the project's merged MR review history. Every rule below was triggered by
a real bug, regression, or API design issue caught in review.

## Step 1: Get the diff

Run these in parallel:

```bash
git diff nv/main...HEAD -- '*.py'
git diff --name-only nv/main...HEAD
```

## Step 2: Read changed files in full

For each changed Python file, read it in full. Also read `gsplat/__init__.py` to understand the
public API surface and `gsplat/cuda/_constants.py` for shared constants.

## Step 3: Review through the Python lens

### Priority 1: Tensor Shape and Type Safety

The most common source of silent bugs in this codebase.

Check for:
- **Validate shapes before flattening**: "Flattening can silently pair the wrong elements when
  shapes differ but `numel()` matches." Always verify `pred.shape == gt.shape` and mask shape
  compatibility before flattening or masking, with a clear `ValueError`.
- **Preserve dtype/device/requires_grad in edge cases**: When `k == 0` or tensors are empty,
  returning `torch.tensor(0.0)` loses the input's dtype, device, and gradient tracking. Use
  `torch.zeros((), device=value.device, dtype=value.dtype)` instead.
- **Contiguity before C++ interop**: Always call `.contiguous()` on tensors before passing to
  C++ extensions. Stride mismatches cause silent corruption or crashes.
- **Validate tensor invariants in Python wrappers**: Assert dtype, contiguity, and shape in
  the Python wrapper before calling into CUDA launchers. The kernel assumes these properties.
- **Float masks rejected in reductions**: `reduce_mean` with mask requires bool or integer
  dtype (not float) to avoid incorrect clamping when mask sum is between 0 and 1.

### Priority 2: API Design

Check for:
- **Remove unused/derivable parameters**: If a parameter can be derived from another argument
  (e.g., `K` from `coeffs.shape`), remove it from the function signature. Fewer arguments =
  cleaner API.
- **Don't couple caching to the library API**: `lru_cache` on library-internal factory functions
  couples optimization concerns to the data model. The library should be stateless; callers
  decide when/how to cache.
- **Explicit constructor parameters over `**kwargs`**: Make APIs self-documenting. Accept
  `(params, angles_map, tiling)` instead of `**params`.
- **New public symbols must be in `__all__`**: Importing into `__init__.py` is not enough. Add
  to `__all__` for proper `from gsplat import *` behavior and API docs.
- **Eliminate duplicate Python wrappers for C++ bound classes**: Don't maintain parallel Python
  classes for pybind11-bound C++ classes. Use the bindings directly with aliases.
- **Remove `export_values()` for scoped enums**: Only needed for legacy non-scoped enums. With
  scoped enums it pollutes the namespace.
- **Proliferating optional parameters is a smell**: When a function accumulates many optional
  params, it signals the need for a configuration object or builder pattern.
- **Separate domain-specific types into their own files**: LiDAR types go in `_torch_lidars.py`,
  camera types in `_torch_cameras.py`. Don't mix domains in one file.

### Priority 3: Losses Module Conventions

The losses subsystem has specific architectural rules.

Check for:
- **Loss functions return unreduced per-element tensors**: Reduction (mean, sum, quantile) is
  a separate step. This gives callers maximum flexibility for masking, weighting, and custom
  aggregation.
- **Losses receive post-activation values**: The loss system always receives post-activation
  values (e.g., `exp(log_scales)`, `sigmoid(opacity_logits)`). Activation happens in the model's
  getter methods, not in the loss function.
- **Docstring-implementation consistency**: If the docstring says "half-extents" but the code
  divides by 2, one of them is wrong. Verify units and semantics match.
- **Division-by-zero guards**: When computing disparity (`1.0 / depth`) or similar, guard against
  zero values using `torch.where`.
- **Validate scheduler parameters at init time**: Check `update_frequency > 0`, `end > start`,
  `total_stages > 0` in `__init__`, not at runtime.

### Priority 4: Python Version Compatibility

Check for:
- **No PEP 646 generalized unpacking on Python 3.10**: `self.data[*args]` syntax is not
  supported. Use `self.data.__getitem__(*args)` with a comment explaining the workaround.
- **`torch.device("cpu")` as a default argument**: Function defaults are evaluated at import
  time. Use `None` as default and create the device inside the function body.
- **`torch.amp.autocast` must be device-aware**: Don't hardcode `device_type="cuda"`. Use
  `input.device.type` instead.

### Priority 5: Code Quality

Check for:
- **`raise ValueError` over bare `assert` for validation**: `assert` statements vanish under
  Python's `-O` flag. Use explicit `if ... raise ValueError` with informative messages.
- **No self-assignments from refactoring**: `ut_params = ut_params` is a no-op. Remove entirely.
- **Prefix unused variables with underscore**: In `autograd.Function.backward`, prefix unused
  saved tensors with `_` (e.g., `_result_point`). Remove dead assignments entirely.
- **`def` instead of lambda assignments**: Convert `clamp_az = lambda v: ...` to
  `def clamp_az(v): return ...` for debuggability (E731).
- **Bind loop variables in lambda captures**: In `lambda a, b: f(a, b, t)` inside a loop,
  bind `t` explicitly: `lambda a, b, t=t: f(a, b, t)`.
- **Use `torch.lerp` over manual interpolation**: `torch.lerp` enables PyTorch to fuse
  operations and skip intermediate allocations.
- **Use `math.prod()` for shape arithmetic**: Not `torch.prod(torch.tensor(...))`.
- **Group imports at file top**: No scattered import blocks through the file.

### Priority 6: Coordinate Conventions

Check for:
- **Consistent coordinate ordering throughout the pipeline**: The LiDAR pipeline mixed
  `[elevation, azimuth]` and `[azimuth, elevation]` conventions. Standardize to a single
  convention across Python, CUDA, tiling, and tests. Document the chosen convention.

### Priority 7: Error Handling

Check for:
- **Gate TODO assertions with real errors**: Replace `# TODO: assert if rays are given` with
  `raise NotImplementedError("Ray inputs require with_eval3d=True")`.
- **Extract duplicated validation to shared helpers**: Identical validation blocks in multiple
  functions should be extracted into a `_validate_and_pad_channels()` helper.
- **Guard `CUDA_HOME` with None check**: `os.path.join(CUDA_HOME, "targets")` will `TypeError`
  if `CUDA_HOME` is `None`.

## Step 4: Format the review

### Blocking
- Shape/dtype mismatches that cause silent corruption
- Missing contiguity checks before CUDA calls
- Public API surface issues (__all__, breaking changes)
- Losses returning reduced values instead of per-element
- Python 3.10 incompatibilities

### Should-Fix
- Unused/derivable parameters in function signatures
- Caching coupled to library internals
- Missing input validation (bare asserts, unguarded divisions)
- Dead code from refactoring

### Nit
- Style issues (lambda assignments, import grouping)
- Minor naming improvements
- `torch.lerp` / `math.prod` opportunities

## Output Format

```markdown
# Python Code Review

Branch: `{branch_name}`
Files reviewed: {count}

---

## Blocking

### {file_path}:{line_number} -- {brief description}

{Explanation with concrete fix.}

```python
# the fix
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

- **Lead with the failure mode**: "This will silently pair wrong elements when shapes differ"
  not "shapes should be validated."
- **Show the fix**: Provide the corrected code, not just the principle.
- **Reference the constant**: Point to `_constants.py:12` or `__init__.py:45`, not just the
  file name.
- **Flag Python version traps explicitly**: "This syntax requires Python 3.11+ but we support
  3.10."
