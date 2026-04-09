---
name: review-testing
description: >
  Domain-specific test review for gsplat -- focused on test quality, parameterization, tolerances,
  CUDA skip guards, gradient verification, reference value testing, and edge case coverage.
  Use when reviewing files under tests/, or any changes that add, modify, or should include tests.
  Also use when a non-test MR lacks test coverage for new functionality.
---

# Testing Code Review (gsplat)

You are reviewing test code in the gsplat Gaussian splatting library. This review enforces
patterns distilled from the project's merged MR review history. Every rule below was triggered by
a real missed bug, flaky test, or insufficient coverage caught in review.

## Step 1: Get the diff

Run these in parallel:

```bash
git diff nv/main...HEAD -- 'tests/**' '*.py'
git diff --name-only nv/main...HEAD
git log nv/main...HEAD --oneline
```

## Step 2: Read changed files in full

For each changed test file, read it in full. Also read `tests/conftest.py` for shared fixtures
and `gsplat/_helper.py` for test utilities (`assert_mismatch_ratio`, `load_test_data`).

If the MR changes non-test code, check whether corresponding tests exist or were added.

## Step 3: Review through the testing lens

### Priority 1: Test Coverage Exists

New functionality without tests is the most common review blocker.

Check for:
- **New functions/classes must have tests**: "I'd like to have tests for this ensuring their
  correctness before approval."
- **Bug fixes must have regression tests**: A test that fails before the fix and passes after.
- **Comments asserting behavior should be actual tests**: "I wouldn't write that as a comment,
  but rather add as unit test."
- **Customer-facing code must be tested**: If targeting external users, test coverage is
  non-negotiable.
- **Gradient flow must be tested for differentiable functions**: Every differentiable loss or
  reduction needs a gradient verification test. Use `torch.autograd.gradcheck` or verify
  `grad is not None` with magnitude checks.

### Priority 2: Test Quality -- Values, Not Just Existence

Tests that run without asserting anything meaningful are worse than no tests.

Check for:
- **Test with hardcoded reference values**: "What if we had a reference set of (perhaps
  hardcoded) values that we could compare against?" Don't just assert `is not None` in gradient
  tests -- verify magnitude ranges against known-good references.
- **Tests must validate quality, not just execution**: "As written these tests will perform
  rendering, but they won't check the quality of results."
- **Tests that silently skip assertions are dangerous**: Tests that only run assertions when
  data is present but silently pass when it's not are worse than no tests -- they give false
  confidence.
- **Gate assertions on actual output shape**: Don't assume RGB channels exist for depth-only
  render modes. Check `render_mode` or output shape before slicing.

### Priority 3: Parameterization and Fixtures

gsplat tests are heavily parameterized. Bad parameterization causes combinatorial explosion or
missed coverage.

Check for:
- **Make key dimensions visible in parametrization**: Important dims (`C`, `N`, `batch_dims`)
  should be `@pytest.mark.parametrize` arguments, not hardcoded inside fixtures.
- **Extract reusable setup into dedicated fixtures**: Gaussian generation, sensor model setup,
  and similar boilerplate should be conftest fixtures, not inline in each test.
- **Test body should focus on behavior, not setup**: If setup code dominates the test body,
  extract it.
- **Parameter count must match function signature**: `product()` generating test parameters
  had 9 elements but the test accepted 8 -- causes `TypeError` at runtime. Always verify
  alignment.
- **Missing boundary values**: "Can we also have values `{min - 0.1, min, min + 0.1}`?"
  Test at, just below, and just above thresholds.
- **Contract/validation tests in a separate class**: Group input validation tests in
  `TestInputContracts`, separate from functional tests.

### Priority 4: Tolerances

Tolerances that are too loose mask real bugs. Tolerances that are too tight cause flaky tests.

Check for:
- **Tighten tolerances after bug fixes**: After fixing a race condition, tolerances were
  tightened from 4e-2 to 1e-3. If your fix makes results more accurate, tighten the tolerance
  to prove it.
- **Document inter-GPU variability**: If a tolerance is relaxed (e.g., 1e-3 to 1.3e-3),
  add a comment explaining why (e.g., "inter-GPU variability on A100 vs V100").
- **Edge case tests document expected behavior**: If empty input returns NaN per PyTorch
  default, decide whether to guard and return 0 instead, then test the chosen behavior.

### Priority 5: CUDA and Feature Skip Guards

Check for:
- **CUDA availability**: `@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")`
  on all GPU-dependent tests.
- **Feature availability**: Use conditional parametrization based on `has_camera_wrappers()`,
  `has_3dgs()`, etc. rather than failing with confusing import errors.
- **Skip with clear messages**: `pytest.skip("Camera wrappers not built")` not a bare skip.
- **Follow existing patterns**: `test_rasterization.py:27` shows the canonical pattern for
  conditional test parametrization based on feature availability.

### Priority 6: Reproducibility

Check for:
- **Deterministic seeds**: Use `torch.Generator` with explicit seed (typically `seed=42`).
  The conftest autouse fixture sets a global seed, but test helpers that create random data
  should also accept a `seed` parameter.
- **Conftest autouse fixture**: Verify that `setup_test_environment()` runs (sets seed=42,
  clears CUDA cache, runs GC). New conftest files should not override this.

### Priority 7: Cross-Validation

Check for:
- **Rendering changes need before/after comparison**: Any change to rasterization constants
  or thresholds must include before/after rendering comparison AND metrics (PSNR/SSIM/LPIPS)
  on a standard scene.
- **Reference implementations must be verified**: Pure-torch reference implementations used
  for testing must themselves be validated against a known-good implementation.
- **Metadata key consistency**: When two code paths set metadata dicts, verify they use the
  same key names.
- **Test large binary assets via Git LFS**: `.npz` files and similar go through LFS with
  `.gitattributes` entries.

## Step 4: Format the review

### Blocking
- New functionality with no tests
- Bug fixes without regression tests
- Tests that silently skip assertions
- Parameter count mismatches in parameterization
- Missing CUDA skip guards (tests that crash without GPU)

### Should-Fix
- Loose tolerances that could mask regressions
- `is not None` gradient tests without magnitude checks
- Setup-heavy test bodies that should use fixtures
- Hardcoded dimensions that should be parameterized

### Nit
- Minor fixture organization improvements
- Additional boundary value suggestions
- Style preferences in test structure

## Output Format

```markdown
# Testing Code Review

Branch: `{branch_name}`
Files reviewed: {count}

---

## Blocking

### {file_path}:{line_number} -- {brief description}

{Explanation of what's missing and why it matters.}

**Required test:**
```python
# skeleton of the missing test
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

{Acknowledge good test design if warranted.}
```

## Voice Calibration

- **Demand coverage**: "I'd like to have tests for this before approval."
- **Question loose assertions**: "What does `is not None` actually prove here? Can we verify
  the gradient magnitude is within [0.01, 10.0]?"
- **Suggest concrete test skeletons**: Don't just say "add a test" -- show the parametrize
  decorator and key assertions.
- **Flag false confidence**: "This test looks green but it's not actually checking anything
  when the data isn't present."
