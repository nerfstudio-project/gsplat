---
name: review-infra
description: >
  Domain-specific build and infrastructure review for gsplat -- focused on CUDA compilation,
  dependency management, header correctness, feature gating, copyright headers, and security
  (no pickle, no personal forks). Use when reviewing setup.py, build.py, pyproject.toml,
  requirements files, Dockerfiles, CI configs, or any change that affects compilation,
  packaging, or dependency resolution.
---

# Build & Infrastructure Code Review (gsplat)

You are reviewing build system and infrastructure code in the gsplat Gaussian splatting library.
This review enforces patterns distilled from the project's merged MR review history. Every rule
below was triggered by a real build failure, security concern, or compatibility issue.

## Step 1: Get the diff

Run these in parallel:

```bash
git diff nv/main...HEAD -- setup.py 'gsplat/cuda/build.py' pyproject.toml '*.cfg' '*.toml' '*.txt' 'Dockerfile*' '.github/**' '*.h' '*.cuh'
git diff --name-only nv/main...HEAD
```

## Step 2: Read changed files in full

For each changed build file, read it in full. Also check `setup.py` and `gsplat/cuda/build.py`
for context on the current build configuration.

## Step 3: Review through the build lens

### Priority 1: Security

Check for:
- **Never use pickle for serialization**: Pickle can execute arbitrary code upon loading. Use
  JSON for build parameters, configuration, or any persisted data. This is an NVIDIA security
  policy.
- **Don't depend on personal forks**: Using personal GitHub forks introduces supply-chain risk.
  Pin to official packages or mirror internally.
- **Pin git dependencies to specific commits**: Always use a commit hash for git-based
  dependencies, not branches or tags, for reproducible builds.
- **Don't expose secrets in build configs**: No API keys, internal URLs, or credentials in
  build scripts or CI configs.

### Priority 2: CUDA Compilation

Check for:
- **`torch` in pyproject.toml build-requires**: `setup.py` imports `torch.utils.cpp_extension`,
  so `torch` must be in `[build-system] requires`. This cannot be solved in setup.py alone.
- **Circular import avoidance**: Use `importlib.util.spec_from_file_location` to dynamically
  load `gsplat/cuda/build.py` without triggering the full package import chain.
- **`FAST_MATH=0` in debug mode**: Debug configurations must explicitly disable fast-math to
  get exact floating-point behavior for debugging.
- **`-Werror` for non-Windows builds**: Treat compiler warnings as errors. Clean up unused
  variables, signed/unsigned mismatches.
- **`__noinline__` for compile time control**: When deeply nested code in unrolled loops causes
  extreme compile times, use `__noinline__` on specific functions rather than removing templates.
  Accept manageable perf hit.
- **Don't duplicate build flags**: One env var per concept. Don't create `DEBUG` if `FAST_COMPILE`
  already controls the same behavior.

### Priority 3: Feature Gating

Check for:
- **Environment variables for conditional compilation**: `BUILD_3DGUT`, `BUILD_3DGS`,
  `BUILD_2DGS`, `BUILD_ADAM`, `BUILD_RELOC` -- provide granular build flags for dev workflow
  speed.
- **Runtime feature checks match build flags**: `has_3dgs()`, `has_2dgs()`, etc. must correctly
  reflect what was built.
- **Always build everything for release/CI**: Granular flags are for development only.

### Priority 4: Dependency Management

Check for:
- **Exclusion ranges over upper-bound pins**: Use `>=0.8.8,!=1.0.9,!=1.0.10` to exclude
  known-buggy versions while allowing future releases. Don't use `<2.0` caps.
- **Guard `CUDA_HOME` with None check**: `os.path.join(CUDA_HOME, "targets")` will `TypeError`
  if `CUDA_HOME` is `None`. Always check `if CUDA_HOME and os.path.isdir(...)`.
- **nvtx/PyTorch conflicts**: `nvtx` package can conflict with PyTorch's own NVTX wrappers.
  Investigate before adopting; consider using Torch's NVTX instead.

### Priority 5: Header Correctness

Check for:
- **Headers must be self-contained**: Every `.h`/`.cuh` file must `#include` all types it uses
  directly. Don't rely on transitive includes (e.g., `<cstdint>` for `uint8_t`).
- **C++ standard justification**: If choosing C++20 over C++17, document the rationale (e.g.,
  `TensorView` utility). Newer standards reduce adoption for downstream projects.

### Priority 6: Copyright Headers

Check for:
- **All new files must have SPDX headers**: Both Python and C/C++/CUDA files.
- **Python format**:
  ```
  # SPDX-FileCopyrightText: Copyright (c) YEAR NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  ```
- **C/C++/CUDA format**: Block comment `/* ... */` style.
- **End date is current year (2026)**: Per project convention.

### Priority 7: Upstream Compatibility

Check for:
- **No internal product names in upstreamable code**: Keep internal names out of code destined
  for the public GitHub repo.
- **Revert-and-reland workflow**: If a feature causes CI failures, revert first to unblock main,
  then fix and reland. Don't leave main broken.

## Step 4: Format the review

### Blocking
- Pickle usage for serialization
- Personal fork dependencies
- Missing `torch` in build-requires
- Circular imports at build time
- Missing copyright headers on new files
- Secrets in build configs

### Should-Fix
- Missing `CUDA_HOME` guards
- Duplicate build flags
- Upper-bound dependency pins
- Non-self-contained headers

### Nit
- Minor flag naming preferences
- Compile time optimization opportunities
- Style preferences in build scripts

## Output Format

```markdown
# Build & Infrastructure Review

Branch: `{branch_name}`
Files reviewed: {count}

---

## Blocking

### {file_path}:{line_number} -- {brief description}

{Explanation with concrete fix.}

---

## Should-Fix

### {file_path}:{line_number} -- {brief description}

{Explanation with alternative.}

---

## Nit

### {file_path}:{line_number} -- {brief description}

{Brief note.}
```

## Voice Calibration

- **Security is non-negotiable**: "Pickle can execute arbitrary code. Use JSON."
- **Be specific about the failure**: "This will `TypeError` when `CUDA_HOME` is None on a
  CPU-only machine."
- **Reference the policy**: "Per NVIDIA security policy, no new pickle usage."
- **Suggest the minimal fix**: Don't restructure the build system when a one-line guard suffices.
