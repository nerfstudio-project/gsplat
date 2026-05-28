<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved. -->
# `arch/` — per-backend overlay tree

This directory holds the AMD ROCm overrides for the gsplat tri-target
build. Backend selection happens at build time via the `GSPLAT_BACKEND`
environment variable, read by `gsplat/cuda/build.py`:

| `GSPLAT_BACKEND` | Default arch list | Sources used |
|---|---|---|
| `cuda` *(default)* | nvcc native | upstream `gsplat/cuda/csrc/*.cu` (NO `arch/` files used) |
| `rocm_radeon` | `gfx1100;gfx1200` | `arch/radeon/...` + `arch/rocm/...` |
| `rocm_instinct` | `gfx942;gfx950` | `arch/instinct/...` + `arch/rocm/...` |

## Subdirectories

- **`radeon/`** — files that diverge between Radeon (RDNA) and Instinct
  (CDNA) builds and were hand-tuned for Radeon. Examples:
  `gsplat/cuda/csrc/IntersectTile.hip`, `gsplat/cuda/include/Common.h`.
- **`instinct/`** — same paths as `radeon/`, hand-tuned for Instinct.
- **`rocm/`** — files identical between Radeon and Instinct but distinct
  from the upstream CUDA version. Used by BOTH AMD backends. Examples:
  `gsplat/cuda/csrc/Adam.cpp`, `gsplat/cuda/include/Cameras.hip.h`,
  `gsplat/cuda/_backend.py`, `MANIFEST.in`.

The corresponding upstream `.cu`/`.cuh`/`.h` files **remain untouched** at
their original paths, so the CUDA backend is unaffected by anything under
`arch/`.

## Adding a new file

1. If the change affects ALL ROCm builds identically and differs from
   upstream → place under `arch/rocm/<original-path>/`.
2. If the change is per-arch (different gfx tunings, wave size, etc.) →
   place a copy under `arch/radeon/<original-path>/` AND another under
   `arch/instinct/<original-path>/`.
3. Never use `#ifdef __gfx_xxx__` to merge the two AMD variants — the
   project policy is to keep them fully separated to preserve hand
   tuning. The ifdef-merging path was rejected during translation.

## How `build.py` finds files

For ROCm builds, `gsplat/cuda/build.py` constructs the source list as:

```
sorted(arch/<variant>/gsplat/cuda/csrc/*.hip)
+ sorted(arch/rocm/gsplat/cuda/csrc/*.cpp)
+ gsplat/cuda/ext.cpp        # upstream, identical across backends
```

and the include search path as:

```
arch/<variant>/gsplat/cuda/include/hip_shim   # cooperative_groups shim
arch/rocm/gsplat/cuda/include                  # shared HIP headers
arch/<variant>/gsplat/cuda/include             # variant-specific headers
gsplat/cuda/include                            # upstream fallback
third_party/cgshim                             # cuda::std polyfills
third_party/glm  OR  /opt/glm                  # GLM (vendored or system)
```

For the CUDA backend nothing under `arch/` is touched.

## Vendored GLM

`third_party/glm/` is a vendored copy of g-truc/glm used by both ROCm
backends to avoid PyTorch's `hipify_python` mirroring/mangling the
header-only library inside the `gsplat/cuda/` subtree. The CUDA
backend continues to use the upstream git submodule at
`gsplat/cuda/csrc/third_party/glm/` (run `git submodule update --init`).
