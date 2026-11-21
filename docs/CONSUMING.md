# Using gsplat as a dependency

This fork is meant to be consumed by other research projects. Four supported
ways, ordered from "just reference the repo" to "fully prebuilt".

| Method | Needs `nvcc` | Compile time | Isolation must be off | Notes |
| --- | --- | --- | --- | --- |
| uv project, git source | yes | ~45 min once per env | yes | pin a tag or commit |
| `uv pip install "gsplat @ git+ssh://…"` | yes | ~45 min once per env | yes | quick one-off |
| Prebuilt wheel (GitHub Release) | no | none | n/a | tied to torch/CUDA/python/arch |
| Local path, editable | yes | incremental | n/a | co-development |

Source installs on a CUDA 12.x toolkit need no extra flags: the build system
pre-includes the `cuda::ceil_div` compatibility header automatically (see
[CUDA12.md](CUDA12.md)). `NVCC_FLAGS` remains available as an override.

## 1. uv project, git source

```toml
# pyproject.toml
[project]
dependencies = ["gsplat"]

[tool.uv.sources]
gsplat = { git = "ssh://git@github.com/Gotti27/gsplat.git", tag = "v1.6.0+leo1" }
```

```bash
UV_NO_BUILD_ISOLATION=1 uv sync
```

Pin a commit instead of a moving branch when reproducibility matters:

```toml
gsplat = { git = "ssh://git@github.com/Gotti27/gsplat.git", rev = "<40-char sha>" }
```

### Why `UV_NO_BUILD_ISOLATION=1` is required

`pyproject.toml` declares `torch>=2.7` as a build requirement. With isolation
enabled, uv installs a *fresh* torch from PyPI into a temporary build
environment and links `gsplat/csrc.so` against it. The resulting extension then
fails to import in your environment, which uses a different torch build (for
example `torch==2.9.1+cu126` from the PyTorch index). Disabling isolation makes
the build use your installed torch.

The same applies to `uv pip`:

```bash
UV_NO_BUILD_ISOLATION=1 uv pip install \
  "gsplat @ git+ssh://git@github.com/Gotti27/gsplat.git@v1.6.0+leo1"
```

### glm submodule

`gsplat/cuda/csrc/third_party/glm` is a submodule and is required by the CUDA
headers. uv and pip clone submodules for VCS installs; if you copy the tree
manually, initialise it first:

```bash
git submodule update --init gsplat/cuda/csrc/third_party/glm
```

## 2. Prebuilt wheel

Download the wheel attached to the matching GitHub Release and reference it
directly (no `nvcc`, no compile, no isolation concerns):

```bash
uv add "gsplat @ https://github.com/Gotti27/gsplat/releases/download/v1.6.0+leo1/gsplat-1.6.0+leo1-cp314-cp314-linux_x86_64.whl"
```

A wheel is **not** universal:

- it is linked against one **torch minor** (e.g. 2.9) and one **CUDA major**
  (e.g. 12.6);
- it is tagged for one **Python minor** (e.g. cp314);
- it contains device code for the architectures in `TORCH_CUDA_ARCH_LIST` at
  build time (e.g. `8.0` for A100).

For a wheel that also runs on newer GPUs without a rebuild, build with
`TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX"` or `"8.0+PTX"`; the trailing PTX makes the
driver JIT the kernels for an unknown compute capability on first use.

## 3. Local path / co-development

For projects that need to modify the CUDA sources:

```toml
[tool.uv.sources]
gsplat = { path = "../gsplat", editable = true }
```

Editable installs do not recompile native modules on import, so after changing
CUDA code run:

```bash
python setup.py build_ext --inplace   # incremental (build/ is reused)
python setup.py develop --no-deps
```

or use the JIT flavour described in [CUDA12.md](CUDA12.md), which recompiles on
demand into `~/.cache/torch_extensions/`.

## Building a wheel

```bash
# on a node with nvcc; can also be submitted with sbatch (see script header)
bash scripts/build_wheel.sh
```

The wheel lands in `dist/`. Attach it to a GitHub Release, then reference the
asset URL (or use `uv pip install --find-links <dir-or-url> gsplat`).

## Verify any install

```bash
python - <<'PY'
import torch, gsplat
from gsplat.cuda._backend import _C
print("gsplat", gsplat.__version__, "ext loaded:", _C is not None)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

If `ext loaded` is `False`, the native extension was not built for this
environment: check that the wheel matches your torch/CUDA/Python, or rebuild
from source with `UV_NO_BUILD_ISOLATION=1`.
