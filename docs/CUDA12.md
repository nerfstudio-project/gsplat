# Building gsplat with a CUDA 12.x toolkit

The CUDA sources in this repository are written against **CCCL 3.0**, which ships
with CUDA 13. On a CUDA 12.x toolkit (CCCL 2.x) the build stops with a
`cuda::ceil_div` error. This page describes the compatibility shim and the full
build recipe. **No gsplat source file needs to be modified.**

## Symptom

```
.../gsplat/cuda/csrc/QuatScaleToCovarCUDA.cu(136): error: namespace "cuda" has no member "ceil_div"
.../gsplat/cuda/csrc/QuatScaleToCovarCUDA.cu(136): error: type name is not allowed
```

The same error appears for every translation unit that computes launch geometry:

| File | Sites |
| --- | --- |
| `gsplat/cuda/csrc/QuatScaleToCovarCUDA.cu` | 4 |
| `gsplat/cuda/csrc/SphericalHarmonics*.cu` | 15 |
| `gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu` | 4 |
| `gsplat/cuda/csrc/IntersectTile.cu` | 4 |

`::cuda::ceil_div<T>(a, b)` is only used on the host side to size kernel
launches (`blocks = ceil_div(n_elements, threads)`), so a trivial integer
implementation is sufficient.

## Fix: force-include the compatibility header

This fork ships `gsplat/cuda/include/CudaCompat.h`:

```cpp
#pragma once

#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ < 13)

#include <cuda_runtime.h>

namespace cuda {
template <typename T>
__host__ __device__ constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}
}  // namespace cuda

#endif  // __CUDACC__ && CUDA < 13
```

It is **inert on CUDA 13 and newer**, so it can be enabled unconditionally.

This fork enables it **automatically**: every native build entry point
appends `-include <path>/CudaCompat.h` to its nvcc flags whenever the header
is present. Plain `pip`/`uv` installs and the runtime JIT build therefore
work without any environment variable. `NVCC_FLAGS` is still honoured if you
need to add extra nvcc flags on top.

The entry points that apply it:

- `gsplat/cuda/build.py` — the main extension (`gsplat.csrc`), used both by
  `setup.py` and by the JIT fallback;
- `gsplat/experimental/render/kernels/cuda/build.py` — the inference renderer;
- `gsplat/geometry/kernels/cuda/build.py` and
  `gsplat/sensors/kernels/cuda/build.py` — the lazily JIT-built subpackages
  (`gsplat/scene` compiles host code only, so it needs no nvcc flags).

Because the flag is added by the build modules themselves, the lazily
JIT-built subpackages get it too, with no action on your side.

## The `glm` submodule must be initialised

`gsplat/cuda/include/{Cameras.h,Common.h,Cameras.cuh}` include `<glm/...>`. A
clone made without `--recurse-submodules` leaves
`gsplat/cuda/csrc/third_party/glm` empty and the build fails on missing glm
headers:

```bash
git submodule update --init gsplat/cuda/csrc/third_party/glm
```

`third_party/googletest` is only required for the C++ gtest suite and can stay
uninitialised.

## Build from source (editable)

```bash
export TORCH_CUDA_ARCH_LIST=8.0      # A100: 8.0, H100: 9.0, RTX 40: 8.9 ...
export MAX_JOBS=$(nproc)

python setup.py build_ext --inplace   # compiles + links gsplat/csrc.so
python setup.py develop --no-deps     # editable link, reuses the built objects
```

or, equivalently, with pip/uv:

```bash
uv pip install --no-build-isolation --no-deps -e .
```

`--no-build-isolation` is required whenever torch was installed from a specific
wheel/index (e.g. `torch==2.9.1+cu126`): build isolation would fetch a different
torch from PyPI. `--no-deps` keeps an already populated environment untouched.

### Why the `build_ext --inplace` variant is recommended

pip/uv build in a throw-away temporary directory, so an interrupted build starts
from scratch. `python setup.py build_ext --inplace` keeps the ninja objects in
`build/temp.linux-x86_64-cpython-*/`, and rerunning it recompiles only what is
missing. On clusters with short job time limits this turns a lost walltime
window into a resumed build. Keep the `build/` directory around (it is
git-ignored).

## JIT-only install (CUDA development)

If you plan to edit the CUDA sources frequently, install the Python package
without compiling and let the JIT build it on first import:

```bash
BUILD_NO_CUDA=1 python setup.py develop --no-deps
# or, with uv/pip:
BUILD_NO_CUDA=1 uv pip install --no-build-isolation --no-deps -e .
```

The JIT cache lives in `~/.cache/torch_extensions/py<minor>-cu<cuda>/`; delete it
to force a full rebuild. The first import compiles, so it must run on a machine
with `nvcc` available.

## Verify the install

```bash
python - <<PY
import torch, gsplat
from gsplat.cuda._backend import _C
print("gsplat", gsplat.__version__, "ext loaded:", _C is not None)
print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="cuda")
s = torch.ones(1, 3, device="cuda")
print(tuple(gsplat.quat_scale_to_covar_preci(q, s)[0].shape))
PY
```

Expected: `ext loaded: True`, the GPU name, and `(1, 3, 3)`.

Core kernel tests:

```bash
python -m pytest tests/test_basic.py -x -q \
    -k "quat_scale_to_covar_preci or test_projection or test_isect or rasterize_to_pixels"
```

`pytest.ini` enables `-p pytest_check` and an `env =` section, so `pytest-check`
and `pytest-env` must be installed.

## Known issues on CUDA 12.x

### The 3DGUT rasterizers compile very slowly

`GSPLAT_NUM_CHANNELS` defaults to 23 channel counts
(`1,2,3,4,5,6,8,9,16,17,21,23,24,32,33,64,65,128,129,256,257,512,513`) and the
3DGUT source files instantiate the full cross-product. Measured with CUDA 12.6 +
GCC 12.2 on an A100 node:

| Translation unit | Compile time |
| --- | --- |
| `RasterizeToPixelsFromWorld3DGSSerialBatchFwd.cu` | ~28 min |
| `RasterizeToPixelsFromWorld3DGSSerialBatchBwd.cu` | ~9 min |
| `RasterizeToPixelsFromWorld3DGSParallelBatchFwd.cu` | >36 min |
| everything else | seconds to ~3 min |

Options:

- give the build job a long time limit (the in-place build resumes);
- restrict the dispatch set, at the cost of runtime-supported channel counts
  and a full rebuild: `export NUM_CHANNELS="1,3,4,6,8,16"`;
- disable 3DGUT entirely (`export BUILD_3DGUT=0`) if you only need 3DGS/2DGS.

### `FAST_MATH` and test tolerances

`gsplat/cuda/build.py` enables `-use_fast_math` by default (`FAST_MATH=1`). This
can make strict per-element comparisons fail, e.g.
`tests/test_basic.py::test_projection` with a 1.7e-6 *relative* deviation
(`max_abs=1.1e-1` on a value of `6.4e4`). Set `FAST_MATH=0` and rebuild if you
need bit-level agreement.

### setuptools >= 82 with PyTorch < 2.9

`torch.utils.cpp_extension` imports `pkg_resources`, which setuptools 82 removed.
Use `setuptools<82` with PyTorch < 2.9 (2.9+ is fine).

### Long jobs that end at the walltime instead of exiting

`examples/simple_trainer.py` builds its train dataloader with `num_workers=4`.
After the training loop the interpreter can block joining those worker
processes, so the job looks hung after the last step even though checkpoints
and stats were already written. Wrapping the command in
`timeout -s TERM <seconds> python ...` makes the job end cleanly.

## Appendix: example HPC setup

Validated on a single A100-SXM-64GB node (`sm_80`), CUDA 12.6, GCC 12.2,
Python 3.14, with both `torch 2.9.1+cu126` and `torch 2.14.1+cu126`
(same 2000-step mip360/garden run: 23.925 dB and 23.915 dB PSNR). Those are
observations from one machine, not requirements.

```bash
# Clusters that use environment modules (skip on systems without `module`):
if command -v module >/dev/null 2>&1; then
    module load cuda/12.6
    module load gcc/12.2.0
fi
export TORCH_CUDA_ARCH_LIST=8.0      # must match your GPU (9.0 for H100, ...)
export MAX_JOBS=$(nproc)
```

A ready-to-use job is provided in `examples/build_gsplat.sh`. Submit it from the
repository root and pass your own allocation:

```bash
mkdir -p logs
sbatch --account=<acct> --partition=<gpu-part> examples/build_gsplat.sh
```

Sizing lessons that matter more than they look:

- request roughly **8-16 CPUs and 64-96 GB per GPU**. Asking for a whole node's
  CPU count and memory to use a single GPU can leave the job pending
  indefinitely, because it then only fits on a completely idle node.
- cold builds need a walltime above the slowest single translation unit: the
  3DGUT rasterizers take 30-40 min each. A long partition works; incremental
  rebuilds fit short debug windows.
- do not cold-build on shared login nodes: per-process CPU-time limits
  (commonly 10 min) kill `nvcc` mid-file.
- keep `build/` between attempts. `setup.py build_ext --inplace` resumes;
  a pip/uv build in an isolated temp dir restarts from zero.
