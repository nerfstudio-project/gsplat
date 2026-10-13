# Windows CUDA 13 source build

This guide describes a tested source build for Windows 10/11, Python 3.12, CUDA 13.0, PyTorch 2.12, and NVIDIA Blackwell GPUs such as the RTX 5090 (`sm_120`). It produces a CPython 3.12 Windows wheel containing the core `gsplat.csrc` extension.

The tested matrix is:

| Component | Version |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.12.1 + cu130 |
| torchvision | 0.27.1 + cu130 |
| CUDA Toolkit | 13.0 |
| Visual Studio | 2022 Build Tools, MSVC 14.44 or newer |
| GPU architecture | `12.0+PTX` (`sm_120`) |

## Prerequisites

Install the following before starting:

- Conda or Miniconda
- Git with submodule support
- Visual Studio 2022 Build Tools with the C++ workload and Windows SDK
- CUDA Toolkit 13.0, including `nvcc` and the CUDA development headers
- An NVIDIA driver that supports the target GPU and CUDA 13 runtime

The build is CPU- and memory-intensive. GPU utilization is normally low because `nvcc` and its CUDA frontend compile host/device code on the CPU. Do not stop `NVDisplay.Container.exe` or `nvcontainer.exe` to resolve a compiler problem; those are NVIDIA driver services, not CUDA build workers.

## Create the environment

Use the official CUDA 13.0 PyTorch index and pin the matching PyTorch pair:

```powershell
conda create -n gsplat_build_env python=3.12 -y
conda activate gsplat_build_env
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install ninja wheel setuptools cmake numpy jaxtyping nvtx "rich>=12"
```

Verify the runtime before compiling:

```powershell
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list())"
```

For an RTX 5090, the output should report CUDA availability, capability `(12, 0)`, and `sm_120` in the architecture list.

## Activate one build shell

Run the Conda activation and Visual Studio setup in the same `cmd.exe` process. Calling `conda.bat` directly avoids accidentally leaving a child shell in `base` while the command prompt appears to have activated the requested environment.

Adjust `CONDA_ROOT` and the Visual Studio path for the local installation:

```bat
@echo off
set "CONDA_ROOT=%USERPROFILE%\miniconda3"
if not exist "%CONDA_ROOT%\condabin\conda.bat" set "CONDA_ROOT=%USERPROFILE%\anaconda3"
call "%CONDA_ROOT%\condabin\conda.bat" activate gsplat_build_env
if errorlevel 1 exit /b 1

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "TORCH_CUDA_ARCH_LIST=12.0+PTX"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "MAX_JOBS=1"
set "BUILD_EXPERIMENTAL=0"
set "PATH=%CONDA_PREFIX%\Scripts;%CONDA_PREFIX%;%CUDA_HOME%\bin;%PATH%"

set CONDA_DEFAULT_ENV
set CONDA_PREFIX
where python
where ninja
where nvcc
where cl
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_capability(0))"
```

`MAX_JOBS=1` is conservative and avoids running several very large CUDA translation units at once. Increase it only when the host has enough RAM and the compiler remains stable.

## Clone and build

From the activated build shell:

```bat
git clone --recursive https://github.com/nerfstudio-project/gsplat.git
cd gsplat
python setup.py bdist_wheel --dist-dir=.\dist
```

The resulting file should have a tag similar to:

```text
dist\gsplat-<version>-cp312-cp312-win_amd64.whl
```

The validated configuration uses `BUILD_EXPERIMENTAL=0`, which builds the core `gsplat.csrc` extension. The separate experimental inference extension is not included in this tested CUDA 13 path.

## Validate the wheel

Inspect and install the wheel without allowing pip to replace the pinned dependencies:

```bat
python -m pip install --force-reinstall --no-deps .\dist\gsplat-*-cp312-cp312-win_amd64.whl
```

Run the import check from outside the repository so the checkout cannot mask the installed package:

```bat
cd /d %TEMP%
gsplat-wheel-smoke
python -c "import sys, torch, gsplat, gsplat.csrc; print(sys.executable); print(gsplat.__file__); print(gsplat.csrc.__file__); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

The native extension should load from the environment's `site-packages` directory, and the GPU should report capability `(12, 0)`.

## CUDA 13 portability notes

The CUDA 13 Windows installer places CCCL headers under `CUDA\include\cccl`. The build helper must add that directory to the extension include paths; otherwise headers such as `cuda/std/limits` are not found.

The current Windows/CUDA 13 source path also needs these portability details:

- Use CUDA's global device `::isfinite` in device code instead of the MSVC host-only `std::isfinite` overload.
- Avoid GCC-only `__builtin_clzll` in host helpers compiled through MSVC.
- Keep declarations and definitions identical when a `bool` parameter is involved; MSVC name decoration distinguishes `bool` from `const bool` in this build path.
- PyTorch 2.12 cu130 does not provide every private CUB radix-sort instantiation used by older extension code. Prefer a public PyTorch sorting dispatcher for the batch-slot permutation when the keys are unique.
- Use fixed-width integer types such as `uint32_t` instead of relying on an unqualified `uint` alias in CUDA headers.

## Troubleshooting process state

A command such as `Wait-Process -Id <pid>` only observes another process. Its prompt returning does not prove that compiler descendants have finished. Before restarting a build, verify that no workspace-specific `python`, `ninja`, `nvcc`, `cicc`, or `cl` processes remain and that the Ninja files are not locked.

A useful completion checklist is:

1. The build log ends without `FAILED`, `LNK`, `Traceback`, or `RuntimeError` diagnostics.
2. The expected wheel exists in `dist`.
3. The wheel contains `gsplat/csrc.pyd`.
4. The wheel imports successfully from outside the source tree.
5. Torch reports CUDA availability and the expected GPU capability.

The wheel remains dependent on a compatible PyTorch installation, CUDA runtime/driver, and Python ABI. It is not a self-contained CUDA runtime distribution.
