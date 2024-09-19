# Installing `gsplat` on Windows

Follow these steps to install `gsplat` on Windows.

## Prerequisites

###  Visual Studio Build Tools

Install Visual Studio Build Tools. If MSVC 143 does not work, you may also need to install MSVC 142 for Visual Studio 2019. 

###  CUDA Toolkit

We recommend installing the CUDA dependencies as part of the Conda python environment creation. This has the advantage to install the CUDA dependencies automatically and in isolation from other python environment on the system, making the process less error prone.

Alternatively, you can install CUDA Toolkit 11.8 using the installer from [here](https://developer.nvidia.com/cuda-11-8-0-download-archive). You will then need to setup the PATH and CUDA_PATH variables accordingly. 

## Python environment setup

We suggest using Conda to create the Python environment as it enables you to install the CUDA dependencies in isolation.

### 1. Create the python environment using Conda.

Create conda `environment.yml` files containing
```
name: <your_conda_environment>
channels:
  - pytorch
  - defaults
  - nvidia/label/cuda-11.8.0
  - conda-forge
dependencies:
  - python=3.10
  - cuda-version==11.8 
  - cudnn==8.9.7.29
  - cuda-toolkit=11.8
  - pytorch=2.1.0
  - pip:
    - numpy==1.26.4
variables:
   CUDA_PATH: ""
``` 

Then run  `conda env create -f environment.yml` to create the conda environment.
Note: `pytorch=2.0.0` also works, but some of the tests require `pytorch=2.1.0` to run.
 
### 2. Activate your conda environment:
    
Activate your environeent using:
```bash
conda activate <your_conda_environment>
```

Replace `<your_conda_environment>` with the name of your conda environment. For example:

```bash
conda activate gsplat
```

## 3. Install `gsplat`

`gsplat` can be installed using either the source package published in `pypi.org`, the wheel published on `pypi.org` or using a clone of the repository.

### Installation using the source package published on `pypi.org`

We recommand install gsplat from the published source package and not the wheel by using
```
pip install --no-binary=gsplat gsplat --no-cache-dir
```
The CUDA code will be compiled during the installation and the cvisual stdio compiler `cl.exe` does not need to be added to the path, because the installation process as an automatic way to find it.
We use `--no-cache-dir` to avoid using a potential wheel from `pypi.org` dowloaded previously that does not have the binaries.

### Installation using the wheel published on `pypi.org`

You can install the `gsplat` using using the wheel published on `pypi.org` by using 
```
pip install gsplat
```
The wheel that does not contain the compiled CUDA binaries. The CUDA code is not compiled during the installation when using wheels, and will be compiled at the first import of `gsplat` wich requires `cl.exe` to be on the path (see section bellow documenting how to do this). 

### Installation using a clone of the repository

1. Clone the `gsplat` repository:
    ```bash
    git clone --recursive https://github.com/nerfstudio-project/gsplat.git
    ```
    Make sure you do not forget the `--recursive` argument to get the `glm/glm.hpp` file.

2. Change into the `gsplat` directory:
    ```bash
    cd gsplat
    ```

3. Install `gsplat` using pip:
    ```bash
    pip install .
    ```

Note: If you an to run the tests or modify the code you will to install the package in edit mode instead using `pip install -e .`

## Run the tests

You will need to clone the gsplat repository locally and install the gsplat package in edit mode using `pip install -e .` for the tests to run because the tests assets are not packaged in the published package. 

Some additional dependencies are required to run all the tests. They can be installed using 
```
pip install --no-binary=nerfacc nerfacc --no-cache-dir
pip install pytest nerfacc git+https://github.com/fraunhoferhhi/PLAS.git imageio torchpq cupy-cuda11x==12.3
```
You can then run the tests using `pytest tests`

Notes
* We use `--no-binary=nerfacc` so that the `nerfacc` CUDA code gets compiled during the installation step. Without this argument the nerfacc CUDA code is compiled on the fly during the first import, which requires the visual studio folder executable `cl.exe` to be on the path (see how to do this bellow).
* the test ` tests/test_compression.py::test_png_compression` currently fails due to some problem in kmeans (`ValueError: Cannot take a larger sample than population when 'replace=False'`)

### Activate the Visual Studio C++ environment

The installation step from source using `pip install --no-binary=gsplat gsplat` has a mechanism to automatically find the path to the visual studio compiler `cl.exe`, an thus one should not need to manually activate the visual studio environment to add `cl.exe` to the path. However, in case the dependency `nerfacc` used in the tests has been installed using the published wheel instead of the source package or if the installation of `gsplat` has been done from the published wheel using `pip install gsplat`, then you may still need to compile some C++ and CUDA code on the fly at runtime (just-in-time compilation) which requires `cl.exe` to be on the path at runtime.

In order to have `cl.exe` on the path you can:
1. Navigate to the directory where vcvars64.bat is located. This path might vary depending on your installation. A common path is:
    ```
    C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build
    ```

2. Run the following command:
    ```
    ./vcvars64.bat
    ```
    If the above command does not work, try activating an older version of VC:
    ```bash
    ./vcvarsall.bat x64 -vcvars_ver=<your_VC++_compiler_toolset_version>
    ```
    Replace `<your_VC++_compiler_toolset_version>` with the version of your VC++ compiler toolset. The version number should appear in the same folder. For example `./vcvarsall.bat x64 -vcvars_ver=14.29`


3. Check that `cl.exe` is in the path by using `where cl.exe`.
