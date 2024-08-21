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

Then run  `conda env create -f environment.yml` 

### 2. Activate your conda environment:
    
```bash
conda activate <your_conda_environment>
```

Replace `<your_conda_environment>` with the name of your conda environment. For example:

```bash
conda activate gsplat
```

### 3. Activate the Visual Studio C++ environment

The installation step using pip has a mechanism to automatically find the path to the visual studio compiler `cl.exe`, an thus one does not need to manually activate the visual studio environment to install `gsplat` with `pip`. However `gsplat` requires to be able to compile C++ code on the fly at runtime (just-in-time compilation) in some instances, which requires `cl.exe` to be on the path at runtime. This is the case for example when using 128  color channels. 

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

## Install `gsplat`

`gsplat` can be installed using either the source package published in `pypi.org` or using a clone of the repository

### Installation using the package pub;ished on `pypi.org`

Run `pip install gsplat`

### Installation using a clone of the repository

1. Clone the `gsplat` repository:
    ```bash
    git clone --recursive https://github.com/nerfstudio-project/gsplat.git
    ```
    MAke sure you do not forget the `--recursive` argument to get the `glm/glm.hpp` file.

2. Change into the `gsplat` directory:
    ```bash
    cd gsplat
    ```

3. Install `gsplat` using pip:
    ```bash
    pip install .
    ```
    you can install in edit mode using `pip install -e .`

## Run the tests

You will need to install the package in edit mode using `pip install -e .` for the tests to run because some of the tests assets are not packaged in the package. You will also need to activate the visual C++ environment as described above before running the tests because some of the test use just-in-time compilation to compile code that has not been compiled during the package installation. 
Some additional dependencies are required to run all the tests. They can be installed using 
```
pip install pytest nerfacc git+https://github.com/fraunhoferhhi/PLAS.git imageio torchpq cupy-cuda11x==12.3
```
You can then run the test using `pytest tests`

Note: the test ` tests/test_compression.py::test_png_compression` currently fails due to some problem in kmeans (`ValueError: Cannot take a larger sample than population when 'replace=False'`)
