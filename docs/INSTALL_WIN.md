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

### Create the python environment using Conda.
Run  `conda env create -f environment.yml -n  <your_conda_environment>` with the `environment.yml` file you will find [here](../environment.yml).

### Activate your conda environment:
    
```bash
conda activate <your_conda_environment>
```

Replace `<your_conda_environment>` with the name of your conda environment. For example:

```bash
conda activate gsplat
```

## Install `gsplat`

`gsplat` can be installed using either the source package published in `pypi.org` or using a clone of the repository

### Installation using the pypi package

Run `pip install gsplat`

### Installation using the Repository

1. Clone the `gsplat` repository:
    ```bash
    git clone --recursive https://github.com/nerfstudio-project/gsplat.git
    ```

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

You will need to install the package in edit mode using `pip install -e .` for the tests to run as some of the tests assets are not packaged in the package.

Some additional dependencies are required to run all the tests. They can be installed using `pip install pytest`

Some more dependencie are required to run the compression tests. They can be installed using `pip install nerfacc git+https://github.com/fraunhoferhhi/PLAS.git imageio torchpq cupy-cuda11x==12.3`

You can then run the test using `pytest tests`

Note: the test ` tests/test_compression.py::test_png_compression` currently fails due to some problem in kmeans (`ValueError: Cannot take a larger sample than population when 'replace=False'`)

## Troubleshoot

We list here some errors that can be uncountered when following the process above with possible solutions.
|Error|Solution|
|-----|--------|
|fatal error C1083: Cannot open include file: 'glm/glm.hpp':  No such file or directory| glm is provided in the thridparty folder when using tje `--recursive` argument when clonning the repository. Alternativeley you can use `git submodule init` and `git submodule update`.
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.1.0 as it may crash. To support both 1.x and 2.x versions of NumPy, modules must be compiled with NumPy 2.0.Some module may need to rebuild instead e.g. with 'pybind11>=2.12'`| install numpy 1.26.4.
subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1| make sure the visual studio compiler `cl.exe` is in the path. 
NerfAcc: No CUDA toolkit found. NerfAcc will be disabled.| make sure `nvcc.exe` in in the path once the python environment has been activated. It should have been installed in the conda environement with the line `cuda-toolkit=11.8` 
TypeError: sparse_coo_tensor() received an invalid combination of arguments - got (indices=Tensor, values=Tensor, size=torch.Size, is_coalesced=bool, ).| `is_coalesced` has been added in pytorch 2.1