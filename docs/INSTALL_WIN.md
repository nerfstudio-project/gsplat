# Installing `gsplat` on Windows

Follow these steps to install `gsplat` on Windows.

## Prerequisites

1. Install Visual Studio Build Tools. If MSVC 143 does not work, you may also need to install MSVC 142 for Visual Studio 2019. 
2. Install CUDA Toolkit 11.8 and setup the CUDA_PATH Variable. The toolkit installer can be downloaded from. 
We recommand to skip this step and instead to use conda to install the CUDA dependencies in isolation in the python environment (see bellow).   


## Create the python environment

We recommand create the python environement using conda because it allows to install the CUDA dependencies in isolation.

Run  `conda env create -f environment.yml` with the following `environment.yml` file

```
name: gsplat
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

2. Activate your conda environment:
    ```bash
    conda activate <your_conda_environment>
    ```
    Replace `<your_conda_environment>` with the name of your conda environment. For example:
    ```bash
    conda activate gsplat
    ```


## Install `gsplat`

gsplat can be installed using the source package in pypi.org or using a clone of the repository

### Installation using the pypi package

`pip install gsplat`

### Installation using the Repository

5. Clone the `gsplat` repository:
    ```bash
    git clone --recursive https://github.com/nerfstudio-project/gsplat.git
    ```

6. Change into the `gsplat` directory:
    ```bash
    cd gsplat
    ```


7. Install `gsplat` using pip:
    ```bash
    pip install .
    ```
    you can install in edit mode using `pip install -e .`

## Run the tests

Some additional dependencies are require to run all the tests. They can be installed using

```
pip install pytest numpy==1.26.4 
```

Some more dependencie are required to run the compression tests:
```
pip install git+https://github.com/fraunhoferhhi/PLAS.git imageio torchpq cupy-cuda11x==12.3
```

## Troubleshoot

Error:
```fatal error C1083: Cannot open include file: 'glm/glm.hpp':  No such file or directory```

Solutions:
glm is provided in the thridparty folder when using tje `--recursive` argument when clonning the repository.
Alternativeley you can use `git submodule init` and `git submodule update`.


Error:
```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.1.0 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
```
Solution:

install numpy 1.26.4.

```subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1```
make sure cl.exd is in the path. Note the cl should be automatically found after PR ?


```NerfAcc: No CUDA toolkit found. NerfAcc will be disabled.```
make sure `nvcc.exe` in in the path once the python environment has been activated. It shoudl have been installed in the conda environement with the line `cuda-toolkit=11.8` 


```TypeError: sparse_coo_tensor() received an invalid combination of arguments - got (indices=Tensor, values=Tensor, size=torch.Size, is_coalesced=bool, )```.
 `is_coalesced` has been added in pytorch 2.1