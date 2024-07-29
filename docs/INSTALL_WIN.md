# Installing `gsplat` on Windows

Follow these steps to install `gsplat` on Windows.

## Prerequisites

1. Install Visual Studio Build Tools. If MSVC 143 does not work, you may also need to install MSVC 142 for Visual Studio 2019. And your CUDA environment should be set up properly.

2. Activate your conda environment:
    ```bash
    conda activate <your_conda_environment>
    ```
    Replace `<your_conda_environment>` with the name of your conda environment. For example:
    ```bash
    conda activate gsplat
    ```

3. Activate your Visual C++ environment:
    Navigate to the directory where `vcvars64.bat` is located. This path might vary depending on your installation. A common path is:
    ```
    C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build
    ```

4. Run the following command:
    ```bash
    ./vcvars64.bat
    ```

    If the above command does not work, try activating an older version of VC:
    ```bash
    ./vcvarsall.bat x64 -vcvars_ver=<your_VC++_compiler_toolset_version>
    ```
    Replace `<your_VC++_compiler_toolset_version>` with the version of your VC++ compiler toolset. The version number should appear in the same folder.
    
    For example:
    ```bash
    ./vcvarsall.bat x64 -vcvars_ver=14.29
    ```

## Clone the Repository

5. Clone the `gsplat` repository:
    ```bash
    git clone --recursive https://github.com/nerfstudio-project/gsplat.git
    ```

6. Change into the `gsplat` directory:
    ```bash
    cd gsplat
    ```

## Install `gsplat`

7. Install `gsplat` using pip:
    ```bash
    pip install .
    ```
