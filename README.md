# gsplat

[![Core Tests.](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml)
[![Docs](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml)

[http://www.gsplat.studio/](http://www.gsplat.studio/)

gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), but we’ve made gsplat even faster, more memory efficient, and with a growing list of new features! 

<div align="center">
  <video src="https://github.com/nerfstudio-project/gsplat/assets/10151885/e60f7603-3c8a-4d05-a3ae-e382507eb043" width="100%" />
</div>

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

The easiest way is to install from PyPI. In this way it will build the CUDA code **on the first run** (JIT).

```bash
pip install gsplat
```

Or install from source. In this way it will build the CUDA code during installation.

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

To install gsplat on Windows, please check [Installing `gsplat` on Windows](#installing-gsplat-on-windows)

## Evaluation

This repo comes with a standalone script that reproduces the official Gaussian Splatting with exactly the same performance on PSNR, SSIM, LPIPS, and converged number of Gaussians. Powered by gsplat’s efficient CUDA implementation, the training takes up to **4x less GPU memory** with up to **2x less time** to finish than the official implementation. Full report can be found [here](https://docs.gsplat.studio/main/tests/eval.html).

```bash
# under examples/
pip install -r requirements.txt
bash benchmark.sh
```

## Examples

We provide a set of examples to get you started! Below you can find the details about
the examples (requires to install some exta dependences via `pip install -r examples/requirements.txt`)

- [Train a 3D Gaussian splatting model on a COLMAP capture.](https://docs.gsplat.studio/main/examples/colmap.html)
- [Fit a 2D image with 3D Gaussians.](https://docs.gsplat.studio/main/examples/image.html)
- [Render a large scene in real-time.](https://docs.gsplat.studio/main/examples/large_scale.html)


## Development and Contribution

This repository was born from the curiosity of people on the Nerfstudio team trying to understand a new rendering technique. We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software.

This project is developed by the following wonderful contributors (unordered):

- [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/) (UC Berkeley): Mentor of the project.
- [Matthew Tancik](https://www.matthewtancik.com/about-me) (Luma AI): Mentor of the project.
- [Vickie Ye](https://people.eecs.berkeley.edu/~vye/) (UC Berkeley): Project lead. v0.1 lead.
- [Matias Turkulainen](https://maturk.github.io/) (Aalto University): Core developer.
- [Ruilong Li](https://www.liruilong.cn/) (UC Berkeley): Core developer. v1.0 lead.
- [Justin Kerr](https://kerrj.github.io/) (UC Berkeley): Core developer.
- [Brent Yi](https://github.com/brentyi) (UC Berkeley): Core developer.
- [Zhuoyang Pan](https://panzhy.com/) (ShanghaiTech University): Core developer.
- [Jianbo Ye](http://www.jianboye.org/) (Amazon): Core developer.

We also have made the mathematical supplement, with conventions and derivations, available [here](https://arxiv.org/abs/2312.02121). If you find this library useful in your projects or papers, please consider citing:

```
@misc{ye2023mathematical,
    title={Mathematical Supplement for the $\texttt{gsplat}$ Library}, 
    author={Vickie Ye and Angjoo Kanazawa},
    year={2023},
    eprint={2312.02121},
    archivePrefix={arXiv},
    primaryClass={cs.MS}
}
```

We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software. Please check [docs/DEV.md](docs/DEV.md) for more info about development.

## Installing `gsplat` on Windows

Follow these steps to install `gsplat` on Windows.

### Prerequisites

1. Install Visual Studio Build Tools. If MSVC 143 does not work, you may also need to install MSVC 142 for Visual Studio 2019.

2. Activate your conda environment:
    ```bash
    conda activate <your_conda_environment>
    ```
    Replace `<your_conda_environment>` with the name of your conda environment. For example:
    ```bash
    conda activate nerfstudio
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

### Clone the Repository

5. Clone the `gsplat` repository:
    ```bash
    git clone --recursive https://github.com/nerfstudio-project/gsplat.git
    ```

6. Change into the `gsplat` directory:
    ```bash
    cd gsplat
    ```

### Install `gsplat`

7. Install `gsplat` using pip:
    ```bash
    pip install .
    ```

