# gsplat

[![Core Tests.](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml)
[![Docs](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml)

[http://www.gsplat.studio/](http://www.gsplat.studio/)

gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). This library contains the neccessary components for efficient 3D to 2D projection, sorting, and alpha compositing of gaussians and their associated backward passes for inverse rendering.

This project was greatly inspired by original paper [3D Gaussian Splatting
for Real-Time Radiance Field Rendering
](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) by Kerbl* and Kopanas* et al. While building this library, we prioritized having a developer friendly Python API. 

![Teaser](/docs/source/imgs/training.gif?raw=true)

## Evaluation
A full evalaution of Nerfstudio's implementation of Gaussian Splatting against the original Inria method can be found [here](https://docs.gsplat.studio/tests/eval.html).


## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

The easist way is to install from PyPI. In this way it will build the CUDA code **on the first run** (JIT).

```bash
pip install gsplat
```

Or install from source. In this way it will build the CUDA code during installation.

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

## Examples

Fit a 2D image with 3D Gaussians.

```bash
pip install -r examples/requirements.txt
python examples/simple_trainer.py
```

## Development and Contribution

This repository was born from the curiosity of people on the Nerfstudio team trying to understand a new rendering technique. This effort was led by Vickie Ye, who wrote the CUDA backend library, and Matias Turkulainen, who wrote the python bindings, library, and documentation. Thank you to Zhuoyang Pan for extensive testing and help on the Python bindings, Ruilong Li for packaging and deployment, and Matt Tancik and Justin Kerr for inspiring Vickie to do this. This library was developed under the guidance of Angjoo Kanazawa at Berkeley. If you find this library useful in your projects or papers, please consider citing this repository:
```
@software{Ye_gsplat,
author = {Ye, Vickie and Turkulainen, Matias, and the Nerfstudio team},
title = {{gsplat}},
url = {https://github.com/nerfstudio-project/gsplat}
}
```

We also have made the mathematical supplement, with conventions and derivations, available [here](https://arxiv.org/abs/2312.02121). If you find the supplement useful, please consider citing:
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
