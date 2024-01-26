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
PSNR
|       | Bicycle | Bonsai | Counter | Flowers | Garden | Kitchen | Stump | Treehill | Avg  |
|-------|---------|--------|---------|---------|--------|---------|-------|----------|------|
| inria-7k   | 24.11   | 29.49  | 27.16   | 20.54   | 26.53  | 31.30   | 26.74 | 22.50    | 26.05 |
| gsplat-7k  | 22.99   | 29.45  | 26.92   | 20.33   | 25.76  | 28.48   | 24.59 | 21.91    | 25.05 |
| inria-30k  | 25.61   | 31.89  | 28.96   | 21.56   | 27.60  | 29.02   | 25.89 | 22.07    | 26.58 |
| gsplat-30k | 24.99   | 32.14  | 28.72   | 21.54   | 27.31  | 31.18   | 25.64 | 22.28    | 26.73 |

LPIPS
|           | Bicycle | Bonsai | Counter | Flowers | Garden | Kitchen | Stump  | Treehill | Avg   |
|-----------|---------|--------|---------|---------|--------|---------|--------|----------|-------|
| inria-7k  | 0.314   | 0.239  | 0.248   | 0.416   | 0.161  | 0.161   | 0.285  | 0.417    | 0.280 |
| gsplat-7k | 0.309   | 0.162  | 0.211   | 0.443   | 0.150  | 0.141   | 0.276  | 0.454    | 0.268 |
| inria-30k | 0.205   | 0.207  | 0.202   | 0.336   | 0.114  | 0.128   | 0.216  | 0.324    | 0.217 |
| gsplat-30k| 0.176   | 0.133  | 0.166   | 0.340   | 0.094  | 0.104   | 0.181  | 0.316    | 0.189 |

SSIM
|           | Bicycle | Bonsai | Counter | Flowers | Garden | Kitchen | Stump  | Treehill | Avg   |
|-----------|---------|--------|---------|---------|--------|---------|--------|----------|-------|
| inria-7k  | 0.689   | 0.915  | 0.875   | 0.530   | 0.834  | 0.901   | 0.731  | 0.589    | 0.758 |
| gsplat-7k | 0.654   | 0.921  | 0.873   | 0.527   | 0.805  | 0.893   | 0.677  | 0.583    | 0.741 |
| inria-30k | 0.777   | 0.936  | 0.905   | 0.607   | 0.865  | 0.925   | 0.773  | 0.634    | 0.803 |
| gsplat-30k| 0.750   | 0.941  | 0.918   | 0.603   | 0.851  | 0.922   | 0.730  | 0.626    | 0.793 |

Time
|           | Bicycle | Bonsai | Counter | Flowers | Garden | Kitchen | Stump  | Treehill | Avg       |
|-----------|---------|--------|---------|---------|--------|---------|--------|----------|-----------|
| inria-7k  | 3:34 | 3:27| 3:05 | 3:02 | 3:51| 4:00 | 3:25| 2:53  | 03:24|
| gsplat-7k | 2:36 | 2:18| 2:06 | 2:14 | 2:49| 2:17 | 2:24| 2:23  | 02:23|
| inria-30k | 25:07| 14:37| 15:24| 17:32| 24:03| 19:02| 20:25| 18:02 | 19:16|
| gsplat-30k| 18:03| 10:13| 9:07 | 13:25| 14:58| 10:02| 15:15| 16:56 | 13:29|


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
