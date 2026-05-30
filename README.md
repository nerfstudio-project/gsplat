# gsplat

[![Core Tests.](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml)
[![Docs](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml)

[http://www.gsplat.studio/](http://www.gsplat.studio/)

gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), but we’ve made gsplat even faster, more memory efficient, and with a growing list of new features! 

<div align="center">
  <video src="https://github.com/nerfstudio-project/gsplat/assets/10151885/64c2e9ca-a9a6-4c7e-8d6f-47eeacd15159" width="100%" />
</div>

## News

### Unreleased

Changes on `main` since the [v1.5.3](https://github.com/nerfstudio-project/gsplat/releases/tag/v1.5.3) tag (not yet on PyPI).

- [May 2026] **Inference Rendering (HiGS)** -- An experimental inference-only rendering path based on HiGS (Hierarchical 3D Gaussian Splatting) is now available under the `experimental` package. The inference path uses macro-tile fused rasterization with fp16 scene packing for low-latency rendering of pre-trained Gaussian scenes. See the [HiGS project page](https://research.nvidia.com/labs/sil/projects/higs/) and the "Inference Rendering" section below.
- [May 2026] Native CUDA **MCMC perturb** (`inject_noise`) speeds up the noise-injection step used in MCMC-style Gaussian optimization.
- [Apr 2026] **AccuTile** adds a conservative ellipse-based tile–Gaussian intersection test on the 3DGS path for tighter work scheduling before rasterization ([PR #927](https://github.com/nerfstudio-project/gsplat/pull/927)).
- [Apr 2026] **NCore v4** capture support, including richer camera models and point-cloud loading via `PointCloudsSourceProtocol`. See the [NCore example](https://docs.gsplat.studio/main/examples/ncore.html).
- [Mar 2026] **LiDAR** rasterization for 3D Gaussian splatting: spinning-lidar camera models, `eval3d` rendering, depth / hit-distance modes, and related tooling (optional SciPy via `pip install "gsplat[lidar]"`).
- [Mar 2026] **TorchScript-oriented** deployment: camera models and distortions are also available through PyTorch **custom operators** and **custom classes**, not only Python callables.
- [Mar 2026] **3DGUT** extensions: [external distortion](https://github.com/nerfstudio-project/gsplat/pull/886) (e.g. windshield-style rigs), optional **per-ray** inputs with gradients, optional **ray-normal** outputs, and refactored render modes / extra signals (see [3DGUT notes](docs/3dgut.md)).
- [Jan 2026] [PPISP](https://research.nvidia.com/labs/sil/projects/ppisp/) is integrated as an alternative way of bilateral grid to compensate the training views.

### v1.5.3

- [May 2025] Arbitrary batching (over multiple scenes and multiple viewpoints) is supported now!! Checkout [here](docs/batch.md) for more details! Kudos to [Junchen Liu](https://junchenliu77.github.io/).
- [May 2025] [Jonathan Stephens](https://x.com/jonstephens85) makes a great [tutorial video](https://www.youtube.com/watch?v=ACPTiP98Pf8) for Windows users on how to install gsplat and get start with 3DGUT.
- [April 2025] [NVIDIA 3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/) is now integrated in gsplat! Checkout [here](docs/3dgut.md) for more details. [[NVIDIA Tech Blog]](https://developer.nvidia.com/blog/revolutionizing-neural-reconstruction-and-rendering-in-gsplat-with-3dgut/) [[NVIDIA Sweepstakes]](https://www.nvidia.com/en-us/research/3dgut-sweepstakes/)

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

The easiest way is to install from PyPI. In this way it will build the CUDA code **on the first run** (JIT).

```bash
pip install gsplat
```

Alternatively you can install gsplat from source. In this way it will build the CUDA code during installation.

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

We also provide [pre-compiled wheels](https://docs.gsplat.studio/whl) for both linux and windows on certain python-torch-CUDA combinations (please check first which versions are supported). Note this way you would have to manually install [gsplat's dependencies](https://github.com/nerfstudio-project/gsplat/blob/6022cf45a19ee307803aaf1f19d407befad2a033/setup.py#L115). For example, to install gsplat for pytorch 2.0 and cuda 11.8 you can run
```
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt20cu118
```

To build gsplat from source on Windows, please check [this instruction](docs/INSTALL_WIN.md).

## Evaluation

This repo comes with a standalone script that reproduces the official Gaussian Splatting with exactly the same performance on PSNR, SSIM, LPIPS, and converged number of Gaussians. Powered by gsplat’s efficient CUDA implementation, the training takes up to **4x less GPU memory** with up to **15% less time** to finish than the official implementation. Full report can be found [here](https://docs.gsplat.studio/main/tests/eval.html).

```bash
cd examples
pip install -r requirements.txt
# download mipnerf_360 benchmark data
python datasets/download_dataset.py
# run batch evaluation
bash benchmarks/basic.sh
```

## Examples

We provide a set of examples to get you started! Below you can find the details about
the examples (requires installing some extrra dependencies via `pip install -r examples/requirements.txt --no-build-isolation`)

- [Train a 3D Gaussian splatting model on a COLMAP capture.](https://docs.gsplat.studio/main/examples/colmap.html)
- [Fit a 2D image with 3D Gaussians.](https://docs.gsplat.studio/main/examples/image.html)
- [Render a large scene in real-time.](https://docs.gsplat.studio/main/examples/large_scale.html)
- [Train on an NCore v4 capture.](https://docs.gsplat.studio/main/examples/ncore.html)


## Inference Rendering

gsplat includes an experimental inference-only rendering path based on HiGS (Hierarchical 3D Gaussian Splatting) in the standalone `experimental` package, designed for low-latency rendering of pre-trained Gaussian scenes where training gradients are not needed. The inference path packs scene data into compact fp16 layouts and uses a macro-tile fused rasterization pipeline for fast single-camera rendering.

```python
from experimental import render_scene, GaussianInferenceScene
```

The `simple_viewer.py` example supports the Inference path via the `--use_gaussian_render_inference_scene` flag. A standalone benchmark comparing Inference rendering against the default `rasterization()` path is available at `examples/benchmarks/gaussian_render_inference_scene/`. For more details, [see project page](https://research.nvidia.com/labs/sil/projects/higs/).

## Development and Contribution

This repository was born from the curiosity of people on the Nerfstudio team trying to understand a new rendering technique. We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software.

This project is developed by the contributors coming from following institutes (unordered):

- UC Berkeley
- NVIDIA
- ShanghaiTech University
- Amazon
- Meta
- IIIT
- LumaAI
- SpectacularAI
- Aalto University
- CMU

We also have a white paper with about the project with benchmarking and mathematical supplement with conventions and derivations, available [here](https://arxiv.org/abs/2409.06765). If you find this library useful in your projects or papers, please consider citing:

```
@article{ye2025gsplat,
  title={gsplat: An open-source library for Gaussian splatting},
  author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen, Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and Angjoo Kanazawa},
  journal={Journal of Machine Learning Research},
  volume={26},
  number={34},
  pages={1--17},
  year={2025}
}
```

We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software. Please check [docs/DEV.md](docs/DEV.md) for more info about development.
