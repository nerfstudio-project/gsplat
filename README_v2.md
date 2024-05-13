# gsplat

WIP: Some of the contents in this markdown should be moved into the webpage.

## Installation

```bash
git clone <URL> --recursive

mamba create --name gsplat -y python=3.10
mamba activate gsplat
mamba install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install pytest ninja

# BUILD_NO_CUDA=1 postpones the compiling to the first run with JIT
BUILD_NO_CUDA=1 pip install -e .

# run tests to trigger JIT compiling and check everything is good.
# currently the test requires nerfacc to be installed:
# pip install git+https://github.com/nerfstudio-project/nerfacc
pytest tests_v2
```

## Examples

### Render 3DGS with a Web-based Viewer (powered by [`viser`](https://github.com/nerfstudio-project/viser) and [`nerfview`](https://github.com/hangg7/nerfview))

This script will render a SFM points initialized, unoptimized scene (`./assets/test_garden.npz`), exported from the Garden scene in the [MipNeRF-360 dataset](https://jonbarron.info/mipnerf360/).

```bash
# install viewer dependency
cd examples_v2
pip install git+https://github.com/hangg7/nerfview

python simple_viewer.py --port 8080
```

### Single Script 3DGS Training.

This script reproduces the [official implementation](https://github.com/graphdeco-inria/gaussian-splatting/) with less training time, less memory footprint and better performance. It also supports a web-based viewer.

| Impl.                   | Training Time | Memory  | SSIM   | PSNR  | LPIPS |
| ----------------------- | ------------- | ------- | ------ | ----- | ----- |
| official implementation | 482s          | 9.11 GB | 0.8237 | 26.11 | 0.129 |
| this repo               | 455s          | 7.52 GB | 0.8330 | 26.28 | 0.123 |

Note: Tested on a 16GB V100-SXM2. LPIPS metric is a different version with the official implementation.

```bash
# install training dependency
cd examples_v2
pip install -r requirements.txt

python simple_trainer.py --data_dir data/360_v2/garden --port 8080 --max_steps 7000
```

### Plug-and-Play in the Official Implementation

If you are developing based on the [official implementation](https://github.com/graphdeco-inria/gaussian-splatting), we provide [a fork version](https://github.com/graphdeco-inria/gaussian-splatting) of it, in which we replace the rasterization backend from `diff-gaussian-rasterization` to `gsplat` with
minimal changes (<100 lines, [commit](https://github.com/liruilong940607/gaussian-splatting/commit/6a50be0fbb7cae3f100cb386c7591ac48f2c288d)), and get some improvements for free:

For example we showcase a 20% training speedup and a noticeable memory reduction, with slightly better performance on the Garden scene, benchmarked on a 16GB V100-SXM2 at 7k steps.

| Backend                       | Training Time | Memory  | SSIM   | PSNR  | LPIPS |
| ----------------------------- | ------------- | ------- | ------ | ----- | ----- |
| `diff-gaussian-rasterization` | 482s          | 9.11 GB | 0.8237 | 26.11 | 0.166 |
| `gsplat`                      | 402s          | 8.78 GB | 0.8366 | 26.18 | 0.163 |

### More Features: Batch Rendering

Our `rasterization` function allows for processing a batch of N cameras in a single pass,
and produce based images in the shape if `[N, H, W, 3]`. See `examples_v2/simple_viewer.py` for an example.

Benchmarks: TODO

### More Features: Extremely Fast Large-scale Scene Rendering

`gsplat` is designed to support large-scale scene. In the example below, we duplicate the Garden scene into a 15x15 grid with 25M Gaussians, and view it in the viewer. The trick behind it is to allow skipping on extremely small GSs (via argument `raidus_clip` in the `rasterization` function) after projected to the image plane, which is crucial for large-scale scenes.

```bash
cd examples_v2
python simple_viewer.py --port 8080 --scene_grid 15
```

Video Comparision: TODO

### More Features: Fast N-D Color Rasterization.

`gsplat` supports N-D color rasterization since a while ago but in `v1.0` we largely improves the speed and memory footprint of it. See the benchmark below.

Benchmarks: TODO

### More Features: Sparse Gradients.

Usually during every iteration of the training, only a portion of the GSs have gradients.
So applying the `torch.optim.Optimizer` to the entire GSs would be wasteful on both compute and memory. The waste is more and more severe as the scene becomes larger and larger. Fortunately, PyTorch has supports to optimize tensors with sparse gradients, using for example `torch.optim.SparseAdam`. However this requires the backward could produce gradients in the sparse tensor layout.

To this end, the `rasterization` function in `gsplat` provides an argument `sparse_grad` to allow the gradients living in a sparse tensor layout to work with optimizers like `torch.optim.SparseAdam`. This can potentially greatly help with training on large-scale scenes.

We provide a preview of this feature by adding `--sparse_grad` argument to the `examples_v2/simple_trainer.py`. However this feature is not yet well optimized and still in active development.

### More Features: Trade-off Between Memory and Runtime.

Our `rasterization` functions supports trading a memory friendly backend via `packed=True` argument. It can greatly save memory usage on both forward and backward pass (e.g., 70% less) on a large scene. The larger the scene is, the more significant memory saving you can get from `packed=True`, with the prince of slightly slower speed (~5%). See `profiling_v2/main.py` for a full comparision.

Benchmark table: TODO
