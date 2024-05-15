# Exploration For Better 3D Gaussian Splatting

## AbsGrad

Uses absolute gradients in the image plane as the criterion for pruning. See [this paper](https://arxiv.org/pdf/2404.10484) for more details.

| Garden at 7k steps (TITAN RTX)         | T(train) | T(render) | Memory  | SSIM   | PSNR  | LPIPS | #GS.  |
| -------------------------------------- | -------- | --------- | ------- | ------ | ----- | ----- | ----- |
| default args                           | 7m07s    | 0.021s/im | 7.54 GB | 0.8332 | 26.29 | 0.123 | 4.46M |
| `--absgrad --grow_grad2d 0.0008`       | 5m50s    | 0.012s/im | 3.80 GB | 0.8365 | 26.44 | 0.121 | 2.17M |
| `--absgrad --grow_grad2d 0.0008` (30k) | --       | 0.013s/im | 4.04 GB | 0.8639 | 27.33 | 0.079 | 2.35M |

| U1 at 7k steps (RTX 2080 Ti)           | T(train) | T(render) | Memory  | SSIM   | PSNR  | LPIPS | #GS.  |
| -------------------------------------- | -------- | --------- | ------- | ------ | ----- | ----- | ----- |
| default args                           | 7m39s    | 0.013s/im | 4.94 GB | 0.6102 | 20.69 | 0.615 | 2.47M |
| default args (30k)                     | --       | 0.019s/im | --      | 0.7518 | 24.67 | 0.385 | 4.18M |
| `--absgrad --grow_grad2d 0.0008`       | 7m16s    | 0.011s/im | 3.41 GB | 0.6055 | 20.29 | 0.636 | 1.72M |
| `--absgrad --grow_grad2d 0.0008` (30k) | --       | 0.014s/im | 4.15 GB | 0.7494 | 24.65 | 0.390 | 2.37M |
| `--absgrad --grow_grad2d 0.0006`       | 8m58s    | 0.011s/im | 4.42 GB | 0.5966 | 19.58 | 0.654 | 2.21M |
| `--absgrad --grow_grad2d 0.0006` (30k) | --       | 0.016s/im | 5.09 GB | 0.7439 | 24.28 | 0.400 | 2.92M |

Note: default args means running `python simple_trainer.py` with:

- Garden: `--data_dir <DATA_DIR> --result_dir results/garden`
- U1: `--data_dir <DATA_DIR> --result_dir results/u1 --data_factor 1 --grow_scale3d 0.001`
