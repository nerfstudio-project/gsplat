# Exploration For Better 3D Gaussian Splatting

## AbsGrad

Uses absolute gradients in the image plane as the criterion for pruning. See [this paper](https://arxiv.org/pdf/2404.10484) for more details.

| Impl.                            | T(train) | T(render) | Memory  | SSIM   | PSNR  | LPIPS | #GS.  |
| -------------------------------- | -------- | --------- | ------- | ------ | ----- | ----- | ----- |
| default args                     | 7m07s    | 0.021s/im | 7.54 GB | 0.8332 | 26.29 | 0.123 | 4.46M |
| `--absgrad --grow_grad2d 0.0008` | 5m50s    | 0.012s/im | 3.80 GB | 0.8365 | 26.44 | 0.121 | 2.17M |

Test environment: Garden scene from MipNeRF360 at 7k steps on a NVIDIA TITAN RTX.
