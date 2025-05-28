# Features: 3DGUT

We now have integrated [NVIDIA 3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/) into gsplat, which extend 3D Gaussian Splatting (3DGS) to support nonlinear camera projections such as distortions in pinhole or fisheye cameras, and [rolling shutter](https://en.wikipedia.org/wiki/Rolling_shutter) effects. This allows user to directly train 3DGS on captured images without the need of undistort them beforehand (though camera calibration -- e.g., using COLMAP -- is still required to get distortion parameters).

https://github.com/user-attachments/assets/291481ec-9546-4d50-a737-19422dbadffd

## How to Use

Here are the instructions on how to use this feature.

### For users directly running `examples` in gsplat:

#### Training

Simplly passing in `--with_ut --with_eval3d` to the `simple_trainer.py` arg list will enable training with 3DGUT! And note in gsplat we only support MCMC densification strategy for 3DGUT:

```
python examples/simple_trainer.py mcmc --with_ut --with_eval3d ... <OTHER ARGS>
```

For benchmarking on MipNeRF360 Dataset, please checkout `examples/benchmarks/3dgut/mcmc.sh`

Note if you are not familiar with how to get started with `simple_trainer.py`, please checkout [README.md](README.md) first!

#### Rendering

Once trained, you could view the 3DGS and play with the distortion effect supported through 3DGUT via our viewer:

```bash
CUDA_VISIBLE_DEVICES=0 python simple_viewer_3dgut.py --ckpt results/benchmark_mcmc_1M_3dgut/garden/ckpt_29999_rank0.pt 
```

Or a more comprehensive nerfstudio-style viewer to export videos. (note changing distortion is not yet supported in this comprehensive viewer!)
```bash
CUDA_VISIBLE_DEVICES=0 python simple_viewer.py --with_ut --with_eval3d --ckpt results/benchmark_mcmc_1M_3dgut/garden/ckpt_29999_rank0.pt 
```

### For users using gsplat' API:
To use the 3DGUT technique The relavant arguments in `rasterization()` function are:
- Setting `with_ut=True` and `with_eval3d=True` to enable 3DGUT (which is consist of two parts: using unscented transform to estimate the camera projection and evaluate Gaussian response in 3D space.)
- To train/render pinhole camera with distortion, setting the distortion parameters to `radial_coeffs`, `tangential_coeffs`, `thin_prism_coeffs`.
- To train/render fisheye camera with distortion, 
setting the distortion parameters to `radial_coeffs` and set `camera_model="fisheye"`
- To enable rolling shutter effects, checks out `rolling_shutter` and `viewmats_rs` on the type of rolling shutters we supported.
