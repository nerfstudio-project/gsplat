# Features: Render Multiple Scenes In a Batch

https://github.com/user-attachments/assets/454ebbf6-daf2-44cd-b8a7-2889d0957b06

`rasterization()` function now supports arbitrary batching. For example now you can render 16 scenes, each to 6 viewpoints in one go:

```python
# 16 differene scenes, each scene has 10k Gaussians.
means: Tensor # (16, 10000, 3)
quats: Tensor # (16, 10000, 4)
scales: Tensor # (16, 10000, 3)
colors: Tensor # (16, 10000, 3)
opacities: Tensor # (16, 10000)
# Each scene render 6 viewpoints. (not shared)
viewmats: Tensor # (16, 6, 4, 4)
Ks: Tensor # (16, 6, 4, 4)
width, height = 300, 200
# Render. 
# Output `renders` is with shape [16, 6, 200, 300, 3]
# Output `alphas` is with shape [16, 6, 200, 300, 1]
renders, alphas, meta = rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, width, height
)
```

Note. The API is designed for the case where all scenes in a batch have the same number of Gaussians. If in your case the number of Gaussians is different across scenes, you can still render them in batch by padding them with zero-opacity Gaussians, or simply use a for-loop.

## Benchmark

benchmark batch feature with 10000 gaussians (`profiling/batch.py`)

| **Model**  | **N Batches** | **Mem (GB)** | **Time [fwd]** | **Time [bwd]** |
|--------|-----------|----------|------------|------------|
| 3DGS   | 1         | 0.010     | 0.00037    | 0.00049    |
| 3DGS   | 4         | 0.040     | 0.00040     | 0.00079    |
| 3DGS   | 16        | 0.161    | 0.00093    | 0.00284    |
| 3DGS   | 64        | 0.642    | 0.00368    | 0.01124    |
| 3DGUT  | 1         | 0.010     | 0.00042    | 0.00070     |
| 3DGUT  | 4         | 0.040     | 0.00057    | 0.00128    |
| 3DGUT  | 16        | 0.162    | 0.00162    | 0.00513    |
| 3DGUT  | 64        | 0.641    | 0.00635    | 0.02031    |
