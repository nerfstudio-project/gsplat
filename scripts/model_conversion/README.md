# Model Conversion Scripts

This directory contains scripts to convert trained Gaussian Splat models to different formats.

## convert_actorshq_to_metalgsplat.py

Converts gsplat-trained checkpoints to metalgsplat binary format for mobile iOS rendering.

### Usage

```bash
python scripts/model_conversion/convert_actorshq_to_metalgsplat.py \
    --checkpoint_path <path_to_checkpoint.pt> \
    --output_dir <output_directory>
```

### Example

```bash
# Convert Actor01 model
python scripts/model_conversion/convert_actorshq_to_metalgsplat.py \
    --checkpoint_path results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/resolution_4/0/ckpts/ckpt_29999_rank0.pt \
    --output_dir results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/resolution_4/0/metalgsplat
```

### Output Files

The script generates the following binary files (all in float32 format):

- **means3d.bin**: 3D positions of Gaussians [N, 3]
- **scales3d.bin**: 3D scales of Gaussians [N, 3]
- **quats.bin**: Quaternion rotations [N, 4] (normalized)
- **colors.bin**: RGB colors [N, 3] (converted from SH DC component, range [0, 1])
- **opacities.bin**: Opacity values [N] (range [0, 1])

### Parameter Transformations

The script applies the following transformations to match the expected metalgsplat format:

1. **Means**: Used as-is (no transformation)
2. **Scales**: Exponential applied (stored as log-space in checkpoint)
3. **Quaternions**: Normalized to unit quaternions
4. **Opacities**: Sigmoid applied (stored as logits in checkpoint)
5. **Colors**: Converted from Spherical Harmonics (SH) DC component to RGB using `rgb = sh * C0 + 0.5` where `C0 = 0.28209479177387814`, then clipped to [0, 1]

### Notes

- The script only uses the DC component (sh0) of the spherical harmonics for color conversion
- Higher-order SH coefficients (shN) are discarded as they are not needed for mobile rendering
- No appearance embedding is included (not present in the training)
- All binary files are saved in float32 format for compatibility with the mobile rasterizer

## convert_splatfacto_to_metalgsplat.py

Converts nerfstudio splatfacto models to metalgsplat format. See the script for usage details.

