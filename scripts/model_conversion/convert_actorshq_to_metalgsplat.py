import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import numpy as np


def sh2rgb(sh: torch.Tensor) -> torch.Tensor:
    """Convert Spherical Harmonics DC component to RGB
    
    This matches the behavior of gsplat's spherical_harmonics() function
    which multiplies SH coefficients by C0 and adds 0.5. The result is NOT
    clamped to [0, 1] - it can be [0, ∞) and should be clamped after blending.
    
    Args:
        sh (torch.Tensor): SH tensor (DC component only)
    
    Returns:
        torch.Tensor: RGB tensor in [0, ∞) range (unbounded upper limit)
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def main(ckpt_path: Path, output_dir: Path):
    """
    Convert a gsplat checkpoint to metalgsplat binary format.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        output_dir: Directory to save the binary files
    """
    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Extract splats dictionary
    splats = ckpt["splats"]
    step = ckpt.get("step", "unknown")
    print(f"Checkpoint step: {step}")
    print(f"Number of Gaussians: {splats['means'].shape[0]}")
    
    # Extract and convert parameters
    # 1. Means: stored as-is, no transformation needed
    means3d = splats["means"].cpu().detach().numpy()  # [N, 3]
    print(f"Means shape: {means3d.shape}")
    
    # 2. Scales: stored as log(scale), need to apply exp
    scales3d = torch.exp(splats["scales"]).cpu().detach().numpy()  # [N, 3]
    print(f"Scales shape: {scales3d.shape}")
    
    # 3. Quats: need to normalize
    quats = splats["quats"]
    quats = F.normalize(quats, dim=-1).cpu().detach().numpy()  # [N, 4]
    print(f"Quats shape: {quats.shape}")
    
    # 4. Opacities: stored as logit, need to apply sigmoid
    opacities = torch.sigmoid(splats["opacities"]).cpu().detach().numpy()  # [N,]
    print(f"Opacities shape: {opacities.shape}")
    
    # 5. Colors: stored as SH coefficients (sh0 is DC component, shN are higher order)
    # For mobile rendering, we only use the DC component (sh0) and convert to RGB
    sh0 = splats["sh0"]  # [N, 1, 3]

    # The  gsplat rasterization calls sh_coeffs_to_color_fast() in the forward pass, which is defined in gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu
    # and it converts the SH coefficients to RGB. Then, it adds 0.5 to the RGB values to make them in the range [0, ∞) (torch.clamp_min in rendering.py line 518).
    '''
    colors = spherical_harmonics(
                sh_degree, dirs, shs, masks=masks
            )  # [..., C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
    colors = torch.clamp_min(colors + 0.5, 0.0)
    '''
    
    # We need to convert the SH coefficients to RGB here.
    colors = sh2rgb(sh0).squeeze(1)  # [N, 3]
    colors = torch.clamp_min(colors, 0.0)  # Only lower bound, no upper bound - exact same as gsplat!
    colors = colors.cpu().detach().numpy()  # Convert to numpy for saving
    print(f"Colors shape: {colors.shape}")
    print(f"Colors range: min={colors.min():.4f}, max={colors.max():.4f}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving binary files to {output_dir}")
    
    # Save as binary files (float32 format)
    means3d.astype("float32").tofile(output_dir / "means3d.bin")
    scales3d.astype("float32").tofile(output_dir / "scales3d.bin")
    quats.astype("float32").tofile(output_dir / "quats.bin")
    colors.astype("float32").tofile(output_dir / "colors.bin")
    opacities.astype("float32").tofile(output_dir / "opacities.bin")
    
    print("\nConversion complete!")
    print(f"Created files:")
    print(f"  - means3d.bin: {means3d.nbytes / (1024**2):.2f} MB")
    print(f"  - scales3d.bin: {scales3d.nbytes / (1024**2):.2f} MB")
    print(f"  - quats.bin: {quats.nbytes / (1024**2):.2f} MB")
    print(f"  - colors.bin: {colors.nbytes / (1024**2):.2f} MB")
    print(f"  - opacities.bin: {opacities.nbytes / (1024**2):.2f} MB")
    print(f"\nTotal size: {(means3d.nbytes + scales3d.nbytes + quats.nbytes + colors.nbytes + opacities.nbytes) / (1024**2):.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert gsplat checkpoint to metalgsplat binary format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--ckpt_path", 
        type=Path, 
        required=True,
        help="Path to the gsplat checkpoint file (.pt)"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        required=True,
        help="Directory to save the binary files"
    )
    
    args = parser.parse_args()
    main(ckpt_path=args.ckpt_path, output_dir=args.output_dir)

'''
# Convert Actor01 Sequence1 resolution_4 frame=0 checkpoint
python scripts/model_conversion/convert_actorshq_to_metalgsplat.py --ckpt_path results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/resolution_4/0/ckpts/ckpt_29999_rank0.pt --output_dir results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/resolution_4/0/metalgsplat

# Convert Actor08 Sequence1 resolution_4 frame=0 checkpoint
python scripts/model_conversion/convert_actorshq_to_metalgsplat.py --ckpt_path results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor08/Sequence1/resolution_4/0/ckpts/ckpt_29999_rank0.pt --output_dir results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor08/Sequence1/resolution_4/0/metalgsplat
'''