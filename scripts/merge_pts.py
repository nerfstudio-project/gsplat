import os
import glob
import torch
import argparse
import open3d as o3d
from pathlib import Path
from collections import defaultdict
import numpy as np

def save_ply(splats: torch.nn.ParameterDict, dir: str, colors: torch.Tensor = None):
    # Convert all tensors to numpy arrays in one go
    print(f"Saving ply to {dir}")
    numpy_data = {k: v.detach().cpu().numpy() for k, v in splats.items()}

    means = numpy_data["means"]
    scales = numpy_data["scales"]
    quats = numpy_data["quats"]
    opacities = numpy_data["opacities"]
    ply_data = {
        "positions": o3d.core.Tensor(means, dtype=o3d.core.Dtype.Float32),
        "normals": o3d.core.Tensor(np.zeros_like(means), dtype=o3d.core.Dtype.Float32),
        "opacity": o3d.core.Tensor(
            opacities.reshape(-1, 1), dtype=o3d.core.Dtype.Float32
        ),
    }

    if colors is not None:
        color = colors.detach().cpu().numpy().copy()  #
        for j in range(color.shape[1]):
            # Needs to be converted to shs as that's what all viewers take.
            ply_data[f"f_dc_{j}"] = o3d.core.Tensor(
                (color[:, j : j + 1] - 0.5) / 0.2820947917738781,
                dtype=o3d.core.Dtype.Float32,
                )
    else:
        sh0 = numpy_data["sh0"].transpose(0, 2, 1).reshape(means.shape[0], -1).copy()
        shN = numpy_data["shN"].transpose(0, 2, 1).reshape(means.shape[0], -1).copy()

        # Add sh0 and shN data
        for i, data in enumerate([sh0, shN]):
            prefix = "f_dc" if i == 0 else "f_rest"
            for j in range(data.shape[1]):
                ply_data[f"{prefix}_{j}"] = o3d.core.Tensor(
                    data[:, j : j + 1], dtype=o3d.core.Dtype.Float32
                )

    # Add scales and quats data
    for name, data in [("scale", scales), ("rot", quats)]:
        for i in range(data.shape[1]):
            ply_data[f"{name}_{i}"] = o3d.core.Tensor(
                data[:, i : i + 1], dtype=o3d.core.Dtype.Float32
            )

    pcd = o3d.t.geometry.PointCloud(ply_data)
    success = o3d.t.io.write_point_cloud(dir, pcd)
    assert success, "Could not save ply file."

def merge_checkpoints(ckpts_folder: str, output_dir: str = None):
    """
    Load and merge checkpoint files from a folder into PLY files.
    
    Args:
        ckpts_folder: Folder containing checkpoint files
        output_dir: Output directory for PLY files. If None, uses ckpts_folder parent
    """
    ckpts_folder = Path(ckpts_folder)
    if not ckpts_folder.exists():
        raise ValueError(f"Checkpoint folder does not exist: {ckpts_folder}")

    # If no output directory specified, create a 'merged_ply' folder next to ckpts
    if output_dir is None:
        output_dir = ckpts_folder.parent / "merged_ply"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find all checkpoint files
    ckpt_files = list(ckpts_folder.glob("ckpt_*_rank*.pt"))
    if not ckpt_files:
        raise ValueError(f"No checkpoint files found in: {ckpts_folder}")

    # Group checkpoints by step number
    step_groups = defaultdict(list)
    for ckpt_file in ckpt_files:
        # Extract step number from filename (assumes format ckpt_STEP_rank*.pt)
        step = int(ckpt_file.name.split('_')[1])
        step_groups[step].append(ckpt_file)

    print(f"Found checkpoints for {len(step_groups)} steps: {sorted(step_groups.keys())}")

    # Process each step
    for step, files in sorted(step_groups.items()):
        print(f"\nProcessing step {step}...")
        print(f"Found {len(files)} rank files:")
        for f in files:
            print(f"  {f}")

        # Load all checkpoints for this step
        ckpts = [torch.load(f, map_location='cpu') for f in files]
        
        # Create a new ParameterDict to store merged parameters
        merged_splats = torch.nn.ParameterDict()
        
        # Get the keys from first checkpoint's splats
        splat_keys = ckpts[0]['splats'].keys()
        
        # Merge parameters for each key
        for key in splat_keys:
            merged_data = torch.cat([ckpt['splats'][key] for ckpt in ckpts])
            merged_splats[key] = torch.nn.Parameter(merged_data)
        
        # Save merged data as PLY
        output_ply = output_dir / f"gaussian_step_{step}.ply"
        save_ply(merged_splats, str(output_ply))
        print(f"Successfully saved merged PLY to {output_ply}")

def main():
    parser = argparse.ArgumentParser(description='Merge checkpoint files from a folder into PLY files')
    parser.add_argument('--ckpts-folder', type=str, required=True,
                      help='Folder containing checkpoint files')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for PLY files (optional)')
    
    args = parser.parse_args()
    merge_checkpoints(args.ckpts_folder, args.output_dir)

if __name__ == '__main__':
    main()
