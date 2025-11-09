#!/usr/bin/env python3
"""
Convert viewer poses to metalgsplat format.
Reads poses from viewer_poses.json and saves each pose as a separate JSON file
with view_matrix and K_matrix in row-major flattened format.
"""

import argparse
import json
import os
from pathlib import Path
import torch
import numpy as np


def convert_poses_to_metalgsplat(viewer_poses_dir: Path, output_subdir: str = "metalgsplat_poses"):
    """
    Convert viewer poses to metalgsplat format.
    
    Args:
        viewer_poses_dir: Directory containing viewer_poses.json
        output_subdir: Subdirectory name to save converted poses
    """
    # Load viewer poses
    poses_file = viewer_poses_dir / "viewer_poses.json"
    
    if not poses_file.exists():
        raise FileNotFoundError(f"Poses file not found: {poses_file}")
    
    with open(poses_file, 'r') as f:
        poses = json.load(f)
    
    print(f"Loaded {len(poses)} poses from {poses_file}")
    
    # Create output directory
    output_dir = viewer_poses_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process each pose
    for i, pose_data in enumerate(poses):
        pose_id = pose_data['pose_id']
        
        # Extract camera-to-world matrix (c2w)
        c2w_matrix = torch.tensor(pose_data["c2w_matrix"], dtype=torch.float32)  # [4, 4]
        
        # Compute view matrix (world-to-camera) = inverse of c2w
        viewmat = torch.inverse(c2w_matrix)  # [4, 4]
        
        # Extract intrinsics matrix K
        K_matrix = torch.tensor(pose_data["K_matrix"], dtype=torch.float32)  # [3, 3]
        
        # Flatten matrices row-wise (row-major order)
        view_matrix_flat = viewmat.flatten().tolist()  # 16 values
        K_matrix_flat = K_matrix.flatten().tolist()  # 9 values
        
        # Create output JSON
        output_data = {
            "view_matrix": view_matrix_flat,
            "K_matrix": K_matrix_flat
        }
        
        # Save to file
        output_file = output_dir / f"{pose_id}.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if (i + 1) % 10 == 0 or i == len(poses) - 1:
            print(f"Processed {i+1}/{len(poses)} poses")
    
    print(f"\nConversion complete!")
    print(f"Saved {len(poses)} pose files to: {output_dir}")
    
    # Print an example for verification
    if len(poses) > 0:
        first_pose_id = poses[0]['pose_id']
        example_file = output_dir / f"{first_pose_id}.json"
        print(f"\nExample pose file: {example_file}")
        with open(example_file, 'r') as f:
            example_data = json.load(f)
        print(f"\nExample pose (pose_id={poses[0]['pose_id']}):")
        print(f"  view_matrix length: {len(example_data['view_matrix'])} values")
        print(f"  K_matrix length: {len(example_data['K_matrix'])} values")
        print(f"  First 4 view_matrix values: {example_data['view_matrix'][:4]}")
        print(f"  First 3 K_matrix values: {example_data['K_matrix'][:3]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert viewer poses to metalgsplat format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--viewer_poses_dir",
        type=Path,
        required=True,
        help="Directory containing viewer_poses.json file"
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="metalgsplat_poses",
        help="Subdirectory name for output pose files"
    )
    
    args = parser.parse_args()
    convert_poses_to_metalgsplat(args.viewer_poses_dir, args.output_subdir)


"""
Example usage:

# This will create a 'metalgsplat_poses' subdirectory with individual JSON files for each pose.
# Stores the converted poses in the 'metalgsplat_poses' subdirectory of the viewer_poses_dir.

# Convert Actor01 Sequence1 resolution_4 frame=0 viewer_poses
python scripts/model_conversion/convert_poses_to_metalgsplat.py --viewer_poses_dir results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/resolution_4/0/viewer_poses

# Convert Actor08 Sequence1 resolution_4 frame=0 viewer_poses
python scripts/model_conversion/convert_poses_to_metalgsplat.py --viewer_poses_dir results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor08/Sequence1/resolution_4/0/viewer_poses
"""

