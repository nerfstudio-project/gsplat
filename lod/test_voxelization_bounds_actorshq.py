#!/usr/bin/env python3
"""
Example script showing how to use camera bounds for voxelization of ActorsHQ dataset.
This script demonstrates the integration of camera bounds calculation with the voxelization process.
"""

import os
import sys
import torch
import numpy as np
from typing import Tuple

# Add the parent directory to the path to import from examples
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from camera_bounds_from_colmap import get_actorshq_bounds, CameraBounds
from voxelization.voxelization import Voxelization
from voxelization.voxel_methods import AggregationMethod


def demo_camera_bounds_voxelization():
    """
    Demonstrate how to use camera bounds for voxelization.
    """
    print("=== ActorsHQ Dataset Voxelization with Camera Bounds ===\n")
    
    # Method 1: Use pre-calculated bounds
    print("1. Using pre-calculated bounds:")
    try:
        bounds = get_actorshq_bounds("viewing_frustum_bounds")
        print(f"   Loaded bounds: {bounds}")
        print(f"   Scene size: {bounds[1] - bounds[0]:.2f} x {bounds[3] - bounds[2]:.2f} x {bounds[5] - bounds[4]:.2f} meters")
        
        # Calculate appropriate voxel sizes for different levels of detail
        scene_size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        print(f"   Max scene dimension: {scene_size:.2f} meters")
        
        # Suggest voxel sizes based on scene size
        voxel_sizes = {
            "coarse": scene_size / 32,    # 32x32x32 grid
            "medium": scene_size / 64,    # 64x64x64 grid
            "fine": scene_size / 128,     # 128x128x128 grid
            "ultra": scene_size / 256     # 256x256x256 grid
        }
        
        print("   Recommended voxel sizes:")
        for level, size in voxel_sizes.items():
            print(f"     {level}: {size:.4f} meters")
        
    except FileNotFoundError as e:
        print(f"   Error: {e}")
        print("   Please run 'python lod/camera_bounds.py' first to calculate bounds.")
        return
    
    # Method 2: Calculate bounds on-the-fly
    print("\n2. Calculating bounds on-the-fly:")
    data_dir = "data/Actor01/Sequence1/0/resolution_4"
    
    if os.path.exists(data_dir):
        bounds_calculator = CameraBounds(data_dir, factor=1, normalize=False)
        dynamic_bounds = bounds_calculator.get_viewing_frustum_bounds(max_depth=5.0, padding=0.5)
        print(f"   Dynamic bounds: {dynamic_bounds}")
        
        # Use the bounds for voxelization
        voxel_size = 0.05  # 5cm voxels
        print(f"\n3. Setting up voxelization with {voxel_size}m voxels:")
        
        voxelizer = Voxelization(voxel_size, bounds=dynamic_bounds)
        print(f"   Voxel grid dimensions: {voxelizer.n_voxels_x} x {voxelizer.n_voxels_y} x {voxelizer.n_voxels_z}")
        print(f"   Total voxels: {voxelizer.n_voxels_x * voxelizer.n_voxels_y * voxelizer.n_voxels_z:,}")
    else:
        print(f"   Data directory {data_dir} not found. Skipping on-the-fly calculation.")


def voxelize_with_camera_bounds(gs_model_path: str, output_dir: str, 
                               voxel_size: float = 0.05, 
                               bounds_method: str = "viewing_frustum_bounds",
                               aggregation_method: AggregationMethod = AggregationMethod.h3dgs):
    """
    Voxelize a Gaussian splat model using camera bounds.
    
    Args:
        gs_model_path: Path to the Gaussian splat model checkpoint
        output_dir: Directory to save voxelized model
        voxel_size: Size of voxels in meters
        bounds_method: Method to use for bounds calculation
        aggregation_method: Method to use for aggregating Gaussians
    """
    print(f"\n=== Voxelizing Model: {gs_model_path} ===")
    
    # Load the trained model
    if not os.path.exists(gs_model_path):
        print(f"Model not found: {gs_model_path}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model = torch.load(gs_model_path, map_location=device, weights_only=True)
    
    # Get bounds for the ActorsHQ dataset
    try:
        bounds = get_actorshq_bounds(bounds_method)
        print(f"Using bounds method: {bounds_method}")
        print(f"Bounds: {bounds}")
    except FileNotFoundError:
        print("Camera bounds not found. Calculating on-the-fly...")
        data_dir = "data/Actor01/Sequence1/0/resolution_4"
        bounds_calculator = CameraBounds(data_dir, factor=1, normalize=False)
        bounds = bounds_calculator.get_viewing_frustum_bounds(max_depth=5.0, padding=0.5)
    
    # Initialize voxelizer with camera bounds
    voxelizer = Voxelization(voxel_size, bounds=bounds)
    voxelizer.read_gs(trained_model)
    
    # Perform voxelization
    print(f"Voxelizing with {aggregation_method} aggregation method...")
    voxelized_model = voxelizer.voxelize(aggregation_method)
    
    # Save the voxelized model
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "voxelized_model.pt")
    voxelizer.save_voxelized_gs(output_path)
    
    # Save voxelization statistics
    voxelizer.voxel_stats(save_dir=output_dir)
    
    # Print statistics
    print(f"\nVoxelization Results:")
    print(f"Original gaussians: {len(trained_model['splats']['means']):,}")
    print(f"Voxelized gaussians: {len(voxelized_model['splats']['means']):,}")
    print(f"Compression ratio: {len(trained_model['splats']['means']) / len(voxelized_model['splats']['means']):.2f}x")
    print(f"Voxel size: {voxel_size} meters")
    print(f"Grid dimensions: {voxelizer.n_voxels_x} x {voxelizer.n_voxels_y} x {voxelizer.n_voxels_z}")


def main():
    """Main function to demonstrate camera bounds usage."""
    
    # Demo 1: Show how to use camera bounds
    demo_camera_bounds_voxelization()
    
    # Demo 2: Example voxelization (uncomment and modify paths as needed)
    # example_model_path = "path/to/your/gaussian_splat_model.pt"
    # example_output_dir = "results/voxelized_actorshq"
    # 
    # if os.path.exists(example_model_path):
    #     voxelize_with_camera_bounds(
    #         gs_model_path=example_model_path,
    #         output_dir=example_output_dir,
    #         voxel_size=0.05,  # 5cm voxels
    #         bounds_method="viewing_frustum_bounds",
    #         aggregation_method=AggregationMethod.h3dgs
    #     )
    # else:
    #     print(f"\nTo run voxelization demo, modify the paths in the main() function.")
    
    print("\n=== Usage Examples ===")
    print("1. To calculate bounds for ActorsHQ dataset:")
    print("   python lod/camera_bounds.py")
    print()
    print("2. To use bounds in your voxelization code:")
    print("   from lod.camera_bounds import get_actorshq_bounds")
    print("   bounds = get_actorshq_bounds('viewing_frustum_bounds')")
    print("   voxelizer = Voxelization(voxel_size=0.05, bounds=bounds)")
    print()
    print("3. Available bounds methods:")
    print("   - 'camera_position_bounds': Minimal coverage of camera positions")
    print("   - 'viewing_frustum_bounds': Coverage of actual viewing volume (recommended)")
    print("   - 'scene_center_bounds': Symmetric coverage around scene center")


if __name__ == "__main__":
    main() 