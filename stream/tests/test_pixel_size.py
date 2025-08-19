#!/usr/bin/env python3

import torch
import numpy as np
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from culling import calc_pixel_size_torch_only, calc_pixel_size_torch_cuda

def calc_pixel_size_method_comparison(means, quats, scales, opacities, colors, viewmat, K, width, height, near=0.01, far=100.0):
    """
    Calculate the pixel size of a Gaussian in the image plane.

    Args:
        means: [N, 3] - Gaussian centers in world coordinates
        quats: [N, 4] - Gaussian quaternions
        scales: [N, 3] - Gaussian scales
        opacities: [N, ] - Gaussian opacities
        colors: [N, K, 3] - Gaussian colors
        viewmat: [4, 4] - view matrix (world to camera)
        K: [3, 3] - camera intrinsic matrix
        width: int - image width
        height: int - image height
        near: float - near clipping plane distance
        far: float - far clipping plane distance

    Returns:
        pixel_size: [N, ] - pixel size of each Gaussian in the image plane
    """
    
    # Warm-up runs to avoid CUDA kernel compilation overhead
    print("Warming up...")
    calc_pixel_size_torch_only(means, quats, scales, opacities, viewmat, K, width, height)
    calc_pixel_size_torch_cuda(means, quats, scales, opacities, viewmat, K, width, height)
    torch.cuda.synchronize()
    
    # Test PyTorch-only implementation with 5 runs
    print("Testing PyTorch-only implementation...")
    torch_times = []
    for run in range(5):
        start_time = time.time()
        pixel_size_torch = calc_pixel_size_torch_only(means, quats, scales, opacities, viewmat, K, width, height)
        torch.cuda.synchronize()
        torch_time = time.time() - start_time
        torch_times.append(torch_time)
        print(f"  Run {run+1}: {torch_time:.6f}s")
    
    avg_torch_time = sum(torch_times) / len(torch_times)
    print(f"PyTorch-only average: {avg_torch_time:.6f} seconds")
    
    # Test PyTorch+CUDA implementation with 5 runs  
    print("Testing PyTorch+CUDA implementation...")
    cuda_times = []
    for run in range(5):
        start_time = time.time()
        pixel_size_cuda = calc_pixel_size_torch_cuda(means, quats, scales, opacities, viewmat, K, width, height)
        torch.cuda.synchronize()
        cuda_time = time.time() - start_time
        cuda_times.append(cuda_time)
        print(f"  Run {run+1}: {cuda_time:.6f}s")
    
    avg_cuda_time = sum(cuda_times) / len(cuda_times)
    print(f"PyTorch+CUDA average: {avg_cuda_time:.6f} seconds")
    
    # Compare results
    diff = torch.abs(pixel_size_torch - pixel_size_cuda)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    print(f"Max difference: {max_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")
    print(f"Results match: {torch.allclose(pixel_size_torch, pixel_size_cuda, atol=1e-6)}")
    
    speedup = avg_torch_time / avg_cuda_time if avg_cuda_time > 0 else float('inf')
    print(f"CUDA speedup: {speedup:.2f}x")
    
    return pixel_size_cuda

def test_pixel_size():
    """Test the pixel size calculation with sample data."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample Gaussian data
    N = 10000  # Number of Gaussians
    
    # Random Gaussian parameters
    torch.manual_seed(42)
    means = torch.randn(N, 3, device=device) * 2.0  # World coordinates
    
    # Random quaternions (normalized)
    quats = torch.randn(N, 4, device=device)
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)
    
    # Random scales
    scales = torch.rand(N, 3, device=device) * 0.1 + 0.01  # Small positive scales
    
    # Random opacities
    opacities = torch.rand(N, device=device) * 0.8 + 0.1  # Between 0.1 and 0.9
    
    # Dummy colors (not used in pixel size calculation)
    colors = torch.randn(N, 3, 3, device=device)
    
    # Camera parameters
    # Simple camera looking down the Z axis
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[2, 3] = -5.0  # Move camera back along Z
    
    # Camera intrinsics
    focal_length = 500.0
    width, height = 800, 600
    K = torch.tensor([
        [focal_length, 0, width/2],
        [0, focal_length, height/2],
        [0, 0, 1]
    ], device=device, dtype=torch.float32)
    
    print(f"Testing with {N} Gaussians")
    print(f"Image size: {width}x{height}")
    print(f"Focal length: {focal_length}")
    
    try:
        # Test our implementation
        pixel_sizes = calc_pixel_size_method_comparison(
            means, quats, scales, opacities, colors, 
            viewmat, K, width, height
        )
        
        # Print statistics
        print(f"\nPixel size statistics:")
        print(f"  Min: {torch.min(pixel_sizes):.6f}")
        print(f"  Max: {torch.max(pixel_sizes):.6f}")
        print(f"  Mean: {torch.mean(pixel_sizes):.6f}")
        print(f"  Std: {torch.std(pixel_sizes):.6f}")
        
        # Check for reasonable values
        valid_mask = torch.isfinite(pixel_sizes) & (pixel_sizes > 0)
        valid_ratio = torch.sum(valid_mask).float() / N
        print(f"  Valid pixel sizes: {valid_ratio:.2%}")
        
        if valid_ratio > 0.99:  # At least 99% should be valid
            print("✅ Test passed! Pixel size calculation appears to work correctly.")
        else:
            print("❌ Test failed! Too many invalid pixel sizes.")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_pixel_size() 