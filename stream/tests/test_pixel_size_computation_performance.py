#!/usr/bin/env python3

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from culling import calc_pixel_size_torch_only, calc_pixel_size_torch_cuda


def create_test_data(N, device):
    """Create test Gaussian data."""
    torch.manual_seed(42)
    
    means = torch.randn(N, 3, device=device) * 2.0
    quats = torch.randn(N, 4, device=device)
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)
    scales = torch.rand(N, 3, device=device) * 0.1 + 0.01
    opacities = torch.rand(N, device=device) * 0.8 + 0.1
    
    # Camera parameters
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[2, 3] = -5.0
    
    focal_length = 500.0
    width, height = 800, 600
    K = torch.tensor([
        [focal_length, 0, width/2],
        [0, focal_length, height/2],
        [0, 0, 1]
    ], device=device, dtype=torch.float32)
    
    return means, quats, scales, opacities, viewmat, K, width, height


def benchmark_implementation(func, *args, num_runs=5):
    """Benchmark a function with multiple runs."""
    # Warmup
    for _ in range(2):
        func(*args)
    
    torch.cuda.synchronize()
    times = []

    print("Warming up...")
    start_time = time.time()
    func(*args)
    torch.cuda.synchronize()
    print(f"Warmup Time - {func.__name__}: {(time.time() - start_time)*1000:.2f}ms")
    
    for _ in range(num_runs):
        start_time = time.time()
        result = func(*args)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)
    
    return np.mean(times), np.std(times), result


def performance_analysis():
    """Analyze performance across different dataset sizes."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test different sizes
    sizes = [1000, 5000, 10000, 25000, 50000, 100000]
    pytorch_times = []
    cuda_times = []
    pytorch_stds = []
    cuda_stds = []
    
    for N in sizes:
        print(f"\nTesting with {N} Gaussians...")
        
        # Create test data
        args = create_test_data(N, device)
        
        # Benchmark PyTorch-only implementation
        pytorch_time, pytorch_std, pytorch_result = benchmark_implementation(
            calc_pixel_size_torch_only, *args
        )
        
        # Benchmark PyTorch+CUDA implementation  
        cuda_time, cuda_std, cuda_result = benchmark_implementation(
            calc_pixel_size_torch_cuda, *args
        )
        
        # Check results match
        diff = torch.abs(pytorch_result - cuda_result)
        max_diff = torch.max(diff)
        
        pytorch_times.append(pytorch_time)
        cuda_times.append(cuda_time)
        pytorch_stds.append(pytorch_std)
        cuda_stds.append(cuda_std)
        
        speedup = pytorch_time / cuda_time
        print(f"  PyTorch-only: {pytorch_time:.4f}±{pytorch_std:.4f}s")
        print(f"  PyTorch+CUDA: {cuda_time:.4f}±{cuda_std:.4f}s") 
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Max difference: {max_diff:.8f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.errorbar(sizes, pytorch_times, yerr=pytorch_stds, label='PyTorch-only', marker='o')
    plt.errorbar(sizes, cuda_times, yerr=cuda_stds, label='PyTorch+CUDA', marker='s')
    plt.xlabel('Number of Gaussians')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time vs Dataset Size')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    speedups = np.array(pytorch_times) / np.array(cuda_times)
    plt.plot(sizes, speedups, 'g-o', label='Speedup (PyTorch / PyTorch+CUDA)')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='No speedup')
    plt.xlabel('Number of Gaussians')
    plt.ylabel('Speedup (PyTorch/CUDA)')
    plt.title('CUDA Speedup vs Dataset Size')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.loglog(sizes, pytorch_times, 'b-o', label='PyTorch-only')
    plt.loglog(sizes, cuda_times, 'r-s', label='PyTorch+CUDA')
    plt.xlabel('Number of Gaussians')
    plt.ylabel('Time (seconds)')
    plt.title('Log-Log Plot: Time vs Dataset Size')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    throughput_pytorch = np.array(sizes) / np.array(pytorch_times)
    throughput_cuda = np.array(sizes) / np.array(cuda_times)
    plt.plot(sizes, throughput_pytorch, 'b-o', label='PyTorch-only')
    plt.plot(sizes, throughput_cuda, 'r-s', label='PyTorch+CUDA')
    plt.xlabel('Number of Gaussians')
    plt.ylabel('Throughput (Gaussians/second)')
    plt.title('Throughput vs Dataset Size')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    filename = 'pixel_size_computation_performance_test.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPerformance plot saved as '{filename}'")
    
    # Summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    best_pytorch_idx = np.argmin(pytorch_times)
    best_cuda_idx = np.argmin(cuda_times)
    max_speedup_idx = np.argmax(speedups)
    
    print(f"Best PyTorch-only performance: {pytorch_times[best_pytorch_idx]:.4f}s @ {sizes[best_pytorch_idx]} Gaussians")
    print(f"Best CUDA performance: {cuda_times[best_cuda_idx]:.4f}s @ {sizes[best_cuda_idx]} Gaussians")
    print(f"Maximum speedup: {speedups[max_speedup_idx]:.2f}x @ {sizes[max_speedup_idx]} Gaussians")
    
    # Find crossover point where CUDA becomes faster
    cuda_faster = speedups > 1.0
    if np.any(cuda_faster):
        crossover_idx = np.where(cuda_faster)[0][0]
        print(f"CUDA becomes faster at: {sizes[crossover_idx]} Gaussians")
    else:
        print("CUDA was not faster for any tested dataset size")


if __name__ == "__main__":
    performance_analysis() 