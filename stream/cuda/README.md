# CUDA-Accelerated Clustering for Stream Operations

This directory contains CUDA-accelerated implementations of clustering algorithms used in Gaussian Splatting stream operations.

## Overview

The CUDA implementation focuses on the `_cluster_center_in_pixel()` function, which is the most computationally intensive part of the merging pipeline. The GPU acceleration provides significant performance improvements for large-scale Gaussian Splatting applications.

### Key Features

- **Parallel Pixel Coordinate Calculation**: Each Gaussian processed independently on GPU
- **Efficient Pixel Grouping**: Sort-based parallel grouping by (x,y) pixel coordinates  
- **Parallel Depth Clustering**: Segmented operations within pixel groups
- **Automatic Fallback**: Gracefully falls back to CPU implementation when CUDA unavailable
- **Memory Efficient**: Optimized memory allocation and data transfer patterns

## Building the CUDA Extension

### Prerequisites

- CUDA Toolkit (11.0 or later recommended)
- PyTorch with CUDA support
- Compatible GPU with Compute Capability 7.0+ (V100, RTX 2080, A100, RTX 3090, etc.)

### Build Instructions

1. **Navigate to the CUDA directory**:
   ```bash
   cd stream/cuda/
   ```

2. **Build the extension**:
   ```bash
   ./build.sh
   ```

3. **Clean build (if needed)**:
   ```bash
   ./build.sh --clean
   ```

### Verify Installation

```python
cd ..  # Back to stream/
python3 -c "from cuda._wrapper import is_cuda_available; print('CUDA extension available:', is_cuda_available())"
```

## Usage

### Basic Usage

```python
from stream.clustering_cuda import cluster_center_in_pixel_cuda_accelerated

# Your candidate Gaussians
candidate_means = torch.randn(1000, 3).cuda()  # [M, 3]
candidate_indices = np.arange(1000)  # [M]
viewmat = torch.eye(4).cuda()  # [4, 4]
K = torch.tensor([[500, 0, 400], [0, 500, 300], [0, 0, 1]]).float().cuda()

# Perform clustering
clusters = cluster_center_in_pixel_cuda_accelerated(
    candidate_means, candidate_indices, viewmat, K,
    width=800, height=600, 
    depth_threshold=0.1, min_cluster_size=2,
    use_cuda=True  # Automatically falls back to CPU if needed
)

print(f"Found {len(clusters)} clusters")
```

### Performance Information

```python
from stream.clustering_cuda import get_clustering_performance_info

info = get_clustering_performance_info()
print(f"CUDA available: {info['cuda_available']}")
print(f"CUDA extension loaded: {info['cuda_extension_loaded']}")
```

### Benchmarking

```python
from stream.clustering_cuda import benchmark_clustering_performance

results = benchmark_clustering_performance(
    candidate_means, candidate_indices, viewmat, K,
    width=800, height=600, num_runs=5
)

print(f"CPU: {results['cpu_mean_ms']:.2f} ms")
if results['gpu_mean_ms']:
    print(f"GPU: {results['gpu_mean_ms']:.2f} ms")
    print(f"Speedup: {results['speedup']:.2f}x")
```

## Implementation Details

### Algorithm Overview

1. **Coordinate Transformation**: Transform candidate Gaussians to camera coordinates
2. **2D Projection**: Project to pixel coordinates using gsplat's projection functions
3. **Parallel Sorting**: Sort by pixel coordinates (x,y) then by depth
4. **Pixel Grouping**: Group Gaussians falling in the same pixel
5. **Depth Clustering**: Within each pixel group, cluster by depth threshold
6. **Result Compaction**: Return clusters in the same format as Python version

### CUDA Kernel Design

- **Block Size**: 256 threads per block (optimized for most GPUs)
- **Memory Pattern**: Coalesced memory access for optimal bandwidth
- **Synchronization**: Minimal global synchronization points
- **Memory Management**: Efficient allocation/deallocation of temporary arrays

### Performance Characteristics

| Dataset Size | Expected Speedup | GPU Memory Usage |
|--------------|------------------|------------------|
| 1K Gaussians | 2-3x            | ~50 MB          |
| 10K Gaussians| 5-8x            | ~200 MB         |
| 100K Gaussians| 10-15x         | ~1.5 GB         |

*Note: Actual performance depends on GPU architecture and clustering complexity*

## File Structure

```
cuda/
├── README.md                   # This file
├── build.sh                    # Build script
├── setup.py                    # Python extension setup
├── ext.cpp                     # C++ extension interface
├── include/
│   └── clustering.cuh          # CUDA kernel headers
├── csrc/
│   └── clustering.cu           # CUDA kernel implementation
└── _wrapper.py                 # Python wrapper functions
```

## Testing

Run the test suite to validate correctness and performance:

```bash
cd ../tests/
python3 test_clustering_cuda.py
```

Tests include:
- Extension availability check
- Correctness validation against CPU implementation
- Performance benchmarking
- Edge case handling
- Fallback behavior verification

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Use smaller batch sizes or reduce precision
candidate_means = candidate_means.half()  # Use FP16
```

**2. Compilation Errors**
```bash
# Check CUDA compatibility
nvcc --version
python3 -c "import torch; print(torch.version.cuda)"

# Clean and rebuild
./build.sh --clean
./build.sh
```

**3. Runtime Errors**
```python
# Enable CUDA debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

### Performance Tips

1. **Batch Size**: Process larger batches (>1000 Gaussians) for better GPU utilization
2. **Memory Layout**: Keep data on GPU to avoid CPU-GPU transfers
3. **Data Types**: Use float32 for best performance; float16 for memory savings
4. **Clustering Parameters**: Tune `depth_threshold` and `min_cluster_size` for your scene

## Contributing

When modifying the CUDA implementation:

1. Update both the kernel code (`clustering.cu`) and tests
2. Ensure compatibility with the Python reference implementation
3. Validate performance improvements with benchmarks
4. Update documentation for any API changes

## License

This CUDA implementation follows the same license as the parent gsplat project.
