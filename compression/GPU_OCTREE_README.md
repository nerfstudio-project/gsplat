# GPU-Optimized Octree Point Cloud Compression

A high-performance GPU-accelerated algorithm for compressing voxelized point clouds using octree spatial indexing and Morton codes (Z-order curves).

## Overview

This implementation achieves **32:1 compression ratio** with **~4ms compression** and **~5ms decompression** time for point clouds with ~700K points on NVIDIA GPUs.

### Key Features
- **GPU-accelerated**: Leverages CUDA for parallel Morton code computation and octree construction
- **Scalable depth**: Supports octree depths from 1 to 10 (resolution: 2³ to 1024³)
- **Memory efficient**: BFS occupancy encoding with 1 byte per octree node

---

## Installation

```{shell}
cd gsplat/compression
mkdir build
# check below for cmake options
```

The CMakeLists.txt now supports 3 ways to specify CUDA installation:

  1. Auto-detection (Default)

```{shell}
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)
```

  2. CMake Option (User-specified)

```{shell}
  cmake -S . -B build -DCUDA_ROOT=/custom/path/to/cuda -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)
```

  3. Environment Variable

```{shell}
  export CUDA_ROOT=/custom/path/to/cuda
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)
```

  Or:

```{shell}
  export CUDA_HOME=/custom/path/to/cuda  # Alternative variable
  cmake -S . -B build
```

  Priority Order:

  1. CMake option (-DCUDA_ROOT=...) - Highest priority
  2. Environment variable ($CUDA_ROOT or $CUDA_HOME)
  3. Auto-detection - Fallback


---

## Algorithm: Compression

### Step 1: Input Preprocessing
```
Input: Point cloud with N points, each point (x, y, z) ∈ [0, 1023]³
       octree_depth: desired tree depth (e.g., 10 for 1024³ resolution)
```

**Quantization** (if octree_depth < 10):
```cpp
shift = 10 - octree_depth
quantized_coord = original_coord >> shift
```
This allows flexible resolution trade-offs: depth 8 → 256³, depth 9 → 512³, depth 10 → 1024³

### Step 2: Morton Code Computation (GPU Parallel)

**Purpose**: Convert 3D coordinates to 1D Morton codes for spatial sorting.

**GPU Kernel**: `compute_morton_codes<<<blocks, threads>>>`
- Each thread processes one point in parallel
- Computes Morton code by interleaving x, y, z bits

**Morton Encoding** (per coordinate):
```
expandBits(v):
  Input:  10-bit coordinate  →  0b0000001011 (e.g., 11)
  Output: 30-bit expanded    →  0b000000000001000000001001
  
  Bits are spread with 2 zeros between each bit to make room for y and z.
```

**Interleaving**:
```
morton_code = expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2)

Example for point (x=3, y=5, z=2):
  x = 0b011 → expanded: 000 000 001 001
  y = 0b101 → expanded: 000 001 000 001 (shifted left by 1)
  z = 0b010 → expanded: 000 000 001 000 (shifted left by 2)
  
  Interleaved: z₂y₂x₂ z₁y₁x₁ z₀y₀x₀ = 010 101 011
```

### Step 3: Spatial Sorting (GPU Parallel)

**Thrust Sort**:
```cpp
thrust::sort_by_key(morton_codes, indices)
```
- Sorts points by Morton code using GPU parallel radix sort
- Groups spatially nearby points together (locality property of Z-order curve)

### Step 4: Octree Construction - Bottom-Up (GPU Parallel)

**Level octree_depth (Leaves)**:
```cpp
unique_leaf_codes = thrust::unique(sorted_morton_codes)
```
- Removes duplicate Morton codes (points at same voxel)
- Each unique code represents a leaf node

**Parent Levels (octree_depth-1 down to 0)**:
```cpp
for depth d from (octree_depth-1) down to 0:
    parent_codes[d] = child_codes[d+1] >> 3  // Parallel shift by 3 bits
    parent_codes[d] = thrust::unique(parent_codes[d])
```

**Why shift by 3?** Each parent in octree has 8 children. Morton code parent-child relationship:
```
Parent code: PPP...PPP (p bits)
Child codes: PPP...PPP CCC where CCC ∈ {000, 001, 010, ..., 111}
                         ↑ 3 bits encode which of 8 children
```

**Example**:
```
Depth 2 node: morton_code = 0b101011
Children (depth 3): 
  0b101011000, 0b101011001, 0b101011010, ..., 0b101011111
Parent (depth 1): 0b101011 >> 3 = 0b101
```

### Step 5: BFS Occupancy Stream Generation (GPU Parallel)

For each level d from 0 to octree_depth-1:

**GPU Kernel**: `compute_occupancy_kernel<<<blocks, threads>>>`
- Each thread processes one parent node
- Computes 8-bit occupancy mask indicating which children exist

**Occupancy Encoding**:
```cpp
for each parent at level d:
    occupancy = 0b00000000
    base_child = parent_code << 3
    
    for child_offset in {0, 1, 2, 3, 4, 5, 6, 7}:
        if (base_child + child_offset) exists in children[d+1]:
            occupancy |= (1 << child_offset)
    
    output[parent_index] = occupancy  // 1 byte per parent
```

**Example**:
```
Parent has children at offsets: 0, 2, 3, 7
Occupancy bits: 0b10001101 = 0x8D
               ↑       ↑↑↑
            bit 7    bits 3,2,0
```

**Binary Search Optimization**: Since children are sorted, use binary search to find first potential child, then scan contiguously.

### Step 6: Serialization

**Compact Binary Format**:
```
[Header: 4 bytes]
  - num_levels: uint32_t

[Level Sizes: num_levels × 4 bytes]
  - level_size[0]: uint32_t (typically 1 for root)
  - level_size[1]: uint32_t
  - ...
  - level_size[octree_depth]: uint32_t (number of leaves)

[BFS Occupancy Stream: variable size]
  - occupancy[level 0]: level_size[0] bytes
  - occupancy[level 1]: level_size[1] bytes
  - ...
  - occupancy[level octree_depth-1]: level_size[octree_depth-1] bytes
  
Total size ≈ 4 + 4×(depth+1) + Σ(level_sizes)
```

**Not stored** (passed as parameter or fixed):
- `octree_depth`: provided during decompression
- Bounding box: fixed [0, 0, 0] to [1023, 1023, 1023]
- Grid dimensions: fixed 1024³

**Compression Ratio Calculation**:
```
Original size = N × 12 bytes (3 uint32_t per point)
Compressed size ≈ 4 + 4×11 + occupancy_stream_size
                ≈ 48 + (total number of non-leaf nodes) bytes

For 720K points → ~8.4 MB original → ~260 KB compressed → 32:1 ratio
```

---

## Algorithm: Decompression

### Step 1: Deserialization (CPU)

Read header and metadata:
```cpp
num_levels = read_uint32()
level_sizes[] = read_uint32_array(num_levels)
bfs_stream[] = read_bytes(remaining)
```

Validate octree depth:
```cpp
if (num_levels != octree_depth + 1):
    error("Depth mismatch!")
```

### Step 2: Octree Reconstruction - Top-Down (Hybrid CPU-GPU)

**Initialize root**:
```cpp
level[0] = {0}  // Root has Morton code 0
```

**Reconstruct levels 1 to octree_depth**:
```cpp
for depth d from 0 to octree_depth-1:
    // Read occupancy for this level
    occupancy_bytes = bfs_stream[offset : offset + level_sizes[d]]
    
    // CPU: Expand occupancy bits to child codes
    children = []
    for p in 0 to level_sizes[d]-1:
        parent_code = level[d][p]
        occ = occupancy_bytes[p]
        base_child = parent_code << 3
        
        for bit in 0 to 7:
            if (occ & (1 << bit)):
                children.append(base_child + bit)
    
    // Upload to GPU
    level[d+1] = children
```

**Why CPU for reconstruction?**
- Variable output size per parent (1-8 children)
- Irregular memory access pattern
- CPU `push_back` is simpler and faster than GPU dynamic allocation
- Data transfer overhead is minimal (occupancy bytes are small)

### Step 3: Leaf Morton Code Decoding (GPU Parallel)

**GPU Kernel**: `reconstruct_points_kernel<<<blocks, threads>>>`
- Each thread processes one leaf node in parallel
- Decodes Morton code back to (x, y, z) coordinates

**Morton Decoding**:
```cpp
compactBits(v):
  Input:  30-bit expanded → 0b000000000001000000001001
  Output: 10-bit compact  → 0b0000001011 (value 11)
  
  Extracts every 3rd bit by masking and shifting.

morton3D_decode(code):
  x = compactBits(code)
  y = compactBits(code >> 1)
  z = compactBits(code >> 2)
```

**Dequantization** (if octree_depth < 10):
```cpp
shift = 10 - octree_depth
x = decoded_x << shift
y = decoded_y << shift
z = decoded_z << shift
```

### Step 4: Output Generation (CPU)

Copy reconstructed coordinates from GPU to CPU and save to PLY file:
```cpp
points = download_from_gpu(x[], y[], z[])
save_ply_geometry(points, output_path)
```

---

## Parallelization Strategy

### Compression Parallelism

| Step | Parallelization | Device |
|------|----------------|--------|
| Morton encoding | Data parallel (1 thread/point) | GPU |
| Sorting | Parallel radix sort | GPU |
| Parent generation | Data parallel (1 thread/node) | GPU |
| Occupancy computation | Data parallel (1 thread/parent) | GPU |

**Key insight**: Morton codes enable massive parallelism by converting irregular 3D octree to regular 1D array operations.

### Decompression Parallelism

| Step | Strategy | Device | Rationale |
|------|----------|--------|-----------|
| Octree reconstruction | Sequential CPU | CPU | Variable branching, small data |
| Morton decoding | Data parallel | GPU | Regular computation, large data |

**Hybrid approach**: Uses CPU for irregular tasks and GPU for data-parallel tasks.

---

### Benchmark Results (NVIDIA GPU, 720K points)

| Metric | Value |
|--------|-------|
| **Compression time** | 4 ms (after warm-up) |
| **Decompression time** | 5 ms |
| **Original size** | 8,444 KB |
| **Compressed size** | 262 KB |
| **Compression ratio** | 32.18:1 |
| **Throughput** | ~180M points/second |

**Comparison with other methods**:
- **2.3× better ratio** than Draco (13.7:1)
- **9× faster** than Draco (38 ms)
- **114× faster** than PCL (455 ms)