# Logic Differences Between CPU and CUDA Clustering Implementations

## Overview
The CPU and CUDA implementations of center-in-pixel clustering have significant logic mismatches that cause both correctness and crash issues.

## Step-by-Step Comparison

### CPU Implementation (_cluster_center_in_pixel in merging.py)

1. **Input**: `candidate_means[M, 3]`, `candidate_indices[M]`
2. **World → Camera**: `means_cam = _world_to_cam(candidate_means)` → `[M, 3]`
3. **Camera → 2D**: `means2d = proj(means_cam)` → `[M, 2]` (continuous coordinates)
4. **Discretize**: `pixel_coords = torch.floor(means2d).long()` → `[M, 2]` (discrete pixel coordinates)
5. **Filter**: Check bounds and depth > 0 → `valid_mask[M]`
6. **Group**: Create `pixel_groups` dict with `(px, py)` keys
7. **Depth cluster**: Within each pixel, cluster by depth threshold

### CUDA Implementation Issues

#### Current Flow:
1. **Input**: Same as CPU
2. **World → Camera**: Same as CPU → `means_cam[M, 3]` ✅
3. **Camera → 2D**: Same as CPU → `means2d[M, 2]` ✅
4. **❌ MISSING**: No discretization step!
5. **❌ BUG**: Passes `means2d` as `pixel_coords` but kernel expects discrete coordinates
6. **❌ CRASH**: Passes `nullptr` for `valid_mask` but kernel dereferences it

#### Kernel Expectations vs Reality:

```cuda
// Kernel signature:
create_sort_keys_kernel(
    const float* pixel_coords,     // Expected: [N, 2] discrete pixel coords for ALL Gaussians
    const float* means_cam,        // Expected: [N, 3] camera coords for ALL Gaussians  
    const int* candidate_indices,  // Expected: [M] indices into the N arrays
    const bool* valid_mask,        // Expected: [N] validity for ALL Gaussians
    int num_candidates,            // Expected: M
    ...
);

// What's actually passed:
create_sort_keys_kernel(
    means2d,                       // Reality: [M, 2] continuous coords for CANDIDATES only
    means_cam,                     // Reality: [M, 3] camera coords for CANDIDATES only
    candidate_indices,             // Reality: [M] original indices ✅
    nullptr,                       // Reality: NULL POINTER ❌
    num_candidates,                // Reality: M ✅
    ...
);
```

## Required Fixes

### Fix 1: Remove valid_mask completely
Since candidates are pre-filtered, we don't need valid_mask in the kernel.

```cuda
// Remove valid_mask parameter and the check:
__global__ void create_sort_keys_kernel(
    const float* pixel_coords,     // [M, 2] - discrete pixel coordinates
    const float* means_cam,        // [M, 3] - camera coordinates
    const int* candidate_indices,  // [M] - original indices
    // REMOVE: const bool* valid_mask,
    int num_candidates,
    ...
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;
    
    int orig_idx = candidate_indices[idx];
    
    // REMOVE: if (!valid_mask[orig_idx]) { ... }
    
    // Get discrete pixel coordinates (already discrete)
    int px = static_cast<int>(pixel_coords[idx * 2 + 0]);
    int py = static_cast<int>(pixel_coords[idx * 2 + 1]);
    ...
}
```

### Fix 2: Add discretization step
In the Python wrapper, convert continuous to discrete coordinates:

```python
# In _wrapper.py or clustering_cuda.py:
pixel_coords_discrete = torch.floor(means2d).int()  # Convert to discrete
```

### Fix 3: Update kernel call
```cuda
create_sort_keys_kernel<<<grid, block>>>(
    pixel_coords_discrete.data_ptr<int>(),  // Now discrete [M, 2]
    means_cam.data_ptr<float>(),            // [M, 3]
    candidate_indices.data_ptr<int>(),      // [M]
    // REMOVE: nullptr,
    num_candidates,
    ...
);
```

### Fix 4: Update coordinate data types
- Use `int*` for discrete pixel coordinates instead of `float*`
- Update kernel to expect integer pixel coordinates

## Summary
The main issue is architectural mismatch:
- **CPU version**: Processes only candidate data throughout
- **CUDA version**: Expects global data arrays indexed by candidate indices
