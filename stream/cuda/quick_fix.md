# Quick Fix for CUDA Kernel Bug

## Problem:
- `create_sort_keys_kernel` expects `valid_mask` parameter
- Calling code passes `nullptr` 
- Kernel tries to access `valid_mask[orig_idx]` → illegal memory access

## Solution 1 (Simplest): Remove valid_mask check
Since candidates are pre-filtered, we don't need valid_mask in the kernel.

### Change in clustering.cu:

1. Remove valid_mask parameter from create_sort_keys_kernel:
```cuda
__global__ void create_sort_keys_kernel(
    const float* pixel_coords,     // [N, 2]
    const float* means_cam,        // [N, 3] 
    const int* candidate_indices,  // [M]
    // REMOVE: const bool* valid_mask,        // [N]
    int num_candidates,
    uint64_t* pixel_hashes,        // [M] - output
    float* depths,                 // [M] - output
    int* indices,                  // [M] - output
    int* valid_count               // [1] - output
)
```

2. Remove the valid_mask check:
```cuda
int orig_idx = candidate_indices[idx];

// REMOVE THIS:
// if (!valid_mask[orig_idx]) {
//     pixel_hashes[idx] = UINT64_MAX;
//     depths[idx] = FLT_MAX;
//     indices[idx] = -1;
//     return;
// }

// Keep the rest of the kernel...
```

3. Update the kernel call:
```cuda
create_sort_keys_kernel<<<grid, block>>>(
    pixel_coords, means_cam, candidate_indices, 
    num_candidates,  // Remove nullptr here
    thrust::raw_pointer_cast(pixel_hashes.data()),
    thrust::raw_pointer_cast(depths.data()),
    thrust::raw_pointer_cast(indices.data()),
    thrust::raw_pointer_cast(valid_count.data())
);
```
