#include "../include/clustering.cuh"
#include "../include/cuda_timer.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <thrust/extrema.h>
#include <cub/cub.cuh>
#include <algorithm>

namespace stream {

// Hash function for pixel coordinates
__device__ __inline__ uint64_t hash_pixel_coord(int x, int y) {
    return (static_cast<uint64_t>(x) << 32) | static_cast<uint64_t>(y);
}

// Extract x coordinate from hash
__device__ __inline__ int extract_x(uint64_t hash) {
    return static_cast<int>(hash >> 32);
}

// Extract y coordinate from hash
__device__ __inline__ int extract_y(uint64_t hash) {
    return static_cast<int>(hash & 0xFFFFFFFF);
}

// Kernel to compute pixel coordinates and validity mask
__global__ void compute_pixel_coords_kernel(
    const float* means_cam,    // [N, 3]
    int num_gaussians,
    int width,
    int height,
    float* pixel_coords,       // [N, 2] - output
    bool* valid_mask           // [N] - output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;
    
    // Extract camera coordinates
    float x_cam = means_cam[idx * 3 + 0];
    float y_cam = means_cam[idx * 3 + 1];
    float z_cam = means_cam[idx * 3 + 2];
    
    // Check if point is in front of camera
    if (z_cam <= 0.0f) {
        valid_mask[idx] = false;
        return;
    }
    
    // Note: Assuming the means_cam are already projected to 2D coordinates
    // If not, we would need to pass K matrix and do projection here
    float x_pixel = x_cam;
    float y_pixel = y_cam;
    
    // Convert to discrete pixel coordinates
    int px = static_cast<int>(floorf(x_pixel));
    int py = static_cast<int>(floorf(y_pixel));
    
    // Check bounds
    bool valid = (px >= 0) && (px < width) && (py >= 0) && (py < height);
    
    pixel_coords[idx * 2 + 0] = static_cast<float>(px);
    pixel_coords[idx * 2 + 1] = static_cast<float>(py);
    valid_mask[idx] = valid;
}

// Kernel to create pixel-depth pairs for sorting
__global__ void create_sort_keys_kernel(
    const int* pixel_coords,       // [M, 2] - discrete pixel coordinates for candidates only
    const float* means_cam,        // [M, 3] - camera coordinates for candidates only 
    const int* candidate_indices,  // [M] - original indices
    int num_candidates,
    uint64_t* pixel_hashes,        // [M] - output: pixel coordinate hashes
    float* depths,                 // [M] - output: depths for sorting
    int* indices,                  // [M] - output: candidate indices for sorting
    int* valid_count               // [1] - output: number of valid candidates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;
    
    // Get discrete pixel coordinates (already filtered and discrete)
    int px = pixel_coords[idx * 2 + 0];
    int py = pixel_coords[idx * 2 + 1];
    
    // Get depth from candidate means
    float depth = means_cam[idx * 3 + 2];
    
    // Get original index
    int orig_idx = candidate_indices[idx];
    
    // Create hash and store data
    pixel_hashes[idx] = hash_pixel_coord(px, py);
    depths[idx] = depth;
    indices[idx] = orig_idx;
    
    // Count all candidates as valid (they're pre-filtered)
    if (idx == 0) {
        *valid_count = num_candidates;
    }
}

// Optimized: One thread block per pixel group for massive parallelism
__global__ void optimized_pixel_clustering_kernel(
    const uint64_t* sorted_pixel_hashes,  // [M] - sorted pixel hashes
    const float* sorted_depths,           // [M] - corresponding sorted depths  
    const int* sorted_indices,            // [M] - corresponding original indices
    const int* pixel_group_starts,       // [num_pixel_groups] - start index of each pixel group
    const int* pixel_group_sizes,        // [num_pixel_groups] - size of each pixel group  
    int num_pixel_groups,
    float depth_threshold,
    int min_cluster_size,
    int* cluster_assignments,             // [M] - output: cluster ID (-1 if not clustered)
    int* cluster_counter                  // [1] - global cluster counter
) {
    int pixel_group_id = blockIdx.x;
    if (pixel_group_id >= num_pixel_groups) return;
    
    int group_start = pixel_group_starts[pixel_group_id];
    int group_size = pixel_group_sizes[pixel_group_id];
    int group_end = group_start + group_size - 1;
    
    // Skip groups too small to cluster
    if (group_size < min_cluster_size) {
        // Mark all as unclustered
        for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
            cluster_assignments[group_start + i] = -1;
        }
        return;
    }
    
    // Shared memory for this pixel group's clustering 
    extern __shared__ int shared_mem[];
    int* temp_cluster_ids = shared_mem;                    // [group_size]
    float* shared_depths = (float*)(shared_mem + group_size); // [group_size]
    
    // Load depths into shared memory for faster access
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        shared_depths[i] = sorted_depths[group_start + i];
        temp_cluster_ids[i] = -1; // Initialize as unclustered
    }
    __syncthreads();
    
    // Only thread 0 does the sequential depth clustering (it's inherently sequential)
    if (threadIdx.x == 0) {
        int cluster_start = 0;
        int current_cluster_id = 0;
        temp_cluster_ids[0] = current_cluster_id;
        
        for (int i = 1; i < group_size; i++) {
            // CORRECT: Match PyTorch - check consecutive pairs (not cluster start)
            float depth_diff = fabsf(shared_depths[i] - shared_depths[i - 1]);
            
            if (depth_diff <= depth_threshold) {
                // Continue current cluster - consecutive elements within threshold
                temp_cluster_ids[i] = current_cluster_id;
            } else {
                // Check if previous cluster was large enough
                int prev_cluster_size = i - cluster_start;
                if (prev_cluster_size < min_cluster_size) {
                    // Mark previous cluster as invalid
                    for (int j = cluster_start; j < i; j++) {
                        temp_cluster_ids[j] = -1;
                    }
                }
                
                // Start new cluster
                current_cluster_id++;
                temp_cluster_ids[i] = current_cluster_id;
                cluster_start = i;
            }
        }
        
        // Handle last cluster
        int last_cluster_size = group_size - cluster_start;
        if (last_cluster_size < min_cluster_size) {
            // Mark last cluster as invalid
            for (int j = cluster_start; j < group_size; j++) {
                temp_cluster_ids[j] = -1;
            }
        }
    }
    __syncthreads();
    
    // All threads cooperate to write results back to global memory
    // First, get base cluster ID for this pixel group
    __shared__ int base_cluster_id;
    __shared__ int local_to_global_mapping[128]; // Map local cluster ID to global offset
    if (threadIdx.x == 0) {
        // FIX: Count only valid (non-negative) unique cluster IDs and create mapping
        const int MAX_CLUSTERS_PER_PIXEL = 128; // Increased limit for safety (was 32)
        bool cluster_exists[128]; // More generous limit to handle dense pixels
        for (int i = 0; i < MAX_CLUSTERS_PER_PIXEL; i++) {
            cluster_exists[i] = false;
            local_to_global_mapping[i] = -1; // Invalid mapping
        }
        
        int num_valid_clusters = 0;
        
        // First pass: identify unique valid cluster IDs
        for (int i = 0; i < group_size; i++) {
            int local_cluster_id = temp_cluster_ids[i];
            if (local_cluster_id >= 0 && local_cluster_id < MAX_CLUSTERS_PER_PIXEL && !cluster_exists[local_cluster_id]) {
                cluster_exists[local_cluster_id] = true;
                num_valid_clusters++;
            }
        }
        
        // Second pass: create mapping from local to global cluster IDs
        int global_offset = 0;
        for (int local_id = 0; local_id < MAX_CLUSTERS_PER_PIXEL; local_id++) {
            if (cluster_exists[local_id]) {
                local_to_global_mapping[local_id] = global_offset++;
            }
        }
        
        // Get base cluster ID from global counter
        if (num_valid_clusters > 0) {
            base_cluster_id = atomicAdd(cluster_counter, num_valid_clusters);
        } else {
            base_cluster_id = -1;
        }
    }
    __syncthreads();
    
    // Write final cluster assignments using proper mapping
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int local_cluster_id = temp_cluster_ids[i];
        if (local_cluster_id >= 0 && base_cluster_id >= 0 && local_cluster_id < 128) {
            int global_offset = local_to_global_mapping[local_cluster_id];
            if (global_offset >= 0) {
                cluster_assignments[group_start + i] = base_cluster_id + global_offset;
            } else {
                cluster_assignments[group_start + i] = -1;
            }
        } else {
            cluster_assignments[group_start + i] = -1;
        }
    }
}

// Optimized boundary marking kernel - much faster than thrust::transform
__global__ void mark_boundaries_kernel(
    const uint64_t* sorted_pixel_hashes,
    int num_valid,
    int* boundary_flags
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_valid) {
        if (idx == 0) {
            boundary_flags[idx] = 1; // First element is always a boundary
        } else {
            // Mark boundary if different from previous element
            boundary_flags[idx] = (sorted_pixel_hashes[idx] != sorted_pixel_hashes[idx - 1]) ? 1 : 0;
        }
    }
}

// Optimized group starts extraction kernel
__global__ void extract_group_starts_kernel(
    const int* boundary_flags,
    const int* prefix_sum,
    int num_valid,
    int num_groups,
    int* group_starts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_valid) return;
    
    // If this is a boundary, record the group start position
    if (boundary_flags[idx] == 1) {
        int group_id = prefix_sum[idx]; // Use prefix sum directly as group ID
        
        if (group_id >= 0 && group_id < num_groups) {
            group_starts[group_id] = idx;
        }
    }
}

// Optimized group sizes computation kernel  
__global__ void compute_group_sizes_kernel(
    const int* group_starts,
    int num_groups,
    int num_valid,
    int* group_sizes
) {
    int group_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_id >= num_groups) return;
    
    int start_pos = group_starts[group_id];
    int end_pos;
    
    if (group_id == num_groups - 1) {
        // Last group extends to end of data
        end_pos = num_valid;
    } else {
        // Use next group's start position
        end_pos = group_starts[group_id + 1];
    }
    
    group_sizes[group_id] = end_pos - start_pos;
}

// Ultra-fast pixel group boundary computation using CUB
void compute_pixel_group_boundaries(
    const thrust::device_vector<uint64_t>& sorted_pixel_hashes,
    int num_valid,
    thrust::device_vector<int>& group_starts,
    thrust::device_vector<int>& group_sizes,
    int& num_pixel_groups
) {
    if (num_valid == 0) {
        num_pixel_groups = 0;
        return;
    }
    
    // Step 1: Mark boundaries using optimized kernel (replaces slow thrust::transform)
    thrust::device_vector<int> boundary_flags(num_valid);
    
    dim3 block(512); // Increase block size for better occupancy
    dim3 grid((num_valid + block.x - 1) / block.x);
    
    mark_boundaries_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(sorted_pixel_hashes.data()),
        num_valid,
        thrust::raw_pointer_cast(boundary_flags.data())
    );
    
    // Step 2: Ultra-fast prefix sum using CUB (much faster than thrust::scan)
    thrust::device_vector<int> prefix_sum(num_valid);
    
    // Determine temporary device storage requirements for CUB prefix sum
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        thrust::raw_pointer_cast(boundary_flags.data()),
        thrust::raw_pointer_cast(prefix_sum.data()),
        num_valid
    );
    
    // Allocate temporary storage
    thrust::device_vector<char> temp_storage(temp_storage_bytes);
    
    // Run CUB exclusive prefix sum (ultra-fast)
    cub::DeviceScan::ExclusiveSum(
        thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes,
        thrust::raw_pointer_cast(boundary_flags.data()),
        thrust::raw_pointer_cast(prefix_sum.data()),
        num_valid
    );
    
    // Step 3: Get total number of groups (last element + boundary flag)
    int last_prefix = 0;
    int last_boundary = 0;
    cudaMemcpy(&last_prefix, thrust::raw_pointer_cast(prefix_sum.data()) + num_valid - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_boundary, thrust::raw_pointer_cast(boundary_flags.data()) + num_valid - 1, sizeof(int), cudaMemcpyDeviceToHost);
    num_pixel_groups = last_prefix + last_boundary;
    
    if (num_pixel_groups == 0) return;
    
    // Step 4: Allocate output arrays
    group_starts.resize(num_pixel_groups);
    group_sizes.resize(num_pixel_groups);
    
    // Step 5: Extract group starts efficiently
    extract_group_starts_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(boundary_flags.data()),
        thrust::raw_pointer_cast(prefix_sum.data()),
        num_valid,
        num_pixel_groups,
        thrust::raw_pointer_cast(group_starts.data())
    );
    
    // Step 6: Compute group sizes efficiently (minimal parallel work)
    dim3 block_sizes(256);
    dim3 grid_sizes((num_pixel_groups + block_sizes.x - 1) / block_sizes.x);
    
    compute_group_sizes_kernel<<<grid_sizes, block_sizes>>>(
        thrust::raw_pointer_cast(group_starts.data()),
        num_pixel_groups,
        num_valid,
        thrust::raw_pointer_cast(group_sizes.data())
    );
    
    cudaDeviceSynchronize();
}

// Main clustering implementation
cudaError_t cluster_center_in_pixel_cuda(
    const float* means_cam,
    const float* pixel_coords,
    const int* candidate_indices,
    int num_candidates,
    const ClusteringConfig& config,
    ClusterResult& result
) {
    // Initialize detailed timing
    CudaTimer timer(true);
    timer.reserve_events(10);
    
    if (num_candidates < config.min_cluster_size) {
        result.num_clusters = 0;
        result.total_clustered = 0;
        return cudaSuccess;
    }
    
    // Stage 1: Allocate temporary device memory
    timer.start_stage("1. Memory Allocation");
    thrust::device_vector<uint64_t> pixel_hashes(num_candidates);
    thrust::device_vector<float> depths(num_candidates);
    thrust::device_vector<int> indices(num_candidates);
    thrust::device_vector<int> valid_count(1);
    timer.end_stage();
    
    // Stage 2: Create sort keys
    timer.start_stage("2. Sort Key Creation Kernel");
    dim3 block(256);
    dim3 grid((num_candidates + block.x - 1) / block.x);
    
    // pixel_coords and means_cam are now candidate-only arrays
    create_sort_keys_kernel<<<grid, block>>>(
        reinterpret_cast<const int*>(pixel_coords), // Now expecting int* for discrete coordinates
        means_cam, 
        candidate_indices, 
        num_candidates,
        thrust::raw_pointer_cast(pixel_hashes.data()),
        thrust::raw_pointer_cast(depths.data()),
        thrust::raw_pointer_cast(indices.data()),
        thrust::raw_pointer_cast(valid_count.data())
    );
    cudaDeviceSynchronize();
    timer.end_stage();
    
    int num_valid = valid_count[0];
    if (num_valid < config.min_cluster_size) {
        result.num_clusters = 0;
        result.total_clustered = 0;
        return cudaSuccess;
    }
    
    // Stage 3: Data validation/filtering
    timer.start_stage("3. Data Validation/Filtering");
    auto valid_end = thrust::remove_if(
        thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.begin(), depths.begin(), indices.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.end(), depths.end(), indices.end())),
        [=] __device__ (const thrust::tuple<uint64_t, float, int>& t) {
            return thrust::get<0>(t) == UINT64_MAX;
        }
    );
    
    num_valid = valid_end - thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.begin(), depths.begin(), indices.begin()));
    timer.end_stage();
    
    // Stage 4: Sorting by pixel hash and depth
    timer.start_stage("4. Sorting Data");
    thrust::sort(
        thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.begin(), depths.begin(), indices.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.begin() + num_valid, depths.begin() + num_valid, indices.begin() + num_valid)),
        [=] __device__ (const thrust::tuple<uint64_t, float, int>& a, const thrust::tuple<uint64_t, float, int>& b) {
            if (thrust::get<0>(a) != thrust::get<0>(b)) {
                return thrust::get<0>(a) < thrust::get<0>(b);
            }
            return thrust::get<1>(a) < thrust::get<1>(b); // Sort by depth within pixel
        }
    );
    timer.end_stage();
    
    // Stage 5: Compute pixel group boundaries
    timer.start_stage("5. Pixel Group Boundaries");
    thrust::device_vector<int> group_starts, group_sizes;
    int num_pixel_groups;
    
    compute_pixel_group_boundaries(pixel_hashes, num_valid, group_starts, group_sizes, num_pixel_groups);
    timer.end_stage();
    
    if (num_pixel_groups == 0) {
        result.num_clusters = 0;
        result.total_clustered = 0;
        return cudaSuccess;
    }
    
    // Stage 6: Setup clustering kernel
    timer.start_stage("6. Clustering Setup");
    thrust::device_vector<int> cluster_assignments(num_valid, -1);
    thrust::device_vector<int> cluster_counter(1, 0);
    
    int max_group_size = *thrust::max_element(group_sizes.begin(), group_sizes.end());
    size_t shared_mem_size = max_group_size * (sizeof(int) + sizeof(float)); // temp_cluster_ids + shared_depths
    
    dim3 grid_opt(num_pixel_groups);
    dim3 block_opt(std::min(256, max_group_size)); // Adjust block size based on group size
    timer.end_stage();
    
    // Stage 7: Main clustering kernel execution  
    timer.start_stage("7. Main Clustering Kernel");
    optimized_pixel_clustering_kernel<<<grid_opt, block_opt, shared_mem_size>>>(
        thrust::raw_pointer_cast(pixel_hashes.data()),
        thrust::raw_pointer_cast(depths.data()),
        thrust::raw_pointer_cast(indices.data()),
        thrust::raw_pointer_cast(group_starts.data()),
        thrust::raw_pointer_cast(group_sizes.data()),
        num_pixel_groups,
        config.depth_threshold,
        config.min_cluster_size,
        thrust::raw_pointer_cast(cluster_assignments.data()),
        thrust::raw_pointer_cast(cluster_counter.data())
    );
    cudaDeviceSynchronize();
    timer.end_stage();
    
    int total_clusters = cluster_counter[0];
    
    // Stage 8: CORRECTED result processing - group indices by cluster ID
    timer.start_stage("8. Result Processing");
    result.num_clusters = total_clusters;
    
    // Count total clustered Gaussians using fast thrust::count_if
    int total_clustered = thrust::count_if(cluster_assignments.begin(), cluster_assignments.end(),
                                          [] __device__ (int x) { return x >= 0; });
    result.total_clustered = total_clustered;
    
    if (total_clustered > 0) {
        // Allocate result memory
        cudaMalloc(&result.cluster_indices, total_clustered * sizeof(int));
        cudaMalloc(&result.cluster_sizes, total_clusters * sizeof(int));
        cudaMalloc(&result.cluster_offsets, (total_clusters + 1) * sizeof(int));
        
        // FIXED: Group indices by cluster ID instead of just copying valid indices
        thrust::device_vector<int> temp_indices(total_clustered);
        thrust::device_vector<int> temp_cluster_ids(total_clustered);
        
        // Extract clustered indices and their cluster IDs together
        auto compact_end = thrust::copy_if(
            thrust::make_zip_iterator(thrust::make_tuple(indices.begin(), cluster_assignments.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(indices.begin() + num_valid, cluster_assignments.begin() + num_valid)),
            thrust::make_zip_iterator(thrust::make_tuple(temp_indices.begin(), temp_cluster_ids.begin())),
            [] __device__ (const thrust::tuple<int, int>& t) {
                return thrust::get<1>(t) >= 0;  // cluster_assignment >= 0
            }
        );
        
        // Sort indices by cluster ID to group them properly
        thrust::sort_by_key(
            temp_cluster_ids.begin(), 
            temp_cluster_ids.begin() + total_clustered,
            temp_indices.begin()
        );
        
        // Copy sorted indices to result
        thrust::copy(temp_indices.begin(), temp_indices.begin() + total_clustered, 
                    thrust::device_pointer_cast(result.cluster_indices));
        
        // Compute cluster sizes using the now-sorted cluster IDs
        thrust::device_vector<int> cluster_sizes(total_clusters, 0);
        
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceHistogram::HistogramEven(
            d_temp_storage, temp_storage_bytes,
            thrust::raw_pointer_cast(temp_cluster_ids.data()),
            thrust::raw_pointer_cast(cluster_sizes.data()),
            total_clusters + 1, 0, total_clusters, total_clustered
        );
        
        thrust::device_vector<char> temp_storage(temp_storage_bytes);
        cub::DeviceHistogram::HistogramEven(
            thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes,
            thrust::raw_pointer_cast(temp_cluster_ids.data()),
            thrust::raw_pointer_cast(cluster_sizes.data()),
            total_clusters + 1, 0, total_clusters, total_clustered
        );
        
        // Copy cluster sizes to result
        thrust::copy(cluster_sizes.begin(), cluster_sizes.end(), thrust::device_pointer_cast(result.cluster_sizes));
        
        // Compute cluster offsets using fast thrust scan
        thrust::exclusive_scan(cluster_sizes.begin(), cluster_sizes.end(), thrust::device_pointer_cast(result.cluster_offsets));
        
        // Set final offset
        cudaMemcpy(result.cluster_offsets + total_clusters, &total_clustered, sizeof(int), cudaMemcpyHostToDevice);
    }
    timer.end_stage();
    
    // Print detailed timing breakdown
    timer.print_results("🔥 CUDA Clustering Microbenchmark 🔥");
    
    return cudaGetLastError();
}

// Debug function: Extract pixel groups after step 2 (grouping + sorting)
cudaError_t extract_pixel_groups_step2(
    const float* means_cam,        // [N, 3] - camera coordinates  
    const float* pixel_coords,     // [N, 2] - 2D pixel coordinates
    const int* candidate_indices,  // [M] - original indices of candidates
    int num_candidates,
    const ClusteringConfig& config,
    int** group_starts,            // [num_groups] - start index of each group
    int** group_sizes,             // [num_groups] - size of each group
    uint64_t** sorted_pixel_hashes,// [num_valid] - sorted pixel hashes
    float** sorted_depths,         // [num_valid] - sorted depths
    int** sorted_indices,          // [num_valid] - sorted original indices
    int* num_groups,               // output: number of pixel groups
    int* num_valid                 // output: number of valid candidates
) {
    if (num_candidates < config.min_cluster_size) {
        *num_groups = 0;
        *num_valid = 0;
        return cudaSuccess;
    }
    
    // Stage 1: Allocate temporary device memory
    thrust::device_vector<uint64_t> pixel_hashes(num_candidates);
    thrust::device_vector<float> depths(num_candidates);
    thrust::device_vector<int> indices(num_candidates);
    thrust::device_vector<int> valid_count(1);
    
    // Stage 2: Create sort keys
    dim3 block(256);
    dim3 grid((num_candidates + block.x - 1) / block.x);
    
    create_sort_keys_kernel<<<grid, block>>>(
        reinterpret_cast<const int*>(pixel_coords),
        means_cam, 
        candidate_indices, 
        num_candidates,
        thrust::raw_pointer_cast(pixel_hashes.data()),
        thrust::raw_pointer_cast(depths.data()),
        thrust::raw_pointer_cast(indices.data()),
        thrust::raw_pointer_cast(valid_count.data())
    );
    cudaDeviceSynchronize();
    
    int num_valid_local = valid_count[0];
    if (num_valid_local < config.min_cluster_size) {
        *num_groups = 0;
        *num_valid = 0;
        return cudaSuccess;
    }
    
    // Stage 3: Data validation/filtering
    auto valid_end = thrust::remove_if(
        thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.begin(), depths.begin(), indices.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.end(), depths.end(), indices.end())),
        [=] __device__ (const thrust::tuple<uint64_t, float, int>& t) {
            return thrust::get<0>(t) == UINT64_MAX;
        }
    );
    
    num_valid_local = valid_end - thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.begin(), depths.begin(), indices.begin()));
    
    // Stage 4: Sorting by pixel hash and depth
    thrust::sort(
        thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.begin(), depths.begin(), indices.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.begin() + num_valid_local, depths.begin() + num_valid_local, indices.begin() + num_valid_local)),
        [=] __device__ (const thrust::tuple<uint64_t, float, int>& a, const thrust::tuple<uint64_t, float, int>& b) {
            if (thrust::get<0>(a) != thrust::get<0>(b)) {
                return thrust::get<0>(a) < thrust::get<0>(b);
            }
            return thrust::get<1>(a) < thrust::get<1>(b); // Sort by depth within pixel
        }
    );
    
    // Stage 5: Compute pixel group boundaries  
    thrust::device_vector<int> group_starts_vec, group_sizes_vec;
    int num_pixel_groups;
    
    compute_pixel_group_boundaries(pixel_hashes, num_valid_local, group_starts_vec, group_sizes_vec, num_pixel_groups);
    
    if (num_pixel_groups == 0) {
        *num_groups = 0;
        *num_valid = 0;
        return cudaSuccess;
    }
    
    // Copy results to host-allocated memory
    *num_groups = num_pixel_groups;
    *num_valid = num_valid_local;
    
    // Allocate host memory for results
    cudaMallocHost(group_starts, num_pixel_groups * sizeof(int));
    cudaMallocHost(group_sizes, num_pixel_groups * sizeof(int));
    cudaMallocHost(sorted_pixel_hashes, num_valid_local * sizeof(uint64_t));
    cudaMallocHost(sorted_depths, num_valid_local * sizeof(float));
    cudaMallocHost(sorted_indices, num_valid_local * sizeof(int));
    
    // Copy data from device to host
    cudaMemcpy(*group_starts, thrust::raw_pointer_cast(group_starts_vec.data()), num_pixel_groups * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(*group_sizes, thrust::raw_pointer_cast(group_sizes_vec.data()), num_pixel_groups * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(*sorted_pixel_hashes, thrust::raw_pointer_cast(pixel_hashes.data()), num_valid_local * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(*sorted_depths, thrust::raw_pointer_cast(depths.data()), num_valid_local * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(*sorted_indices, thrust::raw_pointer_cast(indices.data()), num_valid_local * sizeof(int), cudaMemcpyDeviceToHost);
    
    return cudaGetLastError();
}

// Debug function: Extract cluster assignments after stage 7 (depth clustering)
cudaError_t extract_cluster_assignments_step7(
    const float* means_cam,        // [N, 3] - camera coordinates  
    const float* pixel_coords,     // [N, 2] - 2D pixel coordinates
    const int* candidate_indices,  // [M] - original indices of candidates
    int num_candidates,
    const ClusteringConfig& config,
    int** group_starts,            // [num_groups] - start index of each group
    int** group_sizes,             // [num_groups] - size of each group
    uint64_t** sorted_pixel_hashes,// [num_valid] - sorted pixel hashes
    float** sorted_depths,         // [num_valid] - sorted depths
    int** sorted_indices,          // [num_valid] - sorted original indices
    int** cluster_assignments,     // [num_valid] - cluster ID for each candidate (-1 if not clustered)
    int* num_groups,               // output: number of pixel groups
    int* num_valid,                // output: number of valid candidates
    int* total_clusters            // output: total number of clusters assigned
) {
    if (num_candidates < config.min_cluster_size) {
        *num_groups = 0;
        *num_valid = 0;
        *total_clusters = 0;
        return cudaSuccess;
    }
    
    // Stage 1: Allocate temporary device memory
    thrust::device_vector<uint64_t> pixel_hashes(num_candidates);
    thrust::device_vector<float> depths(num_candidates);
    thrust::device_vector<int> indices(num_candidates);
    thrust::device_vector<int> valid_count(1);
    
    // Stage 2: Create sort keys
    dim3 block(256);
    dim3 grid((num_candidates + block.x - 1) / block.x);
    
    create_sort_keys_kernel<<<grid, block>>>(
        reinterpret_cast<const int*>(pixel_coords),
        means_cam, 
        candidate_indices, 
        num_candidates,
        thrust::raw_pointer_cast(pixel_hashes.data()),
        thrust::raw_pointer_cast(depths.data()),
        thrust::raw_pointer_cast(indices.data()),
        thrust::raw_pointer_cast(valid_count.data())
    );
    cudaDeviceSynchronize();
    
    int num_valid_local = valid_count[0];
    if (num_valid_local < config.min_cluster_size) {
        *num_groups = 0;
        *num_valid = 0;
        *total_clusters = 0;
        return cudaSuccess;
    }
    
    // Stage 3: Data validation/filtering
    auto valid_end = thrust::remove_if(
        thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.begin(), depths.begin(), indices.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.end(), depths.end(), indices.end())),
        [=] __device__ (const thrust::tuple<uint64_t, float, int>& t) {
            return thrust::get<0>(t) == UINT64_MAX;
        }
    );
    
    num_valid_local = valid_end - thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.begin(), depths.begin(), indices.begin()));
    
    // Stage 4: Sorting by pixel hash and depth
    thrust::sort(
        thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.begin(), depths.begin(), indices.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.begin() + num_valid_local, depths.begin() + num_valid_local, indices.begin() + num_valid_local)),
        [=] __device__ (const thrust::tuple<uint64_t, float, int>& a, const thrust::tuple<uint64_t, float, int>& b) {
            if (thrust::get<0>(a) != thrust::get<0>(b)) {
                return thrust::get<0>(a) < thrust::get<0>(b);
            }
            return thrust::get<1>(a) < thrust::get<1>(b); // Sort by depth within pixel
        }
    );
    
    // Stage 5: Compute pixel group boundaries  
    thrust::device_vector<int> group_starts_vec, group_sizes_vec;
    int num_pixel_groups;
    
    compute_pixel_group_boundaries(pixel_hashes, num_valid_local, group_starts_vec, group_sizes_vec, num_pixel_groups);
    
    if (num_pixel_groups == 0) {
        *num_groups = 0;
        *num_valid = 0;
        *total_clusters = 0;
        return cudaSuccess;
    }
    
    // Stage 6: Setup clustering kernel
    thrust::device_vector<int> cluster_assignments_vec(num_valid_local, -1);
    thrust::device_vector<int> cluster_counter(1, 0);
    
    int max_group_size = *thrust::max_element(group_sizes_vec.begin(), group_sizes_vec.end());
    size_t shared_mem_size = max_group_size * (sizeof(int) + sizeof(float)); // temp_cluster_ids + shared_depths
    
    dim3 grid_opt(num_pixel_groups);
    dim3 block_opt(std::min(256, max_group_size)); // Adjust block size based on group size
    
    // Stage 7: Main clustering kernel execution  
    optimized_pixel_clustering_kernel<<<grid_opt, block_opt, shared_mem_size>>>(
        thrust::raw_pointer_cast(pixel_hashes.data()),
        thrust::raw_pointer_cast(depths.data()),
        thrust::raw_pointer_cast(indices.data()),
        thrust::raw_pointer_cast(group_starts_vec.data()),
        thrust::raw_pointer_cast(group_sizes_vec.data()),
        num_pixel_groups,
        config.depth_threshold,
        config.min_cluster_size,
        thrust::raw_pointer_cast(cluster_assignments_vec.data()),
        thrust::raw_pointer_cast(cluster_counter.data())
    );
    cudaDeviceSynchronize();
    
    int total_clusters_local = cluster_counter[0];
    
    // Copy results to host-allocated memory
    *num_groups = num_pixel_groups;
    *num_valid = num_valid_local;
    *total_clusters = total_clusters_local;
    
    // Allocate host memory for results
    cudaMallocHost(group_starts, num_pixel_groups * sizeof(int));
    cudaMallocHost(group_sizes, num_pixel_groups * sizeof(int));
    cudaMallocHost(sorted_pixel_hashes, num_valid_local * sizeof(uint64_t));
    cudaMallocHost(sorted_depths, num_valid_local * sizeof(float));
    cudaMallocHost(sorted_indices, num_valid_local * sizeof(int));
    cudaMallocHost(cluster_assignments, num_valid_local * sizeof(int));
    
    // Copy data from device to host
    cudaMemcpy(*group_starts, thrust::raw_pointer_cast(group_starts_vec.data()), num_pixel_groups * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(*group_sizes, thrust::raw_pointer_cast(group_sizes_vec.data()), num_pixel_groups * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(*sorted_pixel_hashes, thrust::raw_pointer_cast(pixel_hashes.data()), num_valid_local * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(*sorted_depths, thrust::raw_pointer_cast(depths.data()), num_valid_local * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(*sorted_indices, thrust::raw_pointer_cast(indices.data()), num_valid_local * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(*cluster_assignments, thrust::raw_pointer_cast(cluster_assignments_vec.data()), num_valid_local * sizeof(int), cudaMemcpyDeviceToHost);
    
    return cudaGetLastError();
}

// Memory management
cudaError_t allocate_cluster_result(ClusterResult& result, int max_gaussians, int max_clusters) {
    cudaMalloc(&result.cluster_indices, max_gaussians * sizeof(int));
    cudaMalloc(&result.cluster_sizes, max_clusters * sizeof(int));
    cudaMalloc(&result.cluster_offsets, (max_clusters + 1) * sizeof(int));
    return cudaGetLastError();
}

void free_cluster_result(ClusterResult& result) {
    if (result.cluster_indices) cudaFree(result.cluster_indices);
    if (result.cluster_sizes) cudaFree(result.cluster_sizes);
    if (result.cluster_offsets) cudaFree(result.cluster_offsets);
    
    result.cluster_indices = nullptr;
    result.cluster_sizes = nullptr;
    result.cluster_offsets = nullptr;
    result.num_clusters = 0;
    result.total_clustered = 0;
}

} // namespace stream
