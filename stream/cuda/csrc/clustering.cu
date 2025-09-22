#include "../include/clustering.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <cub/cub.cuh>

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

// Fixed depth clustering kernel with proper atomic operations  
__global__ void depth_cluster_kernel(
    const uint64_t* sorted_pixel_hashes,  // [M] - sorted pixel hashes
    const float* sorted_depths,           // [M] - corresponding sorted depths  
    const int* sorted_indices,            // [M] - corresponding original indices
    int num_valid,
    float depth_threshold,
    int min_cluster_size,
    int* cluster_assignments,             // [M] - output: cluster ID for each candidate (-1 if not clustered)
    int* global_cluster_counter           // [1] - global cluster counter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid) return;
    
    // Initialize all assignments to invalid
    cluster_assignments[idx] = -1;
    
    // Find pixel group boundaries
    bool is_group_start = (idx == 0) || (sorted_pixel_hashes[idx] != sorted_pixel_hashes[idx - 1]);
    
    if (!is_group_start) return; // Only process group starts
    
    // Find group end
    int group_end = idx;
    while (group_end + 1 < num_valid && sorted_pixel_hashes[group_end + 1] == sorted_pixel_hashes[idx]) {
        group_end++;
    }
    int group_size = group_end - idx + 1;
    
    if (group_size < min_cluster_size) return; // Skip small groups
    
    // Perform depth clustering within this pixel group (matches CPU logic exactly)
    int cluster_start = idx;
    int current_cluster_id = -1; // Will assign valid IDs later
    
    for (int i = idx; i <= group_end; i++) {
        if (i == idx) {
            // Start first cluster - use a temporary negative ID
            cluster_assignments[i] = -(idx + 1); // Unique negative ID per group
            current_cluster_id = -(idx + 1);
        } else {
            float depth_diff = fabsf(sorted_depths[i] - sorted_depths[i - 1]);
            
            if (depth_diff <= depth_threshold) {
                // Continue current cluster
                cluster_assignments[i] = current_cluster_id;
            } else {
                // Check if previous cluster was large enough
                int prev_cluster_size = i - cluster_start;
                if (prev_cluster_size < min_cluster_size) {
                    // Mark previous cluster as invalid
                    for (int j = cluster_start; j < i; j++) {
                        cluster_assignments[j] = -1;
                    }
                }
                
                // Start new cluster with new temporary negative ID
                current_cluster_id = -(idx + 1) * 1000 - (i - idx); // Unique negative ID
                cluster_assignments[i] = current_cluster_id;
                cluster_start = i;
            }
        }
    }
    
    // Handle last cluster
    int last_cluster_size = group_end - cluster_start + 1;
    if (last_cluster_size < min_cluster_size) {
        // Mark last cluster as invalid
        for (int j = cluster_start; j <= group_end; j++) {
            cluster_assignments[j] = -1;
        }
    }
}

// Second pass: assign final cluster IDs using thrust operations (much faster)
void assign_final_cluster_ids_thrust(
    thrust::device_vector<int>& cluster_assignments,
    int& total_clusters
) {
    // Find all unique temporary cluster IDs (negative values except -1)
    thrust::device_vector<int> temp_assignments = cluster_assignments;
    
    // Keep only valid temporary IDs (negative but not -1)
    auto new_end = thrust::remove_if(temp_assignments.begin(), temp_assignments.end(),
        [] __device__ (int x) { return x == -1 || x >= 0; });
    temp_assignments.resize(new_end - temp_assignments.begin());
    
    if (temp_assignments.size() == 0) {
        total_clusters = 0;
        return;
    }
    
    // Sort and find unique temporary cluster IDs
    thrust::sort(temp_assignments.begin(), temp_assignments.end());
    auto unique_end = thrust::unique(temp_assignments.begin(), temp_assignments.end());
    temp_assignments.resize(unique_end - temp_assignments.begin());
    
    total_clusters = temp_assignments.size();
    
    // Create mapping from old temp ID to new final ID
    thrust::device_vector<int> old_ids = temp_assignments;
    thrust::device_vector<int> new_ids(total_clusters);
    thrust::sequence(new_ids.begin(), new_ids.end(), 0); // 0, 1, 2, ...
    
    // Replace all occurrences of old IDs with new IDs
    for (int i = 0; i < total_clusters; i++) {
        int old_id = old_ids[i];
        int new_id = new_ids[i];
        
        thrust::replace(cluster_assignments.begin(), cluster_assignments.end(), old_id, new_id);
    }
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
    if (num_candidates < config.min_cluster_size) {
        result.num_clusters = 0;
        result.total_clustered = 0;
        return cudaSuccess;
    }
    
    // Allocate temporary device memory
    thrust::device_vector<uint64_t> pixel_hashes(num_candidates);
    thrust::device_vector<float> depths(num_candidates);
    thrust::device_vector<int> indices(num_candidates);
    thrust::device_vector<int> valid_count(1);
    
    // Create sort keys
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
    
    int num_valid = valid_count[0];
    if (num_valid < config.min_cluster_size) {
        result.num_clusters = 0;
        result.total_clustered = 0;
        return cudaSuccess;
    }
    
    // Remove invalid entries
    auto valid_end = thrust::remove_if(
        thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.begin(), depths.begin(), indices.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.end(), depths.end(), indices.end())),
        [=] __device__ (const thrust::tuple<uint64_t, float, int>& t) {
            return thrust::get<0>(t) == UINT64_MAX;
        }
    );
    
    num_valid = valid_end - thrust::make_zip_iterator(thrust::make_tuple(pixel_hashes.begin(), depths.begin(), indices.begin()));
    
    // Sort by pixel hash, then by depth
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
    
    // Allocate clustering result arrays
    thrust::device_vector<int> cluster_assignments(num_valid, -1);
    thrust::device_vector<int> global_cluster_counter(1, 0);
    
    // First pass: depth clustering with temporary IDs
    grid = dim3((num_valid + block.x - 1) / block.x);
    depth_cluster_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(pixel_hashes.data()),
        thrust::raw_pointer_cast(depths.data()),
        thrust::raw_pointer_cast(indices.data()),
        num_valid,
        config.depth_threshold,
        config.min_cluster_size,
        thrust::raw_pointer_cast(cluster_assignments.data()),
        thrust::raw_pointer_cast(global_cluster_counter.data())
    );
    cudaDeviceSynchronize();
    
    // Second pass: assign final cluster IDs using efficient thrust operations
    int total_clusters;
    assign_final_cluster_ids_thrust(cluster_assignments, total_clusters);
    
    // Copy results back
    result.num_clusters = total_clusters;
    
    // Count total clustered Gaussians
    int total_clustered = thrust::count_if(cluster_assignments.begin(), cluster_assignments.end(),
                                          [] __device__ (int x) { return x >= 0; });
    result.total_clustered = total_clustered;
    
    if (total_clustered > 0) {
        // Allocate result memory
        cudaMalloc(&result.cluster_indices, total_clustered * sizeof(int));
        cudaMalloc(&result.cluster_sizes, total_clusters * sizeof(int));
        cudaMalloc(&result.cluster_offsets, (total_clusters + 1) * sizeof(int));
        
        // Compact and copy clustered indices - extract only the indices with valid cluster assignments
        auto compact_end = thrust::copy_if(
            indices.begin(),
            indices.begin() + num_valid,
            cluster_assignments.begin(),
            thrust::device_pointer_cast(result.cluster_indices),
            [] __device__ (int cluster_assignment) {
                return cluster_assignment >= 0;
            }
        );
        
        // Compute cluster sizes by counting occurrences of each cluster ID
        thrust::device_vector<int> cluster_sizes(total_clusters, 0);
        
        // Count elements in each cluster
        for (int cluster_id = 0; cluster_id < total_clusters; cluster_id++) {
            int count = thrust::count(cluster_assignments.begin(), cluster_assignments.end(), cluster_id);
            cluster_sizes[cluster_id] = count;
        }
        
        // Copy cluster sizes to result
        thrust::copy(cluster_sizes.begin(), cluster_sizes.end(), thrust::device_pointer_cast(result.cluster_sizes));
        
        // Compute cluster offsets
        thrust::exclusive_scan(cluster_sizes.begin(), cluster_sizes.end(), thrust::device_pointer_cast(result.cluster_offsets));
        
        // Set final offset
        cudaMemcpy(result.cluster_offsets + total_clusters, &total_clustered, sizeof(int), cudaMemcpyHostToDevice);
    }
    
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
