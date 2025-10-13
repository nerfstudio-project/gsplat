#ifndef STREAM_CLUSTERING_CUH
#define STREAM_CLUSTERING_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace stream {

struct ClusteringConfig {
    float depth_threshold;
    int min_cluster_size;
    int width;
    int height;
};

struct ClusterResult {
    int* cluster_indices;      // Original indices of Gaussians in clusters
    int* cluster_sizes;        // Size of each cluster
    int* cluster_offsets;      // Starting offset of each cluster in cluster_indices
    int num_clusters;          // Total number of valid clusters
    int total_clustered;       // Total number of Gaussians in all clusters
};

// Main clustering function
cudaError_t cluster_center_in_pixel_cuda(
    const float* means_cam,        // [N, 3] - camera coordinates
    const float* pixel_coords,     // [N, 2] - 2D pixel coordinates
    const int* candidate_indices,  // [M] - original indices of candidates
    int num_candidates,
    const ClusteringConfig& config,
    ClusterResult& result
);

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
);

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
);

// Helper functions
cudaError_t compute_pixel_coordinates(
    const float* means_cam,    // [N, 3]
    const float* viewmat,      // [4, 4] - view matrix
    const float* K,            // [3, 3] - intrinsic matrix

    int num_gaussians,
    int width,
    int height,
    float* pixel_coords,       // [N, 2] - output
    bool* valid_mask          // [N] - output
);

// Memory management helpers
cudaError_t allocate_cluster_result(ClusterResult& result, int max_gaussians, int max_clusters);
void free_cluster_result(ClusterResult& result);

} // namespace stream

#endif // STREAM_CLUSTERING_CUH
