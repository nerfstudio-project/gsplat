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
