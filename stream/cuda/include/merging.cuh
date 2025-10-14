#ifndef STREAM_MERGING_CUH
#define STREAM_MERGING_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace stream {

enum class MergeStrategy {
    WEIGHTED_MEAN = 0,
    MOMENT_MATCHING = 1
};

struct MergingConfig {
    MergeStrategy strategy;
    bool weight_by_opacity;     // For weighted_mean strategy
    bool preserve_volume;       // For moment_matching strategy
};

struct MergeResult {
    float* merged_means;        // [num_clusters, 3] - merged Gaussian centers
    float* merged_quats;        // [num_clusters, 4] - merged quaternions  
    float* merged_scales;       // [num_clusters, 3] - merged scales
    float* merged_opacities;    // [num_clusters] - merged opacities
    float* merged_colors;       // [num_clusters, color_dim] - merged colors
    int num_merged;             // Number of merged Gaussians (should equal num_clusters)
    int color_dim;              // Color dimension (3 for RGB, or K*3 for SH)
};

// Main merging function - CUDA accelerated cluster merging
cudaError_t merge_clusters_cuda(
    const int* cluster_indices,     // [total_clustered] - flat array of original indices
    const int* cluster_offsets,     // [num_clusters + 1] - cluster boundaries
    int num_clusters,               // Number of clusters to merge
    int total_clustered,            // Total Gaussians in all clusters
    
    // Original Gaussian data
    const float* means,             // [N, 3] - all Gaussian centers
    const float* quats,             // [N, 4] - all Gaussian quaternions
    const float* scales,            // [N, 3] - all Gaussian scales (linear space)
    const float* opacities,         // [N] - all Gaussian opacities (linear space)  
    const float* colors,            // [N, color_dim] - all Gaussian colors
    int num_gaussians,              // Total number of Gaussians (N)
    int color_dim,                  // Color dimension
    
    const MergingConfig& config,    // Merging configuration
    MergeResult& result             // Output - merged Gaussians
);

// Helper: Merge using weighted mean strategy (CUDA kernel)
__global__ void merge_weighted_mean_kernel(
    const int* cluster_indices,     // [total_clustered] - flat indices
    const int* cluster_offsets,     // [num_clusters + 1] - boundaries
    int num_clusters,
    
    const float* means,             // [N, 3] - input Gaussians
    const float* quats,             // [N, 4]
    const float* scales,            // [N, 3] 
    const float* opacities,         // [N]
    const float* colors,            // [N, color_dim]
    int color_dim,
    bool weight_by_opacity,
    
    float* merged_means,            // [num_clusters, 3] - output
    float* merged_quats,            // [num_clusters, 4]
    float* merged_scales,           // [num_clusters, 3]
    float* merged_opacities,        // [num_clusters]
    float* merged_colors            // [num_clusters, color_dim]
);

// Helper: Merge using moment matching strategy (CUDA kernel)
__global__ void merge_moment_matching_kernel(
    const int* cluster_indices,     // [total_clustered] - flat indices
    const int* cluster_offsets,     // [num_clusters + 1] - boundaries
    int num_clusters,
    
    const float* means,             // [N, 3] - input Gaussians
    const float* quats,             // [N, 4]
    const float* scales,            // [N, 3] 
    const float* opacities,         // [N]
    const float* colors,            // [N, color_dim]
    int color_dim,
    bool preserve_volume,
    
    float* merged_means,            // [num_clusters, 3] - output
    float* merged_quats,            // [num_clusters, 4]
    float* merged_scales,           // [num_clusters, 3]
    float* merged_opacities,        // [num_clusters]
    float* merged_colors            // [num_clusters, color_dim]
);

// Memory management helpers
cudaError_t allocate_merge_result(MergeResult& result, int num_clusters, int color_dim);
void free_merge_result(MergeResult& result);

} // namespace stream

#endif // STREAM_MERGING_CUH
