#include "../include/merging.cuh"
#include "../include/cuda_timer.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

namespace stream {

// CUDA kernel: Merge clusters using weighted mean strategy
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
) {
    int cluster_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_id >= num_clusters) return;
    
    int start = cluster_offsets[cluster_id];
    int end = cluster_offsets[cluster_id + 1];
    int cluster_size = end - start;
    
    if (cluster_size == 0) return;
    
    // Initialize accumulators
    float3 mean_acc = {0.0f, 0.0f, 0.0f};
    float4 quat_acc = {0.0f, 0.0f, 0.0f, 0.0f};
    float3 scale_acc = {0.0f, 0.0f, 0.0f};
    float opacity_sum = 0.0f;
    float total_weight = 0.0f;
    
    // First pass: compute weights and accumulate weighted sums
    for (int i = start; i < end; i++) {
        int idx = cluster_indices[i];
        
        // Determine weight
        float weight = weight_by_opacity ? opacities[idx] : 1.0f;
        total_weight += weight;
        
        // Accumulate weighted sums
        mean_acc.x += means[idx * 3 + 0] * weight;
        mean_acc.y += means[idx * 3 + 1] * weight;
        mean_acc.z += means[idx * 3 + 2] * weight;
        
        quat_acc.x += quats[idx * 4 + 0] * weight;
        quat_acc.y += quats[idx * 4 + 1] * weight;
        quat_acc.z += quats[idx * 4 + 2] * weight;
        quat_acc.w += quats[idx * 4 + 3] * weight;
        
        scale_acc.x += scales[idx * 3 + 0] * weight;
        scale_acc.y += scales[idx * 3 + 1] * weight;
        scale_acc.z += scales[idx * 3 + 2] * weight;
        
        // Accumulate opacity (sum, not weighted)
        opacity_sum += opacities[idx];
    }
    
    // Avoid division by zero
    if (total_weight == 0.0f) {
        total_weight = 1.0f;
    }
    
    // Compute final weighted averages
    merged_means[cluster_id * 3 + 0] = mean_acc.x / total_weight;
    merged_means[cluster_id * 3 + 1] = mean_acc.y / total_weight;
    merged_means[cluster_id * 3 + 2] = mean_acc.z / total_weight;
    
    // Normalize quaternion
    float quat_norm = sqrtf(quat_acc.x * quat_acc.x + quat_acc.y * quat_acc.y + 
                           quat_acc.z * quat_acc.z + quat_acc.w * quat_acc.w);
    if (quat_norm > 0.0f) {
        merged_quats[cluster_id * 4 + 0] = quat_acc.x / quat_norm;
        merged_quats[cluster_id * 4 + 1] = quat_acc.y / quat_norm;
        merged_quats[cluster_id * 4 + 2] = quat_acc.z / quat_norm;
        merged_quats[cluster_id * 4 + 3] = quat_acc.w / quat_norm;
    } else {
        // Fallback to identity quaternion
        merged_quats[cluster_id * 4 + 0] = 0.0f;
        merged_quats[cluster_id * 4 + 1] = 0.0f;
        merged_quats[cluster_id * 4 + 2] = 0.0f;
        merged_quats[cluster_id * 4 + 3] = 1.0f;
    }
    
    merged_scales[cluster_id * 3 + 0] = scale_acc.x / total_weight;
    merged_scales[cluster_id * 3 + 1] = scale_acc.y / total_weight;
    merged_scales[cluster_id * 3 + 2] = scale_acc.z / total_weight;
    
    // Clamp opacity to [0, 1] (sum in linear space)
    merged_opacities[cluster_id] = fminf(opacity_sum, 1.0f);
    
    // Merge colors (weighted average)
    for (int c = 0; c < color_dim; c++) {
        float color_acc = 0.0f;
        float color_weight_sum = 0.0f;
        
        for (int i = start; i < end; i++) {
            int idx = cluster_indices[i];
            float weight = weight_by_opacity ? opacities[idx] : 1.0f;
            color_acc += colors[idx * color_dim + c] * weight;
            color_weight_sum += weight;
        }
        
        merged_colors[cluster_id * color_dim + c] = (color_weight_sum > 0.0f) ? 
            color_acc / color_weight_sum : 0.0f;
    }
}

// CUDA kernel: Merge clusters using moment matching strategy
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
) {
    int cluster_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster_id >= num_clusters) return;
    
    int start = cluster_offsets[cluster_id];
    int end = cluster_offsets[cluster_id + 1];
    int cluster_size = end - start;
    
    if (cluster_size == 0) return;
    
    // First pass: compute total opacity for normalization
    float total_opacity = 0.0f;
    for (int i = start; i < end; i++) {
        int idx = cluster_indices[i];
        total_opacity += opacities[idx];
    }
    
    if (total_opacity == 0.0f) {
        total_opacity = 1.0f; // Avoid division by zero
    }
    
    // Initialize accumulators
    float3 mean_acc = {0.0f, 0.0f, 0.0f};
    float4 quat_acc = {0.0f, 0.0f, 0.0f, 0.0f};
    float3 scale_acc = {0.0f, 0.0f, 0.0f};
    float opacity_sum = 0.0f;
    float total_volume = 0.0f;
    
    // Second pass: accumulate weighted values
    for (int i = start; i < end; i++) {
        int idx = cluster_indices[i];
        
        // Use opacity as normalized weight
        float weight = opacities[idx] / total_opacity;
        
        // Accumulate weighted sums
        mean_acc.x += means[idx * 3 + 0] * weight;
        mean_acc.y += means[idx * 3 + 1] * weight;
        mean_acc.z += means[idx * 3 + 2] * weight;
        
        quat_acc.x += quats[idx * 4 + 0] * weight;
        quat_acc.y += quats[idx * 4 + 1] * weight;
        quat_acc.z += quats[idx * 4 + 2] * weight;
        quat_acc.w += quats[idx * 4 + 3] * weight;
        
        if (!preserve_volume) {
            scale_acc.x += scales[idx * 3 + 0] * weight;
            scale_acc.y += scales[idx * 3 + 1] * weight;
            scale_acc.z += scales[idx * 3 + 2] * weight;
        }
        
        // Accumulate opacity (sum, not weighted)
        opacity_sum += opacities[idx];
        
        // For volume preservation, accumulate weighted volume
        if (preserve_volume) {
            float volume = scales[idx * 3 + 0] * scales[idx * 3 + 1] * scales[idx * 3 + 2];
            total_volume += opacities[idx] * volume;
        }
    }
    
    // Compute final values
    merged_means[cluster_id * 3 + 0] = mean_acc.x;
    merged_means[cluster_id * 3 + 1] = mean_acc.y;
    merged_means[cluster_id * 3 + 2] = mean_acc.z;
    
    // Normalize quaternion
    float quat_norm = sqrtf(quat_acc.x * quat_acc.x + quat_acc.y * quat_acc.y + 
                           quat_acc.z * quat_acc.z + quat_acc.w * quat_acc.w);
    if (quat_norm > 0.0f) {
        merged_quats[cluster_id * 4 + 0] = quat_acc.x / quat_norm;
        merged_quats[cluster_id * 4 + 1] = quat_acc.y / quat_norm;
        merged_quats[cluster_id * 4 + 2] = quat_acc.z / quat_norm;
        merged_quats[cluster_id * 4 + 3] = quat_acc.w / quat_norm;
    } else {
        // Fallback to identity quaternion
        merged_quats[cluster_id * 4 + 0] = 0.0f;
        merged_quats[cluster_id * 4 + 1] = 0.0f;
        merged_quats[cluster_id * 4 + 2] = 0.0f;
        merged_quats[cluster_id * 4 + 3] = 1.0f;
    }
    
    // Handle scales based on preserve_volume flag
    if (preserve_volume) {
        // Calculate radius for spherical Gaussian that preserves total volume
        const float pi = 3.14159265359f;
        float avg_radius = cbrtf(total_volume / ((4.0f / 3.0f) * pi));
        
        merged_scales[cluster_id * 3 + 0] = avg_radius;
        merged_scales[cluster_id * 3 + 1] = avg_radius;
        merged_scales[cluster_id * 3 + 2] = avg_radius;
    } else {
        merged_scales[cluster_id * 3 + 0] = scale_acc.x;
        merged_scales[cluster_id * 3 + 1] = scale_acc.y;
        merged_scales[cluster_id * 3 + 2] = scale_acc.z;
    }
    
    // Clamp opacity to [0, 1] (sum in linear space)
    merged_opacities[cluster_id] = fminf(opacity_sum, 1.0f);
    
    // Merge colors (weighted average using normalized opacity weights)
    for (int c = 0; c < color_dim; c++) {
        float color_acc = 0.0f;
        
        for (int i = start; i < end; i++) {
            int idx = cluster_indices[i];
            float weight = opacities[idx] / total_opacity;
            color_acc += colors[idx * color_dim + c] * weight;
        }
        
        merged_colors[cluster_id * color_dim + c] = color_acc;
    }
}

// Main CUDA merging function
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
) {
    CudaTimer timer(true);
    timer.reserve_events(5);
    
    if (num_clusters == 0) {
        result.num_merged = 0;
        return cudaSuccess;
    }
    
    // Allocate output memory
    timer.start_stage("1. Memory Allocation");
    cudaError_t err = allocate_merge_result(result, num_clusters, color_dim);
    if (err != cudaSuccess) return err;
    timer.end_stage();
    
    // Configure kernel launch parameters
    timer.start_stage("2. Kernel Setup");
    int threads_per_block = 256;
    int blocks = (num_clusters + threads_per_block - 1) / threads_per_block;
    timer.end_stage();
    
    // Launch appropriate kernel based on strategy
    timer.start_stage("3. CUDA Kernel Execution");
    if (config.strategy == MergeStrategy::WEIGHTED_MEAN) {
        merge_weighted_mean_kernel<<<blocks, threads_per_block>>>(
            cluster_indices, cluster_offsets, num_clusters,
            means, quats, scales, opacities, colors, color_dim,
            config.weight_by_opacity,
            result.merged_means, result.merged_quats, result.merged_scales,
            result.merged_opacities, result.merged_colors
        );
    } else if (config.strategy == MergeStrategy::MOMENT_MATCHING) {
        merge_moment_matching_kernel<<<blocks, threads_per_block>>>(
            cluster_indices, cluster_offsets, num_clusters,
            means, quats, scales, opacities, colors, color_dim,
            config.preserve_volume,
            result.merged_means, result.merged_quats, result.merged_scales,
            result.merged_opacities, result.merged_colors
        );
    } else {
        return cudaErrorInvalidValue;
    }
    
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    timer.end_stage();
    
    // Finalize result
    timer.start_stage("4. Result Finalization");
    result.num_merged = num_clusters;
    result.color_dim = color_dim;
    timer.end_stage();
    
    // Print detailed timing
    timer.print_results("🔥 CUDA Merging Microbenchmark 🔥");
    
    return err;
}

// Memory management helpers
cudaError_t allocate_merge_result(MergeResult& result, int num_clusters, int color_dim) {
    cudaError_t err;
    
    // Allocate device memory for merged Gaussians
    err = cudaMalloc(&result.merged_means, num_clusters * 3 * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&result.merged_quats, num_clusters * 4 * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&result.merged_scales, num_clusters * 3 * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&result.merged_opacities, num_clusters * sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMalloc(&result.merged_colors, num_clusters * color_dim * sizeof(float));
    if (err != cudaSuccess) return err;
    
    result.color_dim = color_dim;
    
    return cudaSuccess;
}

void free_merge_result(MergeResult& result) {
    if (result.merged_means) cudaFree(result.merged_means);
    if (result.merged_quats) cudaFree(result.merged_quats);
    if (result.merged_scales) cudaFree(result.merged_scales);
    if (result.merged_opacities) cudaFree(result.merged_opacities);
    if (result.merged_colors) cudaFree(result.merged_colors);
    
    result.merged_means = nullptr;
    result.merged_quats = nullptr;
    result.merged_scales = nullptr;
    result.merged_opacities = nullptr;
    result.merged_colors = nullptr;
    result.num_merged = 0;
    result.color_dim = 0;
}

} // namespace stream
