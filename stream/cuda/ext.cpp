#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "include/clustering.cuh"

// Convert ClusterResult to Python-compatible format
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int, int> 
convert_cluster_result_to_tensors(const stream::ClusterResult& result) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    
    if (result.total_clustered == 0) {
        // Return empty tensors
        return std::make_tuple(
            torch::empty({0}, options),  // cluster_indices
            torch::empty({0}, options),  // cluster_sizes
            torch::empty({0}, options),  // cluster_offsets
            0,                           // num_clusters
            0                            // total_clustered
        );
    }
    
    // Create tensors from device pointers
    torch::Tensor cluster_indices = torch::from_blob(
        result.cluster_indices, 
        {result.total_clustered}, 
        options
    ).clone(); // Clone to ensure tensor owns the memory
    
    torch::Tensor cluster_sizes = torch::from_blob(
        result.cluster_sizes,
        {result.num_clusters},
        options
    ).clone();
    
    torch::Tensor cluster_offsets = torch::from_blob(
        result.cluster_offsets,
        {result.num_clusters + 1},
        options
    ).clone();
    
    return std::make_tuple(
        cluster_indices,
        cluster_sizes,
        cluster_offsets,
        result.num_clusters,
        result.total_clustered
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int, int>
cluster_center_in_pixel(
    torch::Tensor means_cam,           // [M, 3] - candidate means in camera coords
    torch::Tensor pixel_coords,        // [M, 2] - candidate pixel coordinates
    torch::Tensor candidate_indices,   // [M] - original indices
    torch::Tensor viewmat,             // [4, 4] - view matrix
    torch::Tensor K,                   // [3, 3] - intrinsic matrix
    int width,
    int height,
    float depth_threshold,
    int min_cluster_size
) {
    // Validate inputs
    TORCH_CHECK(means_cam.is_cuda(), "means_cam must be on CUDA device");
    TORCH_CHECK(pixel_coords.is_cuda(), "pixel_coords must be on CUDA device");
    TORCH_CHECK(candidate_indices.is_cuda(), "candidate_indices must be on CUDA device");
    TORCH_CHECK(viewmat.is_cuda(), "viewmat must be on CUDA device");
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA device");
    
    TORCH_CHECK(means_cam.dtype() == torch::kFloat32, "means_cam must be float32");
    TORCH_CHECK(pixel_coords.dtype() == torch::kInt32, "pixel_coords must be int32 (discrete coordinates)");
    TORCH_CHECK(candidate_indices.dtype() == torch::kInt32, "candidate_indices must be int32");
    TORCH_CHECK(viewmat.dtype() == torch::kFloat32, "viewmat must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    
    TORCH_CHECK(means_cam.dim() == 2 && means_cam.size(1) == 3, "means_cam must be [M, 3]");
    TORCH_CHECK(pixel_coords.dim() == 2 && pixel_coords.size(1) == 2, "pixel_coords must be [M, 2]");
    TORCH_CHECK(candidate_indices.dim() == 1, "candidate_indices must be [M]");
    TORCH_CHECK(viewmat.dim() == 2 && viewmat.size(0) == 4 && viewmat.size(1) == 4, "viewmat must be [4, 4]");
    TORCH_CHECK(K.dim() == 2 && K.size(0) == 3 && K.size(1) == 3, "K must be [3, 3]");
    
    int num_candidates = means_cam.size(0);
    TORCH_CHECK(pixel_coords.size(0) == num_candidates, "pixel_coords size mismatch");
    TORCH_CHECK(candidate_indices.size(0) == num_candidates, "candidate_indices size mismatch");
    
    // Prepare configuration
    stream::ClusteringConfig config;
    config.depth_threshold = depth_threshold;
    config.min_cluster_size = min_cluster_size;
    config.width = width;
    config.height = height;
    
    // Prepare result structure
    stream::ClusterResult result;
    result.cluster_indices = nullptr;
    result.cluster_sizes = nullptr;
    result.cluster_offsets = nullptr;
    result.num_clusters = 0;
    result.total_clustered = 0;
    
    // Call CUDA function
    cudaError_t cuda_err = stream::cluster_center_in_pixel_cuda(
        means_cam.data_ptr<float>(),
        reinterpret_cast<const float*>(pixel_coords.data_ptr<int>()), // Convert int* to float* for CUDA interface
        candidate_indices.data_ptr<int>(),
        num_candidates,
        config,
        result
    );
    
    TORCH_CHECK(cuda_err == cudaSuccess, "CUDA error in cluster_center_in_pixel: ", cudaGetErrorString(cuda_err));
    
    // Convert result to tensors
    auto tensor_result = convert_cluster_result_to_tensors(result);
    
    // Clean up device memory
    stream::free_cluster_result(result);
    
    return tensor_result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster_center_in_pixel", &cluster_center_in_pixel, 
          "Center-in-pixel clustering using CUDA");
}
