#include <iostream>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "include/clustering.cuh"
#include "include/timer.h"

using namespace std;

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
    cout << "--------------------------------" << endl;
    cout << "Entering ext.cpp" << endl;
    StopWatch sw;
    sw.Restart();

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
    cout << "Preproc CUDA kernel time: " << sw.ElapsedMs() << " ms" << endl;
    
    sw.Restart();
    // Call CUDA function
    cudaError_t cuda_err = stream::cluster_center_in_pixel_cuda(
        means_cam.data_ptr<float>(),
        reinterpret_cast<const float*>(pixel_coords.data_ptr<int>()), // Convert int* to float* for CUDA interface
        candidate_indices.data_ptr<int>(),
        num_candidates,
        config,
        result
    );
    cout << "Clustering CUDA kernel time: " << sw.ElapsedMs() << " ms" << endl;

    sw.Restart();
    TORCH_CHECK(cuda_err == cudaSuccess, "CUDA error in cluster_center_in_pixel: ", cudaGetErrorString(cuda_err));
    
    // Convert result to tensors
    auto tensor_result = convert_cluster_result_to_tensors(result);
    
    // Clean up device memory
    stream::free_cluster_result(result);
    cout << "Postproc CUDA kernel time: " << sw.ElapsedMs() << " ms" << endl;
    cout << "--------------------------------" << endl;
    
    return tensor_result;
}

// Debug function: Extract pixel groups after step 2 (grouping + sorting)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int> extract_pixel_groups_step2(
    const torch::Tensor& means_cam,
    const torch::Tensor& pixel_coords,
    const torch::Tensor& candidate_indices,
    const torch::Tensor& viewmat,
    const torch::Tensor& K,
    int width,
    int height,
    float depth_threshold,
    int min_cluster_size
) {
    // Input validation
    TORCH_CHECK(means_cam.is_cuda(), "means_cam must be on CUDA");
    TORCH_CHECK(pixel_coords.is_cuda(), "pixel_coords must be on CUDA");
    TORCH_CHECK(candidate_indices.is_cuda(), "candidate_indices must be on CUDA");
    TORCH_CHECK(pixel_coords.dtype() == torch::kInt32, "pixel_coords must be int32 (discrete coordinates)");
    
    int num_candidates = means_cam.size(0);
    
    // Setup configuration
    stream::ClusteringConfig config;
    config.depth_threshold = depth_threshold;
    config.min_cluster_size = min_cluster_size;
    config.width = width;
    config.height = height;
    
    // Call CUDA function
    int* group_starts = nullptr;
    int* group_sizes = nullptr;
    uint64_t* sorted_pixel_hashes = nullptr;
    float* sorted_depths = nullptr;
    int* sorted_indices = nullptr;
    int num_groups = 0;
    int num_valid = 0;
    
    cudaError_t cuda_err = stream::extract_pixel_groups_step2(
        means_cam.data_ptr<float>(),
        reinterpret_cast<const float*>(pixel_coords.data_ptr<int>()),
        candidate_indices.data_ptr<int>(),
        num_candidates,
        config,
        &group_starts,
        &group_sizes,
        &sorted_pixel_hashes,
        &sorted_depths,
        &sorted_indices,
        &num_groups,
        &num_valid
    );
    
    TORCH_CHECK(cuda_err == cudaSuccess, "CUDA error in extract_pixel_groups_step2: ", cudaGetErrorString(cuda_err));
    
    // Convert results to PyTorch tensors
    torch::Tensor group_starts_tensor = torch::empty({0}, torch::kInt32);
    torch::Tensor group_sizes_tensor = torch::empty({0}, torch::kInt32);
    torch::Tensor sorted_pixel_hashes_tensor = torch::empty({0}, torch::kInt64);
    torch::Tensor sorted_depths_tensor = torch::empty({0}, torch::kFloat32);
    torch::Tensor sorted_indices_tensor = torch::empty({0}, torch::kInt32);
    
    if (num_groups > 0 && num_valid > 0) {
        // Create tensors from host memory
        group_starts_tensor = torch::from_blob(group_starts, {num_groups}, torch::kInt32).clone();
        group_sizes_tensor = torch::from_blob(group_sizes, {num_groups}, torch::kInt32).clone();
        sorted_pixel_hashes_tensor = torch::from_blob(sorted_pixel_hashes, {num_valid}, torch::kInt64).clone();
        sorted_depths_tensor = torch::from_blob(sorted_depths, {num_valid}, torch::kFloat32).clone();
        sorted_indices_tensor = torch::from_blob(sorted_indices, {num_valid}, torch::kInt32).clone();
        
        // Free host memory
        cudaFreeHost(group_starts);
        cudaFreeHost(group_sizes);
        cudaFreeHost(sorted_pixel_hashes);
        cudaFreeHost(sorted_depths);
        cudaFreeHost(sorted_indices);
    }
    
    return std::make_tuple(group_starts_tensor, group_sizes_tensor, sorted_pixel_hashes_tensor, sorted_depths_tensor, sorted_indices_tensor, num_groups, num_valid);
}

// Debug function: Extract cluster assignments after stage 7 (depth clustering)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int> extract_cluster_assignments_step7(
    const torch::Tensor& means_cam,
    const torch::Tensor& pixel_coords,
    const torch::Tensor& candidate_indices,
    const torch::Tensor& viewmat,
    const torch::Tensor& K,
    int width,
    int height,
    float depth_threshold,
    int min_cluster_size
) {
    // Input validation
    TORCH_CHECK(means_cam.is_cuda(), "means_cam must be on CUDA");
    TORCH_CHECK(pixel_coords.is_cuda(), "pixel_coords must be on CUDA");
    TORCH_CHECK(candidate_indices.is_cuda(), "candidate_indices must be on CUDA");
    TORCH_CHECK(pixel_coords.dtype() == torch::kInt32, "pixel_coords must be int32 (discrete coordinates)");
    
    int num_candidates = means_cam.size(0);
    
    // Setup configuration
    stream::ClusteringConfig config;
    config.depth_threshold = depth_threshold;
    config.min_cluster_size = min_cluster_size;
    config.width = width;
    config.height = height;
    
    // Call CUDA function
    int* group_starts = nullptr;
    int* group_sizes = nullptr;
    uint64_t* sorted_pixel_hashes = nullptr;
    float* sorted_depths = nullptr;
    int* sorted_indices = nullptr;
    int* cluster_assignments = nullptr;
    int num_groups = 0;
    int num_valid = 0;
    int total_clusters = 0;
    
    cudaError_t cuda_err = stream::extract_cluster_assignments_step7(
        means_cam.data_ptr<float>(),
        reinterpret_cast<const float*>(pixel_coords.data_ptr<int>()),
        candidate_indices.data_ptr<int>(),
        num_candidates,
        config,
        &group_starts,
        &group_sizes,
        &sorted_pixel_hashes,
        &sorted_depths,
        &sorted_indices,
        &cluster_assignments,
        &num_groups,
        &num_valid,
        &total_clusters
    );
    
    TORCH_CHECK(cuda_err == cudaSuccess, "CUDA error in extract_cluster_assignments_step7: ", cudaGetErrorString(cuda_err));
    
    // Convert results to PyTorch tensors
    torch::Tensor group_starts_tensor = torch::empty({0}, torch::kInt32);
    torch::Tensor group_sizes_tensor = torch::empty({0}, torch::kInt32);
    torch::Tensor sorted_pixel_hashes_tensor = torch::empty({0}, torch::kInt64);
    torch::Tensor sorted_depths_tensor = torch::empty({0}, torch::kFloat32);
    torch::Tensor sorted_indices_tensor = torch::empty({0}, torch::kInt32);
    torch::Tensor cluster_assignments_tensor = torch::empty({0}, torch::kInt32);
    
    if (num_groups > 0 && num_valid > 0) {
        // Create tensors from host memory
        group_starts_tensor = torch::from_blob(group_starts, {num_groups}, torch::kInt32).clone();
        group_sizes_tensor = torch::from_blob(group_sizes, {num_groups}, torch::kInt32).clone();
        sorted_pixel_hashes_tensor = torch::from_blob(sorted_pixel_hashes, {num_valid}, torch::kInt64).clone();
        sorted_depths_tensor = torch::from_blob(sorted_depths, {num_valid}, torch::kFloat32).clone();
        sorted_indices_tensor = torch::from_blob(sorted_indices, {num_valid}, torch::kInt32).clone();
        cluster_assignments_tensor = torch::from_blob(cluster_assignments, {num_valid}, torch::kInt32).clone();
        
        // Free host memory
        cudaFreeHost(group_starts);
        cudaFreeHost(group_sizes);
        cudaFreeHost(sorted_pixel_hashes);
        cudaFreeHost(sorted_depths);
        cudaFreeHost(sorted_indices);
        cudaFreeHost(cluster_assignments);
    }
    
    return std::make_tuple(group_starts_tensor, group_sizes_tensor, sorted_pixel_hashes_tensor, sorted_depths_tensor, sorted_indices_tensor, cluster_assignments_tensor, num_groups, num_valid, total_clusters);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster_center_in_pixel", &cluster_center_in_pixel, 
          "Center-in-pixel clustering using CUDA");
    m.def("extract_pixel_groups_step2", &extract_pixel_groups_step2,
          "Extract pixel groups after step 2 (grouping + sorting) for debugging");
    m.def("extract_cluster_assignments_step7", &extract_cluster_assignments_step7,
          "Extract cluster assignments after step 7 (depth clustering) for debugging");
}
