#include "compression.h"
#include "utils.h"
#include "timer.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <cstring>

// Functor for bit-shifting to find parent Morton codes
struct ParentShift {
    __host__ __device__
    uint32_t operator()(const uint32_t& code) const {
        return code >> 3;
    }
};

// --- CUDA Error Checking ---
#define checkCudaErrors(val) check_((val), #val, __FILE__, __LINE__)
template<typename T>
void check_(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

// --- Device Functions: Morton Encoding ---

// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
__device__ __host__ uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Inverse of expandBits: extracts coordinates from Morton code
__device__ __host__ uint32_t compactBits(uint32_t v) {
    v &= 0x49249249u; // Keep only every 3rd bit (x: bits 0,3,6,9...)
    v = (v | (v >> 2)) & 0xC30C30C3u;
    v = (v | (v >> 4)) & 0x0F00F00Fu;
    v = (v | (v >> 8)) & 0xFF0000FFu;
    v = (v | (v >> 16)) & 0x0000FFFFu;
    return v;
}

// Extracts x, y, z integer coordinates from Morton code (optimized for voxelized coordinates)
__device__ __host__ void morton3D_decode(uint32_t code, uint32_t& x, uint32_t& y, uint32_t& z) {
    x = compactBits(code);
    y = compactBits(code >> 1);
    z = compactBits(code >> 2);
}

// --- CUDA Kernel: Compute Occupancy ---
__global__ void compute_occupancy_kernel(
    const uint32_t* parents, int num_parents,
    const uint32_t* children, int num_children,
    uint8_t* occupancy_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parents) return;

    uint32_t parent_code = parents[idx];
    uint32_t base_child = parent_code << 3;
    uint8_t occ = 0;

    // Binary search to find the first potential child
    int left = 0;
    int right = num_children;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (children[mid] < base_child) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    // Check contiguous children
    for (int i = 0; left + i < num_children; ++i) {
        uint32_t child_code = children[left + i];
        if (child_code > base_child + 7) break;
        occ |= (1 << (child_code - base_child));
    }

    occupancy_out[idx] = occ;
}

// --- CUDA Kernel: Compute Morton codes (optimized for integer voxelized coordinates) ---
__global__ void compute_morton_codes(const uint32_t* points_x, const uint32_t* points_y, const uint32_t* points_z,
                                     int num_points, 
                                     uint32_t* codes, int* original_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        // Compute Morton code directly (inline computation)
        uint32_t x = points_x[idx];
        uint32_t y = points_y[idx];
        uint32_t z = points_z[idx];
        codes[idx] = expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
        original_indices[idx] = idx;
    }
}

// --- CUDA Kernel: Reconstruct points from octree levels (optimized for integer voxelized coordinates) ---
__global__ void reconstruct_points_kernel(
    const uint32_t* leaf_codes, int num_leaves,
    uint32_t* output_x, uint32_t* output_y, uint32_t* output_z)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_leaves) {
        morton3D_decode(leaf_codes[idx], output_x[idx], output_y[idx], output_z[idx]);
    }
}

// --- Compression Function ---
CompressionResult compress_gpu_octree(const std::vector<Point3D>& points, uint32_t octree_depth) {
    CompressionResult result;
    result.compression_time_ms = 0.0;
    result.compressed_data.clear();

    if (points.empty()) {
        std::cerr << "Error: Empty point cloud" << std::endl;
        return result;
    }

    StopWatch sw;
    int N = points.size();
    // grid_dim = 2^octree_depth, min_b = (0,0,0), max_b = (2^octree_depth-1, 2^octree_depth-1, 2^octree_depth-1) - fixed

    // Convert points to uint32_t format for GPU (coordinates are already integers)
    std::vector<uint32_t> h_points_x(N), h_points_y(N), h_points_z(N);
    for (int i = 0; i < N; ++i) {
        h_points_x[i] = points[i].x;
        h_points_y[i] = points[i].y;
        h_points_z[i] = points[i].z;
    }

    // Upload data to GPU
    thrust::device_vector<uint32_t> d_points_x = h_points_x;
    thrust::device_vector<uint32_t> d_points_y = h_points_y;
    thrust::device_vector<uint32_t> d_points_z = h_points_z;
    thrust::device_vector<uint32_t> d_morton_codes(N);
    thrust::device_vector<int> d_indices(N);

    // Compute Morton codes using integer-based kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    compute_morton_codes<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_points_x.data()),
        thrust::raw_pointer_cast(d_points_y.data()),
        thrust::raw_pointer_cast(d_points_z.data()),
        N,
        thrust::raw_pointer_cast(d_morton_codes.data()),
        thrust::raw_pointer_cast(d_indices.data())
    );
    checkCudaErrors(cudaGetLastError());

    // Sort by Morton code
    thrust::sort_by_key(d_morton_codes.begin(), d_morton_codes.end(), d_indices.begin());

    // Generate all tree levels (bottom-up)
    std::vector<thrust::device_vector<uint32_t>> all_levels(octree_depth + 1);

    // Level octree_depth (Leaves): Unique the sorted point codes
    all_levels[octree_depth] = d_morton_codes;
    auto new_end = thrust::unique(all_levels[octree_depth].begin(), all_levels[octree_depth].end());
    all_levels[octree_depth].resize(new_end - all_levels[octree_depth].begin());

    // Levels (octree_depth-1) down to 0 (Parents):
    for (int d = static_cast<int>(octree_depth) - 1; d >= 0; --d) {
        all_levels[d] = all_levels[d + 1];
        thrust::transform(all_levels[d].begin(), all_levels[d].end(),
                          all_levels[d].begin(),
                          ParentShift());
        auto parent_end = thrust::unique(all_levels[d].begin(), all_levels[d].end());
        all_levels[d].resize(parent_end - all_levels[d].begin());
    }

    // Generate BFS occupancy stream
    thrust::device_vector<uint8_t> d_bfs_stream;
    d_bfs_stream.reserve(N * 2);

    for (uint32_t d = 0; d < octree_depth; ++d) {
        int num_parents = all_levels[d].size();
        int num_children = all_levels[d+1].size();

        thrust::device_vector<uint8_t> d_level_occupancy(num_parents);

        threads = 256;
        blocks = (num_parents + threads - 1) / threads;
        compute_occupancy_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(all_levels[d].data()), num_parents,
            thrust::raw_pointer_cast(all_levels[d+1].data()), num_children,
            thrust::raw_pointer_cast(d_level_occupancy.data())
        );
        checkCudaErrors(cudaGetLastError());

        d_bfs_stream.insert(d_bfs_stream.end(), d_level_occupancy.begin(), d_level_occupancy.end());
    }

    // Copy compressed data to host
    thrust::host_vector<uint8_t> h_bfs_stream = d_bfs_stream;

    // Serialize metadata and compressed data
    // Optimized format for voxelized coordinates: [num_levels (4 bytes)][level_sizes...][bfs_stream...]
    // Note: octree_depth is passed as parameter, not stored. min_b, max_b, and grid_dim are fixed - no need to store
    result.compressed_data.clear();
    
    // Write metadata (reduced size - no octree_depth, bounds or grid_dim needed)
    uint32_t num_levels = octree_depth + 1;
    
    result.compressed_data.resize(sizeof(uint32_t) + sizeof(uint32_t) * num_levels + h_bfs_stream.size());
    size_t offset = 0;
    
    std::memcpy(result.compressed_data.data() + offset, &num_levels, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    
    // Write level sizes
    for (uint32_t d = 0; d <= octree_depth; ++d) {
        uint32_t level_size = all_levels[d].size();
        std::memcpy(result.compressed_data.data() + offset, &level_size, sizeof(uint32_t));
        offset += sizeof(uint32_t);
    }
    
    // Write BFS stream
    std::memcpy(result.compressed_data.data() + offset, h_bfs_stream.data(), h_bfs_stream.size());

    result.compression_time_ms = sw.ElapsedMs();

    return result;
}

// --- Decompression Function ---
DecompressionResult decompress_gpu_octree(const std::vector<uint8_t>& compressed_data, const std::string& output_path, uint32_t octree_depth) {
    DecompressionResult result;
    result.output_path = output_path;
    result.success = false;
    result.decompression_time_ms = 0.0;

    if (compressed_data.empty()) {
        std::cerr << "Error: Empty compressed data" << std::endl;
        return result;
    }

    StopWatch sw;
    size_t offset = 0;

    // Read metadata (optimized format for voxelized coordinates)
    if (compressed_data.size() < sizeof(uint32_t)) {
        std::cerr << "Error: Invalid compressed data format" << std::endl;
        return result;
    }

    uint32_t num_levels;
    std::memcpy(&num_levels, compressed_data.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    
    // Validate that num_levels matches expected octree_depth
    uint32_t expected_num_levels = octree_depth + 1;
    if (num_levels != expected_num_levels) {
        std::cerr << "Error: Octree depth mismatch. Expected " << expected_num_levels 
                  << " levels (depth " << octree_depth << "), but found " << num_levels << " levels" << std::endl;
        return result;
    }

    // Read level sizes
    std::vector<uint32_t> level_sizes(num_levels);
    for (uint32_t i = 0; i < num_levels; ++i) {
        if (offset + sizeof(uint32_t) > compressed_data.size()) {
            std::cerr << "Error: Invalid compressed data format (level sizes)" << std::endl;
            return result;
        }
        std::memcpy(&level_sizes[i], compressed_data.data() + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
    }

    // Read BFS stream
    size_t bfs_stream_size = compressed_data.size() - offset;
    if (bfs_stream_size == 0) {
        std::cerr << "Error: Empty BFS stream" << std::endl;
        return result;
    }

    thrust::host_vector<uint8_t> h_bfs_stream(bfs_stream_size);
    std::memcpy(h_bfs_stream.data(), compressed_data.data() + offset, bfs_stream_size);
    thrust::device_vector<uint8_t> d_bfs_stream = h_bfs_stream;

    // Reconstruct octree levels from BFS stream using GPU kernels
    // We need to reconstruct level by level starting from root
    std::vector<thrust::device_vector<uint32_t>> all_levels(num_levels);
    
    // Start with root (level 0) - single node with code 0
    all_levels[0].resize(1);
    all_levels[0][0] = 0;

    size_t bfs_offset = 0;
    for (uint32_t d = 0; d < octree_depth; ++d) {
        int num_parents = all_levels[d].size();
        if (bfs_offset + num_parents > d_bfs_stream.size()) {
            std::cerr << "Error: BFS stream too short at level " << d << std::endl;
            return result;
        }

        // Copy parents and occupancy to host for CPU processing (simpler and more reliable)
        thrust::host_vector<uint32_t> h_parents = all_levels[d];
        thrust::host_vector<uint8_t> h_level_occupancy(num_parents);
        
        // Copy occupancy bytes from device BFS stream
        thrust::copy(d_bfs_stream.begin() + bfs_offset, 
                     d_bfs_stream.begin() + bfs_offset + num_parents,
                     h_level_occupancy.begin());
        bfs_offset += num_parents;

        // Reconstruct children on CPU (correct and fast for this operation)
        std::vector<uint32_t> h_children;
        h_children.reserve(num_parents * 4); // Average branching factor estimate

        for (int p = 0; p < num_parents; ++p) {
            uint32_t parent_code = h_parents[p];
            uint32_t base_child = parent_code << 3;
            uint8_t occ = h_level_occupancy[p];

            // Expand occupancy bits into child codes
            for (int bit = 0; bit < 8; ++bit) {
                if (occ & (1 << bit)) {
                    h_children.push_back(base_child + bit);
                }
            }
        }

        // Upload children back to GPU
        if (d + 1 < num_levels) {
            all_levels[d + 1] = h_children;
        } else {
            // Last level
            all_levels[d + 1] = h_children;
        }
    }

    // Extract leaf points (last level)
    int num_leaves = all_levels[octree_depth].size();
    if (num_leaves == 0) {
        std::cerr << "Error: No leaf nodes found" << std::endl;
        return result;
    }

    // Reconstruct points from leaf Morton codes using integer-based kernel
    thrust::device_vector<uint32_t> d_reconstructed_x(num_leaves);
    thrust::device_vector<uint32_t> d_reconstructed_y(num_leaves);
    thrust::device_vector<uint32_t> d_reconstructed_z(num_leaves);
    int threads = 256;
    int blocks = (num_leaves + threads - 1) / threads;
    reconstruct_points_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(all_levels[octree_depth].data()),
        num_leaves,
        thrust::raw_pointer_cast(d_reconstructed_x.data()),
        thrust::raw_pointer_cast(d_reconstructed_y.data()),
        thrust::raw_pointer_cast(d_reconstructed_z.data())
    );
    checkCudaErrors(cudaGetLastError());

    // Copy to host
    thrust::host_vector<uint32_t> h_reconstructed_x = d_reconstructed_x;
    thrust::host_vector<uint32_t> h_reconstructed_y = d_reconstructed_y;
    thrust::host_vector<uint32_t> h_reconstructed_z = d_reconstructed_z;

    result.decompression_time_ms = sw.ElapsedMs();

    // Convert to Point3D format
    std::vector<Point3D> points;
    points.reserve(num_leaves);
    for (int i = 0; i < num_leaves; ++i) {
        points.emplace_back(h_reconstructed_x[i], h_reconstructed_y[i], h_reconstructed_z[i]);
    }

    // Save decompressed point cloud using unified PLY writer (file I/O excluded from timing)
    if (!save_ply_geometry(points, output_path)) {
        std::cerr << "Error: Could not save file " << output_path << std::endl;
        return result;
    }

    result.success = true;
    return result;
}

