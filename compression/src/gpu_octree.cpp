#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

// --- Helper Structures ---
struct float3_ { float x, y, z; };

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

// Calculates a 30-bit Morton code for a 3D point.
__device__ __host__ uint32_t morton3D(float3_ p, float3_ min_bound, float3_ max_bound, int grid_dim) {
    float x = (p.x - min_bound.x) / (max_bound.x - min_bound.x);
    float y = (p.y - min_bound.y) / (max_bound.y - min_bound.y);
    float z = (p.z - min_bound.z) / (max_bound.z - min_bound.z);

    x = fminf(fmaxf(x, 0.0f), 1.0f);
    y = fminf(fmaxf(y, 0.0f), 1.0f);
    z = fminf(fmaxf(z, 0.0f), 1.0f);

    uint32_t xx = (uint32_t)(x * (grid_dim - 1));
    uint32_t yy = (uint32_t)(y * (grid_dim - 1));
    uint32_t zz = (uint32_t)(z * (grid_dim - 1));

    return expandBits(xx) | (expandBits(yy) << 1) | (expandBits(zz) << 2);
}

// --- NEW KERNEL: Compute Occupancy ---
// For every parent node, find which of its 8 children exist in the next level.
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

    // 1. Binary search to find the first potential child in the children array.
    // We use a standard lower_bound approach.
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
    
    // 'left' is now the index of the first node >= base_child.
    // 2. Check the next few nodes to see if they match our expected children (base_child + 0..7)
    // Since the 'children' array is sorted, all actual children MUST be contiguous here.
    for (int i = 0; left + i < num_children; ++i) {
        uint32_t child_code = children[left + i];
        
        // If we've gone past the last possible child (base_child + 7), stop early.
        if (child_code > base_child + 7) break;
        
        // If this is one of our children, set the corresponding bit.
        // (child_code - base_child) will be between 0 and 7.
        occ |= (1 << (child_code - base_child));
    }

    occupancy_out[idx] = occ;
}

// --- CUDA Kernel ---
__global__ void compute_morton_codes(const float3_* points, int num_points, 
                                     float3_ min_b, float3_ max_b, int grid_dim, 
                                     uint32_t* codes, int* original_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        codes[idx] = morton3D(points[idx], min_b, max_b, grid_dim);
        original_indices[idx] = idx;
    }
}

// --- Main Host Function ---
int main() {
    // 1. Setup: Generate dummy point cloud
    int N = 1000000;
    
    for(int run = 0; run < 3; ++run) 
    {
        std::cout << "Run " << run + 1 << " of 3" << std::endl;
        std::cout << "Generating " << N << " random points on CPU..." << std::endl;
        std::vector<float3_> h_points(N);
        std::mt19937 gen(1234);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
        for (int i = 0; i < N; ++i) {
            h_points[i] = {dist(gen), dist(gen), dist(gen)};
        }

        float3_ min_b = {1e9, 1e9, 1e9};
        float3_ max_b = {-1e9, -1e9, -1e9};
        for(auto& p : h_points) {
            min_b.x = fminf(min_b.x, p.x); min_b.y = fminf(min_b.y, p.y); min_b.z = fminf(min_b.z, p.z);
            max_b.x = fmaxf(max_b.x, p.x); max_b.y = fmaxf(max_b.y, p.y); max_b.z = fmaxf(max_b.z, p.z);
        }

        int octree_depth = 10;
        int grid_dim = 1 << octree_depth;

        std::cout << "--- Starting GPU Full Octree Construction ---" << std::endl;
        
        // Create timing events
        cudaEvent_t start, stop, temp_start, temp_stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&temp_start);
        cudaEventCreate(&temp_stop);
        float elapsed_ms = 0.0f;

        // Record overall start time
        cudaEventRecord(start);

        // 2. Upload data
        cudaEventRecord(temp_start);
        thrust::device_vector<float3_> d_points = h_points;
        thrust::device_vector<uint32_t> d_morton_codes(N);
        thrust::device_vector<int> d_indices(N);
        cudaEventRecord(temp_stop);
        cudaEventSynchronize(temp_stop);
        cudaEventElapsedTime(&elapsed_ms, temp_start, temp_stop);
        std::cout << "  Data upload: " << elapsed_ms << " ms" << std::endl;

        // 3. Compute Morton codes
        cudaEventRecord(temp_start);
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        compute_morton_codes<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_points.data()), N, 
            min_b, max_b, grid_dim, 
            thrust::raw_pointer_cast(d_morton_codes.data()),
            thrust::raw_pointer_cast(d_indices.data())
        );
        checkCudaErrors(cudaGetLastError());
        cudaEventRecord(temp_stop);
        cudaEventSynchronize(temp_stop);
        cudaEventElapsedTime(&elapsed_ms, temp_start, temp_stop);
        std::cout << "  compute_morton_codes kernel: " << elapsed_ms << " ms" << std::endl;

        // 4. Sort by Morton Key (implicitly groups points by voxel)
        cudaEventRecord(temp_start);
        thrust::sort_by_key(d_morton_codes.begin(), d_morton_codes.end(), d_indices.begin());
        cudaEventRecord(temp_stop);
        cudaEventSynchronize(temp_stop);
        cudaEventElapsedTime(&elapsed_ms, temp_start, temp_stop);
        std::cout << "  thrust::sort_by_key: " << elapsed_ms << " ms" << std::endl;

        // 5. Generate All Tree Levels (Bottom-Up)
        // We store each level in a separate vector. 
        // all_levels[10] = leaves, all_levels[0] = root.
        std::vector<thrust::device_vector<uint32_t>> all_levels(octree_depth + 1);

        // Level 10 (Leaves): Unique the sorted point codes
        cudaEventRecord(temp_start);
        all_levels[octree_depth] = d_morton_codes;
        auto new_end = thrust::unique(all_levels[octree_depth].begin(), all_levels[octree_depth].end());
        all_levels[octree_depth].resize(new_end - all_levels[octree_depth].begin());
        cudaEventRecord(temp_stop);
        cudaEventSynchronize(temp_stop);
        cudaEventElapsedTime(&elapsed_ms, temp_start, temp_stop);
        std::cout << "  Level " << octree_depth << " (leaves) - unique: " << elapsed_ms << " ms" << std::endl;

        // Levels 9 down to 0 (Parents):
        float total_tree_build_time = 0.0f;
        for (int d = octree_depth - 1; d >= 0; --d) {
            cudaEventRecord(temp_start);
            // Initialize current level with children's codes
            all_levels[d] = all_levels[d + 1];

            // Shift children right by 3 to get parent codes
            // No need to re-sort! If children are sorted, their parent codes are also sorted.
            thrust::transform(all_levels[d].begin(), all_levels[d].end(),
                            all_levels[d].begin(),
                            ParentShift());

            // Unique to remove duplicate parents
            auto parent_end = thrust::unique(all_levels[d].begin(), all_levels[d].end());
            all_levels[d].resize(parent_end - all_levels[d].begin());
            cudaEventRecord(temp_stop);
            cudaEventSynchronize(temp_stop);
            cudaEventElapsedTime(&elapsed_ms, temp_start, temp_stop);
            total_tree_build_time += elapsed_ms;
            std::cout << "  Level " << d << " - transform + unique: " << elapsed_ms << " ms" << std::endl;
        }
        std::cout << "  Total tree level generation: " << total_tree_build_time << " ms" << std::endl;

        // --- NEW: BFS Serialization (Occupancy Code Generation) ---
        std::cout << "--- Generating BFS Occupancy Stream on GPU ---" << std::endl;
        
        // This vector will hold the final compressed stream of bytes
        thrust::device_vector<uint8_t> d_bfs_stream;
        // Pre-allocate rough memory to avoid reallocations (optional but good for performance)
        d_bfs_stream.reserve(N * 2); 

        float total_occupancy_time = 0.0f;
        for (int d = 0; d < octree_depth; ++d) {
            int num_parents = all_levels[d].size();
            int num_children = all_levels[d+1].size();

            // Allocate space for this level's occupancy bytes
            thrust::device_vector<uint8_t> d_level_occupancy(num_parents);

            cudaEventRecord(temp_start);
            int threads = 256;
            int blocks = (num_parents + threads - 1) / threads;

            compute_occupancy_kernel<<<blocks, threads>>>(
                thrust::raw_pointer_cast(all_levels[d].data()), num_parents,
                thrust::raw_pointer_cast(all_levels[d+1].data()), num_children,
                thrust::raw_pointer_cast(d_level_occupancy.data())
            );
            checkCudaErrors(cudaGetLastError());
            cudaEventRecord(temp_stop);
            cudaEventSynchronize(temp_stop);
            cudaEventElapsedTime(&elapsed_ms, temp_start, temp_stop);
            total_occupancy_time += elapsed_ms;
            std::cout << "  compute_occupancy_kernel (level " << d << "): " << elapsed_ms << " ms" << std::endl;

            // Append this level's bytes to the master stream
            cudaEventRecord(temp_start);
            d_bfs_stream.insert(d_bfs_stream.end(), d_level_occupancy.begin(), d_level_occupancy.end());
            cudaEventRecord(temp_stop);
            cudaEventSynchronize(temp_stop);
            cudaEventElapsedTime(&elapsed_ms, temp_start, temp_stop);
            if (elapsed_ms > 0.01f) {  // Only print if significant
                std::cout << "  Stream append (level " << d << "): " << elapsed_ms << " ms" << std::endl;
            }
        }
        std::cout << "  Total occupancy kernel time: " << total_occupancy_time << " ms" << std::endl;

        // Overall timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float total_milliseconds = 0;
        cudaEventElapsedTime(&total_milliseconds, start, stop);

        // 6. Statistics
        std::cout << "\n--- GPU Construction & Serialization Finished ---" << std::endl;
        std::cout << "Total time: " << total_milliseconds << " ms" << std::endl;
        
        size_t total_nodes = 0;
        std::cout << "\nTree Statistics:" << std::endl;
        for (int d = 0; d <= octree_depth; ++d) {
            size_t count = all_levels[d].size();
            total_nodes += count;
            std::cout << "Depth " << d << ": " << count << " nodes" << std::endl;
        }
        std::cout << "Total Octree Nodes: " << total_nodes << std::endl;
        std::cout << "Total Serialization Bytes: " << d_bfs_stream.size() << std::endl;

        // (Optional) Download first few bytes to verify
        // thrust::host_vector<uint8_t> h_head(10);
        // thrust::copy_n(d_bfs_stream.begin(), min((size_t)10, d_bfs_stream.size()), h_head.begin());
        // std::cout << "First 10 bytes (hex): ";
        // for(auto b : h_head) printf("%02X ", b);
        // std::cout << std::endl;
    }
    return 0;
}