#include "bindings.h"
#include "helpers.cuh"
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include "utils.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;
#define THREADS 512

// BR = 2^N which is the branching factor of the tree
template <uint32_t DIM, uint32_t BR>
__global__ void
tree_cut2_kernel(const uint32_t n_leaf, // number of leaf nodes in the tree
                 const float *__restrict__ leaf_data, // [n_leaf, DIM]
                 const bool *__restrict__ leaf_mask,  // [n_leaf]
                 const float cut,                     // the cut value
                 // outputs
                 float *__restrict__ selected_data, // [n_leaf]
                 bool *__restrict__ selected_mask   // [n_leaf]
) {
    // The number of threads is n_leaf / 2^N.

    // Each block contains THREADS (e.g., 512) threads.
    // Each thread will process 2^N leaf nodes.
    // So the shared memory (all threads within a block have access to) will
    // store THREADS * 2^N nodes.

    extern __shared__ int sm[];
    // [THREADS * BR * DIM]
    float *sm_data = reinterpret_cast<float *>(sm);
    // [THREADS * BR]
    bool *sm_mask = reinterpret_cast<bool *>(&sm_data[THREADS * BR * DIM]);

    float block_total_data[DIM] = {0.f};
    for (uint32_t block_col = 0; block_col < n_leaf; block_col += BR * THREADS) {

        // Load data into shared memory (BR nodes per thread).
        for (uint32_t i = 0; i < BR; i++) {
            uint32_t offset = threadIdx.x + i * THREADS;
            uint32_t col = block_col + offset;

            // read from global memory to local register
            bool active = leaf_mask[col];
            // write to shared memory
            sm_mask[offset] = active;

            if (active) {
#pragma unroll
                for (uint32_t j = 0; j < DIM; j++) {
                    sm_data[offset * DIM + j] = leaf_data[col * DIM + j];
                }

                // write out the data if this node is selected (big leaf node)
                if (sm_data[offset * DIM] > cut) {
                    selected_mask[col] = true;
#pragma unroll
                    for (uint32_t j = 0; j < DIM; j++) {
                        selected_data[col * DIM + j] = sm_data[offset * DIM + j];
                    }
                    // printf("[Final] thread %d: col %d, data[0] %f\n",
                    // threadIdx.x,
                    //        col, sm_data[offset * DIM]);
                }

                // printf("[LOAD DATA] thread %d: col %d, offset %d, data[0] %f\n",
                //        threadIdx.x, col, offset, sm_data[offset * DIM]);

            } else {
                // If the thread is out of range, set the mask to false and data to
                // zero.
#pragma unroll
                for (uint32_t j = 0; j < DIM; j++) {
                    sm_data[offset * DIM + j] = 0.f;
                }
            }
        }

        // Add the total value of all previous blocks to the first value of this
        // block.
        if (threadIdx.x == 0) {
#pragma unroll
            for (uint32_t i = 0; i < DIM; i++) {
                // A simple add op.
                sm_data[i] += block_total_data[i];
            }
            // printf("[ADD TOTAL] thread %d: data[0] %f\n", threadIdx.x, sm_data[0]);
        }

        // Sync threads to make sure the total value is updated.
        __syncthreads();

        // Parallel reduction (up-sweep).
        for (uint32_t s = THREADS, d = 1; s >= 1; s /= BR, d *= BR) {
            uint32_t offset = (BR * threadIdx.x + 1) * d - 1;
            // accumulate to node [offset + (BR - 1) * d].
            bool valid = block_col + offset + (BR - 1) * d < n_leaf;
            if (threadIdx.x < s && valid) {

                float accum_data[DIM] = {0.f};
#pragma unroll
                for (uint32_t i = 0; i < DIM; i++) {
#pragma unroll
                    for (uint32_t j = 0; j < BR; j++) {
                        // A simple add op.
                        // add node [offset + j * d] to accum_data
                        accum_data[i] += sm_data[(offset + j * d) * DIM + i];

                        if (threadIdx.x < n_leaf && (offset + (BR - 1) * d == 373)) {
                            // printf("[REDUCE] thread %d: add sm node %d to accum data
                            // "
                            //        "living in node %d, "
                            //        "accum_data[0] %f\n",
                            //        threadIdx.x, offset + j * d, offset + (BR - 1) *
                            //        d, sm_data[(offset + j * d) * DIM + i]);
                        }
                    }
                }

                // cut condition.
                if (accum_data[0] <= cut) {
                    // If the accumulated data (parent) is smaller than the
                    // cut, all the children are smaller and useless. So we
                    // can set the output mask to false. Note we only skip the last
                    // child, as the last child is the one that will be
                    // overwritten by the parent.
#pragma unroll
                    for (uint32_t i = 0; i < BR - 1; i++) {
                        uint32_t col = block_col + offset + i * d;
                        if (col < n_leaf) {
                            selected_mask[col] = false;
                        }
                        // printf("[Final] thread %d: col %d, set to false\n",
                        // threadIdx.x,
                        //        col);
                    }
                } else {
                    // If the accumulated data (parent) is larger than the cut, we want
                    // to write out the children which are smaller than the cut.
#pragma unroll
                    for (uint32_t i = 0; i < BR; i++) {
                        uint32_t offset_node = offset + i * d;
                        uint32_t col = block_col + offset_node;
                        if (col < n_leaf && sm_mask[offset_node] &&
                            sm_data[offset_node * DIM] < cut) {
                            selected_mask[col] = true;
#pragma unroll
                            for (uint32_t j = 0; j < DIM; j++) {
                                selected_data[col * DIM + j] =
                                    sm_data[offset_node * DIM + j];
                            }
                            // printf("[Final] thread %d: col %d, data[0] %f\n",
                            //        threadIdx.x, col, sm_data[offset_node * DIM]);
                        }
                    }
                }

                // write the accumulated data to the last child.
#pragma unroll
                for (uint32_t i = 0; i < DIM; i++) {
                    sm_data[(offset + (BR - 1) * d) * DIM + i] = accum_data[i];
                    // now the last node in the children is actually the parent.
                    // So we set its mask to ANY(children).
                    bool active = false;
#pragma unroll
                    for (uint32_t j = 0; j < BR; j++) {
                        uint32_t offset_node = offset + j * d;
                        if (offset_node < n_leaf && sm_mask[offset_node]) {
                            active = true;
                            break;
                        }
                    }
                    sm_mask[offset + (BR - 1) * d] = active;
                }
            }
            __syncthreads();
        }

        // This block has finished. Update the total of it for the next block.
        for (uint32_t i = 0; i < DIM; i++) {
            block_total_data[i] = sm_data[(BR * THREADS - 1) * DIM + i];
        }
        __syncthreads();
    }
}

// Check if a number x is a power of another number b (i.e., x = b^n for some
// integer n)
bool is_power_of(uint32_t x, uint32_t b) {
    if (b <= 1 || x <= 0) {
        return false;
    }
    unsigned int temp = x; // Create a copy of x
    while (temp % b == 0) {
        temp /= b;
    }
    return temp == 1;
}

template <uint32_t DIM, uint32_t BR>
std::vector<torch::Tensor> tree_cut_internel(const torch::Tensor leaf_data,
                                             const torch::Tensor leaf_mask,
                                             const float cut) {
    // Check the input tensor
    TORCH_CHECK(leaf_data.dim() == 2, "leaf_data must be 2D tensor");
    TORCH_CHECK(leaf_mask.dim() == 1, "leaf_mask must be 1D tensor");
    TORCH_CHECK(
        leaf_data.size(0) == leaf_mask.size(0),
        "leaf_data and leaf_mask must have the same size on the first dimension");
    TORCH_CHECK(leaf_data.size(1) == DIM, "leaf_data shape mismatch: expected ", DIM,
                ", got ", leaf_data.size(1));

    // Get the number of leaf nodes
    const uint32_t n_leaf = leaf_data.size(0);
    TORCH_CHECK(is_power_of(n_leaf, BR), "n_leaf must be a power of ", BR, " but got ",
                n_leaf);
    TORCH_CHECK(is_power_of(THREADS, BR), "THREADS must be a power of ", BR,
                " but got ", THREADS);

    // Allocate the output tensor
    torch::Tensor selected_data = torch::zeros({n_leaf, DIM}, leaf_data.options());
    torch::Tensor selected_mask = torch::zeros({n_leaf}, leaf_mask.options());

    // Call the kernel
    if (n_leaf) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        const uint32_t shared_mem = THREADS * BR * (DIM * sizeof(float) + sizeof(bool));
        // Each thread process BR leaf nodes. So the number of threads in total
        // is n_leaf / BR.
        // Each block contains THREADS threads. So the shared memory that each
        // block requests is THREADS * BR * <size of each node>.
        tree_cut2_kernel<DIM, BR><<<1, THREADS, shared_mem, stream>>>(
            n_leaf, leaf_data.data_ptr<float>(), leaf_mask.data_ptr<bool>(), cut,
            selected_data.data_ptr<float>(), selected_mask.data_ptr<bool>());
    }

    // Return the output tensors
    auto outputs = std::vector<torch::Tensor>(2);
    outputs[0] = selected_data;
    outputs[1] = selected_mask;
    return outputs;
}

std::vector<torch::Tensor> tree_cut_tensor(const torch::Tensor leaf_data,
                                           const torch::Tensor leaf_mask,
                                           const uint32_t branch_factor,
                                           const float cut) {
    switch (branch_factor) {
    case 2:
        switch (leaf_data.size(1)) {
        case 1:
            return tree_cut_internel<1, 2>(leaf_data, leaf_mask, cut);
        case 16:
            return tree_cut_internel<16, 2>(leaf_data, leaf_mask, cut);
        }
    case 8:
        switch (leaf_data.size(1)) {
        case 1:
            return tree_cut_internel<1, 8>(leaf_data, leaf_mask, cut);
        case 16:
            return tree_cut_internel<16, 8>(leaf_data, leaf_mask, cut);
        }
    default:
        TORCH_CHECK(false, "Unsupported branch factor");
    }
}