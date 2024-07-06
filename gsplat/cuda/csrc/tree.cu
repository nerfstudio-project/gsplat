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

// /*
// Given all the leaf nodes of the tree, as well as the tree structure, we want to
// find a cut in the tree to locate the nodes (internal or leaf) that satisfy a
// specific condition (e.g., node above the cut is larger than the cut and node
// below the cut is smaller than the cut).

// The tree is currently set as a 2^N branching factor tree. (e.g., N=3 means there
// are 8 children for each node)

// Tutorial:
// https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
// */
// template <uint32_t DIM>
// __global__ void
// tree_cut_kernel(const uint32_t N,      // 2^N is the branching factor of the tree
//                 const uint32_t n_leaf, // number of leaf nodes in the tree
//                 const float *__restrict__ leaf_data, // [n_leaf, DIM]
//                 const bool *__restrict__ leaf_mask,  // [n_leaf]
//                 const float cut,                     // the cut value
//                 // outputs
//                 float *__restrict__ selected_data, // [n_leaf]
//                 bool *__restrict__ selected_mask   // [n_leaf]
// ) {
//     // The number of threads is the same as number of leaves.

//     // In the first step, each thread will check if the leaf node satisfies the
//     // condition (larger than the cut). If it does, it will set the mask to
//     // true, and write out the data (but the thread still stays alive).

//     // In the second step, all children (every 2^N threads) of the same parent
//     // will reduce the data into the 0-th lane of the every 2^N threads, to
//     // calculate the data of the parent node. (using __shfl_down_sync)

//     // In the third step, we check if the parent node satisfies the condition.
//     // This only needs to be done by the 0-th lane of every 2^N threads, as it
//     // is where the parent node data is calculated. If the parent node is larger
//     // than the cut, we send the signal to all the children (2^N threads). The
//     // children which are smaller than the cut will be selected and write out
//     // the data and mask. If the parent node is smaller than the cut, we know
//     // all the children are smaller than the cut, so we don't need to send the
//     // signal to the children. (using __shfl_sync)

//     // In the fourth step, we repeat the second step to calculate the
//     // grandparent node data (every 2^(N+1) threads), and repeat the following
//     // steps.

//     // The process will be repeated until we reach the root node. (all threads
//     // are reduced to 0-th lane)

//     uint32_t idx = cg::this_grid().thread_rank();
//     if (idx >= n_leaf)
//         return;

//     // Step 1

//     // Read the data from global memory
//     float data[DIM] = {0.f};
//     if (leaf_mask[idx]) {
// #pragma unroll
//         for (uint32_t i = 0; i < DIM; i++) {
//             data[i] = leaf_data[idx * DIM + i];
//         }
//     }

//     // Check if the leaf node satisfies the condition.
//     // In this example code we use the first dimension of the data to compare
//     // with the cut. In the hierachical GS case we need to project the GS to
//     // image plane and compare the projected radius with the cut.
//     bool is_larger = data[0] > cut;
//     printf("thread %d: is_larger %d\n", idx, is_larger);

//     // Write out the data and mask if the leaf node satisfies the condition.
//     if (is_larger) {
//         printf("thread %d: selected\n", idx);
// #pragma unroll
//         for (uint32_t i = 0; i < DIM; i++) {
//             selected_data[idx * DIM + i] = data[i];
//         }
//         selected_mask[idx] = true;
//     }

//     // Step 2

//     // Copy the data to parent_data. This is necessary as the following
//     // __shfl_down_sync operation will overwrite the data it applies to.
//     float parent_data[DIM] = {0.f};
// #pragma unroll
//     for (int32_t i = 0; i < DIM; i++) {
//         parent_data[i] = data[i];
//     }

//     // Calculate the parent node data via __shfl_down_sync (every 2^N threads)
// #pragma unroll
//     for (int32_t i = 0; i < DIM; i++) {
//         // e.g., N == 3: branching factor is 8, then we need to
//         // __shfl_down_sync(m, v, 4)
//         // __shfl_down_sync(m, v, 2)
//         // __shfl_down_sync(m, v, 1)
//         // to get the parent node data (simply a sum op in this example).
//         for (int32_t j = N - 1; j >= 0; j--) {
//             float v = __shfl_down_sync(0xFFFFFFFF, parent_data[i], 1 << j);
//             printf("thread %d: j %d, v %f, parent_data[%d] %f\n", idx, j, v, i,
//                    parent_data[i]);
//             parent_data[i] += v;
//         }
//     }
//     printf("thread %d: parent_data[0] %f\n", idx, parent_data[0]);

//     // Step 3

//     // Check if the parent node satisfies the condition.
//     bool is_parent_larger = false;
//     if (idx % (1 << N) == 0) {
//         is_parent_larger = parent_data[0] > cut;
//         printf("thread %d (parent): is_parent_larger: %d\n", idx, is_parent_larger);
//     }

//     // Send the is_parent_larger signal from the 0-th lane of every 2^N threads
//     // to all the 2^N threads, using __shfl_sync.
//     uint32_t lane_id_parent = idx / (1 << N) * (1 << N);
//     is_parent_larger = __shfl_sync(0xFFFFFFFF, is_parent_larger, lane_id_parent);
//     printf("thread %d (children): is_parent_larger: %d\n", idx, is_parent_larger);

//     // Step 4

//     // See if the children satisfy the condition.
//     if (is_parent_larger & !is_larger) {
//         // Write out the data and mask if the child node satisfies the
//         // condition.
//         printf("thread %d: selected\n", idx);
// #pragma unroll
//         for (uint32_t i = 0; i < DIM; i++) {
//             selected_data[idx * DIM + i] = data[i];
//         }
//         selected_mask[idx] = true;
//     }

//     //     else {
//     //         // If not satisfied, then update the data as the parent data, as
//     //         this lane
//     //         // now represents the parent node.
//     //         printf("thread %d: not selected. set parent data as data\n",
//     //         idx);
//     // #pragma unroll
//     //         for (uint32_t i = 0; i < DIM; i++) {
//     //             data[i] = parent_data[i];
//     //         }
//     //     }

//     // Repeat the process until we reach the root node.
//     // To be implemented.
// }

// std::tuple<torch::Tensor, torch::Tensor> tree_cut_tensor(
//     const torch::Tensor &leaf_data,
//     const torch::Tensor &leaf_mask,
//     const int32_t branch_factor,
//     const float cut
// ) {
//     // Check the input tensor
//     TORCH_CHECK(leaf_data.dim() == 2, "leaf_data must be 2D tensor");
//     TORCH_CHECK(leaf_mask.dim() == 1, "leaf_mask must be 1D tensor");
//     TORCH_CHECK(
//         leaf_data.size(0) == leaf_mask.size(0),
//         "leaf_data and leaf_mask must have the same size"
//     );

//     // Get the number of leaf nodes and the dimension of the data
//     const uint32_t n_leaf = leaf_data.size(0);
//     const uint32_t DIM = leaf_data.size(1);
//     const uint32_t N = static_cast<uint32_t>(std::log2(branch_factor));

//     // Allocate the output tensor
//     torch::Tensor selected_data =
//         torch::zeros({n_leaf, DIM}, leaf_data.options());
//     torch::Tensor selected_mask = torch::zeros({n_leaf},
//     leaf_mask.options());

//     if (n_leaf) {
//         at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
//         switch (DIM) {
//         case 1:
//             tree_cut_kernel<1>
//                 <<<(n_leaf + N_THREADS - 1) / N_THREADS,
//                    N_THREADS,
//                    0,
//                    stream>>>(
//                     N,
//                     n_leaf,
//                     leaf_data.data_ptr<float>(),
//                     leaf_mask.data_ptr<bool>(),
//                     cut,
//                     selected_data.data_ptr<float>(),
//                     selected_mask.data_ptr<bool>()
//                 );
//             break;
//         default:
//             TORCH_CHECK(false, "Unsupported dimension of the data");
//         }
//     }

//     return std::make_tuple(selected_data, selected_mask);
// }

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

    // Each block contains N_THREADS (e.g., 256) threads.
    // Each thread will process 2^N leaf nodes.
    // So the shared memory (all threads within a block have access to) will
    // store N_THREADS * 2^N nodes.

    extern __shared__ int sm[];
    // [N_THREADS * BR * DIM]
    float *sm_data = reinterpret_cast<float *>(sm);
    // [N_THREADS * BR]
    bool *sm_mask = reinterpret_cast<bool *>(&sm_data[N_THREADS * BR * DIM]);

    float block_total_data[DIM] = {0.f};
    for (uint32_t block_col = 0; block_col < n_leaf; block_col += BR * N_THREADS) {

        // Load data into shared memory (BR nodes per thread).
        for (uint32_t i = 0; i < BR; i++) {
            uint32_t offset = threadIdx.x + i * N_THREADS;
            uint32_t col = block_col + offset;

            if (col < n_leaf) {
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
                        printf("[Final] thread %d: col %d, data[0] %f\n", threadIdx.x,
                               col, sm_data[offset * DIM]);
                    }
                }
                printf("[LOAD DATA] thread %d: col %d, offset %d, data[0] %f\n",
                       threadIdx.x, col, offset, sm_data[offset * DIM]);

            } else {
                // If the thread is out of range, set the mask to false and data to
                // zero.
                sm_mask[offset] = false;
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
            printf("[ADD TOTAL] thread %d: data[0] %f\n", threadIdx.x, sm_data[0]);
        }

        // Sync threads to make sure the total value is updated.
        __syncthreads();

        // Parallel reduction (up-sweep).
        for (uint32_t s = N_THREADS, d = 1; s >= 1; s /= BR, d *= BR) {
            uint32_t offset = (BR * threadIdx.x + 1) * d - 1;
            // accumulate to node [offset + (BR - 1) * d].
            bool valid = offset + (BR - 1) * d < n_leaf;
            if (threadIdx.x < s && valid) {

                float accum_data[DIM] = {0.f};
#pragma unroll
                for (uint32_t i = 0; i < DIM; i++) {
#pragma unroll
                    for (uint32_t j = 0; j < BR; j++) {
                        // A simple add op.
                        // add node [offset + j * d] to accum_data
                        accum_data[i] += sm_data[(offset + j * d) * DIM + i];

                        if (threadIdx.x < n_leaf) {
                            printf("[REDUCE] thread %d: add sm node %d to accum data "
                                   "living in node %d, "
                                   "accum_data[0] %f\n",
                                   threadIdx.x, offset + j * d, offset + (BR - 1) * d,
                                   accum_data[0]);
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
                    for (uint32_t i = 0; i < BR - 1; i++) {
                        uint32_t col = block_col + offset + i * d;
                        if (col < n_leaf) {
                            selected_mask[col] = false;
                        }
                        printf("[Final] thread %d: col %d, set to false\n", threadIdx.x,
                               col);
                    }
                } else {
                    // If the accumulated data (parent) is larger than the cut, we want
                    // to write out the children which are smaller than the cut.
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
                            printf("[Final] thread %d: col %d, data[0] %f\n",
                                   threadIdx.x, col, sm_data[offset_node * DIM]);
                        }
                    }
                }

                // write the accumulated data to the last child.
                for (uint32_t i = 0; i < DIM; i++) {
                    sm_data[(offset + (BR - 1) * d) * DIM + i] = accum_data[i];
                    // now the last node in the children is actually the parent.
                    // So we set its mask to true.
                    sm_mask[offset + (BR - 1) * d] = true;
                }
            }
            __syncthreads();
        }
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
    TORCH_CHECK(is_power_of(N_THREADS, BR), "N_THREADS must be a power of ", BR,
                " but got ", N_THREADS);

    // Allocate the output tensor
    torch::Tensor selected_data = torch::zeros({n_leaf, DIM}, leaf_data.options());
    torch::Tensor selected_mask = torch::zeros({n_leaf}, leaf_mask.options());

    // Call the kernel
    if (n_leaf) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        const uint32_t shared_mem =
            N_THREADS * BR * (DIM * sizeof(float) + sizeof(bool));
        // Each thread process BR leaf nodes. So the number of threads in total
        // is n_leaf / BR.
        // Each block contains N_THREADS threads. So the shared memory that each
        // block requests is N_THREADS * BR * <size of each node>.
        tree_cut2_kernel<DIM, BR><<<(n_leaf / BR + N_THREADS - 1) / N_THREADS,
                                    N_THREADS, shared_mem, stream>>>(
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
        return tree_cut_internel<1, 2>(leaf_data, leaf_mask, cut);
    case 8:
        return tree_cut_internel<1, 8>(leaf_data, leaf_mask, cut);
    default:
        TORCH_CHECK(false, "Unsupported branch factor");
    }
}