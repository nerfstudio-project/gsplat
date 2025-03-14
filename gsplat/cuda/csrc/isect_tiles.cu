#include "isect_tiles.h"

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <c10/cuda/CUDACachingAllocator.h>

namespace cg = cooperative_groups;

__global__ void isect_tiles(
    // if the data is [C, N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over C * N, only used if packed is False
    const uint32_t C,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const float *__restrict__ means2d,                   // [C, N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [C, N] or [nnz]
    const float *__restrict__ depths,                    // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : C * N)) {
        return;
    }

    const float radius = radii[idx];
    if (radius <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    vec2 mean2d = glm::make_vec2(means2d + 2 * idx);

    float tile_radius = radius / static_cast<float>(tile_size);
    float tile_x = mean2d.x / static_cast<float>(tile_size);
    float tile_y = mean2d.y / static_cast<float>(tile_size);

    // tile_min is inclusive, tile_max is exclusive
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius)), tile_width);
    tile_min.y =
        min(max(0, (uint32_t)floor(tile_y - tile_radius)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius)), tile_height);

    if (first_pass) {
        // first pass only writes out tiles_per_gauss
        tiles_per_gauss[idx] = static_cast<int32_t>(
            (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
        );
        return;
    }

    int64_t cid; // camera id
    if (packed) {
        // parallelize over nnz
        cid = camera_ids[idx];
        // gid = gaussian_ids[idx];
    } else {
        // parallelize over C * N
        cid = idx / N;
        // gid = idx % N;
    }
    const int64_t cid_enc = cid << (32 + tile_n_bits);

    int64_t depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            int64_t tile_id = i * tile_width + j;
            // e.g. tile_n_bits = 22:
            // camera id (10 bits) | tile id (22 bits) | depth (32 bits)
            isect_ids[cur_idx] = cid_enc | (tile_id << 32) | depth_id_enc;
            // the flatten index in [C * N] or [nnz]
            flatten_ids[cur_idx] = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
}


void isect_tiles_launcher(
    uint32_t shmem_size, 
    cudaStream_t stream, 
    uint32_t n_elements, 
    // args
    const bool packed,
    const uint32_t C,
    const uint32_t N,
    const uint32_t nnz,
    const int64_t *__restrict__ camera_ids,   // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    const float *__restrict__ means2d,                   // [C, N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [C, N] or [nnz]
    const float *__restrict__ depths,                    // [C, N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [C, N] or [nnz]
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [C, N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    if (n_elements <= 0) {
        return;
    }
    isect_tiles<<<n_blocks_linear(n_elements), N_THREADS, shmem_size, stream>>>(
        packed,
        C,
        N,
        nnz,
        camera_ids,
        gaussian_ids,
        means2d,
        radii,
        depths,
        cum_tiles_per_gauss,
        tile_size,
        tile_width,
        tile_height,
        tile_n_bits,
        tiles_per_gauss,
        isect_ids,
        flatten_ids
    );
}


__global__ void isect_offset_encode(
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t C,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ offsets // [C, n_tiles]
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects)
        return;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t cid_curr = isect_id_curr >> tile_n_bits;
    int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr = cid_curr * n_tiles + tid_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
    if (idx == n_isects - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < C * n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(n_isects);
    }

    if (idx > 0) {
        // visit the current and previous isect_id and check if the (cid,
        // tile_id) pair changes.
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if (isect_id_prev == isect_id_curr)
            return;

        // write out the offsets between the previous and current tiles
        int64_t cid_prev = isect_id_prev >> tile_n_bits;
        int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        int64_t id_prev = cid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
}

void isect_offset_encode_launcher(
    uint32_t shmem_size, 
    cudaStream_t stream, 
    uint32_t n_elements, 
    // args
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t C,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ offsets // [C, n_tiles]
) {
    if (n_elements <= 0) {
        return;
    }
    isect_offset_encode<<<n_blocks_linear(n_elements), N_THREADS, shmem_size, stream>>>(
        n_isects,
        isect_ids,
        C,
        n_tiles,
        tile_n_bits,
        offsets
    );
}

void radix_sort(
    cudaStream_t stream, 
    // args
    const int64_t n_isects,
    const uint32_t tile_n_bits,
    const uint32_t cam_n_bits,
    const bool double_buffer,
    int64_t *__restrict__ isect_ids,
    int32_t *__restrict__ flatten_ids,
    int64_t *__restrict__ isect_ids_sorted,
    int32_t *__restrict__ flatten_ids_sorted,
    bool &isect_ids_swapped,
    bool &flatten_ids_swapped
) {
    // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
    // DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
    if (double_buffer) {
        // Create a set of DoubleBuffers to wrap pairs of device pointers
        cub::DoubleBuffer<int64_t> d_keys(isect_ids, isect_ids_sorted);
        cub::DoubleBuffer<int32_t> d_values(flatten_ids, flatten_ids_sorted);
        CUB_WRAPPER(
            cub::DeviceRadixSort::SortPairs,
            d_keys,
            d_values,
            n_isects,
            0,
            32 + tile_n_bits + cam_n_bits,
            stream
        );
        switch (d_keys.selector) {
        case 0: // sorted items are stored in isect_ids
            // isect_ids_sorted = isect_ids;
            isect_ids_swapped = true;
            break;
        case 1: // sorted items are stored in isect_ids_sorted
            isect_ids_swapped = false;
            break;
        }
        switch (d_values.selector) {
        case 0: // sorted items are stored in flatten_ids
            // flatten_ids_sorted = flatten_ids;
            flatten_ids_swapped = true;
            break;
        case 1: // sorted items are stored in flatten_ids_sorted
            flatten_ids_swapped = false;
            break;
        }
        // printf("DoubleBuffer d_keys selector: %d\n", d_keys.selector);
        // printf("DoubleBuffer d_values selector: %d\n",
        // d_values.selector);
    } else {
        CUB_WRAPPER(
            cub::DeviceRadixSort::SortPairs,
            isect_ids,
            isect_ids_sorted,
            flatten_ids,
            flatten_ids_sorted,
            n_isects,
            0,
            32 + tile_n_bits + cam_n_bits,
            stream
        );
    }
}