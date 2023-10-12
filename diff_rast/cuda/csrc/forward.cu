#include "forward.cuh"
#include "helpers.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <iostream>

namespace cg = cooperative_groups;

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
__global__ void project_gaussians_forward_kernel(
    const int num_points,
    const float3 *means3d,
    const float3 *scales,
    const float glob_scale,
    const float4 *quats,
    const float *viewmat,
    const float *projmat,
    const float4 intrins,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    float *covs3d,
    float2 *xys,
    float *depths,
    int *radii,
    float3 *conics,
    int32_t *num_tiles_hit
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    float3 p_world = means3d[idx];
    // printf("p_world %d %.2f %.2f %.2f\n", idx, p_world.x, p_world.y,
    // p_world.z);
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view, clip_thresh)) {
        // printf("%d is out of frustum z %.2f, returning\n", idx, p_view.z);
        return;
    }
    // printf("p_view %d %.2f %.2f %.2f\n", idx, p_view.x, p_view.y, p_view.z);

    // compute the projected covariance
    float3 scale = scales[idx];
    float4 quat = quats[idx];
    // printf("%d scale %.2f %.2f %.2f\n", idx, scale.x, scale.y, scale.z);
    // printf("%d quat %.2f %.2f %.2f %.2f\n", idx, quat.w, quat.x, quat.y,
    // quat.z);
    float *cur_cov3d = &(covs3d[6 * idx]);
    scale_rot_to_cov3d(scale, glob_scale, quat, cur_cov3d);

    // project to 2d with ewa approximation
    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;
    float tan_fovx = 0.5 * img_size.x / fx;
    float tan_fovy = 0.5 * img_size.y / fy;
    float3 cov2d = project_cov3d_ewa(
        p_world, cur_cov3d, viewmat, fx, fy, tan_fovx, tan_fovy
    );
    // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);

    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok)
        return; // zero determinant
    // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
    conics[idx] = conic;

    // compute the projected mean
    float2 center = project_pix(projmat, p_world, img_size, {cx, cy});
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        // printf("%d point bbox outside of bounds\n", idx);
        return;
    }

    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    xys[idx] = center;
    // printf(
    //     "point %d x %.2f y %.2f z %.2f, radius %d, # tiles %d, tile_min %d
    //     %d, tile_max %d %d\n", idx, center.x, center.y, depths[idx],
    //     radii[idx], tile_area, tile_min.x, tile_min.y, tile_max.x, tile_max.y
    // );
}

// host function to launch the projection in parallel on device
void project_gaussians_forward_impl(
    const int num_points,
    const float3 *means3d,
    const float3 *scales,
    const float glob_scale,
    const float4 *quats,
    const float *viewmat,
    const float *projmat,
    const float4 intrins,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    float *covs3d,
    float2 *xys,
    float *depths,
    int *radii,
    float3 *conics,
    int *num_tiles_hit
) {
    project_gaussians_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        means3d,
        scales,
        glob_scale,
        quats,
        viewmat,
        projmat,
        intrins,
        img_size,
        tile_bounds,
        clip_thresh,
        covs3d,
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit
    );
}

// kernel to map each intersection from tile ID and depth to a gaussian
// writes output to isect_ids and gaussian_ids
__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2 *xys,
    const float *depths,
    const int *radii,
    const int32_t *cum_tiles_hit,
    const dim3 tile_bounds,
    int64_t *isect_ids,
    int32_t *gaussian_ids
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points)
        return;
    if (radii[idx] <= 0)
        return;
    // get the tile bbox for gaussian
    uint2 tile_min, tile_max;
    float2 center = xys[idx];
    get_tile_bbox(center, radii[idx], tile_bounds, tile_min, tile_max);
    // printf("point %d, %d radius, min %d %d, max %d %d\n", idx, radii[idx],
    // tile_min.x, tile_min.y, tile_max.x, tile_max.y);

    // update the intersection info for all tiles this gaussian hits
    int32_t cur_idx = (idx == 0) ? 0 : cum_tiles_hit[idx - 1];
    // printf("point %d starting at %d\n", idx, cur_idx);
    int64_t depth_id = (int64_t) * (int32_t *)&(depths[idx]);
    for (int i = tile_min.y; i < tile_max.y; ++i) {
        for (int j = tile_min.x; j < tile_max.x; ++j) {
            // isect_id is tile ID and depth as int32
            int64_t tile_id = i * tile_bounds.x + j; // tile within image
            isect_ids[cur_idx] = (tile_id << 32) | depth_id; // tile | depth id
            gaussian_ids[cur_idx] = idx;                     // 3D gaussian id
            ++cur_idx; // handles gaussians that hit more than one tile
        }
    }
    // printf("point %d ending at %d\n", idx, cur_idx);
}

// kernel to map sorted intersection IDs to tile bins
// expect that intersection IDs are sorted by increasing tile ID
// i.e. intersections of a tile are in contiguous chunks
__global__ void get_tile_bin_edges(
    const int num_intersects, const int64_t *isect_ids_sorted, int2 *tile_bins
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_intersects)
        return;
    // save the indices where the tile_id changes
    int32_t cur_tile_idx = (int32_t)(isect_ids_sorted[idx] >> 32);
    if (idx == 0) {
        tile_bins[cur_tile_idx].x = 0;
        return;
    }
    if (idx == num_intersects - 1) {
        tile_bins[cur_tile_idx].y = num_intersects;
        return;
    }
    int32_t prev_tile_idx = (int32_t)(isect_ids_sorted[idx - 1] >> 32);
    if (prev_tile_idx != cur_tile_idx) {
        tile_bins[prev_tile_idx].y = idx;
        tile_bins[cur_tile_idx].x = idx;
        return;
    }
}

// launch on-device prefix sum to get the cumulative number of tiles for
// gaussians
void compute_cumulative_intersects(
    const int num_points,
    const int32_t *num_tiles_hit,
    int32_t &num_intersects,
    int32_t *cum_tiles_hit
) {
    // ref:
    // https://nvlabs.github.io/cub/structcub_1_1_device_scan.html#a9416ac1ea26f9fde669d83ddc883795a
    // allocate sum workspace
    void *sum_ws = nullptr;
    size_t sum_ws_bytes;
    cub::DeviceScan::InclusiveSum(
        sum_ws, sum_ws_bytes, num_tiles_hit, cum_tiles_hit, num_points
    );
    cudaMalloc(&sum_ws, sum_ws_bytes);
    cub::DeviceScan::InclusiveSum(
        sum_ws, sum_ws_bytes, num_tiles_hit, cum_tiles_hit, num_points
    );
    cudaMemcpy(
        &num_intersects,
        &(cum_tiles_hit[num_points - 1]),
        sizeof(int32_t),
        cudaMemcpyDeviceToHost
    );
    cudaFree(sum_ws);
}

// figure out which gaussians, sorted by depth, to render for which tile of the
// output image output gaussian IDs for each tile, sorted by depth, as
// continguous array output start and end indices in this list of gaussians for
// each tile
void bin_and_sort_gaussians(
    const int num_points,
    const int num_intersects,
    const float2 *xys,
    const float *depths,
    const int *radii,
    const int32_t *cum_tiles_hit,
    const dim3 tile_bounds,
    int64_t *isect_ids_unsorted,
    int32_t *gaussian_ids_unsorted,
    int64_t *isect_ids_sorted,
    int32_t *gaussian_ids_sorted,
    int2 *tile_bins
) {
    // for each intersection map the tile ID and depth to a gaussian ID
    // allocate intermediate results
    // int32_t *gaussian_ids_unsorted;
    // int64_t *isect_ids_unsorted; // *isect_ids_sorted;
    // cudaMalloc((void**) &gaussian_ids_unsorted, num_intersects *
    // sizeof(int32_t)); cudaMalloc((void**) &isect_ids_unsorted,
    // num_intersects * sizeof(int64_t)); cudaMalloc((void**)
    // &isect_ids_sorted, num_intersects * sizeof(int64_t));
    map_gaussian_to_intersects<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
        xys,
        depths,
        radii,
        cum_tiles_hit,
        tile_bounds,
        isect_ids_unsorted,
        gaussian_ids_unsorted
    );

    // sort intersections by ascending tile ID and depth with RadixSort
    int32_t max_tile_id = (int32_t)(tile_bounds.x * tile_bounds.y);
    int msb = 32 - __builtin_clz(max_tile_id) + 1;
    // allocate workspace memory
    void *sort_ws = nullptr;
    size_t sort_ws_bytes;
    cub::DeviceRadixSort::SortPairs(
        sort_ws,
        sort_ws_bytes,
        isect_ids_unsorted,
        isect_ids_sorted,
        gaussian_ids_unsorted,
        gaussian_ids_sorted,
        num_intersects,
        0,
        32 + msb
    );
    cudaMalloc(&sort_ws, sort_ws_bytes);
    cub::DeviceRadixSort::SortPairs(
        sort_ws,
        sort_ws_bytes,
        isect_ids_unsorted,
        isect_ids_sorted,
        gaussian_ids_unsorted,
        gaussian_ids_sorted,
        num_intersects,
        0,
        32 + msb
    );
    cudaFree(sort_ws);

    // get the start and end indices for the gaussians in each tile
    // printf("launching tile binning %d %d\n",
    // (num_intersects + N_THREADS - 1) / N_THREADS,
    // N_THREADS);
    get_tile_bin_edges<<<
        (num_intersects + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(num_intersects, isect_ids_sorted, tile_bins);

    // free intermediate work spaces
    // cudaFree(isect_ids_unsorted);
    // cudaFree(isect_ids_sorted);
    // cudaFree(gaussian_ids_unsorted);
}

// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
__global__ void nd_rasterize_forward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t *gaussian_ids_sorted,
    const int2 *tile_bins,
    const float2 *xys,
    const float3 *conics,
    const float *colors,
    const float *opacities,
    float *final_Ts,
    int *final_index,
    float *out_img,
    const float *background
) {
    // current naive implementation where tile data loading is redundant
    // TODO tile data should be shared between tile threads
    int32_t tile_id = blockIdx.y * tile_bounds.x + blockIdx.x;
    unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    if (i >= img_size.y || j >= img_size.x) {
        return;
    }

    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    float T = 1.f;

    // iterate over all gaussians and apply rendering EWA equation (e.q. 2 from
    // paper)
    int idx;
    for (idx = range.x; idx < range.y; ++idx) {
        const int32_t g = gaussian_ids_sorted[idx];
        const float3 conic = conics[g];
        const float2 center = xys[g];
        const float2 delta = {center.x - px, center.y - py};

        // Mahalanobis distance (here referred to as sigma) measures how many
        // standard deviations away distance delta is. sigma = -0.5(d.T * conic
        // * d)
        const float sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
            conic.y * delta.x * delta.y;
        if (sigma < 0.f) {
            continue;
        }
        const float opac = opacities[g];

        const float alpha = min(0.999f, opac * exp(-sigma));

        // break out conditions
        if (alpha < 1.f / 255.f) {
            continue;
        }
        const float next_T = T * (1.f - alpha);
        if (next_T <= 1e-4f) {
            // we want to render the last gaussian that contributes and note
            // that here idx > range.x so we don't underflow
            idx -= 1;
            break;
        }
        const float vis = alpha * T;
        for (int c = 0; c < channels; ++c) {
            out_img[channels * pix_id + c] += colors[channels * g + c] * vis;
        }
        T = next_T;
    }
    final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
    final_index[pix_id] =
        (idx == range.y)
            ? idx - 1
            : idx; // index of in bin of last gaussian in this pixel
    for (int c = 0; c < channels; ++c) {
        out_img[channels * pix_id + c] += T * background[c];
    }
}

// host function to launch parallel rasterization of sorted gaussians on device
void nd_rasterize_forward_impl(
    const dim3 tile_bounds,
    const dim3 block,
    const dim3 img_size,
    const unsigned channels,
    const int32_t *gaussian_ids_sorted,
    const int2 *tile_bins,
    const float2 *xys,
    const float3 *conics,
    const float *colors,
    const float *opacities,
    float *final_Ts,
    int *final_index,
    float *out_img,
    const float *background
) {
    nd_rasterize_forward_kernel<<<tile_bounds, block>>>(
        tile_bounds,
        img_size,
        channels,
        gaussian_ids_sorted,
        tile_bins,
        xys,
        conics,
        colors,
        opacities,
        final_Ts,
        final_index,
        out_img,
        background
    );
}

__global__ void rasterize_forward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t *gaussian_ids_sorted,
    const int2 *tile_bins,
    const float2 *xys,
    const float3 *conics,
    const float3 *colors,
    const float *opacities,
    float *final_Ts,
    int *final_index,
    float3 *out_img,
    const float3 &background
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    float px = (float)j;
    float py = (float)i;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float2 xy_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    __shared__ float opacity_batch[BLOCK_SIZE];

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        int num_done = __syncthreads_count(done);
        if (num_done >= BLOCK_SIZE) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + BLOCK_SIZE * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            xy_batch[tr] = xys[g_id];
            conic_batch[tr] = conics[g_id];
            opacity_batch[tr] = opacities[g_id];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(BLOCK_SIZE, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 conic = conic_batch[t];
            const float2 center = xy_batch[t];
            const float opac = opacity_batch[t];
            const float2 delta = {center.x - px, center.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            const float alpha = min(0.999f, opac * exp(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const float next_T = T * (1.f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done
                // we want to render the last gaussian that contributes and note
                // that here idx > range.x so we don't underflow
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float3 c = colors[g];
            pix_out.x = pix_out.x + c.x * vis;
            pix_out.y = pix_out.y + c.y * vis;
            pix_out.z = pix_out.z + c.z * vis;
            T = next_T;
            cur_idx = batch_start + t;
        }
    }

    if (inside) {
        // add background
        final_Ts[pix_id] = T; // transmittance at last gaussian in this pixel
        final_index[pix_id] =
            cur_idx; // index of in bin of last gaussian in this pixel
        float3 final_color;
        final_color.x = pix_out.x + T * background.x;
        final_color.y = pix_out.y + T * background.y;
        final_color.z = pix_out.z + T * background.z;
        out_img[pix_id] = final_color;
    }
}

// host function to launch parallel rasterization of sorted gaussians on device
void rasterize_forward_impl(
    const dim3 tile_bounds,
    const dim3 block,
    const dim3 img_size,
    const int32_t *gaussian_ids_sorted,
    const int2 *tile_bins,
    const float2 *xys,
    const float3 *conics,
    const float3 *colors,
    const float *opacities,
    float *final_Ts,
    int *final_index,
    float3 *out_img,
    const float3 &background
) {
    rasterize_forward_kernel<<<tile_bounds, block>>>(
        tile_bounds,
        img_size,
        gaussian_ids_sorted,
        tile_bins,
        xys,
        conics,
        colors,
        opacities,
        final_Ts,
        final_index,
        out_img,
        background
    );
}

// device helper to approximate projected 2d cov from 3d mean and cov
__device__ float3 project_cov3d_ewa(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy
) {
    // clip the
    // we expect row major matrices as input, glm uses column major
    // upper 3x3 submatrix
    glm::mat3 W = glm::mat3(
        viewmat[0],
        viewmat[4],
        viewmat[8],
        viewmat[1],
        viewmat[5],
        viewmat[9],
        viewmat[2],
        viewmat[6],
        viewmat[10]
    );
    glm::vec3 p = glm::vec3(viewmat[3], viewmat[7], viewmat[11]);
    glm::vec3 t = W * glm::vec3(mean3d.x, mean3d.y, mean3d.z) + p;

    // clip so that the covariance
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    t.x = t.z * std::min(lim_x, std::max(-lim_x, t.x / t.z));
    t.y = t.z * std::min(lim_y, std::max(-lim_y, t.y / t.z));

    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    glm::mat3 J = glm::mat3(
        fx * rz,
        0.f,
        0.f,
        0.f,
        fy * rz,
        0.f,
        -fx * t.x * rz2,
        -fy * t.y * rz2,
        0.f
    );
    glm::mat3 T = J * W;

    glm::mat3 V = glm::mat3(
        cov3d[0],
        cov3d[1],
        cov3d[2],
        cov3d[1],
        cov3d[3],
        cov3d[4],
        cov3d[2],
        cov3d[4],
        cov3d[5]
    );

    glm::mat3 cov = T * V * glm::transpose(T);

    // add a little blur along axes and save upper triangular elements
    return (float3
    ){float(cov[0][0]) + 0.3f, float(cov[0][1]), float(cov[1][1]) + 0.3f};
}

// device helper to get 3D covariance from scale and quat parameters
__device__ void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, float *cov3d
) {
    // printf("quat %.2f %.2f %.2f %.2f\n", quat.x, quat.y, quat.z, quat.w);
    glm::mat3 R = quat_to_rotmat(quat);
    // printf("R %.2f %.2f %.2f\n", R[0][0], R[1][1], R[2][2]);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    // printf("S %.2f %.2f %.2f\n", S[0][0], S[1][1], S[2][2]);

    glm::mat3 M = R * S;
    glm::mat3 tmp = M * glm::transpose(M);
    // printf("tmp %.2f %.2f %.2f\n", tmp[0][0], tmp[1][1], tmp[2][2]);

    // save upper right because symmetric
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}
