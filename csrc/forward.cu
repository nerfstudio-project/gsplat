#include "helpers.cuh"
#include "forward.cuh"
#include <iostream>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

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
    const float fx,
    const float fy,
    const dim3 img_size,
    const dim3 tile_bounds,
    float *covs3d,
    float2 *xys,
    float *depths,
    int *radii,
    float3 *conics,
    uint32_t *num_tiles_hit
) {
    unsigned idx = cg::this_grid().thread_rank();  // idx of thread within grid
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;
    if (idx >= num_points) {
        return;
    }

    float3 p_world = means3d[idx];
    // printf("p_world %d %.2f %.2f %.2f\n", idx, p_world.x, p_world.y, p_world.z);
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view)) {
        printf("%d is out of frustum z %.2f, returning\n", idx, p_view.z);
        return;
    }
    // printf("p_view %d %.2f %.2f %.2f\n", idx, p_view.x, p_view.y, p_view.z);

    // compute the projected covariance
    float3 scale = scales[idx];
    float4 quat = quats[idx];
    // printf("%d scale %.2f %.2f %.2f\n", idx, scale.x, scale.y, scale.z);
    // printf("%d quat %.2f %.2f %.2f %.2f\n", idx, quat.w, quat.x, quat.y, quat.z);
    float *cur_cov3d = &(covs3d[6 * idx]);
    scale_rot_to_cov3d(scale, glob_scale, quat, cur_cov3d);

    // project to 2d with ewa approximation
    float3 cov2d = project_cov3d_ewa(p_world, cur_cov3d, viewmat, fx, fy);
    // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);

    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok) return; // zero determinant
    // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
    conics[idx] = conic;

    // compute the projected mean
    float2 center = project_pix(projmat, p_world, img_size);
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
    uint32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        printf("%d point bbox outside of bounds\n", idx);
        return;
    }

    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int) radius;
    xys[idx] = center;
    // printf(
    //     "point %d x %.2f y %.2f z %.2f, radius %d, # tiles %d, tile_min %d %d, tile_max %d %d\n",
    //     idx, center.x, center.y, depths[idx], radii[idx],
    //     tile_area, tile_min.x, tile_min.y, tile_max.x, tile_max.y
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
    const float fx,
    const float fy,
    const dim3 img_size,
    const dim3 tile_bounds,
    float *covs3d,
    float2 *xys,
    float *depths,
    int *radii,
    float3 *conics,
    uint32_t *num_tiles_hit
) {
    project_gaussians_forward_kernel
    <<< (num_points + N_THREADS - 1) / N_THREADS, N_THREADS >>> (
        num_points,
        means3d,
        scales,
        glob_scale,
        quats,
        viewmat,
        projmat,
        fx,
        fy,
        img_size,
        tile_bounds,
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
    const uint32_t *cum_tiles_hit,
    const dim3 tile_bounds,
    uint64_t *isect_ids,
    uint32_t *gaussian_ids
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
    // printf("point %d, %d radius, min %d %d, max %d %d\n", idx, radii[idx], tile_min.x, tile_min.y, tile_max.x, tile_max.y);

    // update the intersection info for all tiles this gaussian hits
    uint32_t cur_idx = (idx == 0) ? 0 : cum_tiles_hit[idx - 1];
    // printf("point %d starting at %d\n", idx, cur_idx);
    uint64_t depth_id = (uint64_t) *(uint32_t *) &(depths[idx]);
    for (int i = tile_min.y; i < tile_max.y; ++i) {
        for (int j = tile_min.x; j < tile_max.x; ++j) {
            // isect_id is tile ID and depth as uint32
            uint64_t tile_id = i * tile_bounds.x + j;  // tile within image
            isect_ids[cur_idx] = (tile_id << 32) | depth_id;
            gaussian_ids[cur_idx] = idx;
            ++cur_idx;
        }
    }
    // printf("point %d ending at %d\n", idx, cur_idx);
}


// kernel to map sorted intersection IDs to tile bins
// expect that intersection IDs are sorted by increasing tile ID
// i.e. intersections of a tile are in contiguous chunks
__global__ void get_tile_bin_edges(
    const int num_intersects,
    const uint64_t *isect_ids_sorted,
    uint2 *tile_bins
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_intersects)
        return;
    // save the indices where the tile_id changes
    uint32_t cur_tile_idx = (uint32_t) (isect_ids_sorted[idx] >> 32);
    if (idx == 0) {
        tile_bins[cur_tile_idx].x = 0;
        return;
    }
    if (idx == num_intersects - 1) {
        tile_bins[cur_tile_idx].y = num_intersects;
        return;
    }
    uint32_t prev_tile_idx = (uint32_t) (isect_ids_sorted[idx - 1] >> 32);
    if (prev_tile_idx != cur_tile_idx) {
        tile_bins[prev_tile_idx].y = idx;
        tile_bins[cur_tile_idx].x = idx;
        return;
    }
}


// launch on-device prefix sum to get the cumulative number of tiles for gaussians
void compute_cumulative_intersects(
    const int num_points,
    const uint32_t *num_tiles_hit,
    uint32_t &num_intersects,
    uint32_t *cum_tiles_hit
) {
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
        &num_intersects, &(cum_tiles_hit[num_points-1]), sizeof(uint32_t), cudaMemcpyDeviceToHost
    );
    cudaFree(sum_ws);
}


// figure out which gaussians, sorted by depth, to render for which tile of the output image
// output gaussian IDs for each tile, sorted by depth, as continguous array
// output start and end indices in this list of gaussians for each tile
void bin_and_sort_gaussians(
    const int num_points,
    const int num_intersects,
    const float2 *xys,
    const float *depths,
    const int *radii,
    const uint32_t *cum_tiles_hit,
    const dim3 tile_bounds,
    uint64_t *isect_ids_unsorted,
    uint32_t *gaussian_ids_unsorted,
    uint64_t *isect_ids_sorted,
    uint32_t *gaussian_ids_sorted,
    uint2 *tile_bins
) {
    // for each intersection map the tile ID and depth to a gaussian ID
    // allocate intermediate results
    // uint32_t *gaussian_ids_unsorted;
    // uint64_t *isect_ids_unsorted; // *isect_ids_sorted;
    // cudaMalloc((void**) &gaussian_ids_unsorted, num_intersects * sizeof(uint32_t));
    // cudaMalloc((void**) &isect_ids_unsorted, num_intersects * sizeof(uint64_t));
    // cudaMalloc((void**) &isect_ids_sorted, num_intersects * sizeof(uint64_t));
    map_gaussian_to_intersects <<< (num_points + N_THREADS - 1) / N_THREADS, N_THREADS >>> (
        num_points,
        xys,
        depths,
        radii,
        cum_tiles_hit,
        tile_bounds,
        isect_ids_unsorted,
        gaussian_ids_unsorted
    );

    // sort intersections by ascending tile ID and depth
    // allocate workspace memory
    void *sort_ws = nullptr;
    size_t sort_ws_bytes;
    cub::DeviceRadixSort::SortPairs(
        sort_ws, sort_ws_bytes,
        isect_ids_unsorted, isect_ids_sorted,
        gaussian_ids_unsorted, gaussian_ids_sorted,
        num_intersects
    );
    cudaMalloc(&sort_ws, sort_ws_bytes);
    cub::DeviceRadixSort::SortPairs(
        sort_ws, sort_ws_bytes,
        isect_ids_unsorted, isect_ids_sorted,
        gaussian_ids_unsorted, gaussian_ids_sorted,
        num_intersects
    );
    cudaFree(sort_ws);

    // get the start and end indices for the gaussians in each tile
    get_tile_bin_edges <<< (num_intersects + N_THREADS - 1) / N_THREADS, N_THREADS >>> (
        num_intersects, isect_ids_sorted, tile_bins
    );

    // free intermediate work spaces
    // cudaFree(isect_ids_unsorted);
    // cudaFree(isect_ids_sorted);
    // cudaFree(gaussian_ids_unsorted);
}


// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
__global__ void rasterize_forward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const uint32_t *gaussian_ids_sorted,
    const uint2 *tile_bins,
    const float2 *xys,
    const float3 *conics,
    const float3 *rgbs,
    const float *opacities,
    float *final_Ts,
    int *final_index,
    float3 *out_img
) {
    // current naive implementation where tile data loading is redundant
    // TODO tile data should be shared between tile threads
    uint32_t tile_id = blockIdx.y * tile_bounds.x + blockIdx.x;
    bool bg_white = (blockIdx.x % 2) == (blockIdx.y % 2);
    unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    float px = (float) j;
    float py = (float) i;
    uint32_t pix_id = i * img_size.x + j;

    // which gaussians to look through in this tile
    uint2 range = tile_bins[tile_id];
    float3 rgb, conic;
    float2 center, delta;
    float sigma, opac, alpha, next_T;
    float3 out_color = {0.f, 0.f, 0.f};
    float T = 1.f;
    // iterate over all gaussians
    int idx;
    for (idx = 0; idx < range.y; ++idx) {
        uint32_t g = gaussian_ids_sorted[idx];
        conic = conics[g];
        center = xys[g];
        delta = {center.x - px, center.y - py};
        sigma = 0.5f * (
            conic.x * delta.x * delta.x + conic.z * delta.y * delta.y
        ) - conic.y * delta.x * delta.y;
        if (sigma <= 0.f) {
            continue;
        }
        opac = opacities[g];
        alpha = min(0.999f, opac * exp(-sigma));
        if (alpha < 1.f / 255.f) {
            continue;
        }
        next_T = T * (1.f - alpha);
        if (next_T <= 1e-4f) {
            break;
        }
        rgb = rgbs[g];
        out_color.x += rgb.x * alpha * T;
        out_color.y += rgb.y * alpha * T;
        out_color.z += rgb.z * alpha * T;
        T = next_T;
    }
    final_Ts[pix_id] = T;  // transmittance at last gaussian in this pixel
    final_index[pix_id] = idx;  // index of in bin of last gaussian in this pixel
    if (bg_white) {
        out_color.x += T;
        out_color.y += T;
        out_color.z += T;
    }
    out_img[pix_id] = out_color;
}


// host function to launch parallel rasterization of sorted gaussians on device
void rasterize_forward_impl(
    const dim3 tile_bounds,
    const dim3 block,
    const dim3 img_size,
    const uint32_t *gaussian_ids_sorted,
    const uint2 *tile_bins,
    const float2 *xys,
    const float3 *conics,
    const float3 *rgbs,
    const float *opacities,
    float *final_Ts,
    int *final_index,
    float3 *out_img
) {
    rasterize_forward_kernel <<< tile_bounds, block >>> (
        tile_bounds,
        img_size,
        gaussian_ids_sorted,
        tile_bins,
        xys,
        conics,
        rgbs,
        opacities,
        final_Ts,
        final_index,
        out_img
    );
}


// device helper to approximate projected 2d cov from 3d mean and cov
__device__ float3 project_cov3d_ewa(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy
    // const float tan_fovx,
    // const float tan_fovy,
) {
    // we expect row major matrices as input, glm uses column major
    // upper 3x3 submatrix
    glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );
    glm::vec3 p = glm::vec3(viewmat[3], viewmat[7], viewmat[11]);
    glm::vec3 t = W * glm::vec3(mean3d.x, mean3d.y, mean3d.z) + p;

    // column major
    // we only care about the top 2x2 submatrix
    glm::mat3 J = glm::mat3(
        fx / t.z, 0.f, 0.f,
        0.f, fy / t.z, 0.f,
        -fx * t.x / (t.z * t.z), -fy * t.y / (t.z * t.z), 0.f
    );

    glm::mat3 T = J * W;

    glm::mat3 V = glm::mat3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );

    glm::mat3 cov = T * V * glm::transpose(T);
    // add a little blur along axes and save upper triangular elements
    return (float3) {float(cov[0][0]) + 0.1f, float(cov[0][1]), float(cov[1][1]) + 0.1f};
}

// device helper to get 3D covariance from scale and quat parameters
__device__ void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, float *cov3d
) {
    glm::mat3 R = quat_to_rotmat(quat);
    // printf("R %.2f %.2f %.2f\n", R[0][0], R[1][1], R[2][2]);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    // printf("S %.2f %.2f %.2f\n", S[0][0], S[1][1], S[2][2]);

    glm::mat3 M = R * S;
    glm::mat tmp = M * glm::transpose(M);
    // printf("tmp %.2f %.2f %.2f\n", tmp[0][0], tmp[1][1], tmp[2][2]);

    // save upper right because symmetric
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}
