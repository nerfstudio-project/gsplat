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
    const float *means3d,
    const float *scales,
    const float glob_scale,
    const float *quats,
    const float *viewmat,
    const float *projmat,
    const float fx,
    const float fy,
    const int W,
    const int H,
    const dim3 tile_bounds,
    float *covs3d,
    float *xys,
    float *depths,
    int *radii,
    uint32_t *num_tiles_hit
) {
    unsigned idx = cg::this_grid().thread_rank();  // idx of thread within grid
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;
    if (idx >= num_points) {
        return;
    }

    // float3 p_world = {means3d[3*idx], means3d[3*idx+1], means3d[3*idx+2]};
    float3 *test = (float3 *) means3d;
    float3 p_world = test[idx];
    printf("p_world %d %.2f %.2f %.2f\n", idx, p_world.x, p_world.y, p_world.z);
    float3 p_view;
    if (clip_near_plane(p_world, viewmat, p_view)) {
        printf("%d is out of frustum z %.2f, returning\n", idx, p_view.z);
        return;
    }
    printf("p_view %d %.2f %.2f %.2f\n", idx, p_view.x, p_view.y, p_view.z);

    float4 p_hom = transform_4x4(projmat, p_world);
    float p_w = 1.f / (p_hom.w + 1e-5f);
    float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};
    printf("p_proj %d %.2f %.2f %.2f\n", idx, p_proj.x, p_proj.y, p_proj.z);

    // printf("%.2f %.2f %.2f\n", scales[0], scales[1], scales[2]);
    float3 scale = {scales[3*idx], scales[3*idx+1], scales[3*idx+2]};
    float4 quat = {quats[4*idx+1], quats[4*idx +2], quats[4*idx + 3], quats[4*idx]};
    // printf("0 scale %.2f %.2f %.2f\n", scale.x, scale.y, scale.z);
    // printf("0 quat %.2f %.2f %.2f %.2f\n", quat.w, quat.x, quat.y, quat.z);
    float *cur_cov3d = &(covs3d[6 * idx]);
    compute_cov3d(scale, glob_scale, quat, cur_cov3d);

    // project to 2d with ewa approximation
    float3 cov2d = project_cov3d_ewa(p_world, cur_cov3d, viewmat, fx, fy);
    printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);

    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok) return; // zero determinant
    printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);

    // get the bbox of gaussian in tile coordinates
    float2 center = {ndc2pix(p_proj.x, W), ndc2pix(p_proj.y, H)};
    uint2 bb_min, bb_max;
    get_bbox(center, radius, tile_bounds, bb_min, bb_max);
    uint32_t bb_area = (bb_max.x - bb_min.x) * (bb_max.y - bb_min.y);
    if (bb_area <= 0) {
        printf("bbox outside of bounds\n");
        return;
    }

    num_tiles_hit[idx] = bb_area;
    depths[idx] = p_view.z;
    radii[idx] = (int) ceil(radius);
    xys[2*idx] = center.x;
    xys[2*idx+1] = center.y;
    printf(
        "thread %d x %.2f y %.2f z %.2f, # tiles %d, radii %d\n",
        idx, center.x, center.y, depths[idx], bb_area, radii[idx]
    );
}

// host function to launch the projection in parallel on device
void project_gaussians_forward_impl(
    const int num_points,
    const float *means3d,
    const float *scales,
    const float glob_scale,
    const float *quats,
    const float *viewmat,
    const float *projmat,
    const float fx,
    const float fy,
    const int W,
    const int H,
    const dim3 tile_bounds,
    float *covs3d,
    float *xys,
    float *depths,
    int *radii,
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
        W,
        H,
        tile_bounds,
        covs3d,
        xys,
        depths,
        radii,
        num_tiles_hit
    );
}


// kernel to map each intersection from tile ID and depth to a gaussian
__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float *xys,
    const float *depths,
    const int *radii,
    const uint32_t *cum_tiles_hit,
    const dim3 tile_bounds,
    uint64_t *intersect_ids,
    uint32_t *gaussian_ids
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points)
        return;
    if (radii[idx] <= 0)
        return;
    // get the bbox for gaussian
    uint2 bb_min, bb_max;
    float2 center = {xys[2*idx], xys[2*idx+1]};
    get_bbox(center, radii[idx], tile_bounds, bb_min, bb_max);

    // update the intersection info for all tiles this gaussian hits
    uint32_t cur_idx = (idx == 0) ? 0 : cum_tiles_hit[idx - 1];
    uint64_t depth_id = (uint64_t) *(uint32_t *) &(depths[idx]);
    for (int i = bb_min.y; i < bb_max.y; ++i) {
        for (int j = bb_min.x; j < bb_max.x; ++j) {
            // isect_id is tile ID and depth as uint32
            uint64_t tile_id = i * tile_bounds.x + j;  // tile within image
            intersect_ids[cur_idx] = (tile_id << 32) | depth_id;
            gaussian_ids[cur_idx] = idx;
            ++cur_idx;
        }
    }
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
    cub::DeviceScan::InclusiveSum(sum_ws, sum_ws_bytes, num_tiles_hit, cum_tiles_hit, num_points);
    cudaMalloc(&sum_ws, sum_ws_bytes);
    cub::DeviceScan::InclusiveSum(sum_ws, sum_ws_bytes, num_tiles_hit, cum_tiles_hit, num_points);
    cudaMemcpy(&num_intersects, &(cum_tiles_hit[num_points-1]), sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(sum_ws);
}


// figure out which gaussians, sorted by depth, to render for which tile of the output image
// output gaussian IDs for each tile, sorted by depth, as continguous array
// output start and end indices in this list of gaussians for each tile
void bin_and_sort_gaussians(
    const int num_points,
    const int num_intersects,
    const float *xys,
    const float *depths,
    const int *radii,
    const uint32_t *cum_tiles_hit,
    const dim3 tile_bounds,
    uint32_t *gaussian_ids_sorted,
    uint2 *tile_bins
) {
    // for each intersection map the tile ID and depth to a gaussian ID
    // allocate intermediate results
    uint32_t *gaussian_ids_unsorted;
    uint64_t *isect_ids_unsorted, *isect_ids_sorted;
    cudaMalloc((void**) &gaussian_ids_unsorted, num_intersects * sizeof(uint32_t));
    cudaMalloc((void**) &isect_ids_unsorted, num_intersects * sizeof(uint64_t));
    cudaMalloc((void**) &isect_ids_sorted, num_intersects * sizeof(uint64_t));
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
    cudaFree(isect_ids_unsorted);
    cudaFree(isect_ids_sorted);
    cudaFree(gaussian_ids_unsorted);
}


// kernel function for rendering each gaussian on device
__global__ void render_forward_kernel(
) {
}


// host function to launch parallel rendering of sorted gaussians on device
void render_forward_impl(
) {
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
    // we expect row major matrices as input,
    // glm uses column major
    glm::mat4 viewmat_glm = glm::transpose(glm::make_mat4(viewmat));
    glm::vec4 t = viewmat_glm * glm::vec4(mean3d.x, mean3d.y, mean3d.z, 1.f);
    // printf("viewmat_glm %.2f %.2f %.2f %.2f\n", viewmat_glm[0][0], viewmat_glm[1][1], viewmat_glm[2][2], viewmat_glm[3][3]);
    // printf("t %.2f %.2f %.2f %.2f\n", t[0], t[1], t[2], t[3]);
    // printf("t %.2f %.2f %.2f %.2f\n", t.w, t.x, t.y, t.z);

    // column major
    glm::mat3 J = glm::transpose(glm::mat3(
        fx / t.z, 0.f, -fx * t.x / (t.z * t.z),
        0.f, fy / t.z, -fy * t.y / (t.z * t.z),
        0.f, 0.f, 0.f
    ));
    glm::mat3 W = glm::mat3(
        glm::vec3(viewmat_glm[0]), glm::vec3(viewmat_glm[1]), glm::vec3(viewmat_glm[2])
    );

    glm::mat3 T = J * W;

    glm::mat3 V = glm::mat3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );

    // printf("J %.2f %.2f %.2f\n", J[0][0], J[1][1], J[2][2]);
    // printf("W %.2f %.2f %.2f\n", W[0][0], W[1][1], W[2][2]);
    // printf("V %.2f %.2f %.2f\n", V[0][0], V[1][1], V[2][2]);

    // we only care about the top 2x2 submatrix
    glm::mat3 cov = T * V * glm::transpose(T);
    // add a little blur along axes and save upper triangular elements
    // cov2d[0] = float(cov[0][0]) + 0.1f;
    // cov2d[1] = float(cov[0][1]);
    // cov2d[2] = float(cov[1][1]) + 0.1f;
    return (float3) {float(cov[0][0]) + 0.1f, float(cov[0][1]), float(cov[1][1]) + 0.1f};
}


// device helper to get 3D covariance from scale and quat parameters
__device__ void compute_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, float *cov3d
) {
    // printf("scale %.2f %.2f %.2f\n", scale.x, scale.y, scale.z);
    // printf("quat %.2f %.2f %.2f %.2f\n", quat.w, quat.x, quat.y, quat.z);
    // quat to rotation matrix
    float s = rsqrtf(
        quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
    );
    float w = quat.w * s;
    float x = quat.x * s;
    float y = quat.y * s;
    float z = quat.z * s;

    // glm matrices are column-major
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y),
        2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x),
        2.f * (x * z + w * y), 2.f * (y * z - w * x), 1.f - 2.f * (x * x + x * x)
    );
    // printf("R %.2f %.2f %.2f\n", R[0][0], R[1][1], R[2][2]);

    glm::mat3 S = glm::mat3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
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
