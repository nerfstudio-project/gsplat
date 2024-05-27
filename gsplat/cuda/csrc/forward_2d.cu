#include "forward_2d.cuh"
#include "helpers.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
// WZ: So the only difference between 2DGS and 3DGS here is that 
// we compute the transformation matrix H with 2DGS and no covariance projection for 2DGS
__global__ void project_gaussians_forward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float4 intrins,
    const dim3 img_size,
    const dim3 tile_bounds,
    const unsigned block_width,
    const float clip_thresh,
    float* __restrict__ conv3d,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    int32_t* __restrict__ num_tiles_hit,
    float* __restrict__ transMats
) {
    // printf("Here\n");
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    } 
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    float3 p_world = means3d[idx];
    float3 p_view;

    bool not_ok = clip_near_plane(p_world, viewmat, p_view, clip_thresh);
    // printf("not_ok: %.2f \n", not_ok);
    if (not_ok) return;


    float3 scale = scales[idx];
    float4 quat = quats[idx];


    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;
    float tan_fovx = 0.5 * img_size.x / fx;
    float tan_fovy = 0.5 * img_size.y / fy;

    // printf("img_size.x: %.3f \n", img_size.x);
    // printf("img_size.y: %.3f \n", img_size.y);
    // printf("fx: %.2f, fy: %.2f, tan_fovx: %.2f, tan_fovy: %.2f \n", fx, fy, tan_fovx, tan_fovy);
    // In 3DGS we build 3D covariance matrix and project 
    // Here we build transformation matrix H in 2DGS
    //====== 2DGS Specific ======//
    float* cur_transMats = &(transMats[9 * idx]); //TODO: indexing may have bug
    bool ok;
    float3 normal;

    ok = build_H(
        p_world,
        intrins,
        scale,
        quat,
        viewmat,
        fx,
        fy, 
        tan_fovx,
        tan_fovy,
        cur_transMats,
        normal
    );

    

    if (!ok) return;
    // transMat = transMats + idx * 9;

    int start_index = 0;
    // for (int i = 0; i < 9; ++i) {
    //     printf("cur_transMats[%d] = %f\n", start_index + i, cur_transMats[start_index + i]);
    // }

    // printf("%.2f, %.2f, %.2f \n %.2f, %.2f, %.2f \n %.2f, %.2f, %.2f \n", \
    //     cur_transMats[start_index], cur_transMats[start_index + 1], cur_transMats[start_index + 2], \
    //     cur_transMats[start_index + 3], cur_transMats[start_index + 4], cur_transMats[start_index + 5], \
    //     cur_transMats[start_index + 6], cur_transMats[start_index + 7], cur_transMats[start_index + 8]);

    float2 center;
    float2 extent;
    ok = build_AABB(cur_transMats, center, extent);
    if (!ok) return;
    // printf("%d \n", ok);

    float truncated_R = 3.f;

    float radius = ceil(truncated_R * max(max(extent.x, extent.y), FilterSize));

    // compte the projected mean
    center = project_pix({fx, fy}, p_view, {cx, cy});
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max, block_width);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) return;


    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    xys[idx] = center;
    // compoensation[idx] = comp; // Used in project 3D covariance to 2D; not used in 2DGS
}


// Kernel to map each intersection from tile ID and depth to a gaussian
// writes output to isect_ids and gaussian_ids
// TODO: Hmm what does this function do?
__global__ void map_gaussian_to_intersects(
    const int num_points,
    const float2* __restrict__ xys,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const int32_t* __restrict__ cum_tiles_hit,
    const dim3 tile_bounds,
    const unsigned block_width,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ gaussian_ids
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points) return;

    if (radii[idx] <= 0) return;

    // get the tile bbox for gaussian
    uint2 tile_min, tile_max;
    float2 center = xys[idx];
    get_tile_bbox(center, radii[idx], tile_bounds, tile_min, tile_max, block_width);

    // update the intersection info for all tiles this gaussian hits.
    int32_t cur_idx = (idx == 0) ? 0 : cum_tiles_hit[idx - 1];
    int64_t depth_id = (int64_t) * (int32_t *)&(depths[idx]);
    for (int i = tile_min.y; i < tile_max.y; ++i) {
        for (int j = tile_min.x; j < tile_max.x; ++j) {
            // isect_id is tile ID and depth as int32
            int64_t tile_id = i * tile_bounds.x + j; // tile within image
            isect_ids[cur_idx] = (tile_id << 32) | depth_id; // tile | depth id
            gaussian_ids[cur_idx] = idx;
            ++cur_idx; // handles gaussians that hit more than one tile
        }
    }
}

// Kernel to map sorted intersection IDs to tile bins
// expect that intersection IDs are sorted by increasing tile ID
// i.e. intersections of a tile are in contiguous chunks
__global__ void get_tile_bin_edges(
    const int num_intersects,
    const int64_t* __restrict__ isect_ids_sorted,
    int2* __restrict__ tile_bins
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_intersects) {
        return;
    }
    // save the indices where the tile_id changes
    int32_t cur_tile_idx = (int32_t)(isect_ids_sorted[idx] >> 32);
    if (idx == 0 || idx == num_intersects - 1) {
        if (idx == 0) {
            tile_bins[cur_tile_idx].x = 0;
        }
        if (idx == num_intersects - 1) {
            tile_bins[cur_tile_idx].y = num_intersects;
        }
    }
    if (idx == 0) {
        return;
    }
    int32_t prev_tile_idx = (int32_t)(isect_ids_sorted[idx - 1] >> 32);
    if (prev_tile_idx != cur_tile_idx) {
        tile_bins[prev_tile_idx].y = idx;
        tile_bins[cur_tile_idx].x  = idx;
        return;
    }
}

// Kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
// __global__ void nd_rasterize_forward(
//     const dim3 tile_bounds,
//     const dim3 img_size,
//     const unsigned channels,
//     const int32_t* __restrict__ gaussian_ids_sorted,
//     const int2* __restrict__ tile_bins,
//     const float2* __restrict__ xys,
//     const float* __restrict__ colors,
//     const float* __restrict__ opacities,
//     float* __restrict__ final_Ts,
//     int* __restrict__ final_index,
//     float* __restrict__ out_img,
//     const float* __restrict__ background
// ) {
// //     auto block = cg::this_thread_block();
// //     int32_t tile_id = 
// //         block.group_index().y * tile_bounds.x + block.group_index().x;
// //     unsigned i = 
// //         block.group_index().y * block.group_dim().y + block.thread_index().y;
// //     unsigned j =
// //         block.group_index().x * block.group_dim().x + block.thread_index().x;

// //     float px = (float)j + 0.5;
// //     float py = (float)i + 0.5;
// //     int32_t pix_id = i * img_size.x + j;

// //     // keep not rasterizing threads around for reading data
// //     bool inside = (i < img_size.y && j < img_size.x);
// //     bool done = !inside;

// //     int2 range = tile_bins[tile_id];
// //     const int block_size = block.size();
// //     int num_batches = (range.y - range.x + block_size - 1) / block_size;

// //     // extern: declare a variable or function that is defined in another source file
// //     // or in the host code. It signifies the shared memory is not statically allocated 
// //     // within the kernel but will be allocated dynamically when the kernel is launched.
// //     extern __shared__ int s[];
// //     int32_t* id_batch = (int32_t*)s;
// //     float3* xy_opacity_batch = (float3*)&id_batch[block_size];
// //     #pragma unroll
// //     for(int c = 0; c < channels; ++c) {

// //     }
// }

__global__ void rasterize_forward(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float* __restrict__ transMats,
    const float3* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ final_Ts,
    int* __restrict__ final_index,
    float3* __restrict__ out_img,
    const float3& __restrict__ background
) {
    // each thread draws one pixel, but also timeshares caching gaussians 
    // in a shared tile

    auto block = cg::this_thread_block();
    int32_t tile_id = 
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i = 
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j = 
        block.group_index().x * block.group_dim().x + block.thread_index().x;
    
    float px = (float)j + 0.5;
    float py = (float)i + 0.5;
    int32_t pix_id = i * img_size.x + j;

    // return if out of bounds;
    // keep not rasterizing threads around for reading data
    bool inside = (i < img_size.y && j < img_size.x);
    bool done = !inside;
    
    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    int num_batches = (range.y - range.x + block_size - 1) / block_size;

    // The three lines declare memory arrays within the CUDA kernel function.
    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[MAX_BLOCK_SIZE];

    //====== 2D Splatting ======//
    __shared__ float3 Tu_batch[MAX_BLOCK_SIZE];
    __shared__ float3 Tv_batch[MAX_BLOCK_SIZE];
    __shared__ float3 Tw_batch[MAX_BLOCK_SIZE];

    // current visibility left to render
    float T = 1.f;
    // index of most recent gaussian to write to this thread's pixel
    int cur_idx = 0;

    // collect and process batches of gaussians.
    // each thread loads one gaussian at a time brefore rasterizing its
    // designated pixel

    // Functinality: Get the thread rank within the block. Each thread within a CUDA block
    // has a unique thread rank, which is an integer value ranging from 0 to the block size 
    // minus one. This value is used to determine the index of the data elements to be processed 
    // within shared memory arrays or to perform thread-specific computations.
    int tr = block.thread_rank();
    float3 pix_out = {0.f, 0.f, 0.f};
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        int batch_start = range.x + block_size * b;
        int idx = batch_start + tr;
        if (idx < range.y) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            Tu_batch[tr] = {transMats[9 * g_id + 0], transMats[9 * g_id + 1], transMats[9 * g_id + 2]};
            Tv_batch[tr] = {transMats[9 * g_id + 3], transMats[9 * g_id + 4], transMats[9 * g_id + 5]};
            Tw_batch[tr] = {transMats[9 * g_id + 6], transMats[9 * g_id + 7], transMats[9 * g_id + 8]};
        }

        // Wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        int batch_size = min(block_size, range.y - batch_start);
        for (int t = 0; (t < batch_size) && !done; ++t) {
            const float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;

            // ================================== place where 2D Gaussian changes take place ========== //
            float3 Tu = Tu_batch[t];
            float3 Tv = Tv_batch[t];
            float3 Tw = Tw_batch[t];

            float3 k = {-Tu.x + px * Tw.x, -Tu.y + px * Tw.y, -Tu.z + px * Tw.z};
            float3 l = {-Tv.x + py * Tw.x, -Tv.y + py * Tw.y, -Tv.z + py * Tw.z};

            // printf("px: %.2f, py: %.2f \n", px, py);

            // printf("Tv: %.2f, %.2f, %.2f \n", Tv.x, Tv.y, Tv.z);
            // printf("Tw: %.2f, %.2f, %.2f \n", Tw.x, Tw.y, Tw.z);
            // cross product of two planes is a line
            float3 p = cross_product(k, l);

            // printf("k: %.2f, %.2f, %.2f \n l: %.2f, %.2f, %.2f \n", k.x, k.y, k.z, l.x, l.y, l.z);
            // printf("l: %.2f, %.2f, %.2f \n", l.x, l.y, l.z);
            // printf("p is: %.2f, %.2f, %.2f \n", p.x, p.y, p.z);

            // There is no intersection
            // printf("forward p.z: %.2d \n", p.z);
            // printf("p: %.2f, %.2f, %.2f \n", p.x, p.y, p.z);
            // printf("Tu: %.2f, %.2f, %.2f \n \
            //         Tv: %.2f, %.2f, %.2f \n \
            //         Tw: %.2f, %.2f, %.2f \n \
            //         px: %.2f, py: %.2f \n \
            //         k: %.2f, %.2f, %.2f \n \
            //         l: %.2f, %.2f, %.2f \n \
            //         p: %.2f, %.2f, %.2f \n", 
            //         Tu.x, Tu.y, Tu.z, 
            //         Tv.x, Tv.y, Tv.z, 
            //         Tw.x, Tw.y, Tw.z, 
            //         px, py,
            //         k.x, k.y, k.z,
            //         l.x, l.y, l.z,
            //         p.x, p.y, p.z);
            float2 s;
            if (p.z == 0.0) { 
                // s = {px, py};
                // printf("Here \n");
                continue;
            } else {
                // printf("Here \n");
                // 3D homogeneous point to 2d point on the splat
                s = {p.x / p.z, p.y / p.z};
                // printf("p.z is not zero \n");
                // printf("s: %.2f, %.2f \n", s.x, s.y);
            }
            // printf("p.z: %.2f \n", p.z);
            
            // printf("s: %.2f, %.2f \n", s.x, s.y);
            // 3D distance. Compute Mahalanobis distance in the canonical splat space
            float rho_3d = (s.x * s.x + s.y * s.y);

            // Low pass filter Botsch et al. [2005]. Eq. (11) from 2DGS paper
            float x = xy_opac.x;
            float y = xy_opac.y;
            float2 d = {x - px, y - py};
            // 2d screen distance
            float rho_2d = FilterInvSquare * (d.x * d.x + d.y * d.y);
            float rho = min(rho_3d, rho_2d);
            // if (rho_3d < rho_2d) {
            //     printf("rho_3d < rho_2d \n");
            // }
            // float rho = rho_3d;

            //====== Depth, Normal calculation ======//

            const float sigma = 0.5f * rho;
            const float alpha = min(0.999f, opac * __expf(-sigma));
            // printf("alpha: %.2f \n", alpha);

            if (sigma < 0.f || alpha < 1.f / 255.f) {
                // printf("Here! \n");
                continue;
            }

            const float next_T = T * (1.f  -alpha);
            if (next_T <= 1e-4f) {
                // this pixel is done
                // we want to render the last gaussian that contributes and note
                // that here idx > range.x so we don't underflow

                done = true;
                break;
            }

            //====== Render Depth distortion map ======//

            //====== Render Normal Map ======//
            
            //====== Render Depth Map ======//



            // ========================================================================================= //
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
        final_index[pix_id] = cur_idx; // index of in bin of last gaussian in this pixel
        float3 final_color;
        final_color.x = pix_out.x + T * background.x;
        final_color.y = pix_out.y + T * background.y;
        final_color.z = pix_out.z + T * background.z;
        out_img[pix_id] = final_color;
    } 
}



// Device helper to build the per gaussian transformation matrix
__device__ bool build_H(
    const float3& __restrict__ mean3d,
    const float4 __restrict__ intrins,
    const float3 __restrict__ scale,
    const float4 __restrict__ quat,
    const float* __restrict__ viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy,
    float* transMat,
    float3 &normal
) {
    const glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[1], viewmat[2],
        viewmat[4], viewmat[5], viewmat[6],
        viewmat[8], viewmat[9], viewmat[10]
    );

    const glm::vec3 cam_pos = glm::vec3(viewmat[3], viewmat[7], viewmat[11]); // TODO: the viewmat format is a source of bug, transpose or not?
    const glm::mat4 P = glm::mat4(
        intrins.x, 0.0, 0.0, 0.0,
        0.0, intrins.y, 0.0, 0.0,
        intrins.z, intrins.w, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0
    );

    // printf("intrins: %.2f, %.2f, %.2f, %.2f \n", intrins.x, intrins.y, intrins.z, intrins.w);
    const glm::vec3 p_world = glm::vec3(mean3d.x, mean3d.y, mean3d.z);


    // printf("p_world: %.2f, %.2f, %.2f \n", p_world.x, p_world.y, p_world.z);
    // printf("cam_pos: %.2f, %.2f, %.2f \n", cam_pos.x, cam_pos.y, cam_pos.z);
    // printf("viewmat: %.2f, %.2f, %.2f, %.2f \n \
    //                 %.2f, %.2f, %.2f, %.2f \n \
    //                 %.2f, %.2f, %.2f, %.2f \n \
    //                 %.2f, %.2f, %.2f, %.2f \n", \
    //                 viewmat[0], viewmat[1], viewmat[2], viewmat[3], \
    //                 viewmat[4], viewmat[5], viewmat[6], viewmat[7], \
    //                 viewmat[8], viewmat[9], viewmat[10], viewmat[11], \
    //                 viewmat[12], viewmat[13], viewmat[14], viewmat[15]);
    glm::vec3 p_view = W * p_world + cam_pos;
    // printf("p_world: %.2f, %.2f, %.2f \n p_view: %.2f, %.2f, %.2f \n", \
    //         p_world.x, p_world.y, p_world.z, \
    //         p_view.x, p_view.y, p_view.z);
    glm::mat3 R = quat_to_rotmat(quat) * scale_to_mat({scale.x, scale.y, 1.0f}, 1.0f);
    glm::mat3 M = glm::mat3(W * R[0], W * R[1], p_view);
    glm::vec3 tn = W * R[2];
    // printf("tn: %.2f, %.2f, %.2f \n", tn.x, tn.y, tn.z);
    float cos = glm::dot(-tn, p_view);

    glm::mat4x3 T = glm::transpose(P * glm::mat3x4(
        glm::vec4(M[0], 0.0),
        glm::vec4(M[1], 0.0),
        glm::vec4(M[2], 1.0)
    ));

    transMat[0] = T[0].x;
	transMat[1] = T[0].y;
	transMat[2] = T[0].z;
	transMat[3] = T[1].x;
	transMat[4] = T[1].y;
	transMat[5] = T[1].z;
	transMat[6] = T[2].x;
	transMat[7] = T[2].y;
	transMat[8] = T[2].z;

    // printf("transMat: %.2f, %.2f, %.2f \n %.2f, %.2f, %.2f \n %.2f, %.2f, %.2f\n", \
    //         transMat[0], transMat[1], transMat[2], \
    //         transMat[3], transMat[4], transMat[5], \
    //         transMat[6], transMat[7], transMat[8]);
	normal = {tn.x, tn.y, tn.z};
	return true;
}


// Compute the bounding box of the 2D Gaussian and its center,
// where the center of the bounding box is used to create a low pass filter
// in the image plane
__device__ bool build_AABB(
    const float *transMat,
    float2 & center,
    float2 & extent
) {
    glm::mat4x3 T = glm::mat4x3(
        transMat[0], transMat[1], transMat[2],
        transMat[3], transMat[4], transMat[5],
        transMat[6], transMat[7], transMat[8],
        transMat[6], transMat[7], transMat[8]
    );


    float d = glm::dot(glm::vec3(1.0, 1.0, -1.0), T[3] * T[3]);

    // printf("%d \n", d);
    if (d == 0.0f) {
        // printf("Here \n");
        return false;
    }

    glm::vec3 f = glm::vec3(1.0, 1.0, -1.0) * (1.0f / d);

    glm::vec3 p = glm::vec3(
        glm::dot(f, T[0] * T[3]),
        glm::dot(f, T[1] * T[3]),
        glm::dot(f, T[2] * T[3])
    );

    glm::vec3 h0 = p * p - 
        glm::vec3(
            glm::dot(f, T[0] * T[0]),
            glm::dot(f, T[1] * T[1]),
            glm::dot(f, T[2] * T[2])
        );

    glm::vec3 h = sqrt(max(glm::vec3(0.0), h0)) + glm::vec3(0.0, 0.0, 1e-2);
    center = {p.x, p.y};
    extent = {h.x, h.y};
    return true;
}


// Device helper to get 3D covariance from scale and quat parameters
__device__ void scale_rot_to_cov3d(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    float *cov3d
) {
    glm::mat3 R = quat_to_rotmat(quat);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    
    glm::mat3 M = R * S;
    glm::mat3 tmp = M * glm::transpose(M);

    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];

}