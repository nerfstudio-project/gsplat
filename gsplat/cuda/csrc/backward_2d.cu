#include "backward_2d.cuh"
#include "helpers.cuh"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

inline __device__ void warpSum3(float3& val, cg::thread_block_tile<32>& tile) {
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
}

inline __device__ void warpSum2(float2& val, cg::thread_block_tile<32>& tile) {
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
}

inline __device__ void warpSum(float& val, cg::thread_block_tile<32>& tile) {
    val = cg::reduce(tile, val, cg::plus<float>());
}


__global__ void project_gaussians_backward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float2* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float4 intrins,
    const dim3 img_size,
    const float* __restrict__ cov3d,
    const int* __restrict__ radii,
    const float* __restrict__ ray_transformations,

    // grad input 
    float* __restrict__ v_ray_transformations,
    // const float* __restrict__ v_normal3Ds,

    // grad output
    float3* __restrict__ v_mean3Ds,
    float2* __restrict__ v_scales,
    float4* __restrict__ v_quats,
    float3* __restrict__ v_mean2Ds

) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) return;


    const float* ray_transformation = &(ray_transformations[9 * idx]);
    float* v_ray_transformation = &(v_ray_transformations[9 * idx]);
    // const float* v_normal3D = &(v_normal3Ds[3 * idx]);

    glm::vec3 p_world = glm::vec3(means3d[idx].x, means3d[idx].y, means3d[idx].z);
    float fx = intrins.x;
    float fy = intrins.y;
    float tan_fovx = intrins.z / fx;
    float tan_fovy = intrins.w / fy;
    
    glm::vec3 v_mean3D;
    glm::vec2 v_scale;
    glm::vec4 v_quat;

    build_transform_and_AABB(
        p_world,
        quats[idx],
        scales[idx],
        viewmat,
        intrins,
        radii,
        tan_fovx,
        tan_fovy,
        ray_transformation,

        v_ray_transformation,
        v_mean3D,
        v_scale,
        v_quat,
        v_mean2Ds[idx]
    );

    // build_AABB(
    //     idx,
    //     radii,
    //     img_size.x,
    //     img_size.y,
    //     ray_transformations,

    //     v_mean2Ds,
    //     v_ray_transformations
    // );
    
    // build_H(
    //     p_world,
    //     quats[idx],
    //     scales[idx],
    //     viewmat,
    //     intrins,
    //     tan_fovx,
    //     tan_fovy,
    //     ray_transformation,

    //     // grad input
    //     v_ray_transformation,
    //     // v_normal3D,

    //     // grad output
    //     v_mean3D,
    //     v_scale,
    //     v_quat
    // );

    // Update 
    float3 v_mean3D_float = {v_mean3D.x, v_mean3D.y, v_mean3D.z};
    float2 v_scale_float = {v_scale.x, v_scale.y};
    float4 v_quat_float = {v_quat.x, v_quat.y, v_quat.z, v_quat.w};
    
    v_mean3Ds[idx] = v_mean3D_float;
    v_scales[idx] = v_scale_float;
    v_quats[idx] = v_quat_float;
}


// For simplest 2DGS normal constraint is not used, therefore exclude it for now
__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float* __restrict__ ray_transformations,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,

    // grad input
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,

    // grad_output
    float2* __restrict__ v_mean2D,
    float* __restrict__ v_ray_transformation,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity
) {
    // printf("It is real :) \n");
    auto block = cg::this_thread_block();
    int32_t tile_id = 
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i = 
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j = 
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j + 0.5;
    const float py = (float)i + 0.5;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * img_size.x + j, img_size.x * img_size.y - 1);

    // keep not rasterizing threads around for reading data
    const bool inside = (i < img_size.y && j < img_size.x);

    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float3 buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int bin_final = inside ? final_index[pix_id] : 0;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int2 range = tile_bins[tile_id];
    const int block_size = block.size();
    const int num_batches = (range.y - range.x + block_size - 1) / block_size;

    __shared__ int32_t id_batch[MAX_BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[MAX_BLOCK_SIZE];
    __shared__ float3 rgbs_batch[MAX_BLOCK_SIZE];
    __shared__ float3 Tu_batch[MAX_BLOCK_SIZE];
    __shared__ float3 Tv_batch[MAX_BLOCK_SIZE];
    __shared__ float3 Tw_batch[MAX_BLOCK_SIZE];
    

    // df/d_out for this pixel
    const float3 v_out = v_output[pix_id];
    const float v_out_alpha = v_output_alpha[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const int tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (int b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        const int batch_end = range.y - 1 - block_size * b;
        int batch_size = min(block_size, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            rgbs_batch[tr] = rgbs[g_id];

            //====== 2D Splatting ======//
            Tu_batch[tr] = {ray_transformations[9 * g_id + 0], ray_transformations[9 * g_id + 1], ray_transformations[9 * g_id + 2]};
            Tv_batch[tr] = {ray_transformations[9 * g_id + 3], ray_transformations[9 * g_id + 4], ray_transformations[9 * g_id + 5]};
            Tw_batch[tr] = {ray_transformations[9 * g_id + 6], ray_transformations[9 * g_id + 7], ray_transformations[9 * g_id + 8]};

        }
        // wait for other threads to collext the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (int t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            int valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float vis;
            float gauss_weight_3d;
            float gauss_weight_2d;
            float rho;
            float2 s;
            float2 d;
            float3 k;
            float3 l;
            float3 p;
            float3 Tw;
            if (valid) {
                // Here we compute gaussian weights in forward pass again
                float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                const float2 xy = {xy_opac.x, xy_opac.y};
                const float3 Tu = Tu_batch[t];
                const float3 Tv = Tv_batch[t];
                Tw = Tw_batch[t];
                

                k = {-Tu.x + px * Tw.x, -Tu.y + px * Tw.y, -Tu.z + px * Tw.z};
                l = {-Tv.x + py * Tw.x, -Tv.y + py * Tw.y, -Tv.z + py * Tw.z};
                
                p = cross_product(k, l);
                
                if (p.z == 0.0) {
                   continue;
                } else {
                    s = {p.x / p.z, p.y / p.z};
                }
                gauss_weight_3d = (s.x * s.x + s.y * s.y);
                d = {xy.x - px, xy.y - py};
                gauss_weight_2d = FilterInvSquare * (d.x * d.x + d.y * d.y);

                // compute intersection
                rho = min(gauss_weight_3d, gauss_weight_2d);

                // accumulations
                float sigma = 0.5f * rho;

                vis = __expf(-sigma);
                alpha = min(0.99f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) continue;

            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float v_opacity_local = 0.f;
            float2 v_mean2D_local = {0.f, 0.f};
            float3 v_u_transform_local = {0.f, 0.f, 0.f};
            float3 v_v_transform_local = {0.f, 0.f, 0.f};
            float3 v_w_transform_local = {0.f, 0.f, 0.f};
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                float ra = 1.f / (1.f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
                float v_alpha = 0.f;
                v_rgb_local = {fac * v_out.x, fac * v_out.y, fac * v_out.z};

                const float3 rgb = rgbs_batch[t];
                // contribution from this pixel
                v_alpha += (rgb.x * T - buffer.x * ra) * v_out.x;
                v_alpha += (rgb.y * T - buffer.y * ra) * v_out.y;
                v_alpha += (rgb.z * T - buffer.z * ra) * v_out.z;

                v_alpha += T_final * ra * v_out_alpha;
                // contribution from background pixel
                v_alpha += -T_final * ra * background.x * v_out.x;
                v_alpha += -T_final * ra * background.y * v_out.y;
                v_alpha += -T_final * ra * background.z * v_out.z;
                // update the running sum
                buffer.x += rgb.x * fac;
                buffer.y += rgb.y * fac;
                buffer.z += rgb.z * fac;

                // Helpful reusable temporary variable
                // Need to change this gradient variable name to be readable
                const float v_G = opac * v_alpha;
                float v_depth = 0.0f;

                int32_t g = id_batch[t];

                // ====== 2D Splatting ====== //
                if (gauss_weight_3d <= gauss_weight_2d) {
                    // Update gradients w.r.t. covariance of Gaussian 3x3 (T)
                    const float2 v_s = {
                        v_G * -vis * s.x + v_depth * Tw.x,
                        v_G * -vis * s.y + v_depth * Tw.y
                    };
                    const float3 v_z_w_transform = {s.x, s.y, 1.0};
                    const float v_sz_pz = v_s.x / p.z;
                    const float v_sy_pz = v_s.y / p.z;
                    const float3 v_intersect = {v_sz_pz, v_sy_pz, -(v_sz_pz * s.x + v_sy_pz * s.y)};
                    const float3 v_h_u = cross_product(l, v_intersect);
                    const float3 v_h_v = cross_product(v_intersect, k);

                    const float3 v_u_transform = {-v_h_u.x, -v_h_u.y, -v_h_u.z};
                    const float3 v_v_transform = {-v_h_v.x, -v_h_v.y, -v_h_v.z};
                    const float3 v_w_transform = {
                        px * v_h_u.x + py * v_h_v.x + v_depth * v_z_w_transform.x,
                        px * v_h_u.y + py * v_h_v.y + v_depth * v_z_w_transform.y,
                        px * v_h_u.z + py * v_h_v.z + v_depth * v_z_w_transform.z
                    };
                    v_u_transform_local = {v_u_transform.x, v_u_transform.y, v_u_transform.z};
                    v_v_transform_local = {v_v_transform.x, v_v_transform.y, v_v_transform.z};
                    v_w_transform_local = {v_w_transform.x, v_w_transform.y, v_w_transform.z};

                } else {
                    // Update gradient w.r.t center of Gaussian 2D mean position
                    const float v_G_ddelx = -vis * FilterInvSquare * d.x;
                    const float v_G_ddely = -vis * FilterInvSquare * d.y;
                    v_mean2D_local = {v_G * v_G_ddelx, v_G * v_G_ddely};

                }
                v_opacity_local = vis * v_alpha;
            }
            warpSum3(v_rgb_local, warp);
            warpSum(v_opacity_local, warp);
            warpSum2(v_mean2D_local, warp);
            warpSum3(v_u_transform_local, warp);
            warpSum3(v_v_transform_local, warp);
            warpSum3(v_w_transform_local, warp);
            if (warp.thread_rank() == 0) {
                // printf("Here!!! \n");
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3 * g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3 * g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3 * g + 2, v_rgb_local.z);
                if (gauss_weight_3d <= gauss_weight_2d) {
                    float* v_ray_transformation_ptr = (float*)(v_ray_transformation);
                    atomicAdd(v_ray_transformation_ptr + 9 * g + 0, v_u_transform_local.x);
                    atomicAdd(v_ray_transformation_ptr + 9 * g + 1, v_u_transform_local.y);
                    atomicAdd(v_ray_transformation_ptr + 9 * g + 2, v_u_transform_local.z);
                    atomicAdd(v_ray_transformation_ptr + 9 * g + 3, v_v_transform_local.x);
                    atomicAdd(v_ray_transformation_ptr + 9 * g + 4, v_v_transform_local.y);
                    atomicAdd(v_ray_transformation_ptr + 9 * g + 5, v_v_transform_local.z);
                    atomicAdd(v_ray_transformation_ptr + 9 * g + 6, v_w_transform_local.x);
                    atomicAdd(v_ray_transformation_ptr + 9 * g + 7, v_w_transform_local.y);
                    atomicAdd(v_ray_transformation_ptr + 9 * g + 8, v_w_transform_local.z);

                } else {
                    float* v_mean2D_ptr = (float*)(v_mean2D);
                    atomicAdd(v_mean2D_ptr + 2 * g + 0, v_mean2D_local.x);
                    atomicAdd(v_mean2D_ptr + 2 * g + 1, v_mean2D_local.y);
                }
                atomicAdd(v_opacity + g, v_opacity_local);
            }
        }
    }
}


__device__ void build_H(
    const glm::vec3 & p_world,
    const float4 & quat,
    const float2 & scale,
    const float* viewmat,
    const float4 & intrins,
    float tan_fovx,
    float tan_fovy,
    const float* ray_transformation,

    // grad input
    const float* v_ray_transformation,
    // const float* v_normal3D,

    // grad output
    glm::vec3 & v_mean3D,
    glm::vec2 & v_scale,
    glm::vec4 & v_quat
) {

    // camera information
    const glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[1], viewmat[2],
        viewmat[4], viewmat[5], viewmat[6],
        viewmat[8], viewmat[9], viewmat[10]
    ); // viewmat

    const glm::vec3 cam_pos = glm::vec3(viewmat[3], viewmat[7], viewmat[11]); // camera center
    glm::mat4 P = glm::mat4(
        intrins.x, 0.0, 0.0, 0.0,
        0.0, intrins.y, 0.0, 0.0,
        intrins.z, intrins.w, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0
    );

    glm::mat3 S = scale_to_mat({scale.x, scale.y, 1.0f}, 1.0f);
    glm::mat3 R = quat_to_rotmat(quat);
    glm::mat3 RS = R * S;
    glm::vec3 p_view = W * p_world + cam_pos;
    glm::mat3 M = glm::mat3(W * RS[0], W * RS[1], p_view);

    glm::mat4x3 v_T = glm::mat4x3(
        v_ray_transformation[0], v_ray_transformation[1], v_ray_transformation[2],
        v_ray_transformation[3], v_ray_transformation[4], v_ray_transformation[5],
        v_ray_transformation[6], v_ray_transformation[7], v_ray_transformation[8],
        0.0, 0.0, 0.0
    );

    glm::mat3x4 v_M_aug = glm::transpose(P) * glm::transpose(v_T);
    glm::mat3 v_M = glm::mat3(
        glm::vec3(v_M_aug[0]),
        glm::vec3(v_M_aug[1]),
        glm::vec3(v_M_aug[2])
    );

    glm::mat3 W_t = glm::transpose(W);
    glm::mat3 v_RS= W_t * v_M;
    glm::vec3 v_RS0 = v_RS[0];
    glm::vec3 v_RS1 = v_RS[1];
    glm::vec3 v_intersectw = v_RS[2];
    // glm::vec3 v_tn = W_t * glm::vec3(v_normal3D[0], v_normal3D[1], v_normal3D[2]);
    glm::vec3 v_tn = W_t * glm::vec3(0.0, 0.0, 0.0);

    
    glm::mat3 v_R = glm::mat3(
        v_RS0 * glm::vec3(scale.x),
        v_RS1 * glm::vec3(scale.y),
        v_tn
    );

    v_quat = quat_to_rotmat_vjp(quat, v_R);
    v_scale = glm::vec2(
        (float)glm::dot(v_RS0, R[0]),
        (float)glm::dot(v_RS1, R[1])
    );

    v_mean3D = v_intersectw;
}


__device__ void build_AABB(
    int idx,
    const int * radii,
    const float W, 
    const float H,
    const float * ray_transformations,

    // grad output
    float3 * v_mean2Ds,
    float * v_ray_transformations
) {
    const float* ray_transformation = ray_transformations + 9 * idx;

    const float3 v_mean2D = v_mean2Ds[idx];
    glm::mat4x3 T = glm::mat4x3(
        ray_transformation[0], ray_transformation[1], ray_transformation[2],
        ray_transformation[3], ray_transformation[4], ray_transformation[5],
        ray_transformation[6], ray_transformation[7], ray_transformation[8],
        ray_transformation[6], ray_transformation[7], ray_transformation[8]
    );

    float d = glm::dot(glm::vec3(1.0, 1.0, -1.0), T[3] * T[3]);
    glm::vec3 f = glm::vec3(1.0, 1.0, -1.0) * (1.0f / d);


    glm::vec3 p = glm::vec3(
        glm::dot(f, T[0] * T[3]),
        glm::dot(f, T[1] * T[3]),
        glm::dot(f, T[2] * T[3])
    );

    glm::vec3 v_T0 = v_mean2D.x * f * T[3];
    glm::vec3 v_T1 = v_mean2D.y * f * T[3];
    glm::vec3 v_T3 = v_mean2D.x * f * T[0] + v_mean2D.y * f * T[1];
    glm::vec3 v_f = (v_mean2D.x * T[0] * T[3]) + (v_mean2D.y * T[1] * T[3]);
    float v_d = glm::dot(v_f, f) * (-1.0 / d);
    glm::vec3 dd_dT3 = glm::vec3(1.0, 1.0, -1.0) * T[3] * 2.0f;
    v_T3 += v_d * dd_dT3;
    v_ray_transformations[9 * idx + 0] += v_T0.x;
    v_ray_transformations[9 * idx + 1] += v_T0.y;
    v_ray_transformations[9 * idx + 2] += v_T0.z;
    v_ray_transformations[9 * idx + 3] += v_T1.x;
    v_ray_transformations[9 * idx + 4] += v_T1.y;
    v_ray_transformations[9 * idx + 5] += v_T1.z;
    v_ray_transformations[9 * idx + 6] += v_T3.x;
    v_ray_transformations[9 * idx + 7] += v_T3.y;
    v_ray_transformations[9 * idx + 8] += v_T3.z;

}

__device__ void build_transform_and_AABB(
    const glm::vec3& p_world, 
    const float4& quat,
    const float2& scale,
    const float* viewmat,
    const float4& intrins,
    const int* radii,
    float tan_fovx,
    float tan_fovy,
    const float* ray_transformation,

    // grad 
    float* v_ray_transformation,
    glm::vec3& v_mean3D,
    glm::vec2& v_scale,
    glm::vec4& v_quat,
    float3& v_mean2D
) {

    glm::mat3 M = glm::mat3(
        ray_transformation[0], ray_transformation[1], ray_transformation[2],
        ray_transformation[3], ray_transformation[4], ray_transformation[5],
        ray_transformation[6], ray_transformation[7], ray_transformation[8]
    );

    float distance = glm::dot(glm::vec3(1.0, 1.0, -1.0), M[2] * M[2]);
    glm::vec3 temp_point = glm::vec3(1.0f, 1.0f, -1.0f);
    glm::vec3 f = (1.0f / distance) * temp_point; 

    glm::vec3 p = glm::vec3(
        glm::dot(f, M[0] * M[2]),
        glm::dot(f, M[1] * M[2]),
        glm::dot(f, M[2] * M[2])
    );

    glm::vec3 v_T0 = v_mean2D.x * f * M[2];
    glm::vec3 v_T1 = v_mean2D.y * f * M[2];
    glm::vec3 v_T3 = v_mean2D.x * f * M[0] + v_mean2D.y * f * M[1];
    glm::vec3 v_f = (v_mean2D.x * M[0] * M[2]) + (v_mean2D.y * M[1] * M[2]);
    float v_distance = glm::dot(v_f, f) * (-1.0 / distance);
    glm::vec3 dd_dT3 = glm::vec3(1.0, 1.0, -1.0) * M[2] * 2.0f;
    v_T3 += v_distance * dd_dT3;
    v_ray_transformation[0] += v_T0.x;
    v_ray_transformation[1] += v_T0.y;
    v_ray_transformation[2] += v_T0.z;
    v_ray_transformation[3] += v_T1.x;
    v_ray_transformation[4] += v_T1.y;
    v_ray_transformation[5] += v_T1.z;
    v_ray_transformation[6] += v_T3.x;
    v_ray_transformation[7] += v_T3.y;
    v_ray_transformation[8] += v_T3.z;

    //====== ray transformation gradient ======//
    // camera information
    const glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[1], viewmat[2],
        viewmat[4], viewmat[5], viewmat[6],
        viewmat[8], viewmat[9], viewmat[10]
    ); // viewmat

    const glm::vec3 cam_pos = glm::vec3(viewmat[3], viewmat[7], viewmat[11]); // camera center
    glm::mat3 P = glm::mat3(
        intrins.x, 0.0, 0.0,
        0.0, intrins.y, 0.0,
        intrins.z, intrins.w, 1.0
    );

    glm::mat3 S = scale_to_mat({scale.x, scale.y, 1.0f}, 1.0f);
    glm::mat3 R = quat_to_rotmat(quat);
    glm::mat3 RS = R * S;
    glm::vec3 p_view = W * p_world + cam_pos;
    // glm::mat3 M = glm::mat3(W * RS[0], W * RS[1], p_view);

    glm::mat3 v_T = glm::mat3(
        v_ray_transformation[0], v_ray_transformation[1], v_ray_transformation[2],
        v_ray_transformation[3], v_ray_transformation[4], v_ray_transformation[5],
        v_ray_transformation[6], v_ray_transformation[7], v_ray_transformation[8]
    );

    glm::mat3 v_M_aug = glm::transpose(P) * glm::transpose(v_T);
    glm::mat3 v_M = glm::mat3(
        glm::vec3(v_M_aug[0]),
        glm::vec3(v_M_aug[1]),
        glm::vec3(v_M_aug[2])
    );

    glm::mat3 W_t = glm::transpose(W);
    glm::mat3 v_RS = W_t * v_M;
    glm::vec3 v_RS0 = v_RS[0];
    glm::vec3 v_RS1 = v_RS[1];
    glm::vec3 v_intersectw = v_RS[2];
    // glm::vec3 v_tn = W_t * glm::vec3(v_normal3D[0], v_normal3D[1], v_normal3D[2]);
    glm::vec3 v_tn = W_t * glm::vec3(0.0, 0.0, 0.0);

    
    glm::mat3 v_R = glm::mat3(
        v_RS0 * glm::vec3(scale.x),
        v_RS1 * glm::vec3(scale.y),
        v_tn
    );

    v_quat = quat_to_rotmat_vjp(quat, v_R);
    v_scale = glm::vec2(
        (float)glm::dot(v_RS0, R[0]),
        (float)glm::dot(v_RS1, R[1])
    );

    v_mean3D = v_intersectw;
}