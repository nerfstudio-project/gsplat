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
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float4 intrins,
    const dim3 img_size,
    const float* __restrict__ cov3d,
    const int* __restrict__ radii,
    const float* __restrict__ transMats,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float* __restrict__ v_transMats,
    float3* __restrict__ v_mean3d,
    float3* __restrict__ v_scale,
    float4* __restrict__ v_quat
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) return;

    float3 p_world = means3d[idx];
    float fx = intrins.x;
    float fy = intrins.y;
    float3 p_view = transform_4x3(viewmat, p_world);
    // get v_mean3d from v_xy
    v_mean3d[idx] = transform_4x3_rot_only_transposed(
        viewmat,
        project_pix_vjp({fx, fy}, p_view, v_xy[id])
    );
    
    // get z gradient contribution to mean3d gradient
    // z = viewmat[8] * mean3d.x + viewmat[9] * mean3d.y + viewmat[10] * 
    // mean3d.z = viewmat[11]
    float v_z = v_depth[idx];
    v_mean3d[idx].x += viewmat[8] * v_z;
    v_mean3d[idx].y += viewmat[9] * v_z;
    v_mean3d[idx].z += viewmat[10] * v_z;

    // get v_cov2d
    cov2d_to_conic_vjp(conics[idx], v_conic[idx], v_cov2d[idx]);
    cov2d_to_compensation_vjp(compensation[idx], conics[idx], v_compensation[idx], v_cov2d[idx]);
    // get v_cov3d (and v_mean3d contribution)
    project_cov3d_ewa_vjp(
        p_world,
        &(cov3d[6 * idx]),
        viewmat,
        fx, 
        fy,
        v_cov2d[idx],
        v_mean3d[idx],
        &(v_cov3d[6 * idx])
    );
    // get v_scale and v_quat
    scale_rot_to_cov3d_vjp(
        scales[idx],
        glob_scale,
        quats[idx],
        &(v_cov3d[6 * idx]),
        v_scale[idx],
        v_quat[idx]
    );
}

__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const float2* __restrict__ xys,
    const float* __restrict__ transMats,
    const float3* __restrict__ rgbs,
    const float* __restrict__ opacities,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float2* __restrict__ v_xy_abs,
    float* __restrict__ v_transMats,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity
) {
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
    __shared__ float3 conic_batch[MAX_BLOCK_SIZE];
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
            conic_batch[tr] = conics[g_id];
            rgbs_batch[tr] = rgbs[g_id];

            //====== 2D Splatting ======//
            Tu_batch[tr] = {transMats[9 * g_id + 0], transMats[9 * g_id + 1], transMats[9 * g_id + 2]};
            Tv_batch[tr] = {transMats[9 * g_id + 3], transMats[9 * g_id + 4], transMats[9 * g_id + 5]};
            Tw_batch[tr] = {transMats[9 * g_id + 6], transMats[9 * g_id + 7], transMats[9 * g_id + 8]};

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
            float2 delta;
            float3 conic;
            float vis;
            if (valid) {
                // conic = conic_batch[t];

                // Here we compute gaussian weights in forward pass again
                float3 xy_opac = xy_opacity_batch[t];
                
                const float2 xy = {xy_opac.x, xy_opac.y};
                const float2 Tu = Tu_batch[t];
                const float2 Tv = Tv_batch[t];
                const float2 Tw = Tw_batch[t];
                
                float3 k = px * Tw - Tu;
                float3 l = py * Tw - Tv;
                float3 p = crossProduce(k, l);
                if (p.z == 0.0) continue;
                float2 s = {p.x / p.z, p.y / p.z};
                float rho3d = (s.x * s.x + s.y * s.y);
                float2 d = {xy.x - px, xy.y - py};
                float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);

                // compute intersection
                float rho = min(rho3d, rho2d);

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
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float2 v_xy_abs_local = 0.f;
            float v_opacity_local = 0.f;
            // initialize everything to 0, only set if the lane is valid
            // TODO (WZ) what is the lane?
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
                const float dL_dG = opac * v_alpha;
                float dL_dz = 0.0f;

                // ====== 2D Splatting ====== //
                if (rho3d <= rho2d) {
                    // Update gradients w.r.t. covariance of Gaussian 3x3 (T)
                    const float dL_ds = {
                        dL_dG * -vis * s.x + dL_dz * Tw.x,
                        dL_dG * -vis * s.y + dL_dz * Tw.y
                    };
                    const float3 dz_dTw = {s.x, s.y, 1.0};
                    const float dsx_pz = dL_ds.x / p.z;
                    const float dsy_pz = dL_ds.y / p.z;
                    const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
                    const float3 dL_dk = crossProduct(l, dL_dp);
                    const float3 dL_dl = crossProduct(dL_dp, k);

                    const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
                    const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
                    const float3 dL_dTw = {
                        px * dL_dk.x + py * dL_dl.x + dL_dz * dz_dTw.x,
                        px * dL_dk.y + py * dL_dl.y + dL_dz * dz_dTw.y,
                        px * dL_dk.z + py * dL_dl.z + dL_dz * dz_dTw.z
                    };
                    atomicAdd(&dL_dtransMat[g * 9 + 0], dL_dTu.x);
                    atomicAdd(&dL_dtransMat[g * 9 + 1], dL_dTu.y);
                    atomicAdd(&dL_dtransMat[g * 9 + 2], dL_dTu.z);
                    atomicAdd(&dL_dtransMat[g * 9 + 3], dL_dTv.x);
                    atomicAdd(&dL_dtransMat[g * 9 + 4], dL_dTv.y);
                    atomicAdd(&dL_dtransMat[g * 9 + 5], dL_dTv.z);
                    atomicAdd(&dL_dtransMat[g * 9 + 6], dL_dTw.x);
                    atomicAdd(&dL_dtransMat[g * 9 + 7], dL_dTw.y);
                    atomicAdd(&dL_dtransMat[g * 9 + 8], dL_dTw.z);
                } else {
                    // Update gradient w.r.t center of Gaussian 2D mean position
                    const float dG_ddelx = -vis * FilterInvSquare * d.x;
                    const float dG_ddely = -vis * FilterInvSquare * d.y;
                    atomicAdd(&dL_dmean2D[g].x, dL_dG * dG_ddelx); // not scaled
                    atomicAdd(&dL_dmean2D[g].y, dL_dG * dG_ddely); // not scaled
                    // atomicAdd(&dL_dttransMat[global_id * 9 + 8], dL_dz); // propagate depth loss, disable for now

                }
                // const float v_sigma = -opac * vis * v_alpha;
                // v_conic_local = {0.5f * v_sigma * delta.x * delta.x,
                //                 v_sigma * delta.x * delta.y,
                //                 0.5f * v_sigma * delta.y * delta.y};
                // v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                //                 v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                // v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                v_opacity_local = vis * v_alpha;
            }
            warpSum3(v_rgb_local, warp);
            // warpSum3(v_conic_local, warp);
            // warpSum2(v_xy_local, warp);
            // warpSum2(v_xy_abs_local, warp);
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float* v_rgb_ptr = (float*)(v_rgb);
                atomicAdd(v_rgb_ptr + 3 * g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3 * g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_prt + 3 * g + 2, v_rgb_local.z);

                // float* v_conic_ptr = (float*)(v_conic);
                // atomicAdd(v_conic_ptr + 3 * g + 0, v_conic_local.x);
                // atomicAdd(v_conic_ptr + 3 * g + 1, v_conic_local.y);
                // atomicAdd(v_conic_ptr + 3 * g + 1, v_conic_local.z);

                // float* v_xy_ptr = (float*)(v_xy);
                // atomicAdd(v_xy_ptr + 2 * g + 0, v_xy_local.x);
                // atmoicAdd(v_xy_ptr + 2 * g + 1, v_xy_local.y);

                // float* v_xy_abs_ptr = (float*)(v_xy_abs);
                // atomicAdd(v_xy_abs_ptr + 2 * g + 0, v_xy_abs_local.x);
                // atomicAdd(v_xy_abs_ptr + 2 * g + 1, v_xy_abs_local.y);

                atomicAdd(v_opacity + g, v_opacity_local);
            }
        }
    }
}

// __device__ void build_H(
//     const glm::vec3 & p_world,
//     const glm::vec4 & quat,
//     const glm::vec2 & scale,
//     const float* viewmat,
//     const float* projmat,
//     const int W,
//     const int H,
//     const float* transMat,
//     const float* v_transMat,
//     const float* v_normal3D,
//     glm::vec3 & v_mean3D,
//     glm::vec2 & v_scale,
//     glm::vec4 & v_rot
// ) {

// }

__device__ void build_H(
    const glm::vec3 & p_world,
    const glm::vec4 & quat,
    const glm::vec2 & scale,
    const float* viewmat,
    const float4 & intrins,
    float tan_fovx,
    float tan_fovy,
    const float* transMat,
    const float* dL_dtransMat,
    const float* dL_dnormal3D,
    glm::vec3 & dL_dmean3D,
    glm::vec2 & dL_dscale,
    glm::vec4 & dL_drot
) {
    glm::mat3 dL_dT = glm::mat3(
        dL_dtransMat[0], dL_dtransmat[1], dL_dtransMat[2],
        dL_dtransMat[3], dL_dtransMat[4], dL_dtransMat[5],
        dL_dtransMat[6], dL_dtransMat[7], dL_dtransMat[8]
    );

    glm::mat3x4 dL_dsplat = glm::transpose(dL_dT);

    const glm::mat3 R = quat_to_rotmat(quat);

    float multiplier = 1;

    float3 dL_dtn = transformVec4x3Transpose({dL_dnormal3D[0], dL_dnormal3D[1], dL_dnormal[2]}, viewmat);
    glm::mat3 dL_dRS = glm::mat3(
        glm::vec3(dL_dsplat[0]),
        glm::vec3(dL_dsplat[1]),
        multiplier * glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
    );

    // propagate to scale and quat, mean
    glm::mat3 dL_dR = glm::mat3(
        dL_dRS[0] * glm::vec3(scale.x),
        dL_dRS[1] * glm::vec3(scale.y),
        dL_dRS[2]
    );

    dL_dmean3D = glm::vec3(dL_dsplat[2]);
    dL_drot = quat_ro_rotmat_vjp(quat, dL_dR);
    dL_dscale = glm:vec2(
        (float)glm::dot(dL_dRS[0], R[0]),
        (float)glm::dot(dL_dRS[1], R[1])
    );
}

__device__ void build_AABB(
    int P,
    const int * radii,
    const float W,
    const float H,
    const float * transMats,
    float3 * v_mean2D,
    float * v_transMat
) {
    auto idx = cg::this_grid().thread_rank();

    if (idx >= P || !(radii[idx] > 0)) return ;

    const float* transMat = transMats + 9 * idx; // TODO: why do we need this?

    const float3 v_mean2D = v_mean2D[idx];
    glm::mat4x3 T = glm::mat4x3(
        transMat[0], transMat[1], transMat[2],
        transMat[3], transMat[4], transMat[5],
        transMat[6], transMat[7], transMat[8],
        transMat[6], transMat[7], transMat[8]
    );

    float d = glm::dot(glm::vec3(1.0, 1.0, -1.0), T[3] * T[3]);
    glm::vec3 f = glm::vec3(1.0, 1.0, -1.0) * (1.0f / d);

    glm::vec3 p = glm::vec3(
        glm::dot(f, T[0] * T[3]),
        glm::dot(f, T[1] * T[3]),
        glm::dot(f, T[2] * T[3])
    );

    glm::vec3 dL_dT0 = v_mean2D.x * f * T[3];
    glm::vec3 dL_dT1 = v_mean2D.y * f * T[3];
    glm::vec3 dL_dT3 = v_mean2D.x * f * T[0] + dL_dmean2D.y * f * T[1];
    glm::vec3 dL_df = (v_mean2D.x * T[0] * T[3]) + (v_mean2D.y * T[1] * T[3]);
    float dL_dd = glm::dot(dL_df, f) * (-1.0 / d);
    glm::vec3 dd_dT3 = glm::vec3(1.0, 1.0, -1.0) * T[3] * 2.0f;
    dL_dT3 += dL_dd * dd_dT3;
    v_transMat[9 * idx + 0] += dL_dT0.x;
    v_transMat[9 * idx + 1] += dL_dT0.y;
    v_transMat[9 * idx + 2] += dL_dT0.z;
    v_transMat[9 + idx + 3] += dL_dT1.x;
    v_transMat[9 + idx + 4] += dL_dT1.y;
    v_transMat[9 + idx + 5] += dL_dT1.z;
    v_transMat[9 + idx + 6] += dL_dT3.x;
    v_transMat[9 + idx + 7] += dL_dT3.y;
    v_transMat[9 + idx + 8] += dL_dT3.z;

    // just use to hack the projected 2D gradient here.
    // TODO: What does this mean?
    float z = transMat[8];
    v_mean2D[idx].x = v_transMat[9 * idx + 2] * z * W; // to ndc
    v_mean2D[idx].y = v_transMat[9 * idx + 5] * z * H; // to ndc
}


// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ transMats,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dtransMat,
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	// __shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];


	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
				// collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// compute ray-splat intersection as before
			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

			// compute intersection and depth
			float rho = min(rho3d, rho2d);
			float c_d = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
			if (c_d < near_n) continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			// accumulations

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, opa * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float w = alpha * T;
			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			float dL_dz = 0.0f;
			float dL_dweight = 0;


			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * dL_dalpha;

			if (rho3d <= rho2d) {
				// Update gradients w.r.t. covariance of Gaussian 3x3 (T)
				const float2 dL_ds = {
					dL_dG * -G * s.x + dL_dz * Tw.x,
					dL_dG * -G * s.y + dL_dz * Tw.y
				};
				const float3 dz_dTw = {s.x, s.y, 1.0};
				const float dsx_pz = dL_ds.x / p.z;
				const float dsy_pz = dL_ds.y / p.z;
				const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
				const float3 dL_dk = cross(l, dL_dp);
				const float3 dL_dl = cross(dL_dp, k);

				const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
				const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
				const float3 dL_dTw = {
					pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
					pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
					pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};


				// Update gradients w.r.t. 3D covariance (3x3 matrix)
				atomicAdd(&dL_dtransMat[global_id * 9 + 0],  dL_dTu.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 1],  dL_dTu.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 2],  dL_dTu.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 3],  dL_dTv.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 4],  dL_dTv.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 5],  dL_dTv.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 6],  dL_dTw.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7],  dL_dTw.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dTw.z);
			} else {
				// // Update gradients w.r.t. center of Gaussian 2D mean position
				const float dG_ddelx = -G * FilterInvSquare * d.x;
				const float dG_ddely = -G * FilterInvSquare * d.y;
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); // not scaled
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely); // not scaled
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz); // propagate depth loss
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}