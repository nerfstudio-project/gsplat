#include "backward.cuh"
#include "helpers.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void nd_rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t *__restrict__ gaussians_ids_sorted,
    const int2 *__restrict__ tile_bins,
    const float2 *__restrict__ xys,
    const float3 *__restrict__ conics,
    const float *__restrict__ rgbs,
    const float *__restrict__ opacities,
    const float *__restrict__ background,
    const float *__restrict__ final_Ts,
    const int *__restrict__ final_index,
    const float *__restrict__ v_output,
    float2 *__restrict__ v_xy,
    float3 *__restrict__ v_conic,
    float *__restrict__ v_rgb,
    float *__restrict__ v_opacity,
    float *__restrict__ workspace
) {
    if (channels > MAX_REGISTER_CHANNELS && workspace == nullptr) {
        return;
    }
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

    // which gaussians get gradients for this pixel
    int2 range = tile_bins[tile_id];
    // df/d_out for this pixel
    const float *v_out = &(v_output[channels * pix_id]);
    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float buffer[MAX_REGISTER_CHANNELS] = {0.f};
    float *S;
    if (channels <= MAX_REGISTER_CHANNELS) {
        S = &buffer[0];
    } else {
        S = &workspace[channels * pix_id];
    }
    int bin_final = final_index[pix_id];

    // iterate backward to compute the jacobians wrt rgb, opacity, mean2d, and
    // conic recursively compute T_{n-1} from T_n, where T_i = prod(j < i) (1 -
    // alpha_j), and S_{n-1} from S_n, where S_j = sum_{i > j}(rgb_i * alpha_i *
    // T_i) df/dalpha_i = rgb_i * T_i - S_{i+1| / (1 - alpha_i)
    for (int idx = bin_final - 1; idx >= range.x; --idx) {
        const int32_t g = gaussians_ids_sorted[idx];
        const float3 conic = conics[g];
        const float2 center = xys[g];
        const float2 delta = {center.x - px, center.y - py};
        const float sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
            conic.y * delta.x * delta.y;
        if (sigma < 0.f) {
            continue;
        }
        const float opac = opacities[g];
        const float vis = __expf(-sigma);
        const float alpha = min(0.99f, opac * vis);
        if (alpha < 1.f / 255.f) {
            continue;
        }

        // compute the current T for this gaussian
        const float ra = 1.f / (1.f - alpha);
        T *= ra;
        // rgb = rgbs[g];
        // update v_rgb for this gaussian
        const float fac = alpha * T;
        float v_alpha = 0.f;
        for (int c = 0; c < channels; ++c) {
            // gradient wrt rgb
            atomicAdd(&(v_rgb[channels * g + c]), fac * v_out[c]);
            // contribution from this pixel
            v_alpha += (rgbs[channels * g + c] * T - S[c] * ra) * v_out[c];
            // contribution from background pixel
            v_alpha += -T_final * ra * background[c] * v_out[c];
            // update the running sum
            S[c] += rgbs[channels * g + c] * fac;
        }

        // update v_opacity for this gaussian
        atomicAdd(&(v_opacity[g]), vis * v_alpha);

        // compute vjps for conics and means
        // d_sigma / d_delta = conic * delta
        // d_sigma / d_conic = delta * delta.T
        const float v_sigma = -opac * vis * v_alpha;

        atomicAdd(&(v_conic[g].x), 0.5f * v_sigma * delta.x * delta.x);
        atomicAdd(&(v_conic[g].y), 0.5f * v_sigma * delta.x * delta.y);
        atomicAdd(&(v_conic[g].z), 0.5f * v_sigma * delta.y * delta.y);
        atomicAdd(
            &(v_xy[g].x), v_sigma * (conic.x * delta.x + conic.y * delta.y)
        );
        atomicAdd(
            &(v_xy[g].y), v_sigma * (conic.y * delta.x + conic.z * delta.y)
        );
    }
}

inline __device__ void warpSum3(float3 &val, cg::thread_block_tile<32> &tile) {
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
    val.z = cg::reduce(tile, val.z, cg::plus<float>());
}

inline __device__ void warpSum2(float2 &val, cg::thread_block_tile<32> &tile) {
    val.x = cg::reduce(tile, val.x, cg::plus<float>());
    val.y = cg::reduce(tile, val.y, cg::plus<float>());
}

inline __device__ void warpSum(float &val, cg::thread_block_tile<32> &tile) {
    val = cg::reduce(tile, val, cg::plus<float>());
}

__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t *__restrict__ gaussian_ids_sorted,
    const int2 *__restrict__ tile_bins,
    const float2 *__restrict__ xys,
    const float3 *__restrict__ conics,
    const float3 *__restrict__ rgbs,
    const float *__restrict__ opacities,
    const float3 &__restrict__ background,
    const float *__restrict__ final_Ts,
    const int *__restrict__ final_index,
    const float3 *__restrict__ v_output,
    float2 *__restrict__ v_xy,
    float3 *__restrict__ v_conic,
    float3 *__restrict__ v_rgb,
    float *__restrict__ v_opacity
) {
    auto block = cg::this_thread_block();
    int32_t tile_id =
        block.group_index().y * tile_bounds.x + block.group_index().x;
    unsigned i =
        block.group_index().y * block.group_dim().y + block.thread_index().y;
    unsigned j =
        block.group_index().x * block.group_dim().x + block.thread_index().x;

    const float px = (float)j;
    const float py = (float)i;
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
    const int num_batches = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;

    __shared__ int32_t id_batch[BLOCK_SIZE];
    __shared__ float3 xy_opacity_batch[BLOCK_SIZE];
    __shared__ float3 conic_batch[BLOCK_SIZE];
    __shared__ float3 rgbs_batch[BLOCK_SIZE];

    // df/d_out for this pixel
    const float3 v_out = v_output[pix_id];

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
        const int batch_end = range.y - 1 - BLOCK_SIZE * b;
        int batch_size = min(BLOCK_SIZE, batch_end + 1 - range.x);
        const int idx = batch_end - tr;
        if (idx >= range.x) {
            int32_t g_id = gaussian_ids_sorted[idx];
            id_batch[tr] = g_id;
            const float2 xy = xys[g_id];
            const float opac = opacities[g_id];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_id];
            rgbs_batch[tr] = rgbs[g_id];
        }
        // wait for other threads to collect the gaussians in batch
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
                conic = conic_batch[t];
                float3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                      conic.z * delta.y * delta.y) +
                              conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.99f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = 0;
                }
            }
            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            float3 v_rgb_local = {0.f, 0.f, 0.f};
            float3 v_conic_local = {0.f, 0.f, 0.f};
            float2 v_xy_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
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
                // contribution from background pixel
                v_alpha += -T_final * ra * background.x * v_out.x;
                v_alpha += -T_final * ra * background.y * v_out.y;
                v_alpha += -T_final * ra * background.z * v_out.z;
                // update the running sum
                buffer.x += rgb.x * fac;
                buffer.y += rgb.y * fac;
                buffer.z += rgb.z * fac;

                const float v_sigma = -opac * vis * v_alpha;
                v_conic_local = {
                    0.5f * v_sigma * delta.x * delta.x,
                    0.5f * v_sigma * delta.x * delta.y,
                    0.5f * v_sigma * delta.y * delta.y};
                v_xy_local = {
                    v_sigma * (conic.x * delta.x + conic.y * delta.y),
                    v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                v_opacity_local = vis * v_alpha;
            }
            warpSum3(v_rgb_local, warp);
            warpSum3(v_conic_local, warp);
            warpSum2(v_xy_local, warp);
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                float *v_rgb_ptr = (float *)(v_rgb);
                atomicAdd(v_rgb_ptr + 3 * g + 0, v_rgb_local.x);
                atomicAdd(v_rgb_ptr + 3 * g + 1, v_rgb_local.y);
                atomicAdd(v_rgb_ptr + 3 * g + 2, v_rgb_local.z);

                float *v_conic_ptr = (float *)(v_conic);
                atomicAdd(v_conic_ptr + 3 * g + 0, v_conic_local.x);
                atomicAdd(v_conic_ptr + 3 * g + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 3 * g + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_xy);
                atomicAdd(v_xy_ptr + 2 * g + 0, v_xy_local.x);
                atomicAdd(v_xy_ptr + 2 * g + 1, v_xy_local.y);

                atomicAdd(v_opacity + g, v_opacity_local);
            }
        }
    }
}

__global__ void project_gaussians_backward_kernel(
    const int num_points,
    const float3 *__restrict__ means3d,
    const float3 *__restrict__ scales,
    const float glob_scale,
    const float4 *__restrict__ quats,
    const float *__restrict__ viewmat,
    const float *__restrict__ projmat,
    const float *__restrict__ fullmat,
    const float4 intrins,
    const dim3 img_size,
    const float *__restrict__ cov3d,
    const int *__restrict__ radii,
    const float3 *__restrict__ conics,
    const float2 *__restrict__ v_xy,
    const float *__restrict__ v_depth,
    const float3 *__restrict__ v_conic,
    float3 *__restrict__ v_cov2d,
    float *__restrict__ v_cov3d,
    float3 *__restrict__ v_mean3d,
    float3 *__restrict__ v_scale,
    float4 *__restrict__ v_quat
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) {
        return;
    }
    float3 p_world = means3d[idx];
    float fx = intrins.x;
    float fy = intrins.y;
    float cx = intrins.z;
    float cy = intrins.w;

    // get xy gradient contribution to view space xyz gradient
    float4 v_view =
        project_pix_vjp(projmat, fullmat, p_world, img_size, v_xy[idx]);
    // add depth contribution to view space gradient
    v_view.z += v_depth[idx];

    // get v_cov2d from v_conic
    cov2d_to_conic_vjp(conics[idx], v_conic[idx], v_cov2d[idx]);
    // get v_cov3d (and v_view contribution) from v_cov2d
    project_cov3d_ewa_vjp(
        p_world,
        &(cov3d[6 * idx]),
        viewmat,
        fx,
        fy,
        v_cov2d[idx],
        v_view,
        &(v_cov3d[6 * idx])
    );
    // get v_scale and v_quat from v_cov3d
    scale_rot_to_cov3d_vjp(
        scales[idx],
        glob_scale,
        quats[idx],
        &(v_cov3d[6 * idx]),
        v_scale[idx],
        v_quat[idx]
    );
    // v_view -> v_mean3d in world space
    // v_mean3d = viewmat^T * v_view
    v_mean3d[idx].x =
        viewmat[0] * v_view.x + viewmat[4] * v_view.y + viewmat[8] * v_view.z;
    v_mean3d[idx].y =
        viewmat[1] * v_view.x + viewmat[5] * v_view.y + viewmat[9] * v_view.z;
    v_mean3d[idx].z =
        viewmat[2] * v_view.x + viewmat[6] * v_view.y + viewmat[10] * v_view.z;
}

// output space: 2D covariance, input space: cov3d
__device__ void project_cov3d_ewa_vjp(
    const float3 &__restrict__ mean3d,
    const float *__restrict__ cov3d,
    const float *__restrict__ viewmat,
    const float fx,
    const float fy,
    const float3 &__restrict__ v_cov2d,
    float4 &__restrict__ v_view,
    float *__restrict__ v_cov3d
) {
    // viewmat is row major, glm is column major
    // upper 3x3 submatrix
    // clang-format off
    glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );
    // clang-format on
    glm::vec3 p = glm::vec3(viewmat[3], viewmat[7], viewmat[11]);
    glm::vec3 t = W * glm::vec3(mean3d.x, mean3d.y, mean3d.z) + p;
    float rz = 1.f / t.z;
    float rz2 = rz * rz;

    // column major
    // we only care about the top 2x2 submatrix
    // clang-format off
    glm::mat3 J = glm::mat3(
        fx * rz,         0.f,             0.f,
        0.f,             fy * rz,         0.f,
        -fx * t.x * rz2, -fy * t.y * rz2, 0.f
    );
    glm::mat3 V = glm::mat3(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    );
    // cov = T * V * Tt; G = df/dcov = v_cov
    // -> d/dV = Tt * G * T
    // -> df/dT = G * T * Vt + Gt * T * V
    glm::mat3 v_cov = glm::mat3(
        v_cov2d.x,        0.5f * v_cov2d.y, 0.f,
        0.5f * v_cov2d.y, v_cov2d.z,        0.f,
        0.f,              0.f,              0.f
    );
    // clang-format on

    glm::mat3 T = J * W;
    glm::mat3 Tt = glm::transpose(T);
    glm::mat3 Vt = glm::transpose(V);
    glm::mat3 v_V = Tt * v_cov * T;
    glm::mat3 v_T = v_cov * T * Vt + glm::transpose(v_cov) * T * V;

    // vjp of cov3d parameters
    // v_cov3d_i = v_V : dV/d_cov3d_i
    // where : is frobenius inner product
    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = v_V[0][1] + v_V[1][0];
    v_cov3d[2] = v_V[0][2] + v_V[2][0];
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = v_V[1][2] + v_V[2][1];
    v_cov3d[5] = v_V[2][2];

    // compute df/d_mean3d
    // T = J * W
    glm::mat3 v_J = v_T * glm::transpose(W);
    float rz3 = rz2 * rz;
    v_view.x += (float)(-fx * rz2 * v_J[2][0]);
    v_view.y += (float)(-fy * rz2 * v_J[2][1]);
    v_view.z += (float
    )(-fx * rz2 * v_J[0][0] + 2.f * fx * t.x * rz3 * v_J[2][0] -
      fy * rz2 * v_J[1][1] + 2.f * fy * t.y * rz3 * v_J[2][1]);
}

// given cotangent v in output space (e.g. d_L/d_cov3d) in R(6)
// compute vJp for scale and rotation
__device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float *__restrict__ v_cov3d,
    float3 &__restrict__ v_scale,
    float4 &__restrict__ v_quat
) {
    // cov3d is upper triangular elements of matrix
    // off-diagonal elements count grads from both ij and ji elements,
    // must halve when expanding back into symmetric matrix
    glm::mat3 v_V = glm::mat3(
        v_cov3d[0],
        0.5 * v_cov3d[1],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[1],
        v_cov3d[3],
        0.5 * v_cov3d[4],
        0.5 * v_cov3d[2],
        0.5 * v_cov3d[4],
        v_cov3d[5]
    );
    glm::mat3 R = quat_to_rotmat(quat);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    glm::mat3 M = R * S;
    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    glm::mat3 v_M = 2.f * v_V * M;
    // glm::mat3 v_S = glm::transpose(R) * v_M;
    v_scale.x = (float)glm::dot(R[0], v_M[0]);
    v_scale.y = (float)glm::dot(R[1], v_M[1]);
    v_scale.z = (float)glm::dot(R[2], v_M[2]);

    glm::mat3 v_R = v_M * S;
    v_quat = quat_to_rotmat_vjp(quat, v_R);
}
