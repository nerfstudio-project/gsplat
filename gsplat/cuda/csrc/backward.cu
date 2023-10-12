#include "backward.cuh"
#include "helpers.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <int CHANNELS>
__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t *gaussians_ids_sorted,
    const int2 *tile_bins,
    const float2 *xys,
    const float3 *conics,
    const float *rgbs,
    const float *opacities,
    const float *background,
    const float *final_Ts,
    const int *final_index,
    const float *v_output,
    float2 *v_xy,
    float3 *v_conic,
    float *v_rgb,
    float *v_opacity
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

    // which gaussians get gradients for this pixel
    int2 range = tile_bins[tile_id];
    // df/d_out for this pixel
    const float *v_out = &(v_output[CHANNELS * pix_id]);
    float3 conic;
    float2 center, delta;
    float sigma, vis, fac, opac, alpha, ra;
    float v_alpha, v_sigma;
    // this is the T AFTER the last gaussian in this pixel
    float T_final = final_Ts[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float S[CHANNELS] = {
        0.f}; // TODO: this currently doesn't match the channel count input.
    // S[0] = 0.0;
    // S[1] = 0.0;
    // S[2] = 0.0;
    // float S[channels] = {0.f};
    int bin_final = final_index[pix_id];

    // iterate backward to compute the jacobians wrt rgb, opacity, mean2d, and
    // conic recursively compute T_{n-1} from T_n, where T_i = prod(j < i) (1 -
    // alpha_j), and S_{n-1} from S_n, where S_j = sum_{i > j}(rgb_i * alpha_i *
    // T_i) df/dalpha_i = rgb_i * T_i - S_{i+1| / (1 - alpha_i)
    for (int idx = bin_final - 1; idx >= range.x; --idx) {
        int32_t g = gaussians_ids_sorted[idx];
        conic = conics[g];
        center = xys[g];
        delta = {center.x - px, center.y - py};
        sigma =
            0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
            conic.y * delta.x * delta.y;
        if (sigma < 0.f) {
            continue;
        }
        opac = opacities[g];
        vis = exp(-sigma);
        alpha = min(0.99f, opac * vis);
        if (alpha < 1.f / 255.f) {
            continue;
        }

        // compute the current T for this gaussian
        ra = 1.f / (1.f - alpha);
        T *= ra;
        // rgb = rgbs[g];
        // update v_rgb for this gaussian
        fac = alpha * T;
        v_alpha = 0.f;
        for (int c = 0; c < CHANNELS; ++c) {
            // gradient wrt rgb
            atomicAdd(&(v_rgb[CHANNELS * g + c]), fac * v_out[c]);
            // contribution from this pixel
            v_alpha += (rgbs[CHANNELS * g + c] * T - S[c] * ra) * v_out[c];
            // contribution from background pixel
            v_alpha += -T_final * ra * background[c] * v_out[c];
            // update the running sum
            S[c] += rgbs[CHANNELS * g + c] * fac;
        }

        // v_alpha = (rgb.x * T - S.x * ra) * v_out.x
        //     + (rgb.y * T - S.y * ra) * v_out.y
        //     + (rgb.z * T - S.z * ra) * v_out.z;
        //
        // // update contribution from back
        // S.x += rgb.x * fac;
        // S.y += rgb.y * fac;
        // S.z += rgb.z * fac;

        // update v_opacity for this gaussian
        atomicAdd(&(v_opacity[g]), vis * v_alpha);

        // compute vjps for conics and means
        // d_sigma / d_delta = conic * delta
        // d_sigma / d_conic = delta * delta.T
        v_sigma = -opac * vis * v_alpha;

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

void rasterize_backward_impl(
    const dim3 tile_bounds,
    const dim3 block,
    const dim3 img_size,
    const int *gaussians_ids_sorted,
    const int2 *tile_bins,
    const float2 *xys,
    const float3 *conics,
    const float *rgbs,
    const float *opacities,
    const float *background,
    const float *final_Ts,
    const int *final_index,
    const float *v_output,
    float2 *v_xy,
    float3 *v_conic,
    float *v_rgb,
    float *v_opacity

) {
    rasterize_backward_kernel<3><<<tile_bounds, block>>>(
        tile_bounds,
        img_size,
        gaussians_ids_sorted,
        tile_bins,
        xys,
        conics,
        rgbs,
        opacities,
        background,
        final_Ts,
        final_index,
        v_output,
        v_xy,
        v_conic,
        v_rgb,
        v_opacity
    );
}

__global__ void project_gaussians_backward_kernel(
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
    const float *cov3d,
    const int *radii,
    const float3 *conics,
    const float2 *v_xy,
    const float3 *v_conic,
    float3 *v_cov2d,
    float *v_cov3d,
    float3 *v_mean3d,
    float3 *v_scale,
    float4 *v_quat
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) {
        return;
    }
    float3 p_world = means3d[idx];
    // get v_mean3d from v_xy
    v_mean3d[idx] = project_pix_vjp(projmat, p_world, img_size, v_xy[idx]);

    // get v_cov2d
    cov2d_to_conic_vjp(conics[idx], v_conic[idx], v_cov2d[idx]);
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

void project_gaussians_backward_impl(
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
    const float *cov3d,
    const int *radii,
    const float3 *conics,
    const float2 *v_xy,
    const float3 *v_conic,
    float3 *v_cov2d,
    float *v_cov3d,
    float3 *v_mean3d,
    float3 *v_scale,
    float4 *v_quat
) {
    project_gaussians_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
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
        cov3d,
        radii,
        conics,
        v_xy,
        v_conic,
        v_cov2d,
        v_cov3d,
        v_mean3d,
        v_scale,
        v_quat
    );
}

// output space: 2D covariance, input space: cov3d
__host__ __device__ void project_cov3d_ewa_vjp(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy,
    const float3 &v_cov2d,
    float3 &v_mean3d,
    float *v_cov3d
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
    glm::vec3 v_t = glm::vec3(
        -fx * rz2 * v_J[2][0],
        -fy * rz2 * v_J[2][1],
        -fx * rz2 * v_J[0][0] + 2.f * fx * t.x * rz3 * v_J[2][0] -
            fy * rz2 * v_J[1][1] + 2.f * fy * t.y * rz3 * v_J[2][1]
    );
    // printf("v_t %.2f %.2f %.2f\n", v_t[0], v_t[1], v_t[2]);
    // printf("W %.2f %.2f %.2f\n", W[0][0], W[0][1], W[0][2]);
    v_mean3d.x += (float)glm::dot(v_t, W[0]);
    v_mean3d.y += (float)glm::dot(v_t, W[1]);
    v_mean3d.z += (float)glm::dot(v_t, W[2]);
}

// given cotangent v in output space (e.g. d_L/d_cov3d) in R(6)
// compute vJp for scale and rotation
__host__ __device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float *v_cov3d,
    float3 &v_scale,
    float4 &v_quat
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
