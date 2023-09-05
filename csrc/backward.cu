#include "helpers.cuh"
#include "backward.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;


__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const uint32_t *gaussians_ids_sorted,
    const uint2 *tile_bins,
    const float2 *xys,
    const float3 *conics,
    const float3 *rgbs,
    const float *opacities,
    const float *final_Ts,
    const int *final_index,
    const float3 *v_output,
    float2 *v_xy,
    float3 *v_conic,
    float3 *v_rgb,
    float *v_opacity
) {
    // current naive implementation where tile data loading is redundant
    // TODO tile data should be shared between tile threads
    uint32_t tile_id = blockIdx.y * tile_bounds.x + blockIdx.x;
    unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    float px = (float) j;
    float py = (float) i;
    uint32_t pix_id = i * img_size.x + j;

    // which gaussians get gradients for this pixel
    uint2 range = tile_bins[tile_id];
    // df/d_out for this pixel
    float3 v_out = v_output[pix_id];
    float3 rgb, conic;
    float2 center, delta;
    float sigma, vis, fac, opac, alpha;
    float v_alpha, v_sigma;
    float T = final_Ts[pix_id];
    float3 S = {0.f, 0.f, 0.f};
    int bin_final = final_index[pix_id];
    // iterate backward to compute the jacobians wrt rgb, opacity, mean2d, and conic
    // recursively compute T_{n-1} from T_n, where T_i = prod(j < i) (1 - alpha_j),
    // and S_{n-1} from S_n, where S_j = sum_{i > j}(rgb_i * alpha_i * T_i)
    // df/dalpha_i = rgb_i * T_i - S_{i+1| / (1 - alpha_i)
    for (int idx = bin_final - 1; idx >= range.x; --idx) {
        uint32_t g = gaussians_ids_sorted[idx];
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
        vis = exp(-sigma);
        alpha = min(0.99f, opac * vis);
        if (alpha < 1.f / 255.f) {
            continue;
        }

        T /= (1.f - alpha);
        rgb = rgbs[g];
        // update v_rgb for this gaussian
        fac = alpha * T;
        atomicAdd(&(v_rgb[g].x), fac * v_out.x);
        atomicAdd(&(v_rgb[g].y), fac * v_out.y);
        atomicAdd(&(v_rgb[g].z), fac * v_out.z);

        v_alpha += (rgb.x * T - S.x / (1.f - alpha)) * v_out.x;
        v_alpha += (rgb.y * T - S.y / (1.f - alpha)) * v_out.y;
        v_alpha += (rgb.z * T - S.z / (1.f - alpha)) * v_out.z;

        // update contribution from back
        S.x += rgb.x * fac;
        S.y += rgb.y * fac;
        S.z += rgb.z * fac;

        // update v_opacity for this gaussian
        atomicAdd(&(v_opacity[g]), vis * v_alpha);

        // compute vjps for conics and means
        // d_sigma / d_delta = 2 * conic * delta
        // d_sigma / d_conic = delta * delta.T
        v_sigma = -opac * vis * v_alpha;

        atomicAdd(&(v_conic[g].x), v_sigma * delta.x * delta.x);
        atomicAdd(&(v_conic[g].y), v_sigma * delta.x * delta.y);
        atomicAdd(&(v_conic[g].z), v_sigma * delta.y * delta.y);
        atomicAdd(&(v_xy[g].x), v_sigma * 2.f * (conic.x * delta.x + conic.y * delta.y));
        atomicAdd(&(v_xy[g].y), v_sigma * 2.f * (conic.y * delta.x + conic.z * delta.y));
    }

}

void rasterize_backward_impl(
    const dim3 tile_bounds,
    const dim3 block,
    const dim3 img_size,
    const uint32_t *gaussians_ids_sorted,
    const uint2 *tile_bins,
    const float2 *xys,
    const float3 *conics,
    const float3 *rgbs,
    const float *opacities,
    const float *final_Ts,
    const int *final_index,
    const float3 *v_output,
    float2 *v_xy,
    float3 *v_conic,
    float3 *v_rgb,
    float *v_opacity
   
) {
    rasterize_backward_kernel <<< tile_bounds, block >>> (
        tile_bounds,
        img_size,
        gaussians_ids_sorted,
        tile_bins,
        xys,
        conics,
        rgbs,
        opacities,
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
    unsigned idx = cg::this_grid().thread_rank();  // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) {
        return;
    }
    float3 p_world = means3d[idx];
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
        scales[idx], glob_scale, quats[idx], &(v_cov3d[6 * idx]), v_scale[idx], v_quat[idx]
    );

    // get v_mean3d
    // p_cam = P[:3, :3] * p_world + P[:3, 3:]
    float4 p_cam = transform_4x4(projmat, p_world);
    float w = 1.f / (p_cam.w + 1e-5f);

    float3 v_ndc = {img_size.x * v_xy[idx].x, img_size.y * v_xy[idx].y};
    float4 v_proj = {v_ndc.x * w, v_ndc.y * w, 0., -(v_ndc.x + v_ndc.y) * w * w};
    // df / d_world = df / d_cam * d_cam / d_world
    // = v_proj * P[:3, :3]
    v_mean3d[idx].x += projmat[0] * v_proj.x + projmat[4] * v_proj.y + projmat[8] * v_proj.z;
    v_mean3d[idx].y += projmat[1] * v_proj.x + projmat[5] * v_proj.y + projmat[9] * v_proj.z;
    v_mean3d[idx].z += projmat[2] * v_proj.x + projmat[6] * v_proj.y + projmat[10] * v_proj.z;
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
    project_gaussians_backward_kernel
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
__device__ void project_cov3d_ewa_vjp(
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
    glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );
    glm::vec3 p = glm::vec3(viewmat[3], viewmat[7], viewmat[11]);
    glm::vec3 t = W * glm::vec3(mean3d.x, mean3d.y, mean3d.z) + p;

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

    // df/dcov is nonzero only in upper 2x2 submatrix,
    // bc we crop, so no gradients elsewhere
    glm::mat3 v_cov = glm::mat3(
        v_cov2d.x, 0.5f * v_cov2d.y, 0.f,
        0.5f * v_cov2d.y, v_cov2d.z, 0.f,
        0.f, 0.f, 0.f
    );

    // cov = T * V * Tt; G = df/dcov
    // -> df/dT = G * T * Vt + Gt * T * V
    // -> d/dV = Tt * G * T
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
    float rz = 1.f / t.z;
    float rz2 = rz * rz;
    float rz3 = rz2 * rz;
    glm::vec3 v_t = glm::vec3(
        -fx * rz2 * v_J[0][2],
        -fy * rz2 * v_J[1][2],
        -fx * rz2 * v_J[0][0] + 2.f * fx * t.x * rz3 * v_J[0][2]
        - fy * rz2 * v_J[1][1] + 2.f * fy * t.y * rz3 * v_J[1][2]
    );
    v_mean3d.x += (float) glm::dot(v_t, W[0]);
    v_mean3d.y += (float) glm::dot(v_t, W[1]);
    v_mean3d.z += (float) glm::dot(v_t, W[2]);
}


// given cotangent v in output space (e.g. d_L/d_cov3d) in R(6)
// compute vJp for scale and rotation
__device__ void scale_rot_to_cov3d_vjp(
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
        v_cov3d[0], 0.5 * v_cov3d[1], 0.5 * v_cov3d[2],
        0.5 * v_cov3d[1], v_cov3d[3], 0.5 * v_cov3d[4],
        0.5 * v_cov3d[2], 0.5 * v_cov3d[4], v_cov3d[5]
    );
    glm::mat3 R = quat_to_rotmat(quat);
    glm::mat3 S = scale_to_mat(scale, glob_scale);
    glm::mat3 M = R * S;
    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    glm::mat3 v_M = 2.f * v_V * M;
    // glm::mat3 v_S = glm::transpose(R) * v_M;
    v_scale.x = (float) glm::dot(R[0], v_M[0]);
    v_scale.y = (float) glm::dot(R[1], v_M[1]);
    v_scale.z = (float) glm::dot(R[2], v_M[2]);

    glm::mat3 v_R = v_M * S;
    v_quat = quat_to_rotmat_vjp(quat, v_R);
}
