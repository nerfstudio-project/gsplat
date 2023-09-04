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
    const float3 *v_pix,
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
    float3 rgb, conic;
    float2 center, delta;
    float sigma, vis, fac, opac, alpha;
    float v_alpha, v_sigma;
    float2 dsigma_ddelta;
    float3 dsigma_dconic;
    float T = final_Ts[pix_id];
    float3 S = {0.f, 0.f, 0.f};
    int bin_final = final_index[pix_id];
    // iterate backward to compute the jacobians wrt rgb, opacity, mean2d, and conic
    // recursively compute T_{n-1} from T_n, where T_i = prod(j < i) (1 - alpha_j),
    // and S_{n-1} from S_n, where S_j = sum_{i > j}(c_i * alpha_i * T_i)
    for (idx = bin_final - 1; idx >= range.x; --idx) {
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
        vis = exp(-sigma)
        alpha = min(0.99f, opac * vis);
        if (alpha < 1.f / 255.f) {
            continue;
        }

        T /= (1.f - alpha);
        rgb = rgbs[g];
        // update v_rgb for this gaussian
        fac = alpha * T;
        atomicAdd(&(v_rgb[g].x), fac * v_pixels.x);
        atomicAdd(&(v_rgb[g].y), fac * v_pixels.y);
        atomicAdd(&(v_rgb[g].z), fac * v_pixels.z);

        v_alpha += (rgb.x * T - S.x / (1.f - alpha)) * v_pix.x;
        v_alpha += (rgb.y * T - S.y / (1.f - alpha)) * v_pix.y;
        v_alpha += (rgb.z * T - S.z / (1.f - alpha)) * v_pix.z;

        // update contribution from back
        S.x += rgb.x * fac;
        S.y += rgb.y * fac;
        S.z += rgb.z * fac;

        // update v_opacity for this gaussian
        atomicAdd(&(v_opacity[g]), vis * v_alpha);

        // compute vjps for conics and means
        v_sigma = v_alpha * (-opac * vis);
        dsigma_ddelta.x = 2.f * (conic.x * delta.x + conic.y * delta.y);
        dsigma_ddelta.y = 2.f * (conic.y * delta.x + conic.z * delta.y);
        dsigma_dconic.x = delta.x * delta.x;
        dsigma_dconic.y = delta.x * delta.y;
        dsigma_dconic.z = delta.y * delta.y;

        atomicAdd(&(v_conic[g].x), v_sigma * dsigma_dconic.x);
        atomicAdd(&(v_conic[g].y), v_sigma * dsigma_dconic.y);
        atomicAdd(&(v_conic[g].z), v_sigma * dsigma_dconic.z);
        atomicAdd(&(v_xy[g].x), v_sigma * dsigma_ddelta.x);
        atomicAdd(&(v_xy[g].y), v_sigma * dsigma_ddelta.y);
    }

}


// output space: 2D covariance, input space: cov3d
__device__ void project_cov3d_ewa_vjp(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy
    const float *v_cov2d,
    float *v_mean3d,
    float *v_cov3d
) {
    // df/dcov is nonzero only in upper 2x2 submatrix,
    // bc we crop, so no gradients elsewhere
    glm::mat3 v_cov = glm::mat2(
        v_cov2d[0], 0.5f * v_cov2d[1], 0.f,
        0.5f * v_cov2d[1], v_cov2d[2], 0.f,
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
    // viewmat is row major, glm is column major
    // upper 3x3 submatrix
    glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );
    glm::vec3 p = glm::vec3(viewmat[3], viewmat[7], viewmat[11]);
    glm::vec3 t = W * glm::vec3(mean3d.x, mean3d.y, mean3d.z) + p;
    // T = J * W
    v_J = v_T * glm::transpose(W);
    rz = 1.f / t.z;
    rz2 = rz * rz;
    rz3 = rz2 * rz;
    v_t = glm::vec3(
        -fx * rz2 * v_J[0][2],
        -fy * rz2 * v_J[1][2],
        -fx * rz2 * v_J[0][0] + 2.f * fx * t.x * rz3 * v_J[0][2]
        - fy * rz2 * v_J[1][1] + 2.f * fy * t.y * rz3 * v_J[1][2]
    );
    v_mean = v_t * W;
    v_mean3d[0] = glm::dot(v_t, W[0]);
    v_mean3d[1] = glm::dot(v_t, W[1]);
    v_mean3d[2] = glm::dot(v_t, W[2]);
}


// given cotangent v in output space (e.g. d_L/d_cov3d) in R(6)
// compute vJp for scale and rotation
__device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float *v_cov3d,
    &float3 v_scale,
    &float4 v_quat,
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
    v_scale.x = glm::dot(R[0], v_M[0]);
    v_scale.y = glm::dot(R[1], v_M[1]);
    v_scale.z = glm::dot(R[2], v_M[2]);

    glm::mat3 v_R = v_M * S;
    v_quat = quat_to_rotmat_vjp(quat, v_R);
}
