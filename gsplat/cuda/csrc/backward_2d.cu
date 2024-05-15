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
    const float* projmat,
    const int W,
    const int H,
    const float* transMat,
    const float* dL_dtransMat,
    const float* dL_dnormal3D,
    glm::vec3 & dL_dmean3D,
    glm::vec2 & dL_dscale,
    glm::vec4 & dL_drot,
) {
    // Original implementation using ndc
    // we don't do it here
    glm::mat3 dL_dT = glm::mat3(
        dL_dtransMat[0], dL_dtransMat[1], dL_dtransMat[2],
        dL_dtransMat[3], dL_dtransMat[4], dL_dtransMat[5],
        dL_dtransMat[6], dL_dtransMat[7], dL_dtransMat[8]
    );

    glm::mat3x4 dL_dsplat = glm::transpose(dL_dT);
    const glm::mat3 R = quat_to_rotmat(quat);

    float multiplier = 1;

    float3 dL_dtn = transformVec4x3Transpose({dL_dnormal3D[0], dL_dnormal3D[1], dL_d_normal3D[2]}, viewmat);
    glm::mat3 dL_dRS = glm::mat3(
        glm::vec3(dL_dsplat[0]),
        glm::vec3(dL_dsplat[1]),
        multiplier * glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
    );

    // propagate to scale and quat, mean
    glm::mat3 dL_dR = glm::mat3(
        dL_dRS[0] * glm::vec3(scale.x);
        dL_dRS[1] * glm::vec3(scale.y);
        dL_dRS[2]
    );

    dL_dmean3 = glm::vec3(dL_dsplat[2]);
    dL_drot = quat_to_rotmat_vjp(quat, dL_dR);
    dL_dscale = glm::vec2(
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