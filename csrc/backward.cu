#include "backward.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;


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
    // off-diagonal elements considered 0.5 * (V_ij + V_ji)
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
