#include "backward.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;


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
    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = 0.5f * (v_V[0][1] + v_V[1][0]);
    v_cov3d[2] = 0.5f * (v_V[0][2] + v_V[2][0]);
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = 0.5f * (v_V[1][2] + v_V[2][1]);
    v_cov3d[5] = v_V[2][2];

    // compute df wrt mean3d
    // viewmat is row major, glm is column major
    glm::mat3 W = glm::mat3(
        viewmat[0], viewmat[4], viewmat[8],
        viewmat[1], viewmat[5], viewmat[9],
        viewmat[2], viewmat[6], viewmat[10]
    );
    glm::mat4 P = glm::transpose(glm::make_mat4(viewmat));
    glm::vec4 t = P * glm::vec4(mean3d.x, mean3d.y, mean3d.z, 1.f);
    // T = J * W
    v_J = v_T * glm::transpose(W);
    rz2 = 1.f / (t.z * t.z);
    rz3 = rz2 / t.z;
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
