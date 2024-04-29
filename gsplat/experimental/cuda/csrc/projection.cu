#include "bindings.h"
#include "helpers.cuh"
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

/****************************************************************************
 * Quat-Scale to Covariance and Precision
 ****************************************************************************/

inline __device__ glm::mat3 quat_to_rotmat(const glm::vec4 quat) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return glm::mat3((1.f - 2.f * (y2 + z2)), (2.f * (xy + wz)),
                     (2.f * (xz - wy)), // 1st col
                     (2.f * (xy - wz)), (1.f - 2.f * (x2 + z2)),
                     (2.f * (yz + wx)), // 2nd col
                     (2.f * (xz + wy)), (2.f * (yz - wx)),
                     (1.f - 2.f * (x2 + y2)) // 3rd col
    );
}

inline __device__ void quat_to_rotmat_vjp(const glm::vec4 quat, const glm::mat3 v_R,
                                          glm::vec4 &v_quat) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    v_quat[0] += 2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                        z * (v_R[0][1] - v_R[1][0]));
    v_quat[1] +=
        2.f * (-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
               z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1]));
    v_quat[2] +=
        2.f * (x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
               z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2]));
    v_quat[3] +=
        2.f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
               2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0]));
}

inline __device__ void quat_scale_to_covar_perci(const glm::vec4 quat,
                                                 const glm::vec3 scale,
                                                 // optional outputs
                                                 glm::mat3 *covar, glm::mat3 *perci) {
    glm::mat3 R = quat_to_rotmat(quat);
    if (covar != nullptr) {
        // C = R * S * S * Rt
        glm::mat3 S =
            glm::mat3(scale[0], 0.f, 0.f, 0.f, scale[1], 0.f, 0.f, 0.f, scale[2]);
        glm::mat3 M = R * S;
        *covar = M * glm::transpose(M);
    }
    if (perci != nullptr) {
        // P = R * S^-1 * S^-1 * Rt
        glm::mat3 S = glm::mat3(1.0f / scale[0], 0.f, 0.f, 0.f, 1.0f / scale[1], 0.f,
                                0.f, 0.f, 1.0f / scale[2]);
        glm::mat3 M = R * S;
        *perci = M * glm::transpose(M);
    }
}

inline __device__ void quat_scale_to_covar_vjp(
    // fwd inputs
    const glm::vec4 quat, const glm::vec3 scale,
    // precompute
    const glm::mat3 R,
    // grad outputs
    const glm::mat3 v_covar,
    // grad inputs
    glm::vec4 &v_quat, glm::vec3 &v_scale) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    float sx = scale[0], sy = scale[1], sz = scale[2];

    // M = R * S
    glm::mat3 S = glm::mat3(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    glm::mat3 M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    glm::mat3 v_M = (v_covar + glm::transpose(v_covar)) * M;
    glm::mat3 v_R = v_M * S;

    // grad for (quat, scale) from covar
    quat_to_rotmat_vjp(quat, v_R, v_quat);

    v_scale[0] += R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2];
    v_scale[1] += R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2];
    v_scale[2] += R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2];
}

inline __device__ void quat_scale_to_perci_vjp(
    // fwd inputs
    const glm::vec4 quat, const glm::vec3 scale,
    // precompute
    const glm::mat3 R,
    // grad outputs
    const glm::mat3 v_perci,
    // grad inputs
    glm::vec4 &v_quat, glm::vec3 &v_scale) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    float sx = 1.0f / scale[0], sy = 1.0f / scale[1], sz = 1.0f / scale[2];

    // M = R * S
    glm::mat3 S = glm::mat3(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    glm::mat3 M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    glm::mat3 v_M = (v_perci + glm::transpose(v_perci)) * M;
    glm::mat3 v_R = v_M * S;

    // grad for (quat, scale) from perci
    quat_to_rotmat_vjp(quat, v_R, v_quat);

    v_scale[0] +=
        -sx * sx * (R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2]);
    v_scale[1] +=
        -sy * sy * (R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2]);
    v_scale[2] +=
        -sz * sz * (R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2]);
}

__global__ void
quat_scale_to_covar_perci_fwd_kernel(const int N,
                                     const float *__restrict__ quats,  // [N, 4]
                                     const float *__restrict__ scales, // [N, 3]
                                     const bool triu,
                                     // outputs
                                     float *__restrict__ covars, // [N, 3, 3] or [N, 6]
                                     float *__restrict__ percis  // [N, 3, 3] or [N, 6]
) {
    // parallelize over N.
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    // shift pointers to the current gaussian
    quats += idx * 4;
    scales += idx * 3;

    // compute the matrices
    glm::mat3 covar, perci;
    quat_scale_to_covar_perci(glm::make_vec4(quats), glm::make_vec3(scales),
                              covars ? &covar : nullptr, percis ? &perci : nullptr);

    // write to outputs: glm is column-major but we want row-major
    if (covars != nullptr) {
        if (triu) {
            covars += idx * 6;
            covars[0] = covar[0][0];
            covars[1] = covar[0][1];
            covars[2] = covar[0][2];
            covars[3] = covar[1][1];
            covars[4] = covar[1][2];
            covars[5] = covar[2][2];
        } else {
            covars += idx * 9;
#pragma unroll 3
            for (int i = 0; i < 3; i++) { // rows
#pragma unroll 3
                for (int j = 0; j < 3; j++) { // cols
                    covars[i * 3 + j] = covar[j][i];
                }
            }
        }
    }
    if (percis != nullptr) {
        if (triu) {
            percis += idx * 6;
            percis[0] = perci[0][0];
            percis[1] = perci[0][1];
            percis[2] = perci[0][2];
            percis[3] = perci[1][1];
            percis[4] = perci[1][2];
            percis[5] = perci[2][2];
        } else {
            percis += idx * 9;
#pragma unroll 3
            for (int i = 0; i < 3; i++) { // rows
#pragma unroll 3
                for (int j = 0; j < 3; j++) { // cols
                    percis[i * 3 + j] = perci[j][i];
                }
            }
        }
    }
}

__global__ void quat_scale_to_covar_perci_bwd_kernel(
    const int N,
    // fwd inputs
    const float *__restrict__ quats,  // [N, 4]
    const float *__restrict__ scales, // [N, 3]
    // grad outputs
    const float *__restrict__ v_covars, // [N, 3, 3] or [N, 6]
    const float *__restrict__ v_percis, // [N, 3, 3] or [N, 6]
    const bool triu,
    // grad inputs
    float *__restrict__ v_scales, // [N, 3]
    float *__restrict__ v_quats   // [N, 4]
) {
    // parallelize over N.
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    // shift pointers to the current gaussian
    v_scales += idx * 3;
    v_quats += idx * 4;

    glm::vec4 quat = glm::make_vec4(quats + idx * 4);
    glm::vec3 scale = glm::make_vec3(scales + idx * 3);
    glm::mat3 rotmat = quat_to_rotmat(quat);

    glm::vec4 v_quat(0.f);
    glm::vec3 v_scale(0.f);
    if (v_covars != nullptr) {
        // glm is column-major, input is row-major
        glm::mat3 v_covar;
        if (triu) {
            v_covars += idx * 6;
            v_covar = glm::mat3(v_covars[0], v_covars[1] * .5f, v_covars[2] * .5f,
                                v_covars[1] * .5f, v_covars[3], v_covars[4] * .5f,
                                v_covars[2] * .5f, v_covars[4] * .5f, v_covars[5]);
        } else {
            v_covars += idx * 9;
            v_covar = glm::transpose(glm::make_mat3(v_covars));
        }
        quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar, v_quat, v_scale);
    }
    if (v_percis != nullptr) {
        // glm is column-major, input is row-major
        glm::mat3 v_perci;
        if (triu) {
            v_percis += idx * 6;
            v_perci = glm::mat3(v_percis[0], v_percis[1] * .5f, v_percis[2] * .5f,
                                v_percis[1] * .5f, v_percis[3], v_percis[4] * .5f,
                                v_percis[2] * .5f, v_percis[4] * .5f, v_percis[5]);
        } else {
            v_percis += idx * 9;
            v_perci = glm::transpose(glm::make_mat3(v_percis));
        }
        quat_scale_to_perci_vjp(quat, scale, rotmat, v_perci, v_quat, v_scale);
    }

    // write out results
#pragma unroll 3
    for (int k = 0; k < 3; ++k) {
        v_scales[k] = v_scale[k];
    }
#pragma unroll 4
    for (int k = 0; k < 4; ++k) {
        v_quats[k] = v_quat[k];
    }
}

std::tuple<torch::Tensor, torch::Tensor>
quat_scale_to_covar_perci_fwd_tensor(const torch::Tensor &quats,  // [N, 4]
                                     const torch::Tensor &scales, // [N, 3]
                                     const bool compute_covar, const bool compute_perci,
                                     const bool triu) {
    DEVICE_GUARD(quats);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);

    int N = quats.size(0);

    torch::Tensor covars, percis;
    if (compute_covar) {
        if (triu) {
            covars = torch::empty({N, 6}, quats.options());
        } else {
            covars = torch::empty({N, 3, 3}, quats.options());
        }
    }
    if (compute_perci) {
        if (triu) {
            percis = torch::empty({N, 6}, quats.options());
        } else {
            percis = torch::empty({N, 3, 3}, quats.options());
        }
    }

    if (N > 0) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        quat_scale_to_covar_perci_fwd_kernel<<<(N + N_THREADS - 1) / N_THREADS,
                                               N_THREADS, 0, stream>>>(
            N, quats.data_ptr<float>(), scales.data_ptr<float>(), triu,
            compute_covar ? covars.data_ptr<float>() : nullptr,
            compute_perci ? percis.data_ptr<float>() : nullptr);
    }
    return std::make_tuple(covars, percis);
}

std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_perci_bwd_tensor(
    const torch::Tensor &quats,                  // [N, 4]
    const torch::Tensor &scales,                 // [N, 3]
    const at::optional<torch::Tensor> &v_covars, // [N, 3, 3]
    const at::optional<torch::Tensor> &v_percis, // [N, 3, 3]
    const bool triu) {
    DEVICE_GUARD(quats);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    if (v_covars.has_value()) {
        CHECK_INPUT(v_covars.value());
    }
    if (v_percis.has_value()) {
        CHECK_INPUT(v_percis.value());
    }

    int N = quats.size(0);

    torch::Tensor v_scales = torch::empty({N, 3}, scales.options());
    torch::Tensor v_quats = torch::empty({N, 4}, quats.options());

    if (N > 0) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        quat_scale_to_covar_perci_bwd_kernel<<<(N + N_THREADS - 1) / N_THREADS,
                                               N_THREADS, 0, stream>>>(
            N, quats.data_ptr<float>(), scales.data_ptr<float>(),
            v_covars.has_value() ? v_covars.value().data_ptr<float>() : nullptr,
            v_percis.has_value() ? v_percis.value().data_ptr<float>() : nullptr, triu,
            v_scales.data_ptr<float>(), v_quats.data_ptr<float>());
    }

    return std::make_tuple(v_quats, v_scales);
}

/****************************************************************************
 * Perspective Projection
 ****************************************************************************/

inline __device__ void persp_proj(
    // inputs
    const glm::vec3 mean3d, const glm::mat3 cov3d, const float fx, const float fy,
    const float cx, const float cy, const int width, const int height,
    // outputs
    glm::mat2 &cov2d, glm::vec2 &mean2d) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    float tan_fovx = 0.5f * width / fx;
    float tan_fovy = 0.5f * height / fy;
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;

    float rz = 1.f / z;
    float rz2 = rz * rz;
    float tx = z * min(lim_x, max(-lim_x, x * rz));
    float ty = z * min(lim_y, max(-lim_y, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    glm::mat3x2 J = glm::mat3x2(fx * rz, 0.f,                  // 1st column
                                0.f, fy * rz,                  // 2nd column
                                -fx * tx * rz2, -fy * ty * rz2 // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = glm::vec2({fx * x * rz + cx, fy * y * rz + cy});
}

inline __device__ void persp_proj_vjp(
    // fwd inputs
    const glm::vec3 mean3d, const glm::mat3 cov3d, const float fx, const float fy,
    const float cx, const float cy, const int width, const int height,
    // grad outputs
    const glm::mat2 v_cov2d, const glm::vec2 v_mean2d,
    // grad inputs
    glm::vec3 &v_mean3d, glm::mat3 &v_cov3d) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    float tan_fovx = 0.5f * width / fx;
    float tan_fovy = 0.5f * height / fy;
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;

    float rz = 1.f / z;
    float rz2 = rz * rz;
    float tx = z * min(lim_x, max(-lim_x, x * rz));
    float ty = z * min(lim_y, max(-lim_y, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    glm::mat3x2 J = glm::mat3x2(fx * rz, 0.f,                  // 1st column
                                0.f, fy * rz,                  // 2nd column
                                -fx * tx * rz2, -fy * ty * rz2 // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    v_mean3d += glm::vec3(fx * rz * v_mean2d[0], fy * rz * v_mean2d[1],
                          -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2);

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    float rz3 = rz2 * rz;
    glm::mat3x2 v_J =
        v_cov2d * J * glm::transpose(cov3d) + glm::transpose(v_cov2d) * J * cov3d;

    // fov clipping
    if (x * rz <= lim_x && x * rz >= -lim_x) {
        v_mean3d.x += -fx * rz2 * v_J[2][0];
    } else {
        v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
    }
    if (y * rz <= lim_y && y * rz >= -lim_y) {
        v_mean3d.y += -fy * rz2 * v_J[2][1];
    } else {
        v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
    }
    v_mean3d.z += -fx * rz2 * v_J[0][0] - fy * rz2 * v_J[1][1] +
                  2.f * fx * tx * rz3 * v_J[2][0] + 2.f * fy * ty * rz3 * v_J[2][1];
}

__global__ void persp_proj_fwd_kernel(const int C, const int N,
                                      const float *__restrict__ means,  // [C, N, 3]
                                      const float *__restrict__ covars, // [C, N, 3, 3]
                                      const float *__restrict__ Ks,     // [C, 3, 3]
                                      const int width, const int height,
                                      float *__restrict__ means2d, // [C, N, 2]
                                      float *__restrict__ covars2d // [C, N, 2, 2]
) { // parallelize over C * N.
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const int cid = idx / N; // camera id
    const int gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += idx * 3;
    covars += idx * 9;
    Ks += cid * 9;
    means2d += idx * 2;
    covars2d += idx * 4;

    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    glm::mat2 covar2d;
    glm::vec2 mean2d;
    persp_proj(glm::make_vec3(means), glm::make_mat3(covars), fx, fy, cx, cy, width,
               height, covar2d, mean2d);

// write to outputs: glm is column-major but we want row-major
#pragma unroll 2
    for (int i = 0; i < 2; i++) { // rows
#pragma unroll 2
        for (int j = 0; j < 2; j++) { // cols
            covars2d[i * 2 + j] = covar2d[j][i];
        }
    }
#pragma unroll 2
    for (int i = 0; i < 2; i++) {
        means2d[i] = mean2d[i];
    }
}

__global__ void
persp_proj_bwd_kernel(const int C, const int N,
                      const float *__restrict__ means,  // [C, N, 3]
                      const float *__restrict__ covars, // [C, N, 3, 3]
                      const float *__restrict__ Ks,     // [C, 3, 3]
                      const int width, const int height,
                      const float *__restrict__ v_means2d,  // [C, N, 2]
                      const float *__restrict__ v_covars2d, // [C, N, 2, 2]
                      float *__restrict__ v_means,          // [C, N, 3]
                      float *__restrict__ v_covars          // [C, N, 3, 3]
) {
    // parallelize over C * N.
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const int cid = idx / N; // camera id
    const int gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += idx * 3;
    covars += idx * 9;
    v_means += idx * 3;
    v_covars += idx * 9;
    Ks += cid * 9;
    v_means2d += idx * 2;
    v_covars2d += idx * 4;

    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    glm::mat3 v_covar(0.f);
    glm::vec3 v_mean(0.f);
    persp_proj_vjp(glm::make_vec3(means), glm::make_mat3(covars), fx, fy, cx, cy, width,
                   height, glm::transpose(glm::make_mat2(v_covars2d)),
                   glm::make_vec2(v_means2d), v_mean, v_covar);

// write to outputs: glm is column-major but we want row-major
#pragma unroll 3
    for (int i = 0; i < 3; i++) { // rows
#pragma unroll 3
        for (int j = 0; j < 3; j++) { // cols
            v_covars[i * 3 + j] = v_covar[j][i];
        }
    }

#pragma unroll 3
    for (int i = 0; i < 3; i++) {
        v_means[i] = v_mean[i];
    }
}

std::tuple<torch::Tensor, torch::Tensor>
persp_proj_fwd_tensor(const torch::Tensor &means,  // [C, N, 3]
                      const torch::Tensor &covars, // [C, N, 3, 3]
                      const torch::Tensor &Ks,     // [C, 3, 3]
                      const int width, const int height) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(Ks);

    int C = means.size(0);
    int N = means.size(1);

    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor covars2d = torch::empty({C, N, 2, 2}, covars.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        persp_proj_fwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(),
            Ks.data_ptr<float>(), width, height, means2d.data_ptr<float>(),
            covars2d.data_ptr<float>());
    }
    return std::make_tuple(means2d, covars2d);
}

std::tuple<torch::Tensor, torch::Tensor>
persp_proj_bwd_tensor(const torch::Tensor &means,  // [C, N, 3]
                      const torch::Tensor &covars, // [C, N, 3, 3]
                      const torch::Tensor &Ks,     // [C, 3, 3]
                      const int width, const int height,
                      const torch::Tensor &v_means2d, // [C, N, 2]
                      const torch::Tensor &v_covars2d // [C, N, 2, 2]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(Ks);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_covars2d);

    int C = means.size(0);
    int N = means.size(1);

    torch::Tensor v_means = torch::empty({C, N, 3}, means.options());
    torch::Tensor v_covars = torch::empty({C, N, 3, 3}, means.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        persp_proj_bwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(),
            Ks.data_ptr<float>(), width, height, v_means2d.data_ptr<float>(),
            v_covars2d.data_ptr<float>(), v_means.data_ptr<float>(),
            v_covars.data_ptr<float>());
    }
    return std::make_tuple(v_means, v_covars);
}

/****************************************************************************
 * World to Camera Transformation
 ****************************************************************************/

inline __device__ void pos_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const glm::mat3 R, const glm::vec3 t, const glm::vec3 p, glm::vec3 &p_c) {
    p_c = R * p + t;
}

inline __device__ void pos_world_to_cam_vjp(
    // fwd inputs
    const glm::mat3 R, const glm::vec3 t, const glm::vec3 p,
    // grad outputs
    const glm::vec3 v_p_c,
    // grad inputs
    glm::mat3 &v_R, glm::vec3 &v_t, glm::vec3 &v_p) {
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    v_R += glm::outerProduct(v_p_c, p);
    v_t += v_p_c;
    v_p += glm::transpose(R) * v_p_c;
}

inline __device__ void covar_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const glm::mat3 R, const glm::mat3 covar, glm::mat3 &covar_c) {
    covar_c = R * covar * glm::transpose(R);
}

inline __device__ void covar_world_to_cam_vjp(
    // fwd inputs
    const glm::mat3 R, const glm::mat3 covar,
    // grad outputs
    const glm::mat3 v_covar_c,
    // grad inputs
    glm::mat3 &v_R, glm::mat3 &v_covar) {
    // for D = W * X * WT, G = df/dD
    // df/dX = WT * G * W
    // df/dW
    // = G * (X * WT)T + ((W * X)T * G)T
    // = G * W * XT + (XT * WT * G)T
    // = G * W * XT + GT * W * X
    v_R +=
        v_covar_c * R * glm::transpose(covar) + glm::transpose(v_covar_c) * R * covar;
    v_covar += glm::transpose(R) * v_covar_c * R;
}

__global__ void world_to_cam_fwd_kernel(const int C, const int N,
                                        const float *__restrict__ means,    // [N, 3]
                                        const float *__restrict__ covars,   // [N, 3, 3]
                                        const float *__restrict__ viewmats, // [C, 4, 4]
                                        float *__restrict__ means_c,        // [C, N, 3]
                                        float *__restrict__ covars_c // [C, N, 3, 3]
) { // parallelize over C * N.
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const int cid = idx / N; // camera id
    const int gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    covars += gid * 9;
    viewmats += cid * 16;

    // glm is column-major but input is row-major
    glm::mat3 R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);

    if (means_c != nullptr) {
        glm::vec3 mean_c;
        pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
        means_c += idx * 3;
#pragma unroll 3
        for (int i = 0; i < 3; i++) { // rows
            means_c[i] = mean_c[i];
        }
    }

    // write to outputs: glm is column-major but we want row-major
    if (covars_c != nullptr) {
        glm::mat3 covar_c;
        covar_world_to_cam(R, glm::make_mat3(covars), covar_c);
        covars_c += idx * 9;
#pragma unroll 3
        for (int i = 0; i < 3; i++) { // rows
#pragma unroll 3
            for (int j = 0; j < 3; j++) { // cols
                covars_c[i * 3 + j] = covar_c[j][i];
            }
        }
    }
}

__global__ void
world_to_cam_bwd_kernel(const int C, const int N,
                        const float *__restrict__ means,      // [N, 3]
                        const float *__restrict__ covars,     // [N, 3, 3]
                        const float *__restrict__ viewmats,   // [C, 4, 4]
                        const float *__restrict__ v_means_c,  // [C, N, 3]
                        const float *__restrict__ v_covars_c, // [C, N, 3, 3]
                        float *__restrict__ v_means,          // [N, 3]
                        float *__restrict__ v_covars,         // [N, 3, 3]
                        float *__restrict__ v_viewmats        // [C, 4, 4]
) {
    // parallelize over C * N.
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const int cid = idx / N; // camera id
    const int gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    covars += gid * 9;
    viewmats += cid * 16;

    // glm is column-major but input is row-major
    glm::mat3 R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);

    glm::vec3 v_mean(0.f);
    glm::mat3 v_covar(0.f);
    glm::mat3 v_R(0.f);
    glm::vec3 v_t(0.f);

    if (v_means_c != nullptr) {
        glm::vec3 v_mean_c = glm::make_vec3(v_means_c + idx * 3);
        pos_world_to_cam_vjp(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
    }
    if (v_covars_c != nullptr) {
        glm::mat3 v_covar_c = glm::transpose(glm::make_mat3(v_covars_c + idx * 9));
        covar_world_to_cam_vjp(R, glm::make_mat3(covars), v_covar_c, v_R, v_covar);
    }

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += gid * 3;
#pragma unroll 3
            for (int i = 0; i < 3; i++) {
                atomicAdd(v_means + i, v_mean[i]);
            }
        }
    }
    if (v_covars != nullptr) {
        warpSum(v_covar, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_covars += gid * 9;
#pragma unroll 3
            for (int i = 0; i < 3; i++) { // rows
#pragma unroll 3
                for (int j = 0; j < 3; j++) { // cols
                    atomicAdd(v_covars + i * 3 + j, v_covar[j][i]);
                }
            }
        }
    }
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
#pragma unroll 3
            for (int i = 0; i < 3; i++) { // rows
#pragma unroll 3
                for (int j = 0; j < 3; j++) { // cols
                    atomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                atomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor>
world_to_cam_fwd_tensor(const torch::Tensor &means,   // [N, 3]
                        const torch::Tensor &covars,  // [N, 3, 3]
                        const torch::Tensor &viewmats // [C, 4, 4]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(viewmats);

    int N = means.size(0);
    int C = viewmats.size(0);

    torch::Tensor means_c = torch::empty({C, N, 3}, means.options());
    torch::Tensor covars_c = torch::empty({C, N, 3, 3}, means.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        world_to_cam_fwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                  stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(),
            viewmats.data_ptr<float>(), means_c.data_ptr<float>(),
            covars_c.data_ptr<float>());
    }
    return std::make_tuple(means_c, covars_c);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
world_to_cam_bwd_tensor(const torch::Tensor &means,                    // [N, 3]
                        const torch::Tensor &covars,                   // [N, 3, 3]
                        const torch::Tensor &viewmats,                 // [C, 4, 4]
                        const at::optional<torch::Tensor> &v_means_c,  // [C, N, 3]
                        const at::optional<torch::Tensor> &v_covars_c, // [C, N, 3, 3]
                        const bool means_requires_grad, const bool covars_requires_grad,
                        const bool viewmats_requires_grad) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(viewmats);
    if (v_means_c.has_value()) {
        CHECK_INPUT(v_means_c.value());
    }
    if (v_covars_c.has_value()) {
        CHECK_INPUT(v_covars_c.value());
    }
    int N = means.size(0);
    int C = viewmats.size(0);

    torch::Tensor v_means, v_covars, v_viewmats;
    if (means_requires_grad) {
        v_means = torch::zeros({N, 3}, means.options());
    }
    if (covars_requires_grad) {
        v_covars = torch::zeros({N, 3, 3}, means.options());
    }
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros({C, 4, 4}, means.options());
    }

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        world_to_cam_bwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                  stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(),
            viewmats.data_ptr<float>(),
            v_means_c.has_value() ? v_means_c.value().data_ptr<float>() : nullptr,
            v_covars_c.has_value() ? v_covars_c.value().data_ptr<float>() : nullptr,
            means_requires_grad ? v_means.data_ptr<float>() : nullptr,
            covars_requires_grad ? v_covars.data_ptr<float>() : nullptr,
            viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr);
    }
    return std::make_tuple(v_means, v_covars, v_viewmats);
}

/****************************************************************************
 * Projection of Gaussians
 ****************************************************************************/

inline __device__ float inverse(const glm::mat2 M, glm::mat2 &Minv) {
    float det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (det <= 0.f) {
        return det;
    }
    float invDet = 1.f / det;
    Minv[0][0] = M[1][1] * invDet;
    Minv[0][1] = -M[0][1] * invDet;
    Minv[1][0] = Minv[0][1];
    Minv[1][1] = M[0][0] * invDet;
    return det;
}

template <class T>
inline __device__ void inverse_vjp(const T Minv, const T v_Minv, T &v_M) {
    // P = M^-1
    // df/dM = -P * df/dP * P
    v_M += -Minv * v_Minv * Minv;
}

__global__ void projection_fwd_kernel(const int C, const int N,
                                      const float *__restrict__ means,    // [N, 3]
                                      const float *__restrict__ covars,   // [N, 6]
                                      const float *__restrict__ viewmats, // [C, 4, 4]
                                      const float *__restrict__ Ks,       // [C, 3, 3]
                                      const int32_t image_width,
                                      const int32_t image_height, const float eps2d,
                                      const float near_plane,
                                      // outputs
                                      int32_t *__restrict__ radii, // [C, N]
                                      float *__restrict__ means2d, // [C, N, 2]
                                      float *__restrict__ depths,  // [C, N]
                                      float *__restrict__ conics   // [C, N, 3]
) {
    // parallelize over C * N.
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const int cid = idx / N; // camera id
    const int gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    covars += gid * 6;
    viewmats += cid * 16;
    Ks += cid * 9;

    // glm is column-major but input is row-major
    glm::mat3 R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    glm::vec3 mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane) {
        radii[idx] = 0;
        return;
    }

    // transform Gaussian covariance to camera space
    glm::mat3 covar = glm::mat3(covars[0], covars[1], covars[2], // 1st column
                                covars[1], covars[3], covars[4], // 2nd column
                                covars[2], covars[4], covars[5]  // 3rd column
    );
    glm::mat3 covar_c;
    covar_world_to_cam(R, covar, covar_c);

    // perspective projection
    glm::mat2 covar2d;
    glm::vec2 mean2d;
    persp_proj(mean_c, covar_c, Ks[0], Ks[4], Ks[2], Ks[5], image_width, image_height,
               covar2d, mean2d);
    if (eps2d > 0) {
        // avoid singularity and introduce some bluryness
        covar2d[0][0] += eps2d;
        covar2d[1][1] += eps2d;
    }

    // compute the inverse of the 2d covariance
    glm::mat2 covar2d_inv;
    float det = inverse(covar2d, covar2d_inv);
    if (det <= 0.f) {
        radii[idx] = 0;
        return;
    }

    // take 3 sigma as the radius (non differentiable)
    float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    float v1 = b + sqrt(max(0.1f, b * b - det));
    float v2 = b - sqrt(max(0.1f, b * b - det));
    float radius = ceil(3.f * sqrt(max(v1, v2)));

    // mask out gaussians outside the image region
    if (mean2d.x + radius <= 0 || mean2d.x - radius >= image_width ||
        mean2d.y + radius <= 0 || mean2d.y - radius >= image_height) {
        radii[idx] = 0;
        return;
    }

    // write to outputs
    radii[idx] = (int32_t)radius;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;
    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
}

__global__ void projection_bwd_kernel(
    // fwd inputs
    const int C, const int N,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ covars,   // [N, 6]
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width, const int32_t image_height,
    // fwd outputs
    const int32_t *__restrict__ radii, // [C, N]
    const float *__restrict__ conics,  // [C, N, 3]
    // grad outputs
    const float *__restrict__ v_means2d, // [C, N, 2]
    const float *__restrict__ v_depths,  // [C, N]
    const float *__restrict__ v_conics,  // [C, N, 3]
    // grad inputs
    float *__restrict__ v_means,   // [N, 3]
    float *__restrict__ v_covars,  // [N, 6]
    float *__restrict__ v_viewmats // [C, 4, 4] (optional as it can be very slow)
) {
    // parallelize over C * N.
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= C * N || radii[idx] <= 0) {
        return;
    }
    const int cid = idx / N; // camera id
    const int gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    covars += gid * 6;
    viewmats += cid * 16;
    Ks += cid * 9;

    conics += idx * 3;

    v_means2d += idx * 2;
    v_depths += idx;
    v_conics += idx * 3;

    // vjp: compute the inverse of the 2d covariance
    glm::mat2 covar2d_inv = glm::mat2(conics[0], conics[1], conics[1], conics[2]);
    glm::mat2 v_covar2d_inv =
        glm::mat2(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    glm::mat2 v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    // transform Gaussian to camera space
    glm::mat3 R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);
    glm::mat3 covar = glm::mat3(covars[0], covars[1], covars[2], // 1st column
                                covars[1], covars[3], covars[4], // 2nd column
                                covars[2], covars[4], covars[5]  // 3rd column
    );
    glm::vec3 mean_c;
    pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
    glm::mat3 covar_c;
    covar_world_to_cam(R, covar, covar_c);

    // vjp: perspective projection
    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    glm::mat3 v_covar_c(0.f);
    glm::vec3 v_mean_c(0.f);
    persp_proj_vjp(mean_c, covar_c, fx, fy, cx, cy, image_width, image_height,
                   v_covar2d, glm::make_vec2(v_means2d), v_mean_c, v_covar_c);

    // add contribution from v_depths
    v_mean_c.z += v_depths[0];

    // vjp: transform Gaussian covariance to camera space
    glm::vec3 v_mean(0.f);
    glm::mat3 v_covar(0.f);
    glm::mat3 v_R(0.f);
    glm::vec3 v_t(0.f);
    pos_world_to_cam_vjp(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
    covar_world_to_cam_vjp(R, covar, v_covar_c, v_R, v_covar);

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += gid * 3;
#pragma unroll 3
            for (int i = 0; i < 3; i++) {
                atomicAdd(v_means + i, v_mean[i]);
            }
        }
    }
    if (v_covars != nullptr) {
        warpSum(v_covar, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_covars += gid * 6;
            atomicAdd(v_covars, v_covar[0][0]);
            atomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
            atomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
            atomicAdd(v_covars + 3, v_covar[1][1]);
            atomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
            atomicAdd(v_covars + 5, v_covar[2][2]);
        }
    }
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
#pragma unroll 3
            for (int i = 0; i < 3; i++) { // rows
#pragma unroll 3
                for (int j = 0; j < 3; j++) { // cols
                    atomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                atomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
projection_fwd_tensor(const torch::Tensor &means,    // [N, 3]
                      const torch::Tensor &covars,   // [N, 6]
                      const torch::Tensor &viewmats, // [C, 4, 4]
                      const torch::Tensor &Ks,       // [C, 3, 3]
                      const int image_width, const int image_height, const float eps2d,
                      const float near_plane) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

    int N = means.size(0);    // number of gaussians
    int C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor radii = torch::empty({C, N}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor depths = torch::empty({C, N}, means.options());
    torch::Tensor conics = torch::empty({C, N, 3}, means.options());
    if (C && N) {
        projection_fwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(),
            viewmats.data_ptr<float>(), Ks.data_ptr<float>(), image_width, image_height,
            eps2d, near_plane, radii.data_ptr<int32_t>(), means2d.data_ptr<float>(),
            depths.data_ptr<float>(), conics.data_ptr<float>());
    }
    return std::make_tuple(radii, means2d, depths, conics);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> projection_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,    // [N, 3]
    const torch::Tensor &covars,   // [N, 6]
    const torch::Tensor &viewmats, // [C, 4, 4]
    const torch::Tensor &Ks,       // [C, 3, 3]
    const int image_width, const int image_height,
    // fwd outputs
    const torch::Tensor &radii,  // [C, N]
    const torch::Tensor &conics, // [C, N, 3]
    // grad outputs
    const torch::Tensor &v_means2d, // [C, N, 2]
    const torch::Tensor &v_depths,  // [C, N]
    const torch::Tensor &v_conics,  // [C, N, 3]
    const bool viewmats_requires_grad) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(radii);
    CHECK_INPUT(conics);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_conics);

    int N = means.size(0);    // number of gaussians
    int C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means = torch::zeros_like(means);
    torch::Tensor v_covars = torch::zeros_like(covars);
    torch::Tensor v_viewmats;
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros_like(viewmats);
    }
    if (C && N) {
        projection_bwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(),
            viewmats.data_ptr<float>(), Ks.data_ptr<float>(), image_width, image_height,
            radii.data_ptr<int32_t>(), conics.data_ptr<float>(),
            v_means2d.data_ptr<float>(), v_depths.data_ptr<float>(),
            v_conics.data_ptr<float>(), v_means.data_ptr<float>(),
            v_covars.data_ptr<float>(),
            viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr);
    }
    return std::make_tuple(v_means, v_covars, v_viewmats);
}