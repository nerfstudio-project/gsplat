#include "bindings.h"
#include "reduce.cuh"
#include "quaternion.cuh"
#include "transform.cuh"

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <ATen/cuda/Atomic.cuh>

namespace gsplat {

namespace cg = cooperative_groups;

__device__ float inverse(const mat2 M, mat2 &Minv) {
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

__device__ void inverse_vjp(const mat2 Minv, const mat2 v_Minv, mat2 &v_M) {
    // P = M^-1
    // df/dM = -P * df/dP * P
    v_M += -Minv * v_Minv * Minv;
}


__device__ float add_blur(const float eps2d, mat2 &covar, float &compensation) {
    float det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    float det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}


__device__ void add_blur_vjp(
    const float eps2d,
    const mat2 conic_blur,
    const float compensation,
    const float v_compensation,
    mat2 &v_covar
) {
    // comp = sqrt(det(covar) / det(covar_blur))

    // d [det(M)] / d M = adj(M)
    // d [det(M + aI)] / d M  = adj(M + aI) = adj(M) + a * I
    // d [det(M) / det(M + aI)] / d M
    // = (det(M + aI) * adj(M) - det(M) * adj(M + aI)) / (det(M + aI))^2
    // = adj(M) / det(M + aI) - adj(M + aI) / det(M + aI) * comp^2
    // = (adj(M) - adj(M + aI) * comp^2) / det(M + aI)
    // given that adj(M + aI) = adj(M) + a * I
    // = (adj(M + aI) - aI - adj(M + aI) * comp^2) / det(M + aI)
    // given that adj(M) / det(M) = inv(M)
    // = (1 - comp^2) * inv(M + aI) - aI / det(M + aI)
    // given det(inv(M)) = 1 / det(M)
    // = (1 - comp^2) * inv(M + aI) - aI * det(inv(M + aI))
    // = (1 - comp^2) * conic_blur - aI * det(conic_blur)

    float det_conic_blur = conic_blur[0][0] * conic_blur[1][1] -
                        conic_blur[0][1] * conic_blur[1][0];
    float v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
    float one_minus_sqr_comp = 1 - compensation * compensation;
    v_covar[0][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][0] -
                                    eps2d * det_conic_blur);
    v_covar[0][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][1]);
    v_covar[1][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][0]);
    v_covar[1][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][1] -
                                    eps2d * det_conic_blur);
}


__device__ void ortho_proj(
    // inputs
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2 &cov2d,
    vec2 &mean2d
) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    // mat3x2 is 3 columns x 2 rows.
    mat3x2 J = mat3x2(
        fx,
        0.f, // 1st column
        0.f,
        fy, // 2nd column
        0.f,
        0.f // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2({fx * x + cx, fy * y + cy});
}


__device__ void ortho_proj_vjp(
    // fwd inputs
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2 v_cov2d,
    const vec2 v_mean2d,
    // grad inputs
    vec3 &v_mean3d,
    mat3 &v_cov3d
) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    // mat3x2 is 3 columns x 2 rows.
    mat3x2 J = mat3x2(
        fx,
        0.f, // 1st column
        0.f,
        fy, // 2nd column
        0.f,
        0.f // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * df/dpixx
    // df/dy = fy * df/dpixy
    // df/dz = 0
    v_mean3d += vec3(fx * v_mean2d[0], fy * v_mean2d[1], 0.f);
}


__device__ void persp_proj(
    // inputs
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2 &cov2d,
    vec2 &mean2d
) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    float tan_fovx = 0.5f * width / fx;
    float tan_fovy = 0.5f * height / fy;
    float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    float rz = 1.f / z;
    float rz2 = rz * rz;
    float tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    float ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2 J = mat3x2(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2({fx * x * rz + cx, fy * y * rz + cy});
}


__device__ void persp_proj_vjp(
    // fwd inputs
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2 v_cov2d,
    const vec2 v_mean2d,
    // grad inputs
    vec3 &v_mean3d,
    mat3 &v_cov3d
) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    float tan_fovx = 0.5f * width / fx;
    float tan_fovy = 0.5f * height / fy;
    float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    float rz = 1.f / z;
    float rz2 = rz * rz;
    float tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    float ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2 J = mat3x2(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    v_mean3d += vec3(
        fx * rz * v_mean2d[0],
        fy * rz * v_mean2d[1],
        -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2
    );

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    float rz3 = rz2 * rz;
    mat3x2 v_J = v_cov2d * J * glm::transpose(cov3d) +
                    glm::transpose(v_cov2d) * J * cov3d;

    // fov clipping
    if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
        v_mean3d.x += -fx * rz2 * v_J[2][0];
    } else {
        v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
    }
    if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
        v_mean3d.y += -fy * rz2 * v_J[2][1];
    } else {
        v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
    }
    v_mean3d.z += -fx * rz2 * v_J[0][0] - fy * rz2 * v_J[1][1] +
                    2.f * fx * tx * rz3 * v_J[2][0] +
                    2.f * fy * ty * rz3 * v_J[2][1];
}


__device__ void fisheye_proj(
    // inputs
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2 &cov2d,
    vec2 &mean2d
) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    float eps = 0.0000001f;
    float xy_len = glm::length(glm::vec2({x, y})) + eps;
    float theta = glm::atan(xy_len, z + eps);
    mean2d =
        vec2({x * fx * theta / xy_len + cx, y * fy * theta / xy_len + cy});

    float x2 = x * x + eps;
    float y2 = y * y;
    float xy = x * y;
    float x2y2 = x2 + y2;
    float x2y2z2_inv = 1.f / (x2y2 + z * z);

    float b = glm::atan(xy_len, z) / xy_len / x2y2;
    float a = z * x2y2z2_inv / (x2y2);
    mat3x2 J = mat3x2(
        fx * (x2 * a + y2 * b),
        fy * xy * (a - b),
        fx * xy * (a - b),
        fy * (y2 * a + x2 * b),
        -fx * x * x2y2z2_inv,
        -fy * y * x2y2z2_inv
    );
    cov2d = J * cov3d * glm::transpose(J);
}


__device__ void fisheye_proj_vjp(
    // fwd inputs
    const vec3 mean3d,
    const mat3 cov3d,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2 v_cov2d,
    const vec2 v_mean2d,
    // grad inputs
    vec3 &v_mean3d,
    mat3 &v_cov3d
) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    const float eps = 0.0000001f;
    float x2 = x * x + eps;
    float y2 = y * y;
    float xy = x * y;
    float x2y2 = x2 + y2;
    float len_xy = length(glm::vec2({x, y})) + eps;
    const float x2y2z2 = x2y2 + z * z;
    float x2y2z2_inv = 1.f / x2y2z2;
    float b = glm::atan(len_xy, z) / len_xy / x2y2;
    float a = z * x2y2z2_inv / (x2y2);
    v_mean3d += vec3(
        fx * (x2 * a + y2 * b) * v_mean2d[0] + fy * xy * (a - b) * v_mean2d[1],
        fx * xy * (a - b) * v_mean2d[0] + fy * (y2 * a + x2 * b) * v_mean2d[1],
        -fx * x * x2y2z2_inv * v_mean2d[0] - fy * y * x2y2z2_inv * v_mean2d[1]
    );

    const float theta = glm::atan(len_xy, z);
    const float J_b = theta / len_xy / x2y2;
    const float J_a = z * x2y2z2_inv / (x2y2);
    // mat3x2 is 3 columns x 2 rows.
    mat3x2 J = mat3x2(
        fx * (x2 * J_a + y2 * J_b),
        fy * xy * (J_a - J_b), // 1st column
        fx * xy * (J_a - J_b),
        fy * (y2 * J_a + x2 * J_b), // 2nd column
        -fx * x * x2y2z2_inv,
        -fy * y * x2y2z2_inv // 3rd column
    );
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    mat3x2 v_J = v_cov2d * J * glm::transpose(cov3d) +
                    glm::transpose(v_cov2d) * J * cov3d;
    float l4 = x2y2z2 * x2y2z2;

    float E = -l4 * x2y2 * theta + x2y2z2 * x2y2 * len_xy * z;
    float F = 3 * l4 * theta - 3 * x2y2z2 * len_xy * z - 2 * x2y2 * len_xy * z;

    float A = x * (3 * E + x2 * F);
    float B = y * (E + x2 * F);
    float C = x * (E + y2 * F);
    float D = y * (3 * E + y2 * F);

    float S1 = x2 - y2 - z * z;
    float S2 = y2 - x2 - z * z;
    float inv1 = x2y2z2_inv * x2y2z2_inv;
    float inv2 = inv1 / (x2y2 * x2y2 * len_xy);

    float dJ_dx00 = fx * A * inv2;
    float dJ_dx01 = fx * B * inv2;
    float dJ_dx02 = fx * S1 * inv1;
    float dJ_dx10 = fy * B * inv2;
    float dJ_dx11 = fy * C * inv2;
    float dJ_dx12 = 2.f * fy * xy * inv1;

    float dJ_dy00 = dJ_dx01;
    float dJ_dy01 = fx * C * inv2;
    float dJ_dy02 = 2.f * fx * xy * inv1;
    float dJ_dy10 = dJ_dx11;
    float dJ_dy11 = fy * D * inv2;
    float dJ_dy12 = fy * S2 * inv1;

    float dJ_dz00 = dJ_dx02;
    float dJ_dz01 = dJ_dy02;
    float dJ_dz02 = 2.f * fx * x * z * inv1;
    float dJ_dz10 = dJ_dx12;
    float dJ_dz11 = dJ_dy12;
    float dJ_dz12 = 2.f * fy * y * z * inv1;

    float dL_dtx_raw = dJ_dx00 * v_J[0][0] + dJ_dx01 * v_J[1][0] +
                    dJ_dx02 * v_J[2][0] + dJ_dx10 * v_J[0][1] +
                    dJ_dx11 * v_J[1][1] + dJ_dx12 * v_J[2][1];
    float dL_dty_raw = dJ_dy00 * v_J[0][0] + dJ_dy01 * v_J[1][0] +
                    dJ_dy02 * v_J[2][0] + dJ_dy10 * v_J[0][1] +
                    dJ_dy11 * v_J[1][1] + dJ_dy12 * v_J[2][1];
    float dL_dtz_raw = dJ_dz00 * v_J[0][0] + dJ_dz01 * v_J[1][0] +
                    dJ_dz02 * v_J[2][0] + dJ_dz10 * v_J[0][1] +
                    dJ_dz11 * v_J[1][1] + dJ_dz12 * v_J[2][1];
    v_mean3d.x += dL_dtx_raw;
    v_mean3d.y += dL_dty_raw;
    v_mean3d.z += dL_dtz_raw;
}




/****************************************************************************
 * Perspective Projection Forward Pass
 ****************************************************************************/


__global__ void proj_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,  // [C, N, 3]
    const float *__restrict__ covars, // [C, N, 3, 3]
    const float *__restrict__ Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    float *__restrict__ means2d, // [C, N, 2]
    float *__restrict__ covars2d // [C, N, 2, 2]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    // const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += idx * 3;
    covars += idx * 9;
    Ks += cid * 9;
    means2d += idx * 2;
    covars2d += idx * 4;

    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat2 covar2d(0.f);
    vec2 mean2d(0.f);
    const vec3 mean = glm::make_vec3(means);
    const mat3 covar = glm::make_mat3(covars);

    switch (camera_model) {
        case CameraModelType::PINHOLE: // perspective projection
            persp_proj(mean, covar, fx, fy, cx, cy, width, height, covar2d, mean2d);
            break;
        case CameraModelType::ORTHO: // orthographic projection
            ortho_proj(mean, covar, fx, fy, cx, cy, width, height, covar2d, mean2d);
            break;
        case CameraModelType::FISHEYE: // fisheye projection
            fisheye_proj(mean, covar, fx, fy, cx, cy, width, height, covar2d, mean2d);
            break;
    }

    // write to outputs: glm is column-major but we want row-major
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; i++) { // rows
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t j = 0; j < 2; j++) { // cols
            covars2d[i * 2 + j] = covar2d[j][i];
        }
    }
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; i++) {
        means2d[i] = mean2d[i];
    }
}

std::tuple<torch::Tensor, torch::Tensor> proj_fwd_tensor(
    const torch::Tensor &means,  // [C, N, 3]
    const torch::Tensor &covars, // [C, N, 3, 3]
    const torch::Tensor &Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(covars);
    GSPLAT_CHECK_INPUT(Ks);

    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor covars2d = torch::empty({C, N, 2, 2}, covars.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
                proj_fwd_kernel
                    <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                       GSPLAT_N_THREADS,
                       0,
                       stream>>>(
                        C,
                        N,
                        means.data_ptr<float>(),
                        covars.data_ptr<float>(),
                        Ks.data_ptr<float>(),
                        width,
                        height,
                        camera_model,
                        means2d.data_ptr<float>(),
                        covars2d.data_ptr<float>()
                    );
    }
    return std::make_tuple(means2d, covars2d);
}


/****************************************************************************
 * Perspective Projection Backward Pass
 ****************************************************************************/


 __global__ void proj_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,  // [C, N, 3]
    const float *__restrict__ covars, // [C, N, 3, 3]
    const float *__restrict__ Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const float *__restrict__ v_means2d,  // [C, N, 2]
    const float *__restrict__ v_covars2d, // [C, N, 2, 2]
    float *__restrict__ v_means,          // [C, N, 3]
    float *__restrict__ v_covars          // [C, N, 3, 3]
) {

    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    // const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += idx * 3;
    covars += idx * 9;
    v_means += idx * 3;
    v_covars += idx * 9;
    Ks += cid * 9;
    v_means2d += idx * 2;
    v_covars2d += idx * 4;

    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat3 v_covar(0.f);
    vec3 v_mean(0.f);
    const vec3 mean = glm::make_vec3(means);
    const mat3 covar = glm::make_mat3(covars);
    const vec2 v_mean2d = glm::make_vec2(v_means2d);
    const mat2 v_covar2d = glm::make_mat2(v_covars2d);

    switch (camera_model) {
        case CameraModelType::PINHOLE: // perspective projection
            persp_proj_vjp(
                mean,
                covar,
                fx,
                fy,
                cx,
                cy,
                width,
                height,
                glm::transpose(v_covar2d),
                v_mean2d,
                v_mean,
                v_covar
            );
            break;
        case CameraModelType::ORTHO: // orthographic projection
            ortho_proj_vjp(
                mean,
                covar,
                fx,
                fy,
                cx,
                cy,
                width,
                height,
                glm::transpose(v_covar2d),
                v_mean2d,
                v_mean,
                v_covar
            );
            break;
        case CameraModelType::FISHEYE: // fisheye projection
            fisheye_proj_vjp(
                mean,
                covar,
                fx,
                fy,
                cx,
                cy,
                width,
                height,
                glm::transpose(v_covar2d),
                v_mean2d,
                v_mean,
                v_covar
            );
            break;
    }

    // write to outputs: glm is column-major but we want row-major
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) { // rows
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t j = 0; j < 3; j++) { // cols
            v_covars[i * 3 + j] = v_covar[j][i];
        }
    }

    GSPLAT_PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) {
        v_means[i] = v_mean[i];
    }
}

std::tuple<torch::Tensor, torch::Tensor> proj_bwd_tensor(
    const torch::Tensor &means,  // [C, N, 3]
    const torch::Tensor &covars, // [C, N, 3, 3]
    const torch::Tensor &Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const torch::Tensor &v_means2d, // [C, N, 2]
    const torch::Tensor &v_covars2d // [C, N, 2, 2]
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(covars);
    GSPLAT_CHECK_INPUT(Ks);
    GSPLAT_CHECK_INPUT(v_means2d);
    GSPLAT_CHECK_INPUT(v_covars2d);

    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    torch::Tensor v_means = torch::empty({C, N, 3}, means.options());
    torch::Tensor v_covars = torch::empty({C, N, 3, 3}, means.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
                proj_bwd_kernel
                    <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                       GSPLAT_N_THREADS,
                       0,
                       stream>>>(
                        C,
                        N,
                        means.data_ptr<float>(),
                        covars.data_ptr<float>(),
                        Ks.data_ptr<float>(),
                        width,
                        height,
                        camera_model,
                        v_means2d.data_ptr<float>(),
                        v_covars2d.data_ptr<float>(),
                        v_means.data_ptr<float>(),
                        v_covars.data_ptr<float>()
                    );
    }
    return std::make_tuple(v_means, v_covars);
}



/****************************************************************************
 * Projection of Gaussians (Single Batch) Forward Pass
 ****************************************************************************/


 __global__ void fully_fused_projection_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ covars,   // [N, 6] optional
    const float *__restrict__ quats,    // [N, 4] optional
    const float *__restrict__ scales,   // [N, 3] optional
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ radii,  // [C, N]
    float *__restrict__ means2d,      // [C, N, 2]
    float *__restrict__ depths,       // [C, N]
    float *__restrict__ conics,       // [C, N, 3]
    float *__restrict__ compensations // [C, N] optional
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    // glm is column-major but input is row-major
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx] = 0;
        return;
    }

    // transform Gaussian covariance to camera space
    mat3 covar;
    if (covars != nullptr) {
        covars += gid * 6;
        covar = mat3(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quats += gid * 4;
        scales += gid * 3;
        quat_scale_to_covar_preci(
            glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr
        );
    }
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // perspective projection
    mat2 covar2d;
    vec2 mean2d;

    switch (camera_model) {
        case CameraModelType::PINHOLE: // perspective projection
            persp_proj(
                mean_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
                covar2d,
                mean2d
            );
            break;
        case CameraModelType::ORTHO: // orthographic projection
            ortho_proj(
                mean_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
                covar2d,
                mean2d
            );
            break;
        case CameraModelType::FISHEYE: // fisheye projection
            fisheye_proj(
                mean_c,
                covar_c,
                Ks[0],
                Ks[4],
                Ks[2],
                Ks[5],
                image_width,
                image_height,
                covar2d,
                mean2d
            );
            break;
    }

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2 covar2d_inv;
    inverse(covar2d, covar2d_inv);

    // take 3 sigma as the radius (non differentiable)
    float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    float v1 = b + sqrt(max(0.01f, b * b - det));
    float radius = ceil(3.f * sqrt(v1));
    // float v2 = b - sqrt(max(0.1f, b * b - det));
    // float radius = ceil(3.f * sqrt(max(v1, v2)));

    if (radius <= radius_clip) {
        radii[idx] = 0;
        return;
    }

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
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_fwd_tensor(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6] optional
    const at::optional<torch::Tensor> &quats,  // [N, 4] optional
    const at::optional<torch::Tensor> &scales, // [N, 3] optional
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    if (covars.has_value()) {
        GSPLAT_CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        GSPLAT_CHECK_INPUT(quats.value());
        GSPLAT_CHECK_INPUT(scales.value());
    }
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor radii =
        torch::empty({C, N}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor depths = torch::empty({C, N}, means.options());
    torch::Tensor conics = torch::empty({C, N, 3}, means.options());
    torch::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({C, N}, means.options());
    }
    if (C && N) {
        fully_fused_projection_fwd_kernel
            <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
               GSPLAT_N_THREADS,
               0,
               stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                quats.has_value() ? quats.value().data_ptr<float>() : nullptr,
                scales.has_value() ? scales.value().data_ptr<float>() : nullptr,
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                camera_model,
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<float>(),
                depths.data_ptr<float>(),
                conics.data_ptr<float>(),
                calc_compensations ? compensations.data_ptr<float>() : nullptr
            );
    }
    return std::make_tuple(radii, means2d, depths, conics, compensations);
}



/****************************************************************************
 * Projection of Gaussians (Single Batch) Backward Pass
 ****************************************************************************/


 __global__ void fully_fused_projection_bwd_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ covars,   // [N, 6] optional
    const float *__restrict__ quats,    // [N, 4] optional
    const float *__restrict__ scales,   // [N, 3] optional
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const int32_t *__restrict__ radii,   // [C, N]
    const float *__restrict__ conics,        // [C, N, 3]
    const float *__restrict__ compensations, // [C, N] optional
    // grad outputs
    const float *__restrict__ v_means2d,       // [C, N, 2]
    const float *__restrict__ v_depths,        // [C, N]
    const float *__restrict__ v_conics,        // [C, N, 3]
    const float *__restrict__ v_compensations, // [C, N] optional
    // grad inputs
    float *__restrict__ v_means,   // [N, 3]
    float *__restrict__ v_covars,  // [N, 6] optional
    float *__restrict__ v_quats,   // [N, 4] optional
    float *__restrict__ v_scales,  // [N, 3] optional
    float *__restrict__ v_viewmats // [C, 4, 4] optional
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N || radii[idx] <= 0) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    conics += idx * 3;

    v_means2d += idx * 2;
    v_depths += idx;
    v_conics += idx * 3;

    // vjp: compute the inverse of the 2d covariance
    mat2 covar2d_inv = mat2(conics[0], conics[1], conics[1], conics[2]);
    mat2 v_covar2d_inv =
        mat2(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    mat2 v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    if (v_compensations != nullptr) {
        // vjp: compensation term
        const float compensation = compensations[idx];
        const float v_compensation = v_compensations[idx];
        add_blur_vjp(
            eps2d, covar2d_inv, compensation, v_compensation, v_covar2d
        );
    }

    // transform Gaussian to camera space
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    mat3 covar;
    vec4 quat;
    vec3 scale;
    if (covars != nullptr) {
        covars += gid * 6;
        covar = mat3(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quat = glm::make_vec4(quats + gid * 4);
        scale = glm::make_vec3(scales + gid * 3);
        quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
    }
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // vjp: perspective projection
    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat3 v_covar_c(0.f);
    vec3 v_mean_c(0.f);

    switch (camera_model) {
        case CameraModelType::PINHOLE: // perspective projection
            persp_proj_vjp(
                mean_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                image_width,
                image_height,
                v_covar2d,
                glm::make_vec2(v_means2d),
                v_mean_c,
                v_covar_c
            );
            break;
        case CameraModelType::ORTHO: // orthographic projection
            ortho_proj_vjp(
                mean_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                image_width,
                image_height,
                v_covar2d,
                glm::make_vec2(v_means2d),
                v_mean_c,
                v_covar_c
            );
            break;
        case CameraModelType::FISHEYE: // fisheye projection
            fisheye_proj_vjp(
                mean_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                image_width,
                image_height,
                v_covar2d,
                glm::make_vec2(v_means2d),
                v_mean_c,
                v_covar_c
            );
            break;
    }

    // add contribution from v_depths
    v_mean_c.z += v_depths[0];

    // vjp: transform Gaussian covariance to camera space
    vec3 v_mean(0.f);
    mat3 v_covar(0.f);
    mat3 v_R(0.f);
    vec3 v_t(0.f);
    posW2C_VJP(
        R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean
    );
    covarW2C_VJP(R, covar, v_covar_c, v_R, v_covar);

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += gid * 3;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means + i, v_mean[i]);
            }
        }
    }
    if (v_covars != nullptr) {
        // Output gradients w.r.t. the covariance matrix
        warpSum(v_covar, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_covars += gid * 6;
            gpuAtomicAdd(v_covars, v_covar[0][0]);
            gpuAtomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
            gpuAtomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
            gpuAtomicAdd(v_covars + 3, v_covar[1][1]);
            gpuAtomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
            gpuAtomicAdd(v_covars + 5, v_covar[2][2]);
        }
    } else {
        // Directly output gradients w.r.t. the quaternion and scale
        mat3 rotmat = quat_to_rotmat(quat);
        vec4 v_quat(0.f);
        vec3 v_scale(0.f);
        quat_scale_to_covar_vjp(
            quat, scale, rotmat, v_covar, v_quat, v_scale
        );
        warpSum(v_quat, warp_group_g);
        warpSum(v_scale, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_quats += gid * 4;
            v_scales += gid * 3;
            gpuAtomicAdd(v_quats, v_quat[0]);
            gpuAtomicAdd(v_quats + 1, v_quat[1]);
            gpuAtomicAdd(v_quats + 2, v_quat[2]);
            gpuAtomicAdd(v_quats + 3, v_quat[3]);
            gpuAtomicAdd(v_scales, v_scale[0]);
            gpuAtomicAdd(v_scales + 1, v_scale[1]);
            gpuAtomicAdd(v_scales + 2, v_scale[2]);
        }
    }
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6] optional
    const at::optional<torch::Tensor> &quats,  // [N, 4] optional
    const at::optional<torch::Tensor> &scales, // [N, 3] optional
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const torch::Tensor &radii,                       // [C, N]
    const torch::Tensor &conics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &compensations, // [C, N] optional
    // grad outputs
    const torch::Tensor &v_means2d,                     // [C, N, 2]
    const torch::Tensor &v_depths,                      // [C, N]
    const torch::Tensor &v_conics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &v_compensations, // [C, N] optional
    const bool viewmats_requires_grad
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    if (covars.has_value()) {
        GSPLAT_CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        GSPLAT_CHECK_INPUT(quats.value());
        GSPLAT_CHECK_INPUT(scales.value());
    }
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);
    GSPLAT_CHECK_INPUT(radii);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(v_means2d);
    GSPLAT_CHECK_INPUT(v_depths);
    GSPLAT_CHECK_INPUT(v_conics);
    if (compensations.has_value()) {
        GSPLAT_CHECK_INPUT(compensations.value());
    }
    if (v_compensations.has_value()) {
        GSPLAT_CHECK_INPUT(v_compensations.value());
        assert(compensations.has_value());
    }

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means = torch::zeros_like(means);
    torch::Tensor v_covars, v_quats, v_scales; // optional
    if (covars.has_value()) {
        v_covars = torch::zeros_like(covars.value());
    } else {
        v_quats = torch::zeros_like(quats.value());
        v_scales = torch::zeros_like(scales.value());
    }
    torch::Tensor v_viewmats;
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros_like(viewmats);
    }
    if (C && N) {
        fully_fused_projection_bwd_kernel
            <<<(C * N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
               GSPLAT_N_THREADS,
               0,
               stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                covars.has_value() ? nullptr : quats.value().data_ptr<float>(),
                covars.has_value() ? nullptr : scales.value().data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                eps2d,
                camera_model,
                radii.data_ptr<int32_t>(),
                conics.data_ptr<float>(),
                compensations.has_value()
                    ? compensations.value().data_ptr<float>()
                    : nullptr,
                v_means2d.data_ptr<float>(),
                v_depths.data_ptr<float>(),
                v_conics.data_ptr<float>(),
                v_compensations.has_value()
                    ? v_compensations.value().data_ptr<float>()
                    : nullptr,
                v_means.data_ptr<float>(),
                covars.has_value() ? v_covars.data_ptr<float>() : nullptr,
                covars.has_value() ? nullptr : v_quats.data_ptr<float>(),
                covars.has_value() ? nullptr : v_scales.data_ptr<float>(),
                viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr
            );
    }
    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}



/****************************************************************************
 * Projection of Gaussians (Batched) Forward Pass
 ****************************************************************************/


 __global__ void fully_fused_projection_packed_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ covars,   // [N, 6] Optional
    const float *__restrict__ quats,    // [N, 4] Optional
    const float *__restrict__ scales,   // [N, 3] Optional
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const int32_t
        *__restrict__ block_accum,    // [C * blocks_per_row] packing helper
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ block_cnts, // [C * blocks_per_row] packing helper
    int32_t *__restrict__ indptr,       // [C + 1]
    int64_t *__restrict__ camera_ids,   // [nnz]
    int64_t *__restrict__ gaussian_ids, // [nnz]
    int32_t *__restrict__ radii,        // [nnz]
    float *__restrict__ means2d,            // [nnz, 2]
    float *__restrict__ depths,             // [nnz]
    float *__restrict__ conics,             // [nnz, 3]
    float *__restrict__ compensations       // [nnz] optional
) {
    int32_t blocks_per_row = gridDim.x;

    int32_t row_idx = blockIdx.y; // cid
    int32_t block_col_idx = blockIdx.x;
    int32_t block_idx = row_idx * blocks_per_row + block_col_idx;

    int32_t col_idx = block_col_idx * blockDim.x + threadIdx.x; // gid

    bool valid = (row_idx < C) && (col_idx < N);

    // check if points are with camera near and far plane
    vec3 mean_c;
    mat3 R;
    if (valid) {
        // shift pointers to the current camera and gaussian
        means += col_idx * 3;
        viewmats += row_idx * 16;

        // glm is column-major but input is row-major
        R = mat3(
            viewmats[0],
            viewmats[4],
            viewmats[8], // 1st column
            viewmats[1],
            viewmats[5],
            viewmats[9], // 2nd column
            viewmats[2],
            viewmats[6],
            viewmats[10] // 3rd column
        );
        vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

        // transform Gaussian center to camera space
        posW2C(R, t, glm::make_vec3(means), mean_c);
        if (mean_c.z < near_plane || mean_c.z > far_plane) {
            valid = false;
        }
    }

    // check if the perspective projection is valid.
    mat2 covar2d;
    vec2 mean2d;
    mat2 covar2d_inv;
    float compensation;
    float det;
    if (valid) {
        // transform Gaussian covariance to camera space
        mat3 covar;
        if (covars != nullptr) {
            // if a precomputed covariance is provided
            covars += col_idx * 6;
            covar = mat3(
                covars[0],
                covars[1],
                covars[2], // 1st column
                covars[1],
                covars[3],
                covars[4], // 2nd column
                covars[2],
                covars[4],
                covars[5] // 3rd column
            );
        } else {
            // if not then compute it from quaternions and scales
            quats += col_idx * 4;
            scales += col_idx * 3;
            quat_scale_to_covar_preci(
                glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr
            );
        }
        mat3 covar_c;
        covarW2C(R, covar, covar_c);
        
        Ks += row_idx * 9;
        switch (camera_model) {
            case CameraModelType::PINHOLE: // perspective projection
                persp_proj(
                    mean_c,
                    covar_c,
                    Ks[0],
                    Ks[4],
                    Ks[2],
                    Ks[5],
                    image_width,
                    image_height,
                    covar2d,
                    mean2d
                );
                break;
            case CameraModelType::ORTHO: // orthographic projection
                ortho_proj(
                    mean_c,
                    covar_c,
                    Ks[0],
                    Ks[4],
                    Ks[2],
                    Ks[5],
                    image_width,
                    image_height,
                    covar2d,
                    mean2d
                );
                break;
            case CameraModelType::FISHEYE: // fisheye projection
                fisheye_proj(
                    mean_c,
                    covar_c,
                    Ks[0],
                    Ks[4],
                    Ks[2],
                    Ks[5],
                    image_width,
                    image_height,
                    covar2d,
                    mean2d
                );
                break;
        }

        det = add_blur(eps2d, covar2d, compensation);
        if (det <= 0.f) {
            valid = false;
        } else {
            // compute the inverse of the 2d covariance
            inverse(covar2d, covar2d_inv);
        }
    }

    // check if the points are in the image region
    float radius;
    if (valid) {
        // take 3 sigma as the radius (non differentiable)
        float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
        float v1 = b + sqrt(max(0.1f, b * b - det));
        float v2 = b - sqrt(max(0.1f, b * b - det));
        radius = ceil(3.f * sqrt(max(v1, v2)));

        if (radius <= radius_clip) {
            valid = false;
        }

        // mask out gaussians outside the image region
        if (mean2d.x + radius <= 0 || mean2d.x - radius >= image_width ||
            mean2d.y + radius <= 0 || mean2d.y - radius >= image_height) {
            valid = false;
        }
    }

    int32_t thread_data = static_cast<int32_t>(valid);
    if (block_cnts != nullptr) {
        // First pass: compute the block-wide sum
        int32_t aggregate;
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockReduce<int32_t, GSPLAT_N_THREADS> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            aggregate = BlockReduce(temp_storage).Sum(thread_data);
        } else {
            aggregate = 0;
        }
        if (threadIdx.x == 0) {
            block_cnts[block_idx] = aggregate;
        }
    } else {
        // Second pass: write out the indices of the non zero elements
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockScan<int32_t, GSPLAT_N_THREADS> BlockScan;
            __shared__ typename BlockScan::TempStorage temp_storage;
            BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
        }
        if (valid) {
            if (block_idx > 0) {
                int32_t offset = block_accum[block_idx - 1];
                thread_data += offset;
            }
            // write to outputs
            camera_ids[thread_data] = row_idx;   // cid
            gaussian_ids[thread_data] = col_idx; // gid
            radii[thread_data] = (int32_t)radius;
            means2d[thread_data * 2] = mean2d.x;
            means2d[thread_data * 2 + 1] = mean2d.y;
            depths[thread_data] = mean_c.z;
            conics[thread_data * 3] = covar2d_inv[0][0];
            conics[thread_data * 3 + 1] = covar2d_inv[0][1];
            conics[thread_data * 3 + 2] = covar2d_inv[1][1];
            if (compensations != nullptr) {
                compensations[thread_data] = compensation;
            }
        }
        // lane 0 of the first block in each row writes the indptr
        if (threadIdx.x == 0 && block_col_idx == 0) {
            if (row_idx == 0) {
                indptr[0] = 0;
                indptr[C] = block_accum[C * blocks_per_row - 1];
            } else {
                indptr[row_idx] = block_accum[block_idx - 1];
            }
        }
    }
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_packed_fwd_tensor(
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6]
    const at::optional<torch::Tensor> &quats,  // [N, 3]
    const at::optional<torch::Tensor> &scales, // [N, 3]
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    if (covars.has_value()) {
        GSPLAT_CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        GSPLAT_CHECK_INPUT(quats.value());
        GSPLAT_CHECK_INPUT(scales.value());
    }
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    auto opt = means.options().dtype(torch::kInt32);

    uint32_t nrows = C;
    uint32_t ncols = N;
    uint32_t blocks_per_row = (ncols + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS;

    dim3 threads = {GSPLAT_N_THREADS, 1, 1};
    // limit on the number of blocks: [2**31 - 1, 65535, 65535]
    dim3 blocks = {blocks_per_row, nrows, 1};

    // first pass
    int32_t nnz;
    torch::Tensor block_accum;
    if (C && N) {
        torch::Tensor block_cnts = torch::empty({nrows * blocks_per_row}, opt);
        fully_fused_projection_packed_fwd_kernel
            <<<blocks, threads, 0, stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                quats.has_value() ? quats.value().data_ptr<float>() : nullptr,
                scales.has_value() ? scales.value().data_ptr<float>() : nullptr,
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                nullptr,
                camera_model,
                block_cnts.data_ptr<int32_t>(),
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr
            );
        block_accum = torch::cumsum(block_cnts, 0, torch::kInt32);
        nnz = block_accum[-1].item<int32_t>();
    } else {
        nnz = 0;
    }

    // second pass
    torch::Tensor indptr = torch::empty({C + 1}, opt);
    torch::Tensor camera_ids = torch::empty({nnz}, opt.dtype(torch::kInt64));
    torch::Tensor gaussian_ids = torch::empty({nnz}, opt.dtype(torch::kInt64));
    torch::Tensor radii =
        torch::empty({nnz}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({nnz, 2}, means.options());
    torch::Tensor depths = torch::empty({nnz}, means.options());
    torch::Tensor conics = torch::empty({nnz, 3}, means.options());
    torch::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({nnz}, means.options());
    }

    if (nnz) {
        fully_fused_projection_packed_fwd_kernel
            <<<blocks, threads, 0, stream>>>(
                C,
                N,
                means.data_ptr<float>(),
                covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                quats.has_value() ? quats.value().data_ptr<float>() : nullptr,
                scales.has_value() ? scales.value().data_ptr<float>() : nullptr,
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                block_accum.data_ptr<int32_t>(),
                camera_model,
                nullptr,
                indptr.data_ptr<int32_t>(),
                camera_ids.data_ptr<int64_t>(),
                gaussian_ids.data_ptr<int64_t>(),
                radii.data_ptr<int32_t>(),
                means2d.data_ptr<float>(),
                depths.data_ptr<float>(),
                conics.data_ptr<float>(),
                calc_compensations ? compensations.data_ptr<float>() : nullptr
            );
    } else {
        indptr.fill_(0);
    }

    return std::make_tuple(
        indptr,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        compensations
    );
}



/****************************************************************************
 * Projection of Gaussians (Batched) Backward Pass
 ****************************************************************************/


 __global__ void fully_fused_projection_packed_bwd_kernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const uint32_t nnz,
    const float *__restrict__ means,    // [N, 3]
    const float *__restrict__ covars,   // [N, 6] Optional
    const float *__restrict__ quats,    // [N, 4] Optional
    const float *__restrict__ scales,   // [N, 3] Optional
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ Ks,       // [C, 3, 3]
    const int32_t image_width,
    const int32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const int64_t *__restrict__ camera_ids,   // [nnz]
    const int64_t *__restrict__ gaussian_ids, // [nnz]
    const float *__restrict__ conics,             // [nnz, 3]
    const float *__restrict__ compensations,      // [nnz] optional
    // grad outputs
    const float *__restrict__ v_means2d,       // [nnz, 2]
    const float *__restrict__ v_depths,        // [nnz]
    const float *__restrict__ v_conics,        // [nnz, 3]
    const float *__restrict__ v_compensations, // [nnz] optional
    const bool sparse_grad, // whether the outputs are in COO format [nnz, ...]
    // grad inputs
    float *__restrict__ v_means,   // [N, 3] or [nnz, 3]
    float *__restrict__ v_covars,  // [N, 6] or [nnz, 6] Optional
    float *__restrict__ v_quats,   // [N, 4] or [nnz, 4] Optional
    float *__restrict__ v_scales,  // [N, 3] or [nnz, 3] Optional
    float *__restrict__ v_viewmats // [C, 4, 4] Optional
) {
    // parallelize over nnz.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= nnz) {
        return;
    }
    const int64_t cid = camera_ids[idx];   // camera id
    const int64_t gid = gaussian_ids[idx]; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    conics += idx * 3;

    v_means2d += idx * 2;
    v_depths += idx;
    v_conics += idx * 3;

    // vjp: compute the inverse of the 2d covariance
    mat2 covar2d_inv = mat2(conics[0], conics[1], conics[1], conics[2]);
    mat2 v_covar2d_inv =
        mat2(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    mat2 v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    if (v_compensations != nullptr) {
        // vjp: compensation term
        const float compensation = compensations[idx];
        const float v_compensation = v_compensations[idx];
        add_blur_vjp(
            eps2d, covar2d_inv, compensation, v_compensation, v_covar2d
        );
    }

    // transform Gaussian to camera space
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);
    mat3 covar;
    vec4 quat;
    vec3 scale;
    if (covars != nullptr) {
        // if a precomputed covariance is provided
        covars += gid * 6;
        covar = mat3(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
    } else {
        // if not then compute it from quaternions and scales
        quat = glm::make_vec4(quats + gid * 4);
        scale = glm::make_vec3(scales + gid * 3);
        quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
    }
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat3 v_covar_c(0.f);
    vec3 v_mean_c(0.f);
    switch (camera_model) {
        case CameraModelType::PINHOLE: // perspective projection
            persp_proj_vjp(
                mean_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                image_width,
                image_height,
                v_covar2d,
                glm::make_vec2(v_means2d),
                v_mean_c,
                v_covar_c
            );
            break;
        case CameraModelType::ORTHO: // orthographic projection
            ortho_proj_vjp(
                mean_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                image_width,
                image_height,
                v_covar2d,
                glm::make_vec2(v_means2d),
                v_mean_c,
                v_covar_c
            );
            break;
        case CameraModelType::FISHEYE: // fisheye projection
            fisheye_proj_vjp(
                mean_c,
                covar_c,
                fx,
                fy,
                cx,
                cy,
                image_width,
                image_height,
                v_covar2d,
                glm::make_vec2(v_means2d),
                v_mean_c,
                v_covar_c
            );
            break;
    }

    // add contribution from v_depths
    v_mean_c.z += v_depths[0];

    // vjp: transform Gaussian covariance to camera space
    vec3 v_mean(0.f);
    mat3 v_covar(0.f);
    mat3 v_R(0.f);
    vec3 v_t(0.f);
    posW2C_VJP(
        R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean
    );
    covarW2C_VJP(R, covar, v_covar_c, v_R, v_covar);

    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    if (sparse_grad) {
        // write out results with sparse layout
        if (v_means != nullptr) {
            v_means += idx * 3;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                v_means[i] = v_mean[i];
            }
        }
        if (v_covars != nullptr) {
            v_covars += idx * 6;
            v_covars[0] = v_covar[0][0];
            v_covars[1] = v_covar[0][1] + v_covar[1][0];
            v_covars[2] = v_covar[0][2] + v_covar[2][0];
            v_covars[3] = v_covar[1][1];
            v_covars[4] = v_covar[1][2] + v_covar[2][1];
            v_covars[5] = v_covar[2][2];
        } else {
            mat3 rotmat = quat_to_rotmat(quat);
            vec4 v_quat(0.f);
            vec3 v_scale(0.f);
            quat_scale_to_covar_vjp(
                quat, scale, rotmat, v_covar, v_quat, v_scale
            );
            v_quats += idx * 4;
            v_scales += idx * 3;
            v_quats[0] = v_quat[0];
            v_quats[1] = v_quat[1];
            v_quats[2] = v_quat[2];
            v_quats[3] = v_quat[3];
            v_scales[0] = v_scale[0];
            v_scales[1] = v_scale[1];
            v_scales[2] = v_scale[2];
        }
    } else {
        // write out results with dense layout
        // #if __CUDA_ARCH__ >= 700
        // write out results with warp-level reduction
        auto warp_group_g = cg::labeled_partition(warp, gid);
        if (v_means != nullptr) {
            warpSum(v_mean, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_means += gid * 3;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t i = 0; i < 3; i++) {
                    gpuAtomicAdd(v_means + i, v_mean[i]);
                }
            }
        }
        if (v_covars != nullptr) {
            // Directly output gradients w.r.t. the covariance
            warpSum(v_covar, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_covars += gid * 6;
                gpuAtomicAdd(v_covars, v_covar[0][0]);
                gpuAtomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
                gpuAtomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
                gpuAtomicAdd(v_covars + 3, v_covar[1][1]);
                gpuAtomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
                gpuAtomicAdd(v_covars + 5, v_covar[2][2]);
            }
        } else {
            // Directly output gradients w.r.t. the quaternion and scale
            mat3 rotmat = quat_to_rotmat(quat);
            vec4 v_quat(0.f);
            vec3 v_scale(0.f);
            quat_scale_to_covar_vjp(
                quat, scale, rotmat, v_covar, v_quat, v_scale
            );
            warpSum(v_quat, warp_group_g);
            warpSum(v_scale, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_quats += gid * 4;
                v_scales += gid * 3;
                gpuAtomicAdd(v_quats, v_quat[0]);
                gpuAtomicAdd(v_quats + 1, v_quat[1]);
                gpuAtomicAdd(v_quats + 2, v_quat[2]);
                gpuAtomicAdd(v_quats + 3, v_quat[3]);
                gpuAtomicAdd(v_scales, v_scale[0]);
                gpuAtomicAdd(v_scales + 1, v_scale[1]);
                gpuAtomicAdd(v_scales + 2, v_scale[2]);
            }
        }
    }
    // v_viewmats is always in dense layout
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
fully_fused_projection_packed_bwd_tensor(
    // fwd inputs
    const torch::Tensor &means,                // [N, 3]
    const at::optional<torch::Tensor> &covars, // [N, 6]
    const at::optional<torch::Tensor> &quats,  // [N, 4]
    const at::optional<torch::Tensor> &scales, // [N, 3]
    const torch::Tensor &viewmats,             // [C, 4, 4]
    const torch::Tensor &Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const torch::Tensor &camera_ids,                  // [nnz]
    const torch::Tensor &gaussian_ids,                // [nnz]
    const torch::Tensor &conics,                      // [nnz, 3]
    const at::optional<torch::Tensor> &compensations, // [nnz] optional
    // grad outputs
    const torch::Tensor &v_means2d,                     // [nnz, 2]
    const torch::Tensor &v_depths,                      // [nnz]
    const torch::Tensor &v_conics,                      // [nnz, 3]
    const at::optional<torch::Tensor> &v_compensations, // [nnz] optional
    const bool viewmats_requires_grad,
    const bool sparse_grad
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    if (covars.has_value()) {
        GSPLAT_CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        GSPLAT_CHECK_INPUT(quats.value());
        GSPLAT_CHECK_INPUT(scales.value());
    }
    GSPLAT_CHECK_INPUT(viewmats);
    GSPLAT_CHECK_INPUT(Ks);
    GSPLAT_CHECK_INPUT(camera_ids);
    GSPLAT_CHECK_INPUT(gaussian_ids);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(v_means2d);
    GSPLAT_CHECK_INPUT(v_depths);
    GSPLAT_CHECK_INPUT(v_conics);
    if (compensations.has_value()) {
        GSPLAT_CHECK_INPUT(compensations.value());
    }
    if (v_compensations.has_value()) {
        GSPLAT_CHECK_INPUT(v_compensations.value());
        assert(compensations.has_value());
    }

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras
    uint32_t nnz = camera_ids.size(0);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor v_means, v_covars, v_quats, v_scales, v_viewmats;
    if (sparse_grad) {
        v_means = torch::zeros({nnz, 3}, means.options());
        if (covars.has_value()) {
            v_covars = torch::zeros({nnz, 6}, covars.value().options());
        } else {
            v_quats = torch::zeros({nnz, 4}, quats.value().options());
            v_scales = torch::zeros({nnz, 3}, scales.value().options());
        }
        if (viewmats_requires_grad) {
            v_viewmats = torch::zeros({C, 4, 4}, viewmats.options());
        }
    } else {
        v_means = torch::zeros_like(means);
        if (covars.has_value()) {
            v_covars = torch::zeros_like(covars.value());
        } else {
            v_quats = torch::zeros_like(quats.value());
            v_scales = torch::zeros_like(scales.value());
        }
        if (viewmats_requires_grad) {
            v_viewmats = torch::zeros_like(viewmats);
        }
    }
    if (nnz) {
        fully_fused_projection_packed_bwd_kernel
            <<<(nnz + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
               GSPLAT_N_THREADS,
               0,
               stream>>>(
                C,
                N,
                nnz,
                means.data_ptr<float>(),
                covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                covars.has_value() ? nullptr : quats.value().data_ptr<float>(),
                covars.has_value() ? nullptr : scales.value().data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                image_width,
                image_height,
                eps2d,
                camera_model,
                camera_ids.data_ptr<int64_t>(),
                gaussian_ids.data_ptr<int64_t>(),
                conics.data_ptr<float>(),
                compensations.has_value()
                    ? compensations.value().data_ptr<float>()
                    : nullptr,
                v_means2d.data_ptr<float>(),
                v_depths.data_ptr<float>(),
                v_conics.data_ptr<float>(),
                v_compensations.has_value()
                    ? v_compensations.value().data_ptr<float>()
                    : nullptr,
                sparse_grad,
                v_means.data_ptr<float>(),
                covars.has_value() ? v_covars.data_ptr<float>() : nullptr,
                covars.has_value() ? nullptr : v_quats.data_ptr<float>(),
                covars.has_value() ? nullptr : v_scales.data_ptr<float>(),
                viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr
            );
    }
    return std::make_tuple(v_means, v_covars, v_quats, v_scales, v_viewmats);
}


} // namespace gsplat