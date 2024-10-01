#ifndef GSPLAT_CUDA_UTILS_H
#define GSPLAT_CUDA_UTILS_H

#include "helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#define FILTER_INV_SQUARE 2.0f

namespace gsplat {

template <typename T>
__device__ __host__ inline int sign(T x) {
    return (x > 0) - (x < 0);
}

template <typename T>
inline __device__ mat3<T> quat_to_rotmat(const vec4<T> quat) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    T inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    T x2 = x * x, y2 = y * y, z2 = z * z;
    T xy = x * y, xz = x * z, yz = y * z;
    T wx = w * x, wy = w * y, wz = w * z;
    return mat3<T>(
        (1.f - 2.f * (y2 + z2)),
        (2.f * (xy + wz)),
        (2.f * (xz - wy)), // 1st col
        (2.f * (xy - wz)),
        (1.f - 2.f * (x2 + z2)),
        (2.f * (yz + wx)), // 2nd col
        (2.f * (xz + wy)),
        (2.f * (yz - wx)),
        (1.f - 2.f * (x2 + y2)) // 3rd col
    );
}

template <typename T>
inline __device__ void
quat_to_rotmat_vjp(const vec4<T> quat, const mat3<T> v_R, vec4<T> &v_quat) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    T inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    vec4<T> v_quat_n = vec4<T>(
        2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
               z * (v_R[0][1] - v_R[1][0])),
        2.f *
            (-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
             z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])),
        2.f * (x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
               z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])),
        2.f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
               2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0]))
    );

    vec4<T> quat_n = vec4<T>(w, x, y, z);
    v_quat += (v_quat_n - glm::dot(v_quat_n, quat_n) * quat_n) * inv_norm;
}

template <typename T>
inline __device__ void quat_scale_to_covar_preci(
    const vec4<T> quat,
    const vec3<T> scale,
    // optional outputs
    mat3<T> *covar,
    mat3<T> *preci
) {
    mat3<T> R = quat_to_rotmat<T>(quat);
    if (covar != nullptr) {
        // C = R * S * S * Rt
        mat3<T> S =
            mat3<T>(scale[0], 0.f, 0.f, 0.f, scale[1], 0.f, 0.f, 0.f, scale[2]);
        mat3<T> M = R * S;
        *covar = M * glm::transpose(M);
    }
    if (preci != nullptr) {
        // P = R * S^-1 * S^-1 * Rt
        mat3<T> S = mat3<T>(
            1.0f / scale[0],
            0.f,
            0.f,
            0.f,
            1.0f / scale[1],
            0.f,
            0.f,
            0.f,
            1.0f / scale[2]
        );
        mat3<T> M = R * S;
        *preci = M * glm::transpose(M);
    }
}

template <typename T>
inline __device__ void quat_scale_to_covar_vjp(
    // fwd inputs
    const vec4<T> quat,
    const vec3<T> scale,
    // precompute
    const mat3<T> R,
    // grad outputs
    const mat3<T> v_covar,
    // grad inputs
    vec4<T> &v_quat,
    vec3<T> &v_scale
) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    T sx = scale[0], sy = scale[1], sz = scale[2];

    // M = R * S
    mat3<T> S = mat3<T>(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3<T> M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    mat3<T> v_M = (v_covar + glm::transpose(v_covar)) * M;
    mat3<T> v_R = v_M * S;

    // grad for (quat, scale) from covar
    quat_to_rotmat_vjp<T>(quat, v_R, v_quat);

    v_scale[0] +=
        R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2];
    v_scale[1] +=
        R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2];
    v_scale[2] +=
        R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2];
}

template <typename T>
inline __device__ void quat_scale_to_preci_vjp(
    // fwd inputs
    const vec4<T> quat,
    const vec3<T> scale,
    // precompute
    const mat3<T> R,
    // grad outputs
    const mat3<T> v_preci,
    // grad inputs
    vec4<T> &v_quat,
    vec3<T> &v_scale
) {
    T w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    T sx = 1.0f / scale[0], sy = 1.0f / scale[1], sz = 1.0f / scale[2];

    // M = R * S
    mat3<T> S = mat3<T>(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3<T> M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    mat3<T> v_M = (v_preci + glm::transpose(v_preci)) * M;
    mat3<T> v_R = v_M * S;

    // grad for (quat, scale) from preci
    quat_to_rotmat_vjp<T>(quat, v_R, v_quat);

    v_scale[0] +=
        -sx * sx *
        (R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2]);
    v_scale[1] +=
        -sy * sy *
        (R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2]);
    v_scale[2] +=
        -sz * sz *
        (R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2]);
}

template <typename T>
inline __device__ void ortho_proj(
    // inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2<T> &cov2d,
    vec2<T> &mean2d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx,
        0.f, // 1st column
        0.f,
        fy, // 2nd column
        0.f,
        0.f // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2<T>({fx * x + cx, fy * y + cy});
}

template <typename T>
inline __device__ void ortho_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
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
    v_mean3d += vec3<T>(fx * v_mean2d[0], fy * v_mean2d[1], 0.f);
}

template <typename T>
inline __device__ void persp_proj(
    // inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2<T> &cov2d,
    vec2<T> &mean2d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    T lim_x_neg = cx / fx + 0.3f * tan_fovx;
    T lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    T lim_y_neg = cy / fy + 0.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    T ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * rz,
        0.f, // 1st column
        0.f,
        fy * rz, // 2nd column
        -fx * tx * rz2,
        -fy * ty * rz2 // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = vec2<T>({fx * x * rz + cx, fy * y * rz + cy});
}

template <typename T>
inline __device__ void persp_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T tan_fovx = 0.5f * width / fx;
    T tan_fovy = 0.5f * height / fy;
    T lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    T lim_x_neg = cx / fx + 0.3f * tan_fovx;
    T lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    T lim_y_neg = cy / fy + 0.3f * tan_fovy;

    T rz = 1.f / z;
    T rz2 = rz * rz;
    T tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    T ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
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
    v_mean3d += vec3<T>(
        fx * rz * v_mean2d[0],
        fy * rz * v_mean2d[1],
        -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2
    );

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    T rz3 = rz2 * rz;
    mat3x2<T> v_J = v_cov2d * J * glm::transpose(cov3d) +
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

template <typename T>
inline __device__ void fisheye_proj(
    // inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // outputs
    mat2<T> &cov2d,
    vec2<T> &mean2d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    T eps = 0.0000001f;
    T xy_len = glm::length(glm::vec2({x, y})) + eps;
    T theta = glm::atan(xy_len, z + eps);
    mean2d =
        vec2<T>({x * fx * theta / xy_len + cx, y * fy * theta / xy_len + cy});

    T x2 = x * x + eps;
    T y2 = y * y;
    T xy = x * y;
    T x2y2 = x2 + y2;
    T x2y2z2_inv = 1.f / (x2y2 + z * z);

    T b = glm::atan(xy_len, z) / xy_len / x2y2;
    T a = z * x2y2z2_inv / (x2y2);
    mat3x2<T> J = mat3x2<T>(
        fx * (x2 * a + y2 * b),
        fy * xy * (a - b),
        fx * xy * (a - b),
        fy * (y2 * a + x2 * b),
        -fx * x * x2y2z2_inv,
        -fy * y * x2y2z2_inv
    );
    cov2d = J * cov3d * glm::transpose(J);
}

template <typename T>
inline __device__ void fisheye_proj_vjp(
    // fwd inputs
    const vec3<T> mean3d,
    const mat3<T> cov3d,
    const T fx,
    const T fy,
    const T cx,
    const T cy,
    const uint32_t width,
    const uint32_t height,
    // grad outputs
    const mat2<T> v_cov2d,
    const vec2<T> v_mean2d,
    // grad inputs
    vec3<T> &v_mean3d,
    mat3<T> &v_cov3d
) {
    T x = mean3d[0], y = mean3d[1], z = mean3d[2];

    const T eps = 0.0000001f;
    T x2 = x * x + eps;
    T y2 = y * y;
    T xy = x * y;
    T x2y2 = x2 + y2;
    T len_xy = length(glm::vec2({x, y})) + eps;
    const T x2y2z2 = x2y2 + z * z;
    T x2y2z2_inv = 1.f / x2y2z2;
    T b = glm::atan(len_xy, z) / len_xy / x2y2;
    T a = z * x2y2z2_inv / (x2y2);
    v_mean3d += vec3<T>(
        fx * (x2 * a + y2 * b) * v_mean2d[0] + fy * xy * (a - b) * v_mean2d[1],
        fx * xy * (a - b) * v_mean2d[0] + fy * (y2 * a + x2 * b) * v_mean2d[1],
        -fx * x * x2y2z2_inv * v_mean2d[0] - fy * y * x2y2z2_inv * v_mean2d[1]
    );

    const T theta = glm::atan(len_xy, z);
    const T J_b = theta / len_xy / x2y2;
    const T J_a = z * x2y2z2_inv / (x2y2);
    // mat3x2 is 3 columns x 2 rows.
    mat3x2<T> J = mat3x2<T>(
        fx * (x2 * J_a + y2 * J_b),
        fy * xy * (J_a - J_b), // 1st column
        fx * xy * (J_a - J_b),
        fy * (y2 * J_a + x2 * J_b), // 2nd column
        -fx * x * x2y2z2_inv,
        -fy * y * x2y2z2_inv // 3rd column
    );
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    mat3x2<T> v_J = v_cov2d * J * glm::transpose(cov3d) +
                    glm::transpose(v_cov2d) * J * cov3d;
    T l4 = x2y2z2 * x2y2z2;

    T E = -l4 * x2y2 * theta + x2y2z2 * x2y2 * len_xy * z;
    T F = 3 * l4 * theta - 3 * x2y2z2 * len_xy * z - 2 * x2y2 * len_xy * z;

    T A = x * (3 * E + x2 * F);
    T B = y * (E + x2 * F);
    T C = x * (E + y2 * F);
    T D = y * (3 * E + y2 * F);

    T S1 = x2 - y2 - z * z;
    T S2 = y2 - x2 - z * z;
    T inv1 = x2y2z2_inv * x2y2z2_inv;
    T inv2 = inv1 / (x2y2 * x2y2 * len_xy);

    T dJ_dx00 = fx * A * inv2;
    T dJ_dx01 = fx * B * inv2;
    T dJ_dx02 = fx * S1 * inv1;
    T dJ_dx10 = fy * B * inv2;
    T dJ_dx11 = fy * C * inv2;
    T dJ_dx12 = 2.f * fy * xy * inv1;

    T dJ_dy00 = dJ_dx01;
    T dJ_dy01 = fx * C * inv2;
    T dJ_dy02 = 2.f * fx * xy * inv1;
    T dJ_dy10 = dJ_dx11;
    T dJ_dy11 = fy * D * inv2;
    T dJ_dy12 = fy * S2 * inv1;

    T dJ_dz00 = dJ_dx02;
    T dJ_dz01 = dJ_dy02;
    T dJ_dz02 = 2.f * fx * x * z * inv1;
    T dJ_dz10 = dJ_dx12;
    T dJ_dz11 = dJ_dy12;
    T dJ_dz12 = 2.f * fy * y * z * inv1;

    T dL_dtx_raw = dJ_dx00 * v_J[0][0] + dJ_dx01 * v_J[1][0] +
                   dJ_dx02 * v_J[2][0] + dJ_dx10 * v_J[0][1] +
                   dJ_dx11 * v_J[1][1] + dJ_dx12 * v_J[2][1];
    T dL_dty_raw = dJ_dy00 * v_J[0][0] + dJ_dy01 * v_J[1][0] +
                   dJ_dy02 * v_J[2][0] + dJ_dy10 * v_J[0][1] +
                   dJ_dy11 * v_J[1][1] + dJ_dy12 * v_J[2][1];
    T dL_dtz_raw = dJ_dz00 * v_J[0][0] + dJ_dz01 * v_J[1][0] +
                   dJ_dz02 * v_J[2][0] + dJ_dz10 * v_J[0][1] +
                   dJ_dz11 * v_J[1][1] + dJ_dz12 * v_J[2][1];
    v_mean3d.x += dL_dtx_raw;
    v_mean3d.y += dL_dty_raw;
    v_mean3d.z += dL_dtz_raw;
}

template <typename T>
inline __device__ void pos_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const mat3<T> R,
    const vec3<T> t,
    const vec3<T> p,
    vec3<T> &p_c
) {
    p_c = R * p + t;
}

template <typename T>
inline __device__ void pos_world_to_cam_vjp(
    // fwd inputs
    const mat3<T> R,
    const vec3<T> t,
    const vec3<T> p,
    // grad outputs
    const vec3<T> v_p_c,
    // grad inputs
    mat3<T> &v_R,
    vec3<T> &v_t,
    vec3<T> &v_p
) {
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    v_R += glm::outerProduct(v_p_c, p);
    v_t += v_p_c;
    v_p += glm::transpose(R) * v_p_c;
}

template <typename T>
inline __device__ void covar_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const mat3<T> R,
    const mat3<T> covar,
    mat3<T> &covar_c
) {
    covar_c = R * covar * glm::transpose(R);
}

template <typename T>
inline __device__ void covar_world_to_cam_vjp(
    // fwd inputs
    const mat3<T> R,
    const mat3<T> covar,
    // grad outputs
    const mat3<T> v_covar_c,
    // grad inputs
    mat3<T> &v_R,
    mat3<T> &v_covar
) {
    // for D = W * X * WT, G = df/dD
    // df/dX = WT * G * W
    // df/dW
    // = G * (X * WT)T + ((W * X)T * G)T
    // = G * W * XT + (XT * WT * G)T
    // = G * W * XT + GT * W * X
    v_R += v_covar_c * R * glm::transpose(covar) +
           glm::transpose(v_covar_c) * R * covar;
    v_covar += glm::transpose(R) * v_covar_c * R;
}

template <typename T>
inline __device__ T inverse(const mat2<T> M, mat2<T> &Minv) {
    T det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (det <= 0.f) {
        return det;
    }
    T invDet = 1.f / det;
    Minv[0][0] = M[1][1] * invDet;
    Minv[0][1] = -M[0][1] * invDet;
    Minv[1][0] = Minv[0][1];
    Minv[1][1] = M[0][0] * invDet;
    return det;
}

template <typename T>
inline __device__ void inverse_vjp(const T Minv, const T v_Minv, T &v_M) {
    // P = M^-1
    // df/dM = -P * df/dP * P
    v_M += -Minv * v_Minv * Minv;
}

template <typename T>
inline __device__ T add_blur(const T eps2d, mat2<T> &covar, T &compensation) {
    T det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    T det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}

template <typename T>
inline __device__ void add_blur_vjp(
    const T eps2d,
    const mat2<T> conic_blur,
    const T compensation,
    const T v_compensation,
    mat2<T> &v_covar
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

    T det_conic_blur = conic_blur[0][0] * conic_blur[1][1] -
                       conic_blur[0][1] * conic_blur[1][0];
    T v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
    T one_minus_sqr_comp = 1 - compensation * compensation;
    v_covar[0][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][0] -
                                   eps2d * det_conic_blur);
    v_covar[0][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][1]);
    v_covar[1][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][0]);
    v_covar[1][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][1] -
                                   eps2d * det_conic_blur);
}

template <typename T>
inline __device__ void compute_ray_transforms_aabb_vjp(
    const T *ray_transforms,
    const T *v_means2d,
    const vec3<T> v_normals,
    const mat3<T> W,
    const mat3<T> P,
    const vec3<T> cam_pos,
    const vec3<T> mean_c,
    const vec4<T> quat,
    const vec2<T> scale,
    mat3<T> &_v_ray_transforms,
    vec4<T> &v_quat,
    vec2<T> &v_scale,
    vec3<T> &v_mean
) {
    if (v_means2d[0] != 0 || v_means2d[1] != 0) {
        const T distance = ray_transforms[6] * ray_transforms[6] + ray_transforms[7] * ray_transforms[7] -
                           ray_transforms[8] * ray_transforms[8];
        const T f = 1 / (distance);
        const T dpx_dT00 = f * ray_transforms[6];
        const T dpx_dT01 = f * ray_transforms[7];
        const T dpx_dT02 = -f * ray_transforms[8];
        const T dpy_dT10 = f * ray_transforms[6];
        const T dpy_dT11 = f * ray_transforms[7];
        const T dpy_dT12 = -f * ray_transforms[8];
        const T dpx_dT30 = ray_transforms[0] * (f - 2 * f * f * ray_transforms[6] * ray_transforms[6]);
        const T dpx_dT31 = ray_transforms[1] * (f - 2 * f * f * ray_transforms[7] * ray_transforms[7]);
        const T dpx_dT32 = -ray_transforms[2] * (f + 2 * f * f * ray_transforms[8] * ray_transforms[8]);
        const T dpy_dT30 = ray_transforms[3] * (f - 2 * f * f * ray_transforms[6] * ray_transforms[6]);
        const T dpy_dT31 = ray_transforms[4] * (f - 2 * f * f * ray_transforms[7] * ray_transforms[7]);
        const T dpy_dT32 = -ray_transforms[5] * (f + 2 * f * f * ray_transforms[8] * ray_transforms[8]);

        _v_ray_transforms[0][0] += v_means2d[0] * dpx_dT00;
        _v_ray_transforms[0][1] += v_means2d[0] * dpx_dT01;
        _v_ray_transforms[0][2] += v_means2d[0] * dpx_dT02;
        _v_ray_transforms[1][0] += v_means2d[1] * dpy_dT10;
        _v_ray_transforms[1][1] += v_means2d[1] * dpy_dT11;
        _v_ray_transforms[1][2] += v_means2d[1] * dpy_dT12;
        _v_ray_transforms[2][0] += v_means2d[0] * dpx_dT30 + v_means2d[1] * dpy_dT30;
        _v_ray_transforms[2][1] += v_means2d[0] * dpx_dT31 + v_means2d[1] * dpy_dT31;
        _v_ray_transforms[2][2] += v_means2d[0] * dpx_dT32 + v_means2d[1] * dpy_dT32;
    }

    mat3<T> R = quat_to_rotmat(quat);
    mat3<T> v_M = P * glm::transpose(_v_ray_transforms);
    mat3<T> W_t = glm::transpose(W);
    mat3<T> v_RS = W_t * v_M;
    vec3<T> v_tn = W_t * v_normals;

    // dual visible
    vec3<T> tn = W * R[2];
    T cos = glm::dot(-tn, mean_c);
    T multiplier = cos > 0 ? 1 : -1;
    v_tn *= multiplier;

    mat3<T> v_R = mat3<T>(v_RS[0] * scale[0], v_RS[1] * scale[1], v_tn);

    quat_to_rotmat_vjp<T>(quat, v_R, v_quat);
    v_scale[0] += (T)glm::dot(v_RS[0], R[0]);
    v_scale[1] += (T)glm::dot(v_RS[1], R[1]);

    v_mean += v_RS[2];
}

// Function to check ray-triangle intersection using Möller–Trumbore algorithm
template <typename T>
inline __device__ bool ray_triangle_intersection(
    // ray origin and direction
    const vec3<T> o,
    const vec3<T> d,
    // triangle vertices
    const vec3<T> v0,
    const vec3<T> v1,
    const vec3<T> v2,
    // output intersection t value
    T &t
) {
    const T EPSILON = 0.00000001f;
    t = 0.0f;
    // e1 = v1 - v0
    // e2 = v2 - v0
    vec3<T> e1 = v1 - v0;
    vec3<T> e2 = v2 - v0;

    // h = cross(d, e2)
    // a = dot(e1, h)
    vec3<T> h = glm::cross(d, e2);
    T a = glm::dot(e1, h);
    if (a > -EPSILON && a < EPSILON) return false; // parallel to the triangle

    // f = 1 / a
    // s = o - v0
    // u = f * dot(s, h)
    T f = 1.0f / a;
    vec3<T> s = o - v0;
    T dot_s_h = glm::dot(s, h);
    T u = f * dot_s_h;
    if (u < 0.0f || u > 1.0f) return false; // outside the triangle

    // q = cross(s, e1)
    // v = f * dot(d, q)
    vec3<T> q = glm::cross(s, e1);
    T dot_d_q = glm::dot(d, q);
    T v = f * dot_d_q;
    if (v < 0.0f || u + v > 1.0f) return false; // outside the triangle

    // t = f * dot(e2, q)
    T dot_e2_q = glm::dot(e2, q);
    t = f * dot_e2_q;
    if (t > EPSILON) return true;
    return false; // There is a line intersection, but not a ray intersection
}

template <typename T>
inline __device__ void ray_triangle_intersection_vjp(
    // fwd inputs
    const vec3<T> o,
    const vec3<T> d,
    const vec3<T> v0,
    const vec3<T> v1,
    const vec3<T> v2,
    // grad outputs
    const T &v_t,
    // grad inputs (only backpropagate to triangle vertices)
    vec3<T> &v_v0,
    vec3<T> &v_v1,
    vec3<T> &v_v2
) {
    v_v0 = vec3<T>(0.f);
    v_v1 = vec3<T>(0.f);
    v_v2 = vec3<T>(0.f);

    // we call this function only when there is a ray-triangle intersection
    // so the forward is only used to compute intermediate variables
    vec3<T> e1 = v1 - v0;
    vec3<T> e2 = v2 - v0;
    vec3<T> h = glm::cross(d, e2);
    T a = glm::dot(e1, h);
    T f = 1.0f / a;
    vec3<T> s = o - v0;
    T dot_s_h = glm::dot(s, h);
    T u = f * dot_s_h;
    vec3<T> q = glm::cross(s, e1);
    T dot_d_q = glm::dot(d, q);
    T v = f * dot_d_q;
    T dot_e2_q = glm::dot(e2, q);
    T t = f * dot_e2_q;

    vec3<T> v_e1, v_e2, v_h, v_s, v_q;
    T v_a, v_f;

    // t = f * dot(e2, q), 
    v_f = v_t * dot_e2_q;
    v_e2 = v_t * q * f;
    v_q = v_t * e2 * f;
    
    // v = f * dot(d, q), v is leaf variable
    // q = cross(s, e1)
    v_s = glm::cross(e1, v_q);
    v_e1 = glm::cross(v_q, s);

    // u = f * dot(s, h), u is leaf variable
    // s = o - v0
    // v_o += v_s;
    v_v0 -= v_s;

    // f = 1 / a
    v_a = -v_f / (a * a);

    // a = dot(e1, h)
    v_e1 += v_a * h;
    v_h = v_a * e1;

    // h = cross(d, e2)
    // v_d += glm::cross(e2, v_h);
    v_e2 += glm::cross(v_h, d);

    // e2 = v2 - v0
    v_v2 += v_e2;
    v_v0 -= v_e2;
    
    // e1 = v1 - v0
    v_v1 += v_e1;
    v_v0 -= v_e1;
}

template <typename T>
inline __device__ bool ray_tetra_intersection(
    // ray origin and direction
    const vec3<T> o,
    const vec3<T> d,
    // tetrahedron vertices
    const vec3<T> v0,
    const vec3<T> v1,
    const vec3<T> v2,
    const vec3<T> v3,
    // output intersection face indices and t values
    int32_t &entry_face_idx, // entry face index
    int32_t &exit_face_idx, // exit face index
    T &t_entry, // entry face t value
    T &t_exit  // exit face t value
) {
    entry_face_idx = -1;
    exit_face_idx = -1;
    t_entry = 1e10f;
    t_exit = -1e10f;

    // Test intersection with each of the four faces
    T t_isct;
    vec3<T> faces[4][3] = {
        {v0, v1, v2}, // Face 0
        {v1, v2, v3}, // Face 1
        {v2, v3, v0}, // Face 2
        {v3, v0, v1}  // Face 3
    };

    GSPLAT_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
        bool intersected_face = ray_triangle_intersection(
            o,
            d,
            faces[i][0],
            faces[i][1],
            faces[i][2],
            t_isct
        );
        if (intersected_face) {
            if (t_isct < t_entry) {
                entry_face_idx = i;
                t_entry = t_isct;
            }
            if (t_isct > t_exit) {
                exit_face_idx = i;
                t_exit = t_isct;
            }
        }
    }
    if (entry_face_idx < 0 && exit_face_idx < 0) return false;
    return true;
}

template <typename T>
inline __device__ void ray_tetra_intersection_vjp(
    // fwd inputs
    const vec3<T> o,
    const vec3<T> d,
    const vec3<T> v0,
    const vec3<T> v1,
    const vec3<T> v2,
    const vec3<T> v3,
    // grad outputs
    const T &v_t_entry,
    const T &v_t_exit,
    // grad inputs (only backpropagate to tetrahedron vertices)
    vec3<T> &v_v0,
    vec3<T> &v_v1,
    vec3<T> &v_v2,
    vec3<T> &v_v3
) {
    if (v_t_entry == 0 && v_t_exit == 0) return;

    v_v0 = vec3<T>(0.f);
    v_v1 = vec3<T>(0.f);
    v_v2 = vec3<T>(0.f);
    v_v3 = vec3<T>(0.f);

    // run forward pass to get intersection face indices
    int32_t entry_face_idx = -1;
    int32_t exit_face_idx = -1;
    T t_entry = 1e10f;
    T t_exit = -1e10f;
    bool hit = ray_tetra_intersection(
        o,
        d,
        v0,
        v1,
        v2,
        v3,
        entry_face_idx,
        exit_face_idx,
        t_entry,
        t_exit
    );
    if (!hit) return;

    vec3<T> faces[4][3] = {
        {v0, v1, v2}, // Face 0
        {v1, v2, v3}, // Face 1
        {v2, v3, v0}, // Face 2
        {v3, v0, v1}  // Face 3
    };

    // backpropagate to the tetrahedron vertices

    vec3<T> v_entry_faces[3] = {vec3<T>(0.f), vec3<T>(0.f), vec3<T>(0.f)};
    ray_triangle_intersection_vjp(
        o,
        d,
        faces[entry_face_idx][0],
        faces[entry_face_idx][1],
        faces[entry_face_idx][2],
        v_t_entry,
        v_entry_faces[0],
        v_entry_faces[1],
        v_entry_faces[2]
    );
    switch (entry_face_idx) {
        case 0:
            v_v0 += v_entry_faces[0];
            v_v1 += v_entry_faces[1];
            v_v2 += v_entry_faces[2];
            break;
        case 1:
            v_v1 += v_entry_faces[0];
            v_v2 += v_entry_faces[1];
            v_v3 += v_entry_faces[2];
            break;
        case 2:
            v_v2 += v_entry_faces[0];
            v_v3 += v_entry_faces[1];
            v_v0 += v_entry_faces[2];
            break;
        case 3:
            v_v3 += v_entry_faces[0];
            v_v0 += v_entry_faces[1];
            v_v1 += v_entry_faces[2];
            break;
    }

    vec3<T> v_exit_faces[3] = {vec3<T>(0.f), vec3<T>(0.f), vec3<T>(0.f)};
    ray_triangle_intersection_vjp(
        o,
        d,
        faces[exit_face_idx][0],
        faces[exit_face_idx][1],
        faces[exit_face_idx][2],
        v_t_exit,
        v_exit_faces[0],
        v_exit_faces[1],
        v_exit_faces[2]
    );
    switch (exit_face_idx) {
        case 0:
            v_v0 += v_exit_faces[0];
            v_v1 += v_exit_faces[1];
            v_v2 += v_exit_faces[2];
            break;
        case 1:
            v_v1 += v_exit_faces[0];
            v_v2 += v_exit_faces[1];
            v_v3 += v_exit_faces[2];
            break;
        case 2:
            v_v2 += v_exit_faces[0];
            v_v3 += v_exit_faces[1];
            v_v0 += v_exit_faces[2];
            break;
        case 3:
            v_v3 += v_exit_faces[0];
            v_v0 += v_exit_faces[1];
            v_v1 += v_exit_faces[2];
            break;
    }
}

} // namespace gsplat

#endif // GSPLAT_CUDA_UTILS_H
