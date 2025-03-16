#pragma once

#include "Common.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace gsplat {

namespace cg = cooperative_groups;

///////////////////////////////
// Coordinate Transformations
///////////////////////////////

// Transforms a 3D position from world coordinates to camera coordinates.
// [R | t] is the world-to-camera transformation.
inline __device__ void posW2C(
    const mat3 R,
    const vec3 t,
    const vec3 pW, // Input position in world coordinates
    vec3 &pC       // Output position in camera coordinates
) {
    pC = R * pW + t;
}

// Computes the vector-Jacobian product (VJP) for posW2C.
// This function computes gradients of the transformation with respect to
// inputs.
inline __device__ void posW2C_VJP(
    // Forward inputs
    const mat3 R,
    const vec3 t,
    const vec3 pW, // Input position in world coordinates
    // Gradient output
    const vec3 v_pC, // Gradient of the output position in camera coordinates
    // Gradient inputs (to be accumulated)
    mat3 &v_R, // Gradient w.r.t. R
    vec3 &v_t, // Gradient w.r.t. t
    vec3 &v_pW // Gradient w.r.t. pW
) {
    // Using the rule for differentiating a linear transformation:
    // For D = W * X, G = dL/dD
    // dL/dW = G * X^T, dL/dX = W^T * G
    v_R += glm::outerProduct(v_pC, pW);
    v_t += v_pC;
    v_pW += glm::transpose(R) * v_pC;
}

// Transforms a covariance matrix from world coordinates to camera coordinates.
inline __device__ void covarW2C(
    const mat3 R,
    const mat3 covarW, // Input covariance matrix in world coordinates
    mat3 &covarC       // Output covariance matrix in camera coordinates
) {
    covarC = R * covarW * glm::transpose(R);
}

// Computes the vector-Jacobian product (VJP) for covarW2C.
// This function computes gradients of the transformation with respect to
// inputs.
inline __device__ void covarW2C_VJP(
    // Forward inputs
    const mat3 R,
    const mat3 covarW, // Input covariance matrix in world coordinates
    // Gradient output
    const mat3 v_covarC, // Gradient of the output covariance matrix in camera
                         // coordinates
    // Gradient inputs (to be accumulated)
    mat3 &v_R,     // Gradient w.r.t. rotation matrix
    mat3 &v_covarW // Gradient w.r.t. world covariance matrix
) {
    // Using the rule for differentiating quadratic forms:
    // For D = W * X * W^T, G = dL/dD
    // dL/dX = W^T * G * W
    // dL/dW = G * W * X^T + G^T * W * X
    v_R += v_covarC * R * glm::transpose(covarW) +
           glm::transpose(v_covarC) * R * covarW;
    v_covarW += glm::transpose(R) * v_covarC * R;
}

///////////////////////////////
// Reduce
///////////////////////////////

template <uint32_t DIM, class WarpT>
inline __device__ void warpSum(float *val, WarpT &warp) {
#pragma unroll
    for (uint32_t i = 0; i < DIM; i++) {
        val[i] = cg::reduce(warp, val[i], cg::plus<float>());
    }
}

template <class WarpT> inline __device__ void warpSum(float &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(vec4 &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
    val.z = cg::reduce(warp, val.z, cg::plus<float>());
    val.w = cg::reduce(warp, val.w, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(vec3 &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
    val.z = cg::reduce(warp, val.z, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(vec2 &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
}

template <class WarpT> inline __device__ void warpSum(mat4 &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
    warpSum(val[3], warp);
}

template <class WarpT> inline __device__ void warpSum(mat3 &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
}

template <class WarpT> inline __device__ void warpSum(mat2 &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
}

template <class WarpT> inline __device__ void warpMax(float &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::greater<float>());
}

///////////////////////////////
// Quaternion
///////////////////////////////

inline __device__ mat3 quat_to_rotmat(const vec4 quat) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return mat3(
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

inline __device__ void
quat_to_rotmat_vjp(const vec4 quat, const mat3 v_R, vec4 &v_quat) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    vec4 v_quat_n = vec4(
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

    vec4 quat_n = vec4(w, x, y, z);
    v_quat += (v_quat_n - glm::dot(v_quat_n, quat_n) * quat_n) * inv_norm;
}

inline __device__ void quat_scale_to_covar_preci(
    const vec4 quat,
    const vec3 scale,
    // optional outputs
    mat3 *covar,
    mat3 *preci
) {
    mat3 R = quat_to_rotmat(quat);
    if (covar != nullptr) {
        // C = R * S * S * Rt
        mat3 S =
            mat3(scale[0], 0.f, 0.f, 0.f, scale[1], 0.f, 0.f, 0.f, scale[2]);
        mat3 M = R * S;
        *covar = M * glm::transpose(M);
    }
    if (preci != nullptr) {
        // P = R * S^-1 * S^-1 * Rt
        mat3 S = mat3(
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
        mat3 M = R * S;
        *preci = M * glm::transpose(M);
    }
}

inline __device__ void quat_scale_to_covar_vjp(
    // fwd inputs
    const vec4 quat,
    const vec3 scale,
    // precompute
    const mat3 R,
    // grad outputs
    const mat3 v_covar,
    // grad inputs
    vec4 &v_quat,
    vec3 &v_scale
) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    float sx = scale[0], sy = scale[1], sz = scale[2];

    // M = R * S
    mat3 S = mat3(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3 M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    mat3 v_M = (v_covar + glm::transpose(v_covar)) * M;
    mat3 v_R = v_M * S;

    // grad for (quat, scale) from covar
    quat_to_rotmat_vjp(quat, v_R, v_quat);

    v_scale[0] +=
        R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2];
    v_scale[1] +=
        R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2];
    v_scale[2] +=
        R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2];
}

inline __device__ void quat_scale_to_preci_vjp(
    // fwd inputs
    const vec4 quat,
    const vec3 scale,
    // precompute
    const mat3 R,
    // grad outputs
    const mat3 v_preci,
    // grad inputs
    vec4 &v_quat,
    vec3 &v_scale
) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    float sx = 1.0f / scale[0], sy = 1.0f / scale[1], sz = 1.0f / scale[2];

    // M = R * S
    mat3 S = mat3(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3 M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    mat3 v_M = (v_preci + glm::transpose(v_preci)) * M;
    mat3 v_R = v_M * S;

    // grad for (quat, scale) from preci
    quat_to_rotmat_vjp(quat, v_R, v_quat);

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

///////////////////////////////
// Misc
///////////////////////////////

inline __device__ void
inverse_vjp(const mat2 Minv, const mat2 v_Minv, mat2 &v_M) {
    // P = M^-1
    // df/dM = -P * df/dP * P
    v_M += -Minv * v_Minv * Minv;
}

inline __device__ float
add_blur(const float eps2d, mat2 &covar, float &compensation) {
    float det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    float det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}

inline __device__ void add_blur_vjp(
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

///////////////////////////////
// Projection Related
///////////////////////////////

inline __device__ void ortho_proj(
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

inline __device__ void ortho_proj_vjp(
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

inline __device__ void persp_proj(
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

inline __device__ void persp_proj_vjp(
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

inline __device__ void fisheye_proj(
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
    mean2d = vec2({x * fx * theta / xy_len + cx, y * fy * theta / xy_len + cy});

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

inline __device__ void fisheye_proj_vjp(
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

} // namespace gsplat