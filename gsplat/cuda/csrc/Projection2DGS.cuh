#pragma once

#include "Common.h"
#include "Utils.cuh"

namespace gsplat {

inline __device__ float sum(vec3 a) { return a.x + a.y + a.z; }

inline __device__ void compute_ray_transforms_aabb_vjp(
    const float *ray_transforms,
    const float *v_means2d,
    const vec3 v_normals,
    const mat3 W,
    const mat3 P,
    const vec3 cam_pos,
    const vec3 mean_c,
    const vec4 quat,
    const vec2 scale,
    mat3 &_v_ray_transforms,
    vec4 &v_quat,
    vec2 &v_scale,
    vec3 &v_mean
) {
    if (v_means2d[0] != 0 || v_means2d[1] != 0) {
        const float distance = ray_transforms[6] * ray_transforms[6] +
                               ray_transforms[7] * ray_transforms[7] -
                               ray_transforms[8] * ray_transforms[8];
        const float f = 1 / (distance);
        const float dpx_dT00 = f * ray_transforms[6];
        const float dpx_dT01 = f * ray_transforms[7];
        const float dpx_dT02 = -f * ray_transforms[8];
        const float dpy_dT10 = f * ray_transforms[6];
        const float dpy_dT11 = f * ray_transforms[7];
        const float dpy_dT12 = -f * ray_transforms[8];
        const float dpx_dT30 =
            ray_transforms[0] *
            (f - 2 * f * f * ray_transforms[6] * ray_transforms[6]);
        const float dpx_dT31 =
            ray_transforms[1] *
            (f - 2 * f * f * ray_transforms[7] * ray_transforms[7]);
        const float dpx_dT32 =
            -ray_transforms[2] *
            (f + 2 * f * f * ray_transforms[8] * ray_transforms[8]);
        const float dpy_dT30 =
            ray_transforms[3] *
            (f - 2 * f * f * ray_transforms[6] * ray_transforms[6]);
        const float dpy_dT31 =
            ray_transforms[4] *
            (f - 2 * f * f * ray_transforms[7] * ray_transforms[7]);
        const float dpy_dT32 =
            -ray_transforms[5] *
            (f + 2 * f * f * ray_transforms[8] * ray_transforms[8]);

        _v_ray_transforms[0][0] += v_means2d[0] * dpx_dT00;
        _v_ray_transforms[0][1] += v_means2d[0] * dpx_dT01;
        _v_ray_transforms[0][2] += v_means2d[0] * dpx_dT02;
        _v_ray_transforms[1][0] += v_means2d[1] * dpy_dT10;
        _v_ray_transforms[1][1] += v_means2d[1] * dpy_dT11;
        _v_ray_transforms[1][2] += v_means2d[1] * dpy_dT12;
        _v_ray_transforms[2][0] +=
            v_means2d[0] * dpx_dT30 + v_means2d[1] * dpy_dT30;
        _v_ray_transforms[2][1] +=
            v_means2d[0] * dpx_dT31 + v_means2d[1] * dpy_dT31;
        _v_ray_transforms[2][2] +=
            v_means2d[0] * dpx_dT32 + v_means2d[1] * dpy_dT32;
    }

    mat3 R = quat_to_rotmat(quat);
    mat3 v_M = P * glm::transpose(_v_ray_transforms);
    mat3 W_t = glm::transpose(W);
    mat3 v_RS = W_t * v_M;
    vec3 v_tn = W_t * v_normals;

    // dual visible
    vec3 tn = W * R[2];
    float cos = glm::dot(-tn, mean_c);
    float multiplier = cos > 0 ? 1 : -1;
    v_tn *= multiplier;

    mat3 v_R = mat3(v_RS[0] * scale[0], v_RS[1] * scale[1], v_tn);

    quat_to_rotmat_vjp(quat, v_R, v_quat);
    v_scale[0] += glm::dot(v_RS[0], R[0]);
    v_scale[1] += glm::dot(v_RS[1], R[1]);

    v_mean += v_RS[2];
}

} // namespace gsplat