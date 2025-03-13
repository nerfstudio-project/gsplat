#include "utils.cuh"
#include <cmath>

namespace gsplat {

__device__ float inverse(const mat2 M, mat2 &Minv) {
    float det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (det <= 0.f) return det;
    float invDet = 1.f / det;
    Minv[0][0] = M[1][1] * invDet;
    Minv[0][1] = -M[0][1] * invDet;
    Minv[1][0] = Minv[0][1];
    Minv[1][1] = M[0][0] * invDet;
    return det;
}

__device__ void inverse_vjp(const float Minv, const float v_Minv, float &v_M) {
    v_M += -Minv * v_Minv * Minv;
}

__device__ float add_blur(const float eps2d, mat2 &covar, float &compensation) {
    float det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    float det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(fmaxf(0.f, det_orig / det_blur));
    return det_blur;
}

__device__ void add_blur_vjp(
    const float eps2d,
    const mat2 conic_blur,
    const float compensation,
    const float v_compensation,
    mat2 &v_covar
) {
    float det_conic_blur = conic_blur[0][0] * conic_blur[1][1] -
                       conic_blur[0][1] * conic_blur[1][0];
    float v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
    float one_minus_sqr_comp = 1 - compensation * compensation;
    v_covar[0][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][0] - eps2d * det_conic_blur);
    v_covar[0][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][1]);
    v_covar[1][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][0]);
    v_covar[1][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][1] - eps2d * det_conic_blur);
}

} // namespace gsplat
