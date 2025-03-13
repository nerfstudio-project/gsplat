#ifndef GSPLAT_CUDA_TRANSFORM_CUH
#define GSPLAT_CUDA_TRANSFORM_CUH

#include "types.cuh"

namespace gsplat {


inline __device__ void pos_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const mat3 R,
    const vec3 t,
    const vec3 p,
    vec3 &p_c
) {
    p_c = R * p + t;
}


inline __device__ void pos_world_to_cam_vjp(
    // fwd inputs
    const mat3 R,
    const vec3 t,
    const vec3 p,
    // grad outputs
    const vec3 v_p_c,
    // grad inputs
    mat3 &v_R,
    vec3 &v_t,
    vec3 &v_p
) {
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    v_R += glm::outerProduct(v_p_c, p);
    v_t += v_p_c;
    v_p += glm::transpose(R) * v_p_c;
}


inline __device__ void covar_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const mat3 R,
    const mat3 covar,
    mat3 &covar_c
) {
    covar_c = R * covar * glm::transpose(R);
}


inline __device__ void covar_world_to_cam_vjp(
    // fwd inputs
    const mat3 R,
    const mat3 covar,
    // grad outputs
    const mat3 v_covar_c,
    // grad inputs
    mat3 &v_R,
    mat3 &v_covar
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

} // namespace gsplat

#endif // GSPLAT_CUDA_TRANSFORM_CUH
