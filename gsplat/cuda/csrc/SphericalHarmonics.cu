#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "SphericalHarmonics.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation

template <typename scalar_t>
__device__ void sh_coeffs_to_color_fast(
    const uint32_t degree,  // degree of SH to be evaluated
    const uint32_t c,       // color channel
    const vec3 &dir,        // [3]
    const scalar_t *coeffs, // [K, 3]
    // output
    scalar_t *colors // [3]
) {
    float result = 0.2820947917738781f * coeffs[c];
    if (degree >= 1) {
        // Normally rsqrt is faster than sqrt, but --use_fast_math will optimize
        // sqrt on single precision, so we use sqrt here.
        float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        float x = dir.x * inorm;
        float y = dir.y * inorm;
        float z = dir.z * inorm;

        result +=
            0.48860251190292f * (-y * coeffs[1 * 3 + c] +
                                 z * coeffs[2 * 3 + c] - x * coeffs[3 * 3 + c]);
        if (degree >= 2) {
            float z2 = z * z;

            float fTmp0B = -1.092548430592079f * z;
            float fC1 = x * x - y * y;
            float fS1 = 2.f * x * y;
            float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
            float pSH7 = fTmp0B * x;
            float pSH5 = fTmp0B * y;
            float pSH8 = 0.5462742152960395f * fC1;
            float pSH4 = 0.5462742152960395f * fS1;

            result += pSH4 * coeffs[4 * 3 + c] + pSH5 * coeffs[5 * 3 + c] +
                      pSH6 * coeffs[6 * 3 + c] + pSH7 * coeffs[7 * 3 + c] +
                      pSH8 * coeffs[8 * 3 + c];
            if (degree >= 3) {
                float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                float fTmp1B = 1.445305721320277f * z;
                float fC2 = x * fC1 - y * fS1;
                float fS2 = x * fS1 + y * fC1;
                float pSH12 =
                    z * (1.865881662950577f * z2 - 1.119528997770346f);
                float pSH13 = fTmp0C * x;
                float pSH11 = fTmp0C * y;
                float pSH14 = fTmp1B * fC1;
                float pSH10 = fTmp1B * fS1;
                float pSH15 = -0.5900435899266435f * fC2;
                float pSH9 = -0.5900435899266435f * fS2;

                result +=
                    pSH9 * coeffs[9 * 3 + c] + pSH10 * coeffs[10 * 3 + c] +
                    pSH11 * coeffs[11 * 3 + c] + pSH12 * coeffs[12 * 3 + c] +
                    pSH13 * coeffs[13 * 3 + c] + pSH14 * coeffs[14 * 3 + c] +
                    pSH15 * coeffs[15 * 3 + c];

                if (degree >= 4) {
                    float fTmp0D =
                        z * (-4.683325804901025f * z2 + 2.007139630671868f);
                    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                    float fTmp2B = -1.770130769779931f * z;
                    float fC3 = x * fC2 - y * fS2;
                    float fS3 = x * fS2 + y * fC2;
                    float pSH20 =
                        (1.984313483298443f * z * pSH12 -
                         1.006230589874905f * pSH6);
                    float pSH21 = fTmp0D * x;
                    float pSH19 = fTmp0D * y;
                    float pSH22 = fTmp1C * fC1;
                    float pSH18 = fTmp1C * fS1;
                    float pSH23 = fTmp2B * fC2;
                    float pSH17 = fTmp2B * fS2;
                    float pSH24 = 0.6258357354491763f * fC3;
                    float pSH16 = 0.6258357354491763f * fS3;

                    result += pSH16 * coeffs[16 * 3 + c] +
                              pSH17 * coeffs[17 * 3 + c] +
                              pSH18 * coeffs[18 * 3 + c] +
                              pSH19 * coeffs[19 * 3 + c] +
                              pSH20 * coeffs[20 * 3 + c] +
                              pSH21 * coeffs[21 * 3 + c] +
                              pSH22 * coeffs[22 * 3 + c] +
                              pSH23 * coeffs[23 * 3 + c] +
                              pSH24 * coeffs[24 * 3 + c];
                }
            }
        }
    }

    colors[c] = result;
}

template <typename scalar_t>
__device__ void sh_coeffs_to_color_fast_vjp(
    const uint32_t degree,    // degree of SH to be evaluated
    const uint32_t c,         // color channel
    const vec3 &dir,          // [3]
    const scalar_t *coeffs,   // [K, 3]
    const scalar_t *v_colors, // [3]
    // output
    scalar_t *v_coeffs, // [K, 3]
    vec3 *v_dir         // [3] optional
) {
    float v_colors_local = v_colors[c];

    v_coeffs[c] = 0.2820947917738781f * v_colors_local;
    if (degree < 1) {
        return;
    }
    float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    float x = dir.x * inorm;
    float y = dir.y * inorm;
    float z = dir.z * inorm;
    float v_x = 0.f, v_y = 0.f, v_z = 0.f;

    v_coeffs[1 * 3 + c] = -0.48860251190292f * y * v_colors_local;
    v_coeffs[2 * 3 + c] = 0.48860251190292f * z * v_colors_local;
    v_coeffs[3 * 3 + c] = -0.48860251190292f * x * v_colors_local;

    if (v_dir != nullptr) {
        v_x += -0.48860251190292f * coeffs[3 * 3 + c] * v_colors_local;
        v_y += -0.48860251190292f * coeffs[1 * 3 + c] * v_colors_local;
        v_z += 0.48860251190292f * coeffs[2 * 3 + c] * v_colors_local;
    }
    if (degree < 2) {
        if (v_dir != nullptr) {
            vec3 dir_n = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float z2 = z * z;
    float fTmp0B = -1.092548430592079f * z;
    float fC1 = x * x - y * y;
    float fS1 = 2.f * x * y;
    float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    float pSH7 = fTmp0B * x;
    float pSH5 = fTmp0B * y;
    float pSH8 = 0.5462742152960395f * fC1;
    float pSH4 = 0.5462742152960395f * fS1;
    v_coeffs[4 * 3 + c] = pSH4 * v_colors_local;
    v_coeffs[5 * 3 + c] = pSH5 * v_colors_local;
    v_coeffs[6 * 3 + c] = pSH6 * v_colors_local;
    v_coeffs[7 * 3 + c] = pSH7 * v_colors_local;
    v_coeffs[8 * 3 + c] = pSH8 * v_colors_local;

    float fTmp0B_z, fC1_x, fC1_y, fS1_x, fS1_y, pSH6_z, pSH7_x, pSH7_z, pSH5_y,
        pSH5_z, pSH8_x, pSH8_y, pSH4_x, pSH4_y;
    if (v_dir != nullptr) {
        fTmp0B_z = -1.092548430592079f;
        fC1_x = 2.f * x;
        fC1_y = -2.f * y;
        fS1_x = 2.f * y;
        fS1_y = 2.f * x;
        pSH6_z = 2.f * 0.9461746957575601f * z;
        pSH7_x = fTmp0B;
        pSH7_z = fTmp0B_z * x;
        pSH5_y = fTmp0B;
        pSH5_z = fTmp0B_z * y;
        pSH8_x = 0.5462742152960395f * fC1_x;
        pSH8_y = 0.5462742152960395f * fC1_y;
        pSH4_x = 0.5462742152960395f * fS1_x;
        pSH4_y = 0.5462742152960395f * fS1_y;

        v_x += v_colors_local *
               (pSH4_x * coeffs[4 * 3 + c] + pSH8_x * coeffs[8 * 3 + c] +
                pSH7_x * coeffs[7 * 3 + c]);
        v_y += v_colors_local *
               (pSH4_y * coeffs[4 * 3 + c] + pSH8_y * coeffs[8 * 3 + c] +
                pSH5_y * coeffs[5 * 3 + c]);
        v_z += v_colors_local *
               (pSH6_z * coeffs[6 * 3 + c] + pSH7_z * coeffs[7 * 3 + c] +
                pSH5_z * coeffs[5 * 3 + c]);
    }

    if (degree < 3) {
        if (v_dir != nullptr) {
            vec3 dir_n = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B = 1.445305721320277f * z;
    float fC2 = x * fC1 - y * fS1;
    float fS2 = x * fS1 + y * fC1;
    float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    float pSH13 = fTmp0C * x;
    float pSH11 = fTmp0C * y;
    float pSH14 = fTmp1B * fC1;
    float pSH10 = fTmp1B * fS1;
    float pSH15 = -0.5900435899266435f * fC2;
    float pSH9 = -0.5900435899266435f * fS2;
    v_coeffs[9 * 3 + c] = pSH9 * v_colors_local;
    v_coeffs[10 * 3 + c] = pSH10 * v_colors_local;
    v_coeffs[11 * 3 + c] = pSH11 * v_colors_local;
    v_coeffs[12 * 3 + c] = pSH12 * v_colors_local;
    v_coeffs[13 * 3 + c] = pSH13 * v_colors_local;
    v_coeffs[14 * 3 + c] = pSH14 * v_colors_local;
    v_coeffs[15 * 3 + c] = pSH15 * v_colors_local;

    float fTmp0C_z, fTmp1B_z, fC2_x, fC2_y, fS2_x, fS2_y, pSH12_z, pSH13_x,
        pSH13_z, pSH11_y, pSH11_z, pSH14_x, pSH14_y, pSH14_z, pSH10_x, pSH10_y,
        pSH10_z, pSH15_x, pSH15_y, pSH9_x, pSH9_y;
    if (v_dir != nullptr) {
        fTmp0C_z = -2.285228997322329f * 2.f * z;
        fTmp1B_z = 1.445305721320277f;
        fC2_x = fC1 + x * fC1_x - y * fS1_x;
        fC2_y = x * fC1_y - fS1 - y * fS1_y;
        fS2_x = fS1 + x * fS1_x + y * fC1_x;
        fS2_y = x * fS1_y + fC1 + y * fC1_y;
        pSH12_z = 3.f * 1.865881662950577f * z2 - 1.119528997770346f;
        pSH13_x = fTmp0C;
        pSH13_z = fTmp0C_z * x;
        pSH11_y = fTmp0C;
        pSH11_z = fTmp0C_z * y;
        pSH14_x = fTmp1B * fC1_x;
        pSH14_y = fTmp1B * fC1_y;
        pSH14_z = fTmp1B_z * fC1;
        pSH10_x = fTmp1B * fS1_x;
        pSH10_y = fTmp1B * fS1_y;
        pSH10_z = fTmp1B_z * fS1;
        pSH15_x = -0.5900435899266435f * fC2_x;
        pSH15_y = -0.5900435899266435f * fC2_y;
        pSH9_x = -0.5900435899266435f * fS2_x;
        pSH9_y = -0.5900435899266435f * fS2_y;

        v_x += v_colors_local *
               (pSH9_x * coeffs[9 * 3 + c] + pSH15_x * coeffs[15 * 3 + c] +
                pSH10_x * coeffs[10 * 3 + c] + pSH14_x * coeffs[14 * 3 + c] +
                pSH13_x * coeffs[13 * 3 + c]);

        v_y += v_colors_local *
               (pSH9_y * coeffs[9 * 3 + c] + pSH15_y * coeffs[15 * 3 + c] +
                pSH10_y * coeffs[10 * 3 + c] + pSH14_y * coeffs[14 * 3 + c] +
                pSH11_y * coeffs[11 * 3 + c]);

        v_z += v_colors_local *
               (pSH12_z * coeffs[12 * 3 + c] + pSH13_z * coeffs[13 * 3 + c] +
                pSH11_z * coeffs[11 * 3 + c] + pSH14_z * coeffs[14 * 3 + c] +
                pSH10_z * coeffs[10 * 3 + c]);
    }

    if (degree < 4) {
        if (v_dir != nullptr) {
            vec3 dir_n = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    float fTmp2B = -1.770130769779931f * z;
    float fC3 = x * fC2 - y * fS2;
    float fS3 = x * fS2 + y * fC2;
    float pSH20 = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
    float pSH21 = fTmp0D * x;
    float pSH19 = fTmp0D * y;
    float pSH22 = fTmp1C * fC1;
    float pSH18 = fTmp1C * fS1;
    float pSH23 = fTmp2B * fC2;
    float pSH17 = fTmp2B * fS2;
    float pSH24 = 0.6258357354491763f * fC3;
    float pSH16 = 0.6258357354491763f * fS3;
    v_coeffs[16 * 3 + c] = pSH16 * v_colors_local;
    v_coeffs[17 * 3 + c] = pSH17 * v_colors_local;
    v_coeffs[18 * 3 + c] = pSH18 * v_colors_local;
    v_coeffs[19 * 3 + c] = pSH19 * v_colors_local;
    v_coeffs[20 * 3 + c] = pSH20 * v_colors_local;
    v_coeffs[21 * 3 + c] = pSH21 * v_colors_local;
    v_coeffs[22 * 3 + c] = pSH22 * v_colors_local;
    v_coeffs[23 * 3 + c] = pSH23 * v_colors_local;
    v_coeffs[24 * 3 + c] = pSH24 * v_colors_local;

    float fTmp0D_z, fTmp1C_z, fTmp2B_z, fC3_x, fC3_y, fS3_x, fS3_y, pSH20_z,
        pSH21_x, pSH21_z, pSH19_y, pSH19_z, pSH22_x, pSH22_y, pSH22_z, pSH18_x,
        pSH18_y, pSH18_z, pSH23_x, pSH23_y, pSH23_z, pSH17_x, pSH17_y, pSH17_z,
        pSH24_x, pSH24_y, pSH16_x, pSH16_y;
    if (v_dir != nullptr) {
        fTmp0D_z = 3.f * -4.683325804901025f * z2 + 2.007139630671868f;
        fTmp1C_z = 2.f * 3.31161143515146f * z;
        fTmp2B_z = -1.770130769779931f;
        fC3_x = fC2 + x * fC2_x - y * fS2_x;
        fC3_y = x * fC2_y - fS2 - y * fS2_y;
        fS3_x = fS2 + y * fC2_x + x * fS2_x;
        fS3_y = x * fS2_y + fC2 + y * fC2_y;
        pSH20_z = 1.984313483298443f * (pSH12 + z * pSH12_z) +
                  -1.006230589874905f * pSH6_z;
        pSH21_x = fTmp0D;
        pSH21_z = fTmp0D_z * x;
        pSH19_y = fTmp0D;
        pSH19_z = fTmp0D_z * y;
        pSH22_x = fTmp1C * fC1_x;
        pSH22_y = fTmp1C * fC1_y;
        pSH22_z = fTmp1C_z * fC1;
        pSH18_x = fTmp1C * fS1_x;
        pSH18_y = fTmp1C * fS1_y;
        pSH18_z = fTmp1C_z * fS1;
        pSH23_x = fTmp2B * fC2_x;
        pSH23_y = fTmp2B * fC2_y;
        pSH23_z = fTmp2B_z * fC2;
        pSH17_x = fTmp2B * fS2_x;
        pSH17_y = fTmp2B * fS2_y;
        pSH17_z = fTmp2B_z * fS2;
        pSH24_x = 0.6258357354491763f * fC3_x;
        pSH24_y = 0.6258357354491763f * fC3_y;
        pSH16_x = 0.6258357354491763f * fS3_x;
        pSH16_y = 0.6258357354491763f * fS3_y;

        v_x += v_colors_local *
               (pSH16_x * coeffs[16 * 3 + c] + pSH24_x * coeffs[24 * 3 + c] +
                pSH17_x * coeffs[17 * 3 + c] + pSH23_x * coeffs[23 * 3 + c] +
                pSH18_x * coeffs[18 * 3 + c] + pSH22_x * coeffs[22 * 3 + c] +
                pSH21_x * coeffs[21 * 3 + c]);
        v_y += v_colors_local *
               (pSH16_y * coeffs[16 * 3 + c] + pSH24_y * coeffs[24 * 3 + c] +
                pSH17_y * coeffs[17 * 3 + c] + pSH23_y * coeffs[23 * 3 + c] +
                pSH18_y * coeffs[18 * 3 + c] + pSH22_y * coeffs[22 * 3 + c] +
                pSH19_y * coeffs[19 * 3 + c]);
        v_z += v_colors_local *
               (pSH20_z * coeffs[20 * 3 + c] + pSH21_z * coeffs[21 * 3 + c] +
                pSH19_z * coeffs[19 * 3 + c] + pSH22_z * coeffs[22 * 3 + c] +
                pSH18_z * coeffs[18 * 3 + c] + pSH23_z * coeffs[23 * 3 + c] +
                pSH17_z * coeffs[17 * 3 + c]);

        vec3 dir_n = vec3(x, y, z);
        vec3 v_dir_n = vec3(v_x, v_y, v_z);
        vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

        v_dir->x = v_d.x;
        v_dir->y = v_d.y;
        v_dir->z = v_d.z;
    }
}

template <typename scalar_t>
__global__ void spherical_harmonics_fwd_kernel(
    const uint32_t N,
    const uint32_t K,
    const uint32_t degrees_to_use,
    const vec3 *__restrict__ dirs,       // [N, 3]
    const scalar_t *__restrict__ coeffs, // [N, K, 3]
    const bool *__restrict__ masks,      // [N]
    scalar_t *__restrict__ colors        // [N, 3]
) {
    // parallelize over N * 3
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N * 3) {
        return;
    }
    uint32_t elem_id = idx / 3;
    uint32_t c = idx % 3; // color channel
    if (masks != nullptr && !masks[elem_id]) {
        return;
    }
    sh_coeffs_to_color_fast(
        degrees_to_use,
        c,
        dirs[elem_id],
        coeffs + elem_id * K * 3,
        colors + elem_id * 3
    );
}

void launch_spherical_harmonics_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    // outputs
    at::Tensor colors // [..., 2]
) {
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;

    // parallelize over N * 3
    int64_t n_elements = N * 3;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        dirs.scalar_type(),
        "spherical_harmonics_fwd_kernel",
        [&]() {
            spherical_harmonics_fwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    N,
                    K,
                    degrees_to_use,
                    reinterpret_cast<vec3 *>(dirs.data_ptr<scalar_t>()),
                    coeffs.data_ptr<scalar_t>(),
                    masks.has_value() ? masks.value().data_ptr<bool>()
                                      : nullptr,
                    colors.data_ptr<scalar_t>()
                );
        }
    );
}

template <typename scalar_t>
__global__ void spherical_harmonics_bwd_kernel(
    const uint32_t N,
    const uint32_t K,
    const uint32_t degrees_to_use,
    const vec3 *__restrict__ dirs,         // [N, 3]
    const scalar_t *__restrict__ coeffs,   // [N, K, 3]
    const bool *__restrict__ masks,        // [N]
    const scalar_t *__restrict__ v_colors, // [N, 3
    scalar_t *__restrict__ v_coeffs,       // [N, K, 3]
    scalar_t *__restrict__ v_dirs          // [N, 3] optional
) {
    // parallelize over N * 3
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N * 3) {
        return;
    }
    uint32_t elem_id = idx / 3;
    uint32_t c = idx % 3; // color channel
    if (masks != nullptr && !masks[elem_id]) {
        return;
    }

    vec3 v_dir = {0.f, 0.f, 0.f};
    sh_coeffs_to_color_fast_vjp(
        degrees_to_use,
        c,
        dirs[elem_id],
        coeffs + elem_id * K * 3,
        v_colors + elem_id * 3,
        v_coeffs + elem_id * K * 3,
        v_dirs == nullptr ? nullptr : &v_dir
    );
    if (v_dirs != nullptr) {
        gpuAtomicAdd(v_dirs + elem_id * 3, v_dir.x);
        gpuAtomicAdd(v_dirs + elem_id * 3 + 1, v_dir.y);
        gpuAtomicAdd(v_dirs + elem_id * 3 + 2, v_dir.z);
    }
}

void launch_spherical_harmonics_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    const at::Tensor v_colors,            // [..., 3]
    // outputs
    at::Tensor v_coeffs,            // [..., K, 3]
    at::optional<at::Tensor> v_dirs // [..., 3]
) {
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;

    // parallelize over N * 3
    int64_t n_elements = N * 3;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        dirs.scalar_type(),
        "spherical_harmonics_bwd_kernel",
        [&]() {
            spherical_harmonics_bwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    N,
                    K,
                    degrees_to_use,
                    reinterpret_cast<vec3 *>(dirs.data_ptr<scalar_t>()),
                    coeffs.data_ptr<scalar_t>(),
                    masks.has_value() ? masks.value().data_ptr<bool>()
                                      : nullptr,
                    v_colors.data_ptr<scalar_t>(),
                    v_coeffs.data_ptr<scalar_t>(),
                    v_dirs.has_value() ? v_dirs.value().data_ptr<scalar_t>()
                                       : nullptr
                );
        }
    );
}

} // namespace gsplat
