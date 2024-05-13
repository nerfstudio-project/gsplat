/*
Modified from
https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/sh.cuh
*/

#include "bindings.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// Evaluate spherical harmonics bases at unit direction for high orders using approach
// described by Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013 See
// https://jcgt.org/published/0002/02/06/ for reference implementation
__device__ void
sh_coeffs_to_color_fast(const unsigned degree, // degree of SH to be evaluated
                        const unsigned c,      // color channel
                        const float3 &dir,     // [3]
                        const float *coeffs,   // [K, 3]
                        // output
                        float *colors // [3]
) {
    float result = 0.2820947917738781f * coeffs[c];
    if (degree >= 1) {
        // Normally rsqrt is faster than sqrt, but --use_fast_math will optimize sqrt
        // on single precision, so we use sqrt here.
        float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        float x = dir.x * inorm;
        float y = dir.y * inorm;
        float z = dir.z * inorm;

        result += 0.48860251190292f * (-y * coeffs[1 * 3 + c] + z * coeffs[2 * 3 + c] -
                                       x * coeffs[3 * 3 + c]);
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
                float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
                float pSH13 = fTmp0C * x;
                float pSH11 = fTmp0C * y;
                float pSH14 = fTmp1B * fC1;
                float pSH10 = fTmp1B * fS1;
                float pSH15 = -0.5900435899266435f * fC2;
                float pSH9 = -0.5900435899266435f * fS2;

                result += pSH9 * coeffs[9 * 3 + c] + pSH10 * coeffs[10 * 3 + c] +
                          pSH11 * coeffs[11 * 3 + c] + pSH12 * coeffs[12 * 3 + c] +
                          pSH13 * coeffs[13 * 3 + c] + pSH14 * coeffs[14 * 3 + c] +
                          pSH15 * coeffs[15 * 3 + c];

                if (degree >= 4) {
                    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
                    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                    float fTmp2B = -1.770130769779931f * z;
                    float fC3 = x * fC2 - y * fS2;
                    float fS3 = x * fS2 + y * fC2;
                    float pSH20 =
                        (1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6);
                    float pSH21 = fTmp0D * x;
                    float pSH19 = fTmp0D * y;
                    float pSH22 = fTmp1C * fC1;
                    float pSH18 = fTmp1C * fS1;
                    float pSH23 = fTmp2B * fC2;
                    float pSH17 = fTmp2B * fS2;
                    float pSH24 = 0.6258357354491763f * fC3;
                    float pSH16 = 0.6258357354491763f * fS3;

                    result += pSH16 * coeffs[16 * 3 + c] + pSH17 * coeffs[17 * 3 + c] +
                              pSH18 * coeffs[18 * 3 + c] + pSH19 * coeffs[19 * 3 + c] +
                              pSH20 * coeffs[20 * 3 + c] + pSH21 * coeffs[21 * 3 + c] +
                              pSH22 * coeffs[22 * 3 + c] + pSH23 * coeffs[23 * 3 + c] +
                              pSH24 * coeffs[24 * 3 + c];
                }
            }
        }
    }

    colors[c] = result;
}

__device__ void
sh_coeffs_to_color_fast_vjp(const unsigned degree, // degree of SH to be evaluated
                            const unsigned c,      // color channel
                            const float3 &dir,     // [3]
                            const float *v_colors, // [3]
                            // output
                            float *v_coeffs // [K, 3]
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

    v_coeffs[1 * 3 + c] = -0.48860251190292f * y * v_colors_local;
    v_coeffs[2 * 3 + c] = 0.48860251190292f * z * v_colors_local;
    v_coeffs[3 * 3 + c] = -0.48860251190292f * x * v_colors_local;
    if (degree < 2) {
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
    if (degree < 3) {
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
    if (degree < 4) {
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
}

__global__ void compute_sh_fwd_kernel(const unsigned N, const unsigned K,
                                      const unsigned degrees_to_use,
                                      const float3 *__restrict__ dirs,  // [N, 3]
                                      const float *__restrict__ coeffs, // [N, K, 3]
                                      const bool *__restrict__ masks,   // [N]
                                      float *__restrict__ colors        // [N, 3]
) {
    // parallelize over N * 3
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= N * 3) {
        return;
    }
    unsigned elem_id = idx / 3;
    unsigned c = idx % 3; // color channel
    if (masks != nullptr && !masks[elem_id]) {
        return;
    }
    sh_coeffs_to_color_fast(degrees_to_use, c, dirs[elem_id], coeffs + elem_id * K * 3,
                            colors + elem_id * 3);
}

__global__ void compute_sh_bwd_kernel(const unsigned N, const unsigned K,
                                      const unsigned degrees_to_use,
                                      const float3 *__restrict__ dirs,    // [N, 3]
                                      const bool *__restrict__ masks,     // [N]
                                      const float *__restrict__ v_colors, // [N, 3
                                      float *__restrict__ v_coeffs        // [N, K, 3]
) {
    // parallelize over N * 3
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= N * 3) {
        return;
    }
    unsigned elem_id = idx / 3;
    unsigned c = idx % 3; // color channel
    if (masks != nullptr && !masks[elem_id]) {
        return;
    }
    sh_coeffs_to_color_fast_vjp(degrees_to_use, c, dirs[elem_id],
                                v_colors + elem_id * 3, v_coeffs + elem_id * K * 3);
}

torch::Tensor compute_sh_fwd_tensor(const unsigned degrees_to_use,
                                    torch::Tensor &dirs,              // [..., 3]
                                    torch::Tensor &coeffs,            // [..., K, 3]
                                    at::optional<torch::Tensor> masks // [...]
) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT(coeffs);
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension 3");
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
    const unsigned K = coeffs.size(-2);
    const unsigned N = dirs.numel() / 3;
    torch::Tensor colors = torch::empty_like(dirs); // [..., 3]
    // parallelize over N * 3
    compute_sh_fwd_kernel<<<(N * 3 + N_THREADS - 1) / N_THREADS, N_THREADS>>>(
        N, K, degrees_to_use, (float3 *)dirs.data_ptr<float>(),
        coeffs.data_ptr<float>(),
        masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
        colors.data_ptr<float>());
    return colors; // [..., 3]
}

torch::Tensor compute_sh_bwd_tensor(const unsigned K, const unsigned degrees_to_use,
                                    torch::Tensor &dirs,               // [..., 3]
                                    at::optional<torch::Tensor> masks, // [...]
                                    torch::Tensor &v_colors            // [..., 3]
) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT(v_colors);
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(v_colors.size(-1) == 3, "v_colors must have last dimension 3");
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
    const unsigned N = dirs.numel() / 3;

    // v_coeffs has shape [..., K, 3]
    auto batch_shape = dirs.sizes().vec();
    batch_shape.pop_back();
    batch_shape.push_back(K);
    batch_shape.push_back(3);
    torch::Tensor v_coeffs = torch::zeros(batch_shape, dirs.options());
    compute_sh_bwd_kernel<<<(N * 3 + N_THREADS - 1) / N_THREADS, N_THREADS>>>(
        N, K, degrees_to_use, (float3 *)dirs.data_ptr<float>(),
        masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
        v_colors.data_ptr<float>(), v_coeffs.data_ptr<float>());
    return v_coeffs; // [..., K, 3]
}
