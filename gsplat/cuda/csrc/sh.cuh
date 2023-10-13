#include "cuda_runtime.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
__device__ const float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};
__device__ const float SH_C4[] = {
    2.5033429417967046f,
    -1.7701307697799304,
    0.9461746957575601f,
    -0.6690465435572892f,
    0.10578554691520431f,
    -0.6690465435572892f,
    0.47308734787878004f,
    -1.7701307697799304f,
    0.6258357354491761f};

// This function is used in both host and device code
__host__ __device__ unsigned num_sh_bases(const unsigned degree) {
    if (degree == 0)
        return 1;
    if (degree == 1)
        return 4;
    if (degree == 2)
        return 9;
    if (degree == 3)
        return 16;
    return 25;
}

template <int CHANNELS>
__device__ void sh_coeffs_to_color(
    const unsigned degree,
    const float3 &viewdir,
    const float *coeffs,
    float *colors
) {
    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] = SH_C0 * coeffs[c];
    }
    if (degree < 1) {
        return;
    }

    float norm = sqrt(
        viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z
    );
    float x = viewdir.x / norm;
    float y = viewdir.y / norm;
    float z = viewdir.z / norm;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    // expects CHANNELS * num_bases coefficients
    // supports up to num_bases = 25
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] += SH_C1 * (-y * coeffs[1 * CHANNELS + c] +
                              z * coeffs[2 * CHANNELS + c] -
                              x * coeffs[3 * CHANNELS + c]);
        if (degree < 2) {
            continue;
        }
        colors[c] +=
            (SH_C2[0] * xy * coeffs[4 * CHANNELS + c] +
             SH_C2[1] * yz * coeffs[5 * CHANNELS + c] +
             SH_C2[2] * (2.f * zz - xx - yy) * coeffs[6 * CHANNELS + c] +
             SH_C2[3] * xz * coeffs[7 * CHANNELS + c] +
             SH_C2[4] * (xx - yy) * coeffs[8 * CHANNELS + c]);
        if (degree < 3) {
            continue;
        }
        colors[c] +=
            (SH_C3[0] * y * (3.f * xx - yy) * coeffs[9 * CHANNELS + c] +
             SH_C3[1] * xy * z * coeffs[10 * CHANNELS + c] +
             SH_C3[2] * y * (4.f * zz - xx - yy) * coeffs[11 * CHANNELS + c] +
             SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy) *
                 coeffs[12 * CHANNELS + c] +
             SH_C3[4] * x * (4.f * zz - xx - yy) * coeffs[13 * CHANNELS + c] +
             SH_C3[5] * z * (xx - yy) * coeffs[14 * CHANNELS + c] +
             SH_C3[6] * x * (xx - 3.f * yy) * coeffs[15 * CHANNELS + c]);
        if (degree < 4) {
            continue;
        }
        colors[c] +=
            (SH_C4[0] * xy * (xx - yy) * coeffs[16 * CHANNELS + c] +
             SH_C4[1] * yz * (3.f * xx - yy) * coeffs[17 * CHANNELS + c] +
             SH_C4[2] * xy * (7.f * zz - 1.f) * coeffs[18 * CHANNELS + c] +
             SH_C4[3] * yz * (7.f * zz - 3.f) * coeffs[19 * CHANNELS + c] +
             SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f) *
                 coeffs[20 * CHANNELS + c] +
             SH_C4[5] * xz * (7.f * zz - 3.f) * coeffs[21 * CHANNELS + c] +
             SH_C4[6] * (xx - yy) * (7.f * zz - 1.f) *
                 coeffs[22 * CHANNELS + c] +
             SH_C4[7] * xz * (xx - 3.f * yy) * coeffs[23 * CHANNELS + c] +
             SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy)) *
                 coeffs[24 * CHANNELS + c]);
    }
}

template <int CHANNELS>
__device__ void sh_coeffs_to_color_vjp(
    const unsigned degree,
    const float3 &viewdir,
    const float *v_colors,
    float *v_coeffs
) {
    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    for (int c = 0; c < CHANNELS; ++c) {
        v_coeffs[c] = SH_C0 * v_colors[c];
    }
    if (degree < 1) {
        return;
    }

    float norm = sqrt(
        viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z
    );
    float x = viewdir.x / norm;
    float y = viewdir.y / norm;
    float z = viewdir.z / norm;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;

    for (int c = 0; c < CHANNELS; ++c) {
        float v1 = -SH_C1 * y;
        float v2 = SH_C1 * z;
        float v3 = -SH_C1 * x;
        v_coeffs[1 * CHANNELS + c] = v1 * v_colors[c];
        v_coeffs[2 * CHANNELS + c] = v2 * v_colors[c];
        v_coeffs[3 * CHANNELS + c] = v3 * v_colors[c];
        // printf("c1 %.5e %.5e %.5e\n", v1, v2, v3);
        if (degree < 2) {
            continue;
        }
        float v4 = SH_C2[0] * xy;
        float v5 = SH_C2[1] * yz;
        float v6 = SH_C2[2] * (2.f * zz - xx - yy);
        float v7 = SH_C2[3] * xz;
        float v8 = SH_C2[4] * (xx - yy);
        v_coeffs[4 * CHANNELS + c] = v4 * v_colors[c];
        v_coeffs[5 * CHANNELS + c] = v5 * v_colors[c];
        v_coeffs[6 * CHANNELS + c] = v6 * v_colors[c];
        v_coeffs[7 * CHANNELS + c] = v7 * v_colors[c];
        v_coeffs[8 * CHANNELS + c] = v8 * v_colors[c];
        // printf("c2 %.5e %.5e %.5e %.5e %.5e\n", v4, v5, v6, v7, v8);
        if (degree < 3) {
            continue;
        }
        float v9 = SH_C3[0] * y * (3.f * xx - yy);
        float v10 = SH_C3[1] * xy * z;
        float v11 = SH_C3[2] * y * (4.f * zz - xx - yy);
        float v12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
        float v13 = SH_C3[4] * x * (4.f * zz - xx - yy);
        float v14 = SH_C3[5] * z * (xx - yy);
        float v15 = SH_C3[6] * x * (xx - 3.f * yy);
        v_coeffs[9 * CHANNELS + c] = v9 * v_colors[c];
        v_coeffs[10 * CHANNELS + c] = v10 * v_colors[c];
        v_coeffs[11 * CHANNELS + c] = v11 * v_colors[c];
        v_coeffs[12 * CHANNELS + c] = v12 * v_colors[c];
        v_coeffs[13 * CHANNELS + c] = v13 * v_colors[c];
        v_coeffs[14 * CHANNELS + c] = v14 * v_colors[c];
        v_coeffs[15 * CHANNELS + c] = v15 * v_colors[c];
        // printf(
        //     "c3 %.5e %.5e %.5e %.5e %.5e %.5e %.5e\n",
        //     v9,
        //     v10,
        //     v11,
        //     v12,
        //     v13,
        //     v14,
        //     v15
        // );
        if (degree < 4) {
            continue;
        }
        float v16 = SH_C4[0] * xy * (xx - yy);
        float v17 = SH_C4[1] * yz * (3.f * xx - yy);
        float v18 = SH_C4[2] * xy * (7.f * zz - 1.f);
        float v19 = SH_C4[3] * yz * (7.f * zz - 3.f);
        float v20 = SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f);
        float v21 = SH_C4[5] * xz * (7.f * zz - 3.f);
        float v22 = SH_C4[6] * (xx - yy) * (7.f * zz - 1.f);
        float v23 = SH_C4[7] * xz * (xx - 3.f * yy);
        float v24 = SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy));
        v_coeffs[16 * CHANNELS + c] = v16 * v_colors[c];
        v_coeffs[17 * CHANNELS + c] = v17 * v_colors[c];
        v_coeffs[18 * CHANNELS + c] = v18 * v_colors[c];
        v_coeffs[19 * CHANNELS + c] = v19 * v_colors[c];
        v_coeffs[20 * CHANNELS + c] = v20 * v_colors[c];
        v_coeffs[21 * CHANNELS + c] = v21 * v_colors[c];
        v_coeffs[22 * CHANNELS + c] = v22 * v_colors[c];
        v_coeffs[23 * CHANNELS + c] = v23 * v_colors[c];
        v_coeffs[24 * CHANNELS + c] = v24 * v_colors[c];
        // printf(
        //     "c4 %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e\n",
        //     v16,
        //     v17,
        //     v18,
        //     v19,
        //     v20,
        //     v21,
        //     v22,
        //     v23,
        //     v24
        // );
    }
}

__global__ void compute_sh_forward_kernel(
    const unsigned num_points,
    const unsigned degree,
    const float3 *viewdirs,
    const float *coeffs,
    float *colors
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points) {
        return;
    }
    const unsigned num_channels = 3;
    unsigned num_bases = num_sh_bases(degree);
    unsigned idx_sh = num_bases * num_channels * idx;
    unsigned idx_col = num_channels * idx;

    sh_coeffs_to_color<num_channels>(
        degree, viewdirs[idx], &(coeffs[idx_sh]), &(colors[idx_col])
    );
}

__global__ void compute_sh_backward_kernel(
    const unsigned num_points,
    const unsigned degree,
    const float3 *viewdirs,
    const float *v_colors,
    float *v_coeffs
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points) {
        return;
    }
    const unsigned num_channels = 3;
    unsigned num_bases = num_sh_bases(degree);
    unsigned idx_sh = num_bases * num_channels * idx;
    unsigned idx_col = num_channels * idx;

    sh_coeffs_to_color_vjp<num_channels>(
        degree, viewdirs[idx], &(v_colors[idx_col]), &(v_coeffs[idx_sh])
    );
}
