#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <math.h>

#include "backward.cuh"
#include "tgaimage.h"


float random_float() {
    return (float) std::rand() / RAND_MAX;
}


int main() {
    int num_points = 2048;
    const float fov_x = M_PI / 2.f;
    const int W = 512;
    const int H = 512;
    const float focal = 0.5 * (float) W / tan(0.5 * fov_x);
    const dim3 tile_bounds = {(W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1};
    const dim3 img_size = {W, H, 1};
    const dim3 block = {BLOCK_X, BLOCK_Y, 1};

    int num_cov3d = num_points * 6;
    int num_view = 16;

    float3 *means = new float3[num_points];
    float3 *scales = new float3[num_points];
    float4 *quats = new float4[num_points];
    float3 *rgbs = new float3[num_points];
    float *opacities = new float[num_points];
    float viewmat [] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 8.f,
        0.f, 0.f, 0.f, 1.f
    };

    // silly initialization of gaussians
    // rotate pi/4 about (1, 1, 0)
    float bd = 2.f;
    // float norm = sqrt(2);
    // float theta = M_PI / 4.f;
    // float w = norm / tan(theta / 2.f);
    float u, v, w;
    std::srand(std::time(nullptr));
    for (int i = 0; i < num_points; ++i) {
        means[i] = {bd * (random_float() - 0.5f), bd * (random_float() - 0.5f), bd * (random_float() - 0.5f)};
        scales[i] = {random_float(), random_float(), random_float()};
        rgbs[i] = {random_float(), random_float(), random_float()};

        // float v = (float) i - (float) num_points * 0.5f;
        // means[i] = {v * 0.1f, v * 0.1f, (float) i};
        // scales[i] = {0.5f, 5.f, 1.f};
        // rgbs[i] = {(float) (i % 3 == 0), (float) (i % 3 == 1), (float) (i % 3 == 2)};

        // quats[i] = {w, norm, norm, 0.f};  // w x y z convention
        // quats[i] = {1.f, 0.f, 0.f, 0.f};  // w x y z convention
        // random quat
        u = random_float();
        v = random_float();
        w = random_float();
        quats[i] = {
            sqrt(1.f - u) * sin(2.f * (float) M_PI * v),
            sqrt(1.f - u) * cos(2.f * (float) M_PI * v),
            sqrt(u) * sin(2.f * (float) M_PI * w),
            sqrt(u) * cos(2.f * (float) M_PI * w)
        };

        opacities[i] = 0.9f;
    }

    float3 *scales_d, *means_d, *rgbs_d;
    float4 *quats_d;
    float *viewmat_d, *opacities_d;

    cudaMalloc((void**) &scales_d, num_points * sizeof(float3));
    cudaMalloc((void**) &means_d, num_points * sizeof(float3));
    cudaMalloc((void**) &quats_d, num_points * sizeof(float4));
    cudaMalloc((void**) &rgbs_d, num_points * sizeof(float3));
    cudaMalloc((void**) &opacities_d, num_points * sizeof(float));
    cudaMalloc((void**) &viewmat_d, num_view * sizeof(float));

    cudaMemcpy(scales_d, scales, num_points * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(means_d, means, num_points * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(rgbs_d, rgbs, num_points * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(opacities_d, opacities, num_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(quats_d, quats, num_points * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(viewmat_d, viewmat, num_view * sizeof(float), cudaMemcpyHostToDevice);

    // allocate memory for outputs
    float *covs3d = new float[num_cov3d];
    float2 *xy = new float2[num_points];
    float *z = new float[num_points];
    int *radii = new int[num_points];
    float3 *conics = new float3[num_points];
    uint32_t *num_tiles_hit = new uint32_t[num_points];

    float *covs3d_d, *z_d;
    float2 *xy_d;
    float3 *conics_d;
    int *radii_d;
    uint32_t *num_tiles_hit_d;
    cudaMalloc((void**)&covs3d_d, num_cov3d * sizeof(float));
    cudaMalloc((void**)&xy_d, num_points * sizeof(float2));
    cudaMalloc((void**)&z_d, num_points * sizeof(float));
    cudaMalloc((void**)&radii_d, num_points * sizeof(int));
    cudaMalloc((void**)&conics_d, num_points * sizeof(float3));
    cudaMalloc((void**)&num_tiles_hit_d, num_points * sizeof(uint32_t));

    project_gaussians_forward_impl(
        num_points,
        means_d,
        scales_d,
        1.f,
        quats_d,
        viewmat_d,
        viewmat_d,
        focal,
        focal,
        img_size,
        tile_bounds,
        covs3d_d,
        xy_d,
        z_d,
        radii_d,
        conics_d,
        num_tiles_hit_d
    );
    cudaMemcpy(covs3d, covs3d_d, num_cov3d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xy, xy_d, num_points * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(z, z_d, num_points * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(radii, radii_d, num_points * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(num_tiles_hit, num_tiles_hit_d, num_points * sizeof(uint32_t), cudaMemcpyDeviceToHost);


