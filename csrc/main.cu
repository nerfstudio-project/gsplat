#include <iostream>
#include <cstring>

#include "forward.h"


int main() {
    int num_points = 24;

    int num_mean = num_points * 3;
    int num_scale = num_points * 3;
    int num_quat = num_points * 4;
    int num_cov3d = num_points * 6;
    int num_cov2d = num_points * 3;
    int num_view = 16;

    float *means = new float[num_mean];
    float *scales = new float[num_scale];
    float *quats = new float[num_quat];
    float *covs3d = new float[num_cov3d];
    float *covs2d = new float[num_cov2d];
    float viewmat [] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, -1.f,
        0.f, 0.f, 0.f, 1.f
    };

    for (int i = 0; i < num_points; ++i) {
        // float x [] = {(float) i * 0.2, (float) i * 0.4, (float) i * 0.6};
        float x [] = {0.f, 0.f, 0.f};
        float s [] = {(float) i + 0.1f, (float) i + 0.1f, (float) i + 0.1f};
        float q [] = {1.f, 0.f, 0.f, 0.f};
        std::memcpy(&means[3 * i], &x, sizeof(float) * 3);
        std::memcpy(&scales[3 * i], &s, sizeof(float) * 3);
        // printf("scales %d, %.2f, %.2f, %.2f\n", i, scales[3*i], scales[3*i+1], scales[3*i+2]);
        std::memcpy(&quats[4 * i], &q, sizeof(float) * 4);
        // printf("quats %d, %.2f, %.2f, %.2f, %.2f\n", i, quats[4*i], quats[4*i+1], quats[4*i+2], quats[4*i+3]);
    }

    float *scales_d, *means_d, *quats_d, *viewmat_d, *covs3d_d, *covs2d_d;

    cudaMalloc((void**) &scales_d, num_scale * sizeof(float));
    cudaMalloc((void**) &means_d, num_mean * sizeof(float));
    cudaMalloc((void**) &quats_d, num_quat * sizeof(float));
    cudaMalloc((void**)&viewmat_d, num_view * sizeof(float));

    cudaMemcpy(scales_d, scales, num_scale * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(means_d, means, num_mean * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(quats_d, quats, num_quat * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(viewmat_d, viewmat, num_view * sizeof(float), cudaMemcpyHostToDevice);

    // allocate memory for outputs
    cudaMalloc((void**)&covs3d_d, num_cov3d * sizeof(float));
    cudaMalloc((void**)&covs2d_d, num_cov2d * sizeof(float));

    project_gaussians_forward_impl(num_points, means_d, scales_d, 1.f, quats_d, viewmat_d, covs3d_d, covs2d_d);
    cudaMemcpy(covs3d, covs3d_d, num_cov3d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(covs2d, covs2d_d, num_cov2d * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_points; ++i) {
        printf("covs3d %d ", i);
        for (int j=0; j < 6; ++j) {
            printf("%.2f,", covs3d[6*i+j]);
        }
        printf("\n");

        printf("covs2d %d ", i);
        for (int j=0; j < 3; ++j) {
            printf("%.2f,", covs2d[3*i+j]);
        }
        printf("\n");
    }

    cudaFree(scales_d);
    cudaFree(quats_d);
    cudaFree(covs3d_d);

    delete[] scales;
    delete[] means;
    delete[] quats;
    delete[] covs3d;
    delete[] covs2d;
    return 0;
}
