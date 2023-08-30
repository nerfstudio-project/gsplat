#include <iostream>
#include <cstring>

#include "forward.h"


int main() {
    int num_points = 24;

    int num_scale = num_points * 3;
    int num_quat = num_points * 4;
    int num_cov = num_points * 6;

    float *scales = new float[num_scale];
    float *rots_quat = new float[num_quat];
    float *covs3d = new float[num_cov];

    for (int i = 0; i < num_points; ++i) {
        float s [] = {(float) i, (float) i, (float) i};
        float q [] = {1.f, 0.f, 0.f, 0.f};
        std::memcpy(&scales[3 * i], &s, sizeof(float) * 3);
        // printf("scales %d, %.2f, %.2f, %.2f\n", i, scales[3*i], scales[3*i+1], scales[3*i+2]);
        std::memcpy(&rots_quat[4 * i], &q, sizeof(float) * 4);
        // printf("rots_quat %d, %.2f, %.2f, %.2f, %.2f\n", i, rots_quat[4*i], rots_quat[4*i+1], rots_quat[4*i+2], rots_quat[4*i+3]);
    }

    float glob_scale = 1;

    float *scales_d;
    float *rots_quat_d;
    float *covs3d_d;

    auto stat = cudaMalloc((void**) &scales_d, num_scale * sizeof(float));
    std::cout << stat << std::endl;

    stat = cudaMalloc((void**) &rots_quat_d, num_quat * sizeof(float));
    std::cout << stat << std::endl;

    stat = cudaMalloc((void**)&covs3d_d, num_cov * sizeof(float));
    std::cout << stat << std::endl;

    stat = cudaMemcpy(scales_d, scales, num_scale * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << stat << std::endl;
    stat = cudaMemcpy(rots_quat_d, rots_quat, num_quat * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << stat << std::endl;

    project_gaussians_forward_impl(num_points, scales_d, glob_scale, rots_quat_d, covs3d_d);
    cudaMemcpy(covs3d, covs3d_d, num_cov * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_points; ++i) {
        printf("covs3d %d ", i);
        for (int j=0; j < 6; ++j) {
            printf("%.2f,", covs3d[6*i+j]);
        }
        printf("\n");
    }

    cudaFree(scales_d);
    cudaFree(rots_quat_d);
    cudaFree(covs3d_d);

    delete[] scales;
    delete[] rots_quat;
    delete[] covs3d;
    return 0;
}
